"""Offline smoke tests for the V3 scheduler timezone bootstrap."""

from __future__ import annotations

import ast
import contextlib
import io
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import mock
import zoneinfo

from jarvis_v2.scripts import v3_scheduler_bootstrap as bootstrap
from jarvis_v2.scripts.daemon_gate import (
    V3_DAEMON_ENABLE_ENV,
    V3_SCHEDULER_ENABLE_ENV,
)


def _tzif_payload() -> bytes:
    for root in zoneinfo.TZPATH:
        candidate = Path(root) / "Asia/Seoul"
        try:
            payload = candidate.read_bytes()
        except OSError:
            continue
        if payload.startswith(b"TZif"):
            return payload
    raise SystemExit("host test runtime has no Asia/Seoul TZif fixture")


@contextlib.contextmanager
def _runtime_fixture():
    with TemporaryDirectory(prefix="jarvis-v3-tz-bootstrap-", dir="/private/tmp") as temp:
        root = Path(temp)
        runtime = root / "runtime"
        prefix = runtime / "python"
        interpreter_directory = prefix / "bin"
        asia_directory = prefix / "share" / "zoneinfo" / "Asia"
        interpreter_directory.mkdir(parents=True)
        asia_directory.mkdir(parents=True)
        interpreter = interpreter_directory / "python3"
        tzif = asia_directory / "Seoul"
        interpreter.write_bytes(b"synthetic bundled interpreter")
        tzif.write_bytes(_tzif_payload())
        interpreter.chmod(0o500)
        tzif.chmod(0o400)
        for directory in (
            interpreter_directory,
            asia_directory,
            asia_directory.parent,
            prefix / "share",
            prefix,
            runtime,
            root,
        ):
            directory.chmod(0o500)
        try:
            yield interpreter, tzif
        finally:
            for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o700)
                elif path.exists() and not path.is_symlink():
                    path.chmod(0o600)


@contextlib.contextmanager
def _timezone_restored():
    previous_path = tuple(zoneinfo.TZPATH)
    previous_present = "TZ" in os.environ
    previous_tz = os.environ.get("TZ")
    try:
        yield
    finally:
        zoneinfo.reset_tzpath(previous_path)
        zoneinfo.ZoneInfo.clear_cache()
        if previous_present and previous_tz is not None:
            os.environ["TZ"] = previous_tz
        else:
            os.environ.pop("TZ", None)
        bootstrap.time.tzset()


@contextlib.contextmanager
def _gate_environment(daemon: str | None, scheduler: str | None):
    previous_daemon = os.environ.get(V3_DAEMON_ENABLE_ENV)
    previous_scheduler = os.environ.get(V3_SCHEDULER_ENABLE_ENV)
    try:
        if daemon is None:
            os.environ.pop(V3_DAEMON_ENABLE_ENV, None)
        else:
            os.environ[V3_DAEMON_ENABLE_ENV] = daemon
        if scheduler is None:
            os.environ.pop(V3_SCHEDULER_ENABLE_ENV, None)
        else:
            os.environ[V3_SCHEDULER_ENABLE_ENV] = scheduler
        yield
    finally:
        if previous_daemon is None:
            os.environ.pop(V3_DAEMON_ENABLE_ENV, None)
        else:
            os.environ[V3_DAEMON_ENABLE_ENV] = previous_daemon
        if previous_scheduler is None:
            os.environ.pop(V3_SCHEDULER_ENABLE_ENV, None)
        else:
            os.environ[V3_SCHEDULER_ENABLE_ENV] = previous_scheduler


def _expect_failure(interpreter: Path | str) -> None:
    try:
        with mock.patch.object(bootstrap.sys, "executable", os.fspath(interpreter)):
            bootstrap._select_bundled_timezone()
    except bootstrap.SchedulerTimezoneBootstrapError as exc:
        if str(exc) != "scheduler_timezone_bootstrap_failed" or "/" in str(exc):
            raise SystemExit("timezone bootstrap refusal leaked a path or changed reason")
        return
    raise SystemExit("timezone bootstrap accepted invalid custody")


def test_both_gates_precede_timezone_or_scheduler_work() -> None:
    for daemon, scheduler in ((None, None), ("1", None), ("1", "0")):
        crossed: list[str] = []
        with (
            _gate_environment(daemon, scheduler),
            mock.patch.object(
                bootstrap,
                "_select_bundled_timezone",
                side_effect=lambda: crossed.append("timezone"),
            ),
            mock.patch.object(
                bootstrap,
                "_run_scheduler",
                side_effect=lambda: crossed.append("scheduler"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                bootstrap.main()
            except SystemExit as exc:
                if exc.code != 4:
                    raise SystemExit("bootstrap gate used the wrong refusal exit")
            else:
                raise SystemExit("bootstrap crossed a disabled gate")
        if crossed:
            raise SystemExit("bootstrap performed timezone or scheduler work before both gates")


def test_exact_bundle_timezone_is_selected_before_scheduler_import() -> None:
    with _runtime_fixture() as (interpreter, tzif), _timezone_restored(), _gate_environment(
        "1", "1"
    ):
        order: list[str] = []

        def run_scheduler() -> None:
            order.append("scheduler")
            expected_path = os.fspath(tzif.parent.parent)
            if tuple(zoneinfo.TZPATH) != (expected_path,):
                raise SystemExit("bootstrap did not select one exact bundled TZPATH")
            if os.environ.get("TZ") != f":{os.fspath(tzif)}":
                raise SystemExit("bootstrap did not select the exact libc TZif")

        original_select = bootstrap._select_bundled_timezone

        def select() -> None:
            order.append("timezone")
            original_select()

        with (
            mock.patch.object(bootstrap.sys, "executable", os.fspath(interpreter)),
            mock.patch.object(bootstrap, "_select_bundled_timezone", side_effect=select),
            mock.patch.object(bootstrap, "_run_scheduler", side_effect=run_scheduler),
        ):
            bootstrap.main()
        if order != ["timezone", "scheduler"]:
            raise SystemExit("scheduler import/run was not ordered after timezone selection")


def test_tzif_custody_and_content_fail_closed() -> None:
    with _runtime_fixture() as (interpreter, tzif):
        tzif.parent.chmod(0o700)
        tzif.unlink()
        tzif.parent.chmod(0o500)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, tzif):
        tzif.chmod(0o600)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, tzif):
        tzif.parent.chmod(0o700)
        os.link(tzif, tzif.with_name("Seoul-hardlink"))
        tzif.parent.chmod(0o500)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, tzif):
        tzif.parent.chmod(0o700)
        original = tzif.with_name("Seoul-original")
        tzif.rename(original)
        tzif.symlink_to(original)
        tzif.parent.chmod(0o500)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, tzif):
        tzif.parent.chmod(0o700)
        tzif.unlink()
        tzif.mkdir()
        tzif.parent.chmod(0o500)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, tzif):
        tzif.chmod(0o600)
        tzif.write_bytes(b"TZif2" + (b"\x00" * 64))
        tzif.chmod(0o400)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, tzif):
        tzif.chmod(0o600)
        with tzif.open("wb") as stream:
            stream.truncate(bootstrap.MAX_TZIF_BYTES + 1)
        tzif.chmod(0o400)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, _tzif):
        with mock.patch.object(bootstrap.os, "geteuid", return_value=os.geteuid() + 1):
            _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, _tzif):
        interpreter.parent.parent.parent.parent.chmod(0o700)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, _tzif):
        interpreter.chmod(0o700)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, _tzif):
        interpreter.parent.chmod(0o700)
        os.link(interpreter, interpreter.with_name("python3-hardlink"))
        interpreter.parent.chmod(0o500)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, _tzif):
        interpreter.parent.chmod(0o700)
        original = interpreter.with_name("python3-original")
        interpreter.rename(original)
        interpreter.symlink_to(original)
        interpreter.parent.chmod(0o500)
        _expect_failure(interpreter)

    with _runtime_fixture() as (interpreter, tzif):
        zoneinfo_directory = tzif.parent.parent
        share = zoneinfo_directory.parent
        share.chmod(0o700)
        zoneinfo_directory.chmod(0o700)
        original = zoneinfo_directory.with_name("zoneinfo-original")
        zoneinfo_directory.rename(original)
        zoneinfo_directory.symlink_to(original)
        share.chmod(0o500)
        _expect_failure(interpreter)

    for invalid in (
        Path("runtime/python/bin/python3"),
        Path("/private/tmp/nonexistent/runtime/python/bin/python3"),
        Path("/private/tmp/wrong/runtime/python/bin/python"),
        "/private/tmp/runtime/python/bin/./python3",
        "/private/tmp/runtime/python/bin/../bin/python3",
        "//private/tmp/runtime/python/bin/python3",
    ):
        _expect_failure(invalid)


def test_executable_path_policy_refuses_noncanonical_text_before_open() -> None:
    suffix = "/runtime/python/bin/python3"
    invalid_paths = (
        "/private/tmp/e\u0301" + suffix,
        "/private/tmp/control-\x01" + suffix,
        "/private/tmp/" + ("a" * bootstrap.MAX_EXECUTABLE_PATH_BYTES) + suffix,
    )
    for invalid in invalid_paths:
        with (
            mock.patch.object(bootstrap.sys, "executable", invalid),
            mock.patch.object(
                bootstrap.os,
                "open",
                side_effect=AssertionError("invalid path reached filesystem access"),
            ),
        ):
            try:
                bootstrap._open_timezone_custody()
            except bootstrap.SchedulerTimezoneBootstrapError:
                continue
        raise SystemExit("noncanonical executable path passed lexical validation")


def test_fdopen_failure_closes_duplicated_tzif_descriptor() -> None:
    duplicated: list[int] = []
    real_dup = os.dup
    with _runtime_fixture() as (interpreter, _tzif):
        def tracking_dup(descriptor: int) -> int:
            result = real_dup(descriptor)
            duplicated.append(result)
            return result

        with (
            mock.patch.object(bootstrap.sys, "executable", os.fspath(interpreter)),
            mock.patch.object(bootstrap.os, "dup", side_effect=tracking_dup),
            mock.patch.object(
                bootstrap.os,
                "fdopen",
                side_effect=OSError("injected fdopen failure"),
            ),
        ):
            try:
                bootstrap._select_bundled_timezone()
            except bootstrap.SchedulerTimezoneBootstrapError:
                pass
            else:
                raise SystemExit("fdopen failure passed timezone bootstrap")
    if len(duplicated) != 1:
        raise SystemExit("fdopen failure did not exercise exactly one duplicate")
    try:
        os.fstat(duplicated[0])
    except OSError:
        return
    os.close(duplicated[0])
    raise SystemExit("fdopen failure leaked the duplicated TZif descriptor")


def test_invalid_exact_tzif_cannot_be_masked_by_fallback() -> None:
    fallback_calls: list[str] = []

    class FallbackCapableZoneInfo:
        @classmethod
        def from_file(cls, _stream, *, key):
            raise ValueError(f"invalid exact TZif at /private/{key}")

        @classmethod
        def no_cache(cls, key):
            fallback_calls.append(key)
            return zoneinfo.ZoneInfo(key)

        @classmethod
        def clear_cache(cls):
            return None

    with _runtime_fixture() as (interpreter, tzif):
        tzif.chmod(0o600)
        tzif.write_bytes(b"TZif2" + (b"\x00" * 64))
        tzif.chmod(0o400)
        with mock.patch.object(bootstrap.zoneinfo, "ZoneInfo", FallbackCapableZoneInfo):
            _expect_failure(interpreter)
    if fallback_calls:
        raise SystemExit("invalid exact TZif reached a package/host fallback lookup")


def test_valid_exact_tzif_never_uses_key_lookup_fallback() -> None:
    real_zone_info = zoneinfo.ZoneInfo
    key_lookups: list[str] = []

    class DescriptorOnlyZoneInfo:
        @classmethod
        def from_file(cls, stream, *, key):
            return real_zone_info.from_file(stream, key=key)

        @classmethod
        def no_cache(cls, key):
            key_lookups.append(key)
            return real_zone_info(key)

        @classmethod
        def clear_cache(cls):
            real_zone_info.clear_cache()

    with _runtime_fixture() as (interpreter, _tzif), _timezone_restored():
        with (
            mock.patch.object(bootstrap.sys, "executable", os.fspath(interpreter)),
            mock.patch.object(bootstrap.zoneinfo, "ZoneInfo", DescriptorOnlyZoneInfo),
        ):
            bootstrap._select_bundled_timezone()
    if key_lookups:
        raise SystemExit("valid exact TZif was followed by a fallback-capable key lookup")


def test_post_mutation_failure_restores_timezone_without_path_leak() -> None:
    secret = "/private/tmp/private-timezone-error-sentinel"
    with _runtime_fixture() as (interpreter, _tzif), _timezone_restored():
        previous_path = tuple(zoneinfo.TZPATH)
        previous_present = "TZ" in os.environ
        previous_tz = os.environ.get("TZ")
        real_tzset = bootstrap.time.tzset
        calls = 0

        def fail_once() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError(secret)
            real_tzset()

        try:
            with (
                mock.patch.object(bootstrap.sys, "executable", os.fspath(interpreter)),
                mock.patch.object(bootstrap.time, "tzset", side_effect=fail_once),
            ):
                bootstrap._select_bundled_timezone()
        except bootstrap.SchedulerTimezoneBootstrapError as exc:
            if (
                secret in str(exc)
                or secret in repr(exc)
                or exc.__cause__ is not None
                or tuple(zoneinfo.TZPATH) != previous_path
                or ("TZ" in os.environ) != previous_present
                or os.environ.get("TZ") != previous_tz
            ):
                raise SystemExit("post-mutation refusal leaked data or failed rollback")
        else:
            raise SystemExit("injected post-mutation failure was accepted")


def test_isolated_import_poison_proves_pre_scheduler_failure_boundary() -> None:
    repository = Path(bootstrap.__file__).parents[2]
    child = r'''
import os
import pathlib
import sys

sys.path.insert(0, REPOSITORY)
FORBIDDEN = (
    "jarvis_v2.config",
    "jarvis_v2.automations",
    "jarvis_v2.memory",
    "jarvis_v2.storage",
    "jarvis_v2.tools",
    "jarvis_v2.delivery",
    "jarvis_v2.integrations",
    "jarvis_v2.scripts.run_scheduler",
    "sqlite3",
    "socket",
    "subprocess",
    "requests",
    "urllib",
    "http",
)
class Poison:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(FORBIDDEN):
            raise RuntimeError("forbidden import:" + fullname)
        return None
sys.meta_path.insert(0, Poison())
from jarvis_v2.scripts import v3_scheduler_bootstrap as subject
os.environ.pop("JARVIS_V3_ENABLE_DAEMONS", None)
os.environ.pop("JARVIS_V3_ENABLE_SCHEDULER", None)
try:
    subject.main()
except SystemExit as exc:
    assert exc.code == 4
else:
    raise AssertionError("disabled gate passed")
os.environ["JARVIS_V3_ENABLE_DAEMONS"] = "1"
os.environ["JARVIS_V3_ENABLE_SCHEDULER"] = "1"
sys.executable = "/private/tmp/nonexistent/runtime/python/bin/python3"
try:
    subject.main()
except SystemExit as exc:
    assert exc.code == 5
else:
    raise AssertionError("missing timezone custody passed")
print("ISOLATED_BOOTSTRAP_BOUNDARY_OK")
'''.replace("REPOSITORY", repr(os.fspath(repository)))
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", child],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10.0,
        check=False,
    )
    if (
        result.returncode != 0
        or "ISOLATED_BOOTSTRAP_BOUNDARY_OK" not in result.stdout
        or "forbidden import" in result.stdout + result.stderr
        or "Traceback" in result.stdout + result.stderr
    ):
        raise SystemExit("isolated import-poison bootstrap boundary failed")


def test_failure_is_bounded_and_import_surface_stays_inert() -> None:
    source_path = Path(bootstrap.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module)
    forbidden_imports = {
        "jarvis_v2.config",
        "jarvis_v2.automations.scheduler",
        "jarvis_v2.memory.store",
        "jarvis_v2.tools.registry",
        "requests",
        "socket",
    }
    if top_level_imports & forbidden_imports:
        raise SystemExit("bootstrap gained a config, storage, scheduler, delivery, or network import")

    with _runtime_fixture() as (interpreter, tzif), _gate_environment("1", "1"):
        tzif.chmod(0o600)
        output = io.StringIO()
        with (
            mock.patch.object(bootstrap.sys, "executable", os.fspath(interpreter)),
            mock.patch.object(
                bootstrap,
                "_run_scheduler",
                side_effect=AssertionError("scheduler imported after failed bootstrap"),
            ),
            contextlib.redirect_stdout(output),
        ):
            try:
                bootstrap.main()
            except SystemExit as exc:
                if exc.code != bootstrap.BOOTSTRAP_FAILURE_EXIT:
                    raise SystemExit("bootstrap failure used the wrong bounded exit")
            else:
                raise SystemExit("invalid timezone custody reached scheduler import")
        rendered = output.getvalue()
        if (
            rendered != bootstrap.BOOTSTRAP_FAILURE_MESSAGE + "\n"
            or os.fspath(interpreter) in rendered
            or os.fspath(tzif) in rendered
            or "Traceback" in rendered
            or bootstrap.TARGET_SCHEDULER_MODULE in sys.modules
        ):
            raise SystemExit("bootstrap failure was not path-free and pre-import")


def test_preloaded_scheduler_is_refused_truthfully() -> None:
    sentinel = object()
    output = io.StringIO()
    with (
        _gate_environment("1", "1"),
        mock.patch.dict(sys.modules, {bootstrap.TARGET_SCHEDULER_MODULE: sentinel}),
        mock.patch.object(
            bootstrap,
            "_select_bundled_timezone",
            side_effect=AssertionError("timezone work crossed a dirty import boundary"),
        ),
        mock.patch.object(
            bootstrap,
            "_run_scheduler",
            side_effect=AssertionError("preloaded scheduler was invoked"),
        ),
        contextlib.redirect_stdout(output),
    ):
        try:
            bootstrap.main()
        except SystemExit as exc:
            if exc.code != bootstrap.BOOTSTRAP_FAILURE_EXIT:
                raise SystemExit("preloaded scheduler refusal used the wrong exit")
        else:
            raise SystemExit("preloaded scheduler crossed the bootstrap boundary")
    rendered = output.getvalue()
    if (
        rendered != bootstrap.BOOTSTRAP_FAILURE_MESSAGE + "\n"
        or "No scheduler was imported" in rendered
        or "before scheduler import" in rendered
    ):
        raise SystemExit("preloaded scheduler refusal made a false import claim")


def main() -> None:
    test_both_gates_precede_timezone_or_scheduler_work()
    test_exact_bundle_timezone_is_selected_before_scheduler_import()
    test_tzif_custody_and_content_fail_closed()
    test_executable_path_policy_refuses_noncanonical_text_before_open()
    test_fdopen_failure_closes_duplicated_tzif_descriptor()
    test_invalid_exact_tzif_cannot_be_masked_by_fallback()
    test_valid_exact_tzif_never_uses_key_lookup_fallback()
    test_post_mutation_failure_restores_timezone_without_path_leak()
    test_isolated_import_poison_proves_pre_scheduler_failure_boundary()
    test_failure_is_bounded_and_import_surface_stays_inert()
    test_preloaded_scheduler_is_refused_truthfully()
    print("V3 scheduler timezone bootstrap smoke passed")


if __name__ == "__main__":
    main()
