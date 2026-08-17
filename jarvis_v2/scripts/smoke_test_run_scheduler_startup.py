from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jarvis_v2.scripts import run_scheduler


PRIVATE_MARKERS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        db_path=Path("/private/tmp/jarvis-scheduler.sqlite"),
        obsidian_vault=Path("/\x55sers/example/private/JarvisVault"),
        obsidian_root="Jarvis",
    )


def _assert_bounded_failure(
    label: str,
    *,
    config_error: BaseException | None = None,
    store_error: BaseException | None = None,
    vault_error: BaseException | None = None,
    scheduler_error: BaseException | None = None,
) -> None:
    store = Mock(name="store")
    vault = Mock(name="vault")
    scheduler = Mock(name="scheduler")
    store.init.side_effect = store_error
    vault.init.side_effect = vault_error

    load_config = Mock(name="load_config", return_value=_config(), side_effect=config_error)
    memory_store = Mock(name="MemoryStore", return_value=store)
    obsidian_vault = Mock(name="ObsidianVault", return_value=vault)
    profile_recovery = Mock(name="reconcile_pending_profile_projections")
    goal_recovery = Mock(name="reconcile_pending_goal_projections")
    memory_recovery = Mock(name="reconcile_pending_memory_projections")
    scheduler_type = Mock(name="Scheduler", return_value=scheduler, side_effect=scheduler_error)
    stdout = StringIO()
    stderr = StringIO()

    with (
        patch.object(run_scheduler, "load_config", load_config),
        patch.object(run_scheduler, "MemoryStore", memory_store),
        patch.object(run_scheduler, "ObsidianVault", obsidian_vault),
        patch.object(run_scheduler, "reconcile_pending_profile_projections", profile_recovery),
        patch.object(run_scheduler, "reconcile_pending_goal_projections", goal_recovery),
        patch.object(run_scheduler, "reconcile_pending_memory_projections", memory_recovery),
        patch.object(run_scheduler, "Scheduler", scheduler_type),
        patch.object(run_scheduler.time, "sleep", side_effect=AssertionError("scheduler loop started")),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        try:
            run_scheduler.main()
        except SystemExit as exc:
            if exc.code != 3:
                raise SystemExit(f"{label} should exit 3, got {exc.code}") from exc
        else:
            raise SystemExit(f"{label} should stop during startup")

    output = stdout.getvalue() + stderr.getvalue()
    for expected in (
        "Jarvis scheduler could not start.",
        "<local-path>",
        "safe recovery:",
        "python3 -m jarvis_v2.scripts.bootstrap_memory --check",
    ):
        if expected not in output:
            raise SystemExit(f"{label} missed bounded startup guidance {expected!r}: {output}")
    for marker in PRIVATE_MARKERS:
        if marker in output:
            raise SystemExit(f"{label} leaked local path marker {marker!r}: {output}")
    for error in (config_error, store_error, vault_error, scheduler_error):
        if error is not None and str(error) and str(error) in output:
            raise SystemExit(f"{label} leaked raw startup error text: {output}")
    if "Traceback" in output:
        raise SystemExit(f"{label} emitted a traceback for an expected startup failure: {output}")

    if config_error is not None:
        memory_store.assert_not_called()
        obsidian_vault.assert_not_called()
        scheduler_type.assert_not_called()
    elif store_error is not None:
        store.init.assert_called_once_with()
        vault.init.assert_not_called()
        scheduler_type.assert_not_called()
    elif vault_error is not None:
        store.init.assert_called_once_with()
        vault.init.assert_called_once_with()
        scheduler_type.assert_not_called()
    else:
        store.init.assert_called_once_with()
        vault.init.assert_called_once_with()
        profile_recovery.assert_called_once_with(store, vault, limit=20)
        goal_recovery.assert_called_once_with(store, vault, limit=20)
        memory_recovery.assert_called_once_with(store, vault, limit=20)
        scheduler_type.assert_called_once_with(store, vault, load_config.return_value)


def test_expected_startup_failures_are_bounded() -> None:
    _assert_bounded_failure(
        "configuration failure",
        config_error=PermissionError("/\x55sers/example/private/scheduler.env"),
    )
    _assert_bounded_failure(
        "configuration Unicode failure",
        config_error=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "/private/env marker"),
    )
    _assert_bounded_failure(
        "store initialization failure",
        store_error=sqlite3.OperationalError("unable to open /private/tmp/jarvis-scheduler.sqlite"),
    )
    _assert_bounded_failure(
        "vault initialization failure",
        vault_error=PermissionError("/\x55sers/example/private/JarvisVault"),
    )
    _assert_bounded_failure(
        "scheduler initialization failure",
        scheduler_error=OSError("/var/folders/zc/private-scheduler-state"),
    )


def test_standalone_invalid_utf8_env_failure_is_bounded() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-startup-") as temp:
        root = Path(temp)
        env_path = root / "private-scheduler.env"
        env_path.write_bytes(b"JARVIS_DATA_DIR=private-secret-before-error\n\xff")
        env_path.chmod(0o600)
        child_env = os.environ.copy()
        child_env["JARVIS_V3_ENV"] = str(env_path)
        result = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.run_scheduler"],
            capture_output=True,
            text=True,
            env=child_env,
            timeout=10.0,
        )
        if result.returncode != 3:
            raise SystemExit(
                "standalone scheduler invalid UTF-8 env should exit 3: "
                f"status={result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
            )
        output = result.stdout + result.stderr
        for expected in (
            "Jarvis scheduler could not start.",
            "<local-path>",
            "safe recovery:",
            "python3 -m jarvis_v2.scripts.bootstrap_memory --check",
        ):
            if expected not in output:
                raise SystemExit(f"standalone scheduler missed bounded Unicode guidance {expected!r}: {output}")
        for forbidden in (str(env_path), "private-secret-before-error", "Traceback"):
            if forbidden in output:
                raise SystemExit(f"standalone scheduler Unicode receipt leaked {forbidden!r}: {output}")


def test_unexpected_programmer_error_remains_visible() -> None:
    cases = [
        ("configuration", "scheduler-programmer-error", RuntimeError, "load_config"),
        ("scheduler construction", "scheduler-unicode-programmer-error", UnicodeError, "Scheduler"),
    ]
    for label, marker, error_type, failing_target in cases:
        stdout = StringIO()
        stderr = StringIO()
        replacements = {
            "load_config": Mock(return_value=_config()),
            "MemoryStore": Mock(return_value=Mock()),
            "ObsidianVault": Mock(return_value=Mock()),
            "reconcile_pending_profile_projections": Mock(),
            "reconcile_pending_goal_projections": Mock(),
            "reconcile_pending_memory_projections": Mock(),
            "Scheduler": Mock(return_value=Mock()),
        }
        replacements[failing_target].side_effect = error_type(marker)
        with (
            patch.object(run_scheduler, "load_config", replacements["load_config"]),
            patch.object(run_scheduler, "MemoryStore", replacements["MemoryStore"]),
            patch.object(run_scheduler, "ObsidianVault", replacements["ObsidianVault"]),
            patch.object(
                run_scheduler,
                "reconcile_pending_profile_projections",
                replacements["reconcile_pending_profile_projections"],
            ),
            patch.object(
                run_scheduler,
                "reconcile_pending_memory_projections",
                replacements["reconcile_pending_memory_projections"],
            ),
            patch.object(
                run_scheduler,
                "reconcile_pending_goal_projections",
                replacements["reconcile_pending_goal_projections"],
            ),
            patch.object(run_scheduler, "Scheduler", replacements["Scheduler"]),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            try:
                run_scheduler.main()
            except error_type as exc:
                if str(exc) != marker:
                    raise SystemExit(f"unexpected {label} error changed: {exc}") from exc
            else:
                raise SystemExit(f"unexpected {label} programmer error should be re-raised")
        if stdout.getvalue() or stderr.getvalue():
            raise SystemExit(f"unexpected {label} error should not become a bounded receipt")


def test_retry_sleep_keyboard_interrupt_stops_gracefully() -> None:
    store = Mock(name="store")
    vault = Mock(name="vault")
    scheduler = Mock(name="scheduler")
    scheduler.run_due_jobs.side_effect = RuntimeError("/private/scheduler-tick-secret")
    stdout = StringIO()
    stderr = StringIO()

    with (
        patch.object(run_scheduler, "load_config", return_value=_config()),
        patch.object(run_scheduler, "MemoryStore", return_value=store),
        patch.object(run_scheduler, "ObsidianVault", return_value=vault),
        patch.object(run_scheduler, "reconcile_pending_profile_projections"),
        patch.object(run_scheduler, "reconcile_pending_goal_projections"),
        patch.object(run_scheduler, "reconcile_pending_memory_projections"),
        patch.object(run_scheduler, "Scheduler", return_value=scheduler),
        patch.object(run_scheduler.time, "sleep", side_effect=KeyboardInterrupt()),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        run_scheduler.main()

    output = stdout.getvalue() + stderr.getvalue()
    for expected in (
        "Jarvis V3 scheduler running.",
        "Scheduler tick failed:",
        "Retrying in 60 seconds.",
        "Scheduler stopped.",
    ):
        if expected not in output:
            raise SystemExit(f"retry-sleep interrupt missed {expected!r}: {output}")
    for forbidden in ("/private/scheduler-tick-secret", "Traceback"):
        if forbidden in output:
            raise SystemExit(f"retry-sleep interrupt leaked {forbidden!r}: {output}")
    scheduler.run_due_jobs.assert_called_once_with()
    store.init.assert_called_once_with()
    vault.init.assert_called_once_with()


def test_goal_projection_backlog_retries_each_scheduler_tick() -> None:
    store = Mock(name="store")
    vault = Mock(name="vault")
    scheduler = Mock(name="scheduler")
    scheduler.run_due_jobs.return_value = "No jobs due."
    goal_recovery = Mock(name="reconcile_pending_goal_projections")
    stdout = StringIO()

    with (
        patch.object(run_scheduler, "load_config", return_value=_config()),
        patch.object(run_scheduler, "MemoryStore", return_value=store),
        patch.object(run_scheduler, "ObsidianVault", return_value=vault),
        patch.object(run_scheduler, "reconcile_pending_profile_projections"),
        patch.object(run_scheduler, "reconcile_pending_goal_projections", goal_recovery),
        patch.object(run_scheduler, "reconcile_pending_memory_projections"),
        patch.object(run_scheduler, "Scheduler", return_value=scheduler),
        patch.object(run_scheduler.time, "sleep", side_effect=KeyboardInterrupt()),
        redirect_stdout(stdout),
    ):
        run_scheduler.main()

    if goal_recovery.call_args_list != [
        ((store, vault), {"limit": 20}),
        ((store, vault), {"limit": 20}),
    ]:
        raise SystemExit(
            "scheduler did not retry bounded goal projection backlog after startup: "
            f"{goal_recovery.call_args_list}"
        )
    scheduler.run_due_jobs.assert_called_once_with()


def test_goal_projection_backlog_retries_after_failed_scheduler_tick() -> None:
    store = Mock(name="store")
    vault = Mock(name="vault")
    scheduler = Mock(name="scheduler")
    scheduler.run_due_jobs.side_effect = RuntimeError("PRIVATE-TICK-FAILURE")
    goal_recovery = Mock(name="reconcile_pending_goal_projections")
    stdout = StringIO()

    with (
        patch.object(run_scheduler, "load_config", return_value=_config()),
        patch.object(run_scheduler, "MemoryStore", return_value=store),
        patch.object(run_scheduler, "ObsidianVault", return_value=vault),
        patch.object(run_scheduler, "reconcile_pending_profile_projections"),
        patch.object(run_scheduler, "reconcile_pending_goal_projections", goal_recovery),
        patch.object(run_scheduler, "reconcile_pending_memory_projections"),
        patch.object(run_scheduler, "Scheduler", return_value=scheduler),
        patch.object(run_scheduler.time, "sleep", side_effect=KeyboardInterrupt()),
        redirect_stdout(stdout),
    ):
        run_scheduler.main()

    if goal_recovery.call_args_list != [
        ((store, vault), {"limit": 20}),
        ((store, vault), {"limit": 20}),
    ]:
        raise SystemExit(
            "scheduler failure starved bounded goal projection recovery: "
            f"{goal_recovery.call_args_list}"
        )
    output = stdout.getvalue()
    if "Scheduler tick failed:" not in output or "PRIVATE-TICK-FAILURE" in output:
        raise SystemExit(f"scheduler failure receipt was unsafe or missing: {output}")


def main() -> None:
    old_daemon_enable = os.environ.get("JARVIS_V3_ENABLE_DAEMONS")
    old_scheduler_enable = os.environ.get("JARVIS_V3_ENABLE_SCHEDULER")
    os.environ["JARVIS_V3_ENABLE_DAEMONS"] = "1"
    os.environ["JARVIS_V3_ENABLE_SCHEDULER"] = "1"
    try:
        test_expected_startup_failures_are_bounded()
        test_standalone_invalid_utf8_env_failure_is_bounded()
        test_unexpected_programmer_error_remains_visible()
        test_retry_sleep_keyboard_interrupt_stops_gracefully()
        test_goal_projection_backlog_retries_each_scheduler_tick()
        test_goal_projection_backlog_retries_after_failed_scheduler_tick()
        print("Run scheduler startup smoke passed")
    finally:
        if old_daemon_enable is None:
            os.environ.pop("JARVIS_V3_ENABLE_DAEMONS", None)
        else:
            os.environ["JARVIS_V3_ENABLE_DAEMONS"] = old_daemon_enable
        if old_scheduler_enable is None:
            os.environ.pop("JARVIS_V3_ENABLE_SCHEDULER", None)
        else:
            os.environ["JARVIS_V3_ENABLE_SCHEDULER"] = old_scheduler_enable


if __name__ == "__main__":
    main()
