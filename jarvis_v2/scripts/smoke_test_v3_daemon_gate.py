"""Offline proof that V3 background daemons fail closed before startup."""

from __future__ import annotations

import contextlib
import io
import os

from jarvis_v2.scripts import (
    run_imessage_control,
    run_scheduler,
    run_status_server,
    run_telegram_control,
)
from jarvis_v2.scripts.daemon_gate import (
    V3_DAEMON_ENABLE_ENV,
    V3_SCHEDULER_ENABLE_ENV,
    v3_daemons_enabled,
    v3_scheduler_enabled,
)


INVALID_ENABLE_VALUES = ("", "0", "true", "yes", "invalid", " 1", "1 ", "\t1\n")


def _assert_blocked(module, service: str) -> None:
    original_env = os.environ.pop(V3_DAEMON_ENABLE_ENV, None)
    original_load_config = module.load_config
    runtime_started = False

    def forbidden_load_config():
        nonlocal runtime_started
        runtime_started = True
        raise AssertionError("runtime construction crossed the V3 daemon gate")

    module.load_config = forbidden_load_config
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            try:
                module.main()
            except SystemExit as exc:
                if exc.code != 4:
                    raise SystemExit(f"{service} daemon gate used unexpected exit code: {exc.code}")
            else:
                raise SystemExit(f"{service} daemon gate did not refuse startup")
    finally:
        module.load_config = original_load_config
        if original_env is not None:
            os.environ[V3_DAEMON_ENABLE_ENV] = original_env

    message = output.getvalue()
    for expected in (
        f"Jarvis V3 {service} daemon is disabled before startup.",
        f"Set {V3_DAEMON_ENABLE_ENV}=1 only for a supervised V3 service cutover",
        "No runtime was constructed and no external action was attempted.",
    ):
        if expected not in message:
            raise SystemExit(f"{service} daemon gate missed guidance: {expected}")
    if runtime_started:
        raise SystemExit(f"{service} daemon constructed runtime while disabled")


def _assert_dashboard_blocked(value: str | None) -> None:
    original_env = os.environ.get(V3_DAEMON_ENABLE_ENV)
    original_host = run_status_server.status_host_from_env
    original_runtime = run_status_server.JarvisRuntime
    crossed_gate: list[str] = []

    def forbidden_host() -> str:
        crossed_gate.append("config")
        raise AssertionError("dashboard loaded config while disabled")

    class ForbiddenRuntime:
        def __init__(self) -> None:
            crossed_gate.append("runtime")
            raise AssertionError("dashboard constructed runtime while disabled")

    if value is None:
        os.environ.pop(V3_DAEMON_ENABLE_ENV, None)
    else:
        os.environ[V3_DAEMON_ENABLE_ENV] = value
    run_status_server.status_host_from_env = forbidden_host
    run_status_server.JarvisRuntime = ForbiddenRuntime
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            try:
                run_status_server.main()
            except SystemExit as exc:
                if exc.code != 4:
                    raise SystemExit(
                        f"dashboard daemon gate used unexpected exit code: {exc.code}"
                    )
            else:
                raise SystemExit("dashboard daemon gate did not refuse startup")
    finally:
        run_status_server.status_host_from_env = original_host
        run_status_server.JarvisRuntime = original_runtime
        if original_env is None:
            os.environ.pop(V3_DAEMON_ENABLE_ENV, None)
        else:
            os.environ[V3_DAEMON_ENABLE_ENV] = original_env

    for expected in (
        "Jarvis V3 dashboard daemon is disabled before startup.",
        f"Set {V3_DAEMON_ENABLE_ENV}=1 only for a supervised V3 service cutover",
        "No runtime was constructed and no external action was attempted.",
    ):
        if expected not in output.getvalue():
            raise SystemExit(f"dashboard daemon gate missed guidance: {expected}")
    if crossed_gate:
        raise SystemExit(
            f"dashboard crossed its daemon gate while disabled: {', '.join(crossed_gate)}"
        )


def test_invalid_values_fail_closed() -> None:
    original = os.environ.get(V3_DAEMON_ENABLE_ENV)
    try:
        for value in INVALID_ENABLE_VALUES:
            os.environ[V3_DAEMON_ENABLE_ENV] = value
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    run_scheduler.main()
                except SystemExit as exc:
                    if exc.code != 4:
                        raise SystemExit(f"invalid daemon value {value!r} used exit {exc.code}")
                else:
                    raise SystemExit(f"invalid daemon value {value!r} enabled scheduler")
    finally:
        if original is None:
            os.environ.pop(V3_DAEMON_ENABLE_ENV, None)
        else:
            os.environ[V3_DAEMON_ENABLE_ENV] = original


def test_only_raw_exact_one_enables_gates() -> None:
    original_daemon = os.environ.get(V3_DAEMON_ENABLE_ENV)
    original_scheduler = os.environ.get(V3_SCHEDULER_ENABLE_ENV)
    try:
        for key, check in (
            (V3_DAEMON_ENABLE_ENV, v3_daemons_enabled),
            (V3_SCHEDULER_ENABLE_ENV, v3_scheduler_enabled),
        ):
            os.environ[key] = "1"
            if check() is not True:
                raise SystemExit(f"exact enable value did not activate {key}")
            for value in INVALID_ENABLE_VALUES:
                os.environ[key] = value
                if check() is not False:
                    raise SystemExit(f"non-exact enable value {value!r} activated {key}")
    finally:
        if original_daemon is None:
            os.environ.pop(V3_DAEMON_ENABLE_ENV, None)
        else:
            os.environ[V3_DAEMON_ENABLE_ENV] = original_daemon
        if original_scheduler is None:
            os.environ.pop(V3_SCHEDULER_ENABLE_ENV, None)
        else:
            os.environ[V3_SCHEDULER_ENABLE_ENV] = original_scheduler


def test_scheduler_requires_separate_exact_enable() -> None:
    original_daemon = os.environ.get(V3_DAEMON_ENABLE_ENV)
    original_scheduler = os.environ.get(V3_SCHEDULER_ENABLE_ENV)
    original_load_config = run_scheduler.load_config
    runtime_started = False

    def forbidden_load_config():
        nonlocal runtime_started
        runtime_started = True
        raise AssertionError("scheduler crossed its service-specific gate")

    run_scheduler.load_config = forbidden_load_config
    try:
        os.environ[V3_DAEMON_ENABLE_ENV] = "1"
        for value in INVALID_ENABLE_VALUES:
            os.environ[V3_SCHEDULER_ENABLE_ENV] = value
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    run_scheduler.main()
                except SystemExit as exc:
                    if exc.code != 4:
                        raise SystemExit(
                            f"scheduler-specific value {value!r} used exit {exc.code}"
                        )
                else:
                    raise SystemExit(
                        f"scheduler-specific value {value!r} enabled scheduler execution"
                    )
            rendered = output.getvalue()
            for expected in (
                "Jarvis V3 scheduler execution is disabled before startup.",
                f"Set {V3_SCHEDULER_ENABLE_ENV}=1 only for a separately supervised scheduler cutover",
                "No scheduler was constructed and no scheduled job was run.",
            ):
                if expected not in rendered:
                    raise SystemExit(
                        f"scheduler-specific gate missed guidance {expected!r}: {rendered!r}"
                    )
        if runtime_started:
            raise SystemExit("scheduler constructed runtime without its separate enable")
    finally:
        run_scheduler.load_config = original_load_config
        if original_daemon is None:
            os.environ.pop(V3_DAEMON_ENABLE_ENV, None)
        else:
            os.environ[V3_DAEMON_ENABLE_ENV] = original_daemon
        if original_scheduler is None:
            os.environ.pop(V3_SCHEDULER_ENABLE_ENV, None)
        else:
            os.environ[V3_SCHEDULER_ENABLE_ENV] = original_scheduler


def main() -> None:
    _assert_blocked(run_telegram_control, "Telegram")
    _assert_blocked(run_imessage_control, "iMessage")
    _assert_blocked(run_scheduler, "scheduler")
    _assert_dashboard_blocked(None)
    for value in INVALID_ENABLE_VALUES:
        _assert_dashboard_blocked(value)
    test_invalid_values_fail_closed()
    test_only_raw_exact_one_enables_gates()
    test_scheduler_requires_separate_exact_enable()
    print("V3 daemon activation gate smoke passed (Telegram + iMessage + dashboard + scheduler)")


if __name__ == "__main__":
    main()
