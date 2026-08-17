from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.scripts.daemon_gate import require_v3_daemon_enable
from jarvis_v2.scripts.startup import is_startup_storage_error, print_startup_failure
from jarvis_v2.ui.status_config import (
    MIN_STATUS_AUTH_TOKEN_LENGTH,
    STATUS_AUTH_ENV,
    STATUS_AUTH_USERNAME,
    status_auth_config_from_env,
    status_base_url,
    status_host_from_env,
    status_host_is_loopback,
    status_port_from_env,
)
from jarvis_v2.ui.status_server import make_status_server


def _parser(*, default_host: str, default_port: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Jarvis V3 read-only status dashboard.")
    parser.add_argument("--host", default=default_host, help="Bind host. Defaults to JARVIS_STATUS_HOST or 127.0.0.1.")
    parser.add_argument("--port", type=int, default=default_port, help="Bind port. Defaults to JARVIS_STATUS_PORT or 8766.")
    return parser


_ROOT_DASHBOARD_WRAPPER = Path(__file__).resolve().parents[2] / "launch_jarvis_v3_dashboard.py"


def _help_requested() -> bool:
    return any(argument in {"-h", "--help"} for argument in sys.argv[1:])


def _show_help() -> None:
    _parser(default_host="127.0.0.1", default_port=8766).parse_args()


def _refuse_manual_foreground(reason: str) -> None:
    print(
        "Jarvis V3 dashboard manual launch was refused before environment, "
        f"authentication, or runtime startup: {reason}. Run the exact root launcher "
        "from an attached foreground Terminal.",
        flush=True,
    )
    raise SystemExit(4)


def _require_manual_foreground_attestation(
    wrapper_path: str | os.PathLike[str],
    *,
    caller_frame,
) -> None:
    """Bind manual authority to the exact root wrapper and foreground TTY.

    The caller-frame check prevents importing this module and treating a Boolean
    or public helper call as manual-launch authority.  The TTY process-group
    check makes a LaunchAgent, detached process, or background shell job fail
    before any environment, authentication, storage, or runtime access.
    """

    try:
        declared = Path(wrapper_path)
        declared_lstat = declared.lstat()
        expected_stat = _ROOT_DASHBOARD_WRAPPER.stat()
        declared_resolved = declared.resolve(strict=True)
        expected_resolved = _ROOT_DASHBOARD_WRAPPER.resolve(strict=True)
        caller_name = caller_frame.f_globals.get("__name__")
        caller_file = Path(caller_frame.f_code.co_filename).resolve(strict=True)
        caller_global_file = Path(caller_frame.f_globals.get("__file__", "")).resolve(strict=True)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        _refuse_manual_foreground("root-wrapper provenance could not be verified")

    if (
        declared_resolved != expected_resolved
        or caller_file != expected_resolved
        or caller_global_file != expected_resolved
        or caller_name != "__main__"
        or not stat.S_ISREG(declared_lstat.st_mode)
        or (declared_lstat.st_dev, declared_lstat.st_ino)
        != (expected_stat.st_dev, expected_stat.st_ino)
        or declared_lstat.st_uid != os.getuid()
        or stat.S_IMODE(declared_lstat.st_mode) & 0o022
    ):
        _refuse_manual_foreground("root-wrapper provenance did not match")

    try:
        tty_fd = sys.stdin.fileno()
        attached = tty_fd >= 0 and os.isatty(tty_fd)
        foreground_pgrp = os.tcgetpgrp(tty_fd) if attached else -1
        process_pgrp = os.getpgrp()
    except (AttributeError, OSError, ValueError):
        _refuse_manual_foreground("foreground Terminal ownership could not be verified")
    if not attached:
        _refuse_manual_foreground("standard input is not an attached Terminal")
    if foreground_pgrp <= 0 or foreground_pgrp != process_pgrp:
        _refuse_manual_foreground("this process is not the Terminal foreground process group")


def _run_dashboard() -> None:
    # Help is documentation, not daemon activation. Keep it available to fresh
    # installs without loading config or constructing runtime state.
    if _help_requested():
        _show_help()
        return
    try:
        default_host = status_host_from_env()
        default_port = status_port_from_env()
    except PermissionError:
        print("Jarvis status dashboard refused an insecure environment file.")
        print("If it contains JARVIS_STATUS_AUTH_TOKEN, make it owner-only with: chmod 600 .env")
        raise SystemExit(7) from None

    args = _parser(default_host=default_host, default_port=default_port).parse_args()

    if not status_host_is_loopback(args.host):
        print("Jarvis status dashboard refused a non-loopback bind host.")
        print("Use 127.0.0.1, another 127.0.0.0/8 address, ::1, or localhost.")
        raise SystemExit(6)

    try:
        auth = status_auth_config_from_env()
    except PermissionError:
        print("Jarvis status dashboard refused an insecure environment file.")
        print("If it contains JARVIS_STATUS_AUTH_TOKEN, make it owner-only with: chmod 600 .env")
        raise SystemExit(7) from None
    if not auth.valid:
        print("Jarvis status dashboard authentication is not configured safely.")
        print(
            f"Set {STATUS_AUTH_ENV} to a unique printable secret of at least "
            f"{MIN_STATUS_AUTH_TOKEN_LENGTH} characters, then retry."
        )
        raise SystemExit(5)

    try:
        runtime = JarvisRuntime()
    except Exception as exc:
        if not is_startup_storage_error(exc):
            raise
        print_startup_failure(exc, program="Jarvis status dashboard")
        raise SystemExit(3) from exc
    url = status_base_url(args.host, args.port)
    try:
        server = make_status_server(runtime, args.host, args.port, auth_token=auth.token)
    except OSError as exc:
        print(f"Jarvis status dashboard could not bind {url}.")
        print(f"Diagnostic: {type(exc).__name__}")
        print("Choose another local port with --port or JARVIS_STATUS_PORT.")
        raise SystemExit(4) from exc
    print(f"Jarvis status dashboard: {url}")
    print(
        f"Sign in with username {STATUS_AUTH_USERNAME!r}. Copy the password with the "
        "owner-only helper documented in QUICKSTART.md; do not print the environment value."
    )
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


def main() -> None:
    """Run the service/module entrypoint behind the V3 daemon cutover gate."""

    if _help_requested():
        _show_help()
        return
    require_v3_daemon_enable("dashboard")
    _run_dashboard()


def manual_foreground_main(*, wrapper_path: str | os.PathLike[str]) -> None:
    """Run the operator-invoked root launcher without enabling V3 daemons.

    This authority is deliberately limited to the authenticated loopback server
    in this foreground process.  It neither installs nor enables a LaunchAgent,
    and exiting the process closes the server.
    """

    if _help_requested():
        _show_help()
        return
    _require_manual_foreground_attestation(wrapper_path, caller_frame=sys._getframe(1))
    _run_dashboard()


if __name__ == "__main__":
    main()
