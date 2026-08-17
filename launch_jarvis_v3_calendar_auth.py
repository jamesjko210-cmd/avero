#!/usr/bin/env python3
"""Human-only Google Calendar authorization for Jarvis V3."""

import os
import stat
import sys
from pathlib import Path


_DIRECT_EXECUTION_AUTHORITY = object()


def _require_human_foreground() -> None:
    try:
        tty_fd = sys.stdin.fileno()
        if (
            tty_fd >= 0
            and os.isatty(tty_fd)
            and os.tcgetpgrp(tty_fd) == os.getpgrp()
        ):
            return
    except (AttributeError, OSError, ValueError):
        pass
    print(
        "Jarvis V3 Calendar authorization must be run by the operator in an "
        "attached foreground Terminal.",
        file=sys.stderr,
    )
    raise SystemExit(4)


def _require_exact_root_wrapper(runtime_module, *, caller_frame=None) -> None:
    """Bind auth authority to this repository's exact reviewed root wrapper."""

    try:
        declared = Path(__file__)
        expected = (
            Path(runtime_module.__file__).resolve(strict=True).parents[2]
            / "launch_jarvis_v3_calendar_auth.py"
        )
        declared_lstat = declared.lstat()
        expected_stat = expected.stat()
        declared_resolved = declared.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
        caller = caller_frame if caller_frame is not None else sys._getframe(1)
        caller_file = Path(caller.f_code.co_filename).resolve(strict=True)
        caller_global_file = Path(caller.f_globals.get("__file__", "")).resolve(strict=True)
        caller_name = caller.f_globals.get("__name__")
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        reason = "root-wrapper provenance could not be verified"
    else:
        if (
            declared_resolved == expected_resolved
            and caller_file == expected_resolved
            and caller_global_file == expected_resolved
            and caller_name == "__main__"
            and stat.S_ISREG(declared_lstat.st_mode)
            and (declared_lstat.st_dev, declared_lstat.st_ino)
            == (expected_stat.st_dev, expected_stat.st_ino)
            and declared_lstat.st_uid == os.getuid()
            and not stat.S_IMODE(declared_lstat.st_mode) & 0o022
        ):
            return
        reason = "root-wrapper provenance did not match"
    print(
        "Jarvis V3 Calendar authorization was refused before environment or "
        f"authorization startup: {reason}. Run the exact root launcher from an "
        "attached foreground Terminal.",
        file=sys.stderr,
    )
    raise SystemExit(4)


def _usage() -> str:
    return (
        "Usage: ./launch_jarvis_v3_calendar_auth.py {readonly|full-access}\n"
        "  readonly    authorize bounded Calendar reads only\n"
        "  full-access authorize approval-gated Calendar mutations"
    )


def _load_selected_v3_environment() -> None:
    """Load the explicitly selected owner-only V3 environment for auth paths."""

    if not os.getenv("JARVIS_V3_ENV", "").strip():
        print(
            "Jarvis V3 Calendar authorization requires an explicit owner-only "
            "JARVIS_V3_ENV. Use the command documented in QUICKSTART.md.",
            file=sys.stderr,
        )
        raise SystemExit(78)
    from jarvis_v2.env import load_env

    try:
        load_env()
    except (OSError, RuntimeError, UnicodeError, ValueError):
        print(
            "Jarvis V3 Calendar authorization could not load the selected "
            "owner-only environment safely. Repair it using QUICKSTART.md.",
            file=sys.stderr,
        )
        raise SystemExit(78) from None


def _authorized_main(*, authority: object, runtime_module) -> int:
    if authority is not _DIRECT_EXECUTION_AUTHORITY:
        print(
            "Jarvis V3 Calendar authorization requires the exact root launcher.",
            file=sys.stderr,
        )
        return 4
    _require_exact_root_wrapper(runtime_module, caller_frame=sys._getframe(1))
    arguments = sys.argv[1:]
    if arguments in (["-h"], ["--help"]):
        print(_usage())
        return 0
    if arguments not in (["readonly"], ["full-access"]):
        print(_usage(), file=sys.stderr)
        return 2
    _require_human_foreground()
    if arguments == ["readonly"]:
        from jarvis_v2.scripts.google_calendar_readonly_auth import main as auth_main
    else:
        from jarvis_v2.scripts.google_calendar_reauth import main as auth_main
    return auth_main()


if __name__ == "__main__":
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        print(_usage())
        raise SystemExit(0)
    _require_human_foreground()
    from jarvis_v2.scripts import v3_python_runtime

    _require_exact_root_wrapper(v3_python_runtime)
    v3_python_runtime.reexec_with_v3_python(Path(__file__))
    _require_exact_root_wrapper(v3_python_runtime)
    _require_human_foreground()
    _load_selected_v3_environment()
    raise SystemExit(
        _authorized_main(
            authority=_DIRECT_EXECUTION_AUTHORITY,
            runtime_module=v3_python_runtime,
        )
    )
