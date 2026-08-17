#!/usr/bin/env python3
"""Launch the Jarvis V3 local status dashboard in the foreground.

This explicit operator entrypoint does not install or enable a background
service. The implementation keeps the ``jarvis_v2`` Python package name during
the compatibility migration.
"""

import argparse
import os
import stat
import sys
from pathlib import Path


def _precheck_manual_foreground() -> None:
    """Reject detached starts before selected-environment/runtime imports."""

    reason = "foreground Terminal ownership could not be verified"
    try:
        tty_fd = sys.stdin.fileno()
        if tty_fd < 0 or not os.isatty(tty_fd):
            reason = "standard input is not an attached Terminal"
        else:
            foreground_pgrp = os.tcgetpgrp(tty_fd)
            process_pgrp = os.getpgrp()
            if foreground_pgrp > 0 and foreground_pgrp == process_pgrp:
                return
            reason = "this process is not the Terminal foreground process group"
    except (AttributeError, OSError, ValueError):
        pass
    print(
        "Jarvis V3 dashboard manual launch was refused before environment, "
        f"authentication, or runtime startup: {reason}. Run the exact root launcher "
        "from an attached foreground Terminal.",
        flush=True,
    )
    raise SystemExit(4)


def _show_help() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Jarvis V3 read-only status dashboard."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host. Defaults to JARVIS_STATUS_HOST or 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Bind port. Defaults to JARVIS_STATUS_PORT or 8766.",
    )
    parser.parse_args()


def _require_exact_root_wrapper(runtime_module) -> None:
    """Bind launch authority to this repository's exact reviewed root wrapper."""

    try:
        declared = Path(__file__)
        expected = (
            Path(runtime_module.__file__).resolve(strict=True).parents[2]
            / "launch_jarvis_v3_dashboard.py"
        )
        declared_lstat = declared.lstat()
        expected_stat = expected.stat()
        declared_resolved = declared.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
        caller = sys._getframe(1)
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
        "Jarvis V3 dashboard manual launch was refused before environment, "
        f"authentication, or runtime startup: {reason}. Run the exact root launcher "
        "from an attached foreground Terminal.",
        flush=True,
    )
    raise SystemExit(4)

if __name__ == "__main__":
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        _show_help()
        raise SystemExit(0)
    _precheck_manual_foreground()
    from jarvis_v2.scripts import v3_python_runtime

    _require_exact_root_wrapper(v3_python_runtime)
    v3_python_runtime.reexec_with_v3_python(Path(__file__))
    _require_exact_root_wrapper(v3_python_runtime)

    from jarvis_v2.scripts.run_status_server import manual_foreground_main

    manual_foreground_main(wrapper_path=Path(__file__))
