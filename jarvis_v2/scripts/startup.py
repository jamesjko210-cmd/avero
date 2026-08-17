from __future__ import annotations

import json
import re
import sqlite3
import sys
from typing import Any

from jarvis_v2.v3_commands import (
    V3_BOOTSTRAP_CHECK_COMMAND,
    V3_BOOTSTRAP_WRITE_COMMAND,
    V3_DASHBOARD_COMMAND,
    V3_DASHBOARD_INFO_COMMAND,
    V3_DIAGNOSE_COMMAND,
    V3_ENV_COMMAND_PREFIX,
)

LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)


class StartupRecoveryUnavailable(RuntimeError):
    """A sanitized fail-closed signal for startup projection-audit custody loss."""

    def __init__(self) -> None:
        super().__init__("Startup projection recovery audit is unavailable.")


def safe_startup_text(value: object) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", str(value or ""))


def _startup_storage_message(exc: BaseException) -> str:
    if isinstance(exc, StartupRecoveryUnavailable):
        return (
            "Jarvis startup recovery could not establish its durable audit receipt. Run "
            "`storage status`, repair the reported local storage issue, then retry startup. "
            "After startup succeeds, run `startup recovery report`."
        )
    if isinstance(exc, sqlite3.Error):
        return (
            "Jarvis storage could not open its SQLite database. Set JARVIS_DATA_DIR and "
            "JARVIS_DB_PATH to writable local paths, run "
            f"`{V3_BOOTSTRAP_CHECK_COMMAND}`, then retry startup."
        )
    return (
        "Jarvis configuration or storage could not start. Check that JARVIS_V3_ENV names a readable "
        "environment file, set JARVIS_DATA_DIR and JARVIS_DB_PATH to writable local paths, run "
        f"`{V3_BOOTSTRAP_CHECK_COMMAND}`, then retry startup."
    )


def startup_failure_receipt(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "verified": False,
        "route": "startup",
        "runtime_route": "startup",
        "error": exc.__class__.__name__,
        "exception_type": exc.__class__.__name__,
        "message": _startup_storage_message(exc),
        "db_path": "<local-path>",
        "data_dir": "<local-path>",
        "obsidian_vault": "<local-path>",
        "configured_paths_inspected": False,
        "safe_recovery": [
            "Use the same owner-only V3 environment selected at launch. The commands below use "
            "the standard ~/.jarvis_v3/runtime.env path; if you selected a custom V3 environment, "
            "substitute that same path in every JARVIS_V3_ENV prefix.",
            "Set JARVIS_DATA_DIR and JARVIS_DB_PATH to a writable local folder.",
            f"Run `{V3_BOOTSTRAP_CHECK_COMMAND}` to verify the configured paths without writing.",
            f"Run `{V3_BOOTSTRAP_WRITE_COMMAND}` only after the check reports configured storage ready.",
            f"Run `{V3_DIAGNOSE_COMMAND}` from the Jarvis V3 project folder to verify startup.",
            f"Run `{V3_DASHBOARD_COMMAND}` from the Jarvis V3 project folder for the read-only dashboard.",
            f"Or run `{V3_DASHBOARD_INFO_COMMAND}` for dashboard launch instructions.",
        ],
        "safety_boundary": {
            "started_runtime": False,
            "executed_tools": False,
            "queued_approval": False,
            "approved_request": False,
            "external_side_effect": False,
        },
    }


def print_startup_failure(exc: BaseException, *, json_output: bool = False, program: str = "Jarvis") -> None:
    receipt = startup_failure_receipt(exc)
    if json_output:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    print(f"{program} could not start.", file=sys.stderr)
    print(f"- error: {receipt['error']}: {receipt['message']}", file=sys.stderr)
    print(f"- database: {receipt['db_path']}", file=sys.stderr)
    print("- safe recovery:", file=sys.stderr)
    for command in receipt["safe_recovery"]:
        print(f"  - {command}", file=sys.stderr)


def is_startup_storage_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (StartupRecoveryUnavailable, OSError, sqlite3.Error)):
            return True
        current = current.__cause__ or current.__context__
    return False
