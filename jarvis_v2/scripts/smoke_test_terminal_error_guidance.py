"""Offline proof for the bounded terminal error-guidance review.

This test covers the 20 terminal candidates owned by
``public_error_egress_inventory``.  It intentionally does not claim that every
possible terminal failure in Jarvis is statically discoverable.
"""

from __future__ import annotations

from pathlib import Path

from jarvis_v2.scripts.public_error_egress_inventory import (
    build_public_error_egress_inventory,
)
from jarvis_v2.scripts.startup import (
    V3_BOOTSTRAP_CHECK_COMMAND,
    V3_DASHBOARD_COMMAND,
    V3_DASHBOARD_INFO_COMMAND,
    V3_DIAGNOSE_COMMAND,
    V3_ENV_COMMAND_PREFIX,
    startup_failure_receipt,
)


EXPECTED_TERMINAL_COUNTS_BY_PATH = {
    "jarvis_v2/scripts/ask.py": 3,
    "jarvis_v2/scripts/chat.py": 1,
    "jarvis_v2/scripts/run_status_server.py": 9,
    "jarvis_v2/scripts/run_telegram_control.py": 7,
    "jarvis_v2/scripts/telegram_whoami.py": 1,
}


def _source(root: Path, relative_path: str) -> str:
    return (root / relative_path).read_text(encoding="utf-8")


def test_reviewed_terminal_candidate_census() -> None:
    root = Path(__file__).resolve().parents[2]
    inventory = build_public_error_egress_inventory(root)
    terminal = [
        candidate
        for candidate in inventory.candidates
        if candidate.category == "terminal_error_output"
    ]
    counts = {
        path: sum(candidate.path == path for candidate in terminal)
        for path in EXPECTED_TERMINAL_COUNTS_BY_PATH
    }
    if counts != EXPECTED_TERMINAL_COUNTS_BY_PATH or len(terminal) != 21:
        raise SystemExit(
            f"reviewed terminal error-guidance census drifted: count={len(terminal)} paths={counts}"
        )


def test_terminal_failures_offer_specific_recovery() -> None:
    root = Path(__file__).resolve().parents[2]
    telegram = _source(root, "jarvis_v2/scripts/run_telegram_control.py")
    whoami = _source(root, "jarvis_v2/scripts/telegram_whoami.py")
    dashboard = _source(root, "jarvis_v2/scripts/run_status_server.py")

    required_telegram = (
        "`model status`, then repair the local Ollama service",
        "run `setup check` and repair local storage access",
        "run `list scheduled jobs` and `setup check`",
        "scheduler ticker remains disabled; Telegram control will run without",
        "Set it from @BotFather, then restart the Jarvis Telegram service",
        "Run telegram_whoami, set the returned owner id, then restart",
        "offset custody validation failed before model or",
        "scheduler startup (",
        "Run `setup check`, repair the reported",
        "state issue, then restart",
    )
    required_whoami = (
        "Create or inspect the bot in @BotFather",
        "set TELEGRAM_BOT_TOKEN in .env",
        "re-run telegram_whoami",
    )
    required_dashboard = (
        "Run the exact root launcher",
        "attached foreground Terminal",
        "chmod 600 .env",
        "Use 127.0.0.1",
        "characters, then retry",
        "Choose another local port with --port or JARVIS_STATUS_PORT",
    )
    missing = [
        marker
        for source, markers in (
            (telegram, required_telegram),
            (whoami, required_whoami),
            (dashboard, required_dashboard),
        )
        for marker in markers
        if marker not in source
    ]
    if missing:
        raise SystemExit(f"terminal recovery guidance drifted: {missing}")


def test_shared_startup_failure_is_private_safe_and_actionable() -> None:
    receipt = startup_failure_receipt(OSError("/\x55sers/private/jarvis.sqlite"))
    if receipt.get("ok") is not False or receipt.get("verified") is not False:
        raise SystemExit(f"startup failure boundary drifted: {receipt}")
    rendered = repr(receipt)
    if "/\x55sers/private" in rendered:
        raise SystemExit(f"startup failure leaked a local path: {receipt}")
    recovery = receipt.get("safe_recovery")
    if not isinstance(recovery, list) or len(recovery) < 3:
        raise SystemExit(f"startup recovery steps missing: {receipt}")
    if not any(V3_BOOTSTRAP_CHECK_COMMAND in str(step) for step in recovery):
        raise SystemExit(f"startup recovery lacks a read-only storage check: {receipt}")
    joined = "\n".join(str(step) for step in recovery)
    for expected in (
        V3_DIAGNOSE_COMMAND,
        V3_DASHBOARD_COMMAND,
        V3_DASHBOARD_INFO_COMMAND,
    ):
        if expected not in joined:
            raise SystemExit(f"startup recovery missed canonical V3 command {expected!r}: {receipt}")
    if joined.count(V3_ENV_COMMAND_PREFIX) != 5:
        raise SystemExit(f"startup recovery did not preserve V3 environment custody: {receipt}")
    for forbidden in (
        "`python3 -m jarvis_v2",
        "`python3 launch_jarvis_v3",
        "`./launch_jarvis_v3",
    ):
        if forbidden in joined:
            raise SystemExit(f"startup recovery exposed bare command {forbidden!r}: {receipt}")


def main() -> None:
    test_reviewed_terminal_candidate_census()
    test_terminal_failures_offer_specific_recovery()
    test_shared_startup_failure_is_private_safe_and_actionable()
    print("terminal error guidance smoke test passed")


if __name__ == "__main__":
    main()
