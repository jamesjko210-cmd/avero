"""Offline proof for setup-check Calendar read-only auth guidance."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.tools.system import setup_check


AUTH_COMMAND = (
    'JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" '
    "./launch_jarvis_v3_calendar_auth.py readonly"
)


def _completed(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


def _run_setup(credentials: Path, readonly_token: Path, mutation_token: Path):
    env = {
        "JARVIS_GOOGLE_CREDS": str(credentials),
        "JARVIS_GOOGLE_READONLY_TOKEN": str(readonly_token),
        "JARVIS_GOOGLE_TOKEN": str(mutation_token),
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch("jarvis_v2.tools.system._module_available", return_value=True),
        patch("jarvis_v2.tools.system.subprocess.run", side_effect=_completed),
        patch("jarvis_v2.tools.system.shutil.which", return_value="/usr/bin/pbpaste"),
    ):
        return setup_check({})


def _assert_auth_command(result, *, expected: bool, label: str) -> None:
    commands = result.metadata.get("next_commands") or []
    present = AUTH_COMMAND in commands
    if present is not expected:
        raise SystemExit(f"{label} Calendar auth guidance drifted: {commands}")
    if result.output.count(f"`{AUTH_COMMAND}`") != (1 if expected else 0):
        raise SystemExit(f"{label} Calendar auth command output drifted: {result.output}")
    if expected and (
        "normal macOS Terminal shell prompt" not in result.output
        or "not in Telegram" not in result.output
        or "Jarvis's `You:` prompt" not in result.output
    ):
        raise SystemExit(f"{label} Calendar auth output lost terminal-location guidance")
    if result.metadata.get("calls_external_service") is not False:
        raise SystemExit(f"{label} setup check must remain offline: {result.metadata}")
    if result.metadata.get("writes_files") is not False:
        raise SystemExit(f"{label} setup check must not create an OAuth token: {result.metadata}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-calendar-setup-guidance-") as temp:
        root = Path(temp)
        credentials = root / "credentials.json"
        readonly_token = root / "readonly-token.json"
        mutation_token = root / "mutation-token.json"
        credentials.write_text("{}", encoding="utf-8")

        missing = _run_setup(credentials, readonly_token, mutation_token)
        _assert_auth_command(missing, expected=True, label="missing token")

        readonly_token.write_text("{}", encoding="utf-8")
        readonly_token.chmod(0o644)
        insecure = _run_setup(credentials, readonly_token, mutation_token)
        _assert_auth_command(insecure, expected=True, label="invalid token")

        readonly_token.chmod(0o600)
        connected = _run_setup(credentials, readonly_token, mutation_token)
        _assert_auth_command(connected, expected=False, label="valid token")

        credentials.unlink()
        missing_credentials = _run_setup(credentials, readonly_token, mutation_token)
        _assert_auth_command(missing_credentials, expected=False, label="missing credentials")

    print("setup Calendar auth guidance smoke passed")


if __name__ == "__main__":
    main()
