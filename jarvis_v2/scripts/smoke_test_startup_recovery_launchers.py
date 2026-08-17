"""Offline proof that recovery-audit custody loss has bounded launcher egress."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_v2.agent import runtime as runtime_module
from jarvis_v2.scripts import ask, chat, run_status_server, talk
from jarvis_v2.scripts.startup import (
    V3_BOOTSTRAP_CHECK_COMMAND,
    V3_DASHBOARD_COMMAND,
    V3_DASHBOARD_INFO_COMMAND,
    V3_DIAGNOSE_COMMAND,
    V3_ENV_COMMAND_PREFIX,
    StartupRecoveryUnavailable,
)


PRIVATE_MARKER = "/\x55sers/example/private/startup-recovery.sqlite"


def _run_bounded(label: str, action, argv: list[str], *, json_output: bool = False) -> None:
    stdout = StringIO()
    stderr = StringIO()
    original_argv = sys.argv
    sys.argv = argv
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                action()
            except SystemExit as exc:
                if exc.code != 3:
                    raise SystemExit(f"{label} recovery failure exited {exc.code}, expected 3") from exc
            else:
                raise SystemExit(f"{label} recovery failure did not exit")
    finally:
        sys.argv = original_argv

    output = stdout.getvalue()
    error_output = stderr.getvalue()
    rendered = output + error_output
    for forbidden in (PRIVATE_MARKER, "Traceback", "/\x55sers/example/private"):
        if forbidden in rendered:
            raise SystemExit(f"{label} recovery failure leaked unbounded detail: {rendered!r}")
    if json_output:
        if error_output:
            raise SystemExit(f"{label} JSON recovery failure wrote stderr: {error_output!r}")
        receipt = json.loads(output)
        if (
            receipt.get("route") != "startup"
            or receipt.get("exception_type") != "StartupRecoveryUnavailable"
            or receipt.get("ok") is not False
        ):
            raise SystemExit(f"{label} JSON recovery receipt diverged: {receipt}")
        message = str(receipt.get("message") or "")
        recovery_text = "\n".join(
            str(step) for step in receipt.get("safe_recovery", [])
        )
    else:
        if output:
            raise SystemExit(f"{label} text recovery failure wrote stdout: {output!r}")
        if f"{label} could not start." not in error_output:
            raise SystemExit(f"{label} missed bounded startup heading: {error_output!r}")
        message = error_output
        recovery_text = error_output
    for expected in ("storage status", "startup recovery report"):
        if expected not in message:
            raise SystemExit(f"{label} missed recovery guidance {expected!r}: {message!r}")
    for expected in (
        V3_BOOTSTRAP_CHECK_COMMAND,
        V3_DIAGNOSE_COMMAND,
        V3_DASHBOARD_COMMAND,
        V3_DASHBOARD_INFO_COMMAND,
    ):
        if expected not in recovery_text:
            raise SystemExit(
                f"{label} missed environment-preserving recovery command {expected!r}: "
                f"{recovery_text!r}"
            )
    if recovery_text.count(V3_ENV_COMMAND_PREFIX) != 5:
        raise SystemExit(
            f"{label} recovery did not preserve V3 environment custody: {recovery_text!r}"
        )
    for forbidden in (
        "`python3 -m jarvis_v2",
        "`python3 launch_jarvis_v3",
        "`./launch_jarvis_v3",
    ):
        if forbidden in recovery_text:
            raise SystemExit(
                f"{label} recovery exposed a bare command {forbidden!r}: {recovery_text!r}"
            )


def test_ask_chat_and_dashboard() -> None:
    failure = StartupRecoveryUnavailable()
    failure.__context__ = RuntimeError(PRIVATE_MARKER)

    with patch.object(ask, "JarvisRuntime", side_effect=failure):
        _run_bounded(
            "Jarvis ask",
            ask.main,
            ["jarvis-ask", "--json", "what time is it"],
            json_output=True,
        )

    with patch.object(chat, "JarvisRuntime", side_effect=failure):
        _run_bounded("Jarvis chat", chat.main, ["jarvis-chat"])

    auth = SimpleNamespace(valid=True, token="synthetic-owner-only-status-token-0001")
    with (
        patch.dict(os.environ, {"JARVIS_V3_ENABLE_DAEMONS": "1"}, clear=False),
        patch.object(run_status_server, "status_host_from_env", return_value="127.0.0.1"),
        patch.object(run_status_server, "status_port_from_env", return_value=8766),
        patch.object(run_status_server, "status_host_is_loopback", return_value=True),
        patch.object(run_status_server, "status_auth_config_from_env", return_value=auth),
        patch.object(run_status_server, "JarvisRuntime", side_effect=failure),
    ):
        _run_bounded(
            "Jarvis status dashboard",
            run_status_server.main,
            ["jarvis-status-dashboard"],
        )


def test_talk() -> None:
    failure = StartupRecoveryUnavailable()
    failure.__context__ = RuntimeError(PRIVATE_MARKER)
    with (
        patch.object(talk.shutil, "which", return_value="/usr/local/bin/ffmpeg"),
        patch.object(talk, "_voice_readiness_environment", return_value=({}, None)),
        patch.object(talk, "_voice_readiness_snapshot", return_value={"ready": True}),
        patch.object(talk, "_default_mic_index", return_value="1"),
        patch.object(runtime_module, "JarvisRuntime", side_effect=failure),
    ):
        _run_bounded("Jarvis talk", talk.main, ["jarvis-talk", "--no-speak"])


def main() -> None:
    test_ask_chat_and_dashboard()
    test_talk()
    print("Startup recovery launcher smoke passed")


if __name__ == "__main__":
    main()
