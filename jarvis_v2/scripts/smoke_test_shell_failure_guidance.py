from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.shell import (
    MAX_COMMAND_CHARS,
    MAX_CWD_CHARS,
    _ProcessOutcome,
    run_shell_command,
)


def _outcome(
    *,
    returncode: int | None,
    stdout: bytes = b"",
    stderr: bytes = b"",
    timed_out: bool = False,
    post_start_failed: bool = False,
    exception_type: str | None = None,
) -> _ProcessOutcome:
    return _ProcessOutcome(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_bytes_seen=len(stdout),
        stderr_bytes_seen=len(stderr),
        capture_limit_bytes_per_stream=2000,
        streaming_started=True,
        timed_out=timed_out,
        post_start_failed=post_start_failed,
        exception_type=exception_type,
        process_reaped=True,
    )


def _assert_guidance(result: object, *, unknown: bool, side_effect_possible: bool, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} should fail: {result}")
    metadata = result.metadata
    declaration = metadata.get("recovery_guidance")
    if not isinstance(declaration, dict) or declaration.get("version") != 1:
        raise SystemExit(f"{label} missed canonical guidance: {metadata}")
    action = declaration.get("action")
    if not isinstance(action, str) or not action or action not in result.output:
        raise SystemExit(f"{label} guidance is not user-visible: {result}")
    if declaration.get("commands") != []:
        raise SystemExit(f"{label} must not expose executable recovery commands: {declaration}")
    expected = {
        "outcome_known": not unknown,
        "outcome_unknown": unknown,
        "execution_outcome_unknown": unknown,
        "side_effect_possible": side_effect_possible,
        "retry_safe": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "requires_approval": True,
        "approval_gated": True,
    }
    for key, value in expected.items():
        if metadata.get(key) is not value:
            raise SystemExit(f"{label} should keep {key}={value!r}: {metadata}")
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in result.output or fragment in str(metadata):
            raise SystemExit(f"{label} leaked local path material: {result}")


def _assert_prestart(result: object, label: str) -> None:
    _assert_guidance(result, unknown=False, side_effect_possible=False, label=label)
    for key in ("execution_started", "action_attempted", "process_spawned"):
        if result.metadata.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False: {result.metadata}")
    if result.metadata.get("executes_side_effect") is not False:
        raise SystemExit(f"{label} must keep the pre-start refusal boundary: {result.metadata}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-shell-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        if runtime.registry.get("run_shell_command").risk != RiskLevel.HIGH_RISK:
            raise SystemExit("run_shell_command must remain HIGH_RISK")

    prestart_cases = (
        ("missing command", {"command": ""}),
        ("oversized command", {"command": "x" * (MAX_COMMAND_CHARS + 1)}),
        ("blocked operator", {"command": "echo /tmp/private && echo no"}),
        ("parse error", {"command": "python3 '/private/unterminated"}),
        ("oversized cwd", {"command": "python3", "cwd": "x" * (MAX_CWD_CHARS + 1)}),
    )
    for label, args in prestart_cases:
        with patch("jarvis_v2.tools.shell._run_bounded_process") as runner:
            result = run_shell_command(args)
        if runner.called:
            raise SystemExit(f"{label} must not spawn a process")
        _assert_prestart(result, label)

    with patch(
        "jarvis_v2.tools.shell._run_bounded_process",
        side_effect=FileNotFoundError("/private/missing"),
    ):
        start_failed = run_shell_command(
            {"command": "missing-tool --secret", "cwd": "/private/tmp/shell-guidance"}
        )
    _assert_prestart(start_failed, "start failure")
    if start_failed.metadata.get("start_failed") is not True:
        raise SystemExit(f"start failure lost its classification: {start_failed.metadata}")

    with patch(
        "jarvis_v2.tools.shell._run_bounded_process",
        side_effect=RuntimeError("uncertain /private/detail"),
    ):
        launch_unknown = run_shell_command({"command": "python3 --version"})
    _assert_guidance(launch_unknown, unknown=True, side_effect_possible=True, label="launch exception")
    if launch_unknown.metadata.get("action_attempted") is not True:
        raise SystemExit(f"launch exception lost attempted-action truth: {launch_unknown.metadata}")

    attempted_cases = (
        (
            "timeout",
            _outcome(
                returncode=-15,
                stdout=b"partial /tmp/private-output\n",
                timed_out=True,
            ),
            True,
        ),
        (
            "capture failure",
            _outcome(
                returncode=-15,
                stderr=b"partial /\x55sers/private-output\n",
                post_start_failed=True,
                exception_type="OSError",
            ),
            True,
        ),
        (
            "nonzero exit",
            _outcome(returncode=7, stderr=b"failed /var/folders/private-output\n"),
            False,
        ),
    )
    for label, process_outcome, unknown in attempted_cases:
        with patch("jarvis_v2.tools.shell._run_bounded_process", return_value=process_outcome):
            result = run_shell_command({"command": "python3 --version", "max_chars": 500})
        _assert_guidance(result, unknown=unknown, side_effect_possible=True, label=label)
        if result.metadata.get("execution_started") is not True or result.metadata.get("action_attempted") is not True:
            raise SystemExit(f"{label} lost attempted-command truth: {result.metadata}")
        if "do not rerun automatically" not in result.output:
            raise SystemExit(f"{label} should prohibit automatic replay: {result.output}")

    with patch(
        "jarvis_v2.tools.shell._run_bounded_process",
        return_value=_outcome(returncode=0, stdout=b"ok\n"),
    ) as runner:
        succeeded = run_shell_command({"command": "python3 --version"})
    if not succeeded.ok or runner.call_args.args[0] != ["python3", "--version"]:
        raise SystemExit(f"successful argv execution changed: {succeeded}")
    if runner.call_args.kwargs.get("cwd") != Path("."):
        raise SystemExit(f"successful cwd changed: {runner.call_args}")
    if succeeded.metadata.get("recovery_guidance") is not None:
        raise SystemExit(f"successful command should not expose failure guidance: {succeeded.metadata}")

    print("Shell failure-guidance smoke test passed.")


if __name__ == "__main__":
    main()
