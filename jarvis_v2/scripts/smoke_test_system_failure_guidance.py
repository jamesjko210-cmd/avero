"""Offline canonical-guidance coverage for system tool failures."""

from __future__ import annotations

from jarvis_v2.tools import system


class _FailedProcess:
    returncode = 1
    stdout = ""
    stderr = "private failure /\x55sers/example/secret"


class _FailedClipboardProcess:
    returncode = 1

    def communicate(self, _input: bytes) -> None:
        return None


def _assert_guidance(result, *, outcome_unknown: bool) -> None:
    if result.ok:
        raise SystemExit(f"system failure unexpectedly succeeded: {result}")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("version") != 1:
        raise SystemExit(f"system failure missed canonical guidance: {result.metadata}")
    action = guidance.get("action")
    if not isinstance(action, str) or action not in result.output:
        raise SystemExit(f"system recovery action is not public: {result}")
    expected = {
        "outcome_known": not outcome_unknown,
        "outcome_unknown": outcome_unknown,
        "execution_outcome_unknown": outcome_unknown,
        "side_effect_possible": outcome_unknown,
        "retry_safe": not outcome_unknown,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"system failure has incorrect {key}: {result.metadata}")
    if outcome_unknown and "do not retry automatically" not in result.output:
        raise SystemExit(f"attempted system action lacks no-retry truth: {result}")
    if "/\x55sers/" in result.output or "private failure" in result.output:
        raise SystemExit(f"system failure leaked private diagnostic detail: {result.output}")


def main() -> None:
    original_run = system._run
    original_popen = system.subprocess.Popen
    try:
        def raise_private_failure(*_args, **_kwargs):
            raise RuntimeError("private failure /\x55sers/example/secret")

        system._run = raise_private_failure  # type: ignore[assignment]
        read_exceptions = [
            system.list_running_apps({}),
            system.frontmost_app({}),
        ]
        open_exception = system.open_application({"name": "FakeApp"})

        system._run = lambda *_args, **_kwargs: _FailedProcess()  # type: ignore[assignment]
        read_failures = [
            system.list_running_apps({}),
            system.frontmost_app({}),
            system.get_clipboard({"max_chars": "bad"}),
        ]
        attempted_failures = [
            system.open_application({"name": "FakeApp"}),
            system.volume({"level": 42}),
        ]

        calls: list[object] = []

        def failed_popen(command, *, stdin):
            calls.append((command, stdin))
            return _FailedClipboardProcess()

        system.subprocess.Popen = failed_popen  # type: ignore[assignment]
        attempted_failures.append(system.set_clipboard({"text": "safe test text"}))
        if len(calls) != 1:
            raise SystemExit(f"clipboard failure did not use the mocked seam exactly once: {calls}")

        def reject_unexpected_run(*_args, **_kwargs):
            raise AssertionError("pre-attempt refusal reached the OS execution seam")

        system._run = reject_unexpected_run  # type: ignore[assignment]
        known_no_action = [
            system.open_application({"name": ""}),
            system.open_application({"name": "/\x55sers/example/private/FakeApp.app"}),
            system.volume({"level": "loud"}),
            system.set_clipboard({"text": "x" * (system.MAX_CLIPBOARD_WRITE_CHARS + 1)}),
        ]
    finally:
        system._run = original_run
        system.subprocess.Popen = original_popen

    for result in read_exceptions + read_failures + known_no_action:
        _assert_guidance(result, outcome_unknown=False)
    for result in [open_exception] + attempted_failures:
        _assert_guidance(result, outcome_unknown=True)

    for result in read_exceptions + read_failures:
        if result.metadata.get("reads_private_data") is not True:
            raise SystemExit(f"private system read failure lost privacy truth: {result.metadata}")
    for result in attempted_failures:
        if result.metadata.get("action_attempted") is not True:
            raise SystemExit(f"attempted system action lost attempt truth: {result.metadata}")

    print("System failure guidance smoke passed")


if __name__ == "__main__":
    main()
