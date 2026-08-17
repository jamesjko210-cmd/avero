from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime


PRIVATE_MARKERS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")


def _assert_guided_failure(
    result: Any,
    *,
    label: str,
    reason: str,
    action: str,
    commands: list[str] | None = None,
) -> None:
    expected_commands = commands or []
    if result.ok or result.metadata.get("reason") != reason:
        raise SystemExit(f"{label} did not fail with {reason}: {result}")
    if action not in result.output:
        raise SystemExit(f"{label} omitted its visible recovery action: {result.output}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": expected_commands,
    }:
        raise SystemExit(f"{label} canonical recovery declaration drifted: {result.metadata}")
    expected_flags = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, expected in expected_flags.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} field {key} drifted: {result.metadata}")
    if any(marker in result.output or marker in str(result.metadata) for marker in PRIVATE_MARKERS):
        raise SystemExit(f"{label} exposed a private local path: {result}")
    for key in (
        "calls_model",
        "executes_tools",
        "queues_approval",
        "executes_side_effect",
        "writes_files",
        "writes_memory",
        "writes_notes",
        "controls_computer",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if result.metadata.get(key):
            raise SystemExit(f"{label} unexpectedly set {key}: {result.metadata}")


def _preview(runtime: Any, path: Any) -> Any:
    return runtime.registry.get("preview_tasks_from_note").handler({"path": path})


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-task-preview-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))

        input_cases = (
            ("missing path", _preview(runtime, ""), "missing_path"),
            (
                "oversized path",
                _preview(runtime, "x" * 501),
                "path_too_large",
            ),
            (
                "outside path",
                _preview(runtime, "../outside.md"),
                "unsafe_path",
            ),
        )
        for label, result, reason in input_cases:
            _assert_guided_failure(
                result,
                label=label,
                reason=reason,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            )

        profile_preview = _preview(runtime, "Profile")
        _assert_guided_failure(
            profile_preview,
            label="protected profile preview",
            reason="protected_profile",
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            commands=["read profile"],
        )
        profile_import = runtime.registry.get("import_tasks_from_note").handler(
            {"path": "Profile", "priority": "normal"}
        )
        _assert_guided_failure(
            profile_import,
            label="protected profile import",
            reason="protected_profile",
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            commands=["read profile"],
        )

        folder = runtime.vault.root_path / "Projects" / "Folder.md"
        folder.mkdir(parents=True)
        _assert_guided_failure(
            _preview(runtime, "Projects/Folder.md"),
            label="nonregular note",
            reason="note_not_regular",
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        )

        original_read = runtime.vault.read_note_bounded
        read_failures = (
            (
                "oversized note",
                OverflowError("private oversized detail"),
                "note_too_large",
                LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
            (
                "unsafe note",
                ValueError("Note path must stay inside the Jarvis Obsidian folder."),
                "unsafe_path",
                LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
            (
                "unreadable note",
                OSError("permission denied near /\x55sers/example/private/tasks"),
                "note_read_failed",
                LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
            ),
        )
        try:
            for label, failure, reason, action in read_failures:
                def fail_read(*_args: Any, _failure: Exception = failure, **_kwargs: Any) -> None:
                    raise _failure

                runtime.vault.read_note_bounded = fail_read  # type: ignore[method-assign]
                _assert_guided_failure(
                    _preview(runtime, "Projects/Tasks.md"),
                    label=label,
                    reason=reason,
                    action=action,
                )
        finally:
            runtime.vault.read_note_bounded = original_read  # type: ignore[method-assign]

    print("Jarvis task note preview error-guidance smoke test passed.")


if __name__ == "__main__":
    main()
