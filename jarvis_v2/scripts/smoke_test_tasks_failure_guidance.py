from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import tasks as task_tools


PRIVATE_MARKERS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")


def _assert_prewrite_failure(
    result: Any,
    *,
    label: str,
    reason: str,
    action: str,
    commands: list[str] | None = None,
) -> None:
    expected_commands = commands or []
    if result.ok or result.metadata.get("reason") != reason:
        raise SystemExit(f"{label} did not preserve its refusal reason: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": expected_commands,
    }:
        raise SystemExit(f"{label} omitted canonical recovery guidance: {result}")
    if action not in result.output or any(command not in result.output for command in expected_commands):
        raise SystemExit(f"{label} hid its declared recovery path: {result.output}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "authorizes_task_mutation": False,
        "auto_mutation_effects_started": False,
        "state_changed": False,
        "writes_database": False,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} field {key} drifted: {result.metadata}")
    for key in (
        "queues_approval",
        "approves_request",
        "dismisses_request",
        "writes_files",
        "writes_memory",
        "writes_notes",
        "controls_computer",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if result.metadata.get(key):
            raise SystemExit(f"{label} crossed {key}: {result.metadata}")
    serialized = f"{result.output}\n{result.metadata}"
    if any(marker in serialized for marker in PRIVATE_MARKERS):
        raise SystemExit(f"{label} exposed a private local path: {serialized}")


def _receipt_count(runtime: Any) -> int:
    with runtime.store.connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM auto_mutation_receipts").fetchone()
    return int(row[0])


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-tasks-failure-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        import_tool = runtime.registry.get("import_tasks_from_note").handler

        deterministic_cases = (
            ("missing path", {"path": ""}, "missing_path"),
            ("oversized path", {"path": "x" * 501}, "path_too_large"),
            (
                "private absolute path",
                {"path": "/\x55sers/example/private/task-list.md"},
                "unsafe_path",
            ),
            (
                "bad priority",
                {"path": "Projects/Tasks.md", "priority": "/private/tmp/urgent"},
                "bad_priority",
            ),
        )
        for label, args, reason in deterministic_cases:
            _assert_prewrite_failure(
                import_tool(args),
                label=label,
                reason=reason,
                action=task_tools.TASK_INPUT_RECOVERY_ACTION,
            )

        folder_note = runtime.vault.root_path / "Projects" / "Folder.md"
        folder_note.mkdir(parents=True)
        _assert_prewrite_failure(
            import_tool({"path": "Projects/Folder.md"}),
            label="nonregular note",
            reason="note_not_regular",
            action=task_tools.TASK_INPUT_RECOVERY_ACTION,
        )

        original_read = runtime.vault.read_note_bounded
        try:
            def fail_read(*_args: Any, **_kwargs: Any) -> None:
                raise OSError("permission denied near /\x55sers/example/private/task-list")

            runtime.vault.read_note_bounded = fail_read  # type: ignore[method-assign]
            _assert_prewrite_failure(
                import_tool({"path": "Projects/Tasks.md"}),
                label="unreadable note",
                reason="note_read_failed",
                action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                commands=["setup check"],
            )
        finally:
            runtime.vault.read_note_bounded = original_read  # type: ignore[method-assign]

        profile = import_tool({"path": "Profile"})
        _assert_prewrite_failure(
            profile,
            label="protected profile",
            reason="protected_profile",
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            commands=["read profile"],
        )

        preflight = task_tools.make_import_tasks_from_note_auto_mutation_preflight_result(
            runtime.vault
        )
        _assert_prewrite_failure(
            preflight({"path": "Projects/Tasks.md"}, "note_read_failed"),
            label="semantic preflight read failure",
            reason="note_read_failed",
            action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
            commands=["setup check"],
        )
        _assert_prewrite_failure(
            preflight({"path": "", "priority": "normal"}, "missing_path"),
            label="semantic preflight missing path",
            reason="missing_path",
            action=task_tools.TASK_INPUT_RECOVERY_ACTION,
        )

        if runtime.store.list_tasks(status=None, limit=10) or _receipt_count(runtime):
            raise SystemExit("task import refusals crossed the database or receipt boundary")

    print("Jarvis task failure-guidance smoke test passed (9 prewrite branches).")


if __name__ == "__main__":
    main()
