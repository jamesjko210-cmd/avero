"""Offline smoke coverage for Jarvis-note failure recovery declarations."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.failure_guidance import LOCAL_READ_INPUT_RECOVERY_ACTION
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.notes import (
    NOTE_TOO_LARGE_RECOVERY_ACTION,
    PROFILE_READ_RECOVERY_ACTION,
)


PRIVATE_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")
RETRYABLE_READ_FIELDS = {
    "outcome_known": True,
    "outcome_unknown": False,
    "execution_outcome_unknown": False,
    "side_effect_possible": False,
    "retry_safe": True,
    "automatic_retry_allowed": False,
    "authorizes_retry": False,
}


def _assert_public(value: object, label: str) -> None:
    rendered = str(value)
    if any(fragment in rendered for fragment in PRIVATE_FRAGMENTS):
        raise SystemExit(f"{label} leaked a private local path: {rendered}")


def _assert_retryable_read(
    result: object,
    *,
    action: str,
    commands: list[str],
    reason: str,
    label: str,
) -> None:
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid its public recovery action: {result}")
    expected_guidance = {
        "version": 1,
        "action": action,
        "commands": commands,
    }
    if result.metadata.get("recovery_guidance") != expected_guidance:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    if result.metadata.get("reason") != reason:
        raise SystemExit(f"{label} refusal reason drifted: {result.metadata}")
    for key, expected in RETRYABLE_READ_FIELDS.items():
        if result.metadata.get(key) != expected:
            raise SystemExit(f"{label} field {key} drifted: {result.metadata}")
    if commands:
        if result.metadata.get("next_command") != commands[0]:
            raise SystemExit(f"{label} lost its first recovery command: {result.metadata}")
        if result.metadata.get("recovery_commands") != commands:
            raise SystemExit(f"{label} recovery commands drifted: {result.metadata}")
    _assert_public(result.output, f"{label} output")
    _assert_public(result.metadata, f"{label} metadata")


def _assert_outcome_unknown_write(result: object) -> None:
    if result.ok or "outcome unknown" not in result.output:
        raise SystemExit(f"write I/O failure hid outcome uncertainty: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": result.output,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"write I/O recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": False,
        "outcome_unknown": True,
        "execution_outcome_unknown": True,
        "side_effect_possible": True,
        "retry_safe": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"write I/O field {key} drifted: {result.metadata}")
    if "do not retry automatically" not in result.output:
        raise SystemExit(f"write I/O failure invited an unsafe replay: {result.output}")
    _assert_public(result.output, "write I/O output")
    _assert_public(result.metadata, "write I/O metadata")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-note-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        tools = {
            name: runtime.registry.get(name).handler
            for name in (
                "list_jarvis_notes",
                "search_jarvis_notes",
                "read_jarvis_note",
                "outline_jarvis_note",
                "write_jarvis_note",
            )
        }

        folder_file = runtime.vault.root_path / "Projects" / "Folder.md"
        folder_file.parent.mkdir(parents=True, exist_ok=True)
        folder_file.write_text("# Folder\n", encoding="utf-8")

        input_cases = (
            (
                tools["list_jarvis_notes"]({"folder": "/\x55sers/example/private/notes"}),
                "unsafe_folder",
                "absolute note-list folder",
            ),
            (
                tools["list_jarvis_notes"]({"folder": "Projects/Folder.md"}),
                "not_folder",
                "file used as note-list folder",
            ),
            (
                tools["search_jarvis_notes"]({"query": ""}),
                "missing_query",
                "empty note search",
            ),
            (
                tools["read_jarvis_note"]({"path": ""}),
                "missing_path",
                "empty note read path",
            ),
            (
                tools["read_jarvis_note"]({"path": "../private"}),
                "unsafe_path",
                "unsafe note read path",
            ),
            (
                tools["outline_jarvis_note"]({"path": ""}),
                "missing_path",
                "empty note outline path",
            ),
            (
                tools["outline_jarvis_note"]({"path": "../private"}),
                "unsafe_path",
                "unsafe note outline path",
            ),
        )
        for result, reason, label in input_cases:
            _assert_retryable_read(
                result,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                commands=[],
                reason=reason,
                label=label,
            )

        for tool_name in ("read_jarvis_note", "outline_jarvis_note"):
            protected = tools[tool_name]({"path": "Profile.md"})
            _assert_retryable_read(
                protected,
                action=PROFILE_READ_RECOVERY_ACTION,
                commands=["read profile"],
                reason="protected_profile",
                label=f"{tool_name} protected profile",
            )

        with patch.object(runtime.vault, "read_note_bounded", side_effect=OverflowError):
            for tool_name in ("read_jarvis_note", "outline_jarvis_note"):
                oversized = tools[tool_name]({"path": "Projects/Large.md"})
                _assert_retryable_read(
                    oversized,
                    action=NOTE_TOO_LARGE_RECOVERY_ACTION,
                    commands=[],
                    reason="note_too_large",
                    label=f"{tool_name} oversized note",
                )

        with patch.object(
            runtime.vault,
            "append_note",
            side_effect=OSError("disk failure near /\x55sers/example/private/note"),
        ):
            write_failure = tools["write_jarvis_note"](
                {"path": "Projects/Write.md", "body": "body"}
            )
        _assert_outcome_unknown_write(write_failure)

    print("Note error-guidance smoke test passed.")


if __name__ == "__main__":
    main()
