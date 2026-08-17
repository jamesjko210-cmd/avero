from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.memory import obsidian as obsidian_module
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import notes as notes_module


NOTE_READ_TOOLS = (
    "list_jarvis_notes",
    "search_jarvis_notes",
    "read_jarvis_note",
    "outline_jarvis_note",
)

EXPECTED_SHAPES = {
    "list_jarvis_notes": (
        ("folder", ("string",), False),
        ("limit", ("integer",), False),
    ),
    "search_jarvis_notes": (
        ("query", ("string",), True),
        ("limit", ("integer",), False),
    ),
    "read_jarvis_note": (
        ("path", ("string",), True),
        ("max_chars", ("integer",), False),
    ),
    "outline_jarvis_note": (("path", ("string",), True),),
}


class StaticPlanner:
    def __init__(self, action: PlannedAction):
        self.action = action

    def plan(self, _user_input: str) -> Plan:
        return Plan("Exercise note read argument preflight.", [self.action])


def _install_recording_handlers(runtime: Any) -> list[tuple[str, dict[str, Any]]]:
    private_read_calls: list[tuple[str, dict[str, Any]]] = []
    for name in NOTE_READ_TOOLS:
        tool = runtime.registry.get(name)

        def handler(args: dict[str, Any], *, tool_name: str = name) -> ToolResult:
            private_read_calls.append((tool_name, dict(args)))
            return ToolResult(
                tool_name,
                True,
                f"Mocked {tool_name} result.",
                {
                    "reads_private_data": True,
                    "calls_model": False,
                    "queues_approval": False,
                    "executes_side_effect": False,
                    "authorizes_execution": False,
                    "approval_granted": False,
                },
            )

        runtime.registry._tools[name] = replace(tool, handler=handler)
    return private_read_calls


def _assert_contract_shapes(runtime: Any) -> None:
    for name, expected in EXPECTED_SHAPES.items():
        contract = runtime.registry.get(name).argument_contract
        if contract is None or contract.allow_unknown is not False:
            raise SystemExit(f"{name} does not have a strict argument contract")
        actual = tuple(
            (
                field.name,
                tuple(sorted(argument_type.value for argument_type in field.types)),
                field.required,
            )
            for field in contract.fields
        )
        if actual != expected:
            raise SystemExit(f"{name} argument shape drifted: {actual}")
        if not isinstance(contract.fields, tuple) or any(
            not isinstance(field.types, frozenset) for field in contract.fields
        ):
            raise SystemExit(f"{name} argument contract is not structurally immutable")
        try:
            contract.allow_unknown = True  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            raise SystemExit(f"{name} argument contract can be mutated")


def _assert_valid_executor_inputs(
    runtime: Any,
    private_read_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    cases = (
        ("list_jarvis_notes", {}),
        ("list_jarvis_notes", {"folder": "Projects", "limit": 8}),
        ("search_jarvis_notes", {"query": "launch"}),
        ("search_jarvis_notes", {"query": "launch", "limit": 6}),
        ("read_jarvis_note", {"path": "Projects/Launch Plan"}),
        (
            "read_jarvis_note",
            {"path": "Projects/Launch Plan", "max_chars": 2400},
        ),
        ("outline_jarvis_note", {"path": "Projects/Launch Plan"}),
    )
    for name, args in cases:
        before = len(private_read_calls)
        result = runtime.executor.execute(
            PlannedAction(name, args, "Note read typed compatibility smoke.")
        )
        if (
            not result.ok
            or result.metadata.get("handler_invoked") is not True
            or private_read_calls[before:] != [(name, args)]
        ):
            raise SystemExit(f"{name} rejected compatible arguments: {args} / {result}")


def _assert_malformed_preflight(
    runtime: Any,
    private_read_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    malformed: tuple[tuple[str, Any, str, list[str]], ...] = (
        (
            "list_jarvis_notes",
            ["not", "an", "object"],
            "arguments_not_object",
            [],
        ),
        (
            "list_jarvis_notes",
            {"private_note": "hidden"},
            "unknown_arguments",
            ["<unknown>"],
        ),
        (
            "list_jarvis_notes",
            {"folder": ["Projects"]},
            "type_mismatch",
            ["folder"],
        ),
        ("list_jarvis_notes", {"limit": False}, "type_mismatch", ["limit"]),
        ("list_jarvis_notes", {"limit": "8"}, "type_mismatch", ["limit"]),
        ("list_jarvis_notes", {"limit": 1.5}, "type_mismatch", ["limit"]),
        ("search_jarvis_notes", {}, "missing_required", ["query"]),
        ("search_jarvis_notes", {"query": 7}, "type_mismatch", ["query"]),
        (
            "search_jarvis_notes",
            {"query": "launch", "limit": True},
            "type_mismatch",
            ["limit"],
        ),
        ("read_jarvis_note", {}, "missing_required", ["path"]),
        ("read_jarvis_note", {"path": False}, "type_mismatch", ["path"]),
        (
            "read_jarvis_note",
            {"path": "Projects/Launch Plan", "max_chars": "2400"},
            "type_mismatch",
            ["max_chars"],
        ),
        ("outline_jarvis_note", {}, "missing_required", ["path"]),
        (
            "outline_jarvis_note",
            {"path": {"private": "Projects/Launch Plan"}},
            "type_mismatch",
            ["path"],
        ),
        (
            "outline_jarvis_note",
            {"path": "Projects/Launch Plan", "max_chars": 1200},
            "unknown_arguments",
            ["<unknown>"],
        ),
    )
    for index, (name, args, expected_status, expected_keys) in enumerate(malformed):
        before = len(private_read_calls)
        runtime.planner = StaticPlanner(
            PlannedAction(name, args, "Note read malformed preflight smoke.")  # type: ignore[arg-type]
        )
        result = runtime.handle(f"exercise malformed note read arguments {index}")
        if len(result.tool_results) != 1:
            raise SystemExit(f"{name} runtime preflight lost its rejection result")
        item = result.tool_results[0]
        trace = result.metadata.get("runtime_trace") or {}
        if (
            item.metadata.get("failure_kind") != "tool_arguments_invalid"
            or item.metadata.get("handler_invoked") is not False
            or item.metadata.get("executed_handler") is not False
            or trace.get("executed_handler_count") != 0
            or private_read_calls[before:]
        ):
            raise SystemExit(
                f"{name} malformed arguments reached a private read: {args} / {item}"
            )
        if item.metadata.get("argument_validation_status") != expected_status:
            raise SystemExit(
                f"{name} returned the wrong preflight status: {args} / {item.metadata}"
            )
        actual_keys = (
            item.metadata.get("unknown_arg_keys")
            if expected_status == "unknown_arguments"
            else item.metadata.get("missing_arg_keys")
            if expected_status == "missing_required"
            else item.metadata.get("type_mismatch_arg_keys")
            if expected_status == "type_mismatch"
            else []
        )
        if actual_keys != expected_keys:
            raise SystemExit(
                f"{name} returned the wrong rejected keys: {args} / {item.metadata}"
            )
        if "private_note" in str(item.metadata):
            raise SystemExit("unknown private note argument leaked into rejection metadata")


def _assert_planner_aliases(
    runtime: Any,
    private_read_calls: list[tuple[str, dict[str, Any]]],
    planner: Any,
) -> None:
    runtime.planner = planner
    cases = {
        "show my recent notes": ("list_jarvis_notes", {}),
        "내 노트 보여줘": ("list_jarvis_notes", {}),
        "list jarvis notes in Projects": (
            "list_jarvis_notes",
            {"folder": "Projects"},
        ),
        "search my notes for launch": (
            "search_jarvis_notes",
            {"query": "launch"},
        ),
        "read my note Projects/Launch Plan": (
            "read_jarvis_note",
            {"path": "Projects/Launch Plan"},
        ),
        "outline jarvis note Projects/Launch Plan": (
            "outline_jarvis_note",
            {"path": "Projects/Launch Plan"},
        ),
    }
    for command, expected in cases.items():
        before = len(private_read_calls)
        plan = planner.plan(command)
        actual_plan = [(action.tool_name, action.args) for action in plan.actions]
        if actual_plan != [expected]:
            raise SystemExit(f"note read planner alias drifted: {command!r} -> {actual_plan}")
        result = runtime.handle(command)
        if (
            len(result.tool_results) != 1
            or result.tool_results[0].metadata.get("handler_invoked") is not True
            or private_read_calls[before:] != [expected]
        ):
            raise SystemExit(f"note read planner alias missed typed runtime: {command!r}")


def _assert_note_enumeration_rejects_symlinks(base: Path) -> None:
    runtime = make_temp_runtime(base / "symlink-enumeration")
    outside = runtime.vault.root_path.parent / "outside-private-note.md"
    sentinel = "external-note-symlink-sentinel"
    outside.write_text(sentinel, encoding="utf-8")
    alias = runtime.vault.root_path / "External Alias.md"
    alias.symlink_to(outside)

    original_read_text = Path.read_text
    outside_reads = 0

    def guarded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal outside_reads
        if path.resolve() == outside.resolve():
            outside_reads += 1
        return original_read_text(path, *args, **kwargs)

    with patch.object(Path, "read_text", guarded_read_text):
        search = runtime.handle(f"search my notes for {sentinel}")
        listing = runtime.handle("list jarvis notes")

    if outside_reads:
        raise SystemExit("note enumeration opened an out-of-vault symlink target")
    if not search.tool_results or search.tool_results[0].ok is not True:
        raise SystemExit("note search failed instead of ignoring an out-of-vault symlink")
    if not listing.tool_results or listing.tool_results[0].ok is not True:
        raise SystemExit("note listing failed instead of ignoring an out-of-vault symlink")
    exposed = f"{search.response}\n{listing.response}"
    if search.tool_results[0].metadata.get("count") != 0 or alias.name in exposed:
        raise SystemExit("note enumeration exposed an out-of-vault symlink target")

    outside_identity = (outside.stat().st_dev, outside.stat().st_ino)
    external_descriptor_opens = 0
    original_os_open = obsidian_module.os.open

    def guarded_os_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal external_descriptor_opens
        if dir_fd is None:
            fd = original_os_open(path, flags, mode)
        else:
            fd = original_os_open(path, flags, mode, dir_fd=dir_fd)
        opened_stat = obsidian_module.os.fstat(fd)
        if (opened_stat.st_dev, opened_stat.st_ino) == outside_identity:
            external_descriptor_opens += 1
        return fd

    enumerated_race = runtime.vault.root_path / "Enumerated Race.md"
    enumerated_race.write_text("contained race source", encoding="utf-8")
    original_iter_markdown = notes_module._iter_markdown
    enumerated_swap_happened = False

    def swapping_iter_markdown(root: Path):
        nonlocal enumerated_swap_happened
        for path in original_iter_markdown(root):
            if path == enumerated_race and not enumerated_swap_happened:
                path.unlink()
                path.symlink_to(outside)
                enumerated_swap_happened = True
            yield path

    with (
        patch.object(notes_module, "_iter_markdown", swapping_iter_markdown),
        patch.object(obsidian_module.os, "open", guarded_os_open),
    ):
        enumerated_result = runtime.handle(f"search my notes for {sentinel}")
    if not enumerated_swap_happened:
        raise SystemExit("note search enumeration-race fixture did not run")
    if (
        not enumerated_result.tool_results
        or enumerated_result.tool_results[0].ok is not True
        or enumerated_result.tool_results[0].metadata.get("count") != 0
        or external_descriptor_opens
    ):
        raise SystemExit("note search followed a symlink swapped after enumeration")

    final_open_race = runtime.vault.root_path / "Final Open Race.md"
    final_open_race.write_text("contained final-open source", encoding="utf-8")
    original_os_stat = obsidian_module.os.stat
    target_stat_calls = 0
    final_swap_happened = False

    def swapping_os_stat(path: Any, *args: Any, **kwargs: Any):
        nonlocal target_stat_calls, final_swap_happened
        result = original_os_stat(path, *args, **kwargs)
        if (
            path == final_open_race.name
            and kwargs.get("dir_fd") is not None
            and kwargs.get("follow_symlinks") is False
        ):
            target_stat_calls += 1
            if target_stat_calls == 2:
                final_open_race.unlink()
                final_open_race.symlink_to(outside)
                final_swap_happened = True
        return result

    external_descriptor_opens = 0
    with (
        patch.object(obsidian_module.os, "stat", swapping_os_stat),
        patch.object(obsidian_module.os, "open", guarded_os_open),
    ):
        final_result = runtime.handle(f"search my notes for {sentinel}")
    if not final_swap_happened or target_stat_calls < 2:
        raise SystemExit("note search final-open race fixture did not run")
    if (
        not final_result.tool_results
        or final_result.tool_results[0].ok is not True
        or final_result.tool_results[0].metadata.get("count") != 0
        or external_descriptor_opens
    ):
        raise SystemExit("note search followed a symlink swapped before descriptor open")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-note-read-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        planner = runtime.planner
        _assert_contract_shapes(runtime)
        private_read_calls = _install_recording_handlers(runtime)
        _assert_valid_executor_inputs(runtime, private_read_calls)
        _assert_malformed_preflight(runtime, private_read_calls)
        _assert_planner_aliases(runtime, private_read_calls, planner)
        _assert_note_enumeration_rejects_symlinks(Path(temp))
    print("Note read argument contract smoke passed")


if __name__ == "__main__":
    main()
