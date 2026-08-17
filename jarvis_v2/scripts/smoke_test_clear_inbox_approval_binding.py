from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from jarvis_v2.agent.types import Plan, PlannedAction
from jarvis_v2.memory import obsidian as obsidian_module
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import ToolArgumentType


TOOL_NAME = "clear_obsidian_inbox"
COMMAND = "clear inbox"
RESET_INBOX = "# Inbox\n\n"
ORIGINAL_MARKER = "CLEAR_INBOX_PRIVATE_ORIGINAL_8a17d01c"
CHANGED_MARKER = "CLEAR_INBOX_PRIVATE_CHANGED_94fc67b2"
ORIGINAL_INBOX = f"# Inbox\n\n- {ORIGINAL_MARKER}\n"
CHANGED_INBOX = f"# Inbox\n\n- {CHANGED_MARKER} must survive stale approval\n"
LOCAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9._~-])(?:/" r"Users/|/private/|/var/folders/|/tmp/)"
)


class StaticPlanner:
    def __init__(self, args: dict[str, Any]) -> None:
        self.args = dict(args)

    def plan(self, user_input: str) -> Plan:
        return Plan(
            user_input,
            [PlannedAction(TOOL_NAME, dict(self.args), "clear inbox binding smoke")],
            notes="static_clear_inbox_binding_fixture",
        )


def _inbox_path(runtime) -> Path:
    return runtime.vault.root_path / "Inbox.md"


def _write_inbox(runtime, content: str) -> None:
    _inbox_path(runtime).write_text(content, encoding="utf-8")


def _public_surfaces(value: Any, *, prefix: str | None = None) -> list[tuple[str, Any]]:
    surfaces: list[tuple[str, Any]] = []
    if prefix is None:
        tool_name = getattr(value, "tool_name", None)
        prefix = f"{tool_name} tool" if isinstance(tool_name, str) else "runtime"
    if hasattr(value, "response"):
        surfaces.append((f"{prefix} response", value.response))
    if hasattr(value, "output"):
        surfaces.append((f"{prefix} output", value.output))
    if hasattr(value, "metadata"):
        surfaces.append((f"{prefix} metadata", value.metadata))
    for result in getattr(value, "tool_results", ()):
        surfaces.extend(
            _public_surfaces(result, prefix=f"{result.tool_name} tool")
        )
    return surfaces


def _assert_private_surfaces(
    runtime,
    values: list[Any],
    *,
    context: str,
    markers: tuple[str, ...],
    path_values: list[Any],
    allowed_internal_paths: dict[str, str] | None = None,
) -> None:
    allowed_paths = allowed_internal_paths or {}
    exact_paths = {
        str(runtime.config.data_dir),
        str(runtime.vault.root_path),
        str(_inbox_path(runtime)),
    }

    def absolute_path_locations(value: Any, location: str) -> list[str]:
        if isinstance(value, dict):
            locations: list[str] = []
            for key, item in value.items():
                locations.extend(absolute_path_locations(item, f"{location}.{key}"))
            return locations
        if isinstance(value, (list, tuple)):
            locations = []
            for index, item in enumerate(value):
                locations.extend(absolute_path_locations(item, f"{location}[{index}]"))
            return locations
        if not isinstance(value, str):
            return []
        if allowed_paths.get(location) == value:
            return []
        if any(path and path in value for path in exact_paths) or LOCAL_PATH_RE.search(value):
            return [location]
        return []

    for value_index, value in enumerate(values):
        for label, surface in _public_surfaces(value):
            rendered = json.dumps(surface, sort_keys=True, default=str)
            if any(marker in rendered for marker in markers):
                raise SystemExit(
                    f"{context}: public surface {value_index} {label} exposed raw inbox content"
                )
    for value_index, value in enumerate(path_values):
        for label, surface in _public_surfaces(value):
            locations = absolute_path_locations(surface, label)
            if locations:
                raise SystemExit(
                    f"{context}: clear-inbox surface {value_index} exposed an absolute local path at "
                    + ", ".join(locations[:3])
                )


def _assert_contract(runtime) -> str:
    tool = runtime.registry.get(TOOL_NAME)
    raw = tool.argument_contract
    bound = tool.approval_argument_contract
    if raw is None or bound is None or tool.approval_argument_resolver is None:
        raise SystemExit("clear inbox is missing its two-phase approval contract")
    if raw.allow_unknown or raw.fields:
        raise SystemExit("clear inbox raw arguments are not a strict empty object")
    if bound.allow_unknown or len(bound.fields) != 1:
        raise SystemExit("clear inbox approved arguments are not exactly versioned-digest bound")
    binding = bound.fields[0]
    if (
        binding.name != "target_binding"
        or not binding.required
        or binding.types != frozenset({ToolArgumentType.STRING})
    ):
        raise SystemExit("clear inbox approved contract lacks its required opaque target binding")
    return binding.name


def _queue(runtime, content: str) -> tuple[int, dict[str, Any], Any, str]:
    binding_key = _assert_contract(runtime)
    held = runtime.handle(COMMAND)
    approvals = [
        result
        for result in held.tool_results
        if result.tool_name == TOOL_NAME
        and result.metadata.get("requires_confirmation") is True
    ]
    if len(approvals) != 1:
        raise SystemExit("raw clear inbox did not queue exactly one approval")
    approval_id = approvals[0].metadata.get("approval_id")
    if type(approval_id) is not int:
        raise SystemExit("clear inbox approval has no durable integer id")
    row = runtime.store.get_pending_approval(approval_id)
    try:
        bound_args = json.loads(row["planned_args"]) if row is not None else None
    except (json.JSONDecodeError, TypeError):
        bound_args = None
    if type(bound_args) is not dict or set(bound_args) != {binding_key}:
        raise SystemExit("clear inbox approval did not persist only its versioned digest binding")
    expected_binding = runtime.vault.inbox_clear_binding(
        max_bytes=len(content.encode("utf-8")) + 1
    )
    if bound_args[binding_key] != expected_binding:
        raise SystemExit("clear inbox approval binding does not match the reviewed inbox revision")
    stored = json.dumps(bound_args, sort_keys=True)
    if any(marker in stored for marker in (ORIGINAL_MARKER, CHANGED_MARKER)):
        raise SystemExit("stored clear inbox binding retained raw inbox content")
    if str(runtime.vault.root_path) in stored or str(_inbox_path(runtime)) in stored:
        raise SystemExit("stored clear inbox binding retained an absolute path")
    return approval_id, bound_args, held, binding_key


def test_binding_includes_vault_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-clear-inbox-identity-a-") as first_temp:
        with TemporaryDirectory(prefix="jarvis-clear-inbox-identity-b-") as second_temp:
            first = make_temp_runtime(Path(first_temp))
            second = make_temp_runtime(Path(second_temp))
            _write_inbox(first, ORIGINAL_INBOX)
            _write_inbox(second, ORIGINAL_INBOX)
            first_binding = first.vault.inbox_clear_binding(max_bytes=4096)
            second_binding = second.vault.inbox_clear_binding(max_bytes=4096)
            if (
                not isinstance(first_binding, str)
                or not isinstance(second_binding, str)
                or first_binding == second_binding
            ):
                raise SystemExit("clear inbox binding did not include canonical vault identity")


def test_invalid_utf8_refuses_before_approval() -> None:
    with TemporaryDirectory(prefix="jarvis-clear-inbox-invalid-utf8-") as temp:
        runtime = make_temp_runtime(Path(temp))
        invalid = b"# Inbox\n\n- invalid: \xff\xfe\n"
        _inbox_path(runtime).write_bytes(invalid)
        before_pending = len(runtime.store.list_pending_approvals(limit=100))
        result = runtime.handle(COMMAND)
        matches = [item for item in result.tool_results if item.tool_name == TOOL_NAME]
        if (
            len(matches) != 1
            or matches[0].ok
            or matches[0].metadata.get("reason") != "inbox_unreadable"
            or matches[0].metadata.get("handler_invoked") is not False
            or matches[0].metadata.get("requires_confirmation") is not False
        ):
            raise SystemExit("invalid UTF-8 inbox did not fail before approval")
        if len(runtime.store.list_pending_approvals(limit=100)) != before_pending:
            raise SystemExit("invalid UTF-8 inbox queued an approval")
        if _inbox_path(runtime).read_bytes() != invalid:
            raise SystemExit("invalid UTF-8 inbox changed during approval refusal")
        _assert_private_surfaces(
            runtime,
            [result],
            context="invalid UTF-8 refusal",
            markers=(),
            path_values=[result],
        )


def test_replaced_vault_root_refuses_matching_content() -> None:
    with TemporaryDirectory(prefix="jarvis-clear-inbox-root-swap-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _write_inbox(runtime, ORIGINAL_INBOX)
        approval_id, _args, held, _binding_key = _queue(runtime, ORIGINAL_INBOX)
        review = _review_and_approve(runtime, approval_id)

        original_root = runtime.vault.root_path
        reviewed_root = original_root.with_name(original_root.name + "-reviewed")
        original_root.rename(reviewed_root)
        original_root.mkdir(parents=True)
        replacement = original_root / "Inbox.md"
        replacement.write_text(ORIGINAL_INBOX, encoding="utf-8")

        stale = _approved_rerun(runtime, approval_id)
        result = stale.tool_results[0]
        if (
            result.ok
            or result.metadata.get("reason") != "stale_inbox_binding"
            or result.metadata.get("writes_files") is not False
        ):
            raise SystemExit("matching-content replacement vault did not fail stale approval")
        if replacement.read_text(encoding="utf-8") != ORIGINAL_INBOX:
            raise SystemExit("replacement vault inbox was cleared by a stale approval")
        if (reviewed_root / "Inbox.md").read_text(encoding="utf-8") != ORIGINAL_INBOX:
            raise SystemExit("reviewed vault changed during replacement-root refusal")
        _assert_private_surfaces(
            runtime,
            [held, *review, stale],
            context="replacement vault root",
            markers=(ORIGINAL_MARKER,),
            path_values=[held, stale],
        )


def test_late_invalid_byte_swap_rolls_back_exactly() -> None:
    with TemporaryDirectory(prefix="jarvis-clear-inbox-byte-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        reviewed = b"# Inbox\n\n- exact byte: \xef\xbf\xbd\n"
        replacement = b"# Inbox\n\n- exact byte: \xff\n"
        _inbox_path(runtime).write_bytes(reviewed)
        approval_id, _args, _held, _binding_key = _queue(
            runtime,
            reviewed.decode("utf-8"),
        )
        _review_and_approve(runtime, approval_id)
        original_replace = obsidian_module._replace_text

        def replace_after_byte_swap(root, path, content, **kwargs):
            _inbox_path(runtime).write_bytes(replacement)
            return original_replace(root, path, content, **kwargs)

        with mock.patch.object(
            obsidian_module,
            "_replace_text",
            side_effect=replace_after_byte_swap,
        ):
            stale = _approved_rerun(runtime, approval_id)
        result = stale.tool_results[0]
        if (
            result.ok
            or result.metadata.get("reason") != "stale_inbox_binding"
            or result.metadata.get("writes_files") is not False
        ):
            raise SystemExit("late invalid-byte swap did not fail exact-byte CAS")
        if _inbox_path(runtime).read_bytes() != replacement:
            raise SystemExit("exact-byte CAS did not restore the late replacement bytes")


def _review_and_approve(runtime, approval_id: int) -> tuple[Any, Any, Any]:
    readiness = runtime.registry.get("approval_readiness_packet").handler(
        {"approval_id": approval_id}
    )
    packet = runtime.registry.get("approval_execution_packet").handler(
        {"approval_id": approval_id}
    )
    if not readiness.ok or not packet.ok:
        raise SystemExit("clear inbox approval review setup failed")
    transition = runtime.registry.get("approve_pending_approval").handler(
        {"approval_id": approval_id}
    )
    if (
        not transition.ok
        or transition.metadata.get("approved_approval_id") != approval_id
        or transition.metadata.get("rerun_user_input") != COMMAND
    ):
        raise SystemExit("clear inbox approval transition changed the reviewed request")
    return readiness, packet, transition


def _approved_rerun(runtime, approval_id: int):
    return runtime.handle(
        COMMAND,
        approved=True,
        approved_approval_id=approval_id,
    )


def _assert_failed_claim(runtime, approval_id: int) -> None:
    with runtime.store.connect() as conn:
        row = conn.execute(
            "SELECT outcome FROM approval_execution_claims WHERE approval_id = ?",
            (approval_id,),
        ).fetchone()
    if row is None or row["outcome"] != "failed":
        raise SystemExit("failed clear inbox approval did not finalize its execution claim")


def test_raw_queue_and_stale_rerun() -> None:
    with TemporaryDirectory(prefix="jarvis-clear-inbox-stale-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _write_inbox(runtime, ORIGINAL_INBOX)
        approval_id, _args, held, _binding_key = _queue(runtime, ORIGINAL_INBOX)
        review = _review_and_approve(runtime, approval_id)

        _write_inbox(runtime, CHANGED_INBOX)
        stale = _approved_rerun(runtime, approval_id)
        if len(stale.tool_results) != 1:
            raise SystemExit("stale clear inbox rerun returned the wrong result count")
        result = stale.tool_results[0]
        metadata = result.metadata
        semantic_values = {
            str(metadata.get("reason") or ""),
            str(metadata.get("target_binding_status") or ""),
            str(metadata.get("approval_rerun_block_reason") or ""),
        }
        if result.ok or metadata.get("approval_rerun_blocked") is not True:
            raise SystemExit("changed inbox did not fail its approved rerun closed")
        if not any(value in {"target_changed", "stale", "stale_inbox_binding"} for value in semantic_values):
            raise SystemExit("changed inbox refusal did not report stale binding semantics")
        if not any("target_changed" in value for value in semantic_values):
            raise SystemExit("changed inbox refusal did not identify the changed target")
        if metadata.get("writes_files") is not False or metadata.get("writes_notes") is not False:
            raise SystemExit("stale clear inbox refusal did not report zero file/note writes")
        if _inbox_path(runtime).read_text(encoding="utf-8") != CHANGED_INBOX:
            raise SystemExit("stale clear inbox approval removed or changed the sentinel")
        _assert_failed_claim(runtime, approval_id)
        _assert_private_surfaces(
            runtime,
            [held, *review, stale],
            context="stale rerun",
            markers=(ORIGINAL_MARKER, CHANGED_MARKER),
            path_values=[held, stale],
        )


def test_unchanged_control_clears_exactly() -> None:
    with TemporaryDirectory(prefix="jarvis-clear-inbox-control-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _write_inbox(runtime, ORIGINAL_INBOX)
        approval_id, _args, held, _binding_key = _queue(runtime, ORIGINAL_INBOX)
        review = _review_and_approve(runtime, approval_id)
        cleared = _approved_rerun(runtime, approval_id)
        if (
            not cleared.verified
            or len(cleared.tool_results) != 1
            or cleared.tool_results[0].ok is not True
            or cleared.tool_results[0].metadata.get("handler_invoked") is not True
        ):
            raise SystemExit("unchanged clear inbox approval did not execute exactly once")
        if _inbox_path(runtime).read_text(encoding="utf-8") != RESET_INBOX:
            raise SystemExit("unchanged clear inbox approval did not write the canonical reset body")
        _assert_private_surfaces(
            runtime,
            [held, *review, cleared],
            context="unchanged control",
            markers=(ORIGINAL_MARKER,),
            path_values=[held, cleared],
            allowed_internal_paths={
                "clear_obsidian_inbox tool metadata.path": str(
                    _inbox_path(runtime).resolve()
                )
            },
        )


def test_legacy_and_malformed_bindings_fail_before_handler() -> None:
    with TemporaryDirectory(prefix="jarvis-clear-inbox-malformed-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _write_inbox(runtime, ORIGINAL_INBOX)
        _queued_id, bound_args, held, binding_key = _queue(runtime, ORIGINAL_INBOX)
        cases = (
            ("legacy", {}),
            ("missing binding", {"legacy_target_digest": bound_args[binding_key]}),
            ("malformed binding", {binding_key: None}),
        )
        surfaces: list[Any] = [held]
        clear_surfaces: list[Any] = [held]
        for label, malformed_args in cases:
            approval_id = runtime.store.add_pending_approval(
                runtime.session_id,
                COMMAND,
                TOOL_NAME,
                f"{label} clear inbox binding smoke",
                malformed_args,
            )
            review = _review_and_approve(runtime, approval_id)
            result = _approved_rerun(runtime, approval_id)
            if len(result.tool_results) != 1:
                raise SystemExit(f"{label} clear inbox approval returned the wrong result count")
            tool_result = result.tool_results[0]
            if (
                tool_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or tool_result.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(f"{label} clear inbox approval reached the handler")
            if _inbox_path(runtime).read_text(encoding="utf-8") != ORIGINAL_INBOX:
                raise SystemExit(f"{label} clear inbox approval changed Inbox.md")
            _assert_failed_claim(runtime, approval_id)
            surfaces.extend((*review, result))
            clear_surfaces.append(result)
        _assert_private_surfaces(
            runtime,
            surfaces,
            context="malformed approvals",
            markers=(ORIGINAL_MARKER,),
            path_values=clear_surfaces,
        )


def test_raw_unknown_args_reject_without_queue() -> None:
    with TemporaryDirectory(prefix="jarvis-clear-inbox-unknown-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _write_inbox(runtime, ORIGINAL_INBOX)
        runtime.planner = StaticPlanner(
            {"unexpected": f"{ORIGINAL_MARKER} {runtime.vault.root_path}"}
        )
        before_pending = len(runtime.store.list_pending_approvals(limit=100))
        result = runtime.handle("clear inbox with an unknown raw argument")
        matches = [item for item in result.tool_results if item.tool_name == TOOL_NAME]
        if (
            len(matches) != 1
            or matches[0].metadata.get("failure_kind") != "tool_arguments_invalid"
            or matches[0].metadata.get("handler_invoked") is not False
            or matches[0].metadata.get("requires_confirmation") is not False
        ):
            raise SystemExit("raw clear inbox unknown arguments did not fail before approval")
        if len(runtime.store.list_pending_approvals(limit=100)) != before_pending:
            raise SystemExit("raw clear inbox unknown arguments queued an approval")
        if _inbox_path(runtime).read_text(encoding="utf-8") != ORIGINAL_INBOX:
            raise SystemExit("raw clear inbox unknown arguments changed Inbox.md")
        _assert_private_surfaces(
            runtime,
            [result],
            context="raw unknown arguments",
            markers=(ORIGINAL_MARKER,),
            path_values=[result],
        )


def main() -> None:
    test_binding_includes_vault_identity()
    test_invalid_utf8_refuses_before_approval()
    test_replaced_vault_root_refuses_matching_content()
    test_late_invalid_byte_swap_rolls_back_exactly()
    test_raw_queue_and_stale_rerun()
    test_unchanged_control_clears_exactly()
    test_legacy_and_malformed_bindings_fail_before_handler()
    test_raw_unknown_args_reject_without_queue()
    print("Clear inbox approval binding smoke passed")


if __name__ == "__main__":
    main()
