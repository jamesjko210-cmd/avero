"""Focused offline proof for canonical memory-curator failure guidance."""

from __future__ import annotations

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.memory_curator import (
    MAX_MEMORY_BODY_CHARS,
    MEMORY_PROJECTION_RECOVERY_ACTION,
    MEMORY_READ_RECOVERY_ACTION,
    MEMORY_REFUSAL_RECOVERY_ACTION,
    make_memory_approval_resolvers,
    make_memory_curator_tools,
)


PRIVATE_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")


def _assert_guided(result, *, action: str, commands: list[str], label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded")
    expected = {"version": 1, "action": action, "commands": commands}
    if result.metadata.get("recovery_guidance") != expected:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result.metadata}")
    if action not in result.output:
        raise SystemExit(f"{label} hid its recovery action: {result.output}")
    if result.metadata.get("authorizes_execution") is not False:
        raise SystemExit(f"{label} authorized execution: {result.metadata}")
    if result.metadata.get("approval_granted") is not False:
        raise SystemExit(f"{label} granted approval: {result.metadata}")
    combined = result.output + repr(result.metadata)
    if any(fragment in combined for fragment in PRIVATE_FRAGMENTS):
        raise SystemExit(f"{label} leaked a private local path: {combined}")


def _assert_all_failure_sites_are_centralized() -> None:
    source_path = Path(__file__).parents[1] / "tools" / "memory_curator.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_memory_failure_result"
    ]
    if len(helper_calls) != 23:
        raise SystemExit(
            f"expected 23 centralized memory-curator failure sites, found {len(helper_calls)}"
        )

    raw_failures = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ToolResult"
            and len(node.args) >= 4
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value is False
        ):
            continue
        raw_failures.append(node.lineno)
    if len(raw_failures) != 1:
        raise SystemExit(f"raw failure construction escaped the central helper: {raw_failures}")


class _ProjectionStore:
    def delete_memory_exact_with_projection(self, *_args):
        return SimpleNamespace(status="deleted", projection_targets=[])

    def update_memory_exact_with_projection(self, *_args, **_kwargs):
        return SimpleNamespace(status="updated", projection_targets=[])

    def merge_memories_exact_with_projection(self, *_args):
        return SimpleNamespace(status="merged", projection_targets=[])


class _UnusedVault:
    root_path = Path(".")


def main() -> None:
    _assert_all_failure_sites_are_centralized()

    with TemporaryDirectory(prefix="jarvis-memory-curator-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        tools = {
            name: runtime.registry.get(name).handler
            for name in ("delete_memory", "get_memory", "edit_memory", "merge_memories")
        }
        resolvers = make_memory_approval_resolvers(runtime.store)

        default_cases = (
            ("delete invalid id", tools["delete_memory"]({"memory_id": "/\x55sers/example/private"})),
            ("get missing row", tools["get_memory"]({"memory_id": 999999})),
            ("edit invalid id", tools["edit_memory"]({"memory_id": "/private/example"})),
            (
                "edit oversized body",
                tools["edit_memory"](
                    {"memory_id": 1, "body": "x" * (MAX_MEMORY_BODY_CHARS + 1)}
                ),
            ),
            (
                "edit invalid confidence",
                tools["edit_memory"]({"memory_id": 1, "confidence": "/tmp/private"}),
            ),
            ("edit nonfinite confidence", tools["edit_memory"]({"memory_id": 1, "confidence": "nan"})),
            (
                "merge invalid ids",
                tools["merge_memories"]({"keep_id": "/var/folders/private", "delete_id": 2}),
            ),
            ("merge same id", tools["merge_memories"]({"keep_id": 1, "delete_id": 1})),
            ("delete approval invalid", resolvers[0]({"memory_id": "/\x55sers/example/private"})),
            ("edit approval absent", resolvers[1]({"memory_id": 999999, "title": "changed"})),
            ("merge approval same id", resolvers[2]({"keep_id": 2, "delete_id": 2})),
        )
        for label, result in default_cases:
            _assert_guided(
                result,
                action=MEMORY_REFUSAL_RECOVERY_ACTION,
                commands=[],
                label=label,
            )

        read_invalid = tools["get_memory"]({"memory_id": "/\x55sers/example/private"})
        _assert_guided(
            read_invalid,
            action=MEMORY_READ_RECOVERY_ACTION,
            commands=[],
            label="get invalid id",
        )

    projection_tools = make_memory_curator_tools(_ProjectionStore(), _UnusedVault())
    by_name = {tool.__name__: tool for tool in projection_tools}
    projection_cases = (
        (
            "delete projection pending",
            by_name["delete_memory"](
                {"memory_id": 1, "target_revision": 1, "target_binding": "binding"}
            ),
        ),
        (
            "edit projection pending",
            by_name["edit_memory"](
                {
                    "memory_id": 1,
                    "target_revision": 1,
                    "target_binding": "binding",
                    "title": "changed",
                }
            ),
        ),
        (
            "merge projection pending",
            by_name["merge_memories"](
                {
                    "keep_id": 1,
                    "keep_revision": 1,
                    "keep_binding": "keep",
                    "delete_id": 2,
                    "delete_revision": 1,
                    "delete_binding": "delete",
                }
            ),
        ),
    )
    for label, result in projection_cases:
        _assert_guided(
            result,
            action=MEMORY_PROJECTION_RECOVERY_ACTION,
            commands=["repair memory projections"],
            label=label,
        )
        if result.metadata.get("projection_pending") is not True:
            raise SystemExit(f"{label} lost its committed projection state: {result.metadata}")

    print("Memory curator failure-guidance smoke test passed.")


if __name__ == "__main__":
    main()
