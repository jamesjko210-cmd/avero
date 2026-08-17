"""Focused offline proof for canonical goal-refusal recovery guidance."""

from __future__ import annotations

import ast
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.failure_guidance import LOCAL_READ_INPUT_RECOVERY_ACTION
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools.goals import (
    GOAL_INPUT_RECOVERY_ACTION,
    GOAL_NOT_FOUND_RECOVERY_ACTION,
    add_goal_step_auto_mutation_preflight_result,
    complete_goal_step_auto_mutation_preflight_result,
    create_goal_auto_mutation_preflight_result,
    export_goal_auto_mutation_preflight_result,
    set_goal_status_auto_mutation_preflight_result,
)


PRIVATE_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")


def _assert_guided(result, label: str, expected_action: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("version") != 1:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result.metadata}")
    if guidance.get("action") != expected_action or expected_action not in result.output:
        raise SystemExit(f"{label} hid the expected recovery action: {result}")
    commands = guidance.get("commands")
    if not isinstance(commands, list) or len(commands) > 1:
        raise SystemExit(f"{label} exposed unbounded recovery commands: {guidance}")
    if commands and commands[0] not in result.output:
        raise SystemExit(f"{label} recovery command was not user-visible: {result}")
    for key, expected in (
        ("outcome_known", True),
        ("outcome_unknown", False),
        ("execution_outcome_unknown", False),
        ("side_effect_possible", False),
        ("retry_safe", True),
        ("automatic_retry_allowed", False),
        ("authorizes_retry", False),
        ("authorizes_execution", False),
        ("approval_granted", False),
        ("state_changed", False),
        ("writes_files", False),
        ("writes_database", False),
        ("writes_memory", False),
        ("writes_notes", False),
    ):
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} failure truth drifted for {key}: {result.metadata}")
    combined = result.output + repr(result.metadata)
    if any(fragment in combined for fragment in PRIVATE_FRAGMENTS):
        raise SystemExit(f"{label} leaked a private local path: {combined}")


def _assert_all_false_results_are_guided() -> None:
    source_path = Path(__file__).parents[1] / "tools" / "goals.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    false_results = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ToolResult"
        and len(node.args) >= 4
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value is False
    ]
    if len(false_results) != 5:
        raise SystemExit(f"expected 5 centralized false ToolResults, found {len(false_results)}")
    for node in false_results:
        metadata = node.args[3]
        if not any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in {"declare_failure_guidance", "declare_retryable_local_read_failure"}
            for child in ast.walk(metadata)
        ):
            raise SystemExit(f"goal refusal at line {node.lineno} bypassed canonical guidance")

    centralized = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_goal_refusal_result"
        for node in ast.walk(tree)
    )
    if centralized != 15:
        raise SystemExit(f"expected 15 centralized goal-refusal sites, found {centralized}")


def main() -> None:
    _assert_all_false_results_are_guided()
    with TemporaryDirectory(prefix="jarvis-goal-guidance-") as temp:
        root = Path(temp)
        runtime = JarvisRuntime(
            JarvisConfig(
                data_dir=root,
                db_path=root / "jarvis.sqlite",
                obsidian_vault=root / "Vault",
                obsidian_root="Jarvis",
                use_model_planner=False,
            )
        )
        generic_cases = (
            ("create missing title", "create_goal", {"title": ""}),
            ("create wrong type", "create_goal", {"title": []}),
            ("create path title", "create_goal", {"title": "/\x55sers/example/private-goal"}),
            ("add bad id", "add_goal_step", {"goal_id": "bad", "body": "step"}),
            ("add missing body", "add_goal_step", {"goal_id": 1, "body": ""}),
            ("add invalid unicode", "add_goal_step", {"goal_id": 1, "body": "\ud800"}),
            ("complete bad id", "complete_goal_step", {"step_id": "bad"}),
            ("status bad id", "set_goal_status", {"goal_id": "bad", "status": "done"}),
            ("status invalid", "set_goal_status", {"goal_id": 1, "status": "unknown"}),
            ("export bad id", "export_goal", {"goal_id": "bad"}),
        )
        for label, tool_name, args in generic_cases:
            _assert_guided(
                runtime.registry.get(tool_name).handler(args),
                label,
                GOAL_INPUT_RECOVERY_ACTION,
            )

        for label, tool_name, args in (
            ("status missing goal", "goal_status", {"goal_id": 9999}),
            ("add missing goal", "add_goal_step", {"goal_id": 9999, "body": "step"}),
            ("set missing goal", "set_goal_status", {"goal_id": 9999, "status": "done"}),
            ("export missing goal", "export_goal", {"goal_id": 9999}),
            ("complete missing step", "complete_goal_step", {"step_id": 9999}),
        ):
            _assert_guided(
                runtime.registry.get(tool_name).handler(args),
                label,
                GOAL_NOT_FOUND_RECOVERY_ACTION,
            )

        _assert_guided(
            runtime.registry.get("list_goals").handler({"status": "unknown"}),
            "list invalid status",
            LOCAL_READ_INPUT_RECOVERY_ACTION,
        )
        _assert_guided(
            runtime.registry.get("goal_status").handler({"goal_id": "bad"}),
            "read bad id",
            LOCAL_READ_INPUT_RECOVERY_ACTION,
        )
        if runtime.store.list_goals(limit=20):
            raise SystemExit("goal refusal coverage changed durable goal state")

    preflight_cases = (
        ("create preflight", create_goal_auto_mutation_preflight_result({"title": ""}, "missing_title")),
        (
            "add preflight",
            add_goal_step_auto_mutation_preflight_result({"goal_id": 1, "body": ""}, "missing_body"),
        ),
        (
            "complete preflight",
            complete_goal_step_auto_mutation_preflight_result({"step_id": "bad"}, "bad_step_id"),
        ),
        (
            "status preflight",
            set_goal_status_auto_mutation_preflight_result({"goal_id": 1, "status": "bad"}, "bad_status"),
        ),
        (
            "export preflight",
            export_goal_auto_mutation_preflight_result({"goal_id": "bad"}, "bad_goal_id"),
        ),
    )
    for label, result in preflight_cases:
        _assert_guided(result, label, GOAL_INPUT_RECOVERY_ACTION)
        if result.metadata.get("handler_invoked") is not False:
            raise SystemExit(f"{label} preflight claimed handler execution: {result.metadata}")

    print("Goal failure guidance smoke test passed (19 refusal sites; 17 newly canonicalized).")


if __name__ == "__main__":
    main()
