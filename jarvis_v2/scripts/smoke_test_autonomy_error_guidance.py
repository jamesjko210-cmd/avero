"""Offline proof for canonical autonomy missing-input recovery guidance."""

from __future__ import annotations

import ast
from pathlib import Path

from jarvis_v2.agent.failure_guidance import LOCAL_READ_INPUT_RECOVERY_ACTION
from jarvis_v2.tools.autonomy import make_autonomy_tools


PRIVATE_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")
EXPECTED_TOOLS = {
    "autonomy_plan",
    "risk_preflight",
    "agent_loop_preview",
    "agent_loop_packet",
    "risky_request_lifecycle",
    "action_readiness_packet",
    "execution_contract",
    "argument_contract_packet",
    "verification_packet",
    "execution_acceptance_gate",
    "execution_readiness_matrix",
    "dispatch_decision_packet",
    "command_intake_packet",
    "execution_governor_packet",
    "planner_gap_packet",
    "command_cockpit_packet",
}


def _assert_guided(result, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded")
    guidance = result.metadata.get("recovery_guidance")
    if guidance != {
        "version": 1,
        "action": LOCAL_READ_INPUT_RECOVERY_ACTION,
        "commands": [],
    }:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result.metadata}")
    if LOCAL_READ_INPUT_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} hid its recovery action: {result.output!r}")
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
        ("executes_side_effect", False),
        ("external_side_effect", False),
        ("controls_computer", False),
        ("calls_external_service", False),
    ):
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} truth drifted for {key}: {result.metadata}")
    combined = result.output + repr(result.metadata)
    if any(fragment in combined for fragment in PRIVATE_FRAGMENTS):
        raise SystemExit(f"{label} leaked a private local path: {combined}")


def _assert_source_is_centralized() -> None:
    source_path = Path(__file__).parents[1] / "tools" / "autonomy.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_autonomy_input_refusal"
    ]
    if len(calls) != 16:
        raise SystemExit(f"expected 16 centralized autonomy refusals, found {len(calls)}")
    raw_false_results = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ToolResult"
        and len(node.args) >= 4
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value is False
    ]
    if len(raw_false_results) != 1:
        raise SystemExit(f"expected one centralized false ToolResult, found {len(raw_false_results)}")
    metadata = raw_false_results[0].args[3]
    if not any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "declare_retryable_local_read_failure"
        for child in ast.walk(metadata)
    ):
        raise SystemExit("central autonomy refusal bypassed canonical guidance")


def main() -> None:
    _assert_source_is_centralized()
    tools = make_autonomy_tools(lambda: [], store=None)
    names = {tool.__name__ for tool in tools}
    if names != EXPECTED_TOOLS:
        raise SystemExit(f"autonomy tool set drifted: {sorted(names)}")
    for tool in tools:
        result = tool({})
        if result.tool_name != tool.__name__:
            raise SystemExit(f"{tool.__name__} result identity drifted: {result.tool_name}")
        _assert_guided(result, tool.__name__)

    special_metadata = {
        tool.__name__: tool({}).metadata
        for tool in tools
        if tool.__name__
        in {
            "dispatch_decision_packet",
            "command_intake_packet",
            "execution_governor_packet",
            "planner_gap_packet",
            "command_cockpit_packet",
        }
    }
    for name, metadata in special_metadata.items():
        if metadata.get("reason") != "missing_request":
            raise SystemExit(f"{name} lost its refusal reason: {metadata}")
        handoff = metadata.get(name.removesuffix("_packet") + "_handoff")
        if not isinstance(handoff, dict):
            handoff = metadata.get(name + "_handoff")
        if not isinstance(handoff, dict) or handoff.get("next_command") not in handoff.get(
            "recommended_next_commands", []
        ):
            raise SystemExit(f"{name} lost its front-door recovery handoff: {metadata}")

    print("Autonomy failure guidance smoke test passed (16 missing-input paths).")


if __name__ == "__main__":
    main()
