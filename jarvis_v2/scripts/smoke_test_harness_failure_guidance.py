from __future__ import annotations

import ast
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime
import jarvis_v2.tools.harness as harness_module
from jarvis_v2.tools.harness import (
    HARNESS_CASE_ID_RECOVERY_ACTION,
    HARNESS_CASE_RECOVERY_ACTION,
    HARNESS_EVIDENCE_RECOVERY_ACTION,
    HARNESS_GATE_CONFIGURATION_RECOVERY_ACTION,
    HARNESS_REQUEST_RECOVERY_ACTION,
)


MISSING_REQUEST_TOOLS = (
    "harness_cycle_preview",
    "harness_lifecycle_state",
    "harness_control_surface",
    "execution_proof_bundle",
    "execution_mission_control",
    "execution_case_handoff_packet",
    "save_execution_case",
    "execution_runbook",
)


def _assert_known_no_change(result: object, action: str, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} did not fail closed: {result}")
    expected_guidance = {"version": 1, "action": action, "commands": []}
    if result.metadata.get("recovery_guidance") != expected_guidance:
        raise SystemExit(f"{label} recovery guidance drifted: {result.metadata}")
    if action not in result.output:
        raise SystemExit(f"{label} hid its recovery action: {result.output}")
    for key, expected in {
        "state_changed": False,
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "executes_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_notes": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} truth field {key} drifted: {result.metadata}")
    public = f"{result.output}\n{result.metadata.get('recovery_guidance')}"
    if any(marker in public for marker in ("/\x55sers/", "/private/", "/tmp/")):
        raise SystemExit(f"{label} leaked a private path: {public}")


def _assert_all_literal_failures_are_declared() -> None:
    path = Path(harness_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "ToolResult":
            continue
        if len(node.args) < 2:
            continue
        ok_arg = node.args[1]
        if isinstance(ok_arg, ast.Constant) and ok_arg.value is False:
            failures.append(node)
    undeclared = [
        node.lineno
        for node in failures
        if "_known_no_change_failure_metadata" not in ast.unparse(node)
    ]
    if undeclared:
        raise SystemExit(f"Harness literal failures lack canonical guidance: {undeclared}")
    if len(failures) != 16:
        raise SystemExit(f"Harness failure-site inventory drifted: {len(failures)}")


def main() -> None:
    _assert_all_literal_failures_are_declared()
    with TemporaryDirectory(prefix="jarvis-harness-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))

        for tool_name in MISSING_REQUEST_TOOLS:
            _assert_known_no_change(
                runtime.registry.get(tool_name).handler({}),
                HARNESS_REQUEST_RECOVERY_ACTION,
                f"{tool_name} missing request",
            )

        for raw_id in ("not-a-number", 0):
            _assert_known_no_change(
                runtime.registry.get("inspect_execution_case").handler({"case_id": raw_id}),
                HARNESS_CASE_ID_RECOVERY_ACTION,
                f"execution case id {raw_id!r}",
            )

        _assert_known_no_change(
            runtime.registry.get("append_execution_case_evidence").handler({}),
            HARNESS_CASE_RECOVERY_ACTION,
            "append evidence without case",
        )

        saved = runtime.registry.get("save_execution_case").handler({"request": "safe local test case"})
        if not saved.ok or not saved.metadata.get("case_id"):
            raise SystemExit(f"Could not create isolated evidence case: {saved}")
        case_id = saved.metadata["case_id"]
        evidence_failures = (
            {"case_id": case_id},
            {
                "case_id": case_id,
                "summary": "verification receipt 12 confirmed output",
                "receipt_kind": "approval",
                "receipt_id": "12",
            },
            {
                "case_id": case_id,
                "summary": "verification receipt 12 confirmed output",
                "receipt_kind": "verification",
                "receipt_id": "13",
            },
            {
                "case_id": case_id,
                "summary": "verification receipt 999999 confirmed output",
            },
        )
        for index, args in enumerate(evidence_failures, start=1):
            _assert_known_no_change(
                runtime.registry.get("append_execution_case_evidence").handler(args),
                HARNESS_EVIDENCE_RECOVERY_ACTION,
                f"append evidence refusal {index}",
            )

        original_gates = harness_module.AGI_GATE_EVIDENCE
        try:
            harness_module.AGI_GATE_EVIDENCE = []
            no_gate = runtime.registry.get("agi_next_build_move").handler({})
        finally:
            harness_module.AGI_GATE_EVIDENCE = original_gates
        _assert_known_no_change(
            no_gate,
            HARNESS_GATE_CONFIGURATION_RECOVERY_ACTION,
            "AGI build move without configured gates",
        )

    print("Harness failure-guidance smoke passed: 16 known-no-change refusal sites")


if __name__ == "__main__":
    main()
