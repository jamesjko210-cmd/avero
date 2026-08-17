"""Focused offline proof for canonical utility failure guidance."""

from __future__ import annotations

import ast
from pathlib import Path

from jarvis_v2.agent.failure_guidance import LOCAL_READ_INPUT_RECOVERY_ACTION
from jarvis_v2.tools import utilities


PRIVATE_MARKERS = (
    "/\x55sers/example/private/utility.txt",
    "/private/utility.txt",
    "/var/folders/zc/utility.txt",
    "/tmp/utility.txt",
    "sk_" + "live_SUPERSECRET123",
)


def _assert_canonical_failure(result, *, label: str, source: str, reason: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
    if result.tool_name != source:
        raise SystemExit(f"{label} source drifted: {result.tool_name!r}")
    expected_truth = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, expected in expected_truth.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} outcome truth drifted for {key}: {result.metadata}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": LOCAL_READ_INPUT_RECOVERY_ACTION,
        "commands": [],
    }:
        raise SystemExit(f"{label} guidance drifted: {result.metadata}")
    if LOCAL_READ_INPUT_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} hid the recovery action: {result.output!r}")
    if "next_command" in result.metadata or "recovery_commands" in result.metadata:
        raise SystemExit(f"{label} unexpectedly exposed an executable recovery command: {result.metadata}")
    if result.metadata.get("refusal_reason") != reason:
        raise SystemExit(f"{label} refusal reason drifted: {result.metadata}")
    handoff = result.metadata.get("utility_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed its utility handoff: {result.metadata}")
    if handoff.get("source") != source or handoff.get("status") not in {"refused", "error"}:
        raise SystemExit(f"{label} handoff identity drifted: {handoff}")
    if handoff.get("reason") != reason or handoff.get("refused") is not True:
        raise SystemExit(f"{label} handoff refusal truth drifted: {handoff}")
    for key in (
        "calls_model",
        "executes_tools",
        "queues_approval",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
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
            raise SystemExit(f"{label} utility boundary drifted for {key}: {result.metadata}")
    public_packet = f"{result.output}\n{result.metadata}".lower()
    for marker in PRIVATE_MARKERS:
        if marker.lower() in public_packet:
            raise SystemExit(f"{label} leaked private-looking input: {public_packet}")


def test_failure_branch_families() -> None:
    secret = PRIVATE_MARKERS[-1]
    cases = (
        (utilities.calculate({"expression": "open(1)"}), "calculate evaluation", "calculate", "evaluation_error"),
        (utilities.generate_password({"length": True}), "password bool length", "generate_password", "bad_length"),
        (utilities.generate_password({"length": secret}), "password secret length", "generate_password", "bad_length"),
        (utilities.transform_text({"mode": secret}), "transform missing text", "transform_text", "missing_text"),
        (utilities.transform_text({"text": "/\x55sers/example/private/utility.txt", "mode": "upper"}), "transform local path", "transform_text", "local_path_text"),
        (utilities.transform_text({"text": secret, "mode": "rot13"}), "transform unsupported", "transform_text", "unsupported_mode"),
        (utilities.transform_text({"text": secret, "mode": "repeat", "count": secret}), "transform invalid repeat", "transform_text", "invalid_count"),
        (utilities.transform_text({"text": secret, "mode": "repeat", "count": 21}), "transform repeat range", "transform_text", "count_out_of_range"),
        (utilities.calculate_bmi({"weight": "bad", "weight_unit": "kg", "height": 1.8, "height_unit": "m"}), "bmi bad value", "calculate_bmi", "bad_value"),
        (utilities.calculate_bmi({"weight": 70, "weight_unit": secret, "height": 1.8, "height_unit": "m"}), "bmi unsupported secret unit", "calculate_bmi", "unsupported_units"),
        (utilities.calculate_bmi({"weight": 0, "weight_unit": "kg", "height": 1.8, "height_unit": "m"}), "bmi non-positive", "calculate_bmi", "non_positive_value"),
        (utilities.calculate_bmi({"weight": 70, "weight_unit": "kg", "height": 4, "height_unit": "m"}), "bmi range", "calculate_bmi", "out_of_range"),
        (utilities.convert_units({"value": secret, "from_unit": "km", "to_unit": "m"}), "conversion bad secret value", "convert_units", "bad_value"),
        (utilities.convert_units({"value": "nan", "from_unit": "km", "to_unit": "m"}), "conversion non-finite", "convert_units", "non_finite_value"),
        (utilities.convert_units({"value": 1, "from_unit": secret, "to_unit": "m"}), "conversion secret unit", "convert_units", "unsupported_conversion"),
        (utilities.convert_units({"value": 1, "from_unit": "kg", "to_unit": "m"}), "conversion group mismatch", "convert_units", "unsupported_conversion"),
    )
    if len(cases) != 16:
        raise SystemExit(f"utility failure-guidance scope drifted: {len(cases)}/16")
    for result, label, source, reason in cases:
        _assert_canonical_failure(result, label=label, source=source, reason=reason)


def test_success_behavior_and_bounds_remain_intact() -> None:
    calculation = utilities.calculate({"expression": "12 * (4 + 3)"})
    if not calculation.ok or calculation.output != "12 * (4 + 3) = 84":
        raise SystemExit(f"deterministic calculation drifted: {calculation}")
    conversion = utilities.convert_units({"value": 10, "from_unit": "km", "to_unit": "m"})
    if not conversion.ok or conversion.output != "10 km = 10000 m":
        raise SystemExit(f"deterministic conversion drifted: {conversion}")
    transformed = utilities.transform_text({"text": "Hello Jarvis", "mode": "snake case"})
    if not transformed.ok or transformed.output != "hello_jarvis":
        raise SystemExit(f"deterministic transform drifted: {transformed}")
    password = utilities.generate_password({"length": 9999, "include_symbols": False})
    if not password.ok or len(password.output) != utilities.MAX_PASSWORD_LENGTH or not password.output.isalnum():
        raise SystemExit(f"password bounds drifted: {password.metadata}")
    bmi = utilities.calculate_bmi({"weight": 70, "weight_unit": "kg", "height": 1.75, "height_unit": "m"})
    if not bmi.ok or bmi.metadata.get("bmi_category") != "normal":
        raise SystemExit(f"BMI behavior drifted: {bmi}")


def test_all_literal_failure_constructors_are_canonical() -> None:
    source_path = Path(utilities.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    failures = []
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
        failures.append(node)
        metadata = node.args[3]
        if not (
            isinstance(metadata, ast.Call)
            and isinstance(metadata.func, ast.Name)
            and metadata.func.id == "declare_retryable_local_read_failure"
        ):
            raise SystemExit(
                f"utility literal failure at line {node.lineno} lacks canonical local-read guidance"
            )
    if len(failures) < 24:
        raise SystemExit(f"utility literal failure coverage unexpectedly shrank: {len(failures)}/24")


def main() -> None:
    test_failure_branch_families()
    test_success_behavior_and_bounds_remain_intact()
    test_all_literal_failure_constructors_are_canonical()
    print("Utilities failure-guidance smoke passed: 16 failure branches, 5 bounded successes, and 24 canonical constructors.")


if __name__ == "__main__":
    main()
