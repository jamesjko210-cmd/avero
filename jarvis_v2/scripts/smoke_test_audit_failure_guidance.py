from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.failure_guidance import (
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.audit import make_audit_tools


SECRET_INPUT = "sk_" + "live_auditFailureGuidanceSecret123"


def _assert_guided(result, *, action: str, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} should fail closed: {result}")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("action") != action:
        raise SystemExit(f"{label} missed canonical guidance: {result.metadata}")
    if action not in result.output:
        raise SystemExit(f"{label} hid its recovery action: {result.output}")
    for key, expected in {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "controls_computer": False,
    }.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} truth field {key} drifted: {result.metadata}")
    public = f"{result.output}\n{result.metadata}"
    if SECRET_INPUT in public or any(
        marker in public for marker in ("/\x55sers/", "/private/", "/tmp/")
    ):
        raise SystemExit(f"{label} leaked private input: {public}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-audit-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.log_tool_run(
            "audit-guidance",
            "readiness_report",
            "READ_ONLY",
            True,
            False,
            "synthetic read-only proof",
        )
        tools = make_audit_tools(runtime.store)
        invalid_tools = {
            "verification receipt": tools[1],
            "runtime trace receipt": tools[2],
            "execution recovery": tools[4],
            "after-action learning": tools[5],
            "execution learning closure": tools[8],
        }
        for label, handler in invalid_tools.items():
            key = "message_id" if label == "runtime trace receipt" else "run_id"
            result = handler({key: SECRET_INPUT})
            _assert_guided(
                result,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                label=f"{label} invalid id",
            )

        missing_tools = {
            "verification receipt": tools[1],
            "runtime trace receipt": tools[2],
            "execution recovery": tools[4],
            "after-action learning": tools[5],
            "execution learning closure": tools[8],
        }
        for label, handler in missing_tools.items():
            key = "message_id" if label == "runtime trace receipt" else "run_id"
            result = handler({key: 999_999})
            _assert_guided(
                result,
                action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
                label=f"{label} missing id",
            )
            commands = result.metadata.get("recovery_commands")
            if not isinstance(commands, list) or not commands:
                raise SystemExit(f"{label} lost its existing recovery commands: {result.metadata}")
            if any(command not in result.output for command in commands):
                raise SystemExit(f"{label} recovery commands are not public: {result.output}")

    print("Audit failure-guidance smoke passed: 10 read-only identifier failures")


if __name__ == "__main__":
    main()
