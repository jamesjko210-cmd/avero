from __future__ import annotations

from jarvis_v2.agent.executor import (
    Executor,
    _approval_argument_resolution_failure,
    _internal_mutation_binding_failure,
    _internal_mutation_binding_leak_result,
    _stale_contact_approval_result,
)
from jarvis_v2.agent.types import PlannedAction, RiskLevel, ToolResult
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import (
    TOOL_ARGUMENT_CONTRACT_VERSION,
    Tool,
    ToolArgumentContract,
    ToolArgumentSpec,
    ToolArgumentType,
    ToolRegistry,
)


PRIVATE_MARKERS = (
    "/\x55sers/",
    "/private/",
    "/var/folders/",
    "/tmp/",
    "traceback",
)


def _assert_guidance(
    result: ToolResult,
    *,
    action: str,
    label: str,
    outcome_unknown: bool = False,
) -> None:
    if result.ok is not False or action not in result.output:
        raise SystemExit(f"{label} lost its visible recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": result.output if outcome_unknown else action,
        "commands": [],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    if result.metadata.get("authorizes_retry") is not False:
        raise SystemExit(f"{label} recovery guidance authorized a retry: {result.metadata}")
    public = f"{result.output}\n{result.metadata.get('recovery_guidance')}".lower()
    if any(marker.lower() in public for marker in PRIVATE_MARKERS):
        raise SystemExit(f"{label} leaked private failure detail: {public}")
    if outcome_unknown:
        expected = {
            "outcome_known": False,
            "outcome_unknown": True,
            "execution_outcome_unknown": True,
            "side_effect_possible": True,
            "retry_safe": False,
            "automatic_retry_allowed": False,
        }
        for key, value in expected.items():
            if result.metadata.get(key) is not value:
                raise SystemExit(
                    f"{label} outcome-unknown field {key} drifted: {result.metadata}"
                )


def main() -> None:
    contract = ToolArgumentContract(
        TOOL_ARGUMENT_CONTRACT_VERSION,
        (
            ToolArgumentSpec(
                "body",
                frozenset({ToolArgumentType.STRING}),
                True,
            ),
        ),
        False,
    )
    tool = Tool(
        "typed_tool",
        "Typed fixture.",
        RiskLevel.READ_ONLY,
        lambda _args: ToolResult("typed_tool", True, "ok"),
        "fixture",
        argument_contract=contract,
    )
    registry = ToolRegistry()
    registry.register(tool)
    executor = Executor(registry, PermissionPolicy())

    invalid = executor.execute(
        PlannedAction("typed_tool", {"body": 7}, "invalid typed fixture")
    )
    _assert_guidance(
        invalid,
        action="Correct the request fields and types before retrying.",
        label="argument-contract failure",
    )

    plan_blocked = executor.argument_contract_plan_blocked_result(
        PlannedAction("typed_tool", {"body": "valid"}, "companion fixture")
    )
    _assert_guidance(
        plan_blocked,
        action=(
            "Correct the invalid action's request fields and types, then submit the plan again."
        ),
        label="plan preflight block",
    )

    stale = _stale_contact_approval_result(
        PlannedAction(
            "send_imessage",
            {"to": "Fixture", "message": "hello"},
            "stale recipient fixture",
        )
    )
    _assert_guidance(
        stale,
        action=(
            "Reissue the send request so Jarvis can queue a fresh approval with the resolved "
            "handle."
        ),
        label="stale approved recipient",
    )

    resolver_failure = _approval_argument_resolution_failure(
        PlannedAction("typed_tool", {"body": "valid"}, "resolver fixture"),
        tool,
        status="resolver_unavailable",
    )
    _assert_guidance(
        resolver_failure,
        action="Reissue the request after checking the target.",
        label="approval target binding",
    )

    binding_failure = _internal_mutation_binding_failure(
        PlannedAction("typed_tool", {"body": "valid"}, "binding fixture")
    )
    _assert_guidance(
        binding_failure,
        action="Review the source state, then submit a fresh mutation request.",
        label="internal mutation binding",
    )

    binding_leak = _internal_mutation_binding_leak_result(
        PlannedAction("typed_tool", {"body": "valid"}, "binding leak fixture")
    )
    _assert_guidance(
        binding_leak,
        action="Review the target state before issuing any new mutation request.",
        label="private binding leak",
        outcome_unknown=True,
    )

    print("Executor failure-guidance smoke passed")


if __name__ == "__main__":
    main()
