from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.model_status import MISSING_SPECIALIST_REQUEST_RECOVERY_ACTION


TOOLS = (
    "model_planner_prompt_preview",
    "specialist_action_proposal_contract",
    "specialist_cycle_ledger",
    "specialist_execution_handoff_packet",
    "specialist_execution_readiness",
    "specialist_handoff_quality_gate",
    "specialist_handoff_receipt",
    "specialist_model_draft",
    "specialist_orchestration_packet",
    "specialist_post_run_closure_packet",
    "specialist_proposal_completion_gate",
    "specialist_proposal_gate",
    "specialist_route_quality",
    "specialist_router_contract",
    "specialist_tool_dry_run_packet",
)


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-model-status-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for tool_name in TOOLS:
            result = runtime.registry.get(tool_name).handler({})
            if result.ok or result.metadata.get("reason") != "missing_request":
                raise SystemExit(f"{tool_name} did not fail closed on missing input: {result}")
            guidance = result.metadata.get("recovery_guidance")
            if guidance != {
                "version": 1,
                "action": MISSING_SPECIALIST_REQUEST_RECOVERY_ACTION,
                "commands": [],
            }:
                raise SystemExit(f"{tool_name} recovery guidance drifted: {result.metadata}")
            if MISSING_SPECIALIST_REQUEST_RECOVERY_ACTION not in result.output:
                raise SystemExit(f"{tool_name} hid its recovery action: {result.output}")
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
                    raise SystemExit(
                        f"{tool_name} truth field {key} drifted: {result.metadata}"
                    )
            public = f"{result.output}\n{result.metadata}"
            if any(marker in public for marker in ("/\x55sers/", "/private/", "/tmp/")):
                raise SystemExit(f"{tool_name} leaked a private path: {public}")

    print("Model-status failure-guidance smoke passed: 15 specialist input refusals")


if __name__ == "__main__":
    main()
