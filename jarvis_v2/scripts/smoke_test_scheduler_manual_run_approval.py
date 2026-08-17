from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.types import PlannedAction, RiskLevel, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def _assert_scheduler_action_held(command: str, tool_name: str) -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-manual-approval-") as temp:
        runtime = make_temp_runtime(Path(temp))
        tool = runtime.registry.get(tool_name)
        if tool.risk is not RiskLevel.EXTERNAL_SIDE_EFFECT:
            raise SystemExit(
                f"{tool_name} must be EXTERNAL_SIDE_EFFECT because it may authorize scheduled external delivery; "
                f"found {tool.risk.name}."
            )

        handler_calls = 0

        def forbidden_handler(_args: dict[str, object]) -> ToolResult:
            nonlocal handler_calls
            handler_calls += 1
            raise SystemExit(f"{tool_name} handler ran before explicit approval")

        runtime.registry._tools[tool_name] = replace(tool, handler=forbidden_handler)
        result = runtime.handle(command)

        if len(result.plan.actions) != 1 or result.plan.actions[0].tool_name != tool_name:
            raise SystemExit(f"{command!r} did not route to exactly one {tool_name} action: {result.plan}")
        if len(result.tool_results) != 1:
            raise SystemExit(f"{tool_name} approval hold returned the wrong result count: {result.tool_results}")
        held = result.tool_results[0]
        if (
            result.verified
            or held.ok
            or held.metadata.get("failure_kind") != "approval_required"
            or held.metadata.get("requires_confirmation") is not True
            or held.metadata.get("executed_handler") is not False
            or type(held.metadata.get("approval_id")) is not int
            or handler_calls != 0
        ):
            raise SystemExit(f"{tool_name} crossed or malformed its pre-execution approval hold: {result}")

        malformed_args: tuple[dict[str, object], ...]
        if tool_name == "run_due_jobs":
            malformed_args = ({"injected": True},)
        elif tool_name == "schedule_morning_brief":
            malformed_args = ({"time": 730}, {"time": "07:30", "injected": True})
        else:
            malformed_args = ({}, {"name": 42}, {"name": "Morning Brief", "injected": True})

        pending_before = len(runtime.store.list_pending_approvals(limit=100))
        for args in malformed_args:
            invalid = runtime.executor.execute(PlannedAction(tool_name, args))
            if (
                invalid.ok
                or invalid.metadata.get("failure_kind") != "tool_arguments_invalid"
                or invalid.metadata.get("executed_handler") is not False
                or handler_calls != 0
                or len(runtime.store.list_pending_approvals(limit=100)) != pending_before
            ):
                raise SystemExit(
                    f"Malformed {tool_name} arguments crossed contract, approval, or handler boundary: {args!r} -> {invalid}"
                )


def main() -> None:
    _assert_scheduler_action_held("run job morning brief now", "run_job_now")
    _assert_scheduler_action_held("run due jobs", "run_due_jobs")
    _assert_scheduler_action_held("schedule morning brief at 7:30am", "schedule_morning_brief")
    _assert_scheduler_action_held("resume morning brief", "resume_job")
    print("Scheduler external-delivery approval smoke test passed.")


if __name__ == "__main__":
    main()
