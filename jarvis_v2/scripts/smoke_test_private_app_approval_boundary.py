from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.scripts.test_runtime import (
    approve_pending_runtime_approval,
    make_temp_runtime,
)
from jarvis_v2.tools import computer, system


@dataclass(frozen=True)
class PrivateAppCase:
    tool_name: str
    command: str
    script: str
    stdout: str


CASES = (
    PrivateAppCase(
        tool_name="frontmost_app",
        command="what app is focused",
        script='tell application "System Events" to get name of first process whose frontmost is true',
        stdout="Mock Focus App\n",
    ),
    PrivateAppCase(
        tool_name="list_running_apps",
        command="show running apps",
        script='tell application "System Events" to get name of every process whose background only is false',
        stdout="Mock Notes, Mock Browser\n",
    ),
)


def _held_approval_id(result, case: PrivateAppCase) -> int:
    if (
        len(result.plan.actions) != 1
        or result.plan.actions[0].tool_name != case.tool_name
        or result.plan.actions[0].args != {}
        or len(result.tool_results) != 1
    ):
        raise SystemExit(f"Private-app command routed incorrectly: {case.command!r} -> {result}")

    held = result.tool_results[0]
    approval_id = held.metadata.get("approval_id")
    if (
        result.verified
        or held.tool_name != case.tool_name
        or held.ok
        or held.metadata.get("failure_kind") != "approval_required"
        or held.metadata.get("requires_confirmation") is not True
        or held.metadata.get("risk_level") != RiskLevel.PERSONAL_DATA.name
        or held.metadata.get("planned_args") != {}
        or held.metadata.get("executed_handler") is not False
        or held.metadata.get("handler_invoked") is True
        or type(approval_id) is not int
    ):
        raise SystemExit(f"Private-app command crossed or malformed its approval hold: {result}")
    return approval_id


def assert_private_app_boundary(case: PrivateAppCase) -> None:
    expected_call = ["osascript", "-e", case.script]
    completed = subprocess.CompletedProcess(
        args=expected_call,
        returncode=0,
        stdout=case.stdout,
        stderr="",
    )
    mocked_system_run = Mock(name="mocked_system_run", return_value=completed)
    forbidden_gui = Mock(
        name="forbidden_gui",
        side_effect=AssertionError("private-app reads must not invoke computer control"),
    )

    with TemporaryDirectory(prefix=f"jarvis-{case.tool_name}-approval-") as temp:
        with (
            patch.object(system, "_run", mocked_system_run),
            patch.object(computer, "_pyautogui", forbidden_gui),
        ):
            runtime = make_temp_runtime(Path(temp))
            tool = runtime.registry.get(case.tool_name)
            if tool.risk is not RiskLevel.PERSONAL_DATA:
                raise SystemExit(
                    f"{case.tool_name} must be PERSONAL_DATA; found {tool.risk.name}."
                )

            held = runtime.handle(case.command)
            approval_id = _held_approval_id(held, case)
            if mocked_system_run.call_count != 0 or forbidden_gui.call_count != 0:
                raise SystemExit(
                    f"Unapproved {case.tool_name} reached a system or computer action."
                )

            approval = runtime.store.get_pending_approval(approval_id)
            if (
                approval is None
                or approval["tool_name"] != case.tool_name
                or json.loads(str(approval["planned_args"] or "{}")) != {}
            ):
                raise SystemExit(
                    f"{case.tool_name} approval did not preserve its exact empty-argument action."
                )

            transition = approve_pending_runtime_approval(runtime, approval_id)
            if (
                transition.metadata.get("rerun_user_input") != case.command
                or mocked_system_run.call_count != 0
                or forbidden_gui.call_count != 0
            ):
                raise SystemExit(
                    f"Approval review/transition executed or changed {case.tool_name}: {transition}"
                )

            approved = runtime.handle(
                case.command,
                approved=True,
                approved_approval_id=approval_id,
            )
            if (
                not approved.verified
                or len(approved.plan.actions) != 1
                or approved.plan.actions[0].tool_name != case.tool_name
                or approved.plan.actions[0].args != {}
                or len(approved.tool_results) != 1
                or approved.tool_results[0].tool_name != case.tool_name
                or approved.tool_results[0].ok is not True
                or approved.tool_results[0].metadata.get("handler_invoked") is not True
                or approved.tool_results[0].metadata.get("approval_rerun_exact") is not True
                or approved.tool_results[0].metadata.get("reads_personal_data") is not True
                or approved.tool_results[0].metadata.get("reads_private_data") is not True
            ):
                raise SystemExit(f"Approved exact {case.tool_name} rerun failed: {approved}")
            if mocked_system_run.call_count != 1:
                raise SystemExit(
                    f"Approved exact {case.tool_name} rerun invoked AppleScript "
                    f"{mocked_system_run.call_count} times instead of once."
                )
            mocked_system_run.assert_called_once_with(expected_call)
            if forbidden_gui.call_count != 0:
                raise SystemExit(f"Approved {case.tool_name} invoked computer control.")

            repeated_approval = runtime.registry.get("approve_pending_approval").handler(
                {"approval_id": approval_id}
            )
            if repeated_approval.ok or mocked_system_run.call_count != 1:
                raise SystemExit(
                    f"Repeated approval re-authorized {case.tool_name}: {repeated_approval}"
                )

            replay = runtime.handle(
                case.command,
                approved=True,
                approved_approval_id=approval_id,
            )
            if (
                replay.verified
                or replay.plan.actions
                or replay.tool_results
                or replay.plan.metadata.get("approval_rerun_binding_failure")
                != "approval_already_used"
                or mocked_system_run.call_count != 1
                or forbidden_gui.call_count != 0
            ):
                raise SystemExit(f"Consumed {case.tool_name} approval replayed: {replay}")


def main() -> None:
    for case in CASES:
        assert_private_app_boundary(case)
    print("Private app approval-boundary smoke passed.")


if __name__ == "__main__":
    main()
