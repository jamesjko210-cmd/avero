from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.executor import PermissionPolicy
from jarvis_v2.agent.types import PlannedAction, RiskLevel, ToolResult
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime
from jarvis_v2.tools.registry import Tool


# Tools that MUST always require explicit approval — the ones that touch the
# outside world, the user's real personal stores, or the computer directly.
# Pinned by name (not just by risk tier) so an accidental risk-level downgrade
# of any single one fails loudly here, independent of the general invariant
# below. If a tool is intentionally renamed, update this list in the same commit.
SAFEGUARD_MUST_APPROVE_TOOLS = (
    "send_kakao",
    "send_telegram",
    "send_imessage",
    "send_instagram_dm",
    "send_email",
    "call_contact",
    "call_kakao",
    "call_telegram",
    "call_instagram",
    "frontmost_app",
    "list_running_apps",
    "apple_reminders",
    "set_reminder",      # Schedules a future owner-Telegram delivery.
    "create_reminder",   # Writes to the macOS Reminders app.
    "create_event",
    "update_event",
    "delete_event",
    "click",
    "type_text",
    "move_mouse",
    "observe_act_verify",
    "enable_computer_control",
    "open_application",
    "open_jarvis_vault",
    "run_shell_command",
    "speak",
)


def test_permission_policy_safeguard_invariant() -> None:
    # Fable checkpoint (2026-07-10): the single most important safety property of
    # the whole agent harness is that risky actions cannot execute without
    # explicit approval. Verified live that sends/shell/computer-control all
    # block and route to approval, while confirming the subtle-but-correct split
    # that both Jarvis's internally stored Telegram reminders and macOS
    # Reminders writes stay approval-gated. This test locks that contract across the ENTIRE
    # registry so a future accidental risk-level downgrade of any dangerous tool
    # fails the aggregate immediately instead of silently opening a hole.
    with TemporaryDirectory(prefix="jarvis-safeguard-") as temp:
        runtime = make_temp_runtime(Path(temp))
        policy = PermissionPolicy()  # default max_auto_risk = LOCAL_SAFE
        tools = list(runtime.registry.list())

        # 1. Registry-wide invariant: unapproved execution is allowed IFF the
        #    tool is READ_ONLY or LOCAL_SAFE; PERSONAL_DATA / EXTERNAL_SIDE_EFFECT
        #    / HIGH_RISK must be blocked. And approval always unblocks.
        for tool in tools:
            unapproved = policy.check(tool, approved=False)
            expected_auto = tool.risk <= RiskLevel.LOCAL_SAFE
            if unapproved.allowed != expected_auto:
                raise SystemExit(
                    f"safeguard invariant broken: {tool.name!r} (risk {tool.risk.name}) "
                    f"unapproved-allowed={unapproved.allowed}, expected {expected_auto} "
                    "-- a risk-level downgrade may have opened an approval hole"
                )
            if tool.risk > RiskLevel.LOCAL_SAFE:
                if not unapproved.requires_confirmation:
                    raise SystemExit(
                        f"{tool.name!r} ({tool.risk.name}) blocked but did not flag requires_confirmation"
                    )
                approved = policy.check(tool, approved=True)
                if not approved.allowed:
                    raise SystemExit(
                        f"{tool.name!r} ({tool.risk.name}) stayed blocked even WITH approval"
                    )

        # 2. Explicit pin: every known-dangerous tool that is registered must be
        #    approval-gated by name, independent of the tier invariant above.
        registered = {tool.name: tool for tool in tools}
        for name in SAFEGUARD_MUST_APPROVE_TOOLS:
            tool = registered.get(name)
            if tool is None:
                continue  # not registered in this build/config; skip, don't fail
            if policy.check(tool, approved=False).allowed:
                raise SystemExit(
                    f"CRITICAL safeguard gap: {name!r} (risk {tool.risk.name}) executes WITHOUT approval"
                )


def test_application_launches_stop_before_handlers_without_approval() -> None:
    with TemporaryDirectory(prefix="jarvis-launch-approval-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for name, args in (
            ("open_application", {"name": "SyntheticApp"}),
            ("open_jarvis_vault", {}),
        ):
            registered = runtime.registry.get(name)
            calls: list[dict] = []

            def forbidden_handler(handler_args: dict, *, _name: str = name) -> ToolResult:
                calls.append(dict(handler_args))
                return ToolResult(_name, True, "handler ran")

            runtime.registry._tools[name] = Tool(
                registered.name,
                registered.description,
                registered.risk,
                forbidden_handler,
                registered.toolset,
                registered.auto_mutation_contract,
                registered.argument_contract,
                registered.approval_argument_resolver,
                registered.approval_argument_contract,
            )
            result = runtime.executor.execute(
                PlannedAction(name, args, "synthetic pre-handler launch gate"),
                approved=False,
            )
            if calls:
                raise SystemExit(f"{name} handler ran before explicit approval: {calls}")
            if result.ok or "explicit approval required" not in result.output:
                raise SystemExit(f"{name} did not stop at the pre-handler approval gate: {result}")
            if result.metadata.get("executed_handler") is not False:
                raise SystemExit(f"{name} pre-handler receipt was not explicit: {result.metadata}")
            if name == "open_jarvis_vault":
                injected = runtime.executor.execute(
                    PlannedAction(
                        name,
                        {"path": "/private/tmp/SyntheticVault"},
                        "synthetic injected launch argument",
                    ),
                    approved=False,
                )
                if calls:
                    raise SystemExit(f"{name} injected arguments reached the launch handler: {calls}")
                if injected.ok or injected.metadata.get("requires_confirmation") is not False:
                    raise SystemExit(f"{name} injected arguments reached approval: {injected}")
                if injected.metadata.get("executed_handler") is not False:
                    raise SystemExit(f"{name} injected argument receipt was not pre-handler: {injected}")

        launch_tool = runtime.registry.get("open_application")
        calls: list[dict] = []

        def forbidden_launch(handler_args: dict) -> ToolResult:
            calls.append(dict(handler_args))
            return ToolResult("open_application", True, "handler ran")

        runtime.registry._tools["open_application"] = Tool(
            launch_tool.name,
            launch_tool.description,
            launch_tool.risk,
            forbidden_launch,
            launch_tool.toolset,
            launch_tool.auto_mutation_contract,
            launch_tool.argument_contract,
            launch_tool.approval_argument_resolver,
            launch_tool.approval_argument_contract,
        )
        invalid_cases = (
            {},
            {"name": 42},
            {"name": "/private/tmp/Synthetic.app"},
        )
        for args in invalid_cases:
            result = runtime.executor.execute(
                PlannedAction("open_application", args, "synthetic invalid launch gate"),
                approved=False,
            )
            if calls:
                raise SystemExit(f"open_application invalid input reached the launch handler: {calls}")
            if result.ok or result.metadata.get("requires_confirmation") is not False:
                raise SystemExit(f"open_application invalid input reached approval: {result}")
            if result.metadata.get("executed_handler") is not False:
                raise SystemExit(f"open_application invalid input lacked a pre-handler receipt: {result}")


def test_speech_stops_before_handler_without_approval() -> None:
    with TemporaryDirectory(prefix="jarvis-speech-approval-") as temp:
        runtime = make_temp_runtime(Path(temp))
        registered = runtime.registry.get("speak")
        calls: list[dict] = []

        def forbidden_speech(handler_args: dict) -> ToolResult:
            calls.append(dict(handler_args))
            return ToolResult("speak", True, "handler ran")

        runtime.registry._tools["speak"] = Tool(
            registered.name,
            registered.description,
            registered.risk,
            forbidden_speech,
            registered.toolset,
            registered.auto_mutation_contract,
            registered.argument_contract,
            registered.approval_argument_resolver,
            registered.approval_argument_contract,
        )
        blocked = runtime.executor.execute(
            PlannedAction("speak", {"text": "Synthetic speech"}, "synthetic speech gate"),
            approved=False,
        )
        if calls or blocked.ok or "explicit approval required" not in blocked.output:
            raise SystemExit(f"speak did not stop before its handler: {calls} / {blocked}")
        if blocked.metadata.get("executed_handler") is not False:
            raise SystemExit(f"speak approval receipt was not pre-handler: {blocked}")

        invalid_cases = (
            {},
            {"text": 42},
            {"text": "/private/tmp/SyntheticSpeech.txt"},
            {"text": "x" * 5001},
            {"text": "Synthetic speech", "voice": "/private/tmp/SyntheticVoice"},
            {"text": "Synthetic speech", "unexpected": True},
        )
        for args in invalid_cases:
            refused = runtime.executor.execute(
                PlannedAction("speak", args, "synthetic invalid speech gate"),
                approved=False,
            )
            if calls:
                raise SystemExit(f"invalid speech input reached the speech handler: {calls}")
            if refused.ok or refused.metadata.get("requires_confirmation") is not False:
                raise SystemExit(f"invalid speech input reached approval: {refused}")
            if refused.metadata.get("executed_handler") is not False:
                raise SystemExit(f"invalid speech input lacked a pre-handler receipt: {refused}")


class HostileRow:
    def __init__(self, marker: str):
        self.marker = marker

    def __getitem__(self, key: str) -> object:
        raise RuntimeError(f"{self.marker}:{key}")

    def __str__(self) -> str:
        return self.marker


def assert_operator_limits(metadata: dict, label: str) -> None:
    if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} missed the operator stop-time/work-window override metadata: {metadata}")


def assert_no_future_authority(metadata: dict, label: str) -> None:
    for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False: {metadata}")


def assert_no_local_path_leak(text: str, label: str) -> None:
    if re.search(r"(?<![A-Za-z])/(?:Users|private|var/folders|tmp)/", text):
        raise SystemExit(f"{label} leaked a raw local path:\n{text}")


def assert_approval_failure_recovery(
    result,
    *,
    label: str,
    approval_id: int,
    expected_commands: list[str],
    reason: str,
) -> None:
    if result.ok or result.metadata.get("reason") != reason:
        raise SystemExit(f"{label} missed stable failure metadata: {result}")
    for command in expected_commands:
        if command not in result.output:
            raise SystemExit(f"{label} missed recovery command {command!r}: {result.output}")
    if result.metadata.get("next_command") != "pending approvals":
        raise SystemExit(f"{label} should refresh the queue first: {result.metadata}")
    if result.metadata.get("recovery_commands") != expected_commands:
        raise SystemExit(f"{label} missed ordered recovery metadata: {result.metadata}")
    if result.metadata.get("retry_requires_queue_refresh") is not True:
        raise SystemExit(f"{label} should require a queue refresh: {result.metadata}")
    if result.metadata.get("authorizes_retry") is not False:
        raise SystemExit(f"{label} should not authorize retry: {result.metadata}")
    if result.metadata.get("approval_id") != approval_id:
        raise SystemExit(f"{label} missed the bounded approval id: {result.metadata}")
    if result.metadata.get("writes_files") or result.metadata.get("writes_notes") or result.metadata.get("writes_memory"):
        raise SystemExit(f"{label} failure receipt must not claim a successful write: {result.metadata}")
    assert_no_future_authority(result.metadata, label)
    assert_operator_limits(result.metadata, label)


def test_last_look_rechecks_readiness_after_runtime_restart() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-last-look-restart-") as temp:
        root = Path(temp)
        first_runtime = make_temp_runtime(root)
        approval_id = first_runtime.store.add_pending_approval(
            first_runtime.session_id,
            "run command printf approval-last-look-restart",
            "run_shell_command",
            "approval required",
            planned_args={"command": "printf approval-last-look-restart"},
        )
        # A replacement runtime has no in-memory receipt. Its last-look packet
        # must recheck the durable queue rather than trap the owner in a loop.
        restarted_runtime = make_temp_runtime(root)
        packet = restarted_runtime.registry.get("approval_execution_packet").handler(
            {"approval_id": approval_id}
        )
        if (
            not packet.ok
            or packet.metadata.get("approval_readiness_rechecked") is not True
            or packet.metadata.get("approval_packet_viewed") is not True
            or "freshly rechecked" not in packet.output
        ):
            raise SystemExit(
                "last-look packet did not recover a durable pending approval after a runtime restart: "
                f"{packet}"
            )
        approval = restarted_runtime.registry.get("approve_pending_approval").handler(
            {"approval_id": approval_id}
        )
        if not approval.ok or approval.metadata.get("approved_approval_id") != approval_id:
            raise SystemExit(
                "freshly rechecked last-look packet did not preserve the one-shot approval gate: "
                f"{approval}"
            )


def test_dismiss_pending_approval_idempotency() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-dismiss-idempotency-") as temp:
        runtime = make_temp_runtime(Path(temp))
        dismiss = runtime.registry.get("dismiss_pending_approval").handler

        dismissed_id = runtime.store.add_pending_approval(
            runtime.session_id,
            "dismiss idempotency smoke",
            "run_shell_command",
            "approval required",
            planned_args={"command": "printf dismiss-idempotency-smoke"},
        )
        first = dismiss({"approval_id": dismissed_id})
        dismissed_row = runtime.store.get_approval(dismissed_id)
        if not first.ok or dismissed_row is None or str(dismissed_row["status"]).lower() != "dismissed":
            raise SystemExit(f"Initial dismissal did not persist the dismissed decision: {first}")

        approved_id = runtime.store.add_pending_approval(
            runtime.session_id,
            "approved dismissal refusal smoke",
            "run_shell_command",
            "approval required",
            planned_args={"command": "printf approved-dismissal-smoke"},
        )
        if not runtime.store.set_pending_approval_status(approved_id, "approved"):
            raise SystemExit("Could not prepare approved dismissal refusal fixture.")
        store_type = type(runtime.store)
        original_status_update = store_type.set_pending_approval_status
        unexpected_updates: list[tuple[int, str]] = []

        def forbid_status_update(_store, approval_id, status):
            unexpected_updates.append((approval_id, status))
            raise AssertionError("terminal approval decisions must not be overwritten")

        store_type.set_pending_approval_status = forbid_status_update
        try:
            repeated = dismiss({"approval_id": dismissed_id})
            refused = dismiss({"approval_id": approved_id})
        finally:
            store_type.set_pending_approval_status = original_status_update
        if unexpected_updates:
            raise SystemExit(f"Terminal dismissal attempted an approval status update: {unexpected_updates}")
        if (
            not repeated.ok
            or repeated.metadata.get("reason") != "already_dismissed"
            or repeated.metadata.get("idempotent_noop") is not True
            or repeated.metadata.get("approval_state_changed") is not False
            or repeated.metadata.get("actual_status") != "dismissed"
            or "already dismissed" not in repeated.output.lower()
        ):
            raise SystemExit(f"Repeated dismissal was not a truthful successful no-op: {repeated}")
        if repeated.metadata.get("writes_files") or repeated.metadata.get("writes_notes"):
            raise SystemExit(f"Repeated dismissal no-op must not claim writes: {repeated.metadata}")
        approved_row = runtime.store.get_approval(approved_id)
        if refused.ok or refused.metadata.get("actual_status") != "approved":
            raise SystemExit(f"Dismissal must refuse an already-approved approval: {refused}")
        if approved_row is None or str(approved_row["status"]).lower() != "approved":
            raise SystemExit("Dismissal overwrote the winning approved decision.")


def main() -> None:
    test_permission_policy_safeguard_invariant()
    test_application_launches_stop_before_handlers_without_approval()
    test_speech_stops_before_handler_without_approval()
    test_last_look_rechecks_readiness_after_runtime_restart()
    test_dismiss_pending_approval_idempotency()
    with TemporaryDirectory(prefix="jarvis-approvals-") as temp:
        runtime = make_temp_runtime(Path(temp))
        approvals_note = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Pending Approvals.md"
        approval_review_note = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Approval Review.md"
        pending_views = 0
        polite_readonly_approval_cases = {
            "pending approval please": ("list_pending_approvals", {}),
            "approval list please": ("list_pending_approvals", {}),
            "show latest approvals": ("list_pending_approvals", {}),
            "show me pending approval": ("list_pending_approvals", {}),
            "what needs my approval": ("list_pending_approvals", {}),
            "what is waiting for approval": ("list_pending_approvals", {}),
            "what approvals are pending": ("list_pending_approvals", {}),
            "which approvals are pending": ("list_pending_approvals", {}),
            "what is pending approval": ("list_pending_approvals", {}),
            "anything waiting for approval": ("list_pending_approvals", {}),
            "anything waiting on approval": ("list_pending_approvals", {}),
            "anything I need to approve": ("list_pending_approvals", {}),
            "do I need to approve anything": ("list_pending_approvals", {}),
            "do you need approval from me": ("list_pending_approvals", {}),
            "what do you need me to approve": ("list_pending_approvals", {}),
            "pending approval queue please": ("list_pending_approvals", {}),
            "show waiting approvals": ("list_pending_approvals", {}),
            "show approval queue": ("list_pending_approvals", {}),
            "show latest approval queue": ("list_pending_approvals", {}),
            "do i have pending approvals": ("list_pending_approvals", {}),
            "are there pending approvals": ("list_pending_approvals", {}),
            "approval summary please": ("approval_queue_summary", {}),
            "show me approval summary": ("approval_queue_summary", {}),
            "approval status please": ("approval_queue_summary", {}),
            "approval report please": ("approval_queue_summary", {}),
            "approval queue status please": ("approval_queue_summary", {}),
            "show approval status": ("approval_queue_summary", {}),
            "show approval queue status": ("approval_queue_summary", {}),
            "what approval is blocking Jarvis": ("approval_queue_summary", {}),
            "what approval is blocking completion": ("approval_queue_summary", {}),
            "approval history please": ("approval_history", {}),
            "show me approval history": ("approval_history", {}),
            "latest approval history": ("approval_history", {}),
            "show latest approval history": ("approval_history", {}),
            "approval detail 1 please": ("inspect_pending_approval", {"approval_id": 1}),
            "show me approval detail 1": ("inspect_pending_approval", {"approval_id": 1}),
            "approval readiness 1 please": ("approval_readiness_packet", {"approval_id": 1}),
            "show me approval readiness 1": ("approval_readiness_packet", {"approval_id": 1}),
            "can I approve the latest approval now": ("approval_readiness_packet", {"approval_id": "latest"}),
            "can I approve this approval now": ("approval_readiness_packet", {"approval_id": "latest"}),
            "should I approve the latest approval": ("approval_readiness_packet", {"approval_id": "latest"}),
            "what should I check before approving the latest approval": ("approval_readiness_packet", {"approval_id": "latest"}),
            "what is approval 1 waiting for": ("approval_readiness_packet", {"approval_id": 1}),
            "why is approval 1 blocked": ("approval_readiness_packet", {"approval_id": 1}),
            "approval 1 next step": ("approval_readiness_packet", {"approval_id": 1}),
            "approval latest status": ("approval_readiness_packet", {"approval_id": "latest"}),
            "approval packet 1 please": ("approval_execution_packet", {"approval_id": 1}),
            "can you show me approval packet 1": ("approval_execution_packet", {"approval_id": 1}),
            "show the last look for latest approval": ("approval_execution_packet", {"approval_id": "latest"}),
            "show last look for approval 1": ("approval_execution_packet", {"approval_id": 1}),
            "what happens if I approve latest approval": ("approval_execution_packet", {"approval_id": "latest"}),
            "resume latest approval safely": ("approval_resume_packet", {"approval_id": "latest"}),
            "rerun latest approval packet": ("approval_resume_packet", {"approval_id": "latest"}),
            "approval chain proof 1 please": ("approval_chain_proof", {"approval_id": 1}),
            "show me approval chain proof 1": ("approval_chain_proof", {"approval_id": 1}),
            "prove latest approval ran": ("approval_chain_proof", {"approval_id": "latest"}),
            "approval evidence for latest": ("approval_chain_proof", {"approval_id": "latest"}),
            "next approval step": ("approval_queue_summary", {}),
            "what is the next approval command": ("approval_queue_summary", {}),
            "show me the latest approval readiness": ("approval_readiness_packet", {"approval_id": "latest"}),
            "show latest approval packet": ("approval_execution_packet", {"approval_id": "latest"}),
            "show current approval packet": ("approval_execution_packet", {"approval_id": "latest"}),
            "show newest approval chain proof": ("approval_chain_proof", {"approval_id": "latest"}),
            "show last approval detail": ("inspect_pending_approval", {"approval_id": "latest"}),
        }
        cases = [
            ("run command python3 --version", False),
            ("pending approvals", False),
            ("approval summary", False),
            ("approval detail 1", False),
            ("approval readiness 1", False),
            ("approve approval 1", False),
            ("approval packet 1", False),
            ("approval resume packet 1", False),
            ("approval chain proof 1", False),
            *[(case, False) for case in polite_readonly_approval_cases],
            ("approve approval 1", False),
            ("approve approval 1", False),
            ("run command python3 --version", False),
            ("approval packet 2", False),
            ("dismiss approval 2", False),
            ("pending approvals", False),
        ]
        for command in ["approve anything pending", "approve pending approval", "approve whatever is waiting"]:
            plan = runtime.planner.plan(command)
            if plan.actions:
                raise SystemExit(
                    f"Planner should not infer blanket approval for {command!r}: {[(action.tool_name, action.args) for action in plan.actions]}"
                )
        approved_shell_rerun_seen = False
        for case, approved in cases:
            result = handle_runtime_case(runtime, case, approved=approved)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1400])
            print()
            if case == "run command python3 --version" and "Safety receipt:" in result.response:
                metadata = result.tool_results[0].metadata
                blocked_run_id = metadata.get("logged_tool_run_id")
                approval_id = metadata.get("approval_id")
                if not isinstance(blocked_run_id, int) or not isinstance(approval_id, int):
                    raise SystemExit(f"Blocked risky run should expose run and approval ids: {metadata}")
                if metadata.get("approval_linked_tool_run") is not True:
                    raise SystemExit(f"Blocked risky run should link its audit row to approval id: {metadata}")
                blocked_run = runtime.store.get_tool_run(blocked_run_id)
                if blocked_run is None or blocked_run["approval_id"] != approval_id:
                    raise SystemExit("Blocked tool run audit row was not linked to queued approval.")
                blocked_metadata = json.loads(blocked_run["metadata"] or "{}")
                if blocked_metadata.get("risk_level") != "HIGH_RISK" or blocked_metadata.get("risk_value") != 4:
                    raise SystemExit(f"Blocked tool run audit missed risk metadata: {blocked_metadata}")
                if blocked_metadata.get("toolset") != "code":
                    raise SystemExit(f"Blocked tool run audit missed toolset metadata: {blocked_metadata}")
            if case == "pending approvals":
                pending_views += 1
                if pending_views == 1:
                    for expected in [
                        "Pending approvals:",
                        "shell/code execution",
                        "Why gated",
                        "Safer check",
                        "Decision matrix: approve only if exact and verifiable",
                        "Planned arg keys: command",
                        "Readiness: `approval readiness 1`",
                        "Last look: `approval packet 1`",
                        "Proof after approval: `approval chain proof 1`",
                        "Run if trusted: `approve approval 1`",
                        "Skip: `dismiss approval 1`",
                        "does not grant ongoing permission",
                        "explicit stop times",
                    ]:
                        if expected not in result.response:
                            raise SystemExit(f"Pending approvals response missing expected text: {expected}")
                    metadata = result.tool_results[0].metadata
                    if (
                        metadata.get("executes_tools")
                        or metadata.get("queues_approval")
                        or metadata.get("approves_request")
                        or metadata.get("dismisses_request")
                        or metadata.get("controls_computer")
                        or metadata.get("reads_private_data")
                        or metadata.get("writes_files")
                        or metadata.get("writes_notes")
                        or metadata.get("writes_memory")
                    ):
                        raise SystemExit("Pending approvals should not act or queue approvals.")
                    assert_no_future_authority(metadata, "pending approvals")
                    assert_operator_limits(metadata, "pending approvals")
                    direct_pending = runtime.registry.get("list_pending_approvals").handler({"limit": "bad"})
                    if not direct_pending.ok or direct_pending.metadata.get("limit") != 20:
                        raise SystemExit("Pending approvals did not sanitize a bad limit.")
                    assert_no_future_authority(direct_pending.metadata, "pending approvals direct")
                    bool_pending = runtime.registry.get("list_pending_approvals").handler({"limit": False})
                    if not bool_pending.ok or bool_pending.metadata.get("limit") != 20:
                        raise SystemExit("Pending approvals should treat boolean limits as malformed.")
                    assert_no_future_authority(bool_pending.metadata, "pending approvals bool limit")
                    for expected in [
                        "First safe handoff:",
                        "readiness: `approval readiness 1`",
                        "last look: `approval packet 1`",
                        "proof: `approval chain proof 1`",
                    ]:
                        if expected not in direct_pending.output:
                            raise SystemExit(f"Pending approvals missed first-handoff output: {expected}")
                    expected_chain = [
                        "approval readiness 1",
                        "approval packet 1",
                        "approve approval 1",
                        "approval chain proof 1",
                        "verification receipt <approved run id from approval chain proof 1>",
                    ]
                    if direct_pending.metadata.get("proof_chain_commands_by_approval", {}).get("1") != expected_chain:
                        raise SystemExit(f"Pending approvals missed structured proof-chain metadata: {direct_pending.metadata}")
                    for key, expected in {
                        "first_approval_id": 1,
                        "first_readiness_command": "approval readiness 1",
                        "first_last_look_command": "approval packet 1",
                        "first_proof_command": "approval chain proof 1",
                        "first_approve_command": "approve approval 1",
                        "first_dismiss_command": "dismiss approval 1",
                    }.items():
                        if direct_pending.metadata.get(key) != expected:
                            raise SystemExit(f"Pending approvals missed first-handoff metadata {key}: {direct_pending.metadata}")
                    marker = "APPROVAL_HOSTILE_ROW_SHOULD_NOT_LEAK"
                    readable_rows = runtime.store.list_pending_approvals(status="pending", limit=100)
                    original_list_pending = runtime.store.list_pending_approvals

                    def hostile_pending(status=None, limit=20):  # noqa: ANN001
                        return [HostileRow(marker), *readable_rows]

                    try:
                        runtime.store.list_pending_approvals = hostile_pending  # type: ignore[method-assign]
                        hostile_list = runtime.registry.get("list_pending_approvals").handler({"limit": 5})
                        hostile_summary = runtime.registry.get("approval_queue_summary").handler({"limit": 5})
                        hostile_review = runtime.registry.get("review_pending_approvals").handler({"limit": 5})
                        hostile_saved_review = runtime.registry.get("save_approval_review").handler({"limit": 5})
                    finally:
                        runtime.store.list_pending_approvals = original_list_pending  # type: ignore[method-assign]
                    hostile_metadata = hostile_list.metadata
                    if not hostile_list.ok:
                        raise SystemExit(f"Pending approvals should tolerate malformed approval rows: {hostile_list}")
                    if hostile_metadata.get("count") != 1 or hostile_metadata.get("readable_approval_rows") != 1:
                        raise SystemExit(f"Pending approvals should preserve readable approval count: {hostile_metadata}")
                    if hostile_metadata.get("unreadable_approval_rows") != 1:
                        raise SystemExit(f"Pending approvals missed unreadable approval counter: {hostile_metadata}")
                    for expected in [
                        "approval readiness 1",
                        "approval packet 1",
                        "approval chain proof 1",
                        "unreadable approval row(s) hidden for safety",
                    ]:
                        if expected not in hostile_list.output:
                            raise SystemExit(f"Pending approvals hostile-row output missed {expected!r}: {hostile_list.output}")
                    if marker in hostile_list.output or marker in str(hostile_metadata):
                        raise SystemExit("Pending approvals leaked raw malformed approval row text.")
                    hostile_note_text = approvals_note.read_text(encoding="utf-8")
                    if "Approval #1 - run_shell_command" not in hostile_note_text:
                        raise SystemExit("Pending approvals hostile-row note sync should preserve readable approvals.")
                    if marker in hostile_note_text:
                        raise SystemExit("Pending approvals hostile-row note sync leaked raw malformed row text.")
                    assert_no_future_authority(hostile_metadata, "pending approvals hostile rows")
                    assert_operator_limits(hostile_metadata, "pending approvals hostile rows")
                    hostile_surfaces = {
                        "approval summary hostile rows": hostile_summary,
                        "approval review hostile rows": hostile_review,
                        "save approval review hostile rows": hostile_saved_review,
                    }
                    for label, hostile_result in hostile_surfaces.items():
                        if not hostile_result.ok:
                            raise SystemExit(f"{label} should tolerate malformed approval rows: {hostile_result}")
                        surface_metadata = hostile_result.metadata
                        if surface_metadata.get("count") != 1 or surface_metadata.get("readable_approval_rows") != 1:
                            raise SystemExit(f"{label} should preserve readable approval count: {surface_metadata}")
                        if surface_metadata.get("unreadable_approval_rows") != 1:
                            raise SystemExit(f"{label} missed unreadable approval counter: {surface_metadata}")
                        for expected in [
                            "approval readiness 1",
                            "approval packet 1",
                            "unreadable approval row(s) hidden for safety",
                        ]:
                            if expected not in hostile_result.output:
                                raise SystemExit(f"{label} output missed {expected!r}: {hostile_result.output}")
                        if marker in hostile_result.output or marker in str(surface_metadata):
                            raise SystemExit(f"{label} leaked raw malformed approval row text.")
                        assert_no_future_authority(surface_metadata, label)
                        assert_operator_limits(surface_metadata, label)
                    hostile_review_note = approval_review_note.read_text(encoding="utf-8")
                    if "Approval #1: run_shell_command" not in hostile_review_note:
                        raise SystemExit("Saved approval review hostile-row note should preserve readable approvals.")
                    if marker in hostile_review_note:
                        raise SystemExit("Saved approval review hostile-row note leaked raw malformed row text.")
                    direct_review = runtime.registry.get("review_pending_approvals").handler({"limit": "9999"})
                    if not direct_review.ok or direct_review.metadata.get("limit") != 100:
                        raise SystemExit("Approval review did not clamp a large limit.")
                    bool_review = runtime.registry.get("review_pending_approvals").handler({"limit": True})
                    if not bool_review.ok or bool_review.metadata.get("limit") != 10:
                        raise SystemExit("Approval review should treat boolean limits as malformed.")
                if not approvals_note.exists():
                    raise SystemExit("Pending approvals note was not written to Obsidian.")
                note_text = approvals_note.read_text(encoding="utf-8")
                if pending_views == 1:
                    for expected in [
                        "Approval #1 - run_shell_command",
                        "Category: shell/code execution",
                        "Why gated",
                        "Safer check",
                        "Planned arg keys: command",
                        "does not grant ongoing permission",
                        "`pending approvals`",
                        "`approval readiness 1`",
                        "`approval packet 1`",
                        "`approve approval 1`",
                        "`dismiss approval 1`",
                    ]:
                        if expected not in note_text:
                            raise SystemExit(f"Pending approvals note missing expected text: {expected}")
                    if pending_views == 2 and "No pending approvals." not in note_text:
                        raise SystemExit("Pending approvals note did not clear after dismissal.")
            if case == "approval summary":
                direct_summary = runtime.registry.get("approval_queue_summary").handler({"limit": "bad"})
                if not direct_summary.ok or direct_summary.metadata.get("limit") != 5:
                    raise SystemExit("Approval summary did not sanitize a bad limit.")
                bool_summary = runtime.registry.get("approval_queue_summary").handler({"limit": False})
                if not bool_summary.ok or bool_summary.metadata.get("limit") != 5:
                    raise SystemExit("Approval summary should treat boolean limits as malformed.")
                for expected in [
                    "Approval queue summary:",
                    "pending: 1 shown",
                    "shell/code execution",
                    "approval readiness 1",
                    "approval packet 1",
                    "approval chain proof 1",
                    "Category totals:",
                    "Decision matrix: approve only if exact and verifiable",
                    "explicit stop times",
                    "summary only",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Approval summary missing expected text: {expected}")
                    metadata = result.tool_results[0].metadata
                    if metadata.get("count") != 1 or metadata.get("categories", {}).get("shell/code execution") != 1:
                        raise SystemExit("Approval summary metadata missed pending category count.")
                if metadata.get("proof_chain_commands_by_approval", {}).get("1") != [
                    "approval readiness 1",
                    "approval packet 1",
                    "approve approval 1",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                ]:
                    raise SystemExit(f"Approval summary missed structured proof-chain metadata: {metadata}")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("controls_computer")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("writes_notes")
                    or metadata.get("writes_memory")
                    ):
                        raise SystemExit("Approval summary should not act or queue approvals.")
                assert_no_future_authority(metadata, "approval summary")
                assert_operator_limits(metadata, "approval summary")
                for tool_name, raw_id in [
                    ("approval_readiness_packet", 0),
                    ("approval_execution_packet", -1),
                    ("inspect_pending_approval", 0),
                    ("approval_chain_proof", -7),
                    ("approve_pending_approval", 0),
                    ("dismiss_pending_approval", -3),
                ]:
                    invalid = runtime.registry.get(tool_name).handler({"approval_id": raw_id})
                    if invalid.ok or invalid.metadata.get("reason") != "bad_approval_id":
                        raise SystemExit(f"{tool_name} should reject non-positive approval ids before lookup: {invalid.metadata}")
                    if "positive number" not in invalid.output:
                        raise SystemExit(f"{tool_name} should explain positive approval id requirement: {invalid.output}")
                    if (
                        invalid.metadata.get("executes_tools")
                        or invalid.metadata.get("queues_approval")
                        or invalid.metadata.get("approves_request")
                        or invalid.metadata.get("dismisses_request")
                        or invalid.metadata.get("writes_files")
                        or invalid.metadata.get("writes_notes")
                    ):
                        raise SystemExit(f"{tool_name} invalid id path should remain inert: {invalid.metadata}")
                    assert_no_future_authority(invalid.metadata, f"{tool_name} invalid id")
                    assert_operator_limits(invalid.metadata, f"{tool_name} invalid id")
            if case in polite_readonly_approval_cases:
                expected_tool, expected_args = polite_readonly_approval_cases[case]
                if not result.verified:
                    raise SystemExit(f"Polite read-only approval command should run verified: {case!r}")
                if not result.tool_results or result.tool_results[0].tool_name != expected_tool:
                    raise SystemExit(f"Polite read-only approval command routed to the wrong tool: {case!r} -> {result.tool_results}")
                action = result.plan.actions[0]
                if action.tool_name != expected_tool:
                    raise SystemExit(f"Polite read-only approval plan chose the wrong tool: {case!r} -> {action}")
                for key, expected_value in expected_args.items():
                    if action.args.get(key) != expected_value:
                        raise SystemExit(f"Polite read-only approval command missed stripped args: {case!r} -> {action.args}")
                metadata = result.tool_results[0].metadata
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("controls_computer")
                    or metadata.get("writes_files")
                    or metadata.get("writes_notes")
                    or metadata.get("writes_memory")
                ):
                    raise SystemExit(f"Polite read-only approval command should stay inert: {case!r} -> {metadata}")
                assert_no_future_authority(metadata, f"polite read-only approval {case}")
                assert_operator_limits(metadata, f"polite read-only approval {case}")
            if case == "approval detail 1":
                for expected in [
                    "Approval detail #1",
                    "Original request: run command python3 --version",
                    "Why gated",
                    "Safer check",
                    "Approval decision matrix",
                    "Approve only if",
                    "Dismiss if",
                    "Blocked output",
                    "Readiness: approval readiness 1",
                    "Last look: approval packet 1",
                    "Proof after approval: approval chain proof 1",
                    "Approval proof chain:",
                    "explicit stop times",
                    "verification receipt <approved run id from approval chain proof 1>",
                    "Run if trusted: approve approval 1",
                    "Skip: dismiss approval 1",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Approval detail missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if metadata.get("next_proof_command") != "approval readiness 1":
                    raise SystemExit(f"Approval detail missed next proof command metadata: {metadata}")
                if metadata.get("proof_chain_commands") != [
                    "approval readiness 1",
                    "approval packet 1",
                    "approve approval 1",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                ]:
                    raise SystemExit(f"Approval detail missed structured proof-chain metadata: {metadata}")
                packet_before_readiness = runtime.registry.get("approval_execution_packet").handler(
                    {"approval_id": 1}
                )
                if (
                    not packet_before_readiness.ok
                    or packet_before_readiness.metadata.get("approval_readiness_rechecked") is not True
                    or packet_before_readiness.metadata.get("approval_packet_viewed") is not True
                ):
                    raise SystemExit(
                        "Approval packet should freshly recheck readiness before showing its last look: "
                        f"{packet_before_readiness}"
                    )
                if packet_before_readiness.metadata.get("next_required_command") != "approve approval 1":
                    raise SystemExit(
                        f"Approval packet recheck missed next command: {packet_before_readiness.metadata}"
                    )
                if (
                    packet_before_readiness.metadata.get("approval_readiness_required") is not True
                    or packet_before_readiness.metadata.get("approval_readiness_valid") is not True
                    or packet_before_readiness.metadata.get("approval_packet_viewed") is not True
                ):
                    raise SystemExit(
                        f"Approval packet readiness recheck missed guard metadata: {packet_before_readiness.metadata}"
                    )
                assert_no_future_authority(packet_before_readiness.metadata, "approval packet before readiness")
                assert_operator_limits(packet_before_readiness.metadata, "approval packet before readiness")
                assert_no_future_authority(metadata, "approval detail")
                assert_operator_limits(metadata, "approval detail")
            if case == "approval readiness 1":
                for expected in [
                    "Approval readiness packet #1",
                    "read-only go/no-go packet",
                    "Queue position:",
                    "pending approvals visible: 1",
                    "newest pending approval id: 1",
                    "staleness:",
                    "Stored request:",
                    "planned arg keys: command",
                    "Approval decision matrix",
                    "Approval readiness verdict:",
                    "verdict: LAST_LOOK_REQUIRED",
                    "next safe command: `approval packet 1`",
                    "next required command after approval: `approval chain proof 1`",
                    "Approval proof chain:",
                    "explicit stop times",
                    "verification receipt <approved run id from approval chain proof 1>",
                    "approve command stays disabled",
                    "does not approve, dismiss, rerun, execute tools",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Approval readiness packet missing expected text: {expected}")
                if "next proof command after approval" in result.response:
                    raise SystemExit("Approval readiness packet leaked old proof-first after-approval handoff label.")
                metadata = result.tool_results[0].metadata
                if metadata.get("verdict") != "LAST_LOOK_REQUIRED" or metadata.get("next_command") != "approval packet 1":
                    raise SystemExit(f"Approval readiness packet missed verdict/next command: {metadata}")
                if metadata.get("next_required_command") != "approval packet 1":
                    raise SystemExit(f"Approval readiness packet missed next required command metadata: {metadata}")
                if metadata.get("approval_packet_required") is not True or metadata.get("approval_packet_viewed") is not False:
                    raise SystemExit(f"Approval readiness packet missed last-look metadata: {metadata}")
                if metadata.get("next_proof_command") != "approval chain proof 1":
                    raise SystemExit(f"Approval readiness packet missed next proof command metadata: {metadata}")
                if metadata.get("proof_chain_commands") != [
                    "approval readiness 1",
                    "approval packet 1",
                    "approve approval 1",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                ]:
                    raise SystemExit(f"Approval readiness packet missed structured proof-chain metadata: {metadata}")
                if metadata.get("approval_readiness_receipt_issued") is not True:
                    raise SystemExit(f"Approval readiness packet did not issue a readiness receipt: {metadata}")
                receipt_ttl = metadata.get("approval_readiness_receipt_ttl_seconds")
                if not isinstance(receipt_ttl, int) or receipt_ttl <= 0:
                    raise SystemExit(f"Approval readiness packet missed bounded receipt TTL metadata: {metadata}")
                if metadata.get("pending_approvals") != 1 or metadata.get("newest_pending_approval_id") != 1:
                    raise SystemExit(f"Approval readiness packet missed queue metadata: {metadata}")
                if metadata.get("staleness") not in {"fresh", "review again", "stale", "unknown"}:
                    raise SystemExit(f"Approval readiness packet missed staleness label: {metadata}")
                handoff = metadata.get("approval_readiness_handoff") or {}
                if handoff.get("source") != "approval_readiness_packet":
                    raise SystemExit(f"Approval readiness packet missed structured handoff: {metadata}")
                for nested_key, flat_key in {
                    "approval_id": "approval_id",
                    "status": "status",
                    "tool_name": "tool_name",
                    "category": "category",
                    "planned_arg_keys": "planned_arg_keys",
                    "verdict": "verdict",
                    "next_command": "next_command",
                    "next_required_command": "next_required_command",
                    "next_proof_command": "next_proof_command",
                    "proof_chain_commands": "proof_chain_commands",
                    "approval_packet_required": "approval_packet_required",
                    "approval_packet_viewed": "approval_packet_viewed",
                    "approval_readiness_receipt_issued": "approval_readiness_receipt_issued",
                    "approval_readiness_receipt_ttl_seconds": "approval_readiness_receipt_ttl_seconds",
                }.items():
                    if handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"Approval readiness handoff field {nested_key} diverged from {flat_key}: {metadata}")
                queue = handoff.get("queue") or {}
                for nested_key, flat_key in {
                    "pending_approvals": "pending_approvals",
                    "newest_pending_approval_id": "newest_pending_approval_id",
                    "age_minutes": "age_minutes",
                    "staleness": "staleness",
                }.items():
                    if queue.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"Approval readiness handoff queue field {nested_key} diverged: {metadata}")
                if queue.get("pending_ids") != [1]:
                    raise SystemExit(f"Approval readiness handoff missed pending id order: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if handoff.get(key) is not True or metadata.get(f"approval_readiness_{key}") is not True:
                        raise SystemExit(f"Approval readiness handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "approves_request",
                    "dismisses_request",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "writes_notes",
                    "writes_memory",
                    "reads_private_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if handoff.get(key) is not False:
                        raise SystemExit(f"Approval readiness handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("approval_readiness_authorizes_execution") is not False
                    or metadata.get("approval_readiness_authorizes_completion_claim") is not False
                    or metadata.get("approval_readiness_approval_granted") is not False
                ):
                    raise SystemExit(f"Approval readiness flat authority fields should be false: {metadata}")
                assert_no_future_authority(metadata, "approval readiness")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("controls_computer")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("writes_notes")
                    or metadata.get("writes_memory")
                ):
                    raise SystemExit("Approval readiness packet should not act or queue approvals.")
                assert_operator_limits(metadata, "approval readiness")
                direct_readiness = runtime.registry.get("approval_readiness_packet").handler({"approval_id": "latest"})
                if not direct_readiness.ok or direct_readiness.metadata.get("approval_id") != 1:
                    raise SystemExit(f"Approval readiness direct latest lookup failed: {direct_readiness.metadata}")
                marker = "APPROVAL_READINESS_HOSTILE_QUEUE_ROW_SHOULD_NOT_LEAK"
                original_readiness_snapshot = runtime.store.approval_readiness_snapshot

                def hostile_readiness_snapshot(approval_id, limit=100):  # noqa: ANN001
                    snapshot = original_readiness_snapshot(approval_id, limit=limit)
                    return {
                        **snapshot,
                        "pending_rows": [HostileRow(marker), *(snapshot.get("pending_rows") or [])],
                    }

                try:
                    runtime.store.approval_readiness_snapshot = hostile_readiness_snapshot  # type: ignore[method-assign]
                    hostile_readiness = runtime.registry.get("approval_readiness_packet").handler({"approval_id": 1})
                finally:
                    runtime.store.approval_readiness_snapshot = original_readiness_snapshot  # type: ignore[method-assign]
                hostile_metadata = hostile_readiness.metadata
                if not hostile_readiness.ok:
                    raise SystemExit(f"Approval readiness should return a fail-closed review packet: {hostile_readiness}")
                if hostile_metadata.get("pending_approvals") != 1 or hostile_metadata.get("readable_approval_rows") != 1:
                    raise SystemExit(f"Approval readiness should preserve readable queue count: {hostile_metadata}")
                if hostile_metadata.get("unreadable_approval_rows") != 1:
                    raise SystemExit(f"Approval readiness missed unreadable queue counter: {hostile_metadata}")
                if "unreadable approval row(s) hidden for safety" not in hostile_readiness.output:
                    raise SystemExit(f"Approval readiness should report hidden malformed queue rows: {hostile_readiness.output}")
                if (
                    hostile_metadata.get("verdict") != "UNREADABLE_QUEUE_REVIEW_REQUIRED"
                    or hostile_metadata.get("next_command") != "pending approvals"
                    or hostile_metadata.get("next_required_command") != "pending approvals"
                ):
                    raise SystemExit(f"Approval readiness should fail closed on an unreadable queue: {hostile_metadata}")
                if hostile_metadata.get("approval_readiness_receipt_issued") is not False:
                    raise SystemExit(f"Unreadable queue readiness must not issue a receipt: {hostile_metadata}")
                if marker in hostile_readiness.output or marker in str(hostile_metadata):
                    raise SystemExit("Approval readiness leaked raw malformed queue row text.")
                hostile_queue = (hostile_metadata.get("approval_readiness_handoff") or {}).get("queue") or {}
                if hostile_queue.get("pending_ids") != [1] or hostile_queue.get("unreadable_approval_rows") != 1:
                    raise SystemExit(f"Approval readiness handoff should preserve readable queue ids and hidden-row count: {hostile_metadata}")
                assert_no_future_authority(hostile_metadata, "approval readiness hostile queue")
                assert_operator_limits(hostile_metadata, "approval readiness hostile queue")
                refreshed_readiness = runtime.registry.get("approval_readiness_packet").handler({"approval_id": 1})
                if (
                    not refreshed_readiness.ok
                    or refreshed_readiness.metadata.get("verdict") != "LAST_LOOK_REQUIRED"
                    or refreshed_readiness.metadata.get("approval_readiness_receipt_issued") is not True
                ):
                    raise SystemExit(f"Approval readiness did not recover after queue refresh: {refreshed_readiness}")
            if case == "approval packet 1":
                for expected in [
                    "Approval execution packet #1",
                    "What approval would do",
                    "Rerun exactly",
                    "Last-look checklist",
                    "explicit stop times",
                    "Approval decision matrix",
                    "Approve only if",
                    "Dismiss if",
                    "Approval readiness",
                    "Approval proof chain:",
                    "Prove after approval: approval chain proof 1",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Approval packet missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if metadata.get("approval_packet_viewed") is not True:
                    raise SystemExit("Approval packet did not mark approval as previewed.")
                if metadata.get("next_proof_command") != "approval chain proof 1":
                    raise SystemExit(f"Approval packet missed next proof command metadata: {metadata}")
                if metadata.get("next_required_command") != "approve approval 1":
                    raise SystemExit(f"Approval packet missed next required command metadata: {metadata}")
                if metadata.get("proof_chain_commands") != [
                    "approval readiness 1",
                    "approval packet 1",
                    "approve approval 1",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                ]:
                    raise SystemExit(f"Approval packet missed structured proof-chain metadata: {metadata}")
                handoff = metadata.get("approval_execution_handoff") or {}
                if handoff.get("source") != "approval_execution_packet":
                    raise SystemExit(f"Approval packet missed structured handoff: {metadata}")
                for nested_key, flat_key in {
                    "approval_id": "approval_id",
                    "status": "status",
                    "tool_name": "tool_name",
                    "planned_args": "planned_args",
                    "planned_arg_keys": "planned_arg_keys",
                    "approval_packet_viewed": "approval_packet_viewed",
                    "proof_chain_commands": "proof_chain_commands",
                    "next_required_command": "next_required_command",
                    "next_proof_command": "next_proof_command",
                }.items():
                    if handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"Approval packet handoff field {nested_key} diverged from {flat_key}: {metadata}")
                if handoff.get("category") != "shell/code execution" or handoff.get("decision_commands", {}).get("approve") != "approve approval 1":
                    raise SystemExit(f"Approval packet handoff missed category or decision commands: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if handoff.get(key) is not True or metadata.get(f"approval_execution_{key}") is not True:
                        raise SystemExit(f"Approval packet handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "approves_request",
                    "dismisses_request",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "writes_notes",
                    "writes_memory",
                    "reads_private_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if handoff.get(key) is not False:
                        raise SystemExit(f"Approval packet handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("approval_execution_authorizes_execution") is not False
                    or metadata.get("approval_execution_authorizes_completion_claim") is not False
                    or metadata.get("approval_execution_approval_granted") is not False
                ):
                    raise SystemExit(f"Approval packet flat authority fields should be false: {metadata}")
                assert_no_future_authority(metadata, "approval packet")
                assert_operator_limits(metadata, "approval packet")
            if case == "approval resume packet 1":
                for expected in [
                    "Approval resume packet #1",
                    "one specific queued request",
                    "Resume target:",
                    "approval id: 1",
                    "tool: run_shell_command",
                    "planned arg keys: command",
                    "approval packet viewed: yes",
                    "READY_FOR_ONE_SHOT_APPROVAL_RERUN",
                    "next safe command: `approve approval 1`",
                    "one specific stored request, one approved rerun, no ongoing permission",
                    "Approval proof chain:",
                    "explicit stop times",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Approval resume packet missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if metadata.get("verdict") != "READY_FOR_ONE_SHOT_APPROVAL_RERUN" or metadata.get("next_command") != "approve approval 1":
                    raise SystemExit(f"Approval resume packet missed ready verdict metadata: {metadata}")
                if metadata.get("next_required_command") != "approve approval 1":
                    raise SystemExit(f"Approval resume packet missed next required command metadata: {metadata}")
                if metadata.get("approval_packet_viewed") is not True or metadata.get("approval_packet_required") is not True:
                    raise SystemExit(f"Approval resume packet missed last-look metadata: {metadata}")
                contract = metadata.get("approval_resume_contract") or {}
                for key, expected in {
                    "source": "approval_resume_contract",
                    "approval_id": 1,
                    "status": "pending",
                    "tool_name": "run_shell_command",
                    "exact_rerun_tool_name": "run_shell_command",
                    "exact_rerun_arg_keys": ["command"],
                    "approve_command": "approve approval 1",
                    "last_look_command": "approval packet 1",
                    "proof_command": "approval chain proof 1",
                    "specific_request_only": True,
                    "one_shot": True,
                    "approval_packet_viewed": True,
                    "next_proof_command": "approval chain proof 1",
                }.items():
                    if contract.get(key) != expected:
                        raise SystemExit(f"Approval resume contract missed {key}: {metadata}")
                if contract.get("exact_rerun_args_display") != {"command": "python3 --version"}:
                    raise SystemExit(f"Approval resume contract missed exact args display: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if contract.get(key) is not True or metadata.get(f"approval_resume_{key}") is not True:
                        raise SystemExit(f"Approval resume packet missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "approves_request",
                    "dismisses_request",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "writes_notes",
                    "writes_memory",
                    "reads_private_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if contract.get(key) is not False:
                        raise SystemExit(f"Approval resume contract should keep {key}=False: {metadata}")
                assert_no_future_authority(metadata, "approval resume packet")
                assert_operator_limits(metadata, "approval resume packet")
                marker = "APPROVAL_RESUME_HOSTILE_QUEUE_ROW_SHOULD_NOT_LEAK"
                readable_rows = runtime.store.list_pending_approvals(status="pending", limit=100)
                original_list_pending = runtime.store.list_pending_approvals

                def hostile_pending(status=None, limit=100):  # noqa: ANN001
                    return [HostileRow(marker), *readable_rows]

                try:
                    runtime.store.list_pending_approvals = hostile_pending  # type: ignore[method-assign]
                    hostile_resume = runtime.registry.get("approval_resume_packet").handler({"approval_id": 1})
                finally:
                    runtime.store.list_pending_approvals = original_list_pending  # type: ignore[method-assign]
                hostile_metadata = hostile_resume.metadata
                if not hostile_resume.ok:
                    raise SystemExit(f"Approval resume should tolerate malformed queue rows: {hostile_resume}")
                if hostile_metadata.get("pending_approvals") != 1 or hostile_metadata.get("readable_approval_rows") != 1:
                    raise SystemExit(f"Approval resume should preserve readable queue count: {hostile_metadata}")
                if hostile_metadata.get("unreadable_approval_rows") != 1:
                    raise SystemExit(f"Approval resume missed unreadable queue counter: {hostile_metadata}")
                if "unreadable approval row(s) hidden for safety" not in hostile_resume.output:
                    raise SystemExit(f"Approval resume should report hidden malformed queue rows: {hostile_resume.output}")
                if hostile_metadata.get("next_command") != "approve approval 1" or "approve approval 1" not in hostile_resume.output:
                    raise SystemExit(f"Approval resume should preserve ready next command behind malformed queue rows: {hostile_metadata}")
                if marker in hostile_resume.output or marker in str(hostile_metadata):
                    raise SystemExit("Approval resume leaked raw malformed queue row text.")
                assert_no_future_authority(hostile_metadata, "approval resume hostile queue")
                assert_operator_limits(hostile_metadata, "approval resume hostile queue")
            if case == "approval chain proof 1":
                for expected in [
                    "Approval chain proof #1",
                    "read-only audit packet",
                    "status: pending",
                    "Linked approved reruns:",
                    "none linked",
                    "Verdict: APPROVAL_PENDING_LAST_LOOK_REQUIRED",
                    "Valid execution proof: no",
                    "explicit stop times",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Pending approval chain proof missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if metadata.get("verdict") != "APPROVAL_PENDING_LAST_LOOK_REQUIRED" or metadata.get("valid_execution_proof") is not False:
                    raise SystemExit(f"Pending approval chain proof missed metadata: {metadata}")
                if metadata.get("next_proof_command") != "verification receipt <approved run id from approval chain proof 1>":
                    raise SystemExit(f"Pending approval chain proof missed next proof command: {metadata}")
                if metadata.get("next_required_command") != "verification receipt <approved run id from approval chain proof 1>":
                    raise SystemExit(f"Pending approval chain proof missed next required command: {metadata}")
                if metadata.get("proof_chain_commands") != [
                    "approval readiness 1",
                    "approval packet 1",
                    "approve approval 1",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                ]:
                    raise SystemExit(f"Pending approval chain proof missed proof-chain metadata: {metadata}")
                handoff = metadata.get("approval_chain_proof_handoff") or {}
                if handoff.get("source") != "approval_chain_proof":
                    raise SystemExit(f"Approval chain proof missed structured handoff: {metadata}")
                for nested_key, flat_key in {
                    "approval_id": "approval_id",
                    "status": "status",
                    "tool_name": "tool_name",
                    "category": "category",
                    "planned_arg_keys": "planned_arg_keys",
                    "linked_runs": "linked_runs",
                    "approved_run_ids": "approved_run_ids",
                    "approved_success_count": "approved_success_count",
                    "verdict": "verdict",
                    "valid_execution_proof": "valid_execution_proof",
                    "approval_packet_required": "approval_packet_required",
                    "approval_packet_viewed": "approval_packet_viewed",
                    "proof_chain_commands": "proof_chain_commands",
                    "next_required_command": "next_required_command",
                    "next_proof_command": "next_proof_command",
                    "verification_command": "verification_command",
                }.items():
                    if handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"Approval chain proof handoff field {nested_key} diverged from {flat_key}: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if handoff.get(key) is not True or metadata.get(f"approval_chain_proof_{key}") is not True:
                        raise SystemExit(f"Approval chain proof handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "approves_request",
                    "dismisses_request",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "writes_notes",
                    "writes_memory",
                    "reads_private_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if handoff.get(key) is not False:
                        raise SystemExit(f"Approval chain proof handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("approval_chain_proof_authorizes_execution") is not False
                    or metadata.get("approval_chain_proof_authorizes_completion_claim") is not False
                    or metadata.get("approval_chain_proof_approval_granted") is not False
                ):
                    raise SystemExit(f"Approval chain proof flat authority fields should be false: {metadata}")
                assert_no_future_authority(metadata, "approval chain proof pending")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("controls_computer")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("writes_notes")
                    or metadata.get("writes_memory")
                ):
                    raise SystemExit("Approval chain proof should not act or queue approvals.")
                assert_operator_limits(metadata, "approval chain proof pending")
            if case == "approve approval 1" and "needs a last-look packet" in result.response:
                metadata = result.tool_results[0].metadata
                if metadata.get("approval_packet_required") is not True:
                    raise SystemExit("Approval packet guard did not report approval_packet_required.")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("writes_files")
                    or metadata.get("writes_notes")
                ):
                    raise SystemExit("Approval packet guard should not execute, approve, dismiss, or queue.")
                assert_no_future_authority(metadata, "approval packet guard")
                assert_operator_limits(metadata, "approval packet guard")
                for guard in ["approve approval 1 please", "dismiss approval 1 please"]:
                    guard_plan = runtime.planner.plan(guard)
                    if guard_plan.actions:
                        details = [(action.tool_name, action.args) for action in guard_plan.actions]
                        raise SystemExit(f"Polite retry should not normalize approval decisions into action: {guard!r} -> {details}")
            if case == "approve approval 1" and "Approved run result" in result.response:
                approved_shell_rerun_seen = True
                if "Approved run result" not in result.response or "Python" not in result.response:
                    raise SystemExit("Approved approval did not rerun the planned shell command.")
                metadata = result.tool_results[0].metadata
                if metadata.get("approval_packet_required") is not True or metadata.get("approval_packet_viewed") is not True:
                    raise SystemExit("Approved approval did not require a viewed approval packet.")
                assert_no_future_authority(metadata, "approved approval")
                if metadata.get("approves_request") is not True or metadata.get("writes_files") is not True:
                    raise SystemExit("Approved approval metadata missed approval/write markers.")
                if metadata.get("writes_notes") is not True or metadata.get("queues_approval"):
                    raise SystemExit("Approved approval metadata missed note write/safe queue markers.")
                if metadata.get("exact_rerun_tool_name") != "run_shell_command":
                    raise SystemExit(f"Approved approval should rerun the stored tool directly: {metadata}")
                if metadata.get("exact_rerun_args") != {"command": "python3 --version"}:
                    raise SystemExit(f"Approved approval should preserve exact stored args: {metadata}")
                approved_contract = metadata.get("approval_resume_contract") or {}
                for key, expected in {
                    "source": "approve_pending_approval",
                    "approval_id": 1,
                    "exact_rerun_tool_name": "run_shell_command",
                    "exact_rerun_arg_keys": ["command"],
                    "specific_request_only": True,
                    "one_shot": True,
                    "approval_packet_viewed": True,
                    "approve_command": "approve approval 1",
                    "proof_command": "approval chain proof 1",
                }.items():
                    if approved_contract.get(key) != expected:
                        raise SystemExit(f"Approved approval missed resume contract field {key}: {metadata}")
                if metadata.get("specific_request_only") is not True or metadata.get("one_shot_approval_rerun") is not True:
                    raise SystemExit(f"Approved approval missed one-shot specific-request markers: {metadata}")
                assert_operator_limits(metadata, "approved approval")
                runtime_trace = result.metadata.get("runtime_trace", {})
                trace_results = runtime_trace.get("tool_results") or []
                if not trace_results or not isinstance(trace_results[0].get("approved_rerun_result"), dict):
                    raise SystemExit(f"Approval runtime trace should include nested approved rerun evidence: {runtime_trace}")
                rerun_trace = trace_results[0]["approved_rerun_result"]
                if rerun_trace.get("tool_name") != "run_shell_command" or rerun_trace.get("approval_rerun_exact") is not True:
                    raise SystemExit(f"Nested approved rerun trace missed exact tool evidence: {rerun_trace}")
                if rerun_trace.get("approved_approval_id") != 1 or rerun_trace.get("rerun_arg_keys") != ["command"]:
                    raise SystemExit(f"Nested approved rerun trace missed approval id or arg keys: {rerun_trace}")
                if runtime_trace.get("approved_reruns") != 1 or runtime_trace.get("approved_rerun_run_ids") != [rerun_trace.get("logged_tool_run_id")]:
                    raise SystemExit(f"Approval runtime trace missed top-level approved rerun ids: {runtime_trace}")
                if runtime_trace.get("approved_rerun_approval_ids") != [1]:
                    raise SystemExit(f"Approval runtime trace missed top-level approved approval id: {runtime_trace}")
                execution_claim = runtime.store.get_approval_execution_claim(1)
                if execution_claim is None or execution_claim["outcome"] != "succeeded" or not execution_claim["completed_at"]:
                    raise SystemExit(
                        f"Approved approval missed finalized one-shot execution claim: "
                        f"{dict(execution_claim) if execution_claim else None}"
                    )
                if "claim_token" in json.dumps(result.metadata, sort_keys=True):
                    raise SystemExit(f"Approval runtime metadata exposed the private execution claim token: {result.metadata}")
                trace_receipt = runtime.handle("runtime trace receipt")
                print("[ok] runtime trace receipt after approved rerun")
                print(trace_receipt.response[:1800])
                print()
                trace_message_id = trace_receipt.tool_results[0].metadata.get("message_id")
                if not isinstance(trace_message_id, int):
                    raise SystemExit(f"Runtime trace receipt should expose the traced message id: {trace_receipt.tool_results[0].metadata}")
                for expected in [
                    "approved rerun: run_shell_command ok, approval #1",
                    "approved rerun exact: yes",
                    "approved rerun arg keys: command",
                    "approved rerun output: Python",
                    "approved rerun execution started: yes",
                    "approved rerun output capture: bounded streaming",
                    "cap 24000 bytes/stream",
                    "observed stdout ",
                    "direct process reaped: yes",
                    "approved rerun: yes",
                    "approved rerun approval ids: 1",
                ]:
                    if expected not in trace_receipt.response:
                        raise SystemExit(f"Runtime trace receipt missed approved rerun evidence: {expected}")
                capture_lines = [
                    line
                    for line in trace_receipt.response.splitlines()
                    if "approved rerun output capture:" in line
                ]
                if len(capture_lines) != 1 or "unknown" in capture_lines[0]:
                    raise SystemExit(f"Runtime trace receipt hid numeric shell capture evidence: {trace_receipt.response}")
                trace_metadata = trace_receipt.tool_results[0].metadata
                if trace_metadata.get("approved_reruns") != 1 or trace_metadata.get("approved_rerun_approval_ids") != [1]:
                    raise SystemExit(f"Runtime trace receipt metadata missed approved rerun summary: {trace_metadata}")
                history = runtime.handle("approval history")
                print("[ok] approval history after approved rerun")
                print(history.response[:1800])
                print()
                for expected in [
                    "Approval history",
                    "Approval #1 [approved]",
                    "Approved rerun audit",
                    "Tool run #",
                    "run_shell_command [ok]",
                ]:
                    if expected not in history.response:
                        raise SystemExit(f"Approval history missed approved rerun audit text: {expected}")
                metadata = history.tool_results[0].metadata
                if metadata.get("approved_tool_runs", 0) < 1:
                    raise SystemExit("Approval history metadata missed approved tool run count.")
                assert_operator_limits(metadata, "approval history after approved rerun")
                proof = runtime.handle("approval chain proof 1")
                print("[ok] approval chain proof after approved rerun")
                print(proof.response[:1800])
                print()
                for expected in [
                    "Approval chain proof #1",
                    "status: approved",
                    "Linked approved reruns:",
                    "Tool run #",
                    "Verdict: APPROVAL_CHAIN_PROVEN",
                    "Valid execution proof: yes",
                    "verification receipt",
                    "explicit stop times",
                ]:
                    if expected not in proof.response:
                        raise SystemExit(f"Approved approval chain proof missed expected text: {expected}")
                metadata = proof.tool_results[0].metadata
                if metadata.get("verdict") != "APPROVAL_CHAIN_PROVEN" or metadata.get("valid_execution_proof") is not True or metadata.get("successful_runs", 0) < 1:
                    raise SystemExit(f"Approved approval chain proof missed metadata: {metadata}")
                approved_run_ids = [int(row["id"]) for row in runtime.store.approved_tool_runs_for_approvals([1], limit=10) if row["ok"]]
                if not approved_run_ids:
                    raise SystemExit("Approval chain proof did not leave an approved successful run id.")
                if metadata.get("approved_run_ids") != approved_run_ids or metadata.get("approved_success_count", 0) < 1:
                    raise SystemExit(f"Approved approval chain proof missed approved run id metadata: {metadata}")
                if metadata.get("next_proof_command") != f"verification receipt {approved_run_ids[0]}":
                    raise SystemExit(f"Approved approval chain proof missed verification command metadata: {metadata}")
                handoff = metadata.get("approval_chain_proof_handoff") or {}
                if handoff.get("source") != "approval_chain_proof":
                    raise SystemExit(f"Approved approval chain proof missed structured handoff: {metadata}")
                for nested_key, flat_key in {
                    "approval_id": "approval_id",
                    "status": "status",
                    "tool_name": "tool_name",
                    "category": "category",
                    "planned_arg_keys": "planned_arg_keys",
                    "linked_runs": "linked_runs",
                    "approved_run_ids": "approved_run_ids",
                    "approved_success_count": "approved_success_count",
                    "verdict": "verdict",
                    "valid_execution_proof": "valid_execution_proof",
                    "approval_packet_required": "approval_packet_required",
                    "approval_packet_viewed": "approval_packet_viewed",
                    "proof_chain_commands": "proof_chain_commands",
                    "next_proof_command": "next_proof_command",
                    "verification_command": "verification_command",
                }.items():
                    if handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"Approved approval chain proof handoff field {nested_key} diverged from {flat_key}: {metadata}")
                if handoff.get("approved_reruns", [{}])[0].get("tool_name") != "run_shell_command":
                    raise SystemExit(f"Approved approval chain proof handoff missed approved rerun summary: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if handoff.get(key) is not True or metadata.get(f"approval_chain_proof_{key}") is not True:
                        raise SystemExit(f"Approved approval chain proof handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "approves_request",
                    "dismisses_request",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "writes_notes",
                    "writes_memory",
                    "reads_private_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if handoff.get(key) is not False:
                        raise SystemExit(f"Approved approval chain proof handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("approval_chain_proof_authorizes_execution") is not False
                    or metadata.get("approval_chain_proof_authorizes_completion_claim") is not False
                    or metadata.get("approval_chain_proof_approval_granted") is not False
                ):
                    raise SystemExit(f"Approved approval chain proof flat authority fields should be false: {metadata}")
                assert_no_future_authority(metadata, "approval chain proof approved")
                approved_runs = runtime.store.approved_tool_runs_for_approvals([1], limit=10)
                latest_approved_run = next((row for row in approved_runs if int(row["id"]) == approved_run_ids[0]), None)
                if latest_approved_run is None or latest_approved_run["tool_name"] != "run_shell_command":
                    raise SystemExit("Approved rerun should be logged against the stored tool name.")
                for key in [
                    "executes_tools",
                    "queues_approval",
                    "approves_request",
                    "dismisses_request",
                    "controls_computer",
                    "reads_private_data",
                    "writes_files",
                    "writes_notes",
                    "writes_memory",
                ]:
                    if metadata.get(key):
                        raise SystemExit(f"Approved approval chain proof should report {key}=False.")
                assert_operator_limits(metadata, "approval chain proof approved")
                bundle_without_receipt = runtime.handle("execution proof bundle: run command python3 --version; verification Python version shown; tests smoke_test_approvals passed; evidence recent approved run ok; recovery stop on mismatch; approval approval 1")
                print("[ok] execution proof bundle rejects approval without receipt id")
                print(bundle_without_receipt.response[:1800])
                print()
                if not bundle_without_receipt.verified:
                    raise SystemExit("Execution proof bundle without receipt id should run read-only.")
                for expected in [
                    "approval chain status: approved linked rerun proven",
                    "supplied verification/tool-run ids: none",
                    "risky proof needs a concrete verification/runtime-trace receipt id",
                ]:
                    if expected not in bundle_without_receipt.response:
                        raise SystemExit(f"Execution proof bundle should reject non-receipt verification text: {expected}")
                no_receipt_metadata = bundle_without_receipt.tool_results[0].metadata
                if no_receipt_metadata.get("approval_chain_valid") is not True or no_receipt_metadata.get("has_verified_approval_run") is not False:
                    raise SystemExit(f"Execution proof bundle should validate approval but reject missing receipt link: {no_receipt_metadata}")
                runtime_trace_bundle = runtime.handle(f"execution proof bundle: run command python3 --version; verification runtime trace receipt {trace_message_id} shows approved exact rerun; tests smoke_test_approvals passed; evidence approved rerun visible in trace; recovery stop on mismatch; approval approval 1")
                print("[ok] execution proof bundle with runtime trace approval proof")
                print(runtime_trace_bundle.response[:1800])
                print()
                if not runtime_trace_bundle.verified or runtime_trace_bundle.tool_results[0].tool_name != "execution_proof_bundle":
                    raise SystemExit("Execution proof bundle with runtime trace receipt did not route correctly.")
                for expected in [
                    "approval chain status: approved linked rerun proven",
                    f"supplied runtime-trace message ids: {trace_message_id}",
                    "supplied verification/tool-run ids: none",
                    f"verified approved runs: {approved_run_ids[0]}",
                ]:
                    if expected not in runtime_trace_bundle.response:
                        raise SystemExit(f"Execution proof bundle missed runtime trace proof text: {expected}")
                runtime_trace_metadata = runtime_trace_bundle.tool_results[0].metadata
                if runtime_trace_metadata.get("has_verified_approval_run") is not True:
                    raise SystemExit(f"Runtime trace proof should verify the approved rerun: {runtime_trace_metadata}")
                if runtime_trace_metadata.get("runtime_trace_verified_approval_run_ids") != [approved_run_ids[0]]:
                    raise SystemExit(f"Runtime trace proof should recover the linked approved run id: {runtime_trace_metadata}")
                if runtime_trace_metadata.get("has_missing_supplied_runtime_trace_messages") is not False:
                    raise SystemExit(f"Runtime trace proof should not flag the supplied trace as missing: {runtime_trace_metadata}")
                mixed_verification_bundle = runtime.handle("execution proof bundle: run command python3 --version; verification verification receipt pending; approval packet 1 reviewed; tests smoke_test_approvals passed; evidence approved rerun visible in approval history; recovery stop on mismatch; approval approval 1")
                print("[ok] execution proof bundle ignores unrelated approval ids in verification text")
                print(mixed_verification_bundle.response[:1800])
                print()
                if not mixed_verification_bundle.verified:
                    raise SystemExit("Execution proof bundle with mixed verification text should run read-only.")
                mixed_verification_metadata = mixed_verification_bundle.tool_results[0].metadata
                if mixed_verification_metadata.get("supplied_verification_run_ids") != []:
                    raise SystemExit(f"Verification parser should not treat approval ids as tool-run receipts: {mixed_verification_metadata}")
                if mixed_verification_metadata.get("has_verified_approval_run") is not False:
                    raise SystemExit(f"Mixed verification text must not verify the approved rerun: {mixed_verification_metadata}")
                if "risky proof needs a concrete verification/runtime-trace receipt id" not in mixed_verification_bundle.response:
                    raise SystemExit("Mixed verification text should still ask for a concrete receipt id.")
                bundle = runtime.handle(f"execution proof bundle: run command python3 --version; verification verification receipt {approved_run_ids[0]} shows Python version; tests smoke_test_approvals passed; evidence recent approved run ok; recovery stop on mismatch; approval approval 1")
                print("[ok] execution proof bundle with approval chain")
                print(bundle.response[:1800])
                print()
                if not bundle.verified or bundle.tool_results[0].tool_name != "execution_proof_bundle":
                    raise SystemExit("Execution proof bundle with approval chain did not route correctly.")
                for expected in [
                    "Jarvis execution proof bundle",
                    "approval chain status: approved linked rerun proven",
                    "supplied approval receipt: approval 1",
                    f"supplied verification/tool-run ids: {approved_run_ids[0]}",
                    f"verified approved runs: {approved_run_ids[0]}",
                    "Supplied proof receipts",
                ]:
                    if expected not in bundle.response:
                        raise SystemExit(f"Execution proof bundle missed approval chain text: {expected}")
                metadata = bundle.tool_results[0].metadata
                if metadata.get("supplied_approval_id") != 1 or metadata.get("approval_chain_valid") is not True or not metadata.get("approval_linked_run_ids"):
                    raise SystemExit(f"Execution proof bundle did not validate supplied approval id: {metadata}")
                if metadata.get("has_verified_approval_run") is not True or metadata.get("verified_approval_run_ids") != [approved_run_ids[0]]:
                    raise SystemExit(f"Execution proof bundle did not require the approved-run verification receipt: {metadata}")

        if not approved_shell_rerun_seen:
            raise SystemExit("Natural shell approval flow never executed its exact one-shot rerun assertions.")

        bad_packet = runtime.registry.get("approval_execution_packet").handler({"approval_id": "bad"})
        if bad_packet.ok or bad_packet.metadata.get("reason") != "bad_approval_id":
            raise SystemExit("Approval packet bad id missed safe refusal metadata.")
        if bad_packet.metadata.get("raw_approval_id") != "bad":
            raise SystemExit(f"Approval packet bad id missed bounded raw id metadata: {bad_packet.metadata}")
        assert_no_future_authority(bad_packet.metadata, "approval packet bad id")

        long_bad_readiness = runtime.registry.get("approval_readiness_packet").handler({"approval_id": "a" * 200})
        if long_bad_readiness.ok or long_bad_readiness.metadata.get("raw_approval_id") != ("a" * 77 + "..."):
            raise SystemExit(f"Approval readiness bad id did not bound raw id metadata: {long_bad_readiness.metadata}")
        assert_no_future_authority(long_bad_readiness.metadata, "approval readiness long bad id")
        path_bad_readiness = runtime.registry.get("approval_readiness_packet").handler({"approval_id": "/\x55sers/example/private/approval-id"})
        if path_bad_readiness.ok or path_bad_readiness.metadata.get("raw_approval_id") != "<local-path>":
            raise SystemExit(f"Approval readiness should redact path-shaped bad ids: {path_bad_readiness.metadata}")
        assert_no_future_authority(path_bad_readiness.metadata, "approval readiness path bad id")
        temp_path_bad_readiness = runtime.registry.get("approval_readiness_packet").handler({"approval_id": "/var/folders/zc/jarvis/approval-id"})
        if temp_path_bad_readiness.ok or temp_path_bad_readiness.metadata.get("raw_approval_id") != "<local-path>":
            raise SystemExit(f"Approval readiness should redact temp-root bad ids: {temp_path_bad_readiness.metadata}")

        for tool_name in [
            "dismiss_pending_approval",
            "inspect_pending_approval",
            "approval_chain_proof",
            "approve_pending_approval",
        ]:
            bad_id = runtime.registry.get(tool_name).handler({"approval_id": "bad"})
            if bad_id.ok or bad_id.metadata.get("reason") != "bad_approval_id":
                raise SystemExit(f"{tool_name} bad id missed safe refusal metadata.")
            if bad_id.metadata.get("raw_approval_id") != "bad":
                raise SystemExit(f"{tool_name} bad id missed bounded raw id metadata: {bad_id.metadata}")
            assert_no_future_authority(bad_id.metadata, f"{tool_name} bad id")
            path_bad_id = runtime.registry.get(tool_name).handler({"approval_id": "/private/tmp/jarvis-approval-id"})
            if path_bad_id.ok or path_bad_id.metadata.get("reason") != "bad_approval_id":
                raise SystemExit(f"{tool_name} path bad id missed safe refusal metadata.")
            if path_bad_id.metadata.get("raw_approval_id") != "<local-path>":
                raise SystemExit(f"{tool_name} should redact path-shaped bad ids: {path_bad_id.metadata}")
            assert_no_future_authority(path_bad_id.metadata, f"{tool_name} path bad id")
            temp_path_bad_id = runtime.registry.get(tool_name).handler({"approval_id": "/tmp/jarvis-approval-id"})
            if temp_path_bad_id.ok or temp_path_bad_id.metadata.get("reason") != "bad_approval_id":
                raise SystemExit(f"{tool_name} temp-root bad id missed safe refusal metadata.")
            if temp_path_bad_id.metadata.get("raw_approval_id") != "<local-path>":
                raise SystemExit(f"{tool_name} should redact temp-root bad ids: {temp_path_bad_id.metadata}")

        missing_detail = runtime.registry.get("inspect_pending_approval").handler({"approval_id": 999})
        missing_cases = [
            (
                missing_detail,
                "approval detail missing",
                ["pending approvals", "approval detail 999"],
            ),
            (
                runtime.registry.get("approve_pending_approval").handler({"approval_id": 999}),
                "approval approve missing",
                ["pending approvals", "approval detail 999", "approve approval 999"],
            ),
            (
                runtime.registry.get("dismiss_pending_approval").handler({"approval_id": 999}),
                "approval dismiss missing",
                ["pending approvals", "approval detail 999", "dismiss approval 999"],
            ),
        ]
        for missing_result, label, expected_commands in missing_cases:
            assert_approval_failure_recovery(
                missing_result,
                label=label,
                approval_id=999,
                expected_commands=expected_commands,
                reason="not_found",
            )

        action_tamper_id = runtime.store.add_pending_approval(
            "approval action digest tamper smoke",
            "run command printf reviewed-action",
            "run_shell_command",
            "approval required",
            planned_args={"command": "printf reviewed-action"},
        )
        action_tamper_readiness = runtime.registry.get("approval_readiness_packet").handler(
            {"approval_id": action_tamper_id}
        )
        action_tamper_packet = runtime.registry.get("approval_execution_packet").handler(
            {"approval_id": action_tamper_id}
        )
        if (
            not action_tamper_readiness.ok
            or action_tamper_readiness.metadata.get("approval_readiness_receipt_issued") is not True
            or not action_tamper_packet.ok
            or action_tamper_packet.metadata.get("approval_packet_viewed") is not True
        ):
            raise SystemExit(
                "Approval action-digest tamper fixture did not complete exact review: "
                f"{action_tamper_readiness} {action_tamper_packet}"
            )
        action_tamper_before = runtime.store.approval_readiness_snapshot(action_tamper_id)
        action_tamper_target = action_tamper_before.get("target")
        if action_tamper_target is None:
            raise SystemExit("Approval action-digest tamper fixture disappeared before mutation")
        action_tamper_revision = int(action_tamper_before["queue_revision"])
        action_tamper_updated_at = str(action_tamper_target["updated_at"])
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE pending_approvals SET planned_args = ? WHERE id = ?",
                (
                    json.dumps({"command": "printf unreviewed-action"}, sort_keys=True),
                    action_tamper_id,
                ),
            )
        action_tamper_after = runtime.store.approval_readiness_snapshot(action_tamper_id)
        action_tamper_after_target = action_tamper_after.get("target")
        if (
            action_tamper_after_target is None
            or int(action_tamper_after["queue_revision"]) != action_tamper_revision
            or str(action_tamper_after_target["updated_at"]) != action_tamper_updated_at
        ):
            raise SystemExit(
                "Approval action-digest adversarial fixture changed timestamp or queue revision"
            )
        action_tamper_approve = runtime.registry.get("approve_pending_approval").handler(
            {"approval_id": action_tamper_id}
        )
        action_tamper_row = runtime.store.get_approval(action_tamper_id)
        if (
            action_tamper_approve.ok
            or action_tamper_approve.metadata.get("reason") != "readiness_changed"
            or action_tamper_approve.metadata.get("approval_readiness_valid") is not False
            or action_tamper_approve.metadata.get("approval_packet_viewed") is not False
            or action_tamper_row is None
            or str(action_tamper_row["status"]).lower() != "pending"
            or runtime.store.get_approval_execution_claim(action_tamper_id) is not None
        ):
            raise SystemExit(
                "Unchanged-timestamp approval argument tampering reached approval or execution: "
                f"{action_tamper_approve}"
            )
        assert_no_future_authority(
            action_tamper_approve.metadata,
            "approval action-digest tamper",
        )
        if not runtime.store.set_pending_approval_status(action_tamper_id, "dismissed"):
            raise SystemExit("Approval action-digest tamper fixture could not be dismissed")

        update_failure_id = runtime.store.add_pending_approval(
            "approval update failure smoke",
            "run command printf approval-update-smoke",
            "run_shell_command",
            "approval required",
            planned_args={"command": "printf approval-update-smoke"},
        )
        update_failure_packet = runtime.registry.get("approval_execution_packet").handler(
            {"approval_id": update_failure_id}
        )
        if (
            not update_failure_packet.ok
            or update_failure_packet.metadata.get("approval_readiness_rechecked") is not True
            or update_failure_packet.metadata.get("approval_packet_viewed") is not True
        ):
            raise SystemExit(
                "Approval update-failure fixture should freshly recheck readiness before its last-look packet: "
                f"{update_failure_packet}"
            )
        update_failure_readiness = runtime.registry.get("approval_readiness_packet").handler(
            {"approval_id": update_failure_id}
        )
        if not update_failure_readiness.ok or update_failure_readiness.metadata.get("approval_readiness_receipt_issued") is not True:
            raise SystemExit(f"Approval update-failure fixture readiness failed: {update_failure_readiness}")
        update_failure_packet = runtime.registry.get("approval_execution_packet").handler(
            {"approval_id": update_failure_id}
        )
        if not update_failure_packet.ok:
            raise SystemExit(f"Approval update-failure fixture packet failed after readiness: {update_failure_packet.output}")

        store_type = type(runtime.store)
        original_approve_if_ready = store_type.approve_pending_approval_if_ready

        def false_approve_if_ready(
            _store,
            _approval_id,
            *,
            expected_queue_revision,
            expected_updated_at,
            expected_action_digest,
        ):  # noqa: ARG001
            return False

        store_type.approve_pending_approval_if_ready = false_approve_if_ready
        try:
            changed_approve = runtime.registry.get("approve_pending_approval").handler(
                {"approval_id": update_failure_id}
            )
        finally:
            store_type.approve_pending_approval_if_ready = original_approve_if_ready
        if changed_approve.ok or changed_approve.metadata.get("reason") != "readiness_changed":
            raise SystemExit(f"Approval readiness race should fail closed: {changed_approve}")
        if (
            changed_approve.metadata.get("next_command") != f"approval readiness {update_failure_id}"
            or changed_approve.metadata.get("approval_readiness_valid") is not False
            or changed_approve.metadata.get("approval_packet_viewed") is not False
        ):
            raise SystemExit(f"Approval readiness race missed fresh-review metadata: {changed_approve.metadata}")
        assert_no_future_authority(changed_approve.metadata, "approval readiness race")
        assert_operator_limits(changed_approve.metadata, "approval readiness race")

        retry_readiness = runtime.registry.get("approval_readiness_packet").handler(
            {"approval_id": update_failure_id}
        )
        retry_packet = runtime.registry.get("approval_execution_packet").handler(
            {"approval_id": update_failure_id}
        )
        if not retry_readiness.ok or not retry_packet.ok:
            raise SystemExit(f"Approval failure retry fixture could not refresh review: {retry_readiness} {retry_packet}")

        def raising_approve_if_ready(
            _store,
            _approval_id,
            *,
            expected_queue_revision,
            expected_updated_at,
            expected_action_digest,
        ):  # noqa: ARG001
            raise OSError("approval queue denied near /\x55sers/example/private/approvals.sqlite")

        store_type.approve_pending_approval_if_ready = raising_approve_if_ready
        try:
            failed_approve = runtime.registry.get("approve_pending_approval").handler(
                {"approval_id": update_failure_id}
            )
        finally:
            store_type.approve_pending_approval_if_ready = original_approve_if_ready
        approve_commands = [
            "pending approvals",
            f"approval detail {update_failure_id}",
            "setup check",
            f"approval readiness {update_failure_id}",
            f"approval packet {update_failure_id}",
        ]
        assert_approval_failure_recovery(
            failed_approve,
            label="approval queue approve update failure",
            approval_id=update_failure_id,
            expected_commands=approve_commands,
            reason="approve_failed",
        )
        if failed_approve.metadata.get("approval_decision_recorded") is not False:
            raise SystemExit(f"Failed approval update must report no recorded decision: {failed_approve.metadata}")
        if failed_approve.metadata.get("actual_status") != "pending":
            raise SystemExit(f"Failed approval update must report the reread pending status: {failed_approve.metadata}")
        if failed_approve.metadata.get("retry_requires_fresh_review") is not True:
            raise SystemExit(f"Failed approval update must require a fresh review: {failed_approve.metadata}")
        if failed_approve.metadata.get("exception_type") != "OSError":
            raise SystemExit(f"Approval update failure missed bounded exception type: {failed_approve.metadata}")
        assert_no_local_path_leak(failed_approve.output, "approval update failure")

        def raising_status_update(_store, _approval_id, _status):
            raise OSError("approval queue denied near /\x55sers/example/private/approvals.sqlite")

        original_status_update = store_type.set_pending_approval_status
        store_type.set_pending_approval_status = raising_status_update
        try:
            failed_dismiss = runtime.registry.get("dismiss_pending_approval").handler(
                {"approval_id": update_failure_id}
            )
        finally:
            store_type.set_pending_approval_status = original_status_update
        dismiss_commands = ["pending approvals", f"approval detail {update_failure_id}", "setup check"]
        assert_approval_failure_recovery(
            failed_dismiss,
            label="approval queue dismiss update failure",
            approval_id=update_failure_id,
            expected_commands=dismiss_commands,
            reason="dismiss_failed",
        )
        if failed_dismiss.metadata.get("exception_type") != "OSError":
            raise SystemExit(f"Dismiss update failure missed bounded exception type: {failed_dismiss.metadata}")
        if failed_dismiss.metadata.get("retry_requires_fresh_review") is not False:
            raise SystemExit(f"Dismiss update failure should not invent an approval review: {failed_dismiss.metadata}")
        assert_no_local_path_leak(failed_dismiss.output, "dismiss update failure")
        if "approval queue denied" in failed_dismiss.output:
            raise SystemExit(f"Dismiss update failure leaked raw backend text: {failed_dismiss.output}")

        temp_approval_id = runtime.store.add_pending_approval(
            "/var/folders/zc/jarvis/session",
            "write file /tmp/jarvis/approval-target.txt",
            "write_text_file",
            "blocked output mentioned /var/folders/zc/jarvis/approval-target.txt",
            planned_args={
                "path": "/var/folders/zc/jarvis/approval-target.txt",
                "content": "body from /tmp/jarvis/approval-content",
            },
        )
        temp_outputs = {
            "pending approvals temp": runtime.registry.get("list_pending_approvals").handler({"limit": 5}).output,
            "approval summary temp": runtime.registry.get("approval_queue_summary").handler({"limit": 5}).output,
            "approval detail temp": runtime.registry.get("inspect_pending_approval").handler({"approval_id": temp_approval_id}).output,
            "approval readiness temp": runtime.registry.get("approval_readiness_packet").handler({"approval_id": temp_approval_id}).output,
            "approval packet temp": runtime.registry.get("approval_execution_packet").handler({"approval_id": temp_approval_id}).output,
            "approval history temp": runtime.registry.get("approval_history").handler({"limit": 5}).output,
            "approval chain proof temp": runtime.registry.get("approval_chain_proof").handler({"approval_id": temp_approval_id}).output,
            "approval review temp": runtime.registry.get("review_pending_approvals").handler({"limit": 5}).output,
        }
        for label, output in temp_outputs.items():
            assert_no_local_path_leak(output, label)
            if "<local-path>" not in output:
                raise SystemExit(f"{label} should show redaction markers for temp-root approval text:\n{output}")
        latest_readiness = runtime.registry.get("approval_readiness_packet").handler({"approval_id": "latest"})
        if not latest_readiness.ok or latest_readiness.metadata.get("approval_id") != temp_approval_id:
            raise SystemExit(f"Approval readiness latest should resolve newest pending approval: {latest_readiness.metadata}")
        if latest_readiness.metadata.get("verdict") != "LAST_LOOK_REQUIRED" or latest_readiness.metadata.get("next_command") != f"approval packet {temp_approval_id}":
            raise SystemExit(f"Approval readiness latest missed newest approval next command: {latest_readiness.metadata}")
        latest_detail = runtime.registry.get("inspect_pending_approval").handler({"approval_id": "latest"})
        if not latest_detail.ok or latest_detail.metadata.get("approval_id") != temp_approval_id:
            raise SystemExit(f"Approval detail latest should resolve newest pending approval: {latest_detail.metadata}")
        latest_packet = runtime.registry.get("approval_execution_packet").handler({"approval_id": "latest"})
        if not latest_packet.ok or latest_packet.metadata.get("approval_id") != temp_approval_id:
            raise SystemExit(f"Approval packet latest should resolve newest pending approval: {latest_packet.metadata}")
        latest_chain = runtime.registry.get("approval_chain_proof").handler({"approval_id": "latest"})
        if not latest_chain.ok or latest_chain.metadata.get("approval_id") != temp_approval_id:
            raise SystemExit(f"Approval chain proof latest should resolve newest pending approval: {latest_chain.metadata}")
        latest_surfaces = {
            "latest approval readiness output": latest_readiness.output,
            "latest approval detail output": latest_detail.output,
            "latest approval packet output": latest_packet.output,
            "latest approval chain proof output": latest_chain.output,
            "latest approval readiness metadata": json.dumps(latest_readiness.metadata, sort_keys=True),
            "latest approval detail metadata": json.dumps(latest_detail.metadata, sort_keys=True),
            "latest approval packet metadata": json.dumps(latest_packet.metadata, sort_keys=True),
            "latest approval chain proof metadata": json.dumps(latest_chain.metadata, sort_keys=True),
        }
        for label, text in latest_surfaces.items():
            assert_no_local_path_leak(text, label)
        if latest_packet.metadata.get("planned_args", {}).get("path") != "<local-path>":
            raise SystemExit(f"Approval packet latest should redact planned path metadata: {latest_packet.metadata}")
        if latest_packet.metadata.get("planned_args", {}).get("content") != "body from <local-path>":
            raise SystemExit(f"Approval packet latest should redact planned content metadata: {latest_packet.metadata}")
        pending_note_text = approvals_note.read_text(encoding="utf-8")
        assert_no_local_path_leak(pending_note_text, "pending approvals note temp")
        if "<local-path>" not in pending_note_text:
            raise SystemExit("Pending approvals note should include redaction markers for temp-root approval text.")

        saved_review = runtime.registry.get("save_approval_review").handler({"limit": "bad"})
        if not saved_review.ok or saved_review.metadata.get("limit") != 20:
            raise SystemExit("save_approval_review did not sanitize bad limits.")
        saved_review_path = saved_review.metadata.get("path")
        saved_review_path_display = saved_review.metadata.get("path_display")
        if not saved_review_path or not Path(saved_review_path).exists():
            raise SystemExit(f"save_approval_review missed exact saved path metadata: {saved_review.metadata}")
        if saved_review_path in saved_review.output:
            raise SystemExit("save_approval_review should not print the raw local note path.")
        if saved_review_path_display != "Automations/Approval Review.md":
            raise SystemExit(f"save_approval_review missed vault-relative display metadata: {saved_review.metadata}")
        if "Approval review saved: Automations/Approval Review.md" not in saved_review.output:
            raise SystemExit("save_approval_review should print a vault-relative saved-note label.")
        if str(Path(temp)) in saved_review.output or "/private/tmp/" in saved_review.output or "/\x55sers/" in saved_review.output or "/var/folders/" in saved_review.output or "/tmp/" in saved_review.output:
            raise SystemExit("save_approval_review output should not expose local temp or user paths.")
        saved_note_text = approval_review_note.read_text(encoding="utf-8")
        assert_no_local_path_leak(saved_note_text, "saved approval review note temp")
        if "<local-path>" not in saved_note_text:
            raise SystemExit("Saved approval review note should include redaction markers for temp-root approval text.")
        bool_saved_review = runtime.registry.get("save_approval_review").handler({"limit": True})
        if not bool_saved_review.ok or bool_saved_review.metadata.get("limit") != 20:
            raise SystemExit("save_approval_review should treat boolean limits as malformed.")
        if saved_review.metadata.get("writes_files") is not True or saved_review.metadata.get("writes_notes") is not True:
            raise SystemExit("save_approval_review missed note write metadata.")
        assert_no_future_authority(saved_review.metadata, "save approval review")
        assert_operator_limits(saved_review.metadata, "save approval review")

        approve_display_id = runtime.store.add_pending_approval(
            "/private/tmp/jarvis/session",
            "write file /private/tmp/jarvis/approved-target.txt",
            "write_text_file",
            "blocked output mentioned /private/tmp/jarvis/approved-target.txt",
            planned_args={
                "path": "/private/tmp/jarvis/approved-target.txt",
                "content": "body from /private/tmp/jarvis/approved-content",
                "nested": {"source": "/\x55sers/example/private/source.txt"},
            },
        )
        display_readiness = runtime.registry.get("approval_readiness_packet").handler(
            {"approval_id": approve_display_id}
        )
        if not display_readiness.ok or display_readiness.metadata.get("approval_readiness_receipt_issued") is not True:
            raise SystemExit(f"Approval display fixture readiness failed: {display_readiness}")
        display_packet = runtime.registry.get("approval_execution_packet").handler({"approval_id": approve_display_id})
        if not display_packet.ok:
            raise SystemExit(f"Approval display fixture packet failed: {display_packet.output}")
        approved_display = runtime.registry.get("approve_pending_approval").handler({"approval_id": approve_display_id})
        if not approved_display.ok:
            raise SystemExit(f"Approval display fixture approve failed: {approved_display.output}")
        assert_no_local_path_leak(approved_display.output, "approved path-shaped approval output")
        display_metadata = {
            "rerun_user_input_display": approved_display.metadata.get("rerun_user_input_display"),
            "exact_rerun_args_display": approved_display.metadata.get("exact_rerun_args_display"),
        }
        assert_no_local_path_leak(json.dumps(display_metadata, sort_keys=True), "approved path-shaped approval display metadata")
        if display_metadata["rerun_user_input_display"] != "write file <local-path>":
            raise SystemExit(f"Approved approval missed redacted user input display metadata: {approved_display.metadata}")
        display_args = display_metadata["exact_rerun_args_display"]
        if not isinstance(display_args, dict):
            raise SystemExit(f"Approved approval missed redacted args display metadata: {approved_display.metadata}")
        if display_args.get("path") != "<local-path>" or display_args.get("content") != "body from <local-path>":
            raise SystemExit(f"Approved approval missed redacted args display values: {approved_display.metadata}")
        if display_args.get("nested", {}).get("source") != "<local-path>":
            raise SystemExit(f"Approved approval missed nested redacted args display values: {approved_display.metadata}")
        if approved_display.metadata.get("exact_rerun_args", {}).get("path") != "/private/tmp/jarvis/approved-target.txt":
            raise SystemExit("Approved approval should preserve exact args for the runtime rerun handoff.")
        assert_no_future_authority(approved_display.metadata, "approved path-shaped approval")


if __name__ == "__main__":
    main()
