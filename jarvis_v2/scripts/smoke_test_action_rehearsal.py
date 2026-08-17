from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.rehearsal import (
    _learning_debt_blocks_rehearsal,
    _metadata_bool,
    _rehearsal_handoff,
    _recovery_closure_lines,
    _recovery_closure_metadata,
    _turn_route_metadata,
)


def _assert_action_handoff(metadata: dict, *, source: str = "action_rehearsal") -> None:
    handoff = metadata.get("action_rehearsal_handoff") or {}
    if handoff.get("source") != source:
        raise SystemExit(f"Action rehearsal handoff missed source: {handoff}")
    if metadata.get("action_rehearsal_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"Action rehearsal handoff missed readiness aliases: {handoff} vs {metadata}")
    expected_flat = {
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "next_safe_command": metadata.get("next_command"),
        "next_safe_commands": metadata.get("recommended_next_commands"),
        "next_safe_command_count": len(metadata.get("recommended_next_commands") or []),
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, expected_value in expected_flat.items():
        if handoff.get(key) != expected_value:
            raise SystemExit(f"Action rehearsal handoff {key} diverged: {handoff} vs {metadata}")
        if metadata.get(f"action_rehearsal_{key}") != expected_value:
            raise SystemExit(f"Action rehearsal flat alias {key} diverged: {handoff} vs {metadata}")
    for key in ["mode", "route", "recommendation", "next_command", "safe_to_execute_now", "approval_required"]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"Action rehearsal handoff diverged for {key}: {handoff} vs {metadata}")
    if handoff.get("recommended_next_commands") != metadata.get("recommended_next_commands"):
        raise SystemExit(f"Action rehearsal handoff missed recommended command queue: {handoff}")
    if handoff.get("recommended_next_command_count") != len(metadata.get("recommended_next_commands") or []):
        raise SystemExit(f"Action rehearsal handoff missed recommended command count: {handoff}")
    planned = metadata.get("planned_actions") or []
    if handoff.get("planned_action_count") != len(planned):
        raise SystemExit(f"Action rehearsal handoff missed planned action count: {handoff}")
    if handoff.get("planned_tools") != [str(action.get("tool") or "") for action in planned]:
        raise SystemExit(f"Action rehearsal handoff missed planned tools: {handoff}")
    if handoff.get("risk_counts") != metadata.get("risk_counts"):
        raise SystemExit(f"Action rehearsal handoff missed risk counts: {handoff}")
    for key in [
        "recovery_closure_proof_queue",
        "recovery_closure_proof_queue_count",
        "recovery_closure_next_proof_command",
        "approval_held_review_required",
        "approval_held_review_commands",
        "approval_held_review_command_count",
        "approval_held_review_next_command",
        "approval_held_review_target_run_id",
        "approval_held_review_target_tool_name",
        "execution_learning_failed_or_blocked_action_runs",
        "execution_learning_approval_held_action_runs",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"Action rehearsal handoff missed proof-debt field {key}: {handoff} vs {metadata}")
    expected = {
        "calls_model": False,
        "calls_chat_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_notes": False,
        "writes_memory": False,
        "controls_computer": False,
        "requires_approval": False,
        "speaks": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, expected_value in expected.items():
        if handoff.get(key) != expected_value:
            raise SystemExit(f"Action rehearsal handoff unsafe {key}: {handoff}")
    for value in [handoff.get("next_command"), handoff.get("recommended_next_commands"), handoff.get("planned_tools")]:
        text = str(value)
        if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Action rehearsal handoff leaked local path text: {handoff}")


def main() -> None:
    for malformed_bool in ("true", "false", "yes", 1, 0, None):
        if _metadata_bool(malformed_bool):
            raise SystemExit(f"Malformed rehearsal boolean should default false: {malformed_bool!r}")
    if _metadata_bool(True) is not True or _metadata_bool(False) is not False:
        raise SystemExit("Exact rehearsal booleans should pass through unchanged.")

    for malformed_count in ("not-a-number", True, float("inf"), None):
        if _learning_debt_blocks_rehearsal(
            {"blocks_completion_claim": True, "failed_or_blocked_action_runs": malformed_count}
        ):
            raise SystemExit(f"Malformed learning-debt counts should not block rehearsal: {malformed_count!r}")
    if _learning_debt_blocks_rehearsal(
        {"blocks_completion_claim": "true", "failed_or_blocked_action_runs": "2"}
    ):
        raise SystemExit("Malformed learning-debt block booleans should not block rehearsal.")
    if not _learning_debt_blocks_rehearsal(
        {"blocks_completion_claim": True, "failed_or_blocked_action_runs": "2"}
    ):
        raise SystemExit("Valid learning-debt counts should still block rehearsal.")

    malformed_recovery = {
        "state": "CLOSURE_REQUIRED",
        "ready_to_retry": False,
        "missing": ["verification"],
        "missing_count": 1,
        "required_commands": ["execution health recovery latest"],
        "next_required_command": "execution health recovery latest",
        "blocks_auto_execution": "true",
        "target_run_id": 7,
        "target_tool_name": "demo",
    }
    route = _turn_route_metadata(
        request="remember malformed rehearsal bools",
        mode="tool",
        approval_required=False,
        planned_actions=[{"tool": "remember", "risk": "LOCAL_SAFE"}],
        recovery_closure=malformed_recovery,
    )
    if route.get("route") == "recovery_closure" or route.get("safe_to_execute_now") is not True:
        raise SystemExit(f"Malformed recovery closure booleans should not block action rehearsal route: {route}")
    metadata = _recovery_closure_metadata(malformed_recovery)
    if metadata.get("recovery_closure_blocks_auto_execution") is not False:
        raise SystemExit(f"Malformed recovery closure metadata should project exact false: {metadata}")
    if _recovery_closure_lines(malformed_recovery):
        raise SystemExit("Malformed recovery closure booleans should not render a recovery blocker.")
    handoff = _rehearsal_handoff(
        source="action_rehearsal",
        text="remember malformed rehearsal bools",
        mode="tool",
        route_metadata={**route, "safe_to_execute_now": "true"},
        planned_actions=[{"tool": "remember", "risk": "LOCAL_SAFE"}],
        risk_counts={},
        approval_required=False,
        recovery_closure=malformed_recovery,
        execution_learning_debt={"blocks_completion_claim": "true", "required_commands": []},
    )
    if handoff.get("safe_to_execute_now") is not False:
        raise SystemExit(f"Malformed route safe boolean should project exact false in handoff: {handoff}")
    if handoff.get("recovery_closure_blocks_auto_execution") is not False:
        raise SystemExit(f"Malformed recovery closure boolean should project exact false in handoff: {handoff}")
    if handoff.get("execution_learning_blocks_completion_claim") is not False:
        raise SystemExit(f"Malformed learning debt boolean should project exact false in handoff: {handoff}")

    with TemporaryDirectory(prefix="jarvis-action-rehearsal-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            (
                "rehearse: run command python3 --version",
                ["Jarvis action rehearsal", "run_shell_command", "HIGH_RISK", "approval required", "approval readiness #ID", "approval packet #ID", "approval chain proof #ID", "does not execute tools"],
            ),
            (
                "dry run: remember that action rehearsal is read only",
                ["Jarvis action rehearsal", "remember", "LOCAL_SAFE", "would be allowed by default", "does not execute tools"],
            ),
            (
                "what would Jarvis do: get clipboard",
                ["Jarvis action rehearsal", "get_clipboard", "PERSONAL_DATA", "approval required", "approval readiness #ID", "approval packet #ID", "approval chain proof #ID", "read personal data"],
            ),
        ]
        for case, required in cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run as read-only.")
            missing = [item for item in required if item not in result.response]
            if missing:
                raise SystemExit(f"Action rehearsal missing expected text for '{case}': {missing}")
            metadata = result.tool_results[0].metadata
            _assert_action_handoff(metadata)
            if not metadata.get("next_command"):
                raise SystemExit(f"Action rehearsal missed next_command metadata for '{case}': {metadata}")
            if "run command" in case or "get clipboard" in case:
                if not str(metadata.get("next_command", "")).startswith("execution governor: "):
                    raise SystemExit(f"Risky action rehearsal should route through execution governor first: {metadata}")
                commands = metadata.get("recommended_next_commands", [])
                if not commands or not str(commands[0]).startswith("execution governor: "):
                    raise SystemExit(f"Risky action rehearsal recommended commands should start with execution governor: {metadata}")
                for command in ["approval readiness latest", "approval packet latest", "approval chain proof latest"]:
                    if command not in commands:
                        raise SystemExit(f"Risky action rehearsal missed approval-chain follow-up {command}: {metadata}")
            for key in [
                "calls_model",
                "calls_chat_model",
                "executes_tools",
                "queues_approval",
                "approves_request",
                "dismisses_request",
                "reads_private_data",
                "reads_personal_data",
                "executes_side_effect",
                "external_side_effect",
                "writes_files",
                "writes_notes",
                "writes_memory",
                "controls_computer",
                "requires_approval",
                "speaks",
                "authorizes_execution",
                "authorizes_completion_claim",
                "approval_granted",
            ]:
                if metadata.get(key):
                    raise SystemExit(f"Action rehearsal unsafe metadata {key}: {metadata}")

        if runtime.store.list_pending_approvals(limit=10):
            raise SystemExit("Action rehearsal should not queue pending approvals.")

        memory_result = runtime.handle("search memory for action rehearsal is read only")
        if "No memories found" not in memory_result.response:
            raise SystemExit("Action rehearsal should not create memory while previewing remember.")

        bounded = runtime.registry.get("action_rehearsal").handler({"request": "run command " + ("python3 --version " * 80)})
        print("[ok] direct bounded action_rehearsal")
        print(bounded.output[:900])
        print()
        if not bounded.ok or len(bounded.metadata.get("request", "")) > 500:
            raise SystemExit(f"Action rehearsal request was not bounded: {bounded.metadata}")

        chat_preview = runtime.registry.get("action_rehearsal").handler({"request": "can we talk about memory"})
        _assert_action_handoff(chat_preview.metadata)
        if chat_preview.metadata.get("route") != "chat":
            raise SystemExit(f"Action rehearsal missed chat route metadata: {chat_preview.metadata}")
        if chat_preview.metadata.get("safe_to_execute_now") is not True:
            raise SystemExit(f"Action rehearsal should mark chat preview safe to send: {chat_preview.metadata}")

        build_preview = runtime.registry.get("action_rehearsal").handler(
            {"request": "continue building Jarvis V2 as an agent harness"}
        )
        _assert_action_handoff(build_preview.metadata)
        if build_preview.metadata.get("route") != "auto_tool":
            raise SystemExit(f"Action rehearsal missed harness build route metadata: {build_preview.metadata}")
        if not str(build_preview.metadata.get("next_command", "")).startswith("execution governor: "):
            raise SystemExit(f"Action rehearsal local-safe route should still start with execution governor: {build_preview.metadata}")
        planned = build_preview.metadata.get("planned_actions") or []
        if not planned or planned[0].get("tool") != "harness_build_slice":
            raise SystemExit(f"Action rehearsal missed harness build planned action: {build_preview.metadata}")

    with TemporaryDirectory(prefix="jarvis-action-learning-debt-") as temp:
        runtime = make_temp_runtime(Path(temp))
        run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "remember",
            "LOCAL_SAFE",
            True,
            False,
            "Remembered a useful local fact.",
            metadata={"fact": "learning debt rehearsal fixture"},
        )
        held = runtime.registry.get("action_rehearsal").handler(
            {"request": "remember that previews should close learning debt first"}
        )
        _assert_action_handoff(held.metadata)
        print("[ok] successful-action-learning-context action_rehearsal")
        print(held.output[:1100])
        print()
        if held.metadata.get("route") != "auto_tool":
            raise SystemExit(f"Successful local action learning context should not block rehearsal routing: {held.metadata}")
        if held.metadata.get("safe_to_execute_now") is not True:
            raise SystemExit(f"Successful local action learning context should not mark rehearsal unsafe: {held.metadata}")
        if held.metadata.get("execution_learning_state") != "LEARNING_REVIEW_REQUIRED":
            raise SystemExit(f"Action rehearsal missed learning debt state: {held.metadata}")
        expected_closure = f"execution learning closure {run_id}"
        expected = f"after-action learning packet {run_id}"
        if held.metadata.get("next_command") != "execution governor: remember that previews should close learning debt first":
            raise SystemExit(f"Action rehearsal should keep normal routing for successful action context: {held.metadata}")
        learning_commands = held.metadata.get("execution_learning_required_commands", [])
        if expected_closure not in learning_commands:
            raise SystemExit(f"Action rehearsal missed required learning closure command {expected_closure!r}: {held.metadata}")
        if expected not in learning_commands:
            raise SystemExit(f"Action rehearsal missed required learning command {expected!r}: {held.metadata}")
        if learning_commands.index(expected) > learning_commands.index(expected_closure):
            raise SystemExit(f"Action rehearsal should require after-action learning before learning closure: {held.metadata}")
        if held.metadata.get("execution_learning_proof_queue") != learning_commands:
            raise SystemExit(f"Action rehearsal learning proof queue diverged: {held.metadata}")
        if held.metadata.get("execution_learning_proof_queue_count") != len(learning_commands):
            raise SystemExit(f"Action rehearsal learning proof queue count diverged: {held.metadata}")
        if held.metadata.get("execution_learning_next_proof_command") != held.metadata.get("execution_learning_next_required_command"):
            raise SystemExit(f"Action rehearsal missed learning next proof alias: {held.metadata}")
        if "Execution learning debt:" not in held.output:
            raise SystemExit("Successful action learning context should remain visible in action rehearsal.")

    with TemporaryDirectory(prefix="jarvis-action-approval-held-review-") as temp:
        runtime = make_temp_runtime(Path(temp))
        approval_id = runtime.store.add_pending_approval(
            runtime.session_id,
            "send 가상연락처이 a telegram saying hello",
            "send_telegram",
            "Korean Telegram send fixture requires approval review",
            {"to": "가상연락처이", "body": "hello"},
        )
        held_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "send_telegram",
            "HIGH_RISK",
            False,
            False,
            f"'send_telegram' is risk level HIGH_RISK; explicit approval required. queued as approval #{approval_id}.",
            approval_id=approval_id,
            metadata={"failure_kind": "approval-gate", "requires_confirmation": True},
        )
        held = runtime.registry.get("action_rehearsal").handler(
            {"request": "remember that approval-held sends need approval review first"}
        )
        _assert_action_handoff(held.metadata)
        print("[ok] approval-held-review action_rehearsal")
        print(held.output[:1400])
        print()
        if held.metadata.get("route") != "approval_held_review":
            raise SystemExit(f"Approval-held-only action rehearsal should route to approval review: {held.metadata}")
        if held.metadata.get("recommendation") != "APPROVAL_HELD_REVIEW_REQUIRED":
            raise SystemExit(f"Approval-held-only action rehearsal should name approval review: {held.metadata}")
        if held.metadata.get("safe_to_execute_now") is not False:
            raise SystemExit(f"Approval-held-only action rehearsal should not mark execution safe: {held.metadata}")
        if held.metadata.get("approval_held_review_required") is not True:
            raise SystemExit(f"Action rehearsal missed approval-held review flag: {held.metadata}")
        if held.metadata.get("approval_held_review_target_run_id") != held_run_id:
            raise SystemExit(f"Action rehearsal missed approval-held target run: {held.metadata}")
        if held.metadata.get("approval_held_review_target_tool_name") != "send_telegram":
            raise SystemExit(f"Action rehearsal missed approval-held target tool: {held.metadata}")
        approval_commands = held.metadata.get("approval_held_review_commands") or []
        for command in [
            f"approval readiness {approval_id}",
            f"approval packet {approval_id}",
            f"approval chain proof {approval_id}",
        ]:
            if command not in approval_commands:
                raise SystemExit(f"Action rehearsal missed approval-held review command {command!r}: {held.metadata}")
        if held.metadata.get("next_command") != f"approval readiness {approval_id}":
            raise SystemExit(f"Action rehearsal next command should point at approval readiness: {held.metadata}")
        if f"execution recovery packet {held_run_id}" in str(held.metadata.get("recommended_next_commands")):
            raise SystemExit(f"Action rehearsal should not recover approval-held run #{held_run_id}: {held.metadata}")
        if held.metadata.get("execution_learning_failed_or_blocked_action_runs") != 0:
            raise SystemExit(f"Action rehearsal should not count approval-held send as failed learning debt: {held.metadata}")
        if held.metadata.get("execution_learning_approval_held_action_runs") != 1:
            raise SystemExit(f"Action rehearsal missed approval-held learning context: {held.metadata}")
        for required in [
            "Current blocker: approval-held execution review",
            "Approval-held execution review:",
            f"approval readiness {approval_id}",
        ]:
            if required not in held.output:
                raise SystemExit(f"Action rehearsal missed approval-held review text {required!r}: {held.output}")
        for forbidden in [
            "Execution health recovery closure:",
            f"execution recovery packet {held_run_id}",
        ]:
            if forbidden in held.output:
                raise SystemExit(f"Action rehearsal rendered approval-held run as recovery debt: {held.output}")

        assistant = runtime.registry.get("assistant_turn_rehearsal").handler(
            {"message": "remember that approval-held sends need approval review first"}
        )
        assistant_handoff = assistant.metadata.get("assistant_turn_rehearsal_handoff") or {}
        if assistant.metadata.get("route") != "approval_held_review":
            raise SystemExit(f"Assistant turn rehearsal should route held rows to approval review: {assistant.metadata}")
        if assistant.metadata.get("approval_held_review_target_run_id") != held_run_id:
            raise SystemExit(f"Assistant turn rehearsal missed held target: {assistant.metadata}")
        if assistant_handoff.get("approval_held_review_commands") != assistant.metadata.get("approval_held_review_commands"):
            raise SystemExit(f"Assistant handoff missed approval-held review commands: {assistant.metadata}")
        if "Approval-held execution review:" not in assistant.output:
            raise SystemExit(f"Assistant turn rehearsal missed approval-held review section: {assistant.output}")
        if "Execution health recovery closure:" in assistant.output:
            raise SystemExit(f"Assistant turn rehearsal rendered approval-held row as recovery closure: {assistant.output}")

    with TemporaryDirectory(prefix="jarvis-action-failed-learning-debt-") as temp:
        runtime = make_temp_runtime(Path(temp))
        run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "checkpoint_recovery_execute",
            "LOCAL_SAFE",
            False,
            False,
            "reviewed local-safe step failed verification",
            metadata={"failure_kind": "verification_failed"},
        )
        held = runtime.registry.get("action_rehearsal").handler(
            {"request": "remember that failed previews should close learning debt first"}
        )
        _assert_action_handoff(held.metadata)
        print("[ok] failed-action-learning-debt action_rehearsal")
        print(held.output[:1200])
        print()
        if held.metadata.get("route") != "recovery_closure":
            raise SystemExit(f"Failed action should still hold behind recovery closure first: {held.metadata}")
        if held.metadata.get("safe_to_execute_now") is not False:
            raise SystemExit(f"Failed action learning debt should block safe execution: {held.metadata}")
        if held.metadata.get("execution_learning_state") != "LEARNING_DEBT_AFTER_FAILURE":
            raise SystemExit(f"Action rehearsal missed failed-run learning debt state: {held.metadata}")
        expected_closure = f"execution learning closure {run_id}"
        expected = f"after-action learning packet {run_id}"
        learning_commands = held.metadata.get("execution_learning_required_commands", [])
        if expected_closure not in learning_commands:
            raise SystemExit(f"Action rehearsal missed failed-run learning closure command {expected_closure!r}: {held.metadata}")
        if expected not in learning_commands:
            raise SystemExit(f"Action rehearsal missed failed-run learning command {expected!r}: {held.metadata}")
        if learning_commands.index(expected) > learning_commands.index(expected_closure):
            raise SystemExit(f"Action rehearsal should require failed-run after-action learning before learning closure: {held.metadata}")
        if held.metadata.get("execution_learning_proof_queue") != learning_commands:
            raise SystemExit(f"Action rehearsal failed-run learning proof queue diverged: {held.metadata}")
        if held.metadata.get("execution_learning_proof_queue_count") != len(learning_commands):
            raise SystemExit(f"Action rehearsal failed-run learning proof queue count diverged: {held.metadata}")
        if held.metadata.get("execution_learning_next_proof_command") != held.metadata.get("execution_learning_next_required_command"):
            raise SystemExit(f"Action rehearsal missed failed-run learning next proof alias: {held.metadata}")
        closure_commands = held.metadata.get("recovery_closure_required_commands", [])
        if held.metadata.get("recovery_closure_proof_queue") != closure_commands:
            raise SystemExit(f"Action rehearsal recovery proof queue diverged from closure commands: {held.metadata}")
        if held.metadata.get("recovery_closure_proof_queue_count") != len(closure_commands):
            raise SystemExit(f"Action rehearsal recovery proof queue count diverged: {held.metadata}")
        if held.metadata.get("recovery_closure_next_proof_command") != held.metadata.get("recovery_closure_next_required_command"):
            raise SystemExit(f"Action rehearsal missed next recovery proof command alias: {held.metadata}")
        handoff = held.metadata.get("action_rehearsal_handoff") or {}
        if handoff.get("recovery_closure_proof_queue") != held.metadata.get("recovery_closure_proof_queue"):
            raise SystemExit(f"Action rehearsal handoff missed failed-run recovery proof queue: {held.metadata}")
        if handoff.get("execution_learning_proof_queue") != held.metadata.get("execution_learning_proof_queue"):
            raise SystemExit(f"Action rehearsal handoff missed failed-run learning proof queue: {held.metadata}")
        assistant = runtime.registry.get("assistant_turn_rehearsal").handler(
            {"message": "remember that failed previews should close learning debt first"}
        )
        assistant_handoff = assistant.metadata.get("assistant_turn_rehearsal_handoff") or {}
        if assistant_handoff.get("source") != "assistant_turn_rehearsal":
            raise SystemExit(f"Assistant turn rehearsal handoff missed source: {assistant.metadata}")
        if assistant_handoff.get("recovery_closure_proof_queue") != assistant.metadata.get("recovery_closure_proof_queue"):
            raise SystemExit(f"Assistant turn rehearsal handoff missed recovery proof queue: {assistant.metadata}")
        if assistant_handoff.get("execution_learning_proof_queue") != assistant.metadata.get("execution_learning_proof_queue"):
            raise SystemExit(f"Assistant turn rehearsal handoff missed learning proof queue: {assistant.metadata}")
        for required in ["Execution learning debt:", "next learning required", "learning proof queue"]:
            if required not in held.output:
                raise SystemExit(f"Action rehearsal did not render failed-run learning debt detail {required!r}.")
        for stale in ["next required proof", "next learning proof"]:
            if stale in held.output:
                raise SystemExit(f"Action rehearsal should not use ambiguous proof-first prose {stale!r}.")
        if "proof queue" not in held.output:
            raise SystemExit("Action rehearsal did not render the recovery proof queue.")


if __name__ == "__main__":
    main()
