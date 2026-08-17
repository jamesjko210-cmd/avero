from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.autonomy import (
    _action_readiness_handoff,
    _execution_acceptance_handoff,
    _execution_health_recovery_closure_snapshot,
    _execution_readiness_handoff,
    _metadata_all_bool,
    _metadata_any_bool,
    _metadata_bool,
    _metadata_int,
)


def test_planner_routes_autonomy_plan_aliases() -> None:
    # Real gap found live 2026-07-09: "show autonomy plan" fell through to
    # chat while bare "autonomy plan" worked.
    p = RuleBasedPlanner()
    for q in ("autonomy plan", "show autonomy plan"):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["autonomy_plan"]:
            raise SystemExit(f"autonomy_plan route missed: {q!r} -> {[a.tool_name for a in actions]}")


def test_autonomy_tools_accept_request_arg_aliases() -> None:
    # Fable checkpoint (2026-07-10): found while verifying the core agent-harness
    # planning loop that `autonomy_plan`, `risk_preflight`, `action_readiness_
    # packet`, and `risky_request_lifecycle` accepted ONLY the `request` arg key,
    # while 10 sibling autonomy tools already accept `request`/`order`/`goal`.
    # A model-generated tool call using the natural `goal` or `order` key hit an
    # unhelpful "tell me what to plan" instead of working -- a robustness gap in
    # the agentic path (where the model, not the planner, supplies args). Widened
    # the four to the same alias set. This test proves each now accepts all three
    # keys AND still returns the clarification prompt (ok=False) on genuinely
    # empty args, so the widening did not weaken the empty-input guard.
    with TemporaryDirectory(prefix="jarvis-autonomy-alias-") as temp:
        runtime = make_temp_runtime(Path(temp))
        sample = "organize my tasks and pick the top priority"
        for tool_name in (
            "autonomy_plan",
            "risk_preflight",
            "action_readiness_packet",
            "risky_request_lifecycle",
        ):
            tool = runtime.registry.get(tool_name)
            for key in ("request", "order", "goal"):
                result = tool.handler({key: sample})
                if not result.ok:
                    raise SystemExit(
                        f"{tool_name} should accept the {key!r} arg alias: got ok=False ({result.output!r})"
                    )
            empty = tool.handler({})
            if empty.ok:
                raise SystemExit(
                    f"{tool_name} must still ask for clarification on empty args, got ok=True"
                )


class HostileMetadataValue:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __bool__(self) -> bool:
        raise RuntimeError(self.marker)

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-autonomy-metadata {self.marker}>"


def assert_safe_metadata(metadata: dict, label: str) -> None:
    for key in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "writes_files",
        "writes_notes",
        "writes_memory",
        "executes_side_effect",
        "external_side_effect",
        "controls_computer",
        "calls_external_services",
        "requires_approval",
        "speaks",
    ]:
        if metadata.get(key):
            raise SystemExit(f"{label} should stay read-only, got {key}: {metadata}")
    if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} missed the operator stop-time/work-window override metadata: {metadata}")


def assert_frontdoor_handoff(metadata: dict, label: str, key: str, *, status: str = "ok") -> None:
    ready_key = f"{key}_handoff_ready"
    handoff_key = f"{key}_handoff"
    if metadata.get(ready_key) is not True:
        raise SystemExit(f"{label} missed {key} handoff readiness: {metadata}")
    handoff = metadata.get(handoff_key) or {}
    expected_source = {
        "command_intake": "command_intake_packet",
        "dispatch_decision": "dispatch_decision_packet",
        "execution_governor": "execution_governor_packet",
        "command_cockpit": "command_cockpit_packet",
        "planner_gap": "planner_gap_packet",
    }[key]
    if handoff.get("source") != expected_source or handoff.get("status") != status:
        raise SystemExit(f"{label} missed {key} handoff source/status: {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != [] or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} {key} handoff should be content-free and no-change: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} {key} handoff missed operator readiness: {handoff}")
    commands = handoff.get("recommended_next_commands") or []
    if not commands or handoff.get("recommended_next_command_count") != len(commands):
        raise SystemExit(f"{label} {key} handoff missed recommended command count: {handoff}")
    if handoff.get("next_command") and handoff.get("next_command") != commands[0]:
        raise SystemExit(f"{label} {key} handoff next command should lead recommended commands: {handoff}")
    for field in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "writes_files",
        "writes_database",
        "writes_notes",
        "writes_memory",
        "executes_side_effect",
        "external_side_effect",
        "controls_computer",
        "calls_external_service",
        "calls_external_services",
        "requires_approval",
        "speaks",
        "completes_tasks",
    ]:
        if handoff.get(field) is not False:
            raise SystemExit(f"{label} {key} handoff unsafe {field}: {handoff}")
    if handoff.get("operator_timeboxes_override_priority") is not True or handoff.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} {key} handoff missed operator override flags: {handoff}")
    for proof_key in [
        "recovery_closure_proof_queue",
        "recovery_closure_proof_queue_count",
        "recovery_closure_next_proof_command",
        "approval_held_review_required",
        "approval_held_review_commands",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
    ]:
        if proof_key in metadata and handoff.get(proof_key) != metadata.get(proof_key):
            raise SystemExit(f"{label} {key} handoff missed proof field {proof_key}: {handoff} vs {metadata}")
    text = str(handoff)
    if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"{label} {key} handoff leaked local path text: {handoff}")


def assert_autonomy_exact_metadata_bool_contract() -> None:
    for malformed_bool in ("true", "false", "yes", 1, 0, None):
        if _metadata_bool(malformed_bool):
            raise SystemExit(f"Malformed autonomy boolean should default false: {malformed_bool!r}")
    if _metadata_bool(True) is not True or _metadata_bool(False) is not False:
        raise SystemExit("Exact autonomy booleans should pass through unchanged.")
    if _metadata_any_bool("true", "false", 1, None):
        raise SystemExit("Autonomy any-bool helper should reject malformed truthy values.")
    if _metadata_any_bool(False, True) is not True or _metadata_any_bool(False, False) is not False:
        raise SystemExit("Autonomy any-bool helper should preserve exact booleans.")
    if _metadata_all_bool(True, "true"):
        raise SystemExit("Autonomy all-bool helper should reject malformed truthy values.")
    if _metadata_all_bool(True, True) is not True or _metadata_all_bool(True, False) is not False:
        raise SystemExit("Autonomy all-bool helper should preserve exact booleans.")

    malformed_recovery = {
        "state": "CLOSURE_REQUIRED",
        "ready_to_retry": "true",
        "missing_count": 1,
        "proof_queue": ["execution health recovery latest"],
        "proof_queue_count": 1,
        "next_proof_command": "execution health recovery latest",
        "missing": ["verification"],
        "required_commands": ["execution health recovery latest"],
        "next_required_command": "execution health recovery latest",
        "blocks_auto_execution": "true",
        "target_run_id": 7,
        "target_tool_name": "demo",
    }
    malformed_learning = {
        "state": "LEARNING_DEBT_AFTER_FAILURE",
        "blocks_completion_claim": "true",
        "missing": ["after_action_learning"],
        "missing_count": 1,
        "proof_queue": ["after-action learning packet 7"],
        "required_commands": ["after-action learning packet 7"],
        "next_required_command": "after-action learning packet 7",
    }
    action_handoff = _action_readiness_handoff(
        display_request="remember malformed autonomy bools",
        route="auto_tool",
        recommendation="AUTO_RUN_LOCAL_SAFE",
        reason="exact bool contract",
        recommended_next_commands=["execution governor: remember malformed autonomy bools"],
        safe_to_execute_now="true",
        approval_required="true",
        matched_risks=[],
        missing_checks=[],
        pending_approval_count=0,
        recovery_closure=malformed_recovery,
        recovery_debt_visible="true",
        readiness_preview_allowed_with_recovery_debt="true",
        recovery_closure_blocks_current_action="true",
        risk_gated_tool_count=0,
    )
    for field in [
        "safe_to_execute_now",
        "approval_required",
        "recovery_debt_visible",
        "readiness_preview_allowed_with_recovery_debt",
        "recovery_closure_blocks_current_action",
        "recovery_closure_ready_to_retry",
    ]:
        if action_handoff.get(field) is not False:
            raise SystemExit(f"Action readiness handoff should reject malformed boolean {field}: {action_handoff}")

    readiness_handoff = _execution_readiness_handoff(
        display_request="remember malformed autonomy bools",
        planner_goal="remember malformed autonomy bools",
        verdict="READY_FOR_AUTO_SAFE_TOOL_ROUTE",
        next_command="execution governor: remember malformed autonomy bools",
        matched_risks=[],
        pending_approval_count=0,
        planned_actions=[],
        approval_required="true",
        unknown_tools=0,
        matrix_row_count=1,
        proof_requirement_count=0,
        failed_or_blocked_run_count=0,
        approval_held_run_count=0,
        verification_run_count=0,
        recovery_closure=malformed_recovery,
        recovery_debt_visible="true",
        readiness_matrix_preview_allowed_with_recovery_debt="true",
        recovery_closure_blocks_matrix_execution="true",
        learning_debt=malformed_learning,
        risk_gated_tool_count=0,
        approval_forecast={},
    )
    for field in [
        "approval_required",
        "recovery_closure_ready_to_retry",
        "recovery_closure_blocks_auto_execution",
        "recovery_debt_visible",
        "readiness_matrix_preview_allowed_with_recovery_debt",
        "recovery_closure_blocks_matrix_execution",
        "execution_learning_blocks_completion_claim",
    ]:
        if readiness_handoff.get(field) is not False:
            raise SystemExit(f"Execution readiness handoff should reject malformed boolean {field}: {readiness_handoff}")

    acceptance_handoff = _execution_acceptance_handoff(
        display_request="remember malformed acceptance bools",
        planner_goal="remember malformed acceptance bools",
        verdict="READY_FOR_ACCEPTANCE",
        next_step="operator review",
        next_command="execution contract: remember malformed acceptance bools",
        matched_risks=[],
        planned_action_count=1,
        approval_required="true",
        pending_approval_count=0,
        recent_run_count=1,
        has_recent_success="true",
        has_evidence="true",
        has_tests="true",
        has_recovery="true",
        blocking_reasons=[],
        unknown_tools=0,
        risk_gated_tool_count=0,
        approval_forecast={},
    )
    for field in [
        "approval_required",
        "has_recent_success",
        "has_evidence",
        "has_tests",
        "has_recovery",
    ]:
        if acceptance_handoff.get(field) is not False:
            raise SystemExit(f"Execution acceptance handoff should reject malformed boolean {field}: {acceptance_handoff}")


def assert_recovery_next_required(metadata: dict, output: str, label: str) -> None:
    commands = metadata.get("recovery_closure_required_commands", [])
    if not commands:
        raise SystemExit(f"{label} missed recovery closure command queue: {metadata}")
    if metadata.get("recovery_closure_next_required_command") != commands[0]:
        raise SystemExit(f"{label} missed first recovery closure proof command: {metadata}")
    if metadata.get("recovery_closure_proof_queue") != commands:
        raise SystemExit(f"{label} recovery closure proof queue should mirror required commands: {metadata}")
    if metadata.get("recovery_closure_proof_queue_count") != len(commands):
        raise SystemExit(f"{label} recovery closure proof queue count diverged: {metadata}")
    if metadata.get("recovery_closure_next_proof_command") != commands[0]:
        raise SystemExit(f"{label} missed recovery closure next proof command: {metadata}")
    if "next required proof" in output:
        raise SystemExit(f"{label} should not render ambiguous next required proof prose: {output}")
    if "next required:" not in output:
        raise SystemExit(f"{label} did not render the next required recovery command: {output}")


def assert_recovery_snapshot_proof_aliases(runtime, label: str) -> None:
    snapshot = _execution_health_recovery_closure_snapshot(runtime.store.recent_tool_runs(limit=20))
    commands = snapshot.get("required_commands", [])
    if not commands:
        raise SystemExit(f"{label} recovery snapshot missed command queue: {snapshot}")
    if snapshot.get("proof_queue") != commands:
        raise SystemExit(f"{label} recovery snapshot proof queue should be explicit and mirror commands: {snapshot}")
    if snapshot.get("proof_queue_count") != len(commands):
        raise SystemExit(f"{label} recovery snapshot proof queue count diverged: {snapshot}")
    if snapshot.get("next_proof_command") != snapshot.get("next_required_command"):
        raise SystemExit(f"{label} recovery snapshot next proof should come from explicit alias: {snapshot}")


def assert_learning_debt(metadata: dict, output: str, label: str, *, state: str, run_id: int | None = None, blocks: bool = True) -> None:
    if metadata.get("execution_learning_state") != state:
        raise SystemExit(f"{label} missed execution learning state {state}: {metadata}")
    if metadata.get("execution_learning_blocks_completion_claim") is not blocks:
        raise SystemExit(f"{label} missed execution learning block flag {blocks}: {metadata}")
    learning_commands = metadata.get("execution_learning_required_commands", [])
    if run_id is not None:
        closure = f"execution learning closure {run_id}"
        expected = f"after-action learning packet {run_id}"
        if closure not in learning_commands:
            raise SystemExit(f"{label} missed execution learning closure command {closure!r}: {metadata}")
        if expected not in learning_commands:
            raise SystemExit(f"{label} missed execution learning command {expected!r}: {metadata}")
        if learning_commands.index(expected) > learning_commands.index(closure):
            raise SystemExit(f"{label} should place after-action learning before execution learning closure: {metadata}")
    if metadata.get("execution_learning_proof_queue") != learning_commands:
        raise SystemExit(f"{label} execution learning proof queue diverged from required commands: {metadata}")
    if metadata.get("execution_learning_proof_queue_count") != len(learning_commands):
        raise SystemExit(f"{label} execution learning proof queue count diverged: {metadata}")
    if metadata.get("execution_learning_next_proof_command") != metadata.get("execution_learning_next_required_command"):
        raise SystemExit(f"{label} execution learning next proof alias diverged: {metadata}")
    if "Execution learning debt:" not in output:
        raise SystemExit(f"{label} did not render execution learning debt details.")
    if "next learning proof:" in output:
        raise SystemExit(f"{label} should not render ambiguous next learning proof prose: {output}")
    if "next learning required:" not in output:
        raise SystemExit(f"{label} did not render the next learning required command: {output}")


def assert_risk_preflight_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("risk_preflight_handoff") or {}
    if handoff.get("source") != "risk_preflight":
        raise SystemExit(f"{label} missed risk preflight handoff source: {handoff}")
    for field in ["matched_risks", "likely_approval_required"]:
        if handoff.get(field) != metadata.get(field):
            raise SystemExit(f"{label} handoff diverged for {field}: {handoff} vs {metadata}")
    if handoff.get("matched_risk_count") != len(metadata.get("matched_risks") or []):
        raise SystemExit(f"{label} handoff missed risk count: {handoff}")
    if handoff.get("risk_gated_tool_count") != metadata.get("risk_gated_tools"):
        raise SystemExit(f"{label} handoff missed risk-gated tool count: {handoff}")
    commands = handoff.get("recommended_next_commands") or []
    if handoff.get("recommended_next_command_count") != len(commands):
        raise SystemExit(f"{label} handoff missed recommended-command count: {handoff}")
    if handoff.get("next_command") != (commands[0] if commands else ""):
        raise SystemExit(f"{label} handoff missed next command: {handoff}")
    for field in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "writes_files",
        "writes_database",
        "writes_notes",
        "writes_memory",
        "executes_side_effect",
        "external_side_effect",
        "controls_computer",
        "calls_external_service",
        "calls_external_services",
        "requires_approval",
        "speaks",
        "completes_tasks",
    ]:
        if handoff.get(field) is not False:
            raise SystemExit(f"{label} handoff unsafe {field}: {handoff}")
    text = str(handoff)
    if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"{label} handoff leaked local path text: {handoff}")


def assert_action_readiness_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("action_readiness_handoff") or {}
    if handoff.get("source") != "action_readiness_packet":
        raise SystemExit(f"{label} missed action readiness handoff source: {handoff}")
    for field in [
        "route",
        "recommendation",
        "next_command",
        "recommended_next_commands",
        "safe_to_execute_now",
        "approval_required",
        "matched_risks",
        "missing_checks",
        "recovery_debt_visible",
        "readiness_preview_allowed_with_recovery_debt",
        "recovery_closure_blocks_current_action",
        "approval_held_review_required",
        "approval_held_review_commands",
        "recovery_closure_state",
        "recovery_closure_ready_to_retry",
    ]:
        if handoff.get(field) != metadata.get(field):
            raise SystemExit(f"{label} handoff diverged for {field}: {handoff} vs {metadata}")
    if handoff.get("recommended_next_command_count") != len(metadata.get("recommended_next_commands") or []):
        raise SystemExit(f"{label} handoff missed next-command count: {handoff}")
    if handoff.get("matched_risk_count") != len(metadata.get("matched_risks") or []):
        raise SystemExit(f"{label} handoff missed risk count: {handoff}")
    if handoff.get("missing_check_count") != len(metadata.get("missing_checks") or []):
        raise SystemExit(f"{label} handoff missed missing-check count: {handoff}")
    if handoff.get("pending_approval_count") != metadata.get("pending_approvals"):
        raise SystemExit(f"{label} handoff missed pending approvals: {handoff}")
    if handoff.get("risk_gated_tool_count") != metadata.get("risk_gated_tools"):
        raise SystemExit(f"{label} handoff missed risk-gated tool count: {handoff}")
    if handoff.get("recovery_closure_missing_count") != metadata.get("recovery_closure_missing_count"):
        raise SystemExit(f"{label} handoff missed recovery missing count: {handoff}")
    if handoff.get("recovery_closure_proof_queue") != metadata.get("recovery_closure_proof_queue"):
        raise SystemExit(f"{label} handoff missed recovery proof queue: {handoff}")
    if handoff.get("recovery_closure_proof_queue_count") != metadata.get("recovery_closure_proof_queue_count"):
        raise SystemExit(f"{label} handoff missed recovery proof queue count: {handoff}")
    if handoff.get("recovery_closure_next_proof_command") != metadata.get("recovery_closure_next_proof_command"):
        raise SystemExit(f"{label} handoff missed next recovery proof command: {handoff}")
    for field in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "writes_files",
        "writes_database",
        "writes_notes",
        "writes_memory",
        "executes_side_effect",
        "external_side_effect",
        "controls_computer",
        "calls_external_service",
        "calls_external_services",
        "requires_approval",
        "speaks",
        "completes_tasks",
    ]:
        if handoff.get(field) is not False:
            raise SystemExit(f"{label} handoff unsafe {field}: {handoff}")
    for value in [handoff.get("next_command"), handoff.get("recommended_next_commands")]:
        text = str(value)
        if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"{label} handoff leaked local path text: {handoff}")


def assert_execution_contract_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_contract_handoff") or {}
    if handoff.get("source") != "execution_contract":
        raise SystemExit(f"{label} missed execution contract handoff source: {handoff}")
    for field in [
        "planner_goal",
        "route",
        "approval_required",
        "approval_state",
        "matched_risks",
        "planned_actions",
        "unknown_tools",
    ]:
        if handoff.get(field) != metadata.get(field):
            raise SystemExit(f"{label} handoff diverged for {field}: {handoff} vs {metadata}")
    if handoff.get("matched_risk_count") != len(metadata.get("matched_risks") or []):
        raise SystemExit(f"{label} handoff missed risk count: {handoff}")
    if handoff.get("pending_approval_count") != metadata.get("pending_approvals"):
        raise SystemExit(f"{label} handoff missed pending approvals: {handoff}")
    if handoff.get("planned_action_count") != metadata.get("planned_action_count"):
        raise SystemExit(f"{label} handoff missed planned action count: {handoff}")
    if handoff.get("verification_target_count") != metadata.get("verification_targets"):
        raise SystemExit(f"{label} handoff missed verification target count: {handoff}")
    if handoff.get("recovery_step_count") != metadata.get("recovery_steps"):
        raise SystemExit(f"{label} handoff missed recovery step count: {handoff}")
    if handoff.get("learning_hook_count") != metadata.get("learning_hooks"):
        raise SystemExit(f"{label} handoff missed learning hook count: {handoff}")
    if handoff.get("risk_gated_tool_count") != metadata.get("risk_gated_tools"):
        raise SystemExit(f"{label} handoff missed risk-gated tool count: {handoff}")
    commands = handoff.get("recommended_next_commands") or []
    if handoff.get("recommended_next_command_count") != len(commands):
        raise SystemExit(f"{label} handoff missed recommended-command count: {handoff}")
    if handoff.get("next_command") != (commands[0] if commands else ""):
        raise SystemExit(f"{label} handoff missed next command: {handoff}")
    for field in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "writes_files",
        "writes_database",
        "writes_notes",
        "writes_memory",
        "executes_side_effect",
        "external_side_effect",
        "controls_computer",
        "calls_external_service",
        "calls_external_services",
        "requires_approval",
        "speaks",
        "completes_tasks",
    ]:
        if handoff.get(field) is not False:
            raise SystemExit(f"{label} handoff unsafe {field}: {handoff}")
    text = str(handoff)
    if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"{label} handoff leaked local path text: {handoff}")


def assert_argument_contract_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("argument_contract_handoff") or {}
    if handoff.get("source") != "argument_contract_packet":
        raise SystemExit(f"{label} missed argument contract handoff source: {handoff}")
    for field in [
        "planner_goal",
        "verdict",
        "next_step",
        "next_command",
        "matched_risks",
        "planned_actions",
        "missing_argument_count",
        "argument_warnings",
        "approval_required",
        "unknown_tools",
        "approval_queue_forecast",
        "forecast_new_approvals",
        "forecast_reused_approval_ids",
        "forecast_queue_before",
        "forecast_queue_after_if_sent",
        "forecast_queue_delta_if_sent",
    ]:
        if handoff.get(field) != metadata.get(field):
            raise SystemExit(f"{label} handoff diverged for {field}: {handoff} vs {metadata}")
    if handoff.get("matched_risk_count") != len(metadata.get("matched_risks") or []):
        raise SystemExit(f"{label} handoff missed risk count: {handoff}")
    if handoff.get("pending_approval_count") != metadata.get("pending_approvals"):
        raise SystemExit(f"{label} handoff missed pending approvals: {handoff}")
    if handoff.get("planned_action_count") != metadata.get("planned_action_count"):
        raise SystemExit(f"{label} handoff missed planned action count: {handoff}")
    if handoff.get("risk_gated_tool_count") != metadata.get("risk_gated_tools"):
        raise SystemExit(f"{label} handoff missed risk-gated tool count: {handoff}")
    commands = handoff.get("recommended_next_commands") or []
    if handoff.get("recommended_next_command_count") != len(commands):
        raise SystemExit(f"{label} handoff missed recommended-command count: {handoff}")
    for field in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "writes_files",
        "writes_database",
        "writes_notes",
        "writes_memory",
        "executes_side_effect",
        "external_side_effect",
        "controls_computer",
        "calls_external_service",
        "calls_external_services",
        "requires_approval",
        "speaks",
        "completes_tasks",
    ]:
        if handoff.get(field) is not False:
            raise SystemExit(f"{label} handoff unsafe {field}: {handoff}")
    text = str(handoff)
    if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"{label} handoff leaked local path text: {handoff}")


def assert_verification_packet_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("verification_packet_handoff") or {}
    if handoff.get("source") != "verification_packet":
        raise SystemExit(f"{label} missed verification packet handoff source: {handoff}")
    for field in [
        "planner_goal",
        "verdict",
        "next_step",
        "next_command",
        "matched_risks",
        "planned_actions",
        "approval_required",
        "unknown_tools",
        "approval_queue_forecast",
        "forecast_new_approvals",
        "forecast_reused_approval_ids",
        "forecast_queue_before",
        "forecast_queue_after_if_sent",
        "forecast_queue_delta_if_sent",
    ]:
        if handoff.get(field) != metadata.get(field):
            raise SystemExit(f"{label} handoff diverged for {field}: {handoff} vs {metadata}")
    if handoff.get("matched_risk_count") != len(metadata.get("matched_risks") or []):
        raise SystemExit(f"{label} handoff missed risk count: {handoff}")
    if handoff.get("pending_approval_count") != metadata.get("pending_approvals"):
        raise SystemExit(f"{label} handoff missed pending approvals: {handoff}")
    if handoff.get("planned_action_count") != metadata.get("planned_action_count"):
        raise SystemExit(f"{label} handoff missed planned action count: {handoff}")
    if handoff.get("evidence_requirement_count") != metadata.get("evidence_requirements"):
        raise SystemExit(f"{label} handoff missed evidence requirement count: {handoff}")
    if handoff.get("failure_signal_count") != metadata.get("failure_signals"):
        raise SystemExit(f"{label} handoff missed failure signal count: {handoff}")
    if handoff.get("recovery_step_count") != metadata.get("recovery_steps"):
        raise SystemExit(f"{label} handoff missed recovery step count: {handoff}")
    if handoff.get("risk_gated_tool_count") != metadata.get("risk_gated_tools"):
        raise SystemExit(f"{label} handoff missed risk-gated tool count: {handoff}")
    commands = handoff.get("suggested_commands") or []
    if handoff.get("suggested_command_count") != len(commands):
        raise SystemExit(f"{label} handoff missed suggested-command count: {handoff}")
    for field in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "writes_files",
        "writes_database",
        "writes_notes",
        "writes_memory",
        "executes_side_effect",
        "external_side_effect",
        "controls_computer",
        "calls_external_service",
        "calls_external_services",
        "requires_approval",
        "speaks",
        "completes_tasks",
    ]:
        if handoff.get(field) is not False:
            raise SystemExit(f"{label} handoff unsafe {field}: {handoff}")
    text = str(handoff)
    if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"{label} handoff leaked local path text: {handoff}")


def assert_execution_acceptance_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_acceptance_handoff") or {}
    if handoff.get("source") != "execution_acceptance_gate":
        raise SystemExit(f"{label} missed execution acceptance handoff source: {handoff}")
    for field in [
        "planner_goal",
        "verdict",
        "next_step",
        "next_command",
        "matched_risks",
        "approval_required",
        "has_recent_success",
        "has_evidence",
        "has_tests",
        "has_recovery",
        "unknown_tools",
        "approval_queue_forecast",
        "forecast_new_approvals",
        "forecast_reused_approval_ids",
        "forecast_queue_before",
        "forecast_queue_after_if_sent",
        "forecast_queue_delta_if_sent",
    ]:
        if handoff.get(field) != metadata.get(field):
            raise SystemExit(f"{label} handoff diverged for {field}: {handoff} vs {metadata}")
    if handoff.get("matched_risk_count") != len(metadata.get("matched_risks") or []):
        raise SystemExit(f"{label} handoff missed risk count: {handoff}")
    if handoff.get("planned_action_count") != metadata.get("planned_actions"):
        raise SystemExit(f"{label} handoff missed planned action count: {handoff}")
    if handoff.get("pending_approval_count") != metadata.get("pending_approvals"):
        raise SystemExit(f"{label} handoff missed pending approvals: {handoff}")
    if handoff.get("recent_run_count") != metadata.get("recent_runs"):
        raise SystemExit(f"{label} handoff missed recent run count: {handoff}")
    if handoff.get("blocking_reasons") != metadata.get("blocking_reason_names"):
        raise SystemExit(f"{label} handoff missed blocking reason names: {handoff}")
    if handoff.get("blocking_reason_count") != metadata.get("blocking_reasons"):
        raise SystemExit(f"{label} handoff missed blocking reason count: {handoff}")
    if handoff.get("risk_gated_tool_count") != metadata.get("risk_gated_tools"):
        raise SystemExit(f"{label} handoff missed risk-gated tool count: {handoff}")
    for field in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "writes_files",
        "writes_database",
        "writes_notes",
        "writes_memory",
        "executes_side_effect",
        "external_side_effect",
        "controls_computer",
        "calls_external_service",
        "calls_external_services",
        "requires_approval",
        "speaks",
        "completes_tasks",
    ]:
        if handoff.get(field) is not False:
            raise SystemExit(f"{label} handoff unsafe {field}: {handoff}")
    text = str(handoff)
    if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"{label} handoff leaked local path text: {handoff}")


def assert_execution_readiness_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_readiness_handoff") or {}
    if handoff.get("source") != "execution_readiness_matrix":
        raise SystemExit(f"{label} missed execution readiness handoff source: {handoff}")
    for field in [
        "planner_goal",
        "verdict",
        "next_command",
        "matched_risks",
        "planned_actions",
        "approval_required",
        "unknown_tools",
        "recovery_closure_state",
        "recovery_closure_ready_to_retry",
        "recovery_closure_missing",
        "recovery_closure_missing_count",
        "recovery_closure_required_commands",
        "recovery_closure_next_required_command",
        "recovery_closure_blocks_auto_execution",
        "recovery_debt_visible",
        "readiness_matrix_preview_allowed_with_recovery_debt",
        "recovery_closure_blocks_matrix_execution",
        "approval_held_review_required",
        "approval_held_review_commands",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "recovery_closure_target_run_id",
        "recovery_closure_target_tool_name",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "approval_queue_forecast",
        "forecast_new_approvals",
        "forecast_reused_approval_ids",
        "forecast_queue_before",
        "forecast_queue_after_if_sent",
        "forecast_queue_delta_if_sent",
    ]:
        if handoff.get(field) != metadata.get(field):
            raise SystemExit(f"{label} handoff diverged for {field}: {handoff} vs {metadata}")
    if handoff.get("matched_risk_count") != len(metadata.get("matched_risks") or []):
        raise SystemExit(f"{label} handoff missed risk count: {handoff}")
    if handoff.get("pending_approval_count") != metadata.get("pending_approvals"):
        raise SystemExit(f"{label} handoff missed pending approvals: {handoff}")
    if handoff.get("planned_action_count") != metadata.get("planned_action_count"):
        raise SystemExit(f"{label} handoff missed planned action count: {handoff}")
    if handoff.get("matrix_row_count") != metadata.get("matrix_rows"):
        raise SystemExit(f"{label} handoff missed matrix row count: {handoff}")
    if handoff.get("proof_requirement_count") != metadata.get("proof_requirements"):
        raise SystemExit(f"{label} handoff missed proof requirement count: {handoff}")
    if handoff.get("failed_or_blocked_run_count") != metadata.get("failed_or_blocked_runs"):
        raise SystemExit(f"{label} handoff missed failed/blocked run count: {handoff}")
    if handoff.get("recent_failed_runs") != metadata.get("recent_failed_runs"):
        raise SystemExit(f"{label} handoff missed recent failed run count: {handoff}")
    if handoff.get("recent_approval_held_runs") != metadata.get("recent_approval_held_runs"):
        raise SystemExit(f"{label} handoff missed recent approval-held run count: {handoff}")
    if handoff.get("verification_run_count") != metadata.get("verification_runs"):
        raise SystemExit(f"{label} handoff missed verification run count: {handoff}")
    if handoff.get("recovery_closure_proof_queue_count") != metadata.get("recovery_closure_proof_queue_count"):
        raise SystemExit(f"{label} handoff missed recovery proof queue count: {handoff}")
    if handoff.get("recovery_closure_proof_queue") != metadata.get("recovery_closure_proof_queue"):
        raise SystemExit(f"{label} handoff missed recovery proof queue: {handoff}")
    if handoff.get("recovery_closure_next_proof_command") != metadata.get("recovery_closure_next_proof_command"):
        raise SystemExit(f"{label} handoff missed recovery next proof command: {handoff}")
    if handoff.get("execution_learning_proof_queue_count") != metadata.get("execution_learning_proof_queue_count"):
        raise SystemExit(f"{label} handoff missed learning proof queue count: {handoff}")
    if handoff.get("execution_learning_proof_queue") != metadata.get("execution_learning_proof_queue"):
        raise SystemExit(f"{label} handoff missed learning proof queue: {handoff}")
    if handoff.get("execution_learning_next_proof_command") != metadata.get("execution_learning_next_proof_command"):
        raise SystemExit(f"{label} handoff missed learning next proof command: {handoff}")
    if handoff.get("risk_gated_tool_count") != metadata.get("risk_gated_tools"):
        raise SystemExit(f"{label} handoff missed risk-gated tool count: {handoff}")
    for field in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "writes_files",
        "writes_database",
        "writes_notes",
        "writes_memory",
        "executes_side_effect",
        "external_side_effect",
        "controls_computer",
        "calls_external_service",
        "calls_external_services",
        "requires_approval",
        "speaks",
        "completes_tasks",
    ]:
        if handoff.get(field) is not False:
            raise SystemExit(f"{label} handoff unsafe {field}: {handoff}")
    text = str(handoff)
    if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"{label} handoff leaked local path text: {handoff}")


def assert_frontdoor_packets_separate_approval_held_recent_runs() -> None:
    leak_marker = "AUTONOMY_METADATA_SHOULD_NOT_LEAK /\x55sers/example/private/autonomy.sqlite"
    with TemporaryDirectory(prefix="jarvis-autonomy-approval-held-") as temp:
        runtime = make_temp_runtime(Path(temp))
        recent_rows = [
            {
                "id": 10,
                "tool_name": "send_telegram",
                "risk": "HIGH_RISK",
                "ok": False,
                "approved": False,
                "approval_id": 4,
                "output": "blocked before execution; queued as approval #4",
                "metadata": {
                    "failure_kind": HostileMetadataValue(leak_marker),
                    "failure_stage": "explicit approval required",
                },
            },
            {
                "id": 9,
                "tool_name": "send_telegram",
                "risk": "HIGH_RISK",
                "ok": False,
                "approved": True,
                "approval_id": 3,
                "output": "transport timed out",
                "metadata": "{}",
            },
            {
                "id": 8,
                "tool_name": "safety_status",
                "risk": "READ_ONLY",
                "ok": True,
                "approved": False,
                "approval_id": None,
                "output": "safe",
                "metadata": "{}",
            },
        ]
        runtime.store.recent_tool_runs = lambda limit=20: recent_rows[:limit]
        runtime.store.list_pending_approvals = lambda limit=20: []

        packets = [
            ("dispatch_decision_packet", "dispatch_decision", "recent approval-held runs inspected: 1"),
            ("command_intake_packet", "command_intake", "recent approval-held runs inspected: 1"),
            ("execution_governor_packet", "execution_governor", "recent approval-held runs visible: 1"),
        ]
        for tool_name, handoff_key, approval_line in packets:
            result = runtime.registry.get(tool_name).handler({"request": "what time is it"})
            if not result.ok:
                raise SystemExit(f"{tool_name} approval-held fixture should render read-only packet: {result.output}")
            metadata = result.metadata
            if metadata.get("recent_failed_runs") != 1:
                raise SystemExit(f"{tool_name} should keep true failure count at 1: {metadata}")
            if metadata.get("recent_approval_held_runs") != 1:
                raise SystemExit(f"{tool_name} should expose approval-held count at 1: {metadata}")
            if approval_line not in result.output:
                raise SystemExit(f"{tool_name} output missed approval-held line: {result.output}")
            if "recent failed/blocked runs" not in result.output:
                raise SystemExit(f"{tool_name} output missed true-failure line: {result.output}")
            rendered = f"{result.output}\n{result.metadata}"
            for marker in (leak_marker, "autonomy.sqlite", "/\x55sers/example/private"):
                if marker in rendered:
                    raise SystemExit(f"{tool_name} leaked hostile metadata marker {marker}: {rendered}")
            assert_frontdoor_handoff(metadata, f"{tool_name} approval-held fixture", handoff_key)
            handoff = metadata.get(f"{handoff_key}_handoff") or {}
            if handoff.get("recent_failed_runs") != 1 or handoff.get("recent_approval_held_runs") != 1:
                raise SystemExit(f"{tool_name} handoff missed recent-run bucket parity: {handoff}")

        held_only_rows = [recent_rows[0], recent_rows[2]]
        runtime.store.recent_tool_runs = lambda limit=20: held_only_rows[:limit]
        held_only_packets = [
            ("dispatch_decision_packet", "dispatch_decision", "decision", "APPROVAL_HELD_REVIEW_REQUIRED", "approval_held_review"),
            ("command_intake_packet", "command_intake", "intake_state", "approval_held_review_required", "approval_held_review"),
            ("execution_governor_packet", "execution_governor", "governor_verdict", "APPROVAL_HELD_REVIEW_REQUIRED", "approval_held_review"),
            ("execution_readiness_matrix", None, "verdict", "APPROVAL_HELD_REVIEW_REQUIRED", None),
            ("action_readiness_packet", None, "recommendation", "APPROVAL_HELD_REVIEW_REQUIRED", "approval_held_review"),
            ("command_cockpit_packet", "command_cockpit", "cockpit_verdict", "APPROVAL_HELD_REVIEW_REQUIRED", None),
        ]
        for tool_name, handoff_key, verdict_key, expected_verdict, expected_route in held_only_packets:
            result = runtime.registry.get(tool_name).handler({"request": "what time is it"})
            if not result.ok:
                raise SystemExit(f"{tool_name} approval-held-only fixture should render read-only packet: {result.output}")
            metadata = result.metadata
            if metadata.get(verdict_key) != expected_verdict:
                raise SystemExit(f"{tool_name} should route approval-held-only state to approval review: {metadata}")
            if expected_route is not None and metadata.get("route") != expected_route:
                raise SystemExit(f"{tool_name} approval-held-only route should be {expected_route!r}: {metadata}")
            if metadata.get("safe_to_execute_now", metadata.get("can_auto_run", False)) is not False:
                raise SystemExit(f"{tool_name} approval-held-only packet should not auto-execute: {metadata}")
            if metadata.get("approval_held_review_required") is not True:
                raise SystemExit(f"{tool_name} missed approval-held review metadata: {metadata}")
            if "approval readiness 4" not in str(metadata.get("approval_held_review_commands") or metadata.get("recovery_closure_required_commands") or metadata.get("required_commands") or []):
                raise SystemExit(f"{tool_name} missed approval review commands: {metadata}")
            if "Approval-held execution review:" not in result.output:
                raise SystemExit(f"{tool_name} should render approval-held review section, not generic recovery closure: {result.output}")
            if "Execution health recovery closure:" in result.output or "RECOVERY_CLOSURE_REQUIRED" in result.output:
                raise SystemExit(f"{tool_name} should not label approval-held-only state as recovery closure: {result.output}")
            if "execution recovery packet 10" in result.output:
                raise SystemExit(f"{tool_name} should not recover approval-held run as a failed execution: {result.output}")
            rendered = f"{result.output}\n{metadata}"
            for marker in (leak_marker, "autonomy.sqlite", "/\x55sers/example/private"):
                if marker in rendered:
                    raise SystemExit(f"{tool_name} leaked hostile metadata marker {marker}: {rendered}")
            if handoff_key is not None:
                assert_frontdoor_handoff(metadata, f"{tool_name} approval-held-only fixture", handoff_key)
                handoff = metadata.get(f"{handoff_key}_handoff") or {}
                if handoff.get("approval_held_review_required") is not True:
                    raise SystemExit(f"{tool_name} handoff missed approval-held review flag: {handoff}")
                if handoff.get("approval_held_review_commands") != metadata.get("approval_held_review_commands"):
                    raise SystemExit(f"{tool_name} handoff missed approval-held review commands: {handoff} vs {metadata}")


def assert_execution_recovery_closure_routes_approval_held_runs_to_review() -> None:
    leak_marker = "AUTONOMY_SNAPSHOT_METADATA_SHOULD_NOT_LEAK /\x55sers/example/private/autonomy.sqlite"
    approval_held = {
        "id": 10,
        "tool_name": "send_telegram",
        "risk": "HIGH_RISK",
        "ok": False,
        "approved": False,
        "approval_id": 4,
        "output": "blocked before execution; queued as approval #4",
        "metadata": {
            "failure_kind": HostileMetadataValue(leak_marker),
            "failure_stage": "explicit approval required",
        },
    }
    true_failure = {
        "id": 9,
        "tool_name": "send_email",
        "risk": "EXTERNAL_SIDE_EFFECT",
        "ok": False,
        "approved": True,
        "approval_id": 2,
        "output": "SMTP failed",
        "metadata": '{"failure_kind":"transport-error"}',
    }

    held_snapshot = _execution_health_recovery_closure_snapshot([approval_held])
    if held_snapshot["state"] != "approval_held_review_required":
        raise SystemExit(f"approval-held-only closure should route to approval review: {held_snapshot}")
    if held_snapshot["blocks_auto_execution"] is not True or held_snapshot["ready_to_retry"] is not False:
        raise SystemExit(f"approval-held review should block auto execution without retry readiness: {held_snapshot}")
    if held_snapshot["failed_or_blocked_action_runs"] != 0 or held_snapshot["approval_held_action_runs"] != 1:
        raise SystemExit(f"approval-held closure should not count as true failure: {held_snapshot}")
    for command in [
        "approval readiness 4",
        "approval packet 4",
        "approval chain proof 4",
        "verification receipt <approved run id from approval chain proof 4>",
    ]:
        if command not in held_snapshot["required_commands"]:
            raise SystemExit(f"approval-held closure missed approval command {command!r}: {held_snapshot}")
    if f"execution recovery packet {approval_held['id']}" in held_snapshot["required_commands"]:
        raise SystemExit(f"approval-held closure should not ask for execution recovery: {held_snapshot}")
    if held_snapshot["approval_held_target_run_id"] != approval_held["id"]:
        raise SystemExit(f"approval-held target should be preserved for review: {held_snapshot}")
    rendered_snapshot = str(held_snapshot)
    for marker in (leak_marker, "autonomy.sqlite", "/\x55sers/example/private"):
        if marker in rendered_snapshot:
            raise SystemExit(f"approval-held closure leaked hostile metadata marker {marker}: {held_snapshot}")

    mixed_snapshot = _execution_health_recovery_closure_snapshot([approval_held, true_failure])
    if mixed_snapshot["target_run_id"] != true_failure["id"]:
        raise SystemExit(f"true failure should remain the recovery target in mixed history: {mixed_snapshot}")
    if mixed_snapshot["failed_or_blocked_action_runs"] != 1 or mixed_snapshot["approval_held_action_runs"] != 1:
        raise SystemExit(f"mixed closure should count true failures and approval holds separately: {mixed_snapshot}")
    if f"execution recovery packet {true_failure['id']}" not in mixed_snapshot["required_commands"]:
        raise SystemExit(f"mixed closure should keep recovery packet for true failure: {mixed_snapshot}")
    if f"execution recovery packet {approval_held['id']}" in mixed_snapshot["required_commands"]:
        raise SystemExit(f"mixed closure should not recover approval-held run: {mixed_snapshot}")


def main() -> None:
    test_planner_routes_autonomy_plan_aliases()
    test_autonomy_tools_accept_request_arg_aliases()
    for malformed_count in ("not-a-number", True, float("inf"), None):
        if _metadata_int(malformed_count, 7) != 7:
            raise SystemExit(f"Autonomy metadata counters should default malformed values safely: {malformed_count!r}")
    if _metadata_int("4") != 4:
        raise SystemExit("Autonomy metadata counters should preserve valid numeric strings.")
    assert_autonomy_exact_metadata_bool_contract()
    assert_frontdoor_packets_separate_approval_held_recent_runs()
    assert_execution_recovery_closure_routes_approval_held_runs_to_review()

    with TemporaryDirectory(prefix="jarvis-autonomy-plan-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for tool_name, handoff_key in [
            ("command_intake_packet", "command_intake"),
            ("dispatch_decision_packet", "dispatch_decision"),
            ("execution_governor_packet", "execution_governor"),
            ("command_cockpit_packet", "command_cockpit"),
            ("planner_gap_packet", "planner_gap"),
        ]:
            missing = runtime.registry.get(tool_name).handler({})
            print(f"[ok] direct missing-request {tool_name}")
            print(missing.output[:700])
            print()
            if missing.ok or missing.metadata.get("reason") != "missing_request":
                raise SystemExit(f"{tool_name} should refuse missing requests with reason metadata: {missing.metadata}")
            assert_frontdoor_handoff(missing.metadata, f"Missing-request {tool_name}", handoff_key, status="refused")

        cases = [
            "autonomy plan: use my computer to inspect the screen, run a python script, write a summary file, and tell me what changed",
            "risk preflight: use my computer to inspect the screen, run a python script, write a summary file, and email me the result",
            "risky request lifecycle: use my computer to inspect the screen, run a python script, write a summary file, and email me the result",
            "action readiness: use my computer to inspect the screen, run a python script, write a summary file, and email me the result",
            "action readiness: what time is it",
            "execution contract: use my computer to inspect the screen, run a python script, write a summary file, and email me the result",
            "verification packet: use my computer to inspect the screen, run a python script, write a summary file, and email me the result",
            "command intake: use my computer to inspect the screen, run a python script, write a summary file, and email me the result",
            "command intake: what time is it",
            "execution governor: use my computer to inspect the screen, run a python script, write a summary file, and email me the result",
            "execution governor: what time is it",
            "command cockpit",
            "cockpit packet",
            "command cockpit: use my computer to inspect the screen, run a python script, write a summary file, and email me the result",
            "command cockpit: what time is it",
            "preflight: summarize visible tasks",
            "second loop: use my computer to inspect the screen, run a python script, write a summary file, and email me the result",
            "second loop packet: use my computer to inspect the screen, run a python script, write a summary file, and email me the result",
            "help safety",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:2200])
            print()
            if case.startswith("autonomy plan"):
                required = [
                    "Jarvis autonomy plan",
                    "Safe first steps",
                    "Approval-gated steps",
                    "computer: require explicit approval",
                    "shell/code: require explicit approval",
                    "files: require explicit approval",
                    "Verification",
                    "run_shell_command",
                    "observe_screen",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Autonomy plan missing expected safety context: {missing}")
                assert_safe_metadata(result.tool_results[0].metadata, "Autonomy plan")
            if case.startswith("risk preflight"):
                required = [
                    "Jarvis risk preflight",
                    "Risk signal",
                    "computer: likely approval-gated",
                    "shell/code: likely approval-gated",
                    "files: likely approval-gated",
                    "external side effect: likely approval-gated",
                    "Safe preview path",
                    "This preflight is read-only",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Risk preflight missing expected safety context: {missing}")
                metadata = result.tool_results[0].metadata
                assert_safe_metadata(metadata, "Risk preflight")
                assert_risk_preflight_handoff(metadata, "Risk preflight")
                if not metadata.get("likely_approval_required") or metadata.get("queues_approval"):
                    raise SystemExit("Risk preflight metadata missed approval/read-only flags.")
            if case.startswith("risky request lifecycle"):
                required = [
                    "Jarvis risky request lifecycle",
                    "read-only",
                    "Detected risk areas",
                    "computer",
                    "shell/code",
                    "files",
                    "external side effect",
                    "Lifecycle",
                    "risk preflight",
                    "action rehearsal",
                    "Safety receipt",
                    "approval packet #ID",
                    "recent tool runs",
                    "Stop conditions",
                    "execution governor",
                    "does not call a model",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Risky lifecycle missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                assert_safe_metadata(metadata, "Risky lifecycle")
                if not str(metadata.get("next_command", "")).startswith("execution governor: "):
                    raise SystemExit(f"Risky lifecycle should start with the execution governor: {metadata}")
            if case.startswith("action readiness"):
                required = [
                    "Jarvis action readiness packet",
                    "Recommended preview path",
                    "execution governor",
                    "Go/no-go rule",
                    "does not call a model",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Action readiness missing expected governor-first context: {missing}")
                metadata = result.tool_results[0].metadata
                assert_safe_metadata(metadata, "Action readiness")
                if not str(metadata.get("next_command", "")).startswith("execution governor: "):
                    raise SystemExit(f"Action readiness should route first through execution governor: {metadata}")
                commands = metadata.get("recommended_next_commands", [])
                if not commands or not str(commands[0]).startswith("execution governor: "):
                    raise SystemExit(f"Action readiness recommended commands should begin with execution governor: {metadata}")
                if "use my computer" in case:
                    if metadata.get("approval_required") is not True or metadata.get("safe_to_execute_now") is not False:
                        raise SystemExit(f"Risky action readiness missed approval/safety metadata: {metadata}")
                else:
                    if metadata.get("approval_required") is not False or metadata.get("safe_to_execute_now") is not True:
                        raise SystemExit(f"Low-risk action readiness missed safe metadata: {metadata}")
            if case.startswith("execution contract"):
                required = [
                    "Jarvis execution contract",
                    "steering-wheel contract",
                    "Route:",
                    "Approval state:",
                    "Harness stages",
                    "Perceive",
                    "Ground",
                    "Route",
                    "Plan",
                    "Gate",
                    "Act",
                    "Verify",
                    "Learn",
                    "Planned tool route",
                    "Risk signals",
                    "Verification targets",
                    "Recovery and stop rules",
                    "Learning hooks",
                    "does not call a model",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Execution contract missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                assert_safe_metadata(metadata, "Execution contract")
                assert_execution_contract_handoff(metadata, "Execution contract")
                if not metadata.get("approval_required") or metadata.get("route") not in {"approval_gated", "hold_for_approval_review"}:
                    raise SystemExit(f"Execution contract should detect risky approval-gated route: {metadata}")
                if metadata.get("verification_targets", 0) < 2:
                    raise SystemExit(f"Execution contract missed verification metadata: {metadata}")
            if case.startswith("verification packet"):
                required = [
                    "Jarvis verification packet",
                    "dashboard proof plan",
                    "Verification verdict:",
                    "Planned route to verify",
                    "Evidence requirements",
                    "Failure signals",
                    "Recovery if verification fails",
                    "shell/code evidence",
                    "computer-control evidence",
                    "file evidence",
                    "external-effect evidence",
                    "does not call a model",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Verification packet missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                assert_safe_metadata(metadata, "Verification packet")
                if not metadata.get("approval_required") or metadata.get("verdict") not in {"VERIFY_AFTER_APPROVAL", "VERIFY_AFTER_APPROVAL_REVIEW"}:
                    raise SystemExit(f"Verification packet should detect risky verification route: {metadata}")
                if metadata.get("verdict") == "VERIFY_AFTER_APPROVAL" and not str(metadata.get("next_command", "")).startswith("execution governor: "):
                    raise SystemExit(f"Verification packet should route risky no-pending work through the execution governor: {metadata}")
                if metadata.get("evidence_requirements", 0) < 4 or metadata.get("failure_signals", 0) < 3:
                    raise SystemExit(f"Verification packet missed evidence/failure metadata: {metadata}")
            if case.startswith("command intake"):
                required = [
                    "Jarvis command intake packet",
                    "front-door harness packet",
                    "Intake state:",
                    "Route:",
                    "Risk and approval",
                    "Planned actions",
                    "Proof requirements",
                    "Recommended follow-up packets",
                    "does not call a model",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Command intake packet missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                assert_safe_metadata(metadata, "Command intake packet")
                if "use my computer" in case:
                    if metadata.get("approval_required") is not True or metadata.get("safe_to_execute_now") is not False:
                        raise SystemExit(f"Risky command intake missed approval/safety metadata: {metadata}")
                    if "computer" not in metadata.get("matched_risks", []) or "shell/code" not in metadata.get("matched_risks", []):
                        raise SystemExit(f"Risky command intake missed risk metadata: {metadata}")
                    if metadata.get("intake_state") not in {"needs_exact_route", "approval_required", "blocked_by_existing_approval"}:
                        raise SystemExit(f"Risky command intake should not be treated as ready chat: {metadata}")
                    if metadata.get("intake_state") == "needs_exact_route" and not str(metadata.get("next_command", "")).startswith("execution governor: "):
                        raise SystemExit(f"Risky ambiguous command intake should route through the execution governor first: {metadata}")
                else:
                    if metadata.get("route") not in {"auto_safe_tool_route", "chat_brain"}:
                        raise SystemExit(f"Low-risk command intake should route to safe tools or chat: {metadata}")
                    if metadata.get("approval_required") is not False or metadata.get("safe_to_execute_now") is not True:
                        raise SystemExit(f"Low-risk command intake missed safe metadata: {metadata}")
            if case.startswith("execution governor"):
                required = [
                    "Jarvis execution governor packet",
                    "command-first harness governor",
                    "Governor verdict:",
                    "Can auto-run now:",
                    "Safe to execute now:",
                    "Governor gates",
                    "Required proof before completion",
                    "Stop conditions",
                    "Required follow-up packets",
                    "does not call a model",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Execution governor packet missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                assert_safe_metadata(metadata, "Execution governor packet")
                if "use my computer" in case:
                    if metadata.get("approval_required") is not True or metadata.get("safe_to_execute_now") is not False or metadata.get("can_auto_run") is not False:
                        raise SystemExit(f"Risky execution governor missed approval/safety metadata: {metadata}")
                    if metadata.get("governor_verdict") not in {"APPROVAL_REQUIRED", "HOLD_FOR_APPROVAL_REVIEW", "RECOVERY_REVIEW_FIRST", "PLANNER_GAP"}:
                        raise SystemExit(f"Risky execution governor should not auto-run: {metadata}")
                    if "computer" not in metadata.get("matched_risks", []) or "shell/code" not in metadata.get("matched_risks", []):
                        raise SystemExit(f"Risky execution governor missed risk metadata: {metadata}")
                else:
                    if metadata.get("governor_verdict") == "APPROVAL_HELD_REVIEW_REQUIRED":
                        if metadata.get("route") != "approval_held_review" or metadata.get("safe_to_execute_now") is not False or metadata.get("can_auto_run") is not False:
                            raise SystemExit(f"Low-risk execution governor missed approval-held review metadata: {metadata}")
                        if metadata.get("recovery_closure_blocks_auto_execution") is not True or metadata.get("approval_held_review_required") is not True:
                            raise SystemExit(f"Low-risk execution governor missed approval-held blocker metadata: {metadata}")
                        commands = metadata.get("recovery_closure_required_commands", [])
                        if not commands or not str(commands[0]).startswith("approval readiness "):
                            raise SystemExit(f"Low-risk execution governor missed approval-held command queue: {metadata}")
                        if "approval-held review state:" not in result.response or "approval-held review command queue:" not in result.response:
                            raise SystemExit("Low-risk execution governor did not render approval-held review state.")
                    elif metadata.get("governor_verdict") == "RECOVERY_CLOSURE_REQUIRED":
                        if metadata.get("route") != "recovery_closure" or metadata.get("safe_to_execute_now") is not False or metadata.get("can_auto_run") is not False:
                            raise SystemExit(f"Low-risk execution governor missed recovery-closure hold metadata: {metadata}")
                        if metadata.get("recovery_closure_blocks_auto_execution") is not True:
                            raise SystemExit(f"Low-risk execution governor missed recovery-closure blocker metadata: {metadata}")
                        commands = metadata.get("recovery_closure_required_commands", [])
                        if not commands or not str(commands[0]).startswith("verification receipt "):
                            raise SystemExit(f"Low-risk execution governor missed recovery-closure command queue: {metadata}")
                        if "recovery closure state:" not in result.response or "recovery closure command queue:" not in result.response:
                            raise SystemExit("Low-risk execution governor did not render recovery closure state.")
                    elif metadata.get("governor_verdict") not in {"AUTO_SAFE_READY", "CHAT_READY"}:
                        raise SystemExit(f"Low-risk execution governor should route to safe tools or chat: {metadata}")
                    elif metadata.get("approval_required") is not False or metadata.get("safe_to_execute_now") is not True:
                        raise SystemExit(f"Low-risk execution governor missed safe metadata: {metadata}")
            if case.startswith("command cockpit"):
                required = [
                    "Jarvis command cockpit packet",
                    "command-first dashboard",
                    "Cockpit verdict:",
                    "Cockpit instruments",
                    "intake:",
                    "governor:",
                    "dispatch:",
                    "readiness matrix:",
                    "verification:",
                    "Required proof queue",
                    "execution learning blocks completion claim",
                    "does not call a model",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Command cockpit packet missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                assert_safe_metadata(metadata, "Command cockpit packet")
                if case in {"command cockpit", "cockpit packet"} and metadata.get("request") != "what should Jarvis do next":
                    raise SystemExit(f"Bare command cockpit should use the default safe objective: {metadata}")
                if metadata.get("packet_row_count") != 5 or len(metadata.get("packet_rows", [])) != 5:
                    raise SystemExit(f"Command cockpit missed composed packet rows: {metadata}")
                if metadata.get("proof_queue") != metadata.get("required_commands"):
                    raise SystemExit(f"Command cockpit proof queue should mirror required commands: {metadata}")
                if metadata.get("proof_queue_count") != len(metadata.get("proof_queue", [])):
                    raise SystemExit(f"Command cockpit proof queue count diverged: {metadata}")
                for expected in ["command intake:", "execution governor:", "dispatch decision:", "execution readiness matrix:", "verification packet:"]:
                    if not any(str(command).startswith(expected) for command in metadata.get("required_commands", [])):
                        raise SystemExit(f"Command cockpit missed required command prefix {expected!r}: {metadata}")
                if "use my computer" in case:
                    if metadata.get("approval_required") is not True or metadata.get("can_auto_run") is not False:
                        raise SystemExit(f"Risky command cockpit missed approval/safety metadata: {metadata}")
                    if metadata.get("cockpit_verdict") not in {"APPROVAL_REQUIRED_BEFORE_EXECUTION", "HOLD_FOR_APPROVAL_REVIEW", "APPROVAL_HELD_REVIEW_REQUIRED", "RECOVERY_CLOSURE_REQUIRED", "PREFLIGHT_OR_CLARIFICATION_REQUIRED"}:
                        raise SystemExit(f"Risky command cockpit should not be ready for safe dispatch: {metadata}")
                else:
                    if metadata.get("cockpit_verdict") == "APPROVAL_HELD_REVIEW_REQUIRED":
                        if metadata.get("recovery_closure_blocks_auto_execution") is not True or metadata.get("can_auto_run") is not False:
                            raise SystemExit(f"Low-risk command cockpit missed approval-held hold metadata: {metadata}")
                        if metadata.get("approval_held_review_required") is not True:
                            raise SystemExit(f"Low-risk approval-held command cockpit should expose review metadata: {metadata}")
                        if metadata.get("recovery_debt_visible") is not True:
                            raise SystemExit(f"Low-risk approval-held command cockpit should expose visible review debt: {metadata}")
                        if metadata.get("cockpit_preview_allowed_with_recovery_debt") is not True:
                            raise SystemExit(f"Low-risk approval-held command cockpit should allow the read-only cockpit preview while review debt is visible: {metadata}")
                        if metadata.get("recovery_closure_blocks_cockpit_execution") is not True:
                            raise SystemExit(f"Low-risk approval-held command cockpit should mark cockpit execution blocked by approval review: {metadata}")
                        if metadata.get("recovery_closure_next_required_command") != metadata.get("recovery_closure_next_proof_command"):
                            raise SystemExit(f"Low-risk approval-held command cockpit missed required/proof alias parity: {metadata}")
                        if "next closure proof:" in result.response:
                            raise SystemExit("Low-risk approval-held command cockpit should render next closure required, not next closure proof.")
                        if "next closure required:" not in result.response:
                            raise SystemExit("Low-risk approval-held command cockpit missed next closure required output.")
                        if "approval review blocks cockpit execution: yes" not in result.response or "approval review blocks this cockpit preview: no" not in result.response:
                            raise SystemExit("Low-risk approval-held command cockpit did not render cockpit-execution vs read-only-cockpit review semantics.")
                    elif metadata.get("cockpit_verdict") == "RECOVERY_CLOSURE_REQUIRED":
                        if metadata.get("recovery_closure_blocks_auto_execution") is not True or metadata.get("can_auto_run") is not False:
                            raise SystemExit(f"Low-risk command cockpit missed recovery hold metadata: {metadata}")
                        if metadata.get("recovery_debt_visible") is not True:
                            raise SystemExit(f"Low-risk recovery-held command cockpit should expose visible recovery debt: {metadata}")
                        if metadata.get("cockpit_preview_allowed_with_recovery_debt") is not True:
                            raise SystemExit(f"Low-risk recovery-held command cockpit should allow the read-only cockpit preview while recovery debt is visible: {metadata}")
                        if metadata.get("recovery_closure_blocks_cockpit_execution") is not True:
                            raise SystemExit(f"Low-risk recovery-held command cockpit should mark cockpit execution blocked by recovery debt: {metadata}")
                        if metadata.get("recovery_closure_next_required_command") != metadata.get("recovery_closure_next_proof_command"):
                            raise SystemExit(f"Low-risk recovery-held command cockpit missed required/proof alias parity: {metadata}")
                        if "next closure proof:" in result.response:
                            raise SystemExit("Low-risk recovery-held command cockpit should render next closure required, not next closure proof.")
                        if "next closure required:" not in result.response:
                            raise SystemExit("Low-risk recovery-held command cockpit missed next closure required output.")
                        if "recovery debt blocks cockpit execution: yes" not in result.response or "recovery debt blocks this cockpit preview: no" not in result.response:
                            raise SystemExit("Low-risk recovery-held command cockpit did not render cockpit-execution vs read-only-cockpit recovery semantics.")
                    elif metadata.get("cockpit_verdict") not in {"READY_FOR_SAFE_DISPATCH", "PREFLIGHT_OR_CLARIFICATION_REQUIRED"}:
                        raise SystemExit(f"Low-risk command cockpit should be ready or ask for preflight: {metadata}")
            if case.startswith("preflight:"):
                for expected in ["Jarvis risk preflight", "No obvious risky wording detected", "Possible low-risk path"]:
                    if expected not in result.response:
                        raise SystemExit(f"Low-risk preflight route missing expected context: {expected}")
            if case.startswith("second loop"):
                if case.startswith("second loop packet"):
                    required = [
                        "Jarvis second loop packet",
                        "read-only task packet",
                        "Definition of done",
                        "Loop packet",
                        "Orient",
                        "Define done",
                        "Govern",
                        "Preflight",
                        "Act",
                        "Verify",
                        "Close out",
                        "computer: approval likely required",
                        "shell/code: approval likely required",
                        "files: approval likely required",
                        "external side effect: approval likely required",
                        "Recommended next preview commands",
                        "execution governor",
                        "approval packet #ID",
                        "recent tool runs",
                        "Hard stops",
                        "does not call a model",
                    ]
                    missing = [item for item in required if item not in result.response]
                    if missing:
                        raise SystemExit(f"Second loop packet missing expected context: {missing}")
                    metadata = result.tool_results[0].metadata
                    assert_safe_metadata(metadata, "Second loop packet")
                    if not str(metadata.get("next_command", "")).startswith("execution governor: "):
                        raise SystemExit(f"Second loop packet should start with execution governor: {metadata}")
                    continue
                required = [
                    "Jarvis second loop preview",
                    "task/action loop",
                    "Loop shape",
                    "Load visible context",
                    "Route through execution governor",
                    "Run risk preflight and action rehearsal",
                    "Execute only read-only/local-safe",
                    "Risk gates for this goal",
                    "computer: likely approval-gated",
                    "shell/code: likely approval-gated",
                    "files: likely approval-gated",
                    "external side effect: likely approval-gated",
                    "Stop conditions",
                    "execution governor",
                    "This preview is read-only",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Second loop preview missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                assert_safe_metadata(metadata, "Second loop preview")
                if not str(metadata.get("next_command", "")).startswith("execution governor: "):
                    raise SystemExit(f"Second loop preview should start with execution governor: {metadata}")
            if case == "help safety" and ("autonomy plan" not in result.response or "risk preflight" not in result.response or "risky request lifecycle" not in result.response or "execution contract" not in result.response or "verification packet" not in result.response or "command intake" not in result.response or "second loop" not in result.response or "second loop packet" not in result.response):
                raise SystemExit("Safety help did not include autonomy plan.")

        bare_frontdoor_cases = {
            "autonomy plan": "autonomy_plan",
            "risk preflight": "risk_preflight",
            "action readiness": "action_readiness_packet",
            "execution contract": "execution_contract",
            "argument contract": "argument_contract_packet",
            "verification packet": "verification_packet",
            "execution readiness matrix": "execution_readiness_matrix",
            "dispatch decision": "dispatch_decision_packet",
            "execution governor": "execution_governor_packet",
            "command intake": "command_intake_packet",
            "planner gap": "planner_gap_packet",
            "risky request lifecycle": "risky_request_lifecycle",
            "rehearse": "action_rehearsal",
        }
        for command, expected_tool in bare_frontdoor_cases.items():
            bare_result = runtime.handle(command)
            print(f"[ok={bare_result.verified}] bare {command}")
            print(bare_result.response[:700])
            print()
            if not bare_result.verified:
                raise SystemExit(f"Bare front-door command should route through a read-only default packet: {command!r}")
            if not bare_result.tool_results or bare_result.tool_results[0].tool_name != expected_tool:
                raise SystemExit(f"Bare front-door command routed to the wrong tool: {command!r} -> {bare_result.tool_results}")
            metadata = bare_result.tool_results[0].metadata
            if metadata.get("request") != "what should Jarvis do next" and metadata.get("goal") != "what should Jarvis do next":
                raise SystemExit(f"Bare front-door command missed default safe objective: {command!r} -> {metadata}")
            if expected_tool == "action_rehearsal":
                for field in [
                    "calls_model",
                    "executes_tools",
                    "queues_approval",
                    "approves_request",
                    "reads_personal_data",
                    "external_side_effect",
                    "writes_files",
                    "controls_computer",
                    "requires_approval",
                    "authorizes_execution",
                    "authorizes_completion_claim",
                ]:
                    if metadata.get(field) is not False:
                        raise SystemExit(f"Bare rehearsal should remain read-only/non-authorizing {field}: {metadata}")
            else:
                assert_safe_metadata(metadata, f"Bare {command}")

        polite_operator_cases = {
            "risk preflight please": ("risk_preflight", {"request": "what should Jarvis do next"}),
            "execution governor please": ("execution_governor_packet", {"request": "what should Jarvis do next"}),
            "command cockpit please": ("command_cockpit_packet", {"request": "what should Jarvis do next"}),
            "pending approvals please": ("list_pending_approvals", {}),
            "approval review please": ("review_pending_approvals", {}),
            "safe next actions please": ("safe_next_actions", {}),
            "work queue please": ("work_queue", {}),
            "jarvis doctor please": ("jarvis_doctor", {}),
            "storage status please": ("storage_status", {}),
            "help safety please": ("jarvis_help", {"topic": "safety"}),
            "capability map please": ("capability_map", {"focus": ""}),
            "what should I do now please": ("next_action_packet", {}),
            "send me a priority card please": ("next_action_packet", {}),
            "please pending approvals": ("list_pending_approvals", {}),
            "pls approval review": ("review_pending_approvals", {}),
            "can you safe next actions": ("safe_next_actions", {}),
            "could you work queue": ("work_queue", {}),
            "show me jarvis doctor": ("jarvis_doctor", {}),
            "please show storage status": ("storage_status", {}),
            "please show me help safety": ("jarvis_help", {"topic": "safety"}),
            "could you capability map": ("capability_map", {"focus": ""}),
            "can you send me a priority card": ("next_action_packet", {}),
            "show me risk preflight": ("risk_preflight", {"request": "what should Jarvis do next"}),
            "please execution governor": ("execution_governor_packet", {"request": "what should Jarvis do next"}),
            "please command cockpit": ("command_cockpit_packet", {"request": "what should Jarvis do next"}),
            "can you show me pending approvals": ("list_pending_approvals", {}),
            "could you show me approval review": ("review_pending_approvals", {}),
            "can you please show me safe next actions": ("safe_next_actions", {}),
            "could you please show me work queue": ("work_queue", {}),
            "would you please show jarvis doctor": ("jarvis_doctor", {}),
            "can you please show storage status": ("storage_status", {}),
            "could you please show me help safety": ("jarvis_help", {"topic": "safety"}),
            "would you please show me capability map": ("capability_map", {"focus": ""}),
            "can you please show me risk preflight": ("risk_preflight", {"request": "what should Jarvis do next"}),
            "could you please show me execution governor": ("execution_governor_packet", {"request": "what should Jarvis do next"}),
            "would you please show me command cockpit": ("command_cockpit_packet", {"request": "what should Jarvis do next"}),
            "can you kindly show me pending approvals": ("list_pending_approvals", {}),
            "could you just show me work queue": ("work_queue", {}),
            "would you kindly show jarvis doctor": ("jarvis_doctor", {}),
            "kindly show me help safety": ("jarvis_help", {"topic": "safety"}),
            "just show me capability map": ("capability_map", {"focus": ""}),
            "could you kindly show me execution governor": ("execution_governor_packet", {"request": "what should Jarvis do next"}),
            "harness status please": ("harness_status", {}),
            "show me harness doctrine": ("harness_doctrine", {}),
            "can you show me harness completion": ("harness_completion_assessment", {}),
            "completion audit please": ("completion_audit_packet", {}),
            "show me evidence ledger": ("evidence_ledger", {}),
            "completion claim gate please": ("completion_claim_gate", {}),
            "completion next proof please": ("completion_next_proof_packet", {}),
            "show me harness readiness digest": ("harness_readiness_digest", {}),
            "show me harness operations": ("harness_operations_brief", {"objective": ""}),
            "architecture map please": ("architecture_map", {}),
            "show me roadmap": ("roadmap_report", {}),
            "can you show me brain loop": ("brain_loop_report", {}),
            "show me model routing status": ("model_routing_status", {}),
            "can you show me agi gates": ("agi_gate_report", {}),
            "agi next build move please": ("agi_next_build_move", {"gate": ""}),
            "execution health report please": ("execution_health_report", {}),
            "show me recovery closure checklist": ("recovery_closure_checklist", {}),
            "can you show me after-action learning packet": ("after_action_learning_packet", {}),
            "return brief please": ("return_brief", {}),
            "show me handoff brief": ("handoff_brief", {}),
            "show me build progress": ("build_progress_report", {}),
            "show me build delta": ("build_delta_report", {}),
            "work block checkpoint please": ("work_block_checkpoint", {}),
            "show me focus brief": ("focus_brief", {"objective": ""}),
            "show me work session packet": ("work_session_packet", {"objective": ""}),
            "show me priority stack": ("priority_stack", {}),
            "can you show me continuation packet": ("continuation_packet", {"objective": ""}),
            "show me build target packet": ("build_target_packet", {"objective": ""}),
            "chat continuity brief please": ("chat_continuity_brief", {}),
            "show me chat response health": ("chat_response_health", {}),
            "session learning preview please": ("session_learning_preview", {}),
        }
        for command, (expected_tool, expected_args) in polite_operator_cases.items():
            polite_result = runtime.handle(command)
            print(f"[ok={polite_result.verified}] polite operator {command}")
            print(polite_result.response[:700])
            print()
            if not polite_result.verified:
                raise SystemExit(f"Polite operator command should stay command-first: {command!r}")
            if not polite_result.tool_results or polite_result.tool_results[0].tool_name != expected_tool:
                raise SystemExit(f"Polite operator command routed to the wrong tool: {command!r} -> {polite_result.tool_results}")
            action = polite_result.plan.actions[0]
            for key, expected_value in expected_args.items():
                if action.args.get(key) != expected_value:
                    raise SystemExit(f"Polite operator command missed stripped args: {command!r} -> {action.args}")
            metadata = polite_result.tool_results[0].metadata
            if expected_tool in {"risk_preflight", "execution_governor_packet", "command_cockpit_packet"}:
                assert_safe_metadata(metadata, f"Polite {command}")
                if metadata.get("request") != "what should Jarvis do next":
                    raise SystemExit(f"Polite packet command should use default objective: {command!r} -> {metadata}")

        message_body_guard = runtime.planner.plan("text fixture saying hi please")
        if (
            not message_body_guard.actions
            or message_body_guard.actions[0].tool_name != "send_imessage"
            or message_body_guard.actions[0].args.get("message") != "hi please"
        ):
            details = [(action.tool_name, action.args) for action in message_body_guard.actions]
            raise SystemExit(f"Polite retry should not strip message bodies: {details}")
        leading_message_guard = runtime.planner.plan("please text fixture saying hi please")
        if (
            not leading_message_guard.actions
            or leading_message_guard.actions[0].tool_name != "send_imessage"
            or leading_message_guard.actions[0].args.get("message") != "hi please"
        ):
            details = [(action.tool_name, action.args) for action in leading_message_guard.actions]
            raise SystemExit(f"Leading polite retry should preserve message bodies: {details}")
        stacked_message_guard = runtime.planner.plan("can you please text fixture saying hi please")
        if (
            not stacked_message_guard.actions
            or stacked_message_guard.actions[0].tool_name != "send_imessage"
            or stacked_message_guard.actions[0].args.get("message") != "hi please"
        ):
            details = [(action.tool_name, action.args) for action in stacked_message_guard.actions]
            raise SystemExit(f"Stacked polite retry should preserve message bodies: {details}")
        softener_message_guard = runtime.planner.plan("could you just text fixture saying hi please")
        if (
            not softener_message_guard.actions
            or softener_message_guard.actions[0].tool_name != "send_imessage"
            or softener_message_guard.actions[0].args.get("message") != "hi please"
        ):
            details = [(action.tool_name, action.args) for action in softener_message_guard.actions]
            raise SystemExit(f"Softener polite retry should preserve message bodies: {details}")

        bounded = runtime.registry.get("risk_preflight").handler({"request": "run " + ("python script and email result " * 80)})
        print("[ok] direct bounded risk_preflight")
        print(bounded.output[:900])
        print()
        if not bounded.ok or len(bounded.metadata.get("request", "")) > 600:
            raise SystemExit(f"Risk preflight request was not bounded: {bounded.metadata}")
        assert_safe_metadata(bounded.metadata, "Bounded risk preflight")

        path_autonomy_plan = runtime.registry.get("autonomy_plan").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-autonomy-plan.txt "
                    "/private/tmp/plan-token.txt /var/folders/zc/plan-cache.txt /tmp/plan-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted autonomy_plan")
        print(path_autonomy_plan.output[:900])
        print()
        path_autonomy_text = f"{path_autonomy_plan.output}\n{path_autonomy_plan.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_autonomy_text:
                raise SystemExit(f"Autonomy plan leaked local path fragment {forbidden!r}: {path_autonomy_text}")
        if "<local-path>" not in path_autonomy_text:
            raise SystemExit(f"Autonomy plan should retain a local-path marker after redaction: {path_autonomy_text}")
        if path_autonomy_plan.metadata.get("request") != "run command cat <local-path>" or path_autonomy_plan.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Autonomy plan metadata should expose only the redacted request: {path_autonomy_plan.metadata}")
        if "shell/code" not in path_autonomy_plan.metadata.get("matched_risks", []):
            raise SystemExit(f"Autonomy plan should still classify the original shell request as risky: {path_autonomy_plan.metadata}")
        if path_autonomy_plan.metadata.get("executes_tools") or path_autonomy_plan.metadata.get("queues_approval"):
            raise SystemExit(f"Autonomy plan should remain read-only and non-queuing: {path_autonomy_plan.metadata}")
        assert_safe_metadata(path_autonomy_plan.metadata, "Path-redacted autonomy plan")

        path_preflight = runtime.registry.get("risk_preflight").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-risk-preflight.txt "
                    "/private/tmp/preflight-token.txt /var/folders/zc/preflight-cache.txt /tmp/preflight-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted risk_preflight")
        print(path_preflight.output[:900])
        print()
        path_preflight_text = f"{path_preflight.output}\n{path_preflight.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_preflight_text:
                raise SystemExit(f"Risk preflight leaked local path fragment {forbidden!r}: {path_preflight_text}")
        if "<local-path>" not in path_preflight_text:
            raise SystemExit(f"Risk preflight should retain a local-path marker after redaction: {path_preflight_text}")
        if path_preflight.metadata.get("request") != "run command cat <local-path>" or path_preflight.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Risk preflight metadata should expose only the redacted request: {path_preflight.metadata}")
        if path_preflight.metadata.get("likely_approval_required") is not True or "shell/code" not in path_preflight.metadata.get("matched_risks", []):
            raise SystemExit(f"Risk preflight should still classify the original shell request as risky: {path_preflight.metadata}")
        if path_preflight.metadata.get("executes_tools") or path_preflight.metadata.get("queues_approval"):
            raise SystemExit(f"Risk preflight should remain read-only and non-queuing: {path_preflight.metadata}")
        assert_safe_metadata(path_preflight.metadata, "Path-redacted risk preflight")
        assert_risk_preflight_handoff(path_preflight.metadata, "Path-redacted risk preflight")

        path_lifecycle = runtime.registry.get("risky_request_lifecycle").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-lifecycle.txt "
                    "/private/tmp/lifecycle-token.txt /var/folders/zc/lifecycle-cache.txt /tmp/lifecycle-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted risky_request_lifecycle")
        print(path_lifecycle.output[:900])
        print()
        path_lifecycle_text = f"{path_lifecycle.output}\n{path_lifecycle.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_lifecycle_text:
                raise SystemExit(f"Risky request lifecycle leaked local path fragment {forbidden!r}: {path_lifecycle_text}")
        if "<local-path>" not in path_lifecycle_text:
            raise SystemExit(f"Risky request lifecycle should retain a local-path marker after redaction: {path_lifecycle_text}")
        if path_lifecycle.metadata.get("request") != "run command cat <local-path>" or path_lifecycle.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Risky request lifecycle metadata should expose only the redacted request: {path_lifecycle.metadata}")
        expected_lifecycle_command = "execution governor: run command cat <local-path>"
        if path_lifecycle.metadata.get("next_command") != expected_lifecycle_command:
            raise SystemExit(f"Risky request lifecycle next command should be redacted: {path_lifecycle.metadata}")
        if expected_lifecycle_command not in path_lifecycle.metadata.get("recommended_next_commands", []):
            raise SystemExit(f"Risky request lifecycle should include the redacted governor command: {path_lifecycle.metadata}")
        if "shell/code" not in path_lifecycle.metadata.get("matched_risks", []):
            raise SystemExit(f"Risky request lifecycle should still classify the original shell request as risky: {path_lifecycle.metadata}")
        if path_lifecycle.metadata.get("executes_tools") or path_lifecycle.metadata.get("queues_approval"):
            raise SystemExit(f"Risky request lifecycle should remain read-only and non-queuing: {path_lifecycle.metadata}")
        assert_safe_metadata(path_lifecycle.metadata, "Path-redacted risky request lifecycle")

        path_execution_contract = runtime.registry.get("execution_contract").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-execution-contract.txt "
                    "/private/tmp/contract-token.txt /var/folders/zc/contract-cache.txt /tmp/contract-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted execution_contract")
        print(path_execution_contract.output[:1100])
        print()
        path_contract_text = f"{path_execution_contract.output}\n{path_execution_contract.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_contract_text:
                raise SystemExit(f"Execution contract leaked local path fragment {forbidden!r}: {path_contract_text}")
        if "<local-path>" not in path_contract_text:
            raise SystemExit(f"Execution contract should retain a local-path marker after redaction: {path_contract_text}")
        if path_execution_contract.metadata.get("request") != "run command cat <local-path>" or path_execution_contract.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Execution contract metadata should expose only the redacted request: {path_execution_contract.metadata}")
        if "risk preflight: run command cat <local-path>" not in path_execution_contract.output:
            raise SystemExit(f"Execution contract preview commands should be redacted: {path_execution_contract.output}")
        if "shell/code" not in path_execution_contract.metadata.get("matched_risks", []):
            raise SystemExit(f"Execution contract should still classify the original shell request as risky: {path_execution_contract.metadata}")
        planned_contract_actions = str(path_execution_contract.metadata.get("planned_actions", []))
        if "<local-path>" not in planned_contract_actions or any(fragment in planned_contract_actions for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Execution contract planned actions should redact path-shaped args: {path_execution_contract.metadata}")
        if (
            path_execution_contract.metadata.get("approval_required") is not True
            or path_execution_contract.metadata.get("executes_tools")
            or path_execution_contract.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Execution contract should remain read-only and approval-gated for shell/code: {path_execution_contract.metadata}")
        assert_safe_metadata(path_execution_contract.metadata, "Path-redacted execution contract")
        assert_execution_contract_handoff(path_execution_contract.metadata, "Path-redacted execution contract")

        path_argument_contract = runtime.registry.get("argument_contract_packet").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-argument-contract.txt "
                    "/private/tmp/argument-token.txt /var/folders/zc/argument-cache.txt /tmp/argument-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted argument_contract_packet")
        print(path_argument_contract.output[:1100])
        print()
        path_argument_text = f"{path_argument_contract.output}\n{path_argument_contract.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_argument_text:
                raise SystemExit(f"Argument contract leaked local path fragment {forbidden!r}: {path_argument_text}")
        if "<local-path>" not in path_argument_text:
            raise SystemExit(f"Argument contract should retain a local-path marker after redaction: {path_argument_text}")
        if path_argument_contract.metadata.get("request") != "run command cat <local-path>" or path_argument_contract.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Argument contract metadata should expose only the redacted request: {path_argument_contract.metadata}")
        expected_argument_command = "execution governor: run command cat <local-path>"
        if path_argument_contract.metadata.get("next_command") != expected_argument_command:
            raise SystemExit(f"Argument contract next command should be redacted: {path_argument_contract.metadata}")
        if expected_argument_command not in path_argument_contract.output:
            raise SystemExit(f"Argument contract preview commands should be redacted: {path_argument_contract.output}")
        if "shell/code" not in path_argument_contract.metadata.get("matched_risks", []):
            raise SystemExit(f"Argument contract should still classify the original shell request as risky: {path_argument_contract.metadata}")
        planned_argument_actions = str(path_argument_contract.metadata.get("planned_actions", []))
        if "<local-path>" not in planned_argument_actions or any(fragment in planned_argument_actions for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Argument contract planned actions should redact path-shaped args: {path_argument_contract.metadata}")
        if (
            path_argument_contract.metadata.get("approval_required") is not True
            or path_argument_contract.metadata.get("executes_tools")
            or path_argument_contract.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Argument contract should remain read-only and approval-gated for shell/code: {path_argument_contract.metadata}")
        assert_safe_metadata(path_argument_contract.metadata, "Path-redacted argument contract")
        assert_argument_contract_handoff(path_argument_contract.metadata, "Path-redacted argument contract")

        path_verification = runtime.registry.get("verification_packet").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-verification.txt "
                    "/private/tmp/verification-token.txt /var/folders/zc/verification-cache.txt /tmp/verification-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted verification_packet")
        print(path_verification.output[:1100])
        print()
        path_verification_text = f"{path_verification.output}\n{path_verification.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_verification_text:
                raise SystemExit(f"Verification packet leaked local path fragment {forbidden!r}: {path_verification_text}")
        if "<local-path>" not in path_verification_text:
            raise SystemExit(f"Verification packet should retain a local-path marker after redaction: {path_verification_text}")
        if path_verification.metadata.get("request") != "run command cat <local-path>" or path_verification.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Verification packet metadata should expose only the redacted request: {path_verification.metadata}")
        expected_verification_command = "execution governor: run command cat <local-path>"
        if path_verification.metadata.get("next_command") != expected_verification_command:
            raise SystemExit(f"Verification packet next command should be redacted: {path_verification.metadata}")
        if "execution contract: run command cat <local-path>" not in path_verification.output:
            raise SystemExit(f"Verification packet suggested commands should be redacted: {path_verification.output}")
        if "shell/code" not in path_verification.metadata.get("matched_risks", []):
            raise SystemExit(f"Verification packet should still classify the original shell request as risky: {path_verification.metadata}")
        planned_verification_actions = str(path_verification.metadata.get("planned_actions", []))
        if "<local-path>" not in planned_verification_actions or any(fragment in planned_verification_actions for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Verification packet planned actions should redact path-shaped args: {path_verification.metadata}")
        if (
            path_verification.metadata.get("approval_required") is not True
            or path_verification.metadata.get("executes_tools")
            or path_verification.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Verification packet should remain read-only and approval-gated for shell/code: {path_verification.metadata}")
        assert_safe_metadata(path_verification.metadata, "Path-redacted verification packet")
        assert_verification_packet_handoff(path_verification.metadata, "Path-redacted verification packet")

        path_acceptance = runtime.registry.get("execution_acceptance_gate").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-acceptance.txt "
                    "/private/tmp/acceptance-token.txt /var/folders/zc/acceptance-cache.txt /tmp/acceptance-raw.txt"
                ),
                "evidence": "audit saw /\x55sers/example/private-acceptance.txt and /private/tmp/acceptance-token.txt",
                "tests": "smoke checked /var/folders/zc/acceptance-cache.txt",
                "recovery": "stop before touching /tmp/acceptance-raw.txt",
            }
        )
        print("[ok] direct path-redacted execution_acceptance_gate")
        print(path_acceptance.output[:1100])
        print()
        path_acceptance_text = f"{path_acceptance.output}\n{path_acceptance.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_acceptance_text:
                raise SystemExit(f"Execution acceptance gate leaked local path fragment {forbidden!r}: {path_acceptance_text}")
        if "<local-path>" not in path_acceptance_text:
            raise SystemExit(f"Execution acceptance gate should retain a local-path marker after redaction: {path_acceptance_text}")
        if path_acceptance.metadata.get("request") != "run command cat <local-path>" or path_acceptance.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Execution acceptance metadata should expose only the redacted request: {path_acceptance.metadata}")
        expected_acceptance_command = "execution proof bundle: run command cat <local-path>; verification <target>; tests <smoke>; evidence <receipt>; recovery <stop condition>"
        if path_acceptance.metadata.get("next_command") != expected_acceptance_command:
            raise SystemExit(f"Execution acceptance next command should be redacted: {path_acceptance.metadata}")
        if "args: {'command': 'cat <local-path>'}" not in path_acceptance.output:
            raise SystemExit(f"Execution acceptance route args should be redacted: {path_acceptance.output}")
        if "evidence: audit saw <local-path>" not in path_acceptance.output or "tests: smoke checked <local-path>" not in path_acceptance.output or "recovery: stop before touching <local-path>" not in path_acceptance.output:
            raise SystemExit(f"Execution acceptance supplied proof should be redacted: {path_acceptance.output}")
        if "shell/code" not in path_acceptance.metadata.get("matched_risks", []):
            raise SystemExit(f"Execution acceptance should still classify the original shell request as risky: {path_acceptance.metadata}")
        if (
            path_acceptance.metadata.get("approval_required") is not True
            or path_acceptance.metadata.get("executes_tools")
            or path_acceptance.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Execution acceptance should remain read-only and approval-gated for shell/code: {path_acceptance.metadata}")
        assert_safe_metadata(path_acceptance.metadata, "Path-redacted execution acceptance")
        assert_execution_acceptance_handoff(path_acceptance.metadata, "Path-redacted execution acceptance")

    with TemporaryDirectory(prefix="jarvis-autonomy-closure-") as temp:
        runtime = make_temp_runtime(Path(temp))

        fresh_governor = runtime.registry.get("execution_governor_packet").handler({"request": "what time is it"})
        print("[ok] direct fresh execution_governor_packet")
        print(fresh_governor.output[:900])
        print()
        if fresh_governor.metadata.get("governor_verdict") not in {"AUTO_SAFE_READY", "CHAT_READY"}:
            raise SystemExit(f"Fresh low-risk governor should not be recovery-blocked: {fresh_governor.metadata}")
        if fresh_governor.metadata.get("recovery_closure_state") != "no_recent_execution" or fresh_governor.metadata.get("recovery_closure_blocks_auto_execution") is not False:
            raise SystemExit(f"Fresh low-risk governor missed no-recent-execution metadata: {fresh_governor.metadata}")
        assert_learning_debt(fresh_governor.metadata, fresh_governor.output, "Fresh low-risk governor", state="NO_RECENT_ACTION_RUNS", blocks=False)
        fresh_intake = runtime.registry.get("command_intake_packet").handler({"request": "what time is it"})
        print("[ok] direct fresh command_intake_packet")
        print(fresh_intake.output[:900])
        print()
        if fresh_intake.metadata.get("intake_state") not in {"ready_for_policy_checked_dispatch", "ready_for_chat"}:
            raise SystemExit(f"Fresh low-risk intake should be ready before any execution history: {fresh_intake.metadata}")
        if fresh_intake.metadata.get("recovery_closure_state") != "no_recent_execution" or fresh_intake.metadata.get("recovery_closure_blocks_auto_execution") is not False:
            raise SystemExit(f"Fresh low-risk intake missed no-recent-execution metadata: {fresh_intake.metadata}")
        assert_learning_debt(fresh_intake.metadata, fresh_intake.output, "Fresh low-risk intake", state="NO_RECENT_ACTION_RUNS", blocks=False)
        fresh_dispatch = runtime.registry.get("dispatch_decision_packet").handler({"request": "what time is it"})
        print("[ok] direct fresh dispatch_decision_packet")
        print(fresh_dispatch.output[:900])
        print()
        if fresh_dispatch.metadata.get("decision") not in {"AUTO_RUN_LOCAL_SAFE_IF_SENT_FOR_REAL", "ANSWER_IN_CHAT"}:
            raise SystemExit(f"Fresh low-risk dispatch should be ready before any execution history: {fresh_dispatch.metadata}")
        if fresh_dispatch.metadata.get("recovery_closure_state") != "no_recent_execution" or fresh_dispatch.metadata.get("recovery_closure_blocks_auto_execution") is not False:
            raise SystemExit(f"Fresh low-risk dispatch missed no-recent-execution metadata: {fresh_dispatch.metadata}")
        assert_learning_debt(fresh_dispatch.metadata, fresh_dispatch.output, "Fresh low-risk dispatch", state="NO_RECENT_ACTION_RUNS", blocks=False)

        path_readiness = runtime.registry.get("action_readiness_packet").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-action-readiness.txt "
                    "/private/tmp/action-token.txt /var/folders/zc/action-cache.txt /tmp/action-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted action_readiness_packet")
        print(path_readiness.output[:1100])
        print()
        path_readiness_text = f"{path_readiness.output}\n{path_readiness.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_readiness_text:
                raise SystemExit(f"Action readiness leaked local path fragment {forbidden!r}: {path_readiness_text}")
        if "<local-path>" not in path_readiness_text:
            raise SystemExit(f"Action readiness should retain a local-path marker after redaction: {path_readiness_text}")
        if path_readiness.metadata.get("request") != "run command cat <local-path>" or path_readiness.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Action readiness metadata should expose only the redacted request: {path_readiness.metadata}")
        expected_path_command = "execution governor: run command cat <local-path>"
        if path_readiness.metadata.get("next_command") != expected_path_command:
            raise SystemExit(f"Action readiness next command should be redacted: {path_readiness.metadata}")
        if expected_path_command not in path_readiness.metadata.get("recommended_next_commands", []):
            raise SystemExit(f"Action readiness command list should include the redacted governor command: {path_readiness.metadata}")
        if path_readiness.metadata.get("recommendation") != "PREFLIGHT_REQUIRED" or path_readiness.metadata.get("route") != "preflight_required":
            raise SystemExit(f"Action readiness should still classify the original shell request as risky: {path_readiness.metadata}")
        if (
            path_readiness.metadata.get("safe_to_execute_now") is not False
            or path_readiness.metadata.get("executes_tools")
            or path_readiness.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Action readiness should remain a read-only non-queuing preflight packet: {path_readiness.metadata}")
        assert_safe_metadata(path_readiness.metadata, "Path-redacted action readiness")
        assert_action_readiness_handoff(path_readiness.metadata, "Path-redacted action readiness")

        path_matrix = runtime.registry.get("execution_readiness_matrix").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-readiness-matrix.txt "
                    "/private/tmp/matrix-token.txt /var/folders/zc/matrix-cache.txt /tmp/matrix-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted execution_readiness_matrix")
        print(path_matrix.output[:1100])
        print()
        path_matrix_text = f"{path_matrix.output}\n{path_matrix.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_matrix_text:
                raise SystemExit(f"Execution readiness matrix leaked local path fragment {forbidden!r}: {path_matrix_text}")
        if "<local-path>" not in path_matrix_text:
            raise SystemExit(f"Execution readiness matrix should retain a local-path marker after redaction: {path_matrix_text}")
        if path_matrix.metadata.get("request") != "run command cat <local-path>" or path_matrix.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Execution readiness matrix metadata should expose only the redacted request: {path_matrix.metadata}")
        expected_matrix_command = "execution contract: run command cat <local-path>"
        if path_matrix.metadata.get("next_command") != expected_matrix_command:
            raise SystemExit(f"Execution readiness matrix next command should be redacted: {path_matrix.metadata}")
        if "Args: command='cat <local-path>'" not in path_matrix.output:
            raise SystemExit(f"Execution readiness matrix planned args should be redacted: {path_matrix.output}")
        if "shell/code" not in path_matrix.metadata.get("matched_risks", []):
            raise SystemExit(f"Execution readiness matrix should still classify the original shell request as risky: {path_matrix.metadata}")
        planned_matrix_actions = str(path_matrix.metadata.get("planned_actions", []))
        if "<local-path>" not in planned_matrix_actions or any(fragment in planned_matrix_actions for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Execution readiness matrix planned actions should redact path-shaped args: {path_matrix.metadata}")
        if (
            path_matrix.metadata.get("approval_required") is not True
            or path_matrix.metadata.get("executes_tools")
            or path_matrix.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Execution readiness matrix should remain read-only and approval-gated for shell/code: {path_matrix.metadata}")
        assert_safe_metadata(path_matrix.metadata, "Path-redacted execution readiness matrix")
        assert_execution_readiness_handoff(path_matrix.metadata, "Path-redacted execution readiness matrix")

        path_dispatch = runtime.registry.get("dispatch_decision_packet").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-dispatch.txt "
                    "/private/tmp/dispatch-token.txt /var/folders/zc/dispatch-cache.txt /tmp/dispatch-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted dispatch_decision_packet")
        print(path_dispatch.output[:1100])
        print()
        path_dispatch_text = f"{path_dispatch.output}\n{path_dispatch.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_dispatch_text:
                raise SystemExit(f"Dispatch decision leaked local path fragment {forbidden!r}: {path_dispatch_text}")
        if "<local-path>" not in path_dispatch_text:
            raise SystemExit(f"Dispatch decision should retain a local-path marker after redaction: {path_dispatch_text}")
        if path_dispatch.metadata.get("request") != "run command cat <local-path>" or path_dispatch.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Dispatch decision metadata should expose only the redacted request: {path_dispatch.metadata}")
        if path_dispatch.metadata.get("primary_command") != "run command cat <local-path>":
            raise SystemExit(f"Dispatch decision primary command should be redacted: {path_dispatch.metadata}")
        if "Order: run command cat <local-path>" not in path_dispatch.output:
            raise SystemExit(f"Dispatch decision order should be redacted: {path_dispatch.output}")
        if "Args: command='cat <local-path>'" not in path_dispatch.output:
            raise SystemExit(f"Dispatch decision planned args should be redacted: {path_dispatch.output}")
        if "risk preflight: run command cat <local-path>" not in path_dispatch.output or "execution readiness matrix: run command cat <local-path>" not in path_dispatch.output:
            raise SystemExit(f"Dispatch decision preflight commands should be redacted: {path_dispatch.output}")
        if "shell/code" not in path_dispatch.metadata.get("matched_risks", []):
            raise SystemExit(f"Dispatch decision should still classify the original shell request as risky: {path_dispatch.metadata}")
        planned_dispatch_actions = str(path_dispatch.metadata.get("planned_actions", []))
        if "<local-path>" not in planned_dispatch_actions or any(fragment in planned_dispatch_actions for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Dispatch decision planned actions should redact path-shaped args: {path_dispatch.metadata}")
        if (
            path_dispatch.metadata.get("decision") != "QUEUE_APPROVAL_IF_SENT_FOR_REAL"
            or path_dispatch.metadata.get("approval_required") is not True
            or path_dispatch.metadata.get("can_auto_run")
            or path_dispatch.metadata.get("executes_tools")
            or path_dispatch.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Dispatch decision should remain read-only and approval-gated for shell/code: {path_dispatch.metadata}")
        assert_safe_metadata(path_dispatch.metadata, "Path-redacted dispatch decision")

        path_intake = runtime.registry.get("command_intake_packet").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-intake.txt "
                    "/private/tmp/intake-token.txt /var/folders/zc/intake-cache.txt /tmp/intake-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted command_intake_packet")
        print(path_intake.output[:1100])
        print()
        path_intake_text = f"{path_intake.output}\n{path_intake.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_intake_text:
                raise SystemExit(f"Command intake leaked local path fragment {forbidden!r}: {path_intake_text}")
        if "<local-path>" not in path_intake_text:
            raise SystemExit(f"Command intake should retain a local-path marker after redaction: {path_intake_text}")
        if path_intake.metadata.get("request") != "run command cat <local-path>" or path_intake.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Command intake metadata should expose only the redacted request: {path_intake.metadata}")
        expected_intake_command = "execution governor: run command cat <local-path>"
        if path_intake.metadata.get("next_command") != expected_intake_command:
            raise SystemExit(f"Command intake next command should be redacted: {path_intake.metadata}")
        if "Order: run command cat <local-path>" not in path_intake.output:
            raise SystemExit(f"Command intake order should be redacted: {path_intake.output}")
        if "Args: command='cat <local-path>'" not in path_intake.output:
            raise SystemExit(f"Command intake planned args should be redacted: {path_intake.output}")
        if "dispatch decision: run command cat <local-path>" not in path_intake.output or "verification packet: run command cat <local-path>" not in path_intake.output:
            raise SystemExit(f"Command intake follow-up packets should be redacted: {path_intake.output}")
        if "shell/code" not in path_intake.metadata.get("matched_risks", []):
            raise SystemExit(f"Command intake should still classify the original shell request as risky: {path_intake.metadata}")
        planned_intake_actions = str(path_intake.metadata.get("planned_actions", []))
        if "<local-path>" not in planned_intake_actions or any(fragment in planned_intake_actions for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Command intake planned actions should redact path-shaped args: {path_intake.metadata}")
        if (
            path_intake.metadata.get("intake_state") != "approval_required"
            or path_intake.metadata.get("approval_required") is not True
            or path_intake.metadata.get("can_auto_run")
            or path_intake.metadata.get("executes_tools")
            or path_intake.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Command intake should remain read-only and approval-gated for shell/code: {path_intake.metadata}")
        assert_safe_metadata(path_intake.metadata, "Path-redacted command intake")

        path_governor = runtime.registry.get("execution_governor_packet").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-governor.txt "
                    "/private/tmp/governor-token.txt /var/folders/zc/governor-cache.txt /tmp/governor-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted execution_governor_packet")
        print(path_governor.output[:1100])
        print()
        path_governor_text = f"{path_governor.output}\n{path_governor.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_governor_text:
                raise SystemExit(f"Execution governor leaked local path fragment {forbidden!r}: {path_governor_text}")
        if "<local-path>" not in path_governor_text:
            raise SystemExit(f"Execution governor should retain a local-path marker after redaction: {path_governor_text}")
        if path_governor.metadata.get("request") != "run command cat <local-path>" or path_governor.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Execution governor metadata should expose only the redacted request: {path_governor.metadata}")
        expected_governor_command = "dispatch decision: run command cat <local-path>"
        if path_governor.metadata.get("next_command") != expected_governor_command:
            raise SystemExit(f"Execution governor next command should be redacted: {path_governor.metadata}")
        if "Order: run command cat <local-path>" not in path_governor.output:
            raise SystemExit(f"Execution governor order should be redacted: {path_governor.output}")
        if "Args: command='cat <local-path>'" not in path_governor.output:
            raise SystemExit(f"Execution governor planned args should be redacted: {path_governor.output}")
        if "command intake: run command cat <local-path>" not in path_governor.output or "argument contract: run command cat <local-path>" not in path_governor.output:
            raise SystemExit(f"Execution governor follow-up packets should be redacted: {path_governor.output}")
        if "shell/code" not in path_governor.metadata.get("matched_risks", []):
            raise SystemExit(f"Execution governor should still classify the original shell request as risky: {path_governor.metadata}")
        planned_governor_actions = str(path_governor.metadata.get("planned_actions", []))
        if "<local-path>" not in planned_governor_actions or any(fragment in planned_governor_actions for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Execution governor planned actions should redact path-shaped args: {path_governor.metadata}")
        if (
            path_governor.metadata.get("governor_verdict") != "APPROVAL_REQUIRED"
            or path_governor.metadata.get("approval_required") is not True
            or path_governor.metadata.get("can_auto_run")
            or path_governor.metadata.get("safe_to_execute_now")
            or path_governor.metadata.get("executes_tools")
            or path_governor.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Execution governor should remain read-only and approval-gated for shell/code: {path_governor.metadata}")
        assert_safe_metadata(path_governor.metadata, "Path-redacted execution governor")
        assert_frontdoor_handoff(path_governor.metadata, "Path-redacted execution governor", "execution_governor")
        governor_handoff = path_governor.metadata.get("execution_governor_handoff") or {}
        for field in ["governor_verdict", "route", "next_command", "can_auto_run", "safe_to_execute_now", "reason", "matched_risks", "approval_required", "approval_review_required", "planned_actions", "exact_arguments_ready", "ambiguous_risky_order", "unknown_tools", "recovery_debt_visible", "governor_preview_allowed_with_recovery_debt", "recovery_closure_blocks_governed_execution", "recovery_closure_state", "recovery_closure_ready_to_retry"]:
            if governor_handoff.get(field) != path_governor.metadata.get(field):
                raise SystemExit(f"Execution governor handoff missed {field} parity: {governor_handoff} vs {path_governor.metadata}")
        if governor_handoff.get("planned_action_count") != path_governor.metadata.get("planned_action_count") or governor_handoff.get("pending_approval_count") != path_governor.metadata.get("pending_approvals"):
            raise SystemExit(f"Execution governor handoff missed count parity: {governor_handoff} vs {path_governor.metadata}")

        path_gap = runtime.registry.get("planner_gap_packet").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-gap.txt "
                    "/private/tmp/gap-token.txt /var/folders/zc/gap-cache.txt /tmp/gap-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted planner_gap_packet")
        print(path_gap.output[:1100])
        print()
        path_gap_text = f"{path_gap.output}\n{path_gap.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_gap_text:
                raise SystemExit(f"Planner gap leaked local path fragment {forbidden!r}: {path_gap_text}")
        if "<local-path>" not in path_gap_text:
            raise SystemExit(f"Planner gap should retain a local-path marker after redaction: {path_gap_text}")
        if path_gap.metadata.get("request") != "run command cat <local-path>" or path_gap.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Planner gap metadata should expose only the redacted request: {path_gap.metadata}")
        expected_gap_command = "execution governor: run command cat <local-path>"
        if path_gap.metadata.get("next_command") != expected_gap_command:
            raise SystemExit(f"Planner gap next command should be redacted: {path_gap.metadata}")
        if "Order: run command cat <local-path>" not in path_gap.output:
            raise SystemExit(f"Planner gap order should be redacted: {path_gap.output}")
        if "execution governor: run command cat <local-path>" not in path_gap.output or "verification packet: run command cat <local-path>" not in path_gap.output:
            raise SystemExit(f"Planner gap suggested follow-ups should be redacted: {path_gap.output}")
        if "shell/code" not in path_gap.metadata.get("matched_risks", []):
            raise SystemExit(f"Planner gap should still classify the original shell request as risky: {path_gap.metadata}")
        planned_gap_actions = str(path_gap.metadata.get("planned_actions", []))
        if "<local-path>" not in planned_gap_actions or any(fragment in planned_gap_actions for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Planner gap planned actions should redact path-shaped args: {path_gap.metadata}")
        if (
            path_gap.metadata.get("classification") != "ROUTED"
            or path_gap.metadata.get("approval_required") is not True
            or path_gap.metadata.get("executes_tools")
            or path_gap.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Planner gap should remain read-only and approval-aware for shell/code: {path_gap.metadata}")
        assert_safe_metadata(path_gap.metadata, "Path-redacted planner gap")
        assert_frontdoor_handoff(path_gap.metadata, "Path-redacted planner gap", "planner_gap")
        gap_handoff = path_gap.metadata.get("planner_gap_handoff") or {}
        for field in ["classification", "next_command", "matched_risks", "gap_signals", "planned_actions", "approval_required", "needs_model", "unknown_tools"]:
            if gap_handoff.get(field) != path_gap.metadata.get(field):
                raise SystemExit(f"Planner gap handoff missed {field} parity: {gap_handoff} vs {path_gap.metadata}")
        if gap_handoff.get("planned_action_count") != path_gap.metadata.get("planned_action_count") or gap_handoff.get("suggested_followup_count") != path_gap.metadata.get("suggested_followups"):
            raise SystemExit(f"Planner gap handoff missed count parity: {gap_handoff} vs {path_gap.metadata}")

        path_loop_preview = runtime.registry.get("agent_loop_preview").handler(
            {
                "goal": (
                    "run command cat /\x55sers/example/private-loop-preview.txt "
                    "/private/tmp/loop-preview-token.txt /var/folders/zc/loop-preview-cache.txt /tmp/loop-preview-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted agent_loop_preview")
        print(path_loop_preview.output[:1000])
        print()
        path_loop_preview_text = f"{path_loop_preview.output}\n{path_loop_preview.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_loop_preview_text:
                raise SystemExit(f"Agent loop preview leaked local path fragment {forbidden!r}: {path_loop_preview_text}")
        if "<local-path>" not in path_loop_preview_text:
            raise SystemExit(f"Agent loop preview should retain a local-path marker after redaction: {path_loop_preview_text}")
        if path_loop_preview.metadata.get("goal") != "run command cat <local-path>" or path_loop_preview.metadata.get("display_goal") != "run command cat <local-path>":
            raise SystemExit(f"Agent loop preview metadata should expose only the redacted goal: {path_loop_preview.metadata}")
        expected_loop_command = "execution governor: run command cat <local-path>"
        if path_loop_preview.metadata.get("next_command") != expected_loop_command:
            raise SystemExit(f"Agent loop preview next command should be redacted: {path_loop_preview.metadata}")
        if "Goal: run command cat <local-path>" not in path_loop_preview.output or "risk preflight: run command cat <local-path>" not in path_loop_preview.output:
            raise SystemExit(f"Agent loop preview output should render only redacted commands: {path_loop_preview.output}")
        if "shell/code" not in path_loop_preview.metadata.get("matched_risks", []):
            raise SystemExit(f"Agent loop preview should still classify the original shell request as risky: {path_loop_preview.metadata}")
        recommended_loop_preview = str(path_loop_preview.metadata.get("recommended_next_commands", []))
        if "<local-path>" not in recommended_loop_preview or any(fragment in recommended_loop_preview for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Agent loop preview recommended commands should be redacted: {path_loop_preview.metadata}")
        assert_safe_metadata(path_loop_preview.metadata, "Path-redacted agent loop preview")

        path_loop_packet = runtime.registry.get("agent_loop_packet").handler(
            {
                "goal": (
                    "run command cat /\x55sers/example/private-loop-packet.txt "
                    "/private/tmp/loop-packet-token.txt /var/folders/zc/loop-packet-cache.txt /tmp/loop-packet-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted agent_loop_packet")
        print(path_loop_packet.output[:1000])
        print()
        path_loop_packet_text = f"{path_loop_packet.output}\n{path_loop_packet.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_loop_packet_text:
                raise SystemExit(f"Agent loop packet leaked local path fragment {forbidden!r}: {path_loop_packet_text}")
        if "<local-path>" not in path_loop_packet_text:
            raise SystemExit(f"Agent loop packet should retain a local-path marker after redaction: {path_loop_packet_text}")
        if path_loop_packet.metadata.get("goal") != "run command cat <local-path>" or path_loop_packet.metadata.get("display_goal") != "run command cat <local-path>":
            raise SystemExit(f"Agent loop packet metadata should expose only the redacted goal: {path_loop_packet.metadata}")
        if path_loop_packet.metadata.get("next_command") != expected_loop_command:
            raise SystemExit(f"Agent loop packet next command should be redacted: {path_loop_packet.metadata}")
        if "Goal: run command cat <local-path>" not in path_loop_packet.output or "risky request lifecycle: run command cat <local-path>" not in path_loop_packet.output:
            raise SystemExit(f"Agent loop packet output should render only redacted commands: {path_loop_packet.output}")
        if "shell/code" not in path_loop_packet.metadata.get("matched_risks", []):
            raise SystemExit(f"Agent loop packet should still classify the original shell request as risky: {path_loop_packet.metadata}")
        recommended_loop_packet = str(path_loop_packet.metadata.get("recommended_next_commands", []))
        if "<local-path>" not in recommended_loop_packet or any(fragment in recommended_loop_packet for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Agent loop packet recommended commands should be redacted: {path_loop_packet.metadata}")
        if path_loop_packet.metadata.get("likely_approval_required") is not True:
            raise SystemExit(f"Agent loop packet should preserve approval likelihood from original risky goal: {path_loop_packet.metadata}")
        assert_safe_metadata(path_loop_packet.metadata, "Path-redacted agent loop packet")

        path_cockpit = runtime.registry.get("command_cockpit_packet").handler(
            {
                "request": (
                    "run command cat /\x55sers/example/private-cockpit.txt "
                    "/private/tmp/cockpit-token.txt /var/folders/zc/cockpit-cache.txt /tmp/cockpit-raw.txt"
                )
            }
        )
        print("[ok] direct path-redacted command_cockpit_packet")
        print(path_cockpit.output[:1100])
        print()
        path_cockpit_text = f"{path_cockpit.output}\n{path_cockpit.metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_cockpit_text:
                raise SystemExit(f"Command cockpit leaked local path fragment {forbidden!r}: {path_cockpit_text}")
        if "<local-path>" not in path_cockpit_text:
            raise SystemExit(f"Command cockpit should retain a local-path marker after redaction: {path_cockpit_text}")
        if path_cockpit.metadata.get("request") != "run command cat <local-path>" or path_cockpit.metadata.get("display_request") != "run command cat <local-path>":
            raise SystemExit(f"Command cockpit metadata should expose only the redacted request: {path_cockpit.metadata}")
        expected_cockpit_command = "dispatch decision: run command cat <local-path>"
        if path_cockpit.metadata.get("next_command") != expected_cockpit_command:
            raise SystemExit(f"Command cockpit next command should be redacted: {path_cockpit.metadata}")
        if "Order: run command cat <local-path>" not in path_cockpit.output:
            raise SystemExit(f"Command cockpit order should be redacted: {path_cockpit.output}")
        if "command intake: run command cat <local-path>" not in path_cockpit.output or "verification packet: run command cat <local-path>" not in path_cockpit.output:
            raise SystemExit(f"Command cockpit proof queue should be redacted: {path_cockpit.output}")
        packet_rows = str(path_cockpit.metadata.get("packet_rows", []))
        proof_queue = str(path_cockpit.metadata.get("proof_queue", []))
        required_commands = str(path_cockpit.metadata.get("required_commands", []))
        for label, value in [("packet rows", packet_rows), ("proof queue", proof_queue), ("required commands", required_commands)]:
            if "<local-path>" not in value or any(fragment in value for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
                raise SystemExit(f"Command cockpit {label} should be redacted: {path_cockpit.metadata}")
        if (
            path_cockpit.metadata.get("cockpit_verdict") != "APPROVAL_REQUIRED_BEFORE_EXECUTION"
            or path_cockpit.metadata.get("approval_required") is not True
            or path_cockpit.metadata.get("can_auto_run")
            or path_cockpit.metadata.get("executes_tools")
            or path_cockpit.metadata.get("queues_approval")
        ):
            raise SystemExit(f"Command cockpit should remain read-only and approval-gated for shell/code: {path_cockpit.metadata}")
        assert_safe_metadata(path_cockpit.metadata, "Path-redacted command cockpit")
        assert_frontdoor_handoff(path_cockpit.metadata, "Path-redacted command cockpit", "command_cockpit")
        cockpit_handoff = path_cockpit.metadata.get("command_cockpit_handoff") or {}
        for field in ["cockpit_verdict", "next_command", "can_auto_run", "approval_required", "recovery_debt_visible", "cockpit_preview_allowed_with_recovery_debt", "recovery_closure_blocks_cockpit_execution", "recovery_closure_state", "recovery_closure_ready_to_retry", "execution_learning_blocks_completion_claim", "execution_learning_state", "packet_rows", "governor_verdict", "governor_route", "dispatch_decision", "dispatch_route", "readiness_verdict", "verification_verdict"]:
            if cockpit_handoff.get(field) != path_cockpit.metadata.get(field):
                raise SystemExit(f"Command cockpit handoff missed {field} parity: {cockpit_handoff} vs {path_cockpit.metadata}")
        if cockpit_handoff.get("packet_row_count") != path_cockpit.metadata.get("packet_row_count") or cockpit_handoff.get("pending_approval_count") != path_cockpit.metadata.get("pending_approvals") or cockpit_handoff.get("required_command_count") != path_cockpit.metadata.get("required_command_count"):
            raise SystemExit(f"Command cockpit handoff missed count parity: {cockpit_handoff} vs {path_cockpit.metadata}")

        fresh_risky_intake = runtime.registry.get("command_intake_packet").handler({"request": "run command python3 --version"})
        print("[ok] direct fresh risky command_intake_packet forecast")
        print(fresh_risky_intake.output[:1100])
        print()
        if (
            fresh_risky_intake.metadata.get("forecast_queue_before") != 0
            or fresh_risky_intake.metadata.get("forecast_queue_after_if_sent") != 1
            or fresh_risky_intake.metadata.get("forecast_queue_delta_if_sent") != 1
            or fresh_risky_intake.metadata.get("forecast_new_approvals") != 1
            or fresh_risky_intake.metadata.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Fresh risky intake missed new-approval forecast: {fresh_risky_intake.metadata}")
        if "would queue new approvals if sent: 1" not in fresh_risky_intake.output or "forecast queue after if sent: 1" not in fresh_risky_intake.output:
            raise SystemExit("Fresh risky intake did not render approval forecast.")
        fresh_risky_forecast = fresh_risky_intake.metadata.get("approval_queue_forecast") or []
        if not fresh_risky_forecast or fresh_risky_forecast[0].get("tool_name") != "run_shell_command" or fresh_risky_forecast[0].get("would_queue_new_approval") is not True:
            raise SystemExit(f"Fresh risky intake missed per-action forecast: {fresh_risky_intake.metadata}")

        fresh_risky_governor = runtime.registry.get("execution_governor_packet").handler({"request": "run command python3 --version"})
        print("[ok] direct fresh risky execution_governor_packet forecast")
        print(fresh_risky_governor.output[:1100])
        print()
        if (
            fresh_risky_governor.metadata.get("forecast_queue_before") != 0
            or fresh_risky_governor.metadata.get("forecast_queue_after_if_sent") != 1
            or fresh_risky_governor.metadata.get("forecast_queue_delta_if_sent") != 1
            or fresh_risky_governor.metadata.get("forecast_new_approvals") != 1
            or fresh_risky_governor.metadata.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Fresh risky governor missed new-approval forecast: {fresh_risky_governor.metadata}")
        if "would queue new approvals if sent: 1" not in fresh_risky_governor.output or "forecast queue after if sent: 1" not in fresh_risky_governor.output:
            raise SystemExit("Fresh risky governor did not render approval forecast.")
        fresh_governor_forecast = fresh_risky_governor.metadata.get("approval_queue_forecast") or []
        if not fresh_governor_forecast or fresh_governor_forecast[0].get("tool_name") != "run_shell_command" or fresh_governor_forecast[0].get("would_queue_new_approval") is not True:
            raise SystemExit(f"Fresh risky governor missed per-action forecast: {fresh_risky_governor.metadata}")

        fresh_risky_dispatch = runtime.registry.get("dispatch_decision_packet").handler({"request": "run command python3 --version"})
        print("[ok] direct fresh risky dispatch_decision_packet forecast")
        print(fresh_risky_dispatch.output[:1100])
        print()
        if (
            fresh_risky_dispatch.metadata.get("forecast_queue_before") != 0
            or fresh_risky_dispatch.metadata.get("forecast_queue_after_if_sent") != 1
            or fresh_risky_dispatch.metadata.get("forecast_queue_delta_if_sent") != 1
            or fresh_risky_dispatch.metadata.get("forecast_new_approvals") != 1
            or fresh_risky_dispatch.metadata.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Fresh risky dispatch missed new-approval forecast: {fresh_risky_dispatch.metadata}")
        if "would queue new approvals if sent: 1" not in fresh_risky_dispatch.output or "forecast queue after if sent: 1" not in fresh_risky_dispatch.output:
            raise SystemExit("Fresh risky dispatch did not render approval forecast.")
        fresh_dispatch_forecast = fresh_risky_dispatch.metadata.get("approval_queue_forecast") or []
        if not fresh_dispatch_forecast or fresh_dispatch_forecast[0].get("tool_name") != "run_shell_command" or fresh_dispatch_forecast[0].get("would_queue_new_approval") is not True:
            raise SystemExit(f"Fresh risky dispatch missed per-action forecast: {fresh_risky_dispatch.metadata}")

        fresh_risky_matrix = runtime.registry.get("execution_readiness_matrix").handler({"request": "run command python3 --version"})
        print("[ok] direct fresh risky execution_readiness_matrix forecast")
        print(fresh_risky_matrix.output[:1100])
        print()
        if (
            fresh_risky_matrix.metadata.get("forecast_queue_before") != 0
            or fresh_risky_matrix.metadata.get("forecast_queue_after_if_sent") != 1
            or fresh_risky_matrix.metadata.get("forecast_queue_delta_if_sent") != 1
            or fresh_risky_matrix.metadata.get("forecast_new_approvals") != 1
            or fresh_risky_matrix.metadata.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Fresh risky readiness matrix missed new-approval forecast: {fresh_risky_matrix.metadata}")
        if "Approval forecast: 1 new, 0 reused, queue after 1" not in fresh_risky_matrix.output:
            raise SystemExit("Fresh risky readiness matrix did not render approval forecast.")
        fresh_matrix_forecast = fresh_risky_matrix.metadata.get("approval_queue_forecast") or []
        if not fresh_matrix_forecast or fresh_matrix_forecast[0].get("tool_name") != "run_shell_command" or fresh_matrix_forecast[0].get("would_queue_new_approval") is not True:
            raise SystemExit(f"Fresh risky readiness matrix missed per-action forecast: {fresh_risky_matrix.metadata}")
        assert_execution_readiness_handoff(fresh_risky_matrix.metadata, "Fresh risky execution readiness matrix")

        fresh_risky_argument = runtime.registry.get("argument_contract_packet").handler({"request": "run command python3 --version"})
        print("[ok] direct fresh risky argument_contract_packet forecast")
        print(fresh_risky_argument.output[:1100])
        print()
        if (
            fresh_risky_argument.metadata.get("forecast_queue_before") != 0
            or fresh_risky_argument.metadata.get("forecast_queue_after_if_sent") != 1
            or fresh_risky_argument.metadata.get("forecast_queue_delta_if_sent") != 1
            or fresh_risky_argument.metadata.get("forecast_new_approvals") != 1
            or fresh_risky_argument.metadata.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Fresh risky argument contract missed new-approval forecast: {fresh_risky_argument.metadata}")
        if "Approval queue forecast:" not in fresh_risky_argument.output or "would queue new approvals if sent: 1" not in fresh_risky_argument.output:
            raise SystemExit("Fresh risky argument contract did not render approval forecast.")
        fresh_argument_forecast = fresh_risky_argument.metadata.get("approval_queue_forecast") or []
        if not fresh_argument_forecast or fresh_argument_forecast[0].get("tool_name") != "run_shell_command" or fresh_argument_forecast[0].get("would_queue_new_approval") is not True:
            raise SystemExit(f"Fresh risky argument contract missed per-action forecast: {fresh_risky_argument.metadata}")
        assert_argument_contract_handoff(fresh_risky_argument.metadata, "Fresh risky argument contract")

        fresh_risky_verification = runtime.registry.get("verification_packet").handler({"request": "run command python3 --version"})
        print("[ok] direct fresh risky verification_packet forecast")
        print(fresh_risky_verification.output[:1100])
        print()
        if (
            fresh_risky_verification.metadata.get("forecast_queue_before") != 0
            or fresh_risky_verification.metadata.get("forecast_queue_after_if_sent") != 1
            or fresh_risky_verification.metadata.get("forecast_queue_delta_if_sent") != 1
            or fresh_risky_verification.metadata.get("forecast_new_approvals") != 1
            or fresh_risky_verification.metadata.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Fresh risky verification packet missed new-approval forecast: {fresh_risky_verification.metadata}")
        if "Approval queue forecast:" not in fresh_risky_verification.output or "would queue new approvals if sent: 1" not in fresh_risky_verification.output:
            raise SystemExit("Fresh risky verification packet did not render approval forecast.")
        fresh_verification_forecast = fresh_risky_verification.metadata.get("approval_queue_forecast") or []
        if not fresh_verification_forecast or fresh_verification_forecast[0].get("tool_name") != "run_shell_command" or fresh_verification_forecast[0].get("would_queue_new_approval") is not True:
            raise SystemExit(f"Fresh risky verification packet missed per-action forecast: {fresh_risky_verification.metadata}")
        assert_verification_packet_handoff(fresh_risky_verification.metadata, "Fresh risky verification packet")

        fresh_risky_acceptance = runtime.registry.get("execution_acceptance_gate").handler(
            {
                "request": "run command python3 --version",
                "evidence": "approval packet viewed and recent tool run ok",
                "tests": "smoke test passed",
                "rollback": "stop on non-zero exit",
            }
        )
        print("[ok] direct fresh risky execution_acceptance_gate forecast")
        print(fresh_risky_acceptance.output[:1100])
        print()
        if (
            fresh_risky_acceptance.metadata.get("forecast_queue_before") != 0
            or fresh_risky_acceptance.metadata.get("forecast_queue_after_if_sent") != 1
            or fresh_risky_acceptance.metadata.get("forecast_queue_delta_if_sent") != 1
            or fresh_risky_acceptance.metadata.get("forecast_new_approvals") != 1
            or fresh_risky_acceptance.metadata.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Fresh risky acceptance gate missed new-approval forecast: {fresh_risky_acceptance.metadata}")
        if "would queue new approvals if sent: 1" not in fresh_risky_acceptance.output or "forecast queue after if sent: 1" not in fresh_risky_acceptance.output:
            raise SystemExit("Fresh risky acceptance gate did not render approval forecast.")
        fresh_acceptance_forecast = fresh_risky_acceptance.metadata.get("approval_queue_forecast") or []
        if not fresh_acceptance_forecast or fresh_acceptance_forecast[0].get("tool_name") != "run_shell_command" or fresh_acceptance_forecast[0].get("would_queue_new_approval") is not True:
            raise SystemExit(f"Fresh risky acceptance gate missed per-action forecast: {fresh_risky_acceptance.metadata}")
        assert_execution_acceptance_handoff(fresh_risky_acceptance.metadata, "Fresh risky execution acceptance")

        meta_only = runtime.handle("action readiness: what time is it")
        if not meta_only.verified:
            raise SystemExit(f"Meta-only recovery-closure smoke could not create a readiness packet: {meta_only.response}")
        meta_only_governor = runtime.registry.get("execution_governor_packet").handler({"request": "what time is it"})
        print("[ok] direct meta-only execution_governor_packet")
        print(meta_only_governor.output[:900])
        print()
        if meta_only_governor.metadata.get("recovery_closure_state") != "no_recent_execution" or meta_only_governor.metadata.get("recovery_closure_target_run_id") is not None:
            raise SystemExit(f"Meta-only history should not become a recovery target: {meta_only_governor.metadata}")

        blocked = runtime.handle("run command python3 --version")
        if blocked.verified:
            raise SystemExit("Approval-gated command unexpectedly executed in recovery-closure smoke.")
        run_id = blocked.tool_results[0].metadata.get("logged_tool_run_id")
        if not isinstance(run_id, int):
            raise SystemExit(f"Recovery-closure smoke could not find blocked run id: {blocked.tool_results[0].metadata}")
        held_approval_commands = [
            "approval readiness 1",
            "approval packet 1",
            "approval chain proof 1",
            "verification receipt <approved run id from approval chain proof 1>",
        ]
        forbidden_held_recovery_commands = [
            f"verification receipt {run_id}",
            f"execution recovery packet {run_id}",
            f"execution learning closure {run_id}",
            f"after-action learning packet {run_id}",
        ]

        duplicate_risky_intake = runtime.registry.get("command_intake_packet").handler({"request": "run command python3 --version"})
        print("[ok] direct duplicate risky command_intake_packet forecast")
        print(duplicate_risky_intake.output[:1100])
        print()
        if (
            duplicate_risky_intake.metadata.get("forecast_queue_before") != 1
            or duplicate_risky_intake.metadata.get("forecast_queue_after_if_sent") != 1
            or duplicate_risky_intake.metadata.get("forecast_queue_delta_if_sent") != 0
            or duplicate_risky_intake.metadata.get("forecast_new_approvals") != 0
            or duplicate_risky_intake.metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Duplicate risky intake missed approval reuse forecast: {duplicate_risky_intake.metadata}")
        if "would queue new approvals if sent: 0" not in duplicate_risky_intake.output or "would reuse pending approval ids: 1" not in duplicate_risky_intake.output:
            raise SystemExit("Duplicate risky intake did not render approval reuse forecast.")
        duplicate_intake_forecast = duplicate_risky_intake.metadata.get("approval_queue_forecast") or []
        if not duplicate_intake_forecast or duplicate_intake_forecast[0].get("existing_approval_id") != 1 or duplicate_intake_forecast[0].get("would_reuse_pending_approval") is not True:
            raise SystemExit(f"Duplicate risky intake missed per-action reuse forecast: {duplicate_risky_intake.metadata}")

        duplicate_risky_governor = runtime.registry.get("execution_governor_packet").handler({"request": "run command python3 --version"})
        print("[ok] direct duplicate risky execution_governor_packet forecast")
        print(duplicate_risky_governor.output[:1100])
        print()
        if (
            duplicate_risky_governor.metadata.get("forecast_queue_before") != 1
            or duplicate_risky_governor.metadata.get("forecast_queue_after_if_sent") != 1
            or duplicate_risky_governor.metadata.get("forecast_queue_delta_if_sent") != 0
            or duplicate_risky_governor.metadata.get("forecast_new_approvals") != 0
            or duplicate_risky_governor.metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Duplicate risky governor missed approval reuse forecast: {duplicate_risky_governor.metadata}")
        if "would queue new approvals if sent: 0" not in duplicate_risky_governor.output or "would reuse pending approval ids: 1" not in duplicate_risky_governor.output:
            raise SystemExit("Duplicate risky governor did not render approval reuse forecast.")
        duplicate_governor_forecast = duplicate_risky_governor.metadata.get("approval_queue_forecast") or []
        if not duplicate_governor_forecast or duplicate_governor_forecast[0].get("existing_approval_id") != 1 or duplicate_governor_forecast[0].get("would_reuse_pending_approval") is not True:
            raise SystemExit(f"Duplicate risky governor missed per-action reuse forecast: {duplicate_risky_governor.metadata}")

        duplicate_risky_dispatch = runtime.registry.get("dispatch_decision_packet").handler({"request": "run command python3 --version"})
        print("[ok] direct duplicate risky dispatch_decision_packet forecast")
        print(duplicate_risky_dispatch.output[:1100])
        print()
        if (
            duplicate_risky_dispatch.metadata.get("forecast_queue_before") != 1
            or duplicate_risky_dispatch.metadata.get("forecast_queue_after_if_sent") != 1
            or duplicate_risky_dispatch.metadata.get("forecast_queue_delta_if_sent") != 0
            or duplicate_risky_dispatch.metadata.get("forecast_new_approvals") != 0
            or duplicate_risky_dispatch.metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Duplicate risky dispatch missed approval reuse forecast: {duplicate_risky_dispatch.metadata}")
        if "would queue new approvals if sent: 0" not in duplicate_risky_dispatch.output or "would reuse pending approval ids: 1" not in duplicate_risky_dispatch.output:
            raise SystemExit("Duplicate risky dispatch did not render approval reuse forecast.")
        duplicate_dispatch_forecast = duplicate_risky_dispatch.metadata.get("approval_queue_forecast") or []
        if not duplicate_dispatch_forecast or duplicate_dispatch_forecast[0].get("existing_approval_id") != 1 or duplicate_dispatch_forecast[0].get("would_reuse_pending_approval") is not True:
            raise SystemExit(f"Duplicate risky dispatch missed per-action reuse forecast: {duplicate_risky_dispatch.metadata}")

        duplicate_risky_matrix = runtime.registry.get("execution_readiness_matrix").handler({"request": "run command python3 --version"})
        print("[ok] direct duplicate risky execution_readiness_matrix forecast")
        print(duplicate_risky_matrix.output[:1100])
        print()
        if (
            duplicate_risky_matrix.metadata.get("forecast_queue_before") != 1
            or duplicate_risky_matrix.metadata.get("forecast_queue_after_if_sent") != 1
            or duplicate_risky_matrix.metadata.get("forecast_queue_delta_if_sent") != 0
            or duplicate_risky_matrix.metadata.get("forecast_new_approvals") != 0
            or duplicate_risky_matrix.metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Duplicate risky readiness matrix missed approval reuse forecast: {duplicate_risky_matrix.metadata}")
        if "Approval forecast: 0 new, 1 reused, queue after 1" not in duplicate_risky_matrix.output:
            raise SystemExit("Duplicate risky readiness matrix did not render approval reuse forecast.")
        duplicate_matrix_forecast = duplicate_risky_matrix.metadata.get("approval_queue_forecast") or []
        if not duplicate_matrix_forecast or duplicate_matrix_forecast[0].get("existing_approval_id") != 1 or duplicate_matrix_forecast[0].get("would_reuse_pending_approval") is not True:
            raise SystemExit(f"Duplicate risky readiness matrix missed per-action reuse forecast: {duplicate_risky_matrix.metadata}")
        assert_execution_readiness_handoff(duplicate_risky_matrix.metadata, "Duplicate risky execution readiness matrix")

        duplicate_risky_argument = runtime.registry.get("argument_contract_packet").handler({"request": "run command python3 --version"})
        print("[ok] direct duplicate risky argument_contract_packet forecast")
        print(duplicate_risky_argument.output[:1100])
        print()
        if (
            duplicate_risky_argument.metadata.get("forecast_queue_before") != 1
            or duplicate_risky_argument.metadata.get("forecast_queue_after_if_sent") != 1
            or duplicate_risky_argument.metadata.get("forecast_queue_delta_if_sent") != 0
            or duplicate_risky_argument.metadata.get("forecast_new_approvals") != 0
            or duplicate_risky_argument.metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Duplicate risky argument contract missed approval reuse forecast: {duplicate_risky_argument.metadata}")
        if "Approval queue forecast:" not in duplicate_risky_argument.output or "would queue new approvals if sent: 0" not in duplicate_risky_argument.output or "would reuse pending approval ids: 1" not in duplicate_risky_argument.output:
            raise SystemExit("Duplicate risky argument contract did not render approval reuse forecast.")
        duplicate_argument_forecast = duplicate_risky_argument.metadata.get("approval_queue_forecast") or []
        if not duplicate_argument_forecast or duplicate_argument_forecast[0].get("existing_approval_id") != 1 or duplicate_argument_forecast[0].get("would_reuse_pending_approval") is not True:
            raise SystemExit(f"Duplicate risky argument contract missed per-action reuse forecast: {duplicate_risky_argument.metadata}")
        assert_argument_contract_handoff(duplicate_risky_argument.metadata, "Duplicate risky argument contract")

        duplicate_risky_verification = runtime.registry.get("verification_packet").handler({"request": "run command python3 --version"})
        print("[ok] direct duplicate risky verification_packet forecast")
        print(duplicate_risky_verification.output[:1100])
        print()
        if (
            duplicate_risky_verification.metadata.get("forecast_queue_before") != 1
            or duplicate_risky_verification.metadata.get("forecast_queue_after_if_sent") != 1
            or duplicate_risky_verification.metadata.get("forecast_queue_delta_if_sent") != 0
            or duplicate_risky_verification.metadata.get("forecast_new_approvals") != 0
            or duplicate_risky_verification.metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Duplicate risky verification packet missed approval reuse forecast: {duplicate_risky_verification.metadata}")
        if "Approval queue forecast:" not in duplicate_risky_verification.output or "would queue new approvals if sent: 0" not in duplicate_risky_verification.output or "would reuse pending approval ids: 1" not in duplicate_risky_verification.output:
            raise SystemExit("Duplicate risky verification packet did not render approval reuse forecast.")
        duplicate_verification_forecast = duplicate_risky_verification.metadata.get("approval_queue_forecast") or []
        if not duplicate_verification_forecast or duplicate_verification_forecast[0].get("existing_approval_id") != 1 or duplicate_verification_forecast[0].get("would_reuse_pending_approval") is not True:
            raise SystemExit(f"Duplicate risky verification packet missed per-action reuse forecast: {duplicate_risky_verification.metadata}")
        assert_verification_packet_handoff(duplicate_risky_verification.metadata, "Duplicate risky verification packet")

        duplicate_risky_acceptance = runtime.registry.get("execution_acceptance_gate").handler(
            {
                "request": "run command python3 --version",
                "evidence": "approval packet viewed and recent tool run ok",
                "tests": "smoke test passed",
                "rollback": "stop on non-zero exit",
            }
        )
        print("[ok] direct duplicate risky execution_acceptance_gate forecast")
        print(duplicate_risky_acceptance.output[:1100])
        print()
        if (
            duplicate_risky_acceptance.metadata.get("forecast_queue_before") != 1
            or duplicate_risky_acceptance.metadata.get("forecast_queue_after_if_sent") != 1
            or duplicate_risky_acceptance.metadata.get("forecast_queue_delta_if_sent") != 0
            or duplicate_risky_acceptance.metadata.get("forecast_new_approvals") != 0
            or duplicate_risky_acceptance.metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Duplicate risky acceptance gate missed approval reuse forecast: {duplicate_risky_acceptance.metadata}")
        if "would queue new approvals if sent: 0" not in duplicate_risky_acceptance.output or "would reuse pending approval ids: 1" not in duplicate_risky_acceptance.output:
            raise SystemExit("Duplicate risky acceptance gate did not render approval reuse forecast.")
        duplicate_acceptance_forecast = duplicate_risky_acceptance.metadata.get("approval_queue_forecast") or []
        if not duplicate_acceptance_forecast or duplicate_acceptance_forecast[0].get("existing_approval_id") != 1 or duplicate_acceptance_forecast[0].get("would_reuse_pending_approval") is not True:
            raise SystemExit(f"Duplicate risky acceptance gate missed per-action reuse forecast: {duplicate_risky_acceptance.metadata}")
        assert_execution_acceptance_handoff(duplicate_risky_acceptance.metadata, "Duplicate risky execution acceptance")

        held_governor = runtime.registry.get("execution_governor_packet").handler({"request": "what time is it"})
        print("[ok] direct approval-held execution_governor_packet")
        print(held_governor.output[:1100])
        print()
        if held_governor.metadata.get("governor_verdict") != "APPROVAL_HELD_REVIEW_REQUIRED":
            raise SystemExit(f"Low-risk governor should hold behind approval-held review: {held_governor.metadata}")
        if held_governor.metadata.get("route") != "approval_held_review" or held_governor.metadata.get("safe_to_execute_now") is not False or held_governor.metadata.get("can_auto_run") is not False:
            raise SystemExit(f"Approval-held governor missed hold metadata: {held_governor.metadata}")
        if held_governor.metadata.get("recovery_debt_visible") is not True:
            raise SystemExit(f"Approval-held governor should expose visible review debt: {held_governor.metadata}")
        if held_governor.metadata.get("governor_preview_allowed_with_recovery_debt") is not True:
            raise SystemExit(f"Approval-held governor should allow the read-only governor preview while review debt is visible: {held_governor.metadata}")
        if held_governor.metadata.get("recovery_closure_blocks_governed_execution") is not True:
            raise SystemExit(f"Approval-held governor should mark governed execution blocked by approval review: {held_governor.metadata}")
        required_commands = held_governor.metadata.get("recovery_closure_required_commands", [])
        for expected in held_approval_commands:
            if expected not in required_commands:
                raise SystemExit(f"Approval-held governor missed required command {expected!r}: {held_governor.metadata}")
        for forbidden in forbidden_held_recovery_commands:
            if forbidden in required_commands:
                raise SystemExit(f"Approval-held governor should not recover approval-held run via {forbidden!r}: {held_governor.metadata}")
        assert_recovery_next_required(held_governor.metadata, held_governor.output, "Approval-held governor")
        if held_governor.metadata.get("recovery_closure_missing") != ["approval_review"]:
            raise SystemExit(f"Approval-held governor should request approval review, not recovery proof: {held_governor.metadata}")
        if held_governor.metadata.get("recent_failed_runs") != 0 or held_governor.metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Approval-held governor should count approval-held rows separately from failures: {held_governor.metadata}")
        if "approval-held review state:" not in held_governor.output or "approval-held review command queue:" not in held_governor.output:
            raise SystemExit("Approval-held governor did not render review state and command queue.")
        if "approval review blocks governed execution: yes" not in held_governor.output or "approval review blocks this governor preview: no" not in held_governor.output:
            raise SystemExit("Approval-held governor did not render governed-execution vs read-only-governor review semantics.")
        if "RECOVERY_CLOSURE_REQUIRED" in held_governor.output or "Execution health recovery closure:" in held_governor.output:
            raise SystemExit(f"Approval-held governor should not label held approval as recovery closure: {held_governor.output}")
        assert_learning_debt(held_governor.metadata, held_governor.output, "Approval-held governor", state="LEARNING_LOOP_HAS_APPROVAL_HELD_CONTEXT", blocks=False)
        assert_frontdoor_handoff(held_governor.metadata, "Approval-held governor", "execution_governor")
        governor_handoff = held_governor.metadata.get("execution_governor_handoff") or {}
        for field in ["governor_verdict", "route", "next_command", "can_auto_run", "safe_to_execute_now", "reason", "matched_risks", "approval_required", "approval_review_required", "planned_actions", "exact_arguments_ready", "ambiguous_risky_order", "unknown_tools", "recovery_debt_visible", "governor_preview_allowed_with_recovery_debt", "recovery_closure_blocks_governed_execution", "recovery_closure_state", "recovery_closure_ready_to_retry"]:
            if governor_handoff.get(field) != held_governor.metadata.get(field):
                raise SystemExit(f"Approval-held governor handoff missed {field} parity: {governor_handoff} vs {held_governor.metadata}")
        if governor_handoff.get("planned_action_count") != held_governor.metadata.get("planned_action_count") or governor_handoff.get("pending_approval_count") != held_governor.metadata.get("pending_approvals"):
            raise SystemExit(f"Approval-held governor handoff missed count parity: {governor_handoff} vs {held_governor.metadata}")
        if governor_handoff.get("recovery_closure_proof_queue") != held_governor.metadata.get("recovery_closure_proof_queue"):
            raise SystemExit(f"Approval-held governor handoff missed recovery proof queue: {governor_handoff} vs {held_governor.metadata}")
        if governor_handoff.get("execution_learning_proof_queue") != held_governor.metadata.get("execution_learning_proof_queue"):
            raise SystemExit(f"Approval-held governor handoff missed learning proof queue: {governor_handoff} vs {held_governor.metadata}")
        if governor_handoff.get("recovery_closure_proof_queue_count") != len(required_commands) or governor_handoff.get("recovery_closure_next_proof_command") != held_governor.metadata.get("recovery_closure_next_required_command"):
            raise SystemExit(f"Approval-held governor handoff missed recovery proof queue parity: {governor_handoff} vs {held_governor.metadata}")

        held_intake = runtime.registry.get("command_intake_packet").handler({"request": "what time is it"})
        print("[ok] direct approval-held command_intake_packet")
        print(held_intake.output[:1100])
        print()
        if held_intake.metadata.get("intake_state") != "approval_held_review_required" or held_intake.metadata.get("route") != "approval_held_review":
            raise SystemExit(f"Low-risk intake should hold behind approval-held review: {held_intake.metadata}")
        if held_intake.metadata.get("safe_to_execute_now") is not False or held_intake.metadata.get("can_auto_run") is not False:
            raise SystemExit(f"Approval-held intake missed safe/can-auto-run blocker metadata: {held_intake.metadata}")
        if held_intake.metadata.get("recovery_closure_blocks_auto_execution") is not True:
            raise SystemExit(f"Approval-held intake missed blocker metadata: {held_intake.metadata}")
        if held_intake.metadata.get("recovery_debt_visible") is not True:
            raise SystemExit(f"Approval-held intake should expose visible review debt: {held_intake.metadata}")
        if held_intake.metadata.get("command_intake_preview_allowed_with_recovery_debt") is not True:
            raise SystemExit(f"Approval-held intake should allow the read-only intake preview while review debt is visible: {held_intake.metadata}")
        if held_intake.metadata.get("recovery_closure_blocks_current_order") is not True:
            raise SystemExit(f"Approval-held intake should mark the proposed order blocked by approval review: {held_intake.metadata}")
        for expected in held_approval_commands:
            if expected not in held_intake.metadata.get("recovery_closure_required_commands", []):
                raise SystemExit(f"Approval-held intake missed required command {expected!r}: {held_intake.metadata}")
        for forbidden in forbidden_held_recovery_commands:
            if forbidden in held_intake.metadata.get("recovery_closure_required_commands", []):
                raise SystemExit(f"Approval-held intake should not recover approval-held run via {forbidden!r}: {held_intake.metadata}")
        assert_recovery_next_required(held_intake.metadata, held_intake.output, "Approval-held intake")
        if "approval review blocks proposed order: yes" not in held_intake.output or "approval review blocks this intake preview: no" not in held_intake.output:
            raise SystemExit("Approval-held intake did not render proposed-order vs read-only-intake review semantics.")
        if "Approval-held execution review:" not in held_intake.output or "approval-held review blocks auto-run: yes" not in held_intake.output:
            raise SystemExit("Approval-held intake did not render approval review details.")
        if "RECOVERY_CLOSURE_REQUIRED" in held_intake.output or "Execution health recovery closure:" in held_intake.output:
            raise SystemExit(f"Approval-held intake should not label held approval as recovery closure: {held_intake.output}")
        assert_learning_debt(held_intake.metadata, held_intake.output, "Approval-held intake", state="LEARNING_LOOP_HAS_APPROVAL_HELD_CONTEXT", blocks=False)
        assert_frontdoor_handoff(held_intake.metadata, "Approval-held intake", "command_intake")
        intake_handoff = held_intake.metadata.get("command_intake_handoff") or {}
        for field in ["intake_state", "route", "next_command", "can_auto_run", "safe_to_execute_now", "matched_risks", "approval_required", "approval_review_required", "planned_actions", "exact_arguments_ready", "ambiguous_risky_order", "unknown_tools", "recovery_debt_visible", "command_intake_preview_allowed_with_recovery_debt", "recovery_closure_blocks_current_order", "recovery_closure_state", "recovery_closure_ready_to_retry"]:
            if intake_handoff.get(field) != held_intake.metadata.get(field):
                raise SystemExit(f"Command intake handoff missed {field} parity: {intake_handoff} vs {held_intake.metadata}")
        if intake_handoff.get("planned_action_count") != held_intake.metadata.get("planned_action_count") or intake_handoff.get("pending_approval_count") != held_intake.metadata.get("pending_approvals"):
            raise SystemExit(f"Command intake handoff missed count parity: {intake_handoff} vs {held_intake.metadata}")

        held_dispatch = runtime.registry.get("dispatch_decision_packet").handler({"request": "what time is it"})
        print("[ok] direct approval-held dispatch_decision_packet")
        print(held_dispatch.output[:1100])
        print()
        if held_dispatch.metadata.get("decision") != "APPROVAL_HELD_REVIEW_REQUIRED" or held_dispatch.metadata.get("route") != "approval_held_review":
            raise SystemExit(f"Low-risk dispatch should hold behind approval-held review: {held_dispatch.metadata}")
        if held_dispatch.metadata.get("safe_to_execute_now") is not False or held_dispatch.metadata.get("can_auto_run") is not False:
            raise SystemExit(f"Approval-held dispatch missed safe/can-auto-run blocker metadata: {held_dispatch.metadata}")
        if held_dispatch.metadata.get("recovery_closure_blocks_auto_execution") is not True:
            raise SystemExit(f"Approval-held dispatch missed blocker metadata: {held_dispatch.metadata}")
        if held_dispatch.metadata.get("recovery_debt_visible") is not True:
            raise SystemExit(f"Approval-held dispatch should expose visible review debt: {held_dispatch.metadata}")
        if held_dispatch.metadata.get("dispatch_preview_allowed_with_recovery_debt") is not True:
            raise SystemExit(f"Approval-held dispatch should allow the read-only dispatch preview while review debt is visible: {held_dispatch.metadata}")
        if held_dispatch.metadata.get("recovery_closure_blocks_current_dispatch") is not True:
            raise SystemExit(f"Approval-held dispatch should mark the proposed dispatch blocked by approval review: {held_dispatch.metadata}")
        for expected in held_approval_commands:
            if expected not in held_dispatch.metadata.get("recovery_closure_required_commands", []):
                raise SystemExit(f"Approval-held dispatch missed required command {expected!r}: {held_dispatch.metadata}")
        for forbidden in forbidden_held_recovery_commands:
            if forbidden in held_dispatch.metadata.get("recovery_closure_required_commands", []):
                raise SystemExit(f"Approval-held dispatch should not recover approval-held run via {forbidden!r}: {held_dispatch.metadata}")
        assert_recovery_next_required(held_dispatch.metadata, held_dispatch.output, "Approval-held dispatch")
        if "approval review blocks proposed dispatch: yes" not in held_dispatch.output or "approval review blocks this dispatch preview: no" not in held_dispatch.output:
            raise SystemExit("Approval-held dispatch did not render proposed-dispatch vs read-only-dispatch review semantics.")
        if "Approval-held execution review:" not in held_dispatch.output or "Dispatch decision: APPROVAL_HELD_REVIEW_REQUIRED" not in held_dispatch.output:
            raise SystemExit("Approval-held dispatch did not render approval review decision/details.")
        if "RECOVERY_CLOSURE_REQUIRED" in held_dispatch.output or "Execution health recovery closure:" in held_dispatch.output:
            raise SystemExit(f"Approval-held dispatch should not label held approval as recovery closure: {held_dispatch.output}")
        assert_learning_debt(held_dispatch.metadata, held_dispatch.output, "Approval-held dispatch", state="LEARNING_LOOP_HAS_APPROVAL_HELD_CONTEXT", blocks=False)
        assert_frontdoor_handoff(held_dispatch.metadata, "Approval-held dispatch", "dispatch_decision")
        dispatch_handoff = held_dispatch.metadata.get("dispatch_decision_handoff") or {}
        for field in ["decision", "route", "can_auto_run", "safe_to_execute_now", "primary_command", "matched_risks", "planned_actions", "approval_required", "ambiguous_risky_order", "unknown_tools", "recovery_debt_visible", "dispatch_preview_allowed_with_recovery_debt", "recovery_closure_blocks_current_dispatch", "recovery_closure_state", "recovery_closure_ready_to_retry"]:
            if dispatch_handoff.get(field) != held_dispatch.metadata.get(field):
                raise SystemExit(f"Dispatch handoff missed {field} parity: {dispatch_handoff} vs {held_dispatch.metadata}")
        if dispatch_handoff.get("planned_action_count") != held_dispatch.metadata.get("planned_action_count") or dispatch_handoff.get("pending_approval_count") != held_dispatch.metadata.get("pending_approvals"):
            raise SystemExit(f"Dispatch handoff missed count parity: {dispatch_handoff} vs {held_dispatch.metadata}")

        held_matrix = runtime.registry.get("execution_readiness_matrix").handler({"request": "what time is it"})
        print("[ok] direct approval-held execution_readiness_matrix")
        print(held_matrix.output[:1100])
        print()
        if held_matrix.metadata.get("verdict") != "APPROVAL_HELD_REVIEW_REQUIRED":
            raise SystemExit(f"Low-risk readiness matrix should hold behind approval-held review: {held_matrix.metadata}")
        if held_matrix.metadata.get("recovery_closure_blocks_auto_execution") is not True:
            raise SystemExit(f"Approval-held readiness matrix missed blocker metadata: {held_matrix.metadata}")
        if held_matrix.metadata.get("recovery_debt_visible") is not True:
            raise SystemExit(f"Approval-held readiness matrix should expose visible review debt: {held_matrix.metadata}")
        if held_matrix.metadata.get("readiness_matrix_preview_allowed_with_recovery_debt") is not True:
            raise SystemExit(f"Approval-held readiness matrix should allow the read-only matrix preview while review debt is visible: {held_matrix.metadata}")
        if held_matrix.metadata.get("recovery_closure_blocks_matrix_execution") is not True:
            raise SystemExit(f"Approval-held readiness matrix should mark matrix execution blocked by approval review: {held_matrix.metadata}")
        if held_matrix.metadata.get("failed_or_blocked_runs") != 0 or held_matrix.metadata.get("recent_failed_runs") != 0:
            raise SystemExit(f"Approval-held readiness matrix should not count approval-held rows as failures: {held_matrix.metadata}")
        if held_matrix.metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Approval-held readiness matrix missed approval-held count: {held_matrix.metadata}")
        for expected in held_approval_commands:
            if expected not in held_matrix.metadata.get("recovery_closure_required_commands", []):
                raise SystemExit(f"Approval-held readiness matrix missed approval-review command {expected!r}: {held_matrix.metadata}")
        for forbidden in forbidden_held_recovery_commands:
            if forbidden in held_matrix.metadata.get("recovery_closure_required_commands", []):
                raise SystemExit(f"Approval-held readiness matrix should not recover approval-held run via {forbidden!r}: {held_matrix.metadata}")
        assert_recovery_next_required(held_matrix.metadata, held_matrix.output, "Approval-held readiness matrix")
        if "approval review blocks matrix execution: yes" not in held_matrix.output or "approval review blocks this readiness matrix: no" not in held_matrix.output:
            raise SystemExit("Approval-held readiness matrix did not render matrix-execution vs read-only-matrix review semantics.")
        if "Approval-held execution review:" not in held_matrix.output:
            raise SystemExit("Approval-held readiness matrix did not render approval review details.")
        if "RECOVERY_CLOSURE_REQUIRED" in held_matrix.output or "Execution health recovery closure:" in held_matrix.output:
            raise SystemExit(f"Approval-held readiness matrix should not label held approval as recovery closure: {held_matrix.output}")
        if "0 failed/blocked and 1 approval-held recent run(s)" not in held_matrix.output:
            raise SystemExit(f"Approval-held readiness matrix did not render separated recovery counts: {held_matrix.output}")
        if "recent failed/blocked runs: 0" not in held_matrix.output or "recent approval-held runs: 1" not in held_matrix.output:
            raise SystemExit(f"Approval-held readiness matrix missed separated recovery detail lines: {held_matrix.output}")
        assert_learning_debt(held_matrix.metadata, held_matrix.output, "Approval-held readiness matrix", state="LEARNING_LOOP_HAS_APPROVAL_HELD_CONTEXT", blocks=False)
        assert_execution_readiness_handoff(held_matrix.metadata, "Approval-held execution readiness matrix")

    with TemporaryDirectory(prefix="jarvis-autonomy-learning-context-") as temp:
        runtime = make_temp_runtime(Path(temp))
        run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "remember",
            "LOCAL_SAFE",
            True,
            False,
            "Remembered a useful local fact.",
            metadata={"fact": "front door learning context fixture"},
        )
        frontdoor_cases = [
            ("execution_governor_packet", "governor_verdict", {"AUTO_SAFE_READY", "CHAT_READY"}),
            ("command_intake_packet", "intake_state", {"ready_for_policy_checked_dispatch", "ready_for_chat"}),
            ("dispatch_decision_packet", "decision", {"AUTO_RUN_LOCAL_SAFE_IF_SENT_FOR_REAL", "ANSWER_IN_CHAT"}),
            ("execution_readiness_matrix", "verdict", {"READY_FOR_AUTO_SAFE_TOOL_ROUTE", "READY_FOR_CHAT_BRAIN"}),
            ("command_cockpit_packet", "cockpit_verdict", {"READY_FOR_SAFE_DISPATCH", "PREFLIGHT_OR_CLARIFICATION_REQUIRED"}),
        ]
        for tool_name, verdict_key, allowed in frontdoor_cases:
            packet = runtime.registry.get(tool_name).handler({"request": "what time is it"})
            print(f"[ok] direct learning-context {tool_name}")
            print(packet.output[:1000])
            print()
            if packet.metadata.get(verdict_key) not in allowed:
                raise SystemExit(f"{tool_name} should not block normal routing for successful action learning context: {packet.metadata}")
            if packet.metadata.get("safe_to_execute_now", packet.metadata.get("can_auto_run", True)) is False and tool_name != "execution_readiness_matrix":
                raise SystemExit(f"{tool_name} should remain safe for low-risk routing with successful learning context: {packet.metadata}")
            assert_learning_debt(packet.metadata, packet.output, f"Learning-context {tool_name}", state="LEARNING_REVIEW_REQUIRED", run_id=run_id)
            if tool_name == "command_cockpit_packet":
                assert_frontdoor_handoff(packet.metadata, "Learning-context command cockpit", "command_cockpit")
                cockpit_handoff = packet.metadata.get("command_cockpit_handoff") or {}
                for field in ["cockpit_verdict", "next_command", "can_auto_run", "approval_required", "recovery_debt_visible", "cockpit_preview_allowed_with_recovery_debt", "recovery_closure_blocks_cockpit_execution", "recovery_closure_state", "recovery_closure_ready_to_retry", "execution_learning_blocks_completion_claim", "execution_learning_state", "packet_rows", "governor_verdict", "governor_route", "dispatch_decision", "dispatch_route", "readiness_verdict", "verification_verdict"]:
                    if cockpit_handoff.get(field) != packet.metadata.get(field):
                        raise SystemExit(f"Learning-context command cockpit handoff missed {field} parity: {cockpit_handoff} vs {packet.metadata}")
                if cockpit_handoff.get("packet_row_count") != packet.metadata.get("packet_row_count") or cockpit_handoff.get("pending_approval_count") != packet.metadata.get("pending_approvals") or cockpit_handoff.get("required_command_count") != packet.metadata.get("required_command_count"):
                    raise SystemExit(f"Learning-context command cockpit handoff missed count parity: {cockpit_handoff} vs {packet.metadata}")

    with TemporaryDirectory(prefix="jarvis-action-readiness-closure-") as temp:
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
        assert_recovery_snapshot_proof_aliases(runtime, "Recovery-held action readiness")
        held_action = runtime.registry.get("action_readiness_packet").handler({"request": "what time is it"})
        print("[ok] direct recovery-held action_readiness_packet")
        print(held_action.output[:1100])
        print()
        if held_action.metadata.get("recommendation") != "RECOVERY_CLOSURE_REQUIRED" or held_action.metadata.get("route") != "recovery_closure":
            raise SystemExit(f"Low-risk action readiness should hold behind unresolved recovery closure: {held_action.metadata}")
        if held_action.metadata.get("safe_to_execute_now") is not False:
            raise SystemExit(f"Recovery-held action readiness missed safe-to-execute blocker: {held_action.metadata}")
        if held_action.metadata.get("recovery_debt_visible") is not True:
            raise SystemExit(f"Recovery-held action readiness should expose visible recovery debt: {held_action.metadata}")
        if held_action.metadata.get("readiness_preview_allowed_with_recovery_debt") is not True:
            raise SystemExit(f"Recovery-held action readiness should allow the read-only preview while recovery debt is visible: {held_action.metadata}")
        if held_action.metadata.get("recovery_closure_blocks_current_action") is not True:
            raise SystemExit(f"Recovery-held action readiness should mark the proposed action blocked by recovery debt: {held_action.metadata}")
        for expected in [f"verification receipt {run_id}", f"execution recovery packet {run_id}", f"execution learning closure {run_id}", f"after-action learning packet {run_id}"]:
            if expected not in held_action.metadata.get("recovery_closure_required_commands", []):
                raise SystemExit(f"Recovery-held action readiness missed required command {expected!r}: {held_action.metadata}")
        assert_recovery_next_required(held_action.metadata, held_action.output, "Recovery-held action readiness")
        if "recovery debt blocks proposed action: yes" not in held_action.output or "recovery debt blocks this readiness preview: no" not in held_action.output:
            raise SystemExit("Recovery-held action readiness did not render proposed-action vs read-only-preview recovery semantics.")
        if "Execution health recovery closure:" not in held_action.output or "RECOVERY_CLOSURE_REQUIRED" not in held_action.output:
            raise SystemExit("Recovery-held action readiness did not render recovery closure decision/details.")
        assert_action_readiness_handoff(held_action.metadata, "Recovery-held action readiness")


if __name__ == "__main__":
    main()
