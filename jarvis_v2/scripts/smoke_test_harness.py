from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.config import JarvisConfig
from jarvis_v2.scripts.test_runtime import make_temp_runtime
import jarvis_v2.tools.harness as harness_module
from jarvis_v2.tools.harness import (
    _completion_actionable_queue_from_ordered,
    _completion_proof_queue_from_sources,
    _execution_learning_debt_snapshot,
    _execution_proof_handoff_aliases,
    _metadata_bool,
    _metadata_int,
    make_harness_tools,
)
from jarvis_v2.tools.storage import BOOTSTRAP_CHECK_COMMAND, BOOTSTRAP_WRITE_COMMAND, STORAGE_RECOVERY_CHECK_COMMAND


READ_ONLY_FLAGS = [
    "calls_model",
    "calls_external_service",
    "executes_tools",
    "reads_personal_data",
    "reads_private_data",
    "executes_side_effect",
    "external_side_effect",
    "writes_files",
    "writes_database",
    "writes_memory",
    "writes_notes",
    "queues_approval",
    "requires_approval",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "controls_computer",
    "speaks",
    "completes_tasks",
]
NON_WRITE_AUTHORITY_FLAGS = [
    key for key in READ_ONLY_FLAGS if key not in {"writes_files", "writes_database", "writes_memory", "writes_notes"}
]
EXPECTED_FALLBACK_STORAGE_RECOVERY_MODE = "restart_runtime_to_configured_storage"
EXPECTED_FALLBACK_STORAGE_NEXT_ACTION = (
    "restart or reload Jarvis with the configured durable storage envs, then run `storage status`"
)
EXPECTED_RESTART_ONLY_STORAGE_COMMANDS = [
    "storage status",
    STORAGE_RECOVERY_CHECK_COMMAND,
    "storage status",
]


def assert_harness_metadata_bool_is_exact() -> None:
    if _metadata_bool(True) is not True or _metadata_bool(False) is not False:
        raise SystemExit("Harness metadata bool helper should preserve exact bools.")
    for value in ("true", "false", "yes", "0", 1, 0, ["true"], {"value": True}, None):
        if _metadata_bool(value):
            raise SystemExit(f"Harness metadata bool helper accepted malformed truthy value: {value!r}")
    for value in ("false", 0, None):
        if _metadata_bool(value, default=True) is not True:
            raise SystemExit(f"Harness metadata bool helper should preserve conservative default for malformed value: {value!r}")


def assert_harness_packet_aliases_use_default_request() -> None:
    planner = RuleBasedPlanner()
    expected_routes = {
        "command cockpit please": "command_cockpit_packet",
        "show command cockpit": "command_cockpit_packet",
        "show latest command cockpit": "command_cockpit_packet",
        "command intake please": "command_intake_packet",
        "show command intake": "command_intake_packet",
        "dispatch please": "dispatch_decision_packet",
        "dispatch decision please": "dispatch_decision_packet",
        "show dispatch decision": "dispatch_decision_packet",
        "governor please": "execution_governor_packet",
        "execution governor please": "execution_governor_packet",
        "show execution governor": "execution_governor_packet",
        "planner gap please": "planner_gap_packet",
        "show planner gap": "planner_gap_packet",
        "action readiness please": "action_readiness_packet",
        "show action readiness": "action_readiness_packet",
        "execution contract please": "execution_contract",
        "show execution contract": "execution_contract",
        "argument contract please": "argument_contract_packet",
        "verification packet please": "verification_packet",
        "acceptance gate please": "execution_acceptance_gate",
        "execution readiness matrix please": "execution_readiness_matrix",
    }
    for phrase, expected_tool in expected_routes.items():
        plan = planner.plan(phrase)
        action_tools = [action.tool_name for action in plan.actions]
        if action_tools != [expected_tool]:
            raise SystemExit(f"harness packet alias did not route safely: {phrase!r} -> {action_tools!r}")
        args = plan.actions[0].args
        if args.get("request") != "what should Jarvis do next":
            raise SystemExit(f"harness packet alias kept fake request payload: {phrase!r} -> {args!r}")

    explicit = planner.plan("dispatch decision: run command python3 --version")
    if explicit.actions[0].tool_name != "dispatch_decision_packet" or explicit.actions[0].args.get("request") != "run command python3 --version":
        raise SystemExit(f"explicit dispatch request was not preserved: {explicit.actions}")
    structured = planner.plan("acceptance gate: run command python3 --version; tests smoke passed")
    if structured.actions[0].tool_name != "execution_acceptance_gate":
        raise SystemExit(f"structured acceptance gate did not route: {structured.actions}")
    if structured.actions[0].args.get("request") != "run command python3 --version" or structured.actions[0].args.get("tests") != "smoke passed":
        raise SystemExit(f"structured acceptance gate proof fields were not preserved: {structured.actions[0].args}")


def assert_natural_approval_questions_route_to_rehearsal() -> None:
    planner = RuleBasedPlanner()
    expected_routes = {
        "would this need approval: run command python3 --version": "run command python3 --version",
        "would run command python3 --version need approval": "run command python3 --version",
        "does run command python3 --version need approval": "run command python3 --version",
        "will this require approval: send email to Sam saying hi": "send email to Sam saying hi",
        "check approval for run command python3 --version": "run command python3 --version",
        "preview approval requirements for send imessage to Fixture saying hi": "send imessage to Fixture saying hi",
        "can you safely run command python3 --version": "run command python3 --version",
        "is it safe to send email to Sam saying hi": "send email to Sam saying hi",
        "do i need approval to run command python3 --version": "run command python3 --version",
        "do you need approval to run command python3 --version": "run command python3 --version",
        "does jarvis need approval to send email to Sam saying hi": "send email to Sam saying hi",
        "would i need to approve run command python3 --version": "run command python3 --version",
        "would the operator need to approve send email to Sam saying hi": "send email to Sam saying hi",
        "would this need my approval before sending email to Sam saying hi": "send email to Sam saying hi",
        "would running command python3 --version need approval": "run command python3 --version",
        "does sending email to Sam saying hi require approval": "send email to Sam saying hi",
        "approval preview send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_routes.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural approval rehearsal route should plan one action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "action_rehearsal" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural approval rehearsal route misplanned {phrase!r}: {(action.tool_name, action.args)}")

    tool_detail = planner.plan("does run_shell_command need approval")
    if len(tool_detail.actions) != 1 or tool_detail.actions[0].tool_name != "tool_detail" or tool_detail.actions[0].args != {"name": "run_shell_command"}:
        raise SystemExit(f"tool-specific approval question should stay tool_detail: {tool_detail.actions!r}")

    first_person_tool_detail = planner.plan("do i need approval for run_shell_command")
    if (
        len(first_person_tool_detail.actions) != 1
        or first_person_tool_detail.actions[0].tool_name != "tool_detail"
        or first_person_tool_detail.actions[0].args != {"name": "run_shell_command"}
    ):
        raise SystemExit(f"first-person tool approval question should stay tool_detail: {first_person_tool_detail.actions!r}")

    conversation = planner.plan("can you safely talk about your day")
    if conversation.actions:
        raise SystemExit(f"ordinary safety-colored conversation should not route to action rehearsal: {conversation.actions!r}")

    approval_conversation = planner.plan("do i need approval to talk about lunch")
    if approval_conversation.actions:
        raise SystemExit(f"ordinary approval-colored conversation should not route to action rehearsal: {approval_conversation.actions!r}")


def assert_harness_layer_contract(metadata: dict, label: str) -> None:
    rows = metadata.get("harness_layer_contract_rows") or []
    titles = metadata.get("harness_layer_contract_titles") or []
    expected_titles = ["Intelligence", "Engine", "Agents", "Tools+Memory", "Learning"]
    if titles != expected_titles:
        raise SystemExit(f"{label} layer contract titles drifted: {metadata}")
    if metadata.get("harness_layer_contract_row_count") != len(rows) or len(rows) != len(expected_titles):
        raise SystemExit(f"{label} layer contract row count diverged: {metadata}")
    if metadata.get("harness_layer_contract_non_authorizing") is not True:
        raise SystemExit(f"{label} layer contract should be explicitly non-authorizing: {metadata}")
    if any(row.get("non_authorizing") is not True for row in rows):
        raise SystemExit(f"{label} layer rows should all be non-authorizing: {metadata}")
    status_counts = {
        "ready": metadata.get("harness_layer_contract_ready_count"),
        "partial": metadata.get("harness_layer_contract_partial_count"),
        "missing": metadata.get("harness_layer_contract_missing_count"),
        "unmeasured": metadata.get("harness_layer_contract_unmeasured_count"),
    }
    for status, count in status_counts.items():
        if count != len([row for row in rows if row.get("status") == status]):
            raise SystemExit(f"{label} layer {status} count diverged: {metadata}")
    if sum(int(count or 0) for count in status_counts.values()) != len(rows):
        raise SystemExit(f"{label} layer status counts should cover every row: {metadata}")
    for row in rows:
        if not row.get("key") or not row.get("title") or not row.get("contract") or not row.get("boundary"):
            raise SystemExit(f"{label} layer row missed required fields: {row}")
        evidence = row.get("evidence_tools") or []
        present = row.get("present_evidence") or []
        missing = row.get("missing_evidence") or []
        if row.get("evidence_tool_count") != len(evidence):
            raise SystemExit(f"{label} layer evidence count diverged: {row}")
        if row.get("present_evidence_count") != len(present):
            raise SystemExit(f"{label} layer present evidence count diverged: {row}")
        if row.get("missing_evidence_count") != len(missing):
            raise SystemExit(f"{label} layer missing evidence count diverged: {row}")
        if sorted(present + missing) != sorted(evidence):
            raise SystemExit(f"{label} layer present/missing evidence should partition evidence tools: {row}")
        if row.get("title") == "Intelligence" and "Model drafts do not authorize execution" not in row.get("boundary", ""):
            raise SystemExit(f"{label} intelligence boundary lost non-authorizing language: {row}")
        if row.get("title") == "Agents" and "not companion personas" not in row.get("boundary", ""):
            raise SystemExit(f"{label} agents boundary lost anti-persona language: {row}")
        if row.get("title") == "Learning" and "not self-approval" not in row.get("boundary", ""):
            raise SystemExit(f"{label} learning boundary lost self-approval language: {row}")


def assert_natural_risk_preflight_routes_before_broad_tools() -> None:
    planner = RuleBasedPlanner()
    expected_preflights = {
        "risk preview send email to Sam saying hi": "send email to Sam saying hi",
        "safety preview send email to Sam saying hi": "send email to Sam saying hi",
        "risk check send email to Sam saying hi": "send email to Sam saying hi",
        "safety check send email to Sam saying hi": "send email to Sam saying hi",
        "preflight send email to Sam saying hi": "send email to Sam saying hi",
        "preflight check send email to Sam saying hi": "send email to Sam saying hi",
        "check before sending email to Sam saying hi": "send email to Sam saying hi",
        "check before run command python3 --version": "run command python3 --version",
        "check before you send email to Sam saying hi": "send email to Sam saying hi",
        "risk preview weather Seoul": "weather Seoul",
    }
    for phrase, expected_request in expected_preflights.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural risk preflight should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "risk_preflight" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural risk preflight misplanned {phrase!r}: {(action.tool_name, action.args)}")

    tool_preview = planner.plan("approval preview run_shell_command")
    if len(tool_preview.actions) != 1 or tool_preview.actions[0].tool_name != "tool_detail" or tool_preview.actions[0].args != {"name": "run_shell_command"}:
        raise SystemExit(f"approval preview for an explicit tool should stay tool_detail: {tool_preview.actions!r}")
    email_read = planner.plan("email please")
    if len(email_read.actions) != 1 or email_read.actions[0].tool_name != "read_emails":
        raise SystemExit(f"normal email read route should stay read_emails: {email_read.actions!r}")


def assert_natural_action_readiness_and_governor_routes() -> None:
    planner = RuleBasedPlanner()
    expected_readiness = {
        "go no go send email to Sam saying hi": "send email to Sam saying hi",
        "go/no-go send email to Sam saying hi": "send email to Sam saying hi",
        "go or no go send email to Sam saying hi": "send email to Sam saying hi",
        "is this a go send email to Sam saying hi": "send email to Sam saying hi",
        "ready to send email to Sam saying hi": "send email to Sam saying hi",
        "readiness check send email to Sam saying hi": "send email to Sam saying hi",
        "action readiness send email to Sam saying hi": "send email to Sam saying hi",
        "go no go weather Seoul": "weather Seoul",
    }
    for phrase, expected_request in expected_readiness.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural action readiness should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "action_readiness_packet" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural action readiness misplanned {phrase!r}: {(action.tool_name, action.args)}")

    expected_governor = {
        "govern send email to Sam saying hi": "send email to Sam saying hi",
        "governor check send email to Sam saying hi": "send email to Sam saying hi",
        "execution governor send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_governor.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural governor route should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "execution_governor_packet" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural governor route misplanned {phrase!r}: {(action.tool_name, action.args)}")

    conversation = planner.plan("ready to talk about lunch")
    if conversation.actions:
        raise SystemExit(f"conversation readiness phrasing should stay unclaimed: {conversation.actions!r}")
    safe_question = planner.plan("can jarvis safely send email to Sam saying hi")
    if len(safe_question.actions) != 1 or safe_question.actions[0].tool_name != "action_rehearsal":
        raise SystemExit(f"natural safe-action question should stay action_rehearsal: {safe_question.actions!r}")


def assert_natural_verification_packet_routes() -> None:
    planner = RuleBasedPlanner()
    expected_verification = {
        "how do we verify send email to Sam saying hi": "send email to Sam saying hi",
        "what evidence proves send email to Sam saying hi": "send email to Sam saying hi",
        "what evidence would prove send email to Sam saying hi": "send email to Sam saying hi",
        "what proof proves send email to Sam saying hi": "send email to Sam saying hi",
        "proof plan send email to Sam saying hi": "send email to Sam saying hi",
        "evidence plan send email to Sam saying hi": "send email to Sam saying hi",
        "verification plan send email to Sam saying hi": "send email to Sam saying hi",
        "how do we prove weather Seoul": "weather Seoul",
    }
    for phrase, expected_request in expected_verification.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural verification packet should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "verification_packet" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural verification packet misplanned {phrase!r}: {(action.tool_name, action.args)}")

    generic_proof = planner.plan("prove photosynthesis")
    if generic_proof.actions:
        raise SystemExit(f"generic prove phrase should stay unclaimed: {generic_proof.actions!r}")


def assert_natural_acceptance_gate_routes() -> None:
    planner = RuleBasedPlanner()
    expected_acceptance = {
        "can we call send email to Sam saying hi done": "send email to Sam saying hi",
        "can jarvis call send email to Sam saying hi done": "send email to Sam saying hi",
        "can i mark send email to Sam saying hi done": "send email to Sam saying hi",
        "should we mark send email to Sam saying hi done": "send email to Sam saying hi",
        "should jarvis call send email to Sam saying hi done": "send email to Sam saying hi",
        "is send email to Sam saying hi accepted as done": "send email to Sam saying hi",
        "is send email to Sam saying hi ready to accept": "send email to Sam saying hi",
        "acceptance check send email to Sam saying hi": "send email to Sam saying hi",
        "done gate send email to Sam saying hi": "send email to Sam saying hi",
        "completion acceptance send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_acceptance.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural acceptance gate should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "execution_acceptance_gate" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural acceptance gate misplanned {phrase!r}: {(action.tool_name, action.args)}")

    structured = planner.plan("can we call this done: send email to Sam saying hi; tests smoke passed; evidence receipt 2; recovery none")
    if len(structured.actions) != 1:
        raise SystemExit(f"structured natural acceptance gate should plan one action: {structured.actions!r}")
    structured_action = structured.actions[0]
    expected_args = {
        "request": "send email to Sam saying hi",
        "tests": "smoke passed",
        "evidence": "receipt 2",
        "rollback": "none",
    }
    if structured_action.tool_name != "execution_acceptance_gate" or structured_action.args != expected_args:
        raise SystemExit(f"structured natural acceptance gate lost proof fields: {(structured_action.tool_name, structured_action.args)}")

    conversation = planner.plan("can we call lunch done")
    if conversation.actions:
        raise SystemExit(f"conversation acceptance phrasing should stay unclaimed: {conversation.actions!r}")


def assert_natural_completion_claim_routes() -> None:
    planner = RuleBasedPlanner()
    expected_claims = {
        "are we done?": {},
        "can I claim Jarvis is finished": {},
        "should we call it done": {},
        "can this be marked complete": {},
        "can this be called complete": {},
        "completion claim please": {},
        "claim done please": {},
        "mark Jarvis done please": {},
        "can this be claimed done: Finish Jarvis V2": {"objective": "Finish Jarvis V2"},
    }
    for phrase, expected_args in expected_claims.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural completion claim should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "completion_claim_gate" or action.args != expected_args:
            raise SystemExit(f"natural completion claim misplanned {phrase!r}: {(action.tool_name, action.args)}")

    expected_audits = {
        "prove this is done": {},
        "prove Jarvis is done": {},
        "prove we finished Jarvis": {},
        "prove completion please": {},
        "completion proof please": {},
        "what is left to finish Jarvis": {},
        "what is left before Jarvis is done": {},
        "what do we need to finish Jarvis": {},
        "what do we need before Jarvis is done": {},
        "what proof is missing for Jarvis": {},
        "what evidence is missing for Jarvis": {},
        "what proof do we need for Jarvis": {},
        "what proof do we need before Jarvis is ready": {},
        "what evidence do we need before Jarvis is done": {},
    }
    for phrase, expected_args in expected_audits.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural completion proof should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "completion_audit_packet" or action.args != expected_args:
            raise SystemExit(f"natural completion proof misplanned {phrase!r}: {(action.tool_name, action.args)}")

    for phrase in [
        "what proof is missing for photosynthesis",
        "what evidence do we need before dinner is ready",
    ]:
        plan = planner.plan(phrase)
        if plan.actions and plan.actions[0].tool_name == "completion_audit_packet":
            raise SystemExit(f"natural completion proof should not claim non-Jarvis proof gap: {phrase!r} -> {plan.actions!r}")

    task_completion = planner.plan("mark task 1 done")
    if len(task_completion.actions) != 1 or task_completion.actions[0].tool_name != "complete_task":
        raise SystemExit(f"task completion route should not be stolen by completion claim gate: {task_completion.actions!r}")


def assert_natural_harness_completion_assessment_routes() -> None:
    planner = RuleBasedPlanner()
    for phrase in [
        "how close is Jarvis",
        "how close is Jarvis?",
        "how close is Jarvis to done",
        "how close are we to finishing Jarvis",
        "how close is the harness",
    ]:
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural harness completion assessment should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "harness_completion_assessment" or action.args != {}:
            raise SystemExit(f"natural harness completion assessment misplanned {phrase!r}: {(action.tool_name, action.args)}")

    for phrase in [
        "how close are we",
        "how close is dinner",
    ]:
        plan = planner.plan(phrase)
        if plan.actions and plan.actions[0].tool_name == "harness_completion_assessment":
            raise SystemExit(f"natural harness completion assessment should not claim generic phrasing: {phrase!r} -> {plan.actions!r}")


def assert_natural_completion_next_proof_routes() -> None:
    planner = RuleBasedPlanner()
    expected_next_proof = {
        "what next proof": {},
        "next proof please": {},
        "next proof?": {},
        "next proof for Jarvis?": {"objective": "for Jarvis"},
        "what proof should I run next": {},
        "what should I prove next": {},
        "what evidence next": {},
        "what evidence should we gather next": {},
        "what verification next": {},
        "what verification should we run next": {},
        "next completion evidence": {},
        "next completion check": {},
        "what should we verify next for completion": {},
        "what is the next step to finish Jarvis": {},
        "next step to complete the harness": {},
    }
    for phrase, expected_args in expected_next_proof.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural completion next proof should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "completion_next_proof_packet" or action.args != expected_args:
            raise SystemExit(f"natural completion next proof misplanned {phrase!r}: {(action.tool_name, action.args)}")

    expected_handoff = {
        "operator next command please": {},
        "what should the operator do next?": {},
        "proof handoff please": {},
        "operator next command for Jarvis?": {"objective": "for Jarvis"},
    }
    for phrase, expected_args in expected_handoff.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural operator handoff should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "operator_handoff_packet" or action.args != expected_args:
            raise SystemExit(f"natural operator handoff misplanned {phrase!r}: {(action.tool_name, action.args)}")


def assert_natural_harness_readiness_digest_routes() -> None:
    planner = RuleBasedPlanner()
    expected_readiness = {
        "harness readiness please": {},
        "readiness digest please": {},
        "readiness digest?": {},
        "what is the readiness digest?": {},
        "show readiness digest": {},
        "show harness readiness": {},
        "readiness summary please": {},
        "what is Jarvis readiness?": {},
        "what is blocking completion?": {},
        "completion blockers please": {},
        "what blockers remain?": {},
        "what is blocking Jarvis?": {},
        "harness blockers please": {},
        "status digest please": {},
        "what is missing for Jarvis?": {},
        "what is Jarvis missing": {},
        "what's Jarvis missing?": {},
        "what's missing for the Jarvis harness?": {},
        "why is Jarvis not ready": {},
        "why isn't Jarvis ready": {},
        "why is the harness not ready": {},
        "what would make Jarvis ready": {},
        "what would make the harness ready": {},
        "what is left for Jarvis?": {},
        "what remains for the agent harness?": {},
        "what are the remaining gaps": {},
        "remaining gaps": {},
        "next missing capability": {},
        "what blocks Jarvis now?": {},
        "how done is Jarvis now?": {},
        "harness readiness for Jarvis?": {"objective": "for Jarvis"},
    }
    for phrase, expected_args in expected_readiness.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural harness readiness digest should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "harness_readiness_digest" or action.args != expected_args:
            raise SystemExit(f"natural harness readiness digest misplanned {phrase!r}: {(action.tool_name, action.args)}")
    for phrase in [
        "what is missing for photosynthesis",
        "what remains for dinner",
        "why is dinner not ready",
        "what would make dinner ready",
    ]:
        plan = planner.plan(phrase)
        if plan.actions and plan.actions[0].tool_name == "harness_readiness_digest":
            raise SystemExit(f"natural harness readiness should not claim non-Jarvis phrasing: {phrase!r} -> {plan.actions!r}")


def assert_natural_readiness_report_routes() -> None:
    planner = RuleBasedPlanner()
    for phrase in [
        "is Jarvis ready yet",
        "is Jarvis production ready",
        "is the harness ready",
        "is the agent harness ready?",
    ]:
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural readiness report should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "readiness_report" or action.args != {}:
            raise SystemExit(f"natural readiness report misplanned {phrase!r}: {(action.tool_name, action.args)}")


def assert_natural_execution_prep_packet_routes() -> None:
    planner = RuleBasedPlanner()
    proof_args = {
        "verification": "",
        "tests": "",
        "evidence": "",
        "recovery": "",
        "approval": "",
    }
    expected_proof_bundle = {
        "can we trust send email to Sam saying hi": "send email to Sam saying hi",
        "can Jarvis trust send email to Sam saying hi": "send email to Sam saying hi",
        "is send email to Sam saying hi trustworthy": "send email to Sam saying hi",
        "what proof do we need before running send email to Sam saying hi": "send email to Sam saying hi",
        "what do we need before executing send email to Sam saying hi": "send email to Sam saying hi",
        "show pretrust proof for send email to Sam saying hi": "send email to Sam saying hi",
        "build pre-trust proof for send email to Sam saying hi": "send email to Sam saying hi",
        "proof bundle for send email to Sam saying hi": "send email to Sam saying hi",
        "trust packet for send email to Sam saying hi": "send email to Sam saying hi",
        "before running send email to Sam saying hi what proof do we need": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_proof_bundle.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural execution proof bundle should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        expected_args = {"request": expected_request, **proof_args}
        if action.tool_name != "execution_proof_bundle" or action.args != expected_args:
            raise SystemExit(f"natural execution proof bundle misplanned {phrase!r}: {(action.tool_name, action.args)}")

    expected_mission_control = {
        "mission control for send email to Sam saying hi": "send email to Sam saying hi",
        "set up mission control for send email to Sam saying hi": "send email to Sam saying hi",
        "plan the mission for send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_mission_control.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural execution mission control should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "execution_mission_control" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural execution mission control misplanned {phrase!r}: {(action.tool_name, action.args)}")

    expected_runbook = {
        "runbook for send email to Sam saying hi": "send email to Sam saying hi",
        "give me a runbook for send email to Sam saying hi": "send email to Sam saying hi",
        "what is the runbook for send email to Sam saying hi": "send email to Sam saying hi",
        "steps before running send email to Sam saying hi": "send email to Sam saying hi",
        "operational plan for send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_runbook.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural execution runbook should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "execution_runbook" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural execution runbook misplanned {phrase!r}: {(action.tool_name, action.args)}")


def assert_natural_harness_control_packet_routes() -> None:
    planner = RuleBasedPlanner()
    expected_lifecycle = {
        "what is the lifecycle for send email to Sam saying hi": "send email to Sam saying hi",
        "lifecycle for send email to Sam saying hi": "send email to Sam saying hi",
        "show lifecycle for send email to Sam saying hi": "send email to Sam saying hi",
        "where does this sit in the lifecycle: send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_lifecycle.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural harness lifecycle should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "harness_lifecycle_state" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural harness lifecycle misplanned {phrase!r}: {(action.tool_name, action.args)}")

    expected_control = {
        "how should Jarvis control this action": "what should Jarvis do next",
        "show control surface for send email to Sam saying hi": "send email to Sam saying hi",
        "control surface for send email to Sam saying hi": "send email to Sam saying hi",
        "what controls apply to send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_control.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural harness control should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "harness_control_surface" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural harness control misplanned {phrase!r}: {(action.tool_name, action.args)}")

    expected_operations = {
        "how should Jarvis operate on send email to Sam saying hi": "send email to Sam saying hi",
        "operations brief for send email to Sam saying hi": "send email to Sam saying hi",
        "operator brief for send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_objective in expected_operations.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural harness operations should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "harness_operations_brief" or action.args != {"objective": expected_objective}:
            raise SystemExit(f"natural harness operations misplanned {phrase!r}: {(action.tool_name, action.args)}")

    expected_coding = {
        "how should Codex code this safely": "continue Jarvis harness work",
        "coding discipline for planner routing": "planner routing",
        "what coding discipline applies to planner routing": "planner routing",
    }
    for phrase, expected_objective in expected_coding.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural coding discipline should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "coding_discipline_packet" or action.args != {"objective": expected_objective}:
            raise SystemExit(f"natural coding discipline misplanned {phrase!r}: {(action.tool_name, action.args)}")

    for phrase in [
        "what should we build next for AGI",
        "next AGI build move",
        "what is the next AGI build move",
        "which AGI gate should we improve next",
        "what is the next build move",
        "next harness improvement",
        "what gap should we close next",
        "what should Codex work on next",
        "what should we improve next for Jarvis",
    ]:
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural AGI next move should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "agi_next_build_move" or action.args != {"gate": ""}:
            raise SystemExit(f"natural AGI next move misplanned {phrase!r}: {(action.tool_name, action.args)}")


def assert_natural_execution_case_packet_routes() -> None:
    planner = RuleBasedPlanner()
    expected_save_case = {
        "make a case for send email to Sam saying hi": "send email to Sam saying hi",
        "create a case for send email to Sam saying hi": "send email to Sam saying hi",
        "save a case for send email to Sam saying hi": "send email to Sam saying hi",
        "open a case for send email to Sam saying hi": "send email to Sam saying hi",
        "case file for send email to Sam saying hi": "send email to Sam saying hi",
        "prepare a case file for send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_save_case.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural execution case save should plan one local-safe action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "save_execution_case" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural execution case save misplanned {phrase!r}: {(action.tool_name, action.args)}")

    expected_append_evidence = {
        "attach evidence to latest case: verification receipt 7 passed": "latest",
        "add evidence to latest case: verification receipt 7 passed": "latest",
        "append proof to case 2: approval packet 1 reviewed": "2",
    }
    for phrase, expected_case_id in expected_append_evidence.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural case evidence append should plan one local-safe action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "append_execution_case_evidence":
            raise SystemExit(f"natural case evidence append misrouted {phrase!r}: {(action.tool_name, action.args)}")
        if action.args.get("case_id") != expected_case_id or action.args.get("event_type") != "evidence":
            raise SystemExit(f"natural case evidence append missed case/event metadata {phrase!r}: {action.args}")
        if not str(action.args.get("summary") or "").strip():
            raise SystemExit(f"natural case evidence append missed summary {phrase!r}: {action.args}")

    expected_preview_evidence = {
        "preview evidence for latest case: verification receipt 7 passed": "latest",
        "validate proof for case 2: approval packet 1 reviewed": "2",
    }
    for phrase, expected_case_id in expected_preview_evidence.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural case evidence preview should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "execution_case_evidence_packet":
            raise SystemExit(f"natural case evidence preview misrouted {phrase!r}: {(action.tool_name, action.args)}")
        if action.args.get("case_id") != expected_case_id or action.args.get("event_type") != "evidence":
            raise SystemExit(f"natural case evidence preview missed case/event metadata {phrase!r}: {action.args}")
        if not str(action.args.get("summary") or "").strip():
            raise SystemExit(f"natural case evidence preview missed summary {phrase!r}: {action.args}")

    expected_case_packets = {
        "is the latest case ready": ("execution_case_gate", {"case_id": "latest"}),
        "is case 2 ready": ("execution_case_gate", {"case_id": "2"}),
        "review latest case": ("execution_case_review_packet", {"case_id": "latest"}),
        "human review latest case": ("execution_case_review_packet", {"case_id": "latest"}),
        "close out latest case": ("execution_case_closure_packet", {"case_id": "latest"}),
        "show latest case timeline": ("execution_case_timeline", {"case_id": "latest"}),
        "show case history latest": ("execution_case_timeline", {"case_id": "latest"}),
        "show latest case": ("inspect_execution_case", {"case_id": "latest"}),
    }
    for phrase, expected in expected_case_packets.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural execution case packet should plan one action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if (action.tool_name, action.args) != expected:
            raise SystemExit(f"natural execution case packet misplanned {phrase!r}: {(action.tool_name, action.args)}")

    open_calendar = planner.plan("open Calendar")
    if not open_calendar.actions or open_calendar.actions[0].tool_name != "open_application":
        raise SystemExit(f"case aliases should not steal normal app-open requests: {open_calendar.actions}")


def assert_natural_route_diagnosis_stays_read_only() -> None:
    planner = RuleBasedPlanner()
    expected_diagnosis = {
        "why did this route to shell: run command python3 --version": "run command python3 --version",
        "why would jarvis route this: send email to Sam saying hi": "send email to Sam saying hi",
        "how would jarvis route text fixture saying hi": "text fixture saying hi",
        "diagnose route for text fixture saying hi": "text fixture saying hi",
        "what route would jarvis use for send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_diagnosis.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural route diagnosis should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "command_diagnosis" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural route diagnosis misplanned {phrase!r}: {(action.tool_name, action.args)}")

    expected_rehearsals = {
        "what would jarvis do with run command python3 --version": "run command python3 --version",
        "what would you do with send email to Sam saying hi": "send email to Sam saying hi",
        "dry-run send email to Sam saying hi": "send email to Sam saying hi",
        "simulate send email to Sam saying hi": "send email to Sam saying hi",
        "preview send email to Sam saying hi": "send email to Sam saying hi",
        "walk through send email to Sam saying hi": "send email to Sam saying hi",
        "plan only send email to Sam saying hi": "send email to Sam saying hi",
        "show me what would happen if I send email to Sam saying hi": "send email to Sam saying hi",
        "what happens if I run command python3 --version": "run command python3 --version",
        "before you do it send email to Sam saying hi": "send email to Sam saying hi",
    }
    for phrase, expected_request in expected_rehearsals.items():
        plan = planner.plan(phrase)
        if len(plan.actions) != 1:
            raise SystemExit(f"natural rehearsal should plan one read-only action: {phrase!r} -> {plan.actions!r}")
        action = plan.actions[0]
        if action.tool_name != "action_rehearsal" or action.args != {"request": expected_request}:
            raise SystemExit(f"natural rehearsal misplanned {phrase!r}: {(action.tool_name, action.args)}")

    direct_send = planner.plan("text fixture saying hi")
    if not direct_send.actions or direct_send.actions[0].tool_name != "send_imessage":
        raise SystemExit(f"direct messaging route should remain available: {direct_send.actions!r}")
    direct_email = planner.plan("send email to Sam saying hi")
    if (
        len(direct_email.actions) != 1
        or direct_email.actions[0].tool_name != "send_email"
        or direct_email.actions[0].args != {"to": "Sam", "body": "hi"}
    ):
        raise SystemExit(f"direct approval-gated email route should remain available: {direct_email.actions!r}")
    generic_preview = planner.plan("simulate photosynthesis")
    if generic_preview.actions:
        raise SystemExit(f"generic simulate phrase should stay unclaimed: {generic_preview.actions!r}")
    domain_preview = planner.plan("preview weather Seoul")
    if domain_preview.actions and domain_preview.actions[0].tool_name == "action_rehearsal":
        raise SystemExit(f"domain preview phrase should not be hijacked by action rehearsal: {domain_preview.actions!r}")


def assert_harness_latest_case_readiness_uses_exact_bools() -> None:
    source = Path("jarvis_v2/tools/harness.py").read_text()
    stale_patterns = [
        '= bool(metadata.get("latest_execution_case_learning_blocks_completion_claim"))',
        '= bool(metadata.get("latest_execution_case_evidence_preview_latest_handoff_present"))',
    ]
    for pattern in stale_patterns:
        if pattern in source:
            raise SystemExit(f"Harness latest-case readiness still uses loose bool projection: {pattern}")


def assert_fallback_storage_recovery_mode(metadata: dict, label: str) -> None:
    if metadata.get("storage_recovery_mode") != EXPECTED_FALLBACK_STORAGE_RECOVERY_MODE:
        raise SystemExit(f"{label} missed fallback storage recovery mode: {metadata}")
    if metadata.get("storage_recovery_next_operator_action") != EXPECTED_FALLBACK_STORAGE_NEXT_ACTION:
        raise SystemExit(f"{label} missed fallback storage next operator action: {metadata}")
    if metadata.get("storage_recovery_restart_required") is not True:
        raise SystemExit(f"{label} should require restart after fallback storage recovery: {metadata}")


def assert_completion_proof_queue_uses_explicit_proof_sources() -> None:
    proof_queue = _completion_proof_queue_from_sources(
        recovery_proof_queue=["recovery proof alias"],
        learning_closure_command="execution learning closure 17",
        learning_proof_queue=["learning proof alias"],
        learning_actionable_proof_queue=["after-action learning packet 17", "execution learning closure 17"],
        pending_approval_ids=[41],
        open_task_ids=[9],
        baseline_commands=["completion audit", "evidence ledger", "completion claim gate"],
        selected_agi_closure_commands=["agi next build move: long-running autonomy"],
    )
    expected = [
        "approval readiness 41",
        "approval packet 41",
        "approval chain proof 41",
        "verification receipt <approved run id from approval chain proof 41>",
        "task completion packet 9",
        "recovery proof alias",
        "after-action learning packet 17",
        "execution learning closure 17",
        "completion audit",
        "evidence ledger",
        "completion claim gate",
        "agi next build move: long-running autonomy",
    ]
    if proof_queue != expected:
        raise SystemExit(f"Completion proof queue should use explicit proof sources in order: {proof_queue}")
    if "recovery required mirror" in proof_queue or "learning required mirror" in proof_queue:
        raise SystemExit(f"Completion proof queue leaked required-command mirrors: {proof_queue}")


def assert_completion_actionable_queue_preserves_storage_recheck() -> None:
    ordered_queue = [
        "storage status",
        STORAGE_RECOVERY_CHECK_COMMAND,
        "storage status",
        "execution learning closure 17",
        "execution case gate 1",
    ]
    actionable_queue, inserted = _completion_actionable_queue_from_ordered(
        completion_queue=ordered_queue,
        learning_closure_command="execution learning closure 17",
        learning_actionable_queue=["after-action learning packet 17", "execution learning closure 17"],
    )
    expected = [
        "storage status",
        STORAGE_RECOVERY_CHECK_COMMAND,
        "storage status",
        "after-action learning packet 17",
        "execution learning closure 17",
        "execution case gate 1",
    ]
    if inserted is not True:
        raise SystemExit(f"Completion actionable queue should insert learning prerequisites: {actionable_queue}")
    if actionable_queue != expected:
        raise SystemExit(f"Completion actionable queue should preserve intentional storage recheck: {actionable_queue}")


def assert_completion_proof_surfaces_tolerate_malformed_rows() -> None:
    leak = "JARVIS_ROW_LEAK_SENTINEL"

    class MalformedRow:
        def keys(self):
            raise RuntimeError(leak)

        def __getitem__(self, _key):
            raise RuntimeError(leak)

        def __repr__(self) -> str:
            return leak

    with TemporaryDirectory(prefix="jarvis-harness-malformed-proof-") as temp:
        runtime = make_temp_runtime(Path(temp))
        pending_rows = [
            MalformedRow(),
            {
                "id": 77,
                "tool_name": "run_shell_command",
                "status": "pending",
                "created_at": "2026-07-04T00:00:00Z",
            },
        ]
        task_rows = [
            MalformedRow(),
            {"id": 12, "body": "finish the active readiness task"},
        ]
        recent_rows = [
            MalformedRow(),
            {
                "id": 9,
                "tool_name": "run_shell_command",
                "risk": "HIGH_RISK",
                "ok": False,
                "approved": False,
                "approval_id": None,
                "output": "blocked before execution",
                "metadata": "{}",
            },
            {
                "id": 10,
                "tool_name": "verification_receipt",
                "risk": "READ_ONLY",
                "ok": True,
                "approved": False,
                "approval_id": None,
                "output": "verification receipt for run 9",
                "metadata": '{"run_id": 9}',
            },
        ]
        runtime.store.list_pending_approvals = lambda limit=20: pending_rows[:limit]
        runtime.store.list_tasks = lambda status="open", limit=20: task_rows[:limit] if status == "open" else []
        runtime.store.recent_tool_runs = lambda limit=30: recent_rows[:limit]

        claim = runtime.handle("completion claim gate")
        next_proof = runtime.handle("completion next proof")
        digest = runtime.handle("harness readiness digest")

        for label, result, expected_tool in [
            ("completion claim gate", claim, "completion_claim_gate"),
            ("completion next proof", next_proof, "completion_next_proof_packet"),
            ("harness readiness digest", digest, "harness_readiness_digest"),
        ]:
            if not result.verified or result.tool_results[0].tool_name != expected_tool:
                raise SystemExit(f"{label} should run through {expected_tool}: {result}")
            rendered = result.response + repr(result.tool_results[0].metadata)
            if leak in rendered:
                raise SystemExit(f"{label} leaked malformed row diagnostics: {rendered}")

        claim_metadata = claim.tool_results[0].metadata
        proof_queue = claim_metadata.get("completion_proof_queue") or []
        for expected in [
            "approval readiness 77",
            "approval packet 77",
            "approval chain proof 77",
            "task completion packet 12",
            "verification receipt 9",
            "execution health report",
        ]:
            if expected not in proof_queue:
                raise SystemExit(f"Completion claim proof queue missed {expected!r}: {claim_metadata}")
        if claim_metadata.get("unreadable_recent_tool_run_rows") != 1:
            raise SystemExit(f"Completion claim should count unreadable recent run rows: {claim_metadata}")
        if not any("unreadable recent tool run row" in blocker for blocker in claim_metadata.get("blocker_details", [])):
            raise SystemExit(f"Completion claim should block on unreadable recent run rows: {claim_metadata}")

        next_metadata = next_proof.tool_results[0].metadata
        if next_metadata.get("completion_proof_queue") != proof_queue:
            raise SystemExit(f"Completion next proof should inherit the row-safe proof queue: {next_metadata}")

        digest_metadata = digest.tool_results[0].metadata
        if digest_metadata.get("pending_approval_first_id") != 77:
            raise SystemExit(f"Harness readiness digest should skip malformed approval preview rows: {digest_metadata}")
        if digest_metadata.get("pending_approval_preview_unreadable_rows") != 1:
            raise SystemExit(f"Harness readiness digest should count unreadable approval preview rows: {digest_metadata}")
        if digest_metadata.get("approval_review_first_readiness_command") != "approval readiness 77":
            raise SystemExit(f"Harness readiness digest should keep the first readable approval command: {digest_metadata}")


def assert_harness_control_and_operations_tolerate_malformed_rows() -> None:
    leak = "JARVIS_OPERATOR_ROW_LEAK_SENTINEL"

    class MalformedRow:
        def keys(self):
            raise RuntimeError(leak)

        def __getitem__(self, _key):
            raise RuntimeError(leak)

        def __repr__(self) -> str:
            return leak

    with TemporaryDirectory(prefix="jarvis-harness-operator-malformed-") as temp:
        runtime = make_temp_runtime(Path(temp))
        pending_rows = [
            MalformedRow(),
            {
                "id": 77,
                "tool_name": "run_shell_command",
                "status": "pending",
                "created_at": "2026-07-04T00:00:00Z",
            },
        ]
        recent_rows = [
            MalformedRow(),
            {
                "id": 9,
                "tool_name": "run_shell_command",
                "risk": "HIGH_RISK",
                "ok": False,
                "approved": False,
                "approval_id": None,
                "output": "blocked before execution",
                "metadata": "{}",
            },
            {
                "id": 10,
                "tool_name": "verification_receipt",
                "risk": "READ_ONLY",
                "ok": True,
                "approved": False,
                "approval_id": None,
                "output": "verification receipt for run 9",
                "metadata": '{"run_id": 9}',
            },
        ]
        runtime.store.list_pending_approvals = lambda limit=20: pending_rows[:limit]
        runtime.store.recent_tool_runs = lambda limit=30: recent_rows[:limit]

        control = runtime.handle("control surface for send email to Sam saying hi")
        if not control.verified or control.tool_results[0].tool_name != "harness_control_surface":
            raise SystemExit(f"Harness control should route to the control surface: {control}")
        control_metadata = control.tool_results[0].metadata
        control_rendered = control.response + repr(control_metadata)
        if leak in control_rendered:
            raise SystemExit(f"Harness control leaked malformed row diagnostics: {control_rendered}")
        if control_metadata.get("next_command") != "approval readiness 77":
            raise SystemExit(f"Harness control should skip malformed approval rows and keep approval 77: {control_metadata}")
        if control_metadata.get("unreadable_recent_tool_run_rows") != 1:
            raise SystemExit(f"Harness control should count unreadable recent run rows: {control_metadata}")
        if control_metadata.get("readable_recent_tool_runs") != 2:
            raise SystemExit(f"Harness control should count readable recent run rows: {control_metadata}")
        if control_metadata.get("recent_failed_runs") != 1:
            raise SystemExit(f"Harness control should keep readable failure count: {control_metadata}")
        assert_harness_control_handoff(control_metadata, "malformed operator rows")

        runtime.store.list_pending_approvals = lambda limit=20: []
        operations = runtime.handle("operations brief for summarize local harness smoke evidence")
        if not operations.verified or operations.tool_results[0].tool_name != "harness_operations_brief":
            raise SystemExit(f"Harness operations should route to the operations brief: {operations}")
        operations_metadata = operations.tool_results[0].metadata
        operations_rendered = operations.response + repr(operations_metadata)
        if leak in operations_rendered:
            raise SystemExit(f"Harness operations leaked malformed row diagnostics: {operations_rendered}")
        if operations_metadata.get("next_move") != "review_unreadable_recent_runs":
            raise SystemExit(f"Harness operations should fail closed on unreadable recent rows: {operations_metadata}")
        if operations_metadata.get("next_command") != "recent tool runs":
            raise SystemExit(f"Harness operations should point at recent-run review: {operations_metadata}")
        if operations_metadata.get("safe_to_continue") is not False:
            raise SystemExit(f"Harness operations should stop continuation on unreadable rows: {operations_metadata}")
        if operations_metadata.get("can_auto_run") is not False:
            raise SystemExit(f"Harness operations should block auto-run on unreadable rows: {operations_metadata}")
        if operations_metadata.get("unreadable_recent_tool_run_rows") != 1:
            raise SystemExit(f"Harness operations should count unreadable recent run rows: {operations_metadata}")
        if operations_metadata.get("readable_recent_tool_runs") != 2:
            raise SystemExit(f"Harness operations should count readable recent run rows: {operations_metadata}")
        if operations_metadata.get("recent_failed_runs") != 1:
            raise SystemExit(f"Harness operations should keep readable failure count: {operations_metadata}")
        assert_harness_operations_handoff(operations_metadata, "malformed operator rows")


def assert_execution_proof_bundle_tolerates_malformed_recent_rows() -> None:
    leak = "JARVIS_PROOF_BUNDLE_ROW_LEAK_SENTINEL"

    class MalformedRow:
        def keys(self):
            raise RuntimeError(leak)

        def __getitem__(self, _key):
            raise RuntimeError(leak)

        def __repr__(self) -> str:
            return leak

    with TemporaryDirectory(prefix="jarvis-harness-proof-bundle-malformed-") as temp:
        runtime = make_temp_runtime(Path(temp))
        recent_rows = [
            MalformedRow(),
            {
                "id": 9,
                "tool_name": "run_shell_command",
                "risk": "HIGH_RISK",
                "ok": False,
                "approved": False,
                "approval_id": None,
                "output": "blocked before execution",
                "metadata": "{}",
            },
            {
                "id": 10,
                "tool_name": "verification_receipt",
                "risk": "READ_ONLY",
                "ok": True,
                "approved": False,
                "approval_id": None,
                "output": "verification receipt for run 9",
                "metadata": '{"run_id": 9}',
            },
        ]
        runtime.store.recent_tool_runs = lambda limit=50: recent_rows[:limit]

        result = runtime.handle("execution proof bundle: summarize local harness smoke evidence")
        if not result.verified or result.tool_results[0].tool_name != "execution_proof_bundle":
            raise SystemExit(f"Execution proof bundle should route read-only with malformed recent rows: {result}")
        metadata = result.tool_results[0].metadata
        rendered = result.response + repr(metadata)
        if leak in rendered:
            raise SystemExit(f"Execution proof bundle leaked malformed recent-row text: {rendered}")
        if metadata.get("recent_runs") != 3:
            raise SystemExit(f"Execution proof bundle should preserve total recent-run count: {metadata}")
        if metadata.get("readable_recent_tool_runs") != 2:
            raise SystemExit(f"Execution proof bundle should count readable recent runs: {metadata}")
        if metadata.get("unreadable_recent_tool_run_rows") != 1:
            raise SystemExit(f"Execution proof bundle should count unreadable recent rows: {metadata}")
        if metadata.get("recent_failed_runs") != 1:
            raise SystemExit(f"Execution proof bundle should preserve readable failed-run count: {metadata}")
        if metadata.get("verification_runs") != 1:
            raise SystemExit(f"Execution proof bundle should preserve readable verification count: {metadata}")
        if metadata.get("first_failed_run_id") != 9:
            raise SystemExit(f"Execution proof bundle should keep the first readable failed run id: {metadata}")
        if "execution recovery packet 9" not in (metadata.get("next_proof_commands") or []):
            raise SystemExit(f"Execution proof bundle should keep recovery command for readable failure: {metadata}")
        if not any("unreadable row" in blocker for blocker in metadata.get("blocker_details") or []):
            raise SystemExit(f"Execution proof bundle should block on unreadable recent rows: {metadata}")
        if "unreadable recent tool run rows: 1" not in result.response:
            raise SystemExit(f"Execution proof bundle should render unreadable recent-row count: {result.response}")
        assert_execution_proof_handoff(metadata, "malformed proof bundle rows")
        for key in READ_ONLY_FLAGS:
            if metadata.get(key) is not False:
                raise SystemExit(f"Execution proof bundle malformed-row path unsafe metadata {key}: {metadata}")


def assert_priority_goal_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("priority_goal_handoff") or {}
    if handoff.get("source") != "priority_goal":
        raise SystemExit(f"{label} missed priority-goal handoff source: {metadata}")
    for key in [
        "active_matching_goals",
        "active_matching_goal_rows",
        "active_matching_goal_row_count",
        "priority_memories",
        "priority_memory_rows",
        "priority_memory_row_count",
        "priority",
        "priority_objective",
        "priority_build_order",
        "priority_build_order_count",
        "approval_gates_overridden",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "operator_timeboxes_override_priority",
        "stop_times_override_priority",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} priority-goal handoff field {key} diverged: {metadata}")
    goal_rows = metadata.get("active_matching_goal_rows") or []
    memory_rows = metadata.get("priority_memory_rows") or []
    build_order = metadata.get("priority_build_order") or []
    if metadata.get("active_matching_goal_row_count") != len(goal_rows):
        raise SystemExit(f"{label} priority-goal active-goal row count diverged: {metadata}")
    if metadata.get("priority_memory_row_count") != len(memory_rows):
        raise SystemExit(f"{label} priority-goal memory row count diverged: {metadata}")
    if metadata.get("priority_build_order_count") != len(build_order):
        raise SystemExit(f"{label} priority-goal build-order count diverged: {metadata}")
    if not build_order or "command-first GUI" not in build_order[0]:
        raise SystemExit(f"{label} priority-goal build order lost command-first first step: {metadata}")
    if any(row.get("non_authorizing") is not True for row in goal_rows + memory_rows):
        raise SystemExit(f"{label} priority-goal rows should be non-authorizing: {metadata}")
    if handoff.get("approval_gates_overridden") is not False or handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} priority-goal handoff should not override approvals: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} priority-goal handoff missed review-only contract: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} priority-goal handoff unsafe nested metadata {key}: {metadata}")


def assert_harness_status_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_status_handoff") or {}
    if handoff.get("source") != "harness_status":
        raise SystemExit(f"{label} missed harness-status handoff source: {metadata}")
    for key in [
        "components",
        "component_rows",
        "component_row_count",
        "strong_components",
        "partial_components",
        "missing_components",
        "tools",
        "toolsets",
        "toolset_count_rows",
        "toolset_count_row_count",
        "risk_count_rows",
        "risk_count_row_count",
        "pending_approvals",
        "enabled_jobs",
        "jobs",
        "active_goals",
        "open_tasks",
        "recent_tool_runs",
        "best_next_commands",
        "best_next_command_count",
        "harness_layer_contract_rows",
        "harness_layer_contract_row_count",
        "harness_layer_contract_titles",
        "harness_layer_contract_ready_count",
        "harness_layer_contract_partial_count",
        "harness_layer_contract_missing_count",
        "harness_layer_contract_unmeasured_count",
        "harness_layer_contract_non_authorizing",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "operator_timeboxes_override_priority",
        "stop_times_override_priority",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} harness-status handoff field {key} diverged: {metadata}")
    component_rows = metadata.get("component_rows") or []
    toolset_rows = metadata.get("toolset_count_rows") or []
    risk_rows = metadata.get("risk_count_rows") or []
    best_next_commands = metadata.get("best_next_commands") or []
    if metadata.get("components") != len(component_rows) or metadata.get("component_row_count") != len(component_rows):
        raise SystemExit(f"{label} harness-status component row count diverged: {metadata}")
    if metadata.get("strong_components") != len([row for row in component_rows if row.get("status") == "strong foundation"]):
        raise SystemExit(f"{label} harness-status strong component count diverged: {metadata}")
    if metadata.get("partial_components") != len([row for row in component_rows if row.get("status") == "partial foundation"]):
        raise SystemExit(f"{label} harness-status partial component count diverged: {metadata}")
    if metadata.get("missing_components") != len([row for row in component_rows if row.get("status") == "missing"]):
        raise SystemExit(f"{label} harness-status missing component count diverged: {metadata}")
    if metadata.get("toolset_count_row_count") != len(toolset_rows) or metadata.get("toolsets") != len(toolset_rows):
        raise SystemExit(f"{label} harness-status toolset row count diverged: {metadata}")
    if metadata.get("risk_count_row_count") != len(risk_rows):
        raise SystemExit(f"{label} harness-status risk row count diverged: {metadata}")
    if sum(row.get("count") or 0 for row in risk_rows) != metadata.get("tools"):
        raise SystemExit(f"{label} harness-status risk counts should cover all tools: {metadata}")
    if metadata.get("best_next_command_count") != len(best_next_commands):
        raise SystemExit(f"{label} harness-status best-next command count diverged: {metadata}")
    if not best_next_commands or best_next_commands[0] != "architecture map":
        raise SystemExit(f"{label} harness-status lost architecture-map first command: {metadata}")
    if any(row.get("non_authorizing") is not True for row in component_rows + toolset_rows + risk_rows):
        raise SystemExit(f"{label} harness-status rows should be non-authorizing: {metadata}")
    assert_harness_layer_contract(metadata, label)
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} harness-status handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} harness-status handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} harness-status handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} harness-status handoff unsafe nested metadata {key}: {metadata}")


def assert_harness_cycle_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_cycle_handoff") or {}
    if handoff.get("source") != "harness_cycle_preview":
        raise SystemExit(f"{label} missed harness-cycle handoff source: {metadata}")
    for key in [
        "request",
        "request_chars",
        "likely_route",
        "primary_preview",
        "fallback_preview",
        "safe_to_execute_now",
        "risk_signals",
        "risk_signal_count",
        "approval_required",
        "pending_approvals",
        "pending_approval_review_required",
        "cycle_stages",
        "evidence_tools",
        "evidence_tool_count",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "recovery_closure_state",
        "recovery_closure_ready_to_retry",
        "recovery_closure_missing",
        "recovery_closure_missing_count",
        "recovery_closure_required_commands",
        "recovery_closure_next_required_command",
        "recovery_closure_blocks_auto_execution",
        "recovery_closure_target_run_id",
        "recovery_closure_target_tool_name",
        "recovery_closure_proof_queue",
        "recovery_closure_proof_queue_count",
        "recovery_closure_next_proof_command",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} harness-cycle handoff field {key} diverged: {metadata}")
    if metadata.get("request_chars") != len(metadata.get("request") or ""):
        raise SystemExit(f"{label} harness-cycle request length diverged: {metadata}")
    if metadata.get("risk_signal_count") != len(metadata.get("risk_signals") or []):
        raise SystemExit(f"{label} harness-cycle risk count diverged: {metadata}")
    if metadata.get("evidence_tool_count") != len(metadata.get("evidence_tools") or []):
        raise SystemExit(f"{label} harness-cycle evidence tool count diverged: {metadata}")
    if metadata.get("cycle_stages") != 8:
        raise SystemExit(f"{label} harness-cycle stage count diverged: {metadata}")
    if metadata.get("recovery_closure_proof_queue") != metadata.get("recovery_closure_required_commands"):
        raise SystemExit(f"{label} harness-cycle recovery proof queue diverged: {metadata}")
    if metadata.get("recovery_closure_proof_queue_count") != len(metadata.get("recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} harness-cycle recovery proof count diverged: {metadata}")
    if metadata.get("recovery_closure_missing_count") != len(metadata.get("recovery_closure_missing") or []):
        raise SystemExit(f"{label} harness-cycle recovery missing count diverged: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} harness-cycle handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} harness-cycle handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} harness-cycle handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} harness-cycle handoff unsafe nested metadata {key}: {metadata}")


def assert_no_request_harness_cycle_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_cycle_handoff") or {}
    expected = {
        "source": "harness_cycle_preview",
        "reason": "missing_request",
        "found": False,
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value or handoff.get(key) != value:
            raise SystemExit(f"{label} no-request harness-cycle handoff missed {key}: {metadata}")
    for key in READ_ONLY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} no-request harness-cycle handoff unsafe metadata {key}: {metadata}")


def assert_harness_doctrine_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_doctrine_handoff") or {}
    if handoff.get("source") != "harness_doctrine":
        raise SystemExit(f"{label} missed harness-doctrine handoff source: {metadata}")
    for key in [
        "principles",
        "doctrine_rows",
        "doctrine_row_count",
        "priority",
        "build_filter",
        "build_filter_count",
        "harness_layer_contract_rows",
        "harness_layer_contract_row_count",
        "harness_layer_contract_titles",
        "harness_layer_contract_ready_count",
        "harness_layer_contract_partial_count",
        "harness_layer_contract_missing_count",
        "harness_layer_contract_unmeasured_count",
        "harness_layer_contract_non_authorizing",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "operator_timeboxes_override_priority",
        "stop_times_override_priority",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} harness-doctrine handoff field {key} diverged: {metadata}")
    rows = metadata.get("doctrine_rows") or []
    build_filter = metadata.get("build_filter") or []
    if metadata.get("principles") != len(rows) or metadata.get("doctrine_row_count") != len(rows):
        raise SystemExit(f"{label} harness-doctrine row count diverged: {metadata}")
    if metadata.get("build_filter_count") != len(build_filter):
        raise SystemExit(f"{label} harness-doctrine build-filter count diverged: {metadata}")
    if not rows or any(row.get("non_authorizing") is not True for row in rows):
        raise SystemExit(f"{label} harness-doctrine rows should be non-authorizing: {metadata}")
    assert_harness_layer_contract(metadata, label)
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} harness-doctrine handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} harness-doctrine handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} harness-doctrine handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} harness-doctrine handoff should not grant approval: {metadata}")


def assert_coding_discipline_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("coding_discipline_handoff") or {}
    if handoff.get("source") != "coding_discipline_packet":
        raise SystemExit(f"{label} missed coding-discipline handoff source: {metadata}")
    for key in [
        "objective",
        "objective_chars",
        "likely_files",
        "likely_file_count",
        "verification",
        "verification_supplied",
        "ambiguity",
        "discipline_rows",
        "discipline_row_count",
        "ready_rows",
        "review_rows",
        "ready_for_implementation_review",
        "proof_queue",
        "proof_queue_count",
        "next_proof_command",
        "authorizes_execution",
        "authorizes_edits",
        "authorizes_risky_work",
        "authorizes_test_execution",
        "authorizes_completion_claim",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "approval_granted",
        "operator_timeboxes_override_priority",
        "stop_times_override_priority",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} coding-discipline handoff field {key} diverged: {metadata}")
    rows = metadata.get("discipline_rows") or []
    proof_queue = metadata.get("proof_queue") or []
    likely_files = metadata.get("likely_files") or []
    if metadata.get("objective_chars") != len(metadata.get("objective") or ""):
        raise SystemExit(f"{label} coding-discipline objective length diverged: {metadata}")
    if metadata.get("discipline_row_count") != len(rows):
        raise SystemExit(f"{label} coding-discipline row count diverged: {metadata}")
    if metadata.get("ready_rows") != len([row for row in rows if row.get("status") == "ready"]):
        raise SystemExit(f"{label} coding-discipline ready row count diverged: {metadata}")
    if metadata.get("review_rows") != len([row for row in rows if row.get("status") == "review"]):
        raise SystemExit(f"{label} coding-discipline review row count diverged: {metadata}")
    if metadata.get("likely_file_count") != len(likely_files):
        raise SystemExit(f"{label} coding-discipline file count diverged: {metadata}")
    if metadata.get("proof_queue_count") != len(proof_queue) or metadata.get("next_proof_command") != (proof_queue[0] if proof_queue else None):
        raise SystemExit(f"{label} coding-discipline proof queue diverged: {metadata}")
    if not rows or any(row.get("non_authorizing") is not True for row in rows):
        raise SystemExit(f"{label} coding-discipline rows should be non-authorizing: {metadata}")
    for key in ["authorizes_execution", "authorizes_edits", "authorizes_risky_work", "authorizes_test_execution", "authorizes_completion_claim", "approval_granted"]:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} coding-discipline handoff should not authorize {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} coding-discipline handoff missed review-only contract: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} coding-discipline handoff unsafe nested metadata {key}: {metadata}")


def assert_harness_lifecycle_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_lifecycle_handoff") or {}
    if handoff.get("source") != "harness_lifecycle_state":
        raise SystemExit(f"{label} missed harness-lifecycle handoff source: {metadata}")
    for key in [
        "request",
        "request_chars",
        "likely_route",
        "safe_to_execute_now",
        "approval_required",
        "risk_signals",
        "risk_signal_count",
        "pending_approvals",
        "pending_approval_review_required",
        "lifecycle_stages",
        "ready_stages",
        "held_stages",
        "stage_states",
        "stage_state_count",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "recovery_closure_state",
        "recovery_closure_ready_to_retry",
        "recovery_closure_missing",
        "recovery_closure_missing_count",
        "recovery_closure_required_commands",
        "recovery_closure_next_required_command",
        "recovery_closure_blocks_auto_execution",
        "recovery_closure_target_run_id",
        "recovery_closure_target_tool_name",
        "recovery_closure_proof_queue",
        "recovery_closure_proof_queue_count",
        "recovery_closure_next_proof_command",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} harness-lifecycle handoff field {key} diverged: {metadata}")
    if metadata.get("request_chars") != len(metadata.get("request") or ""):
        raise SystemExit(f"{label} harness-lifecycle request length diverged: {metadata}")
    if metadata.get("risk_signal_count") != len(metadata.get("risk_signals") or []):
        raise SystemExit(f"{label} harness-lifecycle risk count diverged: {metadata}")
    stage_states = metadata.get("stage_states") or []
    if metadata.get("stage_state_count") != len(stage_states) or metadata.get("lifecycle_stages") != len(stage_states):
        raise SystemExit(f"{label} harness-lifecycle stage count diverged: {metadata}")
    if metadata.get("ready_stages") != len([stage for stage in stage_states if stage.get("status") == "ready"]):
        raise SystemExit(f"{label} harness-lifecycle ready stage count diverged: {metadata}")
    if metadata.get("held_stages") != len([stage for stage in stage_states if "held" in str(stage.get("status") or "") or "blocked" in str(stage.get("status") or "")]):
        raise SystemExit(f"{label} harness-lifecycle held stage count diverged: {metadata}")
    if metadata.get("recovery_closure_proof_queue") != metadata.get("recovery_closure_required_commands"):
        raise SystemExit(f"{label} harness-lifecycle recovery proof queue diverged: {metadata}")
    if metadata.get("recovery_closure_proof_queue_count") != len(metadata.get("recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} harness-lifecycle recovery proof count diverged: {metadata}")
    if metadata.get("recovery_closure_missing_count") != len(metadata.get("recovery_closure_missing") or []):
        raise SystemExit(f"{label} harness-lifecycle recovery missing count diverged: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} harness-lifecycle handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} harness-lifecycle handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} harness-lifecycle handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} harness-lifecycle handoff should not grant approval: {metadata}")


def assert_no_request_harness_lifecycle_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_lifecycle_handoff") or {}
    expected = {
        "source": "harness_lifecycle_state",
        "reason": "missing_request",
        "found": False,
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value or handoff.get(key) != value:
            raise SystemExit(f"{label} no-request harness-lifecycle handoff missed {key}: {metadata}")
    for key in READ_ONLY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} no-request harness-lifecycle handoff unsafe metadata {key}: {metadata}")


def assert_harness_control_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_control_handoff") or {}
    if handoff.get("source") != "harness_control_surface":
        raise SystemExit(f"{label} missed harness-control handoff source: {metadata}")
    for key in [
        "request",
        "request_chars",
        "verdict",
        "next_command",
        "can_auto_run",
        "risk_signals",
        "risk_signal_count",
        "approval_required",
        "approval_review_required",
        "pending_approvals",
        "readable_recent_tool_runs",
        "unreadable_recent_tool_run_rows",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "control_groups",
        "ready_groups",
        "partial_groups",
        "missing_groups",
        "group_states",
        "group_state_count",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "recovery_closure_state",
        "recovery_closure_ready_to_retry",
        "recovery_closure_missing",
        "recovery_closure_missing_count",
        "recovery_closure_required_commands",
        "recovery_closure_next_required_command",
        "recovery_closure_blocks_auto_execution",
        "recovery_closure_target_run_id",
        "recovery_closure_target_tool_name",
        "recovery_closure_failed_action_runs",
        "recovery_closure_approval_held_action_runs",
        "recovery_closure_proof_queue",
        "recovery_closure_proof_queue_count",
        "recovery_closure_next_proof_command",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} harness-control handoff field {key} diverged: {metadata}")
    if metadata.get("request_chars") != len(metadata.get("request") or ""):
        raise SystemExit(f"{label} harness-control request length diverged: {metadata}")
    if metadata.get("risk_signal_count") != len(metadata.get("risk_signals") or []):
        raise SystemExit(f"{label} harness-control risk count diverged: {metadata}")
    group_states = metadata.get("group_states") or []
    if metadata.get("group_state_count") != len(group_states) or metadata.get("control_groups") != len(group_states):
        raise SystemExit(f"{label} harness-control group count diverged: {metadata}")
    if metadata.get("ready_groups") != len([group for group in group_states if group.get("status") == "ready"]):
        raise SystemExit(f"{label} harness-control ready group count diverged: {metadata}")
    if metadata.get("partial_groups") != len([group for group in group_states if group.get("status") == "partial"]):
        raise SystemExit(f"{label} harness-control partial group count diverged: {metadata}")
    if metadata.get("missing_groups") != len([group for group in group_states if group.get("status") == "missing"]):
        raise SystemExit(f"{label} harness-control missing group count diverged: {metadata}")
    if metadata.get("recovery_closure_proof_queue") != metadata.get("recovery_closure_required_commands"):
        raise SystemExit(f"{label} harness-control recovery proof queue diverged: {metadata}")
    if metadata.get("recovery_closure_proof_queue_count") != len(metadata.get("recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} harness-control recovery proof count diverged: {metadata}")
    if metadata.get("recovery_closure_missing_count") != len(metadata.get("recovery_closure_missing") or []):
        raise SystemExit(f"{label} harness-control recovery missing count diverged: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} harness-control handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} harness-control handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} harness-control handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} harness-control handoff should not grant approval: {metadata}")


def assert_no_request_harness_control_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_control_handoff") or {}
    expected = {
        "source": "harness_control_surface",
        "reason": "missing_request",
        "found": False,
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value or handoff.get(key) != value:
            raise SystemExit(f"{label} no-request harness-control handoff missed {key}: {metadata}")
    for key in READ_ONLY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} no-request harness-control handoff unsafe metadata {key}: {metadata}")


def assert_harness_operations_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_operations_handoff") or {}
    if handoff.get("source") != "harness_operations_brief":
        raise SystemExit(f"{label} missed harness-operations handoff source: {metadata}")
    for key in [
        "objective",
        "objective_chars",
        "next_move",
        "next_command",
        "safe_to_continue",
        "safe_to_execute_now",
        "can_auto_run",
        "auto_continue_status",
        "stop_condition",
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "readable_recent_tool_runs",
        "unreadable_recent_tool_run_rows",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "required_controls",
        "available_controls",
        "missing_controls",
        "available_control_tools",
        "available_control_tool_count",
        "missing_control_tools",
        "missing_control_tool_count",
        "recommended_control_commands",
        "recommended_control_command_count",
        "implementation_review_commands",
        "implementation_review_command_count",
        "approval_review_required",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "recovery_closure_state",
        "recovery_closure_ready_to_retry",
        "recovery_closure_missing",
        "recovery_closure_missing_count",
        "recovery_closure_required_commands",
        "recovery_closure_next_required_command",
        "recovery_closure_blocks_auto_execution",
        "recovery_closure_target_run_id",
        "recovery_closure_target_tool_name",
        "recovery_closure_failed_action_runs",
        "recovery_closure_approval_held_action_runs",
        "recovery_closure_proof_queue",
        "recovery_closure_proof_queue_count",
        "recovery_closure_next_proof_command",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} harness-operations handoff field {key} diverged: {metadata}")
    if metadata.get("objective_chars") != len(metadata.get("objective") or ""):
        raise SystemExit(f"{label} harness-operations objective length diverged: {metadata}")
    if metadata.get("available_control_tool_count") != len(metadata.get("available_control_tools") or []):
        raise SystemExit(f"{label} harness-operations available control count diverged: {metadata}")
    if metadata.get("missing_control_tool_count") != len(metadata.get("missing_control_tools") or []):
        raise SystemExit(f"{label} harness-operations missing control count diverged: {metadata}")
    if metadata.get("available_controls") != len(metadata.get("available_control_tools") or []):
        raise SystemExit(f"{label} harness-operations available controls diverged: {metadata}")
    if metadata.get("missing_controls") != len(metadata.get("missing_control_tools") or []):
        raise SystemExit(f"{label} harness-operations missing controls diverged: {metadata}")
    if metadata.get("recommended_control_command_count") != len(metadata.get("recommended_control_commands") or []):
        raise SystemExit(f"{label} harness-operations recommended command count diverged: {metadata}")
    if metadata.get("implementation_review_command_count") != len(metadata.get("implementation_review_commands") or []):
        raise SystemExit(f"{label} harness-operations implementation command count diverged: {metadata}")
    if metadata.get("recovery_closure_proof_queue") != metadata.get("recovery_closure_required_commands"):
        raise SystemExit(f"{label} harness-operations recovery proof queue diverged: {metadata}")
    if metadata.get("recovery_closure_proof_queue_count") != len(metadata.get("recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} harness-operations recovery proof count diverged: {metadata}")
    if metadata.get("recovery_closure_missing_count") != len(metadata.get("recovery_closure_missing") or []):
        raise SystemExit(f"{label} harness-operations recovery missing count diverged: {metadata}")
    if metadata.get("recovery_closure_blocks_auto_execution") and metadata.get("can_auto_run") is not False:
        raise SystemExit(f"{label} harness-operations should block auto-run while recovery closure is open: {metadata}")
    expected_auto_status = (
        "yes"
        if metadata.get("can_auto_run") is True
        else "no, review approval first"
        if metadata.get("safe_to_continue") is False
        else "no, close recovery proof first"
    )
    if metadata.get("auto_continue_status") != expected_auto_status:
        raise SystemExit(f"{label} harness-operations auto-continue status drifted: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} harness-operations handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} harness-operations handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} harness-operations handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} harness-operations handoff should not grant approval: {metadata}")


def assert_agi_gate_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("agi_gate_handoff") or {}
    if handoff.get("source") != "agi_gate_report":
        raise SystemExit(f"{label} missed AGI gate handoff source: {metadata}")
    for key in [
        "gates",
        "gate_rows",
        "gate_row_count",
        "strong_gates",
        "partial_gates",
        "missing_gates",
        "pending_approvals",
        "active_goals",
        "open_tasks",
        "gate_statuses",
        "present_evidence_by_gate",
        "missing_evidence_by_gate",
        "next_moves_by_gate",
        "evidence_closure_commands_by_gate",
        "focused_verification_by_gate",
        "real_execution_gaps_by_gate",
        "real_execution_gap_count",
        "best_next_commands",
        "best_next_command_count",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} AGI gate handoff field {key} diverged: {metadata}")
    gate_rows = metadata.get("gate_rows") or []
    if metadata.get("gate_row_count") != len(gate_rows) or metadata.get("gates") != len(gate_rows):
        raise SystemExit(f"{label} AGI gate row count diverged: {metadata}")
    status_counts = {"strong prototype": 0, "partial prototype": 0, "missing": 0}
    for row in gate_rows:
        status_counts[str(row.get("status") or "")] = status_counts.get(str(row.get("status") or ""), 0) + 1
        gate_name = row.get("gate")
        if metadata.get("gate_statuses", {}).get(gate_name) != row.get("status"):
            raise SystemExit(f"{label} AGI gate status map diverged: {metadata}")
        if metadata.get("present_evidence_by_gate", {}).get(gate_name) != row.get("present"):
            raise SystemExit(f"{label} AGI gate present evidence map diverged: {metadata}")
        if metadata.get("missing_evidence_by_gate", {}).get(gate_name) != row.get("missing_evidence"):
            raise SystemExit(f"{label} AGI gate missing evidence map diverged: {metadata}")
        if metadata.get("next_moves_by_gate", {}).get(gate_name) != row.get("next"):
            raise SystemExit(f"{label} AGI gate next-move map diverged: {metadata}")
        if metadata.get("evidence_closure_commands_by_gate", {}).get(gate_name) != row.get("evidence_closure_commands"):
            raise SystemExit(f"{label} AGI gate closure-command map diverged: {metadata}")
        if metadata.get("focused_verification_by_gate", {}).get(gate_name) != row.get("focused_verification"):
            raise SystemExit(f"{label} AGI gate focused-verification map diverged: {metadata}")
        if metadata.get("real_execution_gaps_by_gate", {}).get(gate_name) != row.get("real_execution_gap"):
            raise SystemExit(f"{label} AGI gate real-execution gap map diverged: {metadata}")
        if row.get("non_authorizing") is not True:
            raise SystemExit(f"{label} AGI gate row should be non-authorizing: {metadata}")
    if metadata.get("strong_gates") != status_counts.get("strong prototype", 0):
        raise SystemExit(f"{label} AGI gate strong count diverged: {metadata}")
    if metadata.get("partial_gates") != status_counts.get("partial prototype", 0):
        raise SystemExit(f"{label} AGI gate partial count diverged: {metadata}")
    if metadata.get("missing_gates") != status_counts.get("missing", 0):
        raise SystemExit(f"{label} AGI gate missing count diverged: {metadata}")
    if metadata.get("real_execution_gap_count") != len(metadata.get("real_execution_gaps_by_gate") or {}):
        raise SystemExit(f"{label} AGI gate real-execution gap count diverged: {metadata}")
    if metadata.get("best_next_command_count") != len(metadata.get("best_next_commands") or []):
        raise SystemExit(f"{label} AGI gate best-next command count diverged: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} AGI gate handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} AGI gate handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} AGI gate handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} AGI gate handoff unsafe nested metadata {key}: {metadata}")


def assert_execution_health_recovery_proof_aliases(metadata: dict, label: str) -> None:
    for key in [
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
    ]:
        if key not in metadata:
            raise SystemExit(f"{label} missed explicit recovery proof alias {key}: {metadata}")
    closure_commands = metadata.get("execution_health_recovery_closure_required_commands") or []
    proof_queue = metadata.get("execution_health_recovery_closure_proof_queue")
    if proof_queue != closure_commands:
        raise SystemExit(f"{label} recovery proof queue diverged from closure commands: {metadata}")
    if metadata.get("execution_health_recovery_closure_proof_queue_count") != len(closure_commands):
        raise SystemExit(f"{label} recovery proof queue count diverged: {metadata}")
    next_required = metadata.get("execution_health_recovery_closure_next_required_command")
    if metadata.get("execution_health_recovery_closure_next_proof_command") != next_required:
        raise SystemExit(f"{label} recovery next proof alias diverged: {metadata}")
    if closure_commands and next_required != closure_commands[0]:
        raise SystemExit(f"{label} missed first recovery proof command: {metadata}")
    if metadata.get("execution_health_recovery_closure_blocks_completion_claim"):
        if metadata.get("execution_health_recovery_closure_checklist_command") != "recovery closure checklist":
            raise SystemExit(f"{label} missed recovery closure checklist command: {metadata}")
        if metadata.get("execution_health_recovery_closure_should_open_checklist") is not True:
            raise SystemExit(f"{label} missed recovery closure checklist open flag: {metadata}")
        if metadata.get("execution_health_recovery_closure_target_run_id") is None:
            raise SystemExit(f"{label} missed recovery closure target run: {metadata}")
        if not metadata.get("execution_health_recovery_closure_target_tool_name"):
            raise SystemExit(f"{label} missed recovery closure target tool: {metadata}")
        for key in [
            "execution_health_recovery_closure_target_verification_receipts",
            "execution_health_recovery_closure_target_recovery_packets",
            "execution_health_recovery_closure_target_after_action_learning_packets",
        ]:
            if key not in metadata or not isinstance(metadata.get(key), int):
                raise SystemExit(f"{label} missed recovery closure target proof count {key}: {metadata}")


def assert_execution_health_verification_proof_aliases(metadata: dict, label: str) -> None:
    for key in [
        "execution_health_verification_coverage_state",
        "execution_health_verification_proof_queue",
        "execution_health_verification_proof_queue_count",
        "execution_health_verification_next_required_command",
        "execution_health_verification_next_proof_command",
        "execution_health_verification_blocks_completion_claim",
    ]:
        if key not in metadata:
            raise SystemExit(f"{label} missed execution-health verification alias {key}: {metadata}")
    proof_queue = metadata.get("execution_health_verification_proof_queue") or []
    if metadata.get("execution_health_verification_proof_queue_count") != len(proof_queue):
        raise SystemExit(f"{label} execution-health verification queue count diverged: {metadata}")
    expected_next = proof_queue[0] if proof_queue else ""
    if metadata.get("execution_health_verification_next_proof_command") != expected_next:
        raise SystemExit(f"{label} execution-health verification next proof diverged: {metadata}")
    if metadata.get("execution_health_verification_next_required_command") != expected_next:
        raise SystemExit(f"{label} execution-health verification next required diverged: {metadata}")
    if metadata.get("execution_health_verification_coverage_state") in {"missing", "no_recent_tool_runs"}:
        if proof_queue != ["verification receipt latest"]:
            raise SystemExit(f"{label} missing/no-recent verification coverage should point at latest receipt: {metadata}")
        if metadata.get("execution_health_verification_blocks_completion_claim") is not True:
            raise SystemExit(f"{label} missing/no-recent verification coverage should block completion: {metadata}")
        completion_queue = metadata.get("completion_proof_queue") or []
        if "verification receipt latest" not in completion_queue:
            raise SystemExit(f"{label} completion proof queue missed verification coverage proof: {metadata}")
    elif metadata.get("execution_health_verification_coverage_state") == "present":
        if proof_queue:
            raise SystemExit(f"{label} satisfied or empty verification coverage should not require a verification proof queue: {metadata}")
        if metadata.get("execution_health_verification_blocks_completion_claim") is not False:
            raise SystemExit(f"{label} satisfied or empty verification coverage should not block completion: {metadata}")
    else:
        raise SystemExit(f"{label} has unknown verification coverage state: {metadata}")


def assert_execution_learning_proof_aliases(metadata: dict, label: str) -> None:
    for key in [
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
    ]:
        if key not in metadata:
            raise SystemExit(f"{label} missed explicit learning proof alias {key}: {metadata}")
    learning_commands = metadata.get("execution_learning_required_commands") or []
    if metadata.get("execution_learning_proof_queue") != learning_commands:
        raise SystemExit(f"{label} learning proof queue diverged from required commands: {metadata}")
    if metadata.get("execution_learning_proof_queue_count") != len(learning_commands):
        raise SystemExit(f"{label} learning proof queue count diverged: {metadata}")
    next_required = metadata.get("execution_learning_next_required_command")
    if metadata.get("execution_learning_next_proof_command") != next_required:
        raise SystemExit(f"{label} learning next proof alias diverged: {metadata}")
    if learning_commands and next_required != learning_commands[0]:
        raise SystemExit(f"{label} missed first learning proof command: {metadata}")


def assert_execution_learning_actionable_aliases(metadata: dict, label: str) -> None:
    actionable_commands = metadata.get("execution_learning_actionable_required_commands") or []
    actionable_queue = metadata.get("execution_learning_actionable_proof_queue") or []
    if actionable_queue != actionable_commands:
        raise SystemExit(f"{label} actionable learning queue diverged from required commands: {metadata}")
    if metadata.get("execution_learning_actionable_required_command_count") != len(actionable_commands):
        raise SystemExit(f"{label} actionable learning required count diverged: {metadata}")
    if metadata.get("execution_learning_actionable_proof_queue_count") != len(actionable_queue):
        raise SystemExit(f"{label} actionable learning proof queue count diverged: {metadata}")
    next_evidence = metadata.get("execution_learning_next_evidence_command")
    if metadata.get("execution_learning_actionable_next_required_command") != next_evidence:
        raise SystemExit(f"{label} actionable next required command diverged from next evidence command: {metadata}")
    if metadata.get("execution_learning_actionable_next_proof_command") != next_evidence:
        raise SystemExit(f"{label} actionable next proof command diverged from next evidence command: {metadata}")
    if actionable_commands and next_evidence != actionable_commands[0]:
        raise SystemExit(f"{label} missed first actionable learning proof command: {metadata}")


def assert_selected_agi_target_readiness(metadata: dict, label: str) -> None:
    if not metadata.get("agi_next_gate") or not metadata.get("agi_next_build_command"):
        raise SystemExit(f"{label} missed selected AGI next build target: {metadata}")
    if not metadata.get("agi_next_target_title"):
        raise SystemExit(f"{label} missed selected AGI target title: {metadata}")
    if not metadata.get("agi_focus_selection_source"):
        raise SystemExit(f"{label} missed AGI focus selection source: {metadata}")
    if not metadata.get("agi_focus_selection_reason"):
        raise SystemExit(f"{label} missed AGI focus selection reason: {metadata}")
    expected_selector = f"agi next build move: {metadata.get('agi_next_gate')}"
    if metadata.get("agi_focus_canonical_selector_command") != expected_selector:
        raise SystemExit(f"{label} missed canonical AGI selector command: {metadata}")
    if metadata.get("agi_focus_deliberate_focus_override") not in {True, False}:
        raise SystemExit(f"{label} missed AGI deliberate focus override flag: {metadata}")
    if metadata.get("agi_next_evidence_closure_command_count") != len(metadata.get("agi_next_evidence_closure_commands", [])):
        raise SystemExit(f"{label} missed selected AGI closure command count: {metadata}")
    if metadata.get("agi_next_focused_verification_command_count") != len(metadata.get("agi_next_focused_verification_commands", [])):
        raise SystemExit(f"{label} missed selected AGI focused verification count: {metadata}")
    if metadata.get("agi_next_likely_file_count") != len(metadata.get("agi_next_likely_files", [])):
        raise SystemExit(f"{label} missed selected AGI likely file count: {metadata}")
    if metadata.get("agi_next_acceptance_check_count") != len(metadata.get("agi_next_acceptance_checks", [])):
        raise SystemExit(f"{label} missed selected AGI acceptance count: {metadata}")
    expected_acceptance_preview = list(metadata.get("agi_next_acceptance_checks", []))[:3]
    if metadata.get("agi_next_acceptance_preview") != expected_acceptance_preview:
        raise SystemExit(f"{label} missed selected AGI acceptance preview alias: {metadata}")
    if metadata.get("agi_next_acceptance_preview_count") != len(expected_acceptance_preview):
        raise SystemExit(f"{label} missed selected AGI acceptance preview alias count: {metadata}")
    if metadata.get("agi_next_first_acceptance_check") != (expected_acceptance_preview[0] if expected_acceptance_preview else ""):
        raise SystemExit(f"{label} missed selected AGI first acceptance check alias: {metadata}")
    if metadata.get("agi_next_acceptance_gap_preview") != expected_acceptance_preview:
        raise SystemExit(f"{label} missed selected AGI acceptance gap preview: {metadata}")
    if metadata.get("agi_next_acceptance_gap_preview_count") != len(expected_acceptance_preview):
        raise SystemExit(f"{label} missed selected AGI acceptance gap preview count: {metadata}")
    if metadata.get("agi_next_first_acceptance_gap") != (expected_acceptance_preview[0] if expected_acceptance_preview else ""):
        raise SystemExit(f"{label} missed selected AGI first acceptance gap: {metadata}")
    if bool(metadata.get("agi_next_target_integrity_blocks_start")) is metadata.get("agi_next_target_files_exist"):
        raise SystemExit(f"{label} target-integrity blocker should invert target_files_exist: {metadata}")
    expected_ready = bool(
        metadata.get("agi_next_target_files_exist")
        and metadata.get("agi_next_focused_verification_commands")
        and metadata.get("agi_next_acceptance_checks")
    )
    if metadata.get("agi_next_build_packet_ready_for_review") is not expected_ready:
        raise SystemExit(f"{label} selected AGI readiness flag diverged: {metadata}")
    preflight_keys = {
        "agi_next_implementation_preflight_ready",
        "agi_next_implementation_preflight_blockers",
        "agi_next_implementation_preflight_blocker_count",
        "agi_next_implementation_preflight_next_command",
    }
    if any(key in metadata for key in preflight_keys):
        preflight_blockers = list(metadata.get("agi_next_implementation_preflight_blockers") or [])
        if metadata.get("agi_next_implementation_preflight_blocker_count") != len(preflight_blockers):
            raise SystemExit(f"{label} AGI implementation preflight blocker count diverged: {metadata}")
        expected_preflight_ready = expected_ready and not preflight_blockers
        if metadata.get("agi_next_implementation_preflight_ready") is not expected_preflight_ready:
            raise SystemExit(f"{label} AGI implementation preflight readiness flag diverged: {metadata}")
        if preflight_blockers and not metadata.get("agi_next_implementation_preflight_next_command"):
            raise SystemExit(f"{label} AGI implementation preflight missed next command for blockers: {metadata}")


def assert_harness_completion_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_completion_handoff") or {}
    if handoff.get("source") != "harness_completion_assessment":
        raise SystemExit(f"{label} missed harness completion handoff source: {metadata}")
    for key in [
        "overall_percent",
        "component_percent",
        "gate_percent",
        "execution_readiness_percent",
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "approval_held_review_command",
        "components",
        "incomplete_components",
        "incomplete_component_count",
        "gates",
        "agi_gates",
        "agi_strong_gates",
        "agi_partial_gates",
        "agi_missing_gates",
        "agi_real_execution_gap_count",
        "agi_next_gate",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "real_execution_gap_count",
        "selected_real_execution_gap",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_file_integrity_status",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_missing_target_files",
        "agi_next_missing_target_file_count",
        "agi_next_target_integrity_blocks_start",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_acceptance_gap_preview",
        "agi_next_acceptance_gap_preview_count",
        "agi_next_first_acceptance_gap",
        "agi_next_build_packet_ready_for_review",
        "completion_blockers",
        "completion_blocker_count",
        "completion_blockers_deduplicated",
        "completion_claim_ready",
        "next_proof_commands",
        "next_proof_command",
        "completion_proof_queue",
        "completion_proof_queue_count",
        "completion_next_proof_command",
        "next_completion_proof_command",
        "storage_runtime_fallback_active",
        "storage_runtime_fallback_reason",
        "storage_runtime_fallback_exception_type",
        "storage_runtime_fallback_db_path_display",
        "storage_runtime_fallback_vault_path_display",
        "storage_readiness_blocks_completion_claim",
        "storage_recovery_required",
        "storage_recovery_reason",
        "storage_recovery_mode",
        "storage_recovery_next_operator_action",
        "storage_recovery_restart_required",
        "storage_recovery_check_command",
        "storage_recovery_check_api",
        "storage_recovery_command",
        "storage_readiness_blocker",
        "storage_readiness_next_commands",
        "storage_readiness_next_command_count",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_ready_to_retry",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_checklist_command",
        "execution_health_recovery_closure_should_open_checklist",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_health_recovery_closure_target_run_id",
        "execution_health_recovery_closure_target_tool_name",
        "execution_health_recovery_closure_target_verification_receipts",
        "execution_health_recovery_closure_target_recovery_packets",
        "execution_health_recovery_closure_target_after_action_learning_packets",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_recent_action_runs",
        "execution_learning_failed_or_blocked_action_runs",
        "execution_learning_recent_verification_runs",
        "execution_learning_recent_recovery_runs",
        "execution_learning_recent_after_action_learning_runs",
        "execution_learning_target_run_id",
        "execution_learning_target_tool_name",
        "execution_learning_target_after_action_learning_packets",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_closure_command",
        "execution_learning_evidence_command",
        "execution_learning_after_action_learning_command",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_learning_actionable_required_commands",
        "execution_learning_actionable_required_command_count",
        "execution_learning_actionable_proof_queue",
        "execution_learning_actionable_proof_queue_count",
        "execution_learning_actionable_next_required_command",
        "execution_learning_actionable_next_proof_command",
        "execution_learning_next_evidence_command",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff field diverged for {key}: {metadata}")
    if handoff.get("incomplete_component_count") != len(handoff.get("incomplete_components", [])):
        raise SystemExit(f"{label} handoff missed incomplete component count derivation: {metadata}")
    if handoff.get("completion_blocker_count") != len(handoff.get("completion_blockers", [])):
        raise SystemExit(f"{label} handoff missed completion blocker count derivation: {metadata}")
    if handoff.get("completion_blockers_deduplicated") is not (
        len(handoff.get("completion_blockers", [])) == len(set(handoff.get("completion_blockers", [])))
    ):
        raise SystemExit(f"{label} handoff missed blocker dedupe derivation: {metadata}")
    if handoff.get("completion_claim_ready") is not (not bool(handoff.get("completion_blockers", []))):
        raise SystemExit(f"{label} handoff completion claim readiness diverged from blockers: {metadata}")
    proof_queue = handoff.get("completion_proof_queue") or []
    if handoff.get("completion_proof_queue_count") != len(proof_queue):
        raise SystemExit(f"{label} handoff missed proof queue count derivation: {metadata}")
    if proof_queue and handoff.get("completion_next_proof_command") != proof_queue[0]:
        raise SystemExit(f"{label} handoff missed first completion proof command: {metadata}")
    if proof_queue and handoff.get("next_completion_proof_command") != proof_queue[0]:
        raise SystemExit(f"{label} handoff missed next completion proof alias: {metadata}")
    next_proofs = handoff.get("next_proof_commands") or []
    if next_proofs and handoff.get("next_proof_command") != next_proofs[0]:
        raise SystemExit(f"{label} handoff missed first next proof command: {metadata}")
    assert_selected_agi_target_readiness(handoff, f"{label} handoff")
    assert_execution_health_recovery_proof_aliases(handoff, f"{label} handoff")
    assert_execution_learning_proof_aliases(handoff, f"{label} handoff")
    assert_execution_learning_actionable_aliases(handoff, f"{label} handoff")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} handoff unsafe metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} handoff should not grant approval: {metadata}")


def assert_completion_audit_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("completion_audit_handoff") or {}
    if handoff.get("source") != "completion_audit_packet":
        raise SystemExit(f"{label} missed completion audit handoff source: {metadata}")
    for key in [
        "objective",
        "requirements",
        "strong_evidence",
        "partial_evidence",
        "missing_evidence",
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "approval_held_review_command",
        "next_proof_commands",
        "next_proof_command",
        "completion_proof_queue",
        "completion_proof_queue_count",
        "completion_next_proof_command",
        "next_completion_proof_command",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_ready_to_retry",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_checklist_command",
        "execution_health_recovery_closure_should_open_checklist",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_health_recovery_closure_target_run_id",
        "execution_health_recovery_closure_target_tool_name",
        "execution_health_recovery_closure_target_verification_receipts",
        "execution_health_recovery_closure_target_recovery_packets",
        "execution_health_recovery_closure_target_after_action_learning_packets",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_recent_action_runs",
        "execution_learning_failed_or_blocked_action_runs",
        "execution_learning_recent_verification_runs",
        "execution_learning_recent_recovery_runs",
        "execution_learning_recent_after_action_learning_runs",
        "execution_learning_target_run_id",
        "execution_learning_target_tool_name",
        "execution_learning_target_after_action_learning_packets",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_closure_command",
        "execution_learning_evidence_command",
        "execution_learning_after_action_learning_command",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_learning_actionable_required_commands",
        "execution_learning_actionable_required_command_count",
        "execution_learning_actionable_proof_queue",
        "execution_learning_actionable_proof_queue_count",
        "execution_learning_actionable_next_required_command",
        "execution_learning_actionable_next_proof_command",
        "execution_learning_next_evidence_command",
        "agi_gates",
        "agi_strong_gates",
        "agi_partial_gates",
        "agi_missing_gates",
        "agi_gate_statuses",
        "agi_missing_evidence_by_gate",
        "agi_real_execution_gaps_by_gate",
        "agi_real_execution_gap_count",
        "agi_next_moves_by_gate",
        "agi_evidence_closure_commands_by_gate",
        "agi_focused_verification_by_gate",
        "agi_next_gate",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_file_integrity_status",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_missing_target_files",
        "agi_next_missing_target_file_count",
        "agi_next_target_integrity_blocks_start",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_acceptance_gap_preview",
        "agi_next_acceptance_gap_preview_count",
        "agi_next_first_acceptance_gap",
        "agi_next_build_packet_ready_for_review",
        "completion_claim_ready",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff field diverged for {key}: {metadata}")
    evidence_total = (
        handoff.get("strong_evidence", 0)
        + handoff.get("partial_evidence", 0)
        + handoff.get("missing_evidence", 0)
    )
    if handoff.get("requirements") != evidence_total:
        raise SystemExit(f"{label} handoff evidence counts do not add to requirements: {metadata}")
    proof_queue = handoff.get("completion_proof_queue") or []
    if handoff.get("completion_proof_queue_count") != len(proof_queue):
        raise SystemExit(f"{label} handoff missed proof queue count derivation: {metadata}")
    if proof_queue and handoff.get("completion_next_proof_command") != proof_queue[0]:
        raise SystemExit(f"{label} handoff missed first completion proof command: {metadata}")
    if proof_queue and handoff.get("next_completion_proof_command") != proof_queue[0]:
        raise SystemExit(f"{label} handoff missed next completion proof alias: {metadata}")
    next_proofs = handoff.get("next_proof_commands") or []
    if next_proofs and handoff.get("next_proof_command") != next_proofs[0]:
        raise SystemExit(f"{label} handoff missed first next proof command: {metadata}")
    expected_claim_ready = not (
        handoff.get("partial_evidence")
        or handoff.get("missing_evidence")
        or handoff.get("pending_approvals")
        or handoff.get("open_tasks")
        or handoff.get("agi_real_execution_blocks_completion_claim")
        or handoff.get("recent_approval_held_runs")
        or handoff.get("execution_health_recovery_closure_blocks_completion_claim")
        or handoff.get("execution_learning_blocks_completion_claim")
    )
    if handoff.get("completion_claim_ready") is not expected_claim_ready:
        raise SystemExit(f"{label} handoff completion claim readiness diverged: {metadata}")
    assert_selected_agi_target_readiness(handoff, f"{label} handoff")
    assert_execution_health_recovery_proof_aliases(handoff, f"{label} handoff")
    assert_execution_learning_proof_aliases(handoff, f"{label} handoff")
    assert_execution_learning_actionable_aliases(handoff, f"{label} handoff")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} handoff unsafe metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} handoff should not grant approval: {metadata}")


def assert_agi_next_build_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("agi_next_build_handoff") or {}
    if handoff.get("source") != "agi_next_build_move":
        raise SystemExit(f"{label} missed AGI next-build handoff source: {metadata}")
    for key in [
        "selected_gate",
        "gate_status",
        "target_title",
        "agi_next_gate",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_file_integrity_status",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_missing_target_files",
        "agi_next_missing_target_file_count",
        "agi_next_target_integrity_blocks_start",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_acceptance_gap_preview",
        "agi_next_acceptance_gap_preview_count",
        "agi_next_first_acceptance_gap",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_build_packet_ready_for_review",
        "present_evidence",
        "present_evidence_count",
        "missing_evidence",
        "missing_evidence_count",
        "real_execution_gap",
        "real_execution_gap_count",
        "selected_real_execution_gap",
        "next_safe_build_move",
        "likely_files",
        "likely_file_count",
        "target_file_integrity_status",
        "target_files_checked",
        "target_files_exist",
        "missing_target_files",
        "missing_target_file_count",
        "target_file_rows",
        "target_integrity_blocks_start",
        "focused_tests",
        "focused_test_count",
        "acceptance_checks",
        "acceptance_check_count",
        "build_packet_ready_for_review",
        "pending_approvals",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "approval_held_review_command",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_ready_to_retry",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_checklist_command",
        "execution_health_recovery_closure_should_open_checklist",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_health_recovery_closure_target_run_id",
        "execution_health_recovery_closure_target_tool_name",
        "execution_health_recovery_closure_target_verification_receipts",
        "execution_health_recovery_closure_target_recovery_packets",
        "execution_health_recovery_closure_target_after_action_learning_packets",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_recent_action_runs",
        "execution_learning_failed_or_blocked_action_runs",
        "execution_learning_recent_verification_runs",
        "execution_learning_recent_recovery_runs",
        "execution_learning_recent_after_action_learning_runs",
        "execution_learning_target_run_id",
        "execution_learning_target_tool_name",
        "execution_learning_target_after_action_learning_packets",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_learning_evidence_command",
        "execution_learning_after_action_learning_command",
        "execution_learning_actionable_required_commands",
        "execution_learning_actionable_required_command_count",
        "execution_learning_actionable_proof_queue",
        "execution_learning_actionable_proof_queue_count",
        "execution_learning_actionable_next_required_command",
        "execution_learning_actionable_next_proof_command",
        "execution_learning_next_evidence_command",
        "agi_gates",
        "agi_gate_statuses",
        "agi_real_execution_gap_count",
        "completion_audit_command",
        "evidence_ledger_command",
        "completion_claim_gate_command",
        "proof_handoff_commands",
        "proof_handoff_command_count",
        "evidence_closure_commands",
        "evidence_closure_command_count",
        "focused_verification_commands",
        "focused_verification_command_count",
        "next_command",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} AGI next-build handoff field {key} diverged: {metadata}")
    if metadata.get("present_evidence_count") != len(metadata.get("present_evidence") or []):
        raise SystemExit(f"{label} AGI next-build present evidence count diverged: {metadata}")
    if metadata.get("missing_evidence_count") != len(metadata.get("missing_evidence") or []):
        raise SystemExit(f"{label} AGI next-build missing evidence count diverged: {metadata}")
    if metadata.get("likely_file_count") != len(metadata.get("likely_files") or []):
        raise SystemExit(f"{label} AGI next-build likely file count diverged: {metadata}")
    if metadata.get("target_files_checked") != len(metadata.get("target_file_rows") or []):
        raise SystemExit(f"{label} AGI next-build target row count diverged: {metadata}")
    if metadata.get("missing_target_file_count") != len(metadata.get("missing_target_files") or []):
        raise SystemExit(f"{label} AGI next-build missing target count diverged: {metadata}")
    if metadata.get("focused_test_count") != len(metadata.get("focused_tests") or []):
        raise SystemExit(f"{label} AGI next-build focused test count diverged: {metadata}")
    if metadata.get("focused_verification_command_count") != len(metadata.get("focused_verification_commands") or []):
        raise SystemExit(f"{label} AGI next-build focused verification count diverged: {metadata}")
    if metadata.get("acceptance_check_count") != len(metadata.get("acceptance_checks") or []):
        raise SystemExit(f"{label} AGI next-build acceptance count diverged: {metadata}")
    if metadata.get("proof_handoff_command_count") != len(metadata.get("proof_handoff_commands") or []):
        raise SystemExit(f"{label} AGI next-build proof handoff count diverged: {metadata}")
    if metadata.get("evidence_closure_command_count") != len(metadata.get("evidence_closure_commands") or []):
        raise SystemExit(f"{label} AGI next-build evidence closure count diverged: {metadata}")
    if metadata.get("execution_health_recovery_closure_proof_queue") != metadata.get("execution_health_recovery_closure_required_commands"):
        raise SystemExit(f"{label} AGI next-build recovery proof queue diverged: {metadata}")
    if metadata.get("execution_health_recovery_closure_proof_queue_count") != len(metadata.get("execution_health_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} AGI next-build recovery proof count diverged: {metadata}")
    if metadata.get("execution_learning_proof_queue") != metadata.get("execution_learning_required_commands"):
        raise SystemExit(f"{label} AGI next-build learning proof queue diverged: {metadata}")
    if metadata.get("execution_learning_proof_queue_count") != len(metadata.get("execution_learning_proof_queue") or []):
        raise SystemExit(f"{label} AGI next-build learning proof count diverged: {metadata}")
    if metadata.get("execution_learning_actionable_proof_queue") != metadata.get("execution_learning_actionable_required_commands"):
        raise SystemExit(f"{label} AGI next-build actionable learning proof queue diverged: {metadata}")
    if metadata.get("execution_learning_actionable_proof_queue_count") != len(metadata.get("execution_learning_actionable_proof_queue") or []):
        raise SystemExit(f"{label} AGI next-build actionable learning proof count diverged: {metadata}")
    expected_ready = bool(
        metadata.get("target_files_exist")
        and metadata.get("focused_tests")
        and metadata.get("acceptance_checks")
    )
    if metadata.get("build_packet_ready_for_review") is not expected_ready:
        raise SystemExit(f"{label} AGI next-build readiness flag diverged: {metadata}")
    if bool(metadata.get("target_integrity_blocks_start")) is metadata.get("target_files_exist"):
        raise SystemExit(f"{label} AGI next-build target integrity blocker should invert target_files_exist: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} AGI next-build handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} AGI next-build handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} AGI next-build handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} AGI next-build handoff should not grant approval: {metadata}")
    if metadata.get("agi_next_gate") != metadata.get("selected_gate"):
        raise SystemExit(f"{label} AGI next-build selected gate aliases diverged: {metadata}")
    if metadata.get("agi_next_target_title") != metadata.get("target_title"):
        raise SystemExit(f"{label} AGI next-build target title aliases diverged: {metadata}")
    if metadata.get("agi_next_build_command") != metadata.get("next_command"):
        raise SystemExit(f"{label} AGI next-build command alias diverged: {metadata}")
    if metadata.get("real_execution_gap_count") != metadata.get("agi_real_execution_gap_count"):
        raise SystemExit(f"{label} AGI next-build real-execution gap count aliases diverged: {metadata}")
    if metadata.get("selected_real_execution_gap") != metadata.get("real_execution_gap"):
        raise SystemExit(f"{label} AGI next-build selected real-execution gap alias diverged: {metadata}")
    if metadata.get("selected_real_execution_gap_gate") != metadata.get("selected_gate"):
        raise SystemExit(f"{label} AGI next-build selected real-execution gap gate alias diverged: {metadata}")
    if metadata.get("selected_real_execution_gap_detail") != metadata.get("real_execution_gap"):
        raise SystemExit(f"{label} AGI next-build selected real-execution gap detail alias diverged: {metadata}")
    if metadata.get("agi_next_likely_files") != metadata.get("likely_files"):
        raise SystemExit(f"{label} AGI next-build likely-file aliases diverged: {metadata}")
    if metadata.get("agi_next_target_file_integrity_status") != metadata.get("target_file_integrity_status"):
        raise SystemExit(f"{label} AGI next-build target-integrity aliases diverged: {metadata}")
    if metadata.get("agi_next_focused_verification_commands") != metadata.get("focused_verification_commands"):
        raise SystemExit(f"{label} AGI next-build focused-verification aliases diverged: {metadata}")
    if metadata.get("agi_next_acceptance_checks") != metadata.get("acceptance_checks"):
        raise SystemExit(f"{label} AGI next-build acceptance aliases diverged: {metadata}")
    if metadata.get("agi_next_evidence_closure_commands") != metadata.get("evidence_closure_commands"):
        raise SystemExit(f"{label} AGI next-build closure aliases diverged: {metadata}")
    assert_selected_agi_target_readiness(metadata, label)
    assert_selected_agi_target_readiness(handoff, f"{label} handoff")


def assert_agi_next_build_redacts_path_shaped_target_config() -> None:
    original_target = dict(harness_module.AGI_GATE_BUILD_TARGETS["personal integrations"])
    secret_marker = "AGI_HARNESS_PATH_SECRET"
    try:
        harness_module.AGI_GATE_BUILD_TARGETS["personal integrations"] = {
            **original_target,
            "title": f"Review /\x55sers/example/{secret_marker}/integration target",
            "files": [
                f"/\x55sers/example/{secret_marker}/jarvis_v2/tools/live_connector.py",
                "jarvis_v2/tools/harness.py",
            ],
            "tests": [
                f"python3 -m py_compile /\x55sers/example/{secret_marker}/tool.py",
                "python3 -m jarvis_v2.scripts.smoke_test_harness",
            ],
            "acceptance": [
                f"missing file /private/tmp/{secret_marker}/proof.txt is redacted before handoff",
                "relative target files remain visible for implementation review",
            ],
        }
        with TemporaryDirectory(prefix="jarvis-harness-agi-target-redaction-") as temp:
            runtime = make_temp_runtime(Path(temp))
            result = runtime.handle("agi next build move: personal integrations")
            if not result.verified:
                raise SystemExit(f"Path-shaped AGI target packet should stay read-only/verified: {result.response}")
            metadata = result.tool_results[0].metadata
            combined = result.response + json.dumps(metadata, sort_keys=True)
            for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/", secret_marker]:
                if forbidden in combined:
                    raise SystemExit(f"AGI next build leaked path-shaped target data {forbidden!r}: {combined}")
            if "<local-path>" not in combined:
                raise SystemExit(f"AGI next build should preserve a redacted local-path marker: {combined}")
            if metadata.get("target_title") != "Review <local-path>":
                raise SystemExit(f"AGI next build should redact path-shaped target title: {metadata}")
            if metadata.get("likely_files") != ["<local-path>", "jarvis_v2/tools/harness.py"]:
                raise SystemExit(f"AGI next build should redact only path-shaped likely files: {metadata}")
            if metadata.get("target_file_integrity_status") != "TARGETS_STALE":
                raise SystemExit(f"AGI next build should mark path-shaped target files stale: {metadata}")
            if metadata.get("target_files_exist") is not False or metadata.get("target_integrity_blocks_start") is not True:
                raise SystemExit(f"AGI next build should block start when a target path is redacted/stale: {metadata}")
            if metadata.get("missing_target_files") != ["<local-path>"]:
                raise SystemExit(f"AGI next build should report redacted missing target files: {metadata}")
            rows = metadata.get("target_file_rows") or []
            if rows != [{"path": "<local-path>", "exists": False}, {"path": "jarvis_v2/tools/harness.py", "exists": True}]:
                raise SystemExit(f"AGI next build should preserve relative integrity rows without leaking absolute paths: {metadata}")
            if metadata.get("focused_tests", [None])[0] != "python3 -m py_compile <local-path>":
                raise SystemExit(f"AGI next build should redact path-shaped focused tests: {metadata}")
            if not str(metadata.get("acceptance_checks", [""])[0]).startswith("missing file <local-path>"):
                raise SystemExit(f"AGI next build should redact path-shaped acceptance checks: {metadata}")
            if metadata.get("build_packet_ready_for_review") is not False:
                raise SystemExit(f"AGI next build should not be ready while target integrity is stale: {metadata}")
            assert_agi_next_build_handoff(metadata, "path-shaped AGI next build")
            tools = runtime.registry.list()
            tool_names = {tool.name for tool in tools}
            gate_summary = harness_module._agi_gate_summary(tool_names)
            selected_readiness = harness_module._selected_agi_target_readiness(gate_summary, "personal integrations")
            readiness_text = json.dumps(selected_readiness, sort_keys=True)
            for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/", secret_marker]:
                if forbidden in readiness_text:
                    raise SystemExit(f"Selected AGI readiness leaked path-shaped target data {forbidden!r}: {readiness_text}")
            if "<local-path>" not in readiness_text:
                raise SystemExit(f"Selected AGI readiness should preserve a redacted local-path marker: {readiness_text}")
            if selected_readiness.get("target_title") != "Review <local-path>":
                raise SystemExit(f"Selected AGI readiness should redact target title: {selected_readiness}")
            if selected_readiness.get("target", {}).get("title") != "Review <local-path>":
                raise SystemExit(f"Selected AGI readiness nested target should expose only safe title text: {selected_readiness}")
            if selected_readiness.get("target", {}).get("files") != ["<local-path>", "jarvis_v2/tools/harness.py"]:
                raise SystemExit(f"Selected AGI readiness nested target should redact path-shaped files: {selected_readiness}")
            if selected_readiness.get("verification_commands", [None])[0] != "python3 -m py_compile <local-path>":
                raise SystemExit(f"Selected AGI readiness should redact path-shaped verification commands: {selected_readiness}")
            if selected_readiness.get("target", {}).get("tests", [None])[0] != "python3 -m py_compile <local-path>":
                raise SystemExit(f"Selected AGI readiness nested target should redact path-shaped tests: {selected_readiness}")
            if selected_readiness.get("file_integrity", {}).get("missing") != ["<local-path>"]:
                raise SystemExit(f"Selected AGI readiness should report redacted missing target files: {selected_readiness}")
            for flag in READ_ONLY_FLAGS:
                if metadata.get(flag):
                    raise SystemExit(f"Path-shaped AGI next build should remain read-only for {flag}: {metadata}")
    finally:
        harness_module.AGI_GATE_BUILD_TARGETS["personal integrations"] = original_target


def assert_agi_target_config_malformed_entries_fail_closed() -> None:
    original_target = harness_module.AGI_GATE_BUILD_TARGETS["personal integrations"]
    secret_marker = "AGI_MALFORMED_TARGET_SECRET"

    class ExplodingTarget:
        def get(self, *_args, **_kwargs):
            raise RuntimeError(f"/\x55sers/example/{secret_marker}/get should not be called")

        def __iter__(self):
            raise RuntimeError(f"/private/tmp/{secret_marker}/iter should not be called")

        def __str__(self):
            raise RuntimeError(f"/var/folders/{secret_marker}/str should not be called")

    try:
        harness_module.AGI_GATE_BUILD_TARGETS["personal integrations"] = ExplodingTarget()
        with TemporaryDirectory(prefix="jarvis-harness-malformed-agi-target-") as temp:
            runtime = make_temp_runtime(Path(temp))
            gate_result = runtime.handle("agi gates")
            if not gate_result.verified:
                raise SystemExit(f"Malformed AGI target config should not break gate report: {gate_result.response}")
            build_result = runtime.handle("agi next build move: personal integrations")
            if not build_result.verified:
                raise SystemExit(f"Malformed AGI target config should not break next-build packet: {build_result.response}")
            metadata = build_result.tool_results[0].metadata
            combined = (
                gate_result.response
                + build_result.response
                + json.dumps(gate_result.tool_results[0].metadata, sort_keys=True)
                + json.dumps(metadata, sort_keys=True)
            )
            for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/", secret_marker]:
                if forbidden in combined:
                    raise SystemExit(f"Malformed AGI target config leaked unsafe diagnostics {forbidden!r}: {combined}")
            if metadata.get("likely_files") != [] or metadata.get("focused_tests") != [] or metadata.get("acceptance_checks") != []:
                raise SystemExit(f"Malformed AGI target config should fail closed to empty target fields: {metadata}")
            if metadata.get("build_packet_ready_for_review") is not False:
                raise SystemExit(f"Malformed AGI target config should not be ready for implementation: {metadata}")
            if metadata.get("target_title") != metadata.get("next_safe_build_move"):
                raise SystemExit(f"Malformed AGI target config should fall back to the gate next move title: {metadata}")
            tool_names = {tool.name for tool in runtime.registry.list()}
            selected_readiness = harness_module._selected_agi_target_readiness(
                harness_module._agi_gate_summary(tool_names),
                "personal integrations",
            )
            readiness_text = json.dumps(selected_readiness, sort_keys=True)
            for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/", secret_marker]:
                if forbidden in readiness_text:
                    raise SystemExit(f"Malformed selected AGI readiness leaked unsafe diagnostics {forbidden!r}: {readiness_text}")
            if selected_readiness.get("target", {}).get("files") != []:
                raise SystemExit(f"Malformed selected AGI readiness should expose empty safe file list: {selected_readiness}")
            if selected_readiness.get("target", {}).get("tests") != []:
                raise SystemExit(f"Malformed selected AGI readiness should expose empty safe test list: {selected_readiness}")
            if selected_readiness.get("target", {}).get("acceptance") != []:
                raise SystemExit(f"Malformed selected AGI readiness should expose empty safe acceptance list: {selected_readiness}")
            if selected_readiness.get("build_ready") is not False:
                raise SystemExit(f"Malformed selected AGI readiness should not be build-ready: {selected_readiness}")
    finally:
        harness_module.AGI_GATE_BUILD_TARGETS["personal integrations"] = original_target


def assert_completion_proof_refresh_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("completion_proof_refresh_handoff") or {}
    if handoff.get("source") != "completion_proof_refresh_packet":
        raise SystemExit(f"{label} missed completion proof refresh handoff source: {metadata}")
    if metadata.get("completion_proof_refresh_handoff_ready") is not True or handoff.get("completion_proof_refresh_handoff_ready") is not True:
        raise SystemExit(f"{label} missed completion proof refresh handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested completion proof refresh handoff readiness: {metadata}")
    for key in [
        "objective",
        "claim_gate_verdict",
        "completion_claim_ready",
        "allowed_to_claim",
        "stale_lane_count",
        "stale_lane_names",
        "available_without_recent_evidence_lanes",
        "available_without_recent_evidence_lane_count",
        "partial_evidence_lanes",
        "partial_evidence_lane_count",
        "missing_evidence_lanes",
        "missing_evidence_lane_count",
        "lane_current_evidence_worklist",
        "lane_current_evidence_worklist_count",
        "lane_count",
        "lane_rows",
        "lane_status",
        "lane_refresh_commands",
        "lane_refresh_command_count",
        "ordered_refresh_queue",
        "ordered_refresh_queue_count",
        "resolved_refresh_queue",
        "resolved_refresh_queue_count",
        "first_refresh_command",
        "first_resolved_refresh_command",
        "first_lane_refresh_command",
        "first_resolved_lane_refresh_command",
        "concrete_refresh_commands",
        "concrete_refresh_command_count",
        "placeholder_refresh_commands",
        "placeholder_refresh_command_count",
        "placeholder_resolution_rows",
        "placeholder_resolution_count",
        "actionable_placeholder_count",
        "unresolved_placeholder_count",
        "completion_proof_queue",
        "completion_proof_queue_count",
        "storage_runtime_fallback_active",
        "storage_runtime_fallback_reason",
        "storage_runtime_fallback_exception_type",
        "storage_runtime_fallback_db_path_display",
        "storage_runtime_fallback_vault_path_display",
        "storage_readiness_blocks_completion_claim",
        "storage_readiness_blocker",
        "storage_readiness_next_commands",
        "storage_readiness_next_command_count",
        "storage_readiness_next_proof_command",
        "storage_readiness_proof_queue",
        "storage_readiness_proof_queue_count",
        "storage_readiness_first_proof_command",
        "storage_recovery_required",
        "storage_recovery_reason",
        "storage_recovery_mode",
        "storage_recovery_next_operator_action",
        "storage_recovery_restart_required",
        "storage_recovery_check_tool_command",
        "storage_recovery_check_command",
        "storage_recovery_check_api",
        "storage_recovery_command",
        "storage_issues",
        "storage_issue_count",
        "storage_handoff",
        "next_proof_commands",
        "agi_next_gate",
        "agi_real_execution_gap_count",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_file_integrity_status",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_missing_target_files",
        "agi_next_missing_target_file_count",
        "agi_next_target_integrity_blocks_start",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_acceptance_gap_preview",
        "agi_next_acceptance_gap_preview_count",
        "agi_next_first_acceptance_gap",
        "agi_next_build_packet_ready_for_review",
        "blocker_details",
        "completion_blocker_count",
        "refresh_after_command",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} completion proof refresh handoff field {key} diverged: {metadata}")
    if handoff.get("lane_count") != len(handoff.get("lane_rows") or []):
        raise SystemExit(f"{label} completion proof refresh handoff lane count diverged: {metadata}")
    if handoff.get("lane_current_evidence_worklist_count") != len(handoff.get("lane_current_evidence_worklist") or []):
        raise SystemExit(f"{label} completion proof refresh handoff worklist count diverged: {metadata}")
    if handoff.get("stale_lane_count") != len(handoff.get("stale_lane_names") or []):
        raise SystemExit(f"{label} completion proof refresh handoff stale lane count diverged: {metadata}")
    if handoff.get("ordered_refresh_queue_count") != len(handoff.get("ordered_refresh_queue") or []):
        raise SystemExit(f"{label} completion proof refresh handoff ordered queue count diverged: {metadata}")
    if handoff.get("resolved_refresh_queue_count") != len(handoff.get("resolved_refresh_queue") or []):
        raise SystemExit(f"{label} completion proof refresh handoff resolved queue count diverged: {metadata}")
    if handoff.get("completion_proof_queue_count") != len(handoff.get("completion_proof_queue") or []):
        raise SystemExit(f"{label} completion proof refresh handoff proof queue count diverged: {metadata}")
    if handoff.get("placeholder_resolution_count") != len(handoff.get("placeholder_resolution_rows") or []):
        raise SystemExit(f"{label} completion proof refresh handoff placeholder count diverged: {metadata}")
    if handoff.get("first_refresh_command") != (handoff.get("ordered_refresh_queue") or [""])[0]:
        raise SystemExit(f"{label} completion proof refresh handoff first command diverged: {metadata}")
    if handoff.get("first_resolved_refresh_command") != (handoff.get("resolved_refresh_queue") or [""])[0]:
        raise SystemExit(f"{label} completion proof refresh handoff first resolved command diverged: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} completion proof refresh handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} completion proof refresh handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} completion proof refresh handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} completion proof refresh handoff should not grant approval: {metadata}")
    if handoff.get("completion_claim_ready") is not False or handoff.get("allowed_to_claim") is not False:
        raise SystemExit(f"{label} completion proof refresh handoff should preserve blocked claim state: {metadata}")


def assert_completion_next_proof_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("completion_next_proof_handoff") or {}
    if handoff.get("source") != "completion_next_proof_packet":
        raise SystemExit(f"{label} missed completion next proof handoff source: {metadata}")
    if metadata.get("completion_next_proof_handoff_ready") is not True or handoff.get("completion_next_proof_handoff_ready") is not True:
        raise SystemExit(f"{label} missed completion next proof handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested completion next proof handoff readiness: {metadata}")
    for key in [
        "objective",
        "claim_gate_verdict",
        "completion_claim_ready",
        "allowed_to_claim",
        "blockers",
        "blocker_details",
        "completion_proof_queue",
        "completion_proof_queue_count",
        "completion_ordered_proof_queue",
        "completion_ordered_proof_queue_count",
        "completion_ordered_next_proof_command",
        "completion_actionable_proof_queue",
        "completion_actionable_proof_queue_count",
        "completion_actionable_next_proof_command",
        "completion_actionable_queue_includes_learning_prerequisites",
        "completion_next_proof_uses_actionable_learning_prerequisite",
        "proof_queue",
        "proof_queue_count",
        "next_completion_proof_command",
        "completion_next_proof_command",
        "next_proof_command",
        "command_source",
        "command_reason",
        "first_blocker",
        "queue_position",
        "queue_total",
        "ordered_queue_position",
        "ordered_queue_total",
        "refresh_command",
        "latest_execution_case_found",
        "latest_execution_case_id",
        "latest_execution_case_closure_verdict",
        "latest_execution_case_closure_ready",
        "latest_execution_case_closure_blocks_completion_claim",
        "latest_execution_case_closure_proof_queue",
        "latest_execution_case_closure_proof_queue_count",
        "latest_execution_case_next_closure_proof_command",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_learning_state",
        "execution_learning_missing",
        "execution_learning_closure_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_learning_actionable_required_commands",
        "execution_learning_actionable_required_command_count",
        "execution_learning_actionable_proof_queue",
        "execution_learning_actionable_proof_queue_count",
        "execution_learning_actionable_next_required_command",
        "execution_learning_actionable_next_proof_command",
        "execution_learning_next_evidence_command",
        "execution_learning_blocks_completion_claim",
        "storage_runtime_fallback_active",
        "storage_runtime_fallback_reason",
        "storage_runtime_fallback_exception_type",
        "storage_runtime_fallback_db_path_display",
        "storage_runtime_fallback_vault_path_display",
        "storage_readiness_blocks_completion_claim",
        "storage_recovery_required",
        "storage_recovery_reason",
        "storage_issues",
        "storage_issue_count",
        "storage_recovery_mode",
        "storage_recovery_next_operator_action",
        "storage_recovery_restart_required",
        "storage_recovery_check_command",
        "storage_recovery_check_tool_command",
        "storage_recovery_check_api",
        "storage_recovery_command",
        "storage_readiness_blocker",
        "storage_readiness_next_commands",
        "storage_readiness_next_command_count",
        "storage_readiness_next_required_command",
        "storage_readiness_next_proof_command",
        "storage_readiness_proof_queue",
        "storage_readiness_proof_queue_count",
        "storage_readiness_first_proof_command",
        "storage_handoff",
        "agi_next_gate",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_file_integrity_status",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_missing_target_files",
        "agi_next_missing_target_file_count",
        "agi_next_target_integrity_blocks_start",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_acceptance_gap_preview",
        "agi_next_acceptance_gap_preview_count",
        "agi_next_first_acceptance_gap",
        "agi_next_build_packet_ready_for_review",
        "agi_next_implementation_preflight_ready",
        "agi_next_implementation_preflight_blockers",
        "agi_next_implementation_preflight_blocker_count",
        "agi_next_implementation_preflight_next_command",
        "agi_real_execution_gap_count",
        "real_execution_gap_count",
        "selected_real_execution_gap",
        "selected_real_execution_gap_gate",
        "selected_real_execution_gap_detail",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} completion next proof handoff field {key} diverged: {metadata}")
    ordered_queue = handoff.get("completion_proof_queue") or []
    ordered_alias_queue = handoff.get("completion_ordered_proof_queue") or []
    actionable_queue = handoff.get("completion_actionable_proof_queue") or []
    if ordered_alias_queue != ordered_queue:
        raise SystemExit(f"{label} completion next proof handoff ordered queue alias diverged: {metadata}")
    if handoff.get("completion_ordered_proof_queue_count") != len(ordered_queue):
        raise SystemExit(f"{label} completion next proof handoff ordered queue count diverged: {metadata}")
    if handoff.get("completion_proof_queue_count") != len(ordered_queue):
        raise SystemExit(f"{label} completion next proof handoff proof queue count diverged: {metadata}")
    if handoff.get("completion_actionable_proof_queue_count") != len(actionable_queue):
        raise SystemExit(f"{label} completion next proof handoff actionable queue count diverged: {metadata}")
    if handoff.get("proof_queue") != actionable_queue:
        raise SystemExit(f"{label} completion next proof handoff missed generic proof queue alias: {metadata}")
    if handoff.get("proof_queue_count") != len(actionable_queue):
        raise SystemExit(f"{label} completion next proof handoff missed generic proof queue count: {metadata}")
    if ordered_queue and handoff.get("completion_ordered_next_proof_command") != ordered_queue[0]:
        raise SystemExit(f"{label} completion next proof handoff missed ordered first proof command: {metadata}")
    if actionable_queue:
        for key in ["next_completion_proof_command", "completion_next_proof_command", "next_proof_command"]:
            if handoff.get(key) != actionable_queue[0]:
                raise SystemExit(f"{label} completion next proof handoff missed first proof alias {key}: {metadata}")
        if handoff.get("completion_actionable_next_proof_command") != actionable_queue[0]:
            raise SystemExit(f"{label} completion next proof handoff missed actionable first proof command: {metadata}")
    if handoff.get("completion_actionable_queue_includes_learning_prerequisites"):
        learning_actionable_queue = handoff.get("execution_learning_actionable_proof_queue") or []
        learning_closure_command = handoff.get("execution_learning_closure_command")
        if not learning_actionable_queue or not learning_closure_command:
            raise SystemExit(f"{label} completion next proof handoff claimed inserted learning prerequisites without queue/closure command: {metadata}")
        if learning_closure_command not in ordered_queue:
            raise SystemExit(f"{label} completion next proof handoff inserted learning prerequisites but ordered queue missed closure command: {metadata}")
        closure_index = ordered_queue.index(learning_closure_command)
        ordered_prefix = ordered_queue[:closure_index]
        expected_inserted = [command for command in learning_actionable_queue if command not in ordered_prefix]
        inserted_index = next((index for index, command in enumerate(actionable_queue) if command in expected_inserted), -1)
        if inserted_index < 0:
            raise SystemExit(f"{label} completion next proof handoff missed inserted learning actionable commands: {metadata}")
        if actionable_queue[:inserted_index] != ordered_prefix:
            raise SystemExit(f"{label} completion next proof handoff should preserve ordered prefix before learning insertion: {metadata}")
        if actionable_queue[inserted_index : inserted_index + len(expected_inserted)] != expected_inserted:
            raise SystemExit(f"{label} completion next proof handoff should insert the learning actionable queue at closure position: {metadata}")
    if handoff.get("completion_next_proof_uses_actionable_learning_prerequisite"):
        if handoff.get("completion_ordered_next_proof_command") == handoff.get("next_proof_command"):
            raise SystemExit(f"{label} should distinguish ordered proof from actionable proof when learning prerequisites exist: {metadata}")
        if handoff.get("next_proof_command") != handoff.get("execution_learning_actionable_next_proof_command"):
            raise SystemExit(f"{label} actionable proof should point to next learning evidence command: {metadata}")
    if handoff.get("latest_execution_case_closure_proof_queue_count") != len(handoff.get("latest_execution_case_closure_proof_queue") or []):
        raise SystemExit(f"{label} completion next proof handoff case queue count diverged: {metadata}")
    if handoff.get("execution_health_recovery_closure_proof_queue_count") != len(handoff.get("execution_health_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} completion next proof handoff recovery queue count diverged: {metadata}")
    if handoff.get("execution_learning_proof_queue_count") != len(handoff.get("execution_learning_proof_queue") or []):
        raise SystemExit(f"{label} completion next proof handoff learning queue count diverged: {metadata}")
    if metadata.get("real_execution_gap_count") != metadata.get("agi_real_execution_gap_count"):
        raise SystemExit(f"{label} completion next proof real-execution alias count diverged: {metadata}")
    if handoff.get("real_execution_gap_count") != handoff.get("agi_real_execution_gap_count"):
        raise SystemExit(f"{label} completion next proof handoff real-execution alias count diverged: {metadata}")
    if handoff.get("real_execution_gap_count") != metadata.get("real_execution_gap_count"):
        raise SystemExit(f"{label} completion next proof handoff missed real-execution alias count: {metadata}")
    if not metadata.get("selected_real_execution_gap") or handoff.get("selected_real_execution_gap") != metadata.get("selected_real_execution_gap"):
        raise SystemExit(f"{label} completion next proof missed selected real-execution gap alias: {metadata}")
    if not metadata.get("selected_real_execution_gap_gate") or handoff.get("selected_real_execution_gap_gate") != metadata.get("selected_real_execution_gap_gate"):
        raise SystemExit(f"{label} completion next proof missed selected real-execution gap gate alias: {metadata}")
    if not metadata.get("selected_real_execution_gap_detail") or handoff.get("selected_real_execution_gap_detail") != metadata.get("selected_real_execution_gap_detail"):
        raise SystemExit(f"{label} completion next proof missed selected real-execution gap detail alias: {metadata}")
    if metadata.get("selected_real_execution_gap_gate") != metadata.get("selected_real_execution_gap"):
        raise SystemExit(f"{label} completion next proof gate alias should preserve the legacy selected gap selector: {metadata}")
    if metadata.get("selected_real_execution_gap_detail") == metadata.get("selected_real_execution_gap_gate"):
        raise SystemExit(f"{label} completion next proof detail alias should carry gap text, not just the gate selector: {metadata}")
    assert_execution_learning_actionable_aliases(metadata, label)
    assert_execution_learning_actionable_aliases(handoff, f"{label} handoff")
    if handoff.get("completion_claim_ready") is not False or handoff.get("allowed_to_claim") is not False:
        raise SystemExit(f"{label} completion next proof handoff should preserve blocked claim state: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} completion next proof handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} completion next proof handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} completion next proof handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} completion next proof handoff should not grant approval: {metadata}")


def assert_storage_handoff_contract(metadata: dict, label: str, expected_source: str) -> None:
    handoff = metadata.get("storage_handoff") or {}
    if handoff.get("source") != expected_source:
        raise SystemExit(f"{label} missed storage handoff source: {metadata}")
    if handoff.get("handoff_ready") is not True or handoff.get("storage_handoff_ready") is not True:
        raise SystemExit(f"{label} missed storage handoff readiness flags: {metadata}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} storage handoff should be operator-ready metadata: {metadata}")
    for key in [
        "storage_runtime_fallback_active",
        "storage_runtime_fallback_reason",
        "storage_runtime_fallback_exception_type",
        "storage_runtime_fallback_db_path_display",
        "storage_runtime_fallback_vault_path_display",
        "storage_readiness_blocks_completion_claim",
        "storage_readiness_blocker",
        "storage_readiness_next_commands",
        "storage_readiness_next_command_count",
        "storage_readiness_next_proof_command",
        "storage_readiness_proof_queue",
        "storage_readiness_proof_queue_count",
        "storage_readiness_first_proof_command",
        "storage_recovery_required",
        "storage_recovery_reason",
        "storage_recovery_mode",
        "storage_recovery_next_operator_action",
        "storage_recovery_restart_required",
        "storage_recovery_check_tool_command",
        "storage_recovery_check_command",
        "storage_recovery_check_api",
        "storage_recovery_command",
        "storage_issues",
        "storage_issue_count",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} storage handoff field {key} diverged: {metadata}")
    storage_commands = handoff.get("storage_readiness_next_commands") or []
    proof_queue = handoff.get("storage_readiness_proof_queue") or []
    if handoff.get("storage_readiness_next_command_count") != len(storage_commands):
        raise SystemExit(f"{label} storage handoff command count diverged: {metadata}")
    if handoff.get("storage_readiness_proof_queue_count") != len(proof_queue):
        raise SystemExit(f"{label} storage handoff proof queue count diverged: {metadata}")
    if proof_queue and handoff.get("storage_readiness_first_proof_command") != proof_queue[0]:
        raise SystemExit(f"{label} storage handoff first proof diverged from queue head: {metadata}")
    if proof_queue and handoff.get("storage_readiness_next_required_command") != proof_queue[0]:
        raise SystemExit(f"{label} storage handoff next required diverged from queue head: {metadata}")
    if proof_queue and handoff.get("storage_readiness_next_proof_command") != proof_queue[0]:
        raise SystemExit(f"{label} storage handoff next proof diverged from queue head: {metadata}")
    if handoff.get("next_commands") != (proof_queue or storage_commands):
        raise SystemExit(f"{label} storage handoff generic next commands diverged: {metadata}")
    if proof_queue and handoff.get("next_proof_command") != proof_queue[0]:
        raise SystemExit(f"{label} storage handoff missed generic next proof command: {metadata}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} storage handoff should not report state changes: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} storage handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} storage handoff should not grant approval: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    expected_boundaries = {
        "metadata_only": True,
        "reads_database_file": False,
        "reads_db_file_contents": False,
        "reads_vault_files": False,
        "scans_obsidian_vault": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "approves_request": False,
        "approves_requests": False,
        "dismisses_request": False,
        "dismisses_approvals": False,
        "controls_computer": False,
        "calls_external_service": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
    }
    for key, expected in expected_boundaries.items():
        if boundaries.get(key) is not expected:
            raise SystemExit(f"{label} storage handoff boundary {key} diverged: {metadata}")


def assert_evidence_ledger_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("evidence_ledger_handoff") or {}
    if handoff.get("source") != "evidence_ledger":
        raise SystemExit(f"{label} missed evidence ledger handoff source: {metadata}")
    for key in [
        "objective",
        "lanes",
        "strong_evidence",
        "partial_evidence",
        "missing_evidence",
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "recent_runs",
        "recent_successes",
        "recent_failures",
        "verification_runs",
        "after_action_learning_runs",
        "evidence_completion_runs",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_ready_to_retry",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_checklist_command",
        "execution_health_recovery_closure_should_open_checklist",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_health_recovery_closure_target_run_id",
        "execution_health_recovery_closure_target_tool_name",
        "execution_health_recovery_closure_target_verification_receipts",
        "execution_health_recovery_closure_target_recovery_packets",
        "execution_health_recovery_closure_target_after_action_learning_packets",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_recent_action_runs",
        "execution_learning_failed_or_blocked_action_runs",
        "execution_learning_recent_verification_runs",
        "execution_learning_recent_recovery_runs",
        "execution_learning_recent_after_action_learning_runs",
        "execution_learning_target_run_id",
        "execution_learning_target_tool_name",
        "execution_learning_target_after_action_learning_packets",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_learning_evidence_command",
        "execution_learning_after_action_learning_command",
        "execution_learning_actionable_required_commands",
        "execution_learning_actionable_required_command_count",
        "execution_learning_actionable_proof_queue",
        "execution_learning_actionable_proof_queue_count",
        "execution_learning_actionable_next_required_command",
        "execution_learning_actionable_next_proof_command",
        "execution_learning_next_evidence_command",
        "latest_execution_case_found",
        "latest_execution_case_id",
        "latest_execution_case_verdict",
        "latest_execution_case_ready",
        "latest_execution_case_closure_verdict",
        "latest_execution_case_closure_ready",
        "latest_execution_case_closure_blocks_completion_claim",
        "latest_execution_case_closure_proof_queue",
        "latest_execution_case_closure_proof_queue_count",
        "latest_execution_case_next_closure_proof_command",
        "latest_execution_case_missing_proofs",
        "latest_execution_case_next_proof_command",
        "latest_execution_case_next_proof_commands",
        "latest_execution_case_mission_command_queue",
        "latest_execution_case_mission_command_count",
        "latest_execution_case_evidence_preview_gate_count",
        "latest_execution_case_evidence_preview_ready_count",
        "latest_execution_case_evidence_preview_blocked_count",
        "latest_execution_case_evidence_preview_verdicts",
        "latest_execution_case_evidence_preview_latest_verdict",
        "latest_execution_case_evidence_preview_latest_event_id",
        "latest_execution_case_evidence_preview_latest_handoff",
        "latest_execution_case_evidence_preview_latest_handoff_present",
        "latest_execution_case_approval_queue_forecast",
        "latest_execution_case_forecast_new_approvals",
        "latest_execution_case_forecast_reused_approval_ids",
        "latest_execution_case_forecast_queue_before",
        "latest_execution_case_forecast_queue_after_if_sent",
        "latest_execution_case_forecast_queue_delta_if_sent",
        "latest_execution_case_recovery_closure_state",
        "latest_execution_case_recovery_closure_missing",
        "latest_execution_case_recovery_closure_next_required_command",
        "latest_execution_case_recovery_closure_required_commands",
        "latest_execution_case_recovery_closure_proof_queue",
        "latest_execution_case_recovery_closure_proof_queue_count",
        "latest_execution_case_recovery_closure_next_proof_command",
        "latest_execution_case_recovery_closure_blocks_completion_claim",
        "latest_execution_case_learning_state",
        "latest_execution_case_learning_missing",
        "latest_execution_case_learning_next_required_command",
        "latest_execution_case_learning_required_commands",
        "latest_execution_case_learning_proof_queue",
        "latest_execution_case_learning_proof_queue_count",
        "latest_execution_case_learning_next_proof_command",
        "latest_execution_case_learning_blocks_completion_claim",
        "next_proof_commands",
        "agi_gates",
        "agi_strong_gates",
        "agi_partial_gates",
        "agi_missing_gates",
        "agi_gate_statuses",
        "agi_missing_evidence_by_gate",
        "agi_real_execution_gaps_by_gate",
        "agi_real_execution_gap_count",
        "agi_next_moves_by_gate",
        "agi_evidence_closure_commands_by_gate",
        "agi_focused_verification_by_gate",
        "agi_next_gate",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_file_integrity_status",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_missing_target_files",
        "agi_next_missing_target_file_count",
        "agi_next_target_integrity_blocks_start",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_acceptance_gap_preview",
        "agi_next_acceptance_gap_preview_count",
        "agi_next_first_acceptance_gap",
        "agi_next_build_packet_ready_for_review",
        "claim_state",
        "completion_claim_ready",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff field diverged for {key}: {metadata}")
    if handoff.get("lanes") != handoff.get("strong_evidence", 0) + handoff.get("partial_evidence", 0) + handoff.get("missing_evidence", 0):
        raise SystemExit(f"{label} handoff lane evidence counts do not add up: {metadata}")
    if handoff.get("recent_runs") != handoff.get("recent_successes", 0) + handoff.get("recent_failures", 0):
        raise SystemExit(f"{label} handoff recent run counts diverged: {metadata}")
    if handoff.get("latest_execution_case_closure_proof_queue_count") != len(handoff.get("latest_execution_case_closure_proof_queue") or []):
        raise SystemExit(f"{label} handoff latest case closure queue count diverged: {metadata}")
    if handoff.get("latest_execution_case_mission_command_count") != len(handoff.get("latest_execution_case_mission_command_queue") or []):
        raise SystemExit(f"{label} handoff latest case mission queue count diverged: {metadata}")
    if handoff.get("latest_execution_case_recovery_closure_proof_queue_count") != len(handoff.get("latest_execution_case_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} handoff latest case recovery queue count diverged: {metadata}")
    if handoff.get("latest_execution_case_learning_proof_queue_count") != len(handoff.get("latest_execution_case_learning_proof_queue") or []):
        raise SystemExit(f"{label} handoff latest case learning queue count diverged: {metadata}")
    if handoff.get("latest_execution_case_evidence_preview_gate_count", 0) > 0:
        if handoff.get("latest_execution_case_evidence_preview_ready_count", 0) + handoff.get("latest_execution_case_evidence_preview_blocked_count", 0) != handoff.get("latest_execution_case_evidence_preview_gate_count"):
            raise SystemExit(f"{label} handoff evidence preview count diverged: {metadata}")
    if handoff.get("completion_claim_ready") is not (handoff.get("claim_state") == "READY_FOR_HUMAN_COMPLETION_REVIEW"):
        raise SystemExit(f"{label} handoff completion readiness diverged from claim state: {metadata}")
    if handoff.get("next_proof_commands", {}).get("steering") != "dispatch decision: <next real order>":
        raise SystemExit(f"{label} handoff missed steering proof command: {metadata}")
    assert_selected_agi_target_readiness(handoff, f"{label} handoff")
    assert_execution_health_recovery_proof_aliases(handoff, f"{label} handoff")
    assert_execution_learning_proof_aliases(handoff, f"{label} handoff")
    assert_execution_learning_actionable_aliases(handoff, f"{label} handoff")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} handoff unsafe metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} handoff should not grant approval: {metadata}")


def assert_completion_claim_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("completion_claim_handoff") or {}
    if handoff.get("source") != "completion_claim_gate":
        raise SystemExit(f"{label} missed completion claim handoff source: {metadata}")
    if metadata.get("completion_claim_handoff_ready") is not True or handoff.get("completion_claim_handoff_ready") is not True:
        raise SystemExit(f"{label} missed completion claim handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested completion claim handoff readiness: {metadata}")
    for key in [
        "objective",
        "proposed_claim",
        "verdict",
        "allowed_to_claim",
        "completion_claim_ready",
        "blockers",
        "completion_blocker_count",
        "blocker_details",
        "completion_blockers_deduplicated",
        "safe_claim",
        "completion_proof_queue",
        "completion_proof_queue_count",
        "completion_next_proof_command",
        "next_completion_proof_command",
        "next_proof_commands",
        "next_proof_command",
        "next_proof_command_count",
        "pending_approvals",
        "open_tasks",
        "active_goals",
        "execution_learning_blocks_completion_claim",
        "execution_health_recovery_closure_blocks_completion_claim",
        "storage_runtime_fallback_active",
        "storage_runtime_fallback_reason",
        "storage_runtime_fallback_exception_type",
        "storage_runtime_fallback_db_path_display",
        "storage_runtime_fallback_vault_path_display",
        "storage_readiness_blocks_completion_claim",
        "storage_recovery_required",
        "storage_recovery_reason",
        "storage_recovery_mode",
        "storage_recovery_next_operator_action",
        "storage_recovery_restart_required",
        "storage_recovery_check_command",
        "storage_recovery_check_api",
        "storage_recovery_command",
        "storage_readiness_blocker",
        "storage_readiness_next_commands",
        "storage_readiness_next_command_count",
        "storage_readiness_next_proof_command",
        "latest_execution_case_ready",
        "latest_execution_case_verdict",
        "latest_execution_case_closure_verdict",
        "latest_execution_case_closure_ready",
        "latest_execution_case_closure_blocks_completion_claim",
        "latest_execution_case_evidence_preview_gate_count",
        "latest_execution_case_evidence_preview_ready_count",
        "latest_execution_case_evidence_preview_blocked_count",
        "latest_execution_case_evidence_preview_verdicts",
        "latest_execution_case_evidence_preview_latest_verdict",
        "latest_execution_case_evidence_preview_latest_event_id",
        "latest_execution_case_evidence_preview_latest_handoff",
        "latest_execution_case_evidence_preview_latest_handoff_present",
        "agi_next_gate",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_file_integrity_status",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_missing_target_files",
        "agi_next_missing_target_file_count",
        "agi_next_target_integrity_blocks_start",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_acceptance_gap_preview",
        "agi_next_acceptance_gap_preview_count",
        "agi_next_first_acceptance_gap",
        "agi_next_build_packet_ready_for_review",
        "selected_real_execution_gap_gate",
        "selected_real_execution_gap_detail",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} completion claim handoff field {key} diverged: {metadata}")
    if handoff.get("blocker_count") != metadata.get("blockers"):
        raise SystemExit(f"{label} completion claim handoff blocker count diverged: {metadata}")
    if handoff.get("completion_proof_queue_count") != len(handoff.get("completion_proof_queue") or []):
        raise SystemExit(f"{label} completion claim handoff proof queue count diverged: {metadata}")
    storage_commands = handoff.get("storage_readiness_next_commands") or []
    if storage_commands:
        if handoff.get("storage_readiness_next_proof_command") != storage_commands[0]:
            raise SystemExit(f"{label} completion claim handoff missed first storage proof command: {metadata}")
    elif handoff.get("storage_readiness_next_proof_command") not in {"", None}:
        raise SystemExit(f"{label} completion claim handoff should not expose a storage proof command without storage queue: {metadata}")
    if handoff.get("next_proof_command_count") != len(handoff.get("next_proof_commands") or {}):
        raise SystemExit(f"{label} completion claim handoff next proof count diverged: {metadata}")
    proof_queue = handoff.get("completion_proof_queue") or []
    if proof_queue:
        if handoff.get("completion_next_proof_command") != proof_queue[0]:
            raise SystemExit(f"{label} completion claim handoff missed completion next proof command: {metadata}")
        if handoff.get("next_completion_proof_command") != proof_queue[0]:
            raise SystemExit(f"{label} completion claim handoff missed next completion proof command: {metadata}")
    if handoff.get("completion_claim_ready") is not False or handoff.get("allowed_to_claim") is not False:
        raise SystemExit(f"{label} completion claim handoff should preserve blocked claim state: {metadata}")
    if handoff.get("latest_execution_case_evidence_preview_gate_count", 0) > 0:
        if handoff.get("latest_execution_case_evidence_preview_ready_count", 0) + handoff.get("latest_execution_case_evidence_preview_blocked_count", 0) != handoff.get("latest_execution_case_evidence_preview_gate_count"):
            raise SystemExit(f"{label} completion claim handoff evidence preview count diverged: {metadata}")
        if handoff.get("latest_execution_case_evidence_preview_latest_handoff_present") is not True:
            raise SystemExit(f"{label} completion claim handoff missed evidence preview handoff flag: {metadata}")
        if (handoff.get("latest_execution_case_evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"{label} completion claim handoff missed evidence preview handoff source: {metadata}")
    if not handoff.get("agi_next_build_command"):
        raise SystemExit(f"{label} completion claim handoff missed selected AGI build command: {metadata}")
    if handoff.get("real_execution_gap_count") != handoff.get("agi_real_execution_gap_count"):
        raise SystemExit(f"{label} completion claim handoff missed neutral real-execution gap alias: {metadata}")
    if handoff.get("real_execution_gap_count") != metadata.get("real_execution_gap_count"):
        raise SystemExit(f"{label} completion claim handoff real-execution gap alias diverged: {metadata}")
    if not handoff.get("selected_real_execution_gap") or handoff.get("selected_real_execution_gap") != metadata.get("selected_real_execution_gap"):
        raise SystemExit(f"{label} completion claim handoff missed selected real-execution gap alias: {metadata}")
    assert_selected_agi_target_readiness(handoff, f"{label} completion claim handoff")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} completion claim handoff unsafe nested metadata {key}: {metadata}")


def assert_operator_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("operator_handoff") or {}
    if handoff.get("source") != "operator_handoff_packet":
        raise SystemExit(f"{label} missed operator handoff source: {metadata}")
    if metadata.get("operator_handoff_ready") is not True or handoff.get("operator_handoff_ready") is not True:
        raise SystemExit(f"{label} missed operator handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested operator handoff readiness: {metadata}")
    for key in [
        "source_tool",
        "objective",
        "next_operator_command",
        "command_source",
        "command_reason",
        "first_blocker",
        "queue_position",
        "queue_total",
        "refresh_command",
        "claim_gate_verdict",
        "completion_claim_ready",
        "allowed_to_claim",
        "latest_execution_case_review_state",
        "latest_execution_case_review_verdict",
        "latest_execution_case_review_next_safe_command",
        "latest_execution_case_review_approval_required",
        "latest_execution_case_review_checklist_items",
        "latest_execution_case_review_blocker_count",
        "latest_execution_case_review_handoff",
        "latest_execution_case_review_handoff_present",
        "latest_execution_case_evidence_preview_gate_count",
        "latest_execution_case_evidence_preview_ready_count",
        "latest_execution_case_evidence_preview_blocked_count",
        "latest_execution_case_evidence_preview_verdicts",
        "latest_execution_case_evidence_preview_latest_verdict",
        "latest_execution_case_evidence_preview_latest_event_id",
        "latest_execution_case_evidence_preview_latest_handoff",
        "latest_execution_case_evidence_preview_latest_handoff_present",
        "latest_execution_case_closure_verdict",
        "latest_execution_case_closure_ready",
        "latest_execution_case_closure_blocks_completion_claim",
        "latest_execution_case_closure_proof_queue",
        "latest_execution_case_closure_proof_queue_count",
        "latest_execution_case_next_closure_proof_command",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_learning_state",
        "execution_learning_closure_command",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_next_proof_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "storage_runtime_fallback_active",
        "storage_runtime_fallback_reason",
        "storage_runtime_fallback_exception_type",
        "storage_runtime_fallback_db_path_display",
        "storage_runtime_fallback_vault_path_display",
        "storage_readiness_blocks_completion_claim",
        "storage_recovery_required",
        "storage_recovery_reason",
        "storage_issues",
        "storage_issue_count",
        "storage_recovery_mode",
        "storage_recovery_next_operator_action",
        "storage_recovery_restart_required",
        "storage_recovery_check_command",
        "storage_recovery_check_api",
        "storage_recovery_command",
        "storage_readiness_blocker",
        "storage_readiness_next_commands",
        "storage_readiness_next_command_count",
        "storage_readiness_next_proof_command",
        "storage_readiness_proof_queue",
        "storage_readiness_proof_queue_count",
        "storage_readiness_first_proof_command",
        "storage_handoff",
        "agi_next_gate",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_target_integrity_blocks_start",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_acceptance_gap_preview",
        "agi_next_acceptance_gap_preview_count",
        "agi_next_first_acceptance_gap",
        "agi_next_target_file_integrity_status",
        "agi_next_build_packet_ready_for_review",
        "agi_next_implementation_preflight_ready",
        "agi_next_implementation_preflight_blockers",
        "agi_next_implementation_preflight_blocker_count",
        "agi_next_implementation_preflight_next_command",
        "agi_real_execution_gap_count",
        "real_execution_gap_count",
        "selected_real_execution_gap",
        "completion_proof_queue",
        "completion_proof_queue_count",
        "completion_next_proof_handoff",
        "completion_next_proof_handoff_present",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} operator handoff field {key} diverged: {metadata}")
    for key in [
        "execution_health_recovery_closure_missing",
        "execution_learning_missing",
        "agi_next_likely_files",
        "agi_next_missing_target_files",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} operator handoff list {key} diverged: {metadata}")
    if handoff.get("execution_health_recovery_closure_missing_count") != len(metadata.get("execution_health_recovery_closure_missing") or []):
        raise SystemExit(f"{label} operator handoff missed recovery missing count: {metadata}")
    if handoff.get("execution_health_recovery_closure_proof_queue_count") != len(metadata.get("execution_health_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} operator handoff missed recovery proof queue count: {metadata}")
    if handoff.get("execution_health_recovery_closure_required_commands") and handoff.get("execution_health_recovery_closure_next_required_command") != handoff.get("execution_health_recovery_closure_required_commands")[0]:
        raise SystemExit(f"{label} operator handoff recovery next required diverged from required-command queue head: {metadata}")
    if handoff.get("execution_health_recovery_closure_next_proof_command") and handoff.get("execution_health_recovery_closure_proof_queue"):
        if handoff.get("execution_health_recovery_closure_next_proof_command") != handoff.get("execution_health_recovery_closure_proof_queue")[0]:
            raise SystemExit(f"{label} operator handoff recovery next proof diverged from queue head: {metadata}")
    if handoff.get("execution_learning_missing_count") != len(metadata.get("execution_learning_missing") or []):
        raise SystemExit(f"{label} operator handoff missed learning missing count: {metadata}")
    if handoff.get("execution_learning_proof_queue_count") != len(metadata.get("execution_learning_proof_queue") or []):
        raise SystemExit(f"{label} operator handoff missed learning proof queue count: {metadata}")
    if handoff.get("execution_learning_required_commands") and handoff.get("execution_learning_next_required_command") != handoff.get("execution_learning_required_commands")[0]:
        raise SystemExit(f"{label} operator handoff learning next required diverged from required-command queue head: {metadata}")
    if handoff.get("latest_execution_case_closure_proof_queue_count") != len(metadata.get("latest_execution_case_closure_proof_queue") or []):
        raise SystemExit(f"{label} operator handoff missed latest case closure queue count: {metadata}")
    if handoff.get("latest_execution_case_review_handoff_present") and (handoff.get("latest_execution_case_review_handoff") or {}).get("source") != "execution_case_review_packet":
        raise SystemExit(f"{label} operator handoff missed latest case review handoff source: {metadata}")
    if handoff.get("latest_execution_case_evidence_preview_gate_count", 0) > 0:
        if handoff.get("latest_execution_case_evidence_preview_ready_count", 0) + handoff.get("latest_execution_case_evidence_preview_blocked_count", 0) != handoff.get("latest_execution_case_evidence_preview_gate_count"):
            raise SystemExit(f"{label} operator handoff evidence preflight counts diverged: {metadata}")
        if handoff.get("latest_execution_case_evidence_preview_latest_handoff_present") is not True:
            raise SystemExit(f"{label} operator handoff missed evidence preflight handoff flag: {metadata}")
        if (handoff.get("latest_execution_case_evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"{label} operator handoff missed evidence preflight handoff source: {metadata}")
    if handoff.get("storage_readiness_proof_queue_count") != len(handoff.get("storage_readiness_proof_queue") or []):
        raise SystemExit(f"{label} operator handoff storage proof queue count diverged: {metadata}")
    if handoff.get("storage_readiness_proof_queue") and handoff.get("storage_readiness_first_proof_command") != handoff.get("storage_readiness_proof_queue")[0]:
        raise SystemExit(f"{label} operator handoff missed first storage proof command: {metadata}")
    if handoff.get("agi_next_likely_file_count") != metadata.get("agi_next_likely_file_count"):
        raise SystemExit(f"{label} operator handoff missed likely file count: {metadata}")
    if handoff.get("agi_next_missing_target_file_count") != metadata.get("agi_next_missing_target_file_count"):
        raise SystemExit(f"{label} operator handoff missed missing file count: {metadata}")
    if handoff.get("agi_next_acceptance_check_count") != metadata.get("agi_next_acceptance_check_count"):
        raise SystemExit(f"{label} operator handoff missed acceptance count: {metadata}")
    if handoff.get("agi_next_focused_verification_command_count") != metadata.get("agi_next_focused_verification_command_count"):
        raise SystemExit(f"{label} operator handoff missed verification count: {metadata}")
    if metadata.get("real_execution_gap_count") != metadata.get("agi_real_execution_gap_count"):
        raise SystemExit(f"{label} operator real-execution alias count diverged: {metadata}")
    if handoff.get("real_execution_gap_count") != handoff.get("agi_real_execution_gap_count"):
        raise SystemExit(f"{label} operator handoff real-execution alias count diverged: {metadata}")
    if not metadata.get("selected_real_execution_gap") or handoff.get("selected_real_execution_gap") != metadata.get("selected_real_execution_gap"):
        raise SystemExit(f"{label} operator handoff missed selected real-execution gap alias: {metadata}")
    if not metadata.get("selected_real_execution_gap_gate") or handoff.get("selected_real_execution_gap_gate") != metadata.get("selected_real_execution_gap_gate"):
        raise SystemExit(f"{label} operator handoff missed selected real-execution gap gate alias: {metadata}")
    if not metadata.get("selected_real_execution_gap_detail") or handoff.get("selected_real_execution_gap_detail") != metadata.get("selected_real_execution_gap_detail"):
        raise SystemExit(f"{label} operator handoff missed selected real-execution gap detail alias: {metadata}")
    gap_by_gate = metadata.get("agi_real_execution_gaps_by_gate") or {}
    if gap_by_gate and metadata.get("selected_real_execution_gap_detail") != gap_by_gate.get(metadata.get("selected_real_execution_gap_gate")):
        raise SystemExit(f"{label} operator detail should match selected gate gap text: {metadata}")
    if not gap_by_gate and metadata.get("selected_real_execution_gap_detail") == metadata.get("selected_real_execution_gap_gate"):
        raise SystemExit(f"{label} operator detail should carry gap text, not just the gate selector: {metadata}")
    if (handoff.get("completion_next_proof_handoff") or {}).get("real_execution_gap_count") != metadata.get("real_execution_gap_count"):
        raise SystemExit(f"{label} operator inherited completion next-proof real-execution alias diverged: {metadata}")
    assert_selected_agi_target_readiness(handoff, f"{label} operator handoff")
    if handoff.get("completion_next_proof_handoff_present") is not True:
        raise SystemExit(f"{label} operator handoff missed inherited completion next-proof handoff flag: {metadata}")
    if (handoff.get("completion_next_proof_handoff") or {}).get("source") != "completion_next_proof_packet":
        raise SystemExit(f"{label} operator handoff missed inherited completion next-proof handoff source: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} operator handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} operator handoff missed non-authorizing contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} operator handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} operator handoff should not grant approval: {metadata}")
    if metadata.get("next_operator_command") and metadata.get("next_operator_command") not in metadata.get("completion_proof_queue", []):
        raise SystemExit(f"{label} next operator command should come from proof queue: {metadata}")


def assert_harness_readiness_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("harness_readiness_handoff") or {}
    if handoff.get("source") != "harness_readiness_digest":
        raise SystemExit(f"{label} missed readiness handoff source: {metadata}")
    if metadata.get("harness_readiness_handoff_ready") is not True or handoff.get("harness_readiness_handoff_ready") is not True:
        raise SystemExit(f"{label} missed harness readiness handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested harness readiness handoff readiness: {metadata}")
    for key in [
        "source_tool",
        "objective",
        "readiness_state",
        "readiness_verdict",
        "completion_claim_ready",
        "allowed_to_claim",
        "top_risk",
        "next_command",
        "next_proof_command",
        "completion_next_proof_command",
        "command_source",
        "command_reason",
        "blocker_details",
        "blocker_count",
        "readiness_rows",
        "readiness_row_count",
        "command_ladder",
        "command_ladder_count",
        "command_ladder_first_command",
        "command_ladder_storage_prefix",
        "command_ladder_storage_prefix_count",
        "completion_proof_queue",
        "completion_proof_queue_count",
        "completion_next_proof_handoff",
        "completion_next_proof_handoff_present",
        "latest_execution_case_found",
        "latest_execution_case_closure_verdict",
        "latest_execution_case_closure_ready",
        "latest_execution_case_closure_blocks_completion_claim",
        "latest_execution_case_closure_proof_queue",
        "latest_execution_case_closure_proof_queue_count",
        "latest_execution_case_next_closure_proof_command",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_learning_state",
        "execution_learning_missing",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_next_proof_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_blocks_completion_claim",
        "storage_runtime_fallback_active",
        "storage_runtime_fallback_reason",
        "storage_runtime_fallback_exception_type",
        "storage_runtime_fallback_db_path_display",
        "storage_runtime_fallback_vault_path_display",
        "storage_readiness_blocks_completion_claim",
        "storage_recovery_required",
        "storage_recovery_reason",
        "storage_recovery_mode",
        "storage_recovery_next_operator_action",
        "storage_recovery_restart_required",
        "storage_recovery_check_command",
        "storage_recovery_check_api",
        "storage_recovery_command",
        "storage_readiness_blocker",
        "storage_readiness_next_commands",
        "storage_readiness_next_command_count",
        "storage_readiness_next_proof_command",
        "storage_readiness_proof_queue",
        "storage_readiness_proof_queue_count",
        "storage_readiness_first_proof_command",
        "storage_handoff",
        "agi_next_gate",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_target_integrity_blocks_start",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_acceptance_gap_preview",
        "agi_next_acceptance_gap_preview_count",
        "agi_next_first_acceptance_gap",
        "agi_next_missing_target_files",
        "agi_next_missing_target_file_count",
        "agi_next_target_file_integrity_status",
        "agi_next_build_packet_ready_for_review",
        "agi_next_implementation_preflight_ready",
        "agi_next_implementation_preflight_blockers",
        "agi_next_implementation_preflight_blocker_count",
        "agi_next_implementation_preflight_next_command",
        "agi_real_execution_gap_count",
        "real_execution_gap_count",
        "selected_real_execution_gap",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} readiness handoff field {key} diverged: {metadata}")
    if metadata.get("readiness_verdict") != metadata.get("readiness_state"):
        raise SystemExit(f"{label} readiness verdict alias diverged from state: {metadata}")
    if metadata.get("completion_next_proof_command") != metadata.get("next_proof_command"):
        raise SystemExit(f"{label} readiness completion next-proof alias diverged: {metadata}")
    if handoff.get("execution_health_recovery_closure_missing_count") != len(metadata.get("execution_health_recovery_closure_missing") or []):
        raise SystemExit(f"{label} readiness handoff missed recovery missing count: {metadata}")
    if handoff.get("execution_health_recovery_closure_proof_queue_count") != len(metadata.get("execution_health_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} readiness handoff missed recovery proof queue count: {metadata}")
    if handoff.get("execution_health_recovery_closure_required_commands") and handoff.get("execution_health_recovery_closure_next_required_command") != handoff.get("execution_health_recovery_closure_required_commands")[0]:
        raise SystemExit(f"{label} readiness handoff recovery next required diverged from required-command queue head: {metadata}")
    if handoff.get("execution_health_recovery_closure_next_proof_command") and handoff.get("execution_health_recovery_closure_proof_queue"):
        if handoff.get("execution_health_recovery_closure_next_proof_command") != handoff.get("execution_health_recovery_closure_proof_queue")[0]:
            raise SystemExit(f"{label} readiness handoff recovery next proof diverged from queue head: {metadata}")
    if handoff.get("execution_learning_missing_count") != len(metadata.get("execution_learning_missing") or []):
        raise SystemExit(f"{label} readiness handoff missed learning missing count: {metadata}")
    if handoff.get("execution_learning_proof_queue_count") != len(metadata.get("execution_learning_proof_queue") or []):
        raise SystemExit(f"{label} readiness handoff missed learning proof queue count: {metadata}")
    if handoff.get("execution_learning_required_commands") and handoff.get("execution_learning_next_required_command") != handoff.get("execution_learning_required_commands")[0]:
        raise SystemExit(f"{label} readiness handoff learning next required diverged from required-command queue head: {metadata}")
    if handoff.get("latest_execution_case_closure_proof_queue_count") != len(metadata.get("latest_execution_case_closure_proof_queue") or []):
        raise SystemExit(f"{label} readiness handoff missed latest case closure queue count: {metadata}")
    if handoff.get("completion_next_proof_handoff_present") is not True:
        raise SystemExit(f"{label} readiness handoff missed inherited completion next-proof handoff flag: {metadata}")
    if (handoff.get("completion_next_proof_handoff") or {}).get("source") != "completion_next_proof_packet":
        raise SystemExit(f"{label} readiness handoff missed inherited completion next-proof handoff source: {metadata}")
    if metadata.get("real_execution_gap_count") != metadata.get("agi_real_execution_gap_count"):
        raise SystemExit(f"{label} readiness real-execution alias count diverged: {metadata}")
    if handoff.get("real_execution_gap_count") != handoff.get("agi_real_execution_gap_count"):
        raise SystemExit(f"{label} readiness handoff real-execution alias count diverged: {metadata}")
    if not metadata.get("selected_real_execution_gap") or handoff.get("selected_real_execution_gap") != metadata.get("selected_real_execution_gap"):
        raise SystemExit(f"{label} readiness handoff missed selected real-execution gap alias: {metadata}")
    if not metadata.get("selected_real_execution_gap_gate") or handoff.get("selected_real_execution_gap_gate") != metadata.get("selected_real_execution_gap_gate"):
        raise SystemExit(f"{label} readiness handoff missed selected real-execution gap gate alias: {metadata}")
    if not metadata.get("selected_real_execution_gap_detail") or handoff.get("selected_real_execution_gap_detail") != metadata.get("selected_real_execution_gap_detail"):
        raise SystemExit(f"{label} readiness handoff missed selected real-execution gap detail alias: {metadata}")
    gap_by_gate = metadata.get("agi_real_execution_gaps_by_gate") or {}
    if gap_by_gate and metadata.get("selected_real_execution_gap_detail") != gap_by_gate.get(metadata.get("selected_real_execution_gap_gate")):
        raise SystemExit(f"{label} readiness detail should match selected gate gap text: {metadata}")
    if not gap_by_gate and metadata.get("selected_real_execution_gap_detail") == metadata.get("selected_real_execution_gap_gate"):
        raise SystemExit(f"{label} readiness detail should carry gap text, not just the gate selector: {metadata}")
    if (handoff.get("completion_next_proof_handoff") or {}).get("real_execution_gap_count") != metadata.get("real_execution_gap_count"):
        raise SystemExit(f"{label} readiness inherited completion next-proof real-execution alias diverged: {metadata}")
    assert_selected_agi_target_readiness(handoff, f"{label} readiness handoff")
    if metadata.get("readiness_row_count") != len(metadata.get("readiness_rows") or []):
        raise SystemExit(f"{label} readiness row count diverged: {metadata}")
    readiness_rows = {row.get("name"): row for row in metadata.get("readiness_rows") or []}
    execution_case_row = readiness_rows.get("execution case") or {}
    expected_case_concrete_next = ""
    case_next = metadata.get("latest_execution_case_next_closure_proof_command")
    proof_queue = metadata.get("completion_proof_queue") or []
    if case_next in proof_queue:
        case_next_index = proof_queue.index(case_next)
        if case_next_index + 1 < len(proof_queue):
            expected_case_concrete_next = proof_queue[case_next_index + 1]
    if metadata.get("latest_execution_case_concrete_next_closure_proof_command") != expected_case_concrete_next:
        raise SystemExit(f"{label} readiness latest case concrete next command diverged from proof queue: {metadata}")
    expected_case_row_next = (
        expected_case_concrete_next
        or case_next
        or "save execution case: <next real order>"
    )
    if metadata.get("latest_execution_case_row_next_closure_command") != expected_case_row_next:
        raise SystemExit(f"{label} readiness latest case row next alias diverged: {metadata}")
    if handoff.get("latest_execution_case_concrete_next_closure_proof_command") != expected_case_concrete_next:
        raise SystemExit(f"{label} readiness handoff missed latest case concrete next command: {metadata}")
    if handoff.get("latest_execution_case_row_next_closure_command") != expected_case_row_next:
        raise SystemExit(f"{label} readiness handoff missed latest case row next command: {metadata}")
    if execution_case_row.get("next") != expected_case_row_next:
        raise SystemExit(f"{label} readiness execution case row missed concrete next command: {metadata}")
    verification_row = readiness_rows.get("verification coverage") or {}
    if metadata.get("execution_health_verification_blocks_completion_claim"):
        if verification_row.get("state") != "blocked":
            raise SystemExit(f"{label} readiness verification row should be blocked: {metadata}")
        expected_verification_next = (
            metadata.get("execution_health_verification_row_next_command")
            or metadata.get("execution_health_verification_concrete_next_proof_command")
            or metadata.get("execution_health_verification_next_required_command")
            or metadata.get("execution_health_verification_next_proof_command")
        )
        if verification_row.get("next") != expected_verification_next:
            raise SystemExit(f"{label} readiness verification row missed next proof command: {metadata}")
    else:
        if verification_row.get("state") != "ready":
            raise SystemExit(f"{label} readiness verification row should be ready: {metadata}")
        if verification_row.get("next") != "none":
            raise SystemExit(f"{label} readiness verification row should not advertise stale proof work: {metadata}")
    learning_row = readiness_rows.get("learning loop") or {}
    latest_case_closure_learning_blocks = (
        _metadata_bool(metadata.get("latest_execution_case_closure_blocks_completion_claim"))
        and any("learning" in str(command).lower() for command in (metadata.get("latest_execution_case_closure_proof_queue") or []))
    )
    learning_row_blocked = (
        _metadata_bool(metadata.get("execution_learning_blocks_completion_claim"))
        or _metadata_bool(metadata.get("latest_execution_case_learning_blocks_completion_claim"))
        or latest_case_closure_learning_blocks
    )
    expected_learning_next = (
        metadata.get("execution_learning_actionable_next_proof_command")
        or metadata.get("execution_learning_actionable_next_required_command")
        or metadata.get("execution_learning_next_proof_command")
        or metadata.get("execution_learning_next_required_command")
        or (
            metadata.get("latest_execution_case_learning_actionable_next_proof_command")
            or metadata.get("latest_execution_case_learning_actionable_next_required_command")
            or metadata.get("latest_execution_case_learning_next_proof_command")
            or metadata.get("latest_execution_case_learning_next_required_command")
            or metadata.get("latest_execution_case_next_closure_proof_command")
            if learning_row_blocked
            else ""
        )
        or ("execution learning closure" if learning_row_blocked else "none")
    )
    if learning_row_blocked:
        if learning_row.get("state") != "blocked" or learning_row.get("next") != expected_learning_next:
            raise SystemExit(f"{label} readiness learning row missed latest learning proof debt: {metadata}")
    elif learning_row.get("state") != "ready" or learning_row.get("next") != "none":
        raise SystemExit(f"{label} readiness learning row should not advertise stale proof work: {metadata}")
    if handoff.get("command_ladder") and metadata.get("next_command") != handoff.get("command_ladder")[0]:
        raise SystemExit(f"{label} readiness handoff should put next command first in ladder: {metadata}")
    if metadata.get("command_ladder") and metadata.get("command_ladder_first_command") != metadata.get("command_ladder")[0]:
        raise SystemExit(f"{label} readiness handoff missed first command alias: {metadata}")
    if metadata.get("command_ladder_storage_prefix_count") != len(metadata.get("command_ladder_storage_prefix") or []):
        raise SystemExit(f"{label} readiness handoff storage prefix count diverged: {metadata}")
    if handoff.get("storage_readiness_proof_queue_count") != len(handoff.get("storage_readiness_proof_queue") or []):
        raise SystemExit(f"{label} readiness handoff storage proof queue count diverged: {metadata}")
    if handoff.get("storage_readiness_proof_queue") and handoff.get("storage_readiness_first_proof_command") != handoff.get("storage_readiness_proof_queue")[0]:
        raise SystemExit(f"{label} readiness handoff missed first storage proof command: {metadata}")
    if metadata.get("completion_proof_queue") and metadata.get("next_command") != metadata.get("completion_proof_queue")[0]:
        raise SystemExit(f"{label} readiness handoff should follow completion proof queue: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} readiness handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} readiness handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} readiness handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} readiness handoff unsafe nested metadata {key}: {metadata}")


def assert_execution_proof_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_proof_handoff") or {}
    if handoff.get("source") != "execution_proof_bundle":
        raise SystemExit(f"{label} missed proof handoff source: {metadata}")
    if metadata.get("execution_proof_handoff_ready") is not True or handoff.get("execution_proof_handoff_ready") is not True:
        raise SystemExit(f"{label} missed proof handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested proof handoff readiness: {metadata}")
    for key in [
        "request",
        "proof_state",
        "can_trust_execution",
        "approval_required",
        "risk_signals",
        "planned_actions",
        "planned_action_count",
        "approval_queue_forecast",
        "forecast_new_approvals",
        "forecast_reused_approval_ids",
        "forecast_queue_before",
        "forecast_queue_after_if_sent",
        "forecast_queue_delta_if_sent",
        "supplied_approval",
        "supplied_approval_id",
        "approval_chain_status",
        "approval_chain_valid",
        "approval_linked_run_ids",
        "supplied_verification",
        "supplied_tests",
        "supplied_evidence",
        "supplied_recovery",
        "missing_receipts",
        "missing_receipt_count",
        "pending_approvals",
        "open_tasks",
        "recent_runs",
        "readable_recent_tool_runs",
        "unreadable_recent_tool_run_rows",
        "recent_failed_runs",
        "verification_runs",
        "proof_lane_rows",
        "proof_lanes",
        "strong_lanes",
        "partial_lanes",
        "missing_lanes",
        "proof_requirement_rows",
        "proof_requirements",
        "missing_requirement_names",
        "missing_requirement_count",
        "next_proof_commands",
        "next_proof_command",
        "first_pending_approval_id",
        "first_failed_run_id",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_ready_to_retry",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_checklist_command",
        "execution_health_recovery_closure_should_open_checklist",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_health_recovery_closure_target_run_id",
        "execution_health_recovery_closure_target_tool_name",
        "execution_health_recovery_closure_target_verification_receipts",
        "execution_health_recovery_closure_target_recovery_packets",
        "execution_health_recovery_closure_target_after_action_learning_packets",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_recent_action_runs",
        "execution_learning_failed_or_blocked_action_runs",
        "execution_learning_recent_verification_runs",
        "execution_learning_recent_recovery_runs",
        "execution_learning_recent_after_action_learning_runs",
        "execution_learning_target_run_id",
        "execution_learning_target_tool_name",
        "execution_learning_target_after_action_learning_packets",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_closure_command",
        "execution_learning_evidence_command",
        "execution_learning_after_action_learning_command",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_learning_actionable_required_commands",
        "execution_learning_actionable_required_command_count",
        "execution_learning_actionable_proof_queue",
        "execution_learning_actionable_proof_queue_count",
        "execution_learning_actionable_next_required_command",
        "execution_learning_actionable_next_proof_command",
        "execution_learning_next_evidence_command",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} proof handoff field {key} diverged: {metadata}")
    if handoff.get("risk_signal_count") != len(metadata.get("risk_signals") or []):
        raise SystemExit(f"{label} proof handoff risk signal count diverged: {metadata}")
    if handoff.get("blockers") != metadata.get("blocker_details"):
        raise SystemExit(f"{label} proof handoff blocker detail list diverged: {metadata}")
    if handoff.get("blocker_count") != metadata.get("blocker_count"):
        raise SystemExit(f"{label} proof handoff blocker count diverged: {metadata}")
    if metadata.get("proof_lanes") != len(metadata.get("proof_lane_rows") or []):
        raise SystemExit(f"{label} proof handoff lane rows diverged from lane count: {metadata}")
    if metadata.get("proof_requirements") != len(metadata.get("proof_requirement_rows") or []):
        raise SystemExit(f"{label} proof handoff requirement rows diverged from requirement count: {metadata}")
    if metadata.get("missing_requirement_count") != len(metadata.get("missing_requirement_names") or []):
        raise SystemExit(f"{label} proof handoff missing requirement count diverged: {metadata}")
    if metadata.get("next_proof_commands") and metadata.get("next_proof_command") != metadata.get("next_proof_commands")[0]:
        raise SystemExit(f"{label} proof handoff next command should be first queue item: {metadata}")
    if metadata.get("missing_receipt_count") != len(metadata.get("missing_receipts") or []):
        raise SystemExit(f"{label} proof handoff missing receipt count diverged: {metadata}")
    assert_execution_health_recovery_proof_aliases(handoff, f"{label} proof handoff")
    assert_execution_learning_proof_aliases(handoff, f"{label} proof handoff")
    assert_execution_learning_actionable_aliases(handoff, f"{label} proof handoff")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} proof handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} proof handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} proof handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} proof handoff should not grant approval: {metadata}")


def assert_execution_proof_aliases_require_explicit_counts() -> None:
    handoff = {
        "execution_health_recovery_closure_missing": ["verification_receipt"],
        "execution_health_recovery_closure_proof_queue": ["recovery closure checklist"],
        "execution_learning_missing": ["after_action_learning"],
        "execution_learning_proof_queue": ["after-action learning packet 7"],
        "execution_learning_actionable_proof_queue": ["execution learning closure 7"],
    }
    aliases = _execution_proof_handoff_aliases(handoff)
    for key in [
        "execution_proof_recovery_closure_missing_count",
        "execution_proof_recovery_closure_proof_queue_count",
        "execution_proof_learning_missing_count",
        "execution_proof_learning_proof_queue_count",
        "execution_proof_learning_actionable_proof_queue_count",
    ]:
        if aliases.get(key) is not None:
            raise SystemExit(f"Execution proof alias helper synthesized missing count {key}: {aliases}")


INHERITED_EXECUTION_PROOF_ALIAS_KEYS = [
    "execution_proof_recovery_closure_state",
    "execution_proof_recovery_closure_ready_to_retry",
    "execution_proof_recovery_closure_missing",
    "execution_proof_recovery_closure_missing_count",
    "execution_proof_recovery_closure_proof_queue",
    "execution_proof_recovery_closure_proof_queue_count",
    "execution_proof_recovery_closure_next_required_command",
    "execution_proof_recovery_closure_next_proof_command",
    "execution_proof_recovery_closure_blocks_completion_claim",
    "execution_proof_learning_state",
    "execution_proof_learning_blocks_completion_claim",
    "execution_proof_learning_missing",
    "execution_proof_learning_missing_count",
    "execution_proof_learning_proof_queue",
    "execution_proof_learning_proof_queue_count",
    "execution_proof_learning_next_required_command",
    "execution_proof_learning_next_proof_command",
    "execution_proof_learning_actionable_proof_queue",
    "execution_proof_learning_actionable_proof_queue_count",
    "execution_proof_learning_actionable_next_required_command",
    "execution_proof_learning_actionable_next_proof_command",
]


def assert_inherited_execution_proof_aliases(metadata: dict, label: str, handoff_key: str) -> None:
    proof_handoff = metadata.get("execution_proof_handoff") or {}
    if proof_handoff.get("source") != "execution_proof_bundle":
        raise SystemExit(f"{label} missed inherited proof handoff source: {metadata}")
    handoff = metadata.get(handoff_key) or {}
    for key in INHERITED_EXECUTION_PROOF_ALIAS_KEYS:
        if metadata.get(key) != handoff.get(key):
            raise SystemExit(f"{label} inherited proof alias {key} diverged from nested handoff: {metadata}")
    alias_to_proof_key = {
        "execution_proof_recovery_closure_state": "execution_health_recovery_closure_state",
        "execution_proof_recovery_closure_ready_to_retry": "execution_health_recovery_closure_ready_to_retry",
        "execution_proof_recovery_closure_missing": "execution_health_recovery_closure_missing",
        "execution_proof_recovery_closure_missing_count": "execution_health_recovery_closure_missing_count",
        "execution_proof_recovery_closure_proof_queue": "execution_health_recovery_closure_proof_queue",
        "execution_proof_recovery_closure_proof_queue_count": "execution_health_recovery_closure_proof_queue_count",
        "execution_proof_recovery_closure_next_required_command": "execution_health_recovery_closure_next_required_command",
        "execution_proof_recovery_closure_next_proof_command": "execution_health_recovery_closure_next_proof_command",
        "execution_proof_recovery_closure_blocks_completion_claim": "execution_health_recovery_closure_blocks_completion_claim",
        "execution_proof_learning_state": "execution_learning_state",
        "execution_proof_learning_blocks_completion_claim": "execution_learning_blocks_completion_claim",
        "execution_proof_learning_missing": "execution_learning_missing",
        "execution_proof_learning_missing_count": "execution_learning_missing_count",
        "execution_proof_learning_proof_queue": "execution_learning_proof_queue",
        "execution_proof_learning_proof_queue_count": "execution_learning_proof_queue_count",
        "execution_proof_learning_next_required_command": "execution_learning_next_required_command",
        "execution_proof_learning_next_proof_command": "execution_learning_next_proof_command",
        "execution_proof_learning_actionable_proof_queue": "execution_learning_actionable_proof_queue",
        "execution_proof_learning_actionable_proof_queue_count": "execution_learning_actionable_proof_queue_count",
        "execution_proof_learning_actionable_next_required_command": "execution_learning_actionable_next_required_command",
        "execution_proof_learning_actionable_next_proof_command": "execution_learning_actionable_next_proof_command",
    }
    for alias, proof_key in alias_to_proof_key.items():
        if metadata.get(alias) != proof_handoff.get(proof_key):
            raise SystemExit(f"{label} inherited proof alias {alias} diverged from source proof handoff: {metadata}")
    if metadata.get("execution_proof_recovery_closure_proof_queue_count") != len(metadata.get("execution_proof_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} inherited recovery proof queue count diverged: {metadata}")
    if metadata.get("execution_proof_learning_proof_queue_count") != len(metadata.get("execution_proof_learning_proof_queue") or []):
        raise SystemExit(f"{label} inherited learning proof queue count diverged: {metadata}")
    if metadata.get("execution_proof_learning_actionable_proof_queue_count") != len(metadata.get("execution_proof_learning_actionable_proof_queue") or []):
        raise SystemExit(f"{label} inherited actionable learning proof queue count diverged: {metadata}")
    if metadata.get("execution_proof_recovery_closure_proof_queue") and metadata.get("execution_proof_recovery_closure_next_proof_command") != metadata["execution_proof_recovery_closure_proof_queue"][0]:
        raise SystemExit(f"{label} inherited recovery next proof command diverged: {metadata}")
    if metadata.get("execution_proof_learning_proof_queue") and metadata.get("execution_proof_learning_next_proof_command") != metadata["execution_proof_learning_proof_queue"][0]:
        raise SystemExit(f"{label} inherited learning next proof command diverged: {metadata}")
    if metadata.get("execution_proof_learning_actionable_proof_queue") and metadata.get("execution_proof_learning_actionable_next_proof_command") != metadata["execution_proof_learning_actionable_proof_queue"][0]:
        raise SystemExit(f"{label} inherited actionable learning next proof command diverged: {metadata}")
    if metadata.get("execution_proof_learning_actionable_next_proof_command") != metadata.get("execution_proof_learning_actionable_next_required_command"):
        raise SystemExit(f"{label} inherited actionable learning next required/proof aliases diverged: {metadata}")


def assert_execution_mission_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_mission_handoff") or {}
    if handoff.get("source") != "execution_mission_control":
        raise SystemExit(f"{label} missed mission handoff source: {metadata}")
    if metadata.get("execution_mission_handoff_ready") is not True or handoff.get("execution_mission_handoff_ready") is not True:
        raise SystemExit(f"{label} missed mission handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested mission handoff readiness: {metadata}")
    for key in [
        "request",
        "mission_state",
        "go_no_go",
        "next_command",
        "risk_signals",
        "approval_required",
        "pending_approvals",
        "open_tasks",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "approval_held_review_command",
        "recent_verification_runs",
        "lifecycle_rows",
        "lifecycle_stages",
        "route_packets",
        "route_packet_count",
        "proof_packets",
        "proof_packet_count",
        "recovery_packets",
        "recovery_packet_count",
        "learning_packets",
        "learning_packet_count",
        "mission_packet_groups",
        "mission_command_queue",
        "mission_command_count",
        "governor_command",
        "cockpit_command",
        "dispatch_command",
        "runbook_command",
        "proof_bundle_command",
        "acceptance_command",
        "audit_command",
        "recovery_command",
        "learning_closure_command",
        "learning_command",
        "execution_proof_handoff",
        "execution_proof_handoff_present",
        "execution_proof_state",
        "execution_proof_can_trust_execution",
        "execution_proof_next_command",
        "execution_proof_missing_receipt_count",
        "execution_proof_blocker_count",
        *INHERITED_EXECUTION_PROOF_ALIAS_KEYS,
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "blocker_details",
        "blockers",
        "blocker_count",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} mission handoff field {key} diverged: {metadata}")
    if handoff.get("risk_signal_count") != len(metadata.get("risk_signals") or []):
        raise SystemExit(f"{label} mission handoff risk signal count diverged: {metadata}")
    if metadata.get("lifecycle_stages") != len(metadata.get("lifecycle_rows") or []):
        raise SystemExit(f"{label} mission lifecycle count diverged: {metadata}")
    if metadata.get("route_packet_count") != len(metadata.get("route_packets") or []):
        raise SystemExit(f"{label} route packet count diverged: {metadata}")
    if metadata.get("proof_packet_count") != len(metadata.get("proof_packets") or []):
        raise SystemExit(f"{label} proof packet count diverged: {metadata}")
    if metadata.get("recovery_packet_count") != len(metadata.get("recovery_packets") or []):
        raise SystemExit(f"{label} recovery packet count diverged: {metadata}")
    if metadata.get("learning_packet_count") != len(metadata.get("learning_packets") or []):
        raise SystemExit(f"{label} learning packet count diverged: {metadata}")
    if metadata.get("mission_command_count") != len(metadata.get("mission_command_queue") or []):
        raise SystemExit(f"{label} mission command count diverged: {metadata}")
    if metadata.get("mission_command_queue") and metadata.get("next_command") != metadata.get("mission_command_queue")[0]:
        raise SystemExit(f"{label} mission next command should be first queue item: {metadata}")
    if metadata.get("blocker_count") != len(metadata.get("blocker_details") or []):
        raise SystemExit(f"{label} mission blocker count diverged: {metadata}")
    if handoff.get("execution_proof_handoff_present") is not True:
        raise SystemExit(f"{label} mission handoff missed inherited execution proof handoff flag: {metadata}")
    if (handoff.get("execution_proof_handoff") or {}).get("source") != "execution_proof_bundle":
        raise SystemExit(f"{label} mission handoff missed inherited execution proof handoff source: {metadata}")
    proof_handoff = handoff.get("execution_proof_handoff") or {}
    if handoff.get("execution_proof_state") != proof_handoff.get("proof_state"):
        raise SystemExit(f"{label} mission proof state diverged from inherited proof handoff: {metadata}")
    if handoff.get("execution_proof_next_command") != proof_handoff.get("next_proof_command"):
        raise SystemExit(f"{label} mission proof next command diverged from inherited proof handoff: {metadata}")
    assert_inherited_execution_proof_aliases(metadata, label, "execution_mission_handoff")
    groups = metadata.get("mission_packet_groups") or {}
    for group, key in [
        ("route", "route_packets"),
        ("proof", "proof_packets"),
        ("recovery", "recovery_packets"),
        ("learning", "learning_packets"),
    ]:
        if groups.get(group) != metadata.get(key):
            raise SystemExit(f"{label} mission packet group {group} diverged: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} mission handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} mission handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} mission handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} mission handoff should not grant approval: {metadata}")


def assert_execution_case_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_case_handoff") or {}
    if handoff.get("source") != "execution_case_handoff_packet":
        raise SystemExit(f"{label} missed execution case handoff source: {metadata}")
    if metadata.get("execution_case_handoff_ready") is not True or handoff.get("execution_case_handoff_ready") is not True:
        raise SystemExit(f"{label} missed execution case handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested execution case handoff readiness: {metadata}")
    for key in [
        "request",
        "handoff_verdict",
        "mission_state",
        "go_no_go",
        "ready_to_save_case",
        "next_command",
        "cockpit_command",
        "save_case_command",
        "inspect_command",
        "gate_command",
        "review_command",
        "timeline_command",
        "mission_command_queue",
        "mission_command_count",
        "handoff_proof_queue",
        "handoff_proof_queue_count",
        "proof_queue",
        "proof_queue_count",
        "approval_required",
        "risk_signals",
        "risk_signal_count",
        "pending_approvals",
        "blockers",
        "blocker_details",
        "blocker_count",
        "mission_handoff_present",
        "execution_proof_handoff_present",
        "execution_proof_state",
        "execution_proof_next_command",
        "execution_proof_can_trust_execution",
        *INHERITED_EXECUTION_PROOF_ALIAS_KEYS,
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "writes_case",
        "creates_case",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} case handoff field {key} diverged: {metadata}")
    if handoff.get("mission_handoff") != metadata.get("mission_handoff"):
        raise SystemExit(f"{label} case handoff nested mission handoff diverged: {metadata}")
    if handoff.get("mission_handoff_present") is not True:
        raise SystemExit(f"{label} case handoff should preserve mission handoff: {metadata}")
    if handoff.get("execution_proof_handoff") != metadata.get("execution_proof_handoff"):
        raise SystemExit(f"{label} case handoff nested execution proof handoff diverged: {metadata}")
    if handoff.get("execution_proof_handoff_present") is not True:
        raise SystemExit(f"{label} case handoff should preserve execution proof handoff: {metadata}")
    proof_handoff = handoff.get("execution_proof_handoff") or {}
    if proof_handoff.get("source") != "execution_proof_bundle":
        raise SystemExit(f"{label} case handoff missed inherited execution proof source: {metadata}")
    if handoff.get("execution_proof_state") != proof_handoff.get("proof_state"):
        raise SystemExit(f"{label} case handoff proof state diverged from inherited proof handoff: {metadata}")
    if handoff.get("execution_proof_next_command") != proof_handoff.get("next_proof_command"):
        raise SystemExit(f"{label} case handoff proof next command diverged from inherited proof handoff: {metadata}")
    assert_inherited_execution_proof_aliases(metadata, label, "execution_case_handoff")
    if metadata.get("risk_signal_count") != len(metadata.get("risk_signals") or []):
        raise SystemExit(f"{label} case handoff risk signal count diverged: {metadata}")
    if metadata.get("handoff_proof_queue_count") != len(metadata.get("handoff_proof_queue") or []):
        raise SystemExit(f"{label} case handoff proof queue count diverged: {metadata}")
    if metadata.get("proof_queue") != metadata.get("handoff_proof_queue"):
        raise SystemExit(f"{label} case handoff proof queue alias diverged: {metadata}")
    if metadata.get("mission_command_count") != len(metadata.get("mission_command_queue") or []):
        raise SystemExit(f"{label} case handoff mission queue count diverged: {metadata}")
    if metadata.get("blocker_count") != len(metadata.get("blocker_details") or []):
        raise SystemExit(f"{label} case handoff blocker count diverged: {metadata}")
    if metadata.get("writes_case") is not False or metadata.get("creates_case") is not False:
        raise SystemExit(f"{label} case handoff should not write/create a case: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} case handoff unsafe nested metadata {key}: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} case handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} case handoff should not authorize execution or completion claims: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} case handoff should not grant approval: {metadata}")


def assert_saved_execution_case_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("saved_execution_case_handoff") or {}
    if handoff.get("source") != "save_execution_case":
        raise SystemExit(f"{label} missed saved-case handoff source: {metadata}")
    for key in [
        "case_id",
        "request",
        "mission_state",
        "go_no_go",
        "next_command",
        "initial_governor_command",
        "initial_cockpit_command",
        "mission_command_queue",
        "mission_command_count",
        "execution_mission_handoff_present",
        "execution_proof_handoff_present",
        "execution_proof_state",
        "execution_proof_next_command",
        "execution_proof_can_trust_execution",
        *INHERITED_EXECUTION_PROOF_ALIAS_KEYS,
        "proof_bundle_command",
        "acceptance_command",
        "audit_command",
        "recovery_command",
        "learning_closure_command",
        "learning_command",
        "initial_evidence_supplied",
        "initial_evidence_attached",
        "initial_evidence_ready_to_append",
        "initial_evidence_event_id",
        "initial_evidence_event_type",
        "initial_evidence_receipt_kind",
        "initial_evidence_receipt_kind_normalized",
        "initial_evidence_receipt_id",
        "initial_evidence_receipt_target_status",
        "initial_evidence_receipt_target_exists",
        "initial_evidence_receipt_target_issue",
        "initial_evidence_conflict_reason",
        "initial_evidence_skip_reason",
        "evidence_events_created",
        "approval_required",
        "risk_signals",
        "risk_signal_count",
        "pending_approvals",
        "blockers",
        "note_path",
        "local_case_creation_only",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "writes_case",
        "creates_case",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} saved-case handoff field {key} diverged: {metadata}")
    if metadata.get("mission_command_count") != len(metadata.get("mission_command_queue") or []):
        raise SystemExit(f"{label} saved-case mission command count diverged: {metadata}")
    if handoff.get("execution_mission_handoff") != metadata.get("execution_mission_handoff"):
        raise SystemExit(f"{label} saved-case mission handoff diverged: {metadata}")
    if handoff.get("execution_mission_handoff_present") is not True:
        raise SystemExit(f"{label} saved-case missed mission handoff presence: {metadata}")
    if (handoff.get("execution_mission_handoff") or {}).get("source") != "execution_mission_control":
        raise SystemExit(f"{label} saved-case missed mission handoff source: {metadata}")
    if handoff.get("execution_proof_handoff") != metadata.get("execution_proof_handoff"):
        raise SystemExit(f"{label} saved-case proof handoff diverged: {metadata}")
    if handoff.get("execution_proof_handoff_present") is not True:
        raise SystemExit(f"{label} saved-case missed proof handoff presence: {metadata}")
    proof_handoff = handoff.get("execution_proof_handoff") or {}
    if proof_handoff.get("source") != "execution_proof_bundle":
        raise SystemExit(f"{label} saved-case missed proof handoff source: {metadata}")
    if handoff.get("execution_proof_state") != proof_handoff.get("proof_state"):
        raise SystemExit(f"{label} saved-case proof state diverged from inherited proof handoff: {metadata}")
    if handoff.get("execution_proof_next_command") != proof_handoff.get("next_proof_command"):
        raise SystemExit(f"{label} saved-case proof next command diverged from inherited proof handoff: {metadata}")
    assert_inherited_execution_proof_aliases(metadata, label, "saved_execution_case_handoff")
    if metadata.get("risk_signal_count") != len(metadata.get("risk_signals") or []):
        raise SystemExit(f"{label} saved-case risk signal count diverged: {metadata}")
    if handoff.get("writes_case") is not True or handoff.get("creates_case") is not True:
        raise SystemExit(f"{label} saved-case handoff should declare case write/create: {metadata}")
    if handoff.get("local_case_creation_only") is not True:
        raise SystemExit(f"{label} saved-case handoff should declare local case creation only: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} saved-case handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} saved-case handoff should not grant approval: {metadata}")
    for key in ["writes_files", "writes_database", "writes_notes"]:
        if handoff.get(key) is not True:
            raise SystemExit(f"{label} saved-case handoff missed local write flag {key}: {metadata}")
    for key in [
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "speaks",
        "completes_tasks",
    ]:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} saved-case handoff unsafe nested metadata {key}: {metadata}")


def assert_inspect_execution_case_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("inspect_execution_case_handoff") or {}
    if handoff.get("source") != "inspect_execution_case":
        raise SystemExit(f"{label} missed inspect-case handoff source: {metadata}")
    for key in [
        "found",
        "case_id",
        "request",
        "mission_state",
        "go_no_go",
        "next_command",
        "risk_signals",
        "risk_signal_count",
        "note_path",
        "events",
        "mission_command_queue",
        "mission_command_count",
        "execution_mission_handoff_present",
        "execution_proof_handoff_present",
        "execution_proof_state",
        "execution_proof_next_command",
        "execution_proof_can_trust_execution",
        *INHERITED_EXECUTION_PROOF_ALIAS_KEYS,
        "initial_governor_command",
        "proof_bundle_command",
        "acceptance_command",
        "audit_command",
        "recovery_command",
        "learning_command",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_ready_to_retry",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} inspect-case handoff field {key} diverged: {metadata}")
    if metadata.get("risk_signal_count") != len(metadata.get("risk_signals") or []):
        raise SystemExit(f"{label} inspect-case risk signal count diverged: {metadata}")
    if metadata.get("mission_command_count") != len(metadata.get("mission_command_queue") or []):
        raise SystemExit(f"{label} inspect-case mission command count diverged: {metadata}")
    if handoff.get("execution_mission_handoff") != metadata.get("execution_mission_handoff"):
        raise SystemExit(f"{label} inspect-case mission handoff diverged: {metadata}")
    if handoff.get("execution_mission_handoff_present") is not True:
        raise SystemExit(f"{label} inspect-case missed mission handoff presence: {metadata}")
    if (handoff.get("execution_mission_handoff") or {}).get("source") != "execution_mission_control":
        raise SystemExit(f"{label} inspect-case missed mission handoff source: {metadata}")
    if handoff.get("execution_proof_handoff") != metadata.get("execution_proof_handoff"):
        raise SystemExit(f"{label} inspect-case proof handoff diverged: {metadata}")
    if handoff.get("execution_proof_handoff_present") is not True:
        raise SystemExit(f"{label} inspect-case missed proof handoff presence: {metadata}")
    proof_handoff = handoff.get("execution_proof_handoff") or {}
    if proof_handoff.get("source") != "execution_proof_bundle":
        raise SystemExit(f"{label} inspect-case missed proof handoff source: {metadata}")
    if handoff.get("execution_proof_state") != proof_handoff.get("proof_state"):
        raise SystemExit(f"{label} inspect-case proof state diverged from inherited proof handoff: {metadata}")
    if handoff.get("execution_proof_next_command") != proof_handoff.get("next_proof_command"):
        raise SystemExit(f"{label} inspect-case proof next command diverged from inherited proof handoff: {metadata}")
    assert_inherited_execution_proof_aliases(metadata, label, "inspect_execution_case_handoff")
    if metadata.get("execution_health_recovery_closure_proof_queue_count") != len(metadata.get("execution_health_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} inspect-case recovery proof count diverged: {metadata}")
    if metadata.get("execution_learning_proof_queue_count") != len(metadata.get("execution_learning_proof_queue") or []):
        raise SystemExit(f"{label} inspect-case learning proof count diverged: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} inspect-case handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} inspect-case handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} inspect-case handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} inspect-case handoff unsafe nested metadata {key}: {metadata}")


def assert_no_case_inspect_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("inspect_execution_case_handoff") or {}
    if handoff.get("source") != "inspect_execution_case" or handoff.get("found") is not False:
        raise SystemExit(f"{label} missed no-case inspect handoff: {metadata}")
    for key, expected in {
        "case_id": None,
        "request": "",
        "mission_state": "NO_CASE",
        "go_no_go": "UNKNOWN",
        "next_command": "",
        "risk_signals": [],
        "risk_signal_count": 0,
        "note_path": "",
        "events": 0,
        "mission_command_queue": [],
        "mission_command_count": 0,
        "execution_mission_handoff": {},
        "execution_mission_handoff_present": False,
        "execution_proof_handoff": {},
        "execution_proof_handoff_present": False,
        "execution_proof_state": "unknown",
        "execution_proof_next_command": "",
        "execution_proof_can_trust_execution": False,
        "execution_health_recovery_closure_state": "unknown",
        "execution_health_recovery_closure_ready_to_retry": False,
        "execution_health_recovery_closure_missing": [],
        "execution_health_recovery_closure_missing_count": 0,
        "execution_health_recovery_closure_proof_queue": [],
        "execution_health_recovery_closure_proof_queue_count": 0,
        "execution_health_recovery_closure_next_required_command": "",
        "execution_health_recovery_closure_next_proof_command": "",
        "execution_health_recovery_closure_blocks_completion_claim": False,
        "execution_learning_state": "unknown",
        "execution_learning_blocks_completion_claim": False,
        "execution_learning_missing": [],
        "execution_learning_missing_count": 0,
        "execution_learning_proof_queue": [],
        "execution_learning_proof_queue_count": 0,
        "execution_learning_next_required_command": "",
        "execution_learning_next_proof_command": "",
        "evidence_preview_gate_count": 0,
        "evidence_preview_ready_count": 0,
        "evidence_preview_blocked_count": 0,
        "evidence_preview_verdicts": [],
        "evidence_preview_latest_verdict": "",
        "evidence_preview_latest_event_id": None,
        "evidence_preview_latest_handoff": {},
        "evidence_preview_latest_handoff_present": False,
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }.items():
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} no-case inspect handoff missed {key}: {metadata}")
    for key in [
        "evidence_preview_gate_count",
        "evidence_preview_ready_count",
        "evidence_preview_blocked_count",
        "evidence_preview_verdicts",
        "evidence_preview_latest_verdict",
        "evidence_preview_latest_event_id",
        "evidence_preview_latest_handoff",
        "evidence_preview_latest_handoff_present",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if metadata.get(key) != handoff.get(key):
            raise SystemExit(f"{label} no-case inspect evidence-preflight top-level/handoff diverged for {key}: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} no-case inspect handoff unsafe nested metadata {key}: {metadata}")


def assert_execution_case_evidence_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_case_evidence_handoff") or {}
    if handoff.get("source") != "execution_case_evidence_packet":
        raise SystemExit(f"{label} missed case-evidence handoff source: {metadata}")
    for key in [
        "found",
        "case_id",
        "request",
        "verdict",
        "ready_to_append",
        "event_type",
        "summary",
        "explicit_receipt_kind",
        "explicit_receipt_id",
        "inferred_receipt_kind",
        "inferred_receipt_id",
        "receipt_kind",
        "receipt_kind_normalized",
        "receipt_id",
        "receipt_target_status",
        "receipt_target_exists",
        "receipt_target_issue",
        "target_lookup_command",
        "conflict_reason",
        "append_command",
        "next_command",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} case-evidence handoff field {key} diverged: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} case-evidence handoff missed preview-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} case-evidence handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} case-evidence handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} case-evidence handoff unsafe nested metadata {key}: {metadata}")
    for key in ["writes_case", "writes_case_event", "writes_evidence"]:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} case-evidence handoff should be read-only for {key}: {metadata}")


def assert_case_evidence_preview_contract(metadata: dict, label: str) -> None:
    if (
        metadata.get("draft_only") is not True
        or metadata.get("requires_manual_send") is not True
        or metadata.get("loads_without_execution") is not True
    ):
        raise SystemExit(f"{label} missed preview-only contract: {metadata}")
    if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} should not authorize execution/completion: {metadata}")
    if metadata.get("approval_granted") is not False:
        raise SystemExit(f"{label} should not grant approval: {metadata}")


def assert_append_case_evidence_contract(metadata: dict, label: str) -> None:
    handoff = metadata.get("append_execution_case_evidence_handoff") or {}
    if handoff.get("source") != "append_execution_case_evidence":
        raise SystemExit(f"{label} missed append-evidence handoff source: {metadata}")
    for key in [
        "case_id",
        "event_id",
        "event_type",
        "summary",
        "receipt_kind",
        "receipt_id",
        "inferred_receipt_kind",
        "inferred_receipt_id",
        "evidence_preview_verdict",
        "evidence_preview_ready_to_append",
        "note_path",
        "local_evidence_append_only",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "writes_case",
        "writes_case_event",
        "writes_evidence",
        "writes_files",
        "writes_database",
        "writes_notes",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} append-evidence handoff field {key} diverged: {metadata}")
    if metadata.get("local_evidence_append_only") is not True:
        raise SystemExit(f"{label} should declare local evidence append only: {metadata}")
    if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} should not authorize execution/completion: {metadata}")
    if metadata.get("approval_granted") is not False:
        raise SystemExit(f"{label} should not grant approval: {metadata}")
    for key in ["writes_case", "writes_case_event", "writes_evidence", "writes_files", "writes_database", "writes_notes"]:
        if metadata.get(key) is not True or handoff.get(key) is not True:
            raise SystemExit(f"{label} missed local write metadata {key}: {metadata}")
    for key in NON_WRITE_AUTHORITY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} unsafe append metadata {key}: {metadata}")


def assert_append_case_evidence_block_contract(metadata: dict, label: str) -> None:
    if metadata.get("local_evidence_append_only") is not False:
        raise SystemExit(f"{label} blocked append should not declare local append completion: {metadata}")
    if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} should not authorize execution/completion: {metadata}")
    if metadata.get("approval_granted") is not False:
        raise SystemExit(f"{label} should not grant approval: {metadata}")
    for key in ["writes_case", "writes_case_event", "writes_evidence", "writes_files", "writes_database", "writes_notes"]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} blocked append should not write {key}: {metadata}")


def assert_execution_case_gate_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_case_gate_handoff") or {}
    if handoff.get("source") != "execution_case_gate":
        raise SystemExit(f"{label} missed case-gate handoff source: {metadata}")
    if metadata.get("execution_case_gate_handoff_ready") is not True or handoff.get("execution_case_gate_handoff_ready") is not True:
        raise SystemExit(f"{label} missed case-gate handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested case-gate handoff readiness: {metadata}")
    for key in [
        "found",
        "case_id",
        "verdict",
        "request",
        "mission_state",
        "go_no_go",
        "next_command",
        "next_safe_command",
        "risk_signals",
        "approval_required",
        "planned_action_count",
        "forecast_new_approvals",
        "forecast_reused_approval_ids",
        "forecast_queue_before",
        "forecast_queue_after_if_sent",
        "forecast_queue_delta_if_sent",
        "pending_approvals",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "approval_held_review_command",
        "events",
        "evidence_preview_gate_count",
        "evidence_preview_ready_count",
        "evidence_preview_blocked_count",
        "evidence_preview_verdicts",
        "evidence_preview_latest_verdict",
        "evidence_preview_latest_event_id",
        "evidence_preview_latest_handoff",
        "evidence_preview_latest_handoff_present",
        "has_verification_evidence",
        "has_approval_evidence",
        "has_approval_chain_evidence",
        "approval_chain_status",
        "approval_evidence_ids",
        "approval_linked_run_ids",
        "verification_receipt_ids",
        "runtime_trace_receipt_message_ids",
        "missing_verification_receipt_ids",
        "missing_runtime_trace_receipt_message_ids",
        "has_missing_verification_receipts",
        "has_missing_runtime_trace_receipts",
        "verified_approval_run_ids",
        "has_verified_approval_run",
        "has_recovery_evidence",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_proof_handoff",
        "execution_proof_handoff_present",
        "execution_proof_state",
        "execution_proof_next_command",
        "execution_proof_can_trust_execution",
        *INHERITED_EXECUTION_PROOF_ALIAS_KEYS,
        "case_proof_requirements",
        "missing_case_proofs",
        "next_case_proof_commands",
        "next_case_proof_command",
        "mission_command_queue",
        "mission_command_count",
        "initial_governor_command",
        "proof_bundle_command",
        "acceptance_command",
        "audit_command",
        "recovery_command",
        "learning_command",
        "blockers",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} case-gate handoff field {key} diverged: {metadata}")
    if metadata.get("mission_command_count") != len(metadata.get("mission_command_queue") or []):
        raise SystemExit(f"{label} case-gate mission command count diverged: {metadata}")
    if metadata.get("execution_health_recovery_closure_proof_queue_count") != len(metadata.get("execution_health_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} case-gate recovery proof count diverged: {metadata}")
    if metadata.get("execution_learning_proof_queue_count") != len(metadata.get("execution_learning_proof_queue") or []):
        raise SystemExit(f"{label} case-gate learning proof count diverged: {metadata}")
    if metadata.get("evidence_preview_gate_count") != len(metadata.get("evidence_preview_verdicts") or []):
        raise SystemExit(f"{label} case-gate evidence preview verdict count diverged: {metadata}")
    if metadata.get("evidence_preview_ready_count", 0) + metadata.get("evidence_preview_blocked_count", 0) != metadata.get("evidence_preview_gate_count"):
        raise SystemExit(f"{label} case-gate evidence preview ready/blocked count diverged: {metadata}")
    if metadata.get("evidence_preview_gate_count", 0) > 0:
        if metadata.get("evidence_preview_latest_handoff_present") is not True:
            raise SystemExit(f"{label} case-gate missed latest evidence preview handoff flag: {metadata}")
        if (metadata.get("evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"{label} case-gate missed latest evidence preview handoff source: {metadata}")
    assert_inherited_execution_proof_aliases(metadata, label, "execution_case_gate_handoff")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} case-gate handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} case-gate handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} case-gate handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} case-gate handoff unsafe nested metadata {key}: {metadata}")


def assert_no_case_gate_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_case_gate_handoff") or {}
    if handoff.get("source") != "execution_case_gate" or handoff.get("found") is not False:
        raise SystemExit(f"{label} missed no-case gate handoff: {metadata}")
    if metadata.get("execution_case_gate_handoff_ready") is not True or handoff.get("execution_case_gate_handoff_ready") is not True:
        raise SystemExit(f"{label} missed no-case gate handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested no-case gate handoff readiness: {metadata}")
    for key, expected in {
        "case_id": None,
        "request": "",
        "verdict": "CASE_NOT_FOUND",
        "mission_state": "NO_CASE",
        "go_no_go": "UNKNOWN",
        "next_command": "",
        "next_safe_command": "save execution case: <order>",
        "risk_signals": [],
        "approval_required": False,
        "planned_action_count": 0,
        "forecast_new_approvals": 0,
        "forecast_reused_approval_ids": [],
        "forecast_queue_before": 0,
        "forecast_queue_after_if_sent": 0,
        "forecast_queue_delta_if_sent": 0,
        "pending_approvals": 0,
        "recent_failed_runs": 0,
        "recent_approval_held_runs": 0,
        "approval_held_review_command": "",
        "events": 0,
        "evidence_preview_gate_count": 0,
        "evidence_preview_ready_count": 0,
        "evidence_preview_blocked_count": 0,
        "evidence_preview_verdicts": [],
        "evidence_preview_latest_verdict": "",
        "evidence_preview_latest_event_id": None,
        "evidence_preview_latest_handoff": {},
        "evidence_preview_latest_handoff_present": False,
        "has_verification_evidence": False,
        "has_approval_evidence": False,
        "has_approval_chain_evidence": False,
        "approval_chain_status": "missing",
        "approval_evidence_ids": [],
        "approval_linked_run_ids": [],
        "verification_receipt_ids": [],
        "runtime_trace_receipt_message_ids": [],
        "missing_verification_receipt_ids": [],
        "missing_runtime_trace_receipt_message_ids": [],
        "has_missing_verification_receipts": False,
        "has_missing_runtime_trace_receipts": False,
        "verified_approval_run_ids": [],
        "has_verified_approval_run": False,
        "has_recovery_evidence": False,
        "execution_health_recovery_closure_state": "unknown",
        "execution_health_recovery_closure_missing": [],
        "execution_health_recovery_closure_missing_count": 0,
        "execution_health_recovery_closure_proof_queue": [],
        "execution_health_recovery_closure_proof_queue_count": 0,
        "execution_health_recovery_closure_next_required_command": "",
        "execution_health_recovery_closure_next_proof_command": "",
        "execution_health_recovery_closure_blocks_completion_claim": False,
        "execution_learning_state": "unknown",
        "execution_learning_blocks_completion_claim": False,
        "execution_learning_missing": [],
        "execution_learning_missing_count": 0,
        "execution_learning_proof_queue": [],
        "execution_learning_proof_queue_count": 0,
        "execution_learning_next_required_command": "",
        "execution_learning_next_proof_command": "",
        "case_proof_requirements": ["case"],
        "missing_case_proofs": ["case"],
        "next_case_proof_commands": ["save execution case: <order>"],
        "next_case_proof_command": "save execution case: <order>",
        "mission_command_queue": [],
        "mission_command_count": 0,
        "initial_governor_command": "",
        "proof_bundle_command": "",
        "acceptance_command": "",
        "audit_command": "",
        "recovery_command": "",
        "learning_command": "",
        "blockers": 1,
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }.items():
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} no-case gate handoff missed {key}: {metadata}")
    for key in handoff:
        if key in {"source", *READ_ONLY_FLAGS}:
            continue
        if key in metadata and handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} no-case gate top-level/handoff diverged for {key}: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} no-case gate handoff unsafe nested metadata {key}: {metadata}")


def assert_execution_case_review_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_case_review_handoff") or {}
    if handoff.get("source") != "execution_case_review_packet":
        raise SystemExit(f"{label} missed case-review handoff source: {metadata}")
    if metadata.get("execution_case_review_handoff_ready") is not True or handoff.get("execution_case_review_handoff_ready") is not True:
        raise SystemExit(f"{label} missed case-review handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested case-review handoff readiness: {metadata}")
    for key in [
        "found",
        "case_id",
        "request",
        "review_state",
        "verdict",
        "mission_state",
        "go_no_go",
        "next_safe_command",
        "risk_signals",
        "approval_required",
        "forecast_new_approvals",
        "forecast_reused_approval_ids",
        "forecast_queue_before",
        "forecast_queue_after_if_sent",
        "forecast_queue_delta_if_sent",
        "pending_approvals",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "approval_held_review_command",
        "events",
        "evidence_preview_gate_count",
        "evidence_preview_ready_count",
        "evidence_preview_blocked_count",
        "evidence_preview_verdicts",
        "evidence_preview_latest_verdict",
        "evidence_preview_latest_event_id",
        "evidence_preview_latest_handoff",
        "evidence_preview_latest_handoff_present",
        "has_verification_evidence",
        "has_approval_evidence",
        "has_approval_chain_evidence",
        "approval_chain_status",
        "approval_evidence_ids",
        "approval_linked_run_ids",
        "verification_receipt_ids",
        "missing_verification_receipt_ids",
        "has_missing_verification_receipts",
        "verified_approval_run_ids",
        "has_verified_approval_run",
        "has_recovery_evidence",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_proof_handoff",
        "execution_proof_handoff_present",
        "execution_proof_state",
        "execution_proof_next_command",
        "execution_proof_can_trust_execution",
        *INHERITED_EXECUTION_PROOF_ALIAS_KEYS,
        "missing_case_proofs",
        "next_case_proof_commands",
        "next_case_proof_command",
        "mission_command_queue",
        "mission_command_count",
        "initial_governor_command",
        "proof_bundle_command",
        "acceptance_command",
        "audit_command",
        "recovery_command",
        "learning_command",
        "blockers",
        "checklist_items",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} case-review handoff field {key} diverged: {metadata}")
    if handoff.get("gate_handoff_present") is not True or not isinstance(handoff.get("gate_handoff"), dict):
        raise SystemExit(f"{label} case-review handoff missed inherited gate handoff: {metadata}")
    if handoff["gate_handoff"].get("source") != "execution_case_gate":
        raise SystemExit(f"{label} case-review inherited gate handoff has wrong source: {metadata}")
    if metadata.get("mission_command_count") != len(metadata.get("mission_command_queue") or []):
        raise SystemExit(f"{label} case-review mission command count diverged: {metadata}")
    for key in [
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
    ]:
        if key not in metadata or key not in handoff:
            raise SystemExit(f"{label} missed explicit case-review proof alias {key}: {metadata}")
    if metadata.get("execution_health_recovery_closure_proof_queue_count") != len(metadata.get("execution_health_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} case-review recovery proof count diverged: {metadata}")
    if metadata.get("execution_health_recovery_closure_required_commands") and metadata.get("execution_health_recovery_closure_proof_queue") != metadata.get("execution_health_recovery_closure_required_commands"):
        raise SystemExit(f"{label} case-review recovery proof queue diverged from required commands: {metadata}")
    if metadata.get("execution_health_recovery_closure_required_commands") and metadata.get("execution_health_recovery_closure_next_proof_command") != metadata.get("execution_health_recovery_closure_next_required_command"):
        raise SystemExit(f"{label} case-review recovery next proof diverged from next required command: {metadata}")
    if metadata.get("execution_learning_proof_queue_count") != len(metadata.get("execution_learning_proof_queue") or []):
        raise SystemExit(f"{label} case-review learning proof count diverged: {metadata}")
    if metadata.get("execution_learning_required_commands") and metadata.get("execution_learning_proof_queue") != metadata.get("execution_learning_required_commands"):
        raise SystemExit(f"{label} case-review learning proof queue diverged from required commands: {metadata}")
    if metadata.get("execution_learning_required_commands") and metadata.get("execution_learning_next_proof_command") != metadata.get("execution_learning_next_required_command"):
        raise SystemExit(f"{label} case-review learning next proof diverged from next required command: {metadata}")
    if metadata.get("evidence_preview_gate_count") != len(metadata.get("evidence_preview_verdicts") or []):
        raise SystemExit(f"{label} case-review evidence preview verdict count diverged: {metadata}")
    if metadata.get("evidence_preview_ready_count", 0) + metadata.get("evidence_preview_blocked_count", 0) != metadata.get("evidence_preview_gate_count"):
        raise SystemExit(f"{label} case-review evidence preview ready/blocked count diverged: {metadata}")
    if metadata.get("evidence_preview_gate_count", 0) > 0:
        if metadata.get("evidence_preview_latest_handoff_present") is not True:
            raise SystemExit(f"{label} case-review missed latest evidence preview handoff flag: {metadata}")
        if (metadata.get("evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"{label} case-review missed latest evidence preview handoff source: {metadata}")
    assert_inherited_execution_proof_aliases(metadata, label, "execution_case_review_handoff")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} case-review handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} case-review handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} case-review handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} case-review handoff unsafe nested metadata {key}: {metadata}")


def assert_no_case_review_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_case_review_handoff") or {}
    if handoff.get("source") != "execution_case_review_packet" or handoff.get("found") is not False:
        raise SystemExit(f"{label} missed no-case review handoff: {metadata}")
    if metadata.get("execution_case_review_handoff_ready") is not True or handoff.get("execution_case_review_handoff_ready") is not True:
        raise SystemExit(f"{label} missed no-case review handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested no-case review handoff readiness: {metadata}")
    assert_no_case_gate_handoff(
        {"execution_case_gate_handoff": handoff.get("gate_handoff") or {}, **{key: handoff.get(key) for key in handoff}},
        f"{label} inherited gate",
    )
    for key, expected in {
        "case_id": None,
        "request": "",
        "review_state": "NO_CASE",
        "verdict": "CASE_NOT_FOUND",
        "checklist_items": 0,
        "gate_handoff_present": False,
        "evidence_preview_gate_count": 0,
        "evidence_preview_ready_count": 0,
        "evidence_preview_blocked_count": 0,
        "evidence_preview_verdicts": [],
        "evidence_preview_latest_verdict": "",
        "evidence_preview_latest_event_id": None,
        "evidence_preview_latest_handoff": {},
        "evidence_preview_latest_handoff_present": False,
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }.items():
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} no-case review handoff missed {key}: {metadata}")
    if handoff.get("gate_handoff", {}).get("source") != "execution_case_gate":
        raise SystemExit(f"{label} no-case review missed inherited gate defaults: {metadata}")
    for key in [
        "review_state",
        "checklist_items",
        "evidence_preview_gate_count",
        "evidence_preview_ready_count",
        "evidence_preview_blocked_count",
        "evidence_preview_verdicts",
        "evidence_preview_latest_verdict",
        "evidence_preview_latest_event_id",
        "evidence_preview_latest_handoff",
        "evidence_preview_latest_handoff_present",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if metadata.get(key) != handoff.get(key):
            raise SystemExit(f"{label} no-case review top-level/handoff diverged for {key}: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} no-case review handoff unsafe nested metadata {key}: {metadata}")


def assert_execution_case_closure_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_case_closure_handoff") or {}
    if handoff.get("source") != "execution_case_closure_packet":
        raise SystemExit(f"{label} missed case-closure handoff source: {metadata}")
    if metadata.get("execution_case_closure_handoff_ready") is not True or handoff.get("execution_case_closure_handoff_ready") is not True:
        raise SystemExit(f"{label} missed case-closure handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested case-closure handoff readiness: {metadata}")
    for key in [
        "found",
        "case_id",
        "request",
        "closure_verdict",
        "gate_verdict",
        "review_state",
        "case_closure_ready",
        "case_closure_blocks_completion_claim",
        "missing_case_proofs",
        "missing_case_proof_count",
        "next_case_proof_commands",
        "next_case_proof_command",
        "closure_proof_queue",
        "closure_proof_queue_count",
        "next_closure_proof_command",
        "mission_command_queue",
        "mission_command_count",
        "events",
        "evidence_preview_gate_count",
        "evidence_preview_ready_count",
        "evidence_preview_blocked_count",
        "evidence_preview_verdicts",
        "evidence_preview_latest_verdict",
        "evidence_preview_latest_event_id",
        "evidence_preview_latest_handoff",
        "evidence_preview_latest_handoff_present",
        "approval_required",
        "has_verification_evidence",
        "has_approval_evidence",
        "has_approval_chain_evidence",
        "approval_chain_status",
        "approval_evidence_ids",
        "approval_linked_run_ids",
        "verification_receipt_ids",
        "runtime_trace_receipt_message_ids",
        "missing_verification_receipt_ids",
        "missing_runtime_trace_receipt_message_ids",
        "has_missing_verification_receipts",
        "has_missing_runtime_trace_receipts",
        "verified_approval_run_ids",
        "has_verified_approval_run",
        "has_recovery_evidence",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "approval_held_review_command",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_proof_handoff",
        "execution_proof_handoff_present",
        "execution_proof_state",
        "execution_proof_next_command",
        "execution_proof_can_trust_execution",
        *INHERITED_EXECUTION_PROOF_ALIAS_KEYS,
        "blockers",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} case-closure handoff field {key} diverged: {metadata}")
    if handoff.get("gate_handoff_present") is not True or not isinstance(handoff.get("gate_handoff"), dict):
        raise SystemExit(f"{label} case-closure handoff missed inherited gate handoff: {metadata}")
    if handoff["gate_handoff"].get("source") != "execution_case_gate":
        raise SystemExit(f"{label} case-closure inherited gate handoff has wrong source: {metadata}")
    if handoff.get("review_handoff_present") is not True or not isinstance(handoff.get("review_handoff"), dict):
        raise SystemExit(f"{label} case-closure handoff missed inherited review handoff: {metadata}")
    if handoff["review_handoff"].get("source") != "execution_case_review_packet":
        raise SystemExit(f"{label} case-closure inherited review handoff has wrong source: {metadata}")
    if metadata.get("missing_case_proof_count") != len(metadata.get("missing_case_proofs") or []):
        raise SystemExit(f"{label} case-closure missing proof count diverged: {metadata}")
    if metadata.get("closure_proof_queue_count") != len(metadata.get("closure_proof_queue") or []):
        raise SystemExit(f"{label} case-closure proof queue count diverged: {metadata}")
    if metadata.get("mission_command_count") != len(metadata.get("mission_command_queue") or []):
        raise SystemExit(f"{label} case-closure mission command count diverged: {metadata}")
    for key in [
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
    ]:
        if key not in metadata or key not in handoff:
            raise SystemExit(f"{label} missed explicit case-closure proof/required alias {key}: {metadata}")
    if metadata.get("execution_health_recovery_closure_proof_queue_count") != len(metadata.get("execution_health_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} case-closure recovery proof count diverged: {metadata}")
    if metadata.get("execution_health_recovery_closure_required_commands") and metadata.get("execution_health_recovery_closure_proof_queue") != metadata.get("execution_health_recovery_closure_required_commands"):
        raise SystemExit(f"{label} case-closure recovery proof queue diverged from required commands: {metadata}")
    if metadata.get("execution_health_recovery_closure_required_commands") and metadata.get("execution_health_recovery_closure_next_proof_command") != metadata.get("execution_health_recovery_closure_next_required_command"):
        raise SystemExit(f"{label} case-closure recovery next proof diverged from next required command: {metadata}")
    if metadata.get("execution_learning_proof_queue_count") != len(metadata.get("execution_learning_proof_queue") or []):
        raise SystemExit(f"{label} case-closure learning proof count diverged: {metadata}")
    if metadata.get("execution_learning_required_commands") and metadata.get("execution_learning_proof_queue") != metadata.get("execution_learning_required_commands"):
        raise SystemExit(f"{label} case-closure learning proof queue diverged from required commands: {metadata}")
    if metadata.get("execution_learning_required_commands") and metadata.get("execution_learning_next_proof_command") != metadata.get("execution_learning_next_required_command"):
        raise SystemExit(f"{label} case-closure learning next proof diverged from next required command: {metadata}")
    if metadata.get("evidence_preview_gate_count") != len(metadata.get("evidence_preview_verdicts") or []):
        raise SystemExit(f"{label} case-closure evidence preview verdict count diverged: {metadata}")
    if metadata.get("evidence_preview_ready_count", 0) + metadata.get("evidence_preview_blocked_count", 0) != metadata.get("evidence_preview_gate_count"):
        raise SystemExit(f"{label} case-closure evidence preview ready/blocked count diverged: {metadata}")
    if metadata.get("evidence_preview_gate_count", 0) > 0:
        if metadata.get("evidence_preview_latest_handoff_present") is not True:
            raise SystemExit(f"{label} case-closure missed latest evidence preview handoff flag: {metadata}")
        if (metadata.get("evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"{label} case-closure missed latest evidence preview handoff source: {metadata}")
    assert_inherited_execution_proof_aliases(metadata, label, "execution_case_closure_handoff")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} case-closure handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} case-closure handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} case-closure handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} case-closure handoff unsafe nested metadata {key}: {metadata}")


def assert_no_case_closure_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_case_closure_handoff") or {}
    if handoff.get("source") != "execution_case_closure_packet" or handoff.get("found") is not False:
        raise SystemExit(f"{label} missed no-case closure handoff: {metadata}")
    if metadata.get("execution_case_closure_handoff_ready") is not True or handoff.get("execution_case_closure_handoff_ready") is not True:
        raise SystemExit(f"{label} missed no-case closure handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested no-case closure handoff readiness: {metadata}")
    for key, expected in {
        "case_id": None,
        "request": "",
        "closure_verdict": "CASE_CLOSURE_NOT_FOUND",
        "gate_verdict": "CASE_NOT_FOUND",
        "review_state": "NO_CASE",
        "case_closure_ready": False,
        "case_closure_blocks_completion_claim": True,
        "missing_case_proofs": ["case"],
        "missing_case_proof_count": 1,
        "next_case_proof_commands": ["save execution case: <order>"],
        "next_case_proof_command": "save execution case: <order>",
        "closure_proof_queue": ["save execution case: <order>"],
        "closure_proof_queue_count": 1,
        "next_closure_proof_command": "save execution case: <order>",
        "evidence_preview_gate_count": 0,
        "evidence_preview_ready_count": 0,
        "evidence_preview_blocked_count": 0,
        "evidence_preview_verdicts": [],
        "evidence_preview_latest_verdict": "",
        "evidence_preview_latest_event_id": None,
        "evidence_preview_latest_handoff": {},
        "evidence_preview_latest_handoff_present": False,
        "gate_handoff_present": False,
        "review_handoff_present": False,
        "reason": "missing_case",
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }.items():
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} no-case closure handoff missed {key}: {metadata}")
    if handoff.get("gate_handoff", {}).get("source") != "execution_case_gate":
        raise SystemExit(f"{label} no-case closure missed inherited gate defaults: {metadata}")
    if handoff.get("review_handoff", {}).get("source") != "execution_case_review_packet":
        raise SystemExit(f"{label} no-case closure missed inherited review defaults: {metadata}")
    for key in [
        "closure_verdict",
        "gate_verdict",
        "review_state",
        "case_closure_ready",
        "case_closure_blocks_completion_claim",
        "missing_case_proofs",
        "missing_case_proof_count",
        "closure_proof_queue",
        "closure_proof_queue_count",
        "next_closure_proof_command",
        "evidence_preview_gate_count",
        "evidence_preview_ready_count",
        "evidence_preview_blocked_count",
        "evidence_preview_verdicts",
        "evidence_preview_latest_verdict",
        "evidence_preview_latest_event_id",
        "evidence_preview_latest_handoff",
        "evidence_preview_latest_handoff_present",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if metadata.get(key) != handoff.get(key):
            raise SystemExit(f"{label} no-case closure top-level/handoff diverged for {key}: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} no-case closure handoff unsafe nested metadata {key}: {metadata}")


def assert_execution_case_timeline_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_case_timeline_handoff") or {}
    if handoff.get("source") != "execution_case_timeline":
        raise SystemExit(f"{label} missed case-timeline handoff source: {metadata}")
    if metadata.get("execution_case_timeline_handoff_ready") is not True or handoff.get("execution_case_timeline_handoff_ready") is not True:
        raise SystemExit(f"{label} missed case-timeline handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested case-timeline handoff readiness: {metadata}")
    for key in [
        "found",
        "case_id",
        "request",
        "review_state",
        "verdict",
        "mission_state",
        "go_no_go",
        "next_safe_command",
        "risk_signals",
        "events",
        "recent_runs",
        "pending_approvals",
        "blockers",
        "approval_required",
        "approval_queue_forecast",
        "forecast_new_approvals",
        "forecast_reused_approval_ids",
        "forecast_queue_before",
        "forecast_queue_after_if_sent",
        "forecast_queue_delta_if_sent",
        "has_approval_evidence",
        "has_approval_chain_evidence",
        "approval_chain_status",
        "approval_evidence_ids",
        "approval_linked_run_ids",
        "verification_receipt_ids",
        "runtime_trace_receipt_message_ids",
        "missing_verification_receipt_ids",
        "missing_runtime_trace_receipt_message_ids",
        "has_missing_verification_receipts",
        "has_missing_runtime_trace_receipts",
        "verified_approval_run_ids",
        "has_verified_approval_run",
        "has_execution_evidence",
        "has_verification_evidence",
        "has_recovery_evidence",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_health_recovery_closure_target_run_id",
        "execution_health_recovery_closure_target_tool_name",
        "execution_health_recovery_closure_target_verification_receipts",
        "execution_health_recovery_closure_target_recovery_packets",
        "execution_health_recovery_closure_target_after_action_learning_packets",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_target_run_id",
        "execution_learning_target_tool_name",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_proof_handoff",
        "execution_proof_handoff_present",
        "execution_proof_state",
        "execution_proof_next_command",
        "execution_proof_can_trust_execution",
        *INHERITED_EXECUTION_PROOF_ALIAS_KEYS,
        "missing_case_proofs",
        "next_case_proof_commands",
        "next_case_proof_command",
        "mission_command_queue",
        "mission_command_count",
        "lifecycle_stages",
        "complete_stages",
        "missing_stages",
        "missing_stage_count",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "approval_held_review_command",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if key in metadata and handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} case-timeline handoff field {key} diverged: {metadata}")
    if handoff.get("timeline_items") != handoff.get("evidence_event_count", 0) + 1:
        raise SystemExit(f"{label} case-timeline handoff timeline item count diverged: {metadata}")
    if handoff.get("risk_signal_count") != len(handoff.get("risk_signals") or []):
        raise SystemExit(f"{label} case-timeline handoff risk count diverged: {metadata}")
    if handoff.get("forecast_reused_approval_count") != len(handoff.get("forecast_reused_approval_ids") or []):
        raise SystemExit(f"{label} case-timeline handoff reused approval count diverged: {metadata}")
    if handoff.get("missing_case_proof_count") != len(handoff.get("missing_case_proofs") or []):
        raise SystemExit(f"{label} case-timeline handoff missing proof count diverged: {metadata}")
    if handoff.get("mission_command_count") != len(handoff.get("mission_command_queue") or []):
        raise SystemExit(f"{label} case-timeline handoff mission command count diverged: {metadata}")
    for key in [
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
    ]:
        if key not in metadata or key not in handoff:
            raise SystemExit(f"{label} missed explicit case-timeline proof alias {key}: {metadata}")
    if handoff.get("execution_health_recovery_closure_proof_queue_count") != len(handoff.get("execution_health_recovery_closure_proof_queue") or []):
        raise SystemExit(f"{label} case-timeline handoff recovery proof count diverged: {metadata}")
    if handoff.get("execution_health_recovery_closure_required_commands") and handoff.get("execution_health_recovery_closure_proof_queue") != handoff.get("execution_health_recovery_closure_required_commands"):
        raise SystemExit(f"{label} case-timeline recovery proof queue diverged from required commands: {metadata}")
    if handoff.get("execution_health_recovery_closure_required_commands") and handoff.get("execution_health_recovery_closure_next_proof_command") != handoff.get("execution_health_recovery_closure_next_required_command"):
        raise SystemExit(f"{label} case-timeline recovery next proof diverged from next required command: {metadata}")
    if handoff.get("execution_learning_proof_queue_count") != len(handoff.get("execution_learning_proof_queue") or []):
        raise SystemExit(f"{label} case-timeline handoff learning proof count diverged: {metadata}")
    if handoff.get("execution_learning_required_commands") and handoff.get("execution_learning_proof_queue") != handoff.get("execution_learning_required_commands"):
        raise SystemExit(f"{label} case-timeline learning proof queue diverged from required commands: {metadata}")
    if handoff.get("execution_learning_required_commands") and handoff.get("execution_learning_next_proof_command") != handoff.get("execution_learning_next_required_command"):
        raise SystemExit(f"{label} case-timeline learning next proof diverged from next required command: {metadata}")
    if handoff.get("complete_stage_count") != len(handoff.get("complete_stages") or []):
        raise SystemExit(f"{label} case-timeline handoff complete stage count diverged: {metadata}")
    if handoff.get("missing_stage_count") != len(handoff.get("missing_stages") or []):
        raise SystemExit(f"{label} case-timeline handoff missing stage count diverged: {metadata}")
    if handoff.get("gate_handoff_present") is not True or not isinstance(handoff.get("gate_handoff"), dict):
        raise SystemExit(f"{label} case-timeline handoff missed inherited gate handoff: {metadata}")
    if handoff["gate_handoff"].get("source") != "execution_case_gate":
        raise SystemExit(f"{label} case-timeline inherited gate handoff has wrong source: {metadata}")
    assert_inherited_execution_proof_aliases(metadata, label, "execution_case_timeline_handoff")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} case-timeline handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} case-timeline handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} case-timeline handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} case-timeline handoff unsafe nested metadata {key}: {metadata}")


def assert_no_case_timeline_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_case_timeline_handoff") or {}
    if handoff.get("source") != "execution_case_timeline" or handoff.get("found") is not False:
        raise SystemExit(f"{label} missed no-case timeline handoff: {metadata}")
    if metadata.get("execution_case_timeline_handoff_ready") is not True or handoff.get("execution_case_timeline_handoff_ready") is not True:
        raise SystemExit(f"{label} missed no-case timeline handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested no-case timeline handoff readiness: {metadata}")
    for key, expected in {
        "case_id": None,
        "request": "",
        "review_state": "NO_CASE",
        "verdict": "CASE_NOT_FOUND",
        "timeline_items": 0,
        "evidence_event_count": 0,
        "recent_run_count": 0,
        "events": 0,
        "recent_runs": 0,
        "missing_stages": [],
        "missing_stage_count": 0,
        "approval_required": False,
        "approval_chain_status": "missing",
        "mission_command_queue": [],
        "mission_command_count": 0,
        "evidence_preview_gate_count": 0,
        "evidence_preview_ready_count": 0,
        "evidence_preview_blocked_count": 0,
        "evidence_preview_verdicts": [],
        "evidence_preview_latest_verdict": "",
        "evidence_preview_latest_event_id": None,
        "evidence_preview_latest_handoff": {},
        "evidence_preview_latest_handoff_present": False,
        "gate_handoff_present": False,
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }.items():
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} no-case timeline handoff missed {key}: {metadata}")
    for key in [
        "evidence_preview_gate_count",
        "evidence_preview_ready_count",
        "evidence_preview_blocked_count",
        "evidence_preview_verdicts",
        "evidence_preview_latest_verdict",
        "evidence_preview_latest_event_id",
        "evidence_preview_latest_handoff",
        "evidence_preview_latest_handoff_present",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if metadata.get(key) != handoff.get(key):
            raise SystemExit(f"{label} no-case timeline evidence-preflight top-level/handoff diverged for {key}: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} no-case timeline handoff unsafe nested metadata {key}: {metadata}")


def assert_execution_runbook_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_runbook_handoff") or {}
    if handoff.get("source") != "execution_runbook":
        raise SystemExit(f"{label} missed execution-runbook handoff source: {metadata}")
    if metadata.get("execution_runbook_handoff_ready") is not True or handoff.get("execution_runbook_handoff_ready") is not True:
        raise SystemExit(f"{label} missed execution-runbook handoff ready flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested execution-runbook handoff readiness: {metadata}")
    for key in [
        "request",
        "runbook_state",
        "next_step",
        "risk_signals",
        "approval_required",
        "pending_approvals",
        "recent_failed_runs",
        "recent_approval_held_runs",
        "approval_held_review_command",
        "recent_verification_runs",
        "before_step_names",
        "before_steps",
        "during_step_names",
        "during_steps",
        "after_step_names",
        "after_steps",
        "route_packets",
        "proof_packets",
        "recovery_packets",
        "blocker_details",
        "blocker_count",
        "blockers",
        "learning_closure_step",
        "execution_learning_closure_command",
        "execution_health_recovery_closure_state",
        "execution_health_recovery_closure_ready_to_retry",
        "execution_health_recovery_closure_missing",
        "execution_health_recovery_closure_missing_count",
        "execution_health_recovery_closure_required_commands",
        "execution_health_recovery_closure_next_required_command",
        "execution_health_recovery_closure_checklist_command",
        "execution_health_recovery_closure_should_open_checklist",
        "execution_health_recovery_closure_proof_queue",
        "execution_health_recovery_closure_proof_queue_count",
        "execution_health_recovery_closure_next_proof_command",
        "execution_health_recovery_closure_blocks_completion_claim",
        "execution_health_recovery_closure_target_run_id",
        "execution_health_recovery_closure_target_tool_name",
        "execution_health_recovery_closure_target_verification_receipts",
        "execution_health_recovery_closure_target_recovery_packets",
        "execution_health_recovery_closure_target_after_action_learning_packets",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_recent_action_runs",
        "execution_learning_failed_or_blocked_action_runs",
        "execution_learning_recent_verification_runs",
        "execution_learning_recent_recovery_runs",
        "execution_learning_recent_after_action_learning_runs",
        "execution_learning_target_run_id",
        "execution_learning_target_tool_name",
        "execution_learning_target_after_action_learning_packets",
        "execution_learning_missing",
        "execution_learning_missing_count",
        "execution_learning_evidence_command",
        "execution_learning_after_action_learning_command",
        "execution_learning_required_commands",
        "execution_learning_next_required_command",
        "execution_learning_proof_queue",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
        "execution_learning_actionable_required_commands",
        "execution_learning_actionable_required_command_count",
        "execution_learning_actionable_proof_queue",
        "execution_learning_actionable_proof_queue_count",
        "execution_learning_actionable_next_required_command",
        "execution_learning_actionable_next_proof_command",
        "execution_learning_next_evidence_command",
        "runbook_proof_queue",
        "runbook_proof_queue_count",
        "runbook_next_proof_command",
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} execution-runbook handoff field {key} diverged: {metadata}")
    if handoff.get("risk_signal_count") != len(handoff.get("risk_signals") or []):
        raise SystemExit(f"{label} execution-runbook handoff risk count diverged: {metadata}")
    if handoff.get("route_packet_count") != len(handoff.get("route_packets") or []):
        raise SystemExit(f"{label} execution-runbook handoff route packet count diverged: {metadata}")
    if handoff.get("proof_packet_count") != len(handoff.get("proof_packets") or []):
        raise SystemExit(f"{label} execution-runbook handoff proof packet count diverged: {metadata}")
    if handoff.get("recovery_packet_count") != len(handoff.get("recovery_packets") or []):
        raise SystemExit(f"{label} execution-runbook handoff recovery packet count diverged: {metadata}")
    if handoff.get("before_steps") != len(handoff.get("before_step_names") or []):
        raise SystemExit(f"{label} execution-runbook handoff before-step count diverged: {metadata}")
    if handoff.get("during_steps") != len(handoff.get("during_step_names") or []):
        raise SystemExit(f"{label} execution-runbook handoff during-step count diverged: {metadata}")
    if handoff.get("after_steps") != len(handoff.get("after_step_names") or []):
        raise SystemExit(f"{label} execution-runbook handoff after-step count diverged: {metadata}")
    if handoff.get("blocker_count") != len(handoff.get("blocker_details") or []):
        raise SystemExit(f"{label} execution-runbook handoff blocker count diverged: {metadata}")
    assert_execution_health_recovery_proof_aliases(handoff, f"{label} execution-runbook handoff")
    assert_execution_learning_proof_aliases(handoff, f"{label} execution-runbook handoff")
    assert_execution_learning_actionable_aliases(handoff, f"{label} execution-runbook handoff")
    if handoff.get("runbook_proof_queue_count") != len(handoff.get("runbook_proof_queue") or []):
        raise SystemExit(f"{label} execution-runbook handoff proof queue count diverged: {metadata}")
    if handoff.get("runbook_proof_queue") and handoff.get("runbook_next_proof_command") != handoff["runbook_proof_queue"][0]:
        raise SystemExit(f"{label} execution-runbook handoff next proof command diverged: {metadata}")
    if str(handoff.get("next_step") or "").strip("`") not in (handoff.get("runbook_proof_queue") or []):
        raise SystemExit(f"{label} execution-runbook handoff proof queue missed next step: {metadata}")
    for command in (handoff.get("execution_health_recovery_closure_proof_queue") or []):
        if command not in (handoff.get("runbook_proof_queue") or []):
            raise SystemExit(f"{label} execution-runbook handoff missed recovery proof command in runbook queue: {metadata}")
    for command in (handoff.get("execution_learning_actionable_proof_queue") or []):
        if command not in (handoff.get("runbook_proof_queue") or []):
            raise SystemExit(f"{label} execution-runbook handoff missed actionable learning command in runbook queue: {metadata}")
    if handoff.get("draft_only") is not True or handoff.get("requires_manual_send") is not True or handoff.get("loads_without_execution") is not True:
        raise SystemExit(f"{label} execution-runbook handoff missed review-only contract: {metadata}")
    if handoff.get("authorizes_execution") is not False or handoff.get("authorizes_completion_claim") is not False:
        raise SystemExit(f"{label} execution-runbook handoff should not authorize execution/completion: {metadata}")
    if handoff.get("approval_granted") is not False:
        raise SystemExit(f"{label} execution-runbook handoff should not grant approval: {metadata}")
    for key in READ_ONLY_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} execution-runbook handoff unsafe nested metadata {key}: {metadata}")


def assert_no_request_execution_runbook_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("execution_runbook_handoff") or {}
    expected = {
        "source": "execution_runbook",
        "execution_runbook_handoff_ready": True,
        "reason": "missing_request",
        "found": False,
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value or handoff.get(key) != value:
            raise SystemExit(f"{label} no-request execution-runbook handoff missed {key}: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} no-request execution-runbook handoff missed nested readiness: {metadata}")
    for key in READ_ONLY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} no-request execution-runbook handoff unsafe metadata {key}: {metadata}")


def assert_harness_operator_surfaces_separate_approval_held_runs() -> None:
    with TemporaryDirectory(prefix="jarvis-harness-approval-held-") as temp:
        runtime = make_temp_runtime(Path(temp))
        failed_id = runtime.store.log_tool_run(
            session_id=runtime.session_id,
            tool_name="checkpoint_recovery_execute",
            risk="LOCAL_SAFE",
            ok=False,
            approved=False,
            output="local recovery verification failed",
            metadata={"failure_kind": "verification_failed"},
        )
        held_id = runtime.store.log_tool_run(
            session_id=runtime.session_id,
            tool_name="send_telegram",
            risk="EXTERNAL_SIDE_EFFECT",
            ok=False,
            approved=False,
            approval_id=42,
            output="queued as approval #42",
            metadata={
                "approval_required": True,
                "approval_id": 42,
                "failure_kind": "approval_required",
            },
        )

        learning_debt = _execution_learning_debt_snapshot(runtime.store.recent_tool_runs(limit=20))
        if learning_debt.get("failed_or_blocked_action_runs") != 1:
            raise SystemExit(f"Harness learning debt should exclude approval-held rows from failure count: {learning_debt}")
        if learning_debt.get("approval_held_action_runs") != 1:
            raise SystemExit(f"Harness learning debt should expose approval-held action count: {learning_debt}")
        if learning_debt.get("target_run_id") != failed_id:
            raise SystemExit(f"Harness learning debt should target true failure #{failed_id}, not held #{held_id}: {learning_debt}")
        if f"execution recovery packet {held_id}" in (learning_debt.get("required_commands") or []):
            raise SystemExit(f"Harness learning debt should not request recovery for approval-held row #{held_id}: {learning_debt}")

        control = runtime.handle("harness control: summarize local harness proof state")
        if not control.verified or control.tool_results[0].tool_name != "harness_control_surface":
            raise SystemExit(f"Harness control should route with mixed failed/held rows: {control}")
        control_metadata = control.tool_results[0].metadata
        if control_metadata.get("recent_failed_runs") != 1 or control_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Harness control should separate failed and approval-held rows: {control_metadata}")
        if control_metadata.get("recovery_closure_target_run_id") != failed_id:
            raise SystemExit(f"Harness control recovery should target true failure #{failed_id}: {control_metadata}")
        if f"execution recovery packet {held_id}" in (control_metadata.get("recovery_closure_required_commands") or []):
            raise SystemExit(f"Harness control should not recover approval-held row #{held_id}: {control_metadata}")
        if "recent failed/blocked runs: 1" not in control.response or "recent approval-held runs: 1" not in control.response:
            raise SystemExit(f"Harness control should render separated attention counts: {control.response}")
        assert_harness_control_handoff(control_metadata, "mixed approval-held harness control")

        agi_next = runtime.handle("agi next build move")
        if not agi_next.verified or agi_next.tool_results[0].tool_name != "agi_next_build_move":
            raise SystemExit(f"AGI next build should route with mixed failed/held rows: {agi_next}")
        agi_next_metadata = agi_next.tool_results[0].metadata
        if agi_next_metadata.get("recent_failed_runs") != 1 or agi_next_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"AGI next build should separate failed and approval-held rows: {agi_next_metadata}")
        if agi_next_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"AGI next build should surface held approval review command: {agi_next_metadata}")
        if f"execution recovery packet {held_id}" in agi_next.response:
            raise SystemExit(f"AGI next build should not route held row #{held_id} to recovery: {agi_next.response}")
        if "recent failed/blocked runs: 1" not in agi_next.response or "recent approval-held runs: 1" not in agi_next.response:
            raise SystemExit(f"AGI next build should render separated attention counts: {agi_next.response}")
        assert_agi_next_build_handoff(agi_next_metadata, "mixed approval-held AGI next build")

        completion = runtime.handle("harness completion")
        if not completion.verified or completion.tool_results[0].tool_name != "harness_completion_assessment":
            raise SystemExit(f"Harness completion should route with mixed failed/held rows: {completion}")
        completion_metadata = completion.tool_results[0].metadata
        if completion_metadata.get("recent_failed_runs") != 1 or completion_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Harness completion should separate failed and approval-held rows: {completion_metadata}")
        if completion_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Harness completion should surface held approval review command: {completion_metadata}")
        if "approval readiness 42" not in (completion_metadata.get("completion_proof_queue") or []):
            raise SystemExit(f"Harness completion proof queue should include approval review: {completion_metadata}")
        if f"execution recovery packet {held_id}" in completion.response or f"execution recovery packet {held_id}" in " ".join(completion_metadata.get("completion_proof_queue") or []):
            raise SystemExit(f"Harness completion should not route held row #{held_id} to recovery: {completion_metadata}")
        if "recent failed/blocked runs: 1" not in completion.response or "recent approval-held runs: 1" not in completion.response:
            raise SystemExit(f"Harness completion should render separated attention counts: {completion.response}")
        assert_harness_completion_handoff(completion_metadata, "mixed approval-held harness completion")

        audit = runtime.handle("completion audit")
        if not audit.verified or audit.tool_results[0].tool_name != "completion_audit_packet":
            raise SystemExit(f"Completion audit should route with mixed failed/held rows: {audit}")
        audit_metadata = audit.tool_results[0].metadata
        if audit_metadata.get("recent_failed_runs") != 1 or audit_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Completion audit should separate failed and approval-held rows: {audit_metadata}")
        if audit_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Completion audit should surface held approval review command: {audit_metadata}")
        if "approval readiness 42" not in (audit_metadata.get("completion_proof_queue") or []):
            raise SystemExit(f"Completion audit proof queue should include approval review: {audit_metadata}")
        if f"execution recovery packet {held_id}" in audit.response or f"execution recovery packet {held_id}" in " ".join(audit_metadata.get("completion_proof_queue") or []):
            raise SystemExit(f"Completion audit should not route held row #{held_id} to recovery: {audit_metadata}")
        if "recent failed/blocked tool runs: 1" not in audit.response or "recent approval-held tool runs: 1" not in audit.response:
            raise SystemExit(f"Completion audit should render separated attention counts: {audit.response}")
        assert_completion_audit_handoff(audit_metadata, "mixed approval-held completion audit")

        mission = runtime.handle("execution mission control: summarize local harness proof state")
        if not mission.verified or mission.tool_results[0].tool_name != "execution_mission_control":
            raise SystemExit(f"Execution mission should route with mixed failed/held rows: {mission}")
        mission_metadata = mission.tool_results[0].metadata
        if mission_metadata.get("recent_failed_runs") != 1 or mission_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution mission should separate failed and approval-held rows: {mission_metadata}")
        if mission_metadata.get("next_command") != f"execution recovery packet {failed_id}":
            raise SystemExit(f"Execution mission should target true failure first in mixed rows: {mission_metadata}")
        if mission_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Execution mission should expose held approval review command: {mission_metadata}")
        if "approval readiness 42" not in (mission_metadata.get("mission_command_queue") or []):
            raise SystemExit(f"Execution mission queue should include approval review command: {mission_metadata}")
        if f"execution recovery packet {held_id}" in mission.response or f"execution recovery packet {held_id}" in " ".join(mission_metadata.get("mission_command_queue") or []):
            raise SystemExit(f"Execution mission should not route held row #{held_id} to recovery: {mission_metadata}")
        if "recent failed/blocked runs: 1" not in mission.response or "recent approval-held runs: 1" not in mission.response:
            raise SystemExit(f"Execution mission should render separated attention counts: {mission.response}")
        assert_execution_mission_handoff(mission_metadata, "mixed approval-held execution mission")

        runbook = runtime.handle("execution runbook: summarize local harness proof state")
        if not runbook.verified or runbook.tool_results[0].tool_name != "execution_runbook":
            raise SystemExit(f"Execution runbook should route with mixed failed/held rows: {runbook}")
        runbook_metadata = runbook.tool_results[0].metadata
        if runbook_metadata.get("recent_failed_runs") != 1 or runbook_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution runbook should separate failed and approval-held rows: {runbook_metadata}")
        if runbook_metadata.get("next_step") != f"execution recovery packet {failed_id}":
            raise SystemExit(f"Execution runbook should target true failure first in mixed rows: {runbook_metadata}")
        if runbook_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Execution runbook should expose held approval review command: {runbook_metadata}")
        if "approval readiness 42" not in (runbook_metadata.get("runbook_proof_queue") or []):
            raise SystemExit(f"Execution runbook queue should include approval review command: {runbook_metadata}")
        if f"execution recovery packet {held_id}" in runbook.response or f"execution recovery packet {held_id}" in " ".join(runbook_metadata.get("runbook_proof_queue") or []):
            raise SystemExit(f"Execution runbook should not route held row #{held_id} to recovery: {runbook_metadata}")
        if "recent failed/blocked runs: 1" not in runbook.response or "recent approval-held runs: 1" not in runbook.response:
            raise SystemExit(f"Execution runbook should render separated attention counts: {runbook.response}")
        assert_execution_runbook_handoff(runbook_metadata, "mixed approval-held execution runbook")

        saved_case = runtime.handle("save execution case: summarize local harness proof state")
        if not saved_case.verified:
            raise SystemExit(f"Expected mixed approval-held execution case save to run local-safe: {saved_case}")
        saved_case_id = saved_case.tool_results[0].metadata.get("case_id")
        gate = runtime.registry.get("execution_case_gate").handler({"case_id": saved_case_id})
        if not gate.ok:
            raise SystemExit(f"Mixed approval-held execution case gate should run: {gate}")
        gate_metadata = gate.metadata
        if gate_metadata.get("recent_failed_runs") != 1 or gate_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution case gate should separate failed and approval-held rows: {gate_metadata}")
        if gate_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Execution case gate should expose held approval review command: {gate_metadata}")
        if "approval readiness 42" not in (gate_metadata.get("next_case_proof_commands") or []):
            raise SystemExit(f"Execution case gate proof commands should include approval review: {gate_metadata}")
        if f"execution recovery packet {held_id}" in gate.output or f"execution recovery packet {held_id}" in " ".join(gate_metadata.get("next_case_proof_commands") or []):
            raise SystemExit(f"Execution case gate should not route held row #{held_id} to recovery: {gate_metadata}")
        if "recent failed/blocked runs: 1" not in gate.output or "recent approval-held runs: 1" not in gate.output:
            raise SystemExit(f"Execution case gate should render separated attention counts: {gate.output}")
        assert_execution_case_gate_handoff(gate_metadata, "mixed approval-held execution case gate")

        review = runtime.registry.get("execution_case_review_packet").handler({"case_id": saved_case_id})
        if not review.ok:
            raise SystemExit(f"Mixed approval-held execution case review should run: {review}")
        review_metadata = review.metadata
        if review_metadata.get("recent_failed_runs") != 1 or review_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution case review should separate failed and approval-held rows: {review_metadata}")
        if review_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Execution case review should expose held approval review command: {review_metadata}")
        if f"execution recovery packet {held_id}" in review.output or f"execution recovery packet {held_id}" in " ".join(review_metadata.get("next_case_proof_commands") or []):
            raise SystemExit(f"Execution case review should not route held row #{held_id} to recovery: {review_metadata}")
        if "recent failed/blocked runs: 1" not in review.output or "recent approval-held runs: 1" not in review.output:
            raise SystemExit(f"Execution case review should render separated attention counts: {review.output}")
        assert_execution_case_review_handoff(review_metadata, "mixed approval-held execution case review")

        closure = runtime.registry.get("execution_case_closure_packet").handler({"case_id": saved_case_id})
        if not closure.ok:
            raise SystemExit(f"Mixed approval-held execution case closure should run: {closure}")
        closure_metadata = closure.metadata
        if closure_metadata.get("recent_failed_runs") != 1 or closure_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution case closure should separate failed and approval-held rows: {closure_metadata}")
        if closure_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Execution case closure should expose held approval review command: {closure_metadata}")
        if "approval readiness 42" not in (closure_metadata.get("closure_proof_queue") or []):
            raise SystemExit(f"Execution case closure proof queue should include approval review: {closure_metadata}")
        if f"execution recovery packet {held_id}" in closure.output or f"execution recovery packet {held_id}" in " ".join(closure_metadata.get("closure_proof_queue") or []):
            raise SystemExit(f"Execution case closure should not route held row #{held_id} to recovery: {closure_metadata}")
        if "recent failed/blocked runs: 1" not in closure.output or "recent approval-held runs: 1" not in closure.output:
            raise SystemExit(f"Execution case closure should render separated attention counts: {closure.output}")
        assert_execution_case_closure_handoff(closure_metadata, "mixed approval-held execution case closure")

        timeline = runtime.registry.get("execution_case_timeline").handler({"case_id": saved_case_id})
        if not timeline.ok:
            raise SystemExit(f"Mixed approval-held execution case timeline should run: {timeline}")
        timeline_metadata = timeline.metadata
        if timeline_metadata.get("recent_failed_runs") != 1 or timeline_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution case timeline should separate failed and approval-held rows: {timeline_metadata}")
        if timeline_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Execution case timeline should expose held approval review command: {timeline_metadata}")
        if f"execution recovery packet {held_id}" in timeline.output:
            raise SystemExit(f"Execution case timeline should not route held row #{held_id} to recovery: {timeline.output}")
        if "recent failed/blocked runs: 1" not in timeline.output or "recent approval-held runs: 1" not in timeline.output:
            raise SystemExit(f"Execution case timeline should render separated attention counts: {timeline.output}")
        if "approval-held" not in timeline.output:
            raise SystemExit(f"Execution case timeline should label approval-held runtime rows: {timeline.output}")
        assert_execution_case_timeline_handoff(timeline_metadata, "mixed approval-held execution case timeline")

    with TemporaryDirectory(prefix="jarvis-harness-approval-held-only-") as temp:
        runtime = make_temp_runtime(Path(temp))
        held_id = runtime.store.log_tool_run(
            session_id=runtime.session_id,
            tool_name="send_telegram",
            risk="EXTERNAL_SIDE_EFFECT",
            ok=False,
            approved=False,
            approval_id=42,
            output="queued as approval #42",
            metadata={
                "approval_required": True,
                "approval_id": 42,
                "failure_stage": "approval_required",
            },
        )

        learning_debt = _execution_learning_debt_snapshot(runtime.store.recent_tool_runs(limit=20))
        if learning_debt.get("failed_or_blocked_action_runs") != 0 or learning_debt.get("approval_held_action_runs") != 1:
            raise SystemExit(f"Harness learning debt should classify approval-held-only rows separately: {learning_debt}")
        if learning_debt.get("blocks_completion_claim") is not False:
            raise SystemExit(f"Approval-held-only learning state should not create failure learning debt: {learning_debt}")
        if learning_debt.get("target_run_id") is not None:
            raise SystemExit(f"Approval-held-only learning state should not target held run #{held_id}: {learning_debt}")
        if any(str(held_id) in command and "recovery" in command for command in learning_debt.get("required_commands") or []):
            raise SystemExit(f"Approval-held-only learning state should not request recovery for held run #{held_id}: {learning_debt}")

        control = runtime.handle("harness control: summarize local harness proof state")
        control_metadata = control.tool_results[0].metadata
        if control_metadata.get("verdict") != "APPROVAL_REVIEW_REQUIRED":
            raise SystemExit(f"Harness control should review approval-held rows without recovery: {control_metadata}")
        if control_metadata.get("next_command") != "approval readiness 42":
            raise SystemExit(f"Harness control should surface held approval id: {control_metadata}")
        if control_metadata.get("recent_failed_runs") != 0 or control_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Harness control approval-held-only counts drifted: {control_metadata}")
        if control_metadata.get("recovery_closure_blocks_auto_execution") is not False:
            raise SystemExit(f"Harness control should not open recovery closure for approval-held-only rows: {control_metadata}")
        assert_harness_control_handoff(control_metadata, "approval-held-only harness control")

        operations = runtime.handle("harness operations: summarize local harness proof state")
        operations_metadata = operations.tool_results[0].metadata
        if operations_metadata.get("next_move") != "review_approval_held_run":
            raise SystemExit(f"Harness operations should pick approval review for held-only rows: {operations_metadata}")
        if operations_metadata.get("next_command") != "approval readiness 42":
            raise SystemExit(f"Harness operations should surface held approval id: {operations_metadata}")
        if operations_metadata.get("recent_failed_runs") != 0 or operations_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Harness operations approval-held-only counts drifted: {operations_metadata}")
        if operations_metadata.get("safe_to_continue") is not False or operations_metadata.get("can_auto_run") is not False:
            raise SystemExit(f"Harness operations should pause for approval review on held-only rows: {operations_metadata}")
        if "recent failed/blocked runs: 0" not in operations.response or "recent approval-held runs: 1" not in operations.response:
            raise SystemExit(f"Harness operations should render separated attention counts: {operations.response}")
        assert_harness_operations_handoff(operations_metadata, "approval-held-only harness operations")

        agi_next = runtime.handle("agi next build move")
        agi_next_metadata = agi_next.tool_results[0].metadata
        if agi_next_metadata.get("recent_failed_runs") != 0 or agi_next_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"AGI next build approval-held-only counts drifted: {agi_next_metadata}")
        if agi_next_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"AGI next build should review held-only approval rows: {agi_next_metadata}")
        if f"execution recovery packet {held_id}" in agi_next.response:
            raise SystemExit(f"AGI next build should not recover approval-held-only row #{held_id}: {agi_next.response}")
        if "recent failed/blocked runs: 0" not in agi_next.response or "recent approval-held runs: 1" not in agi_next.response:
            raise SystemExit(f"AGI next build should render approval-held-only counts: {agi_next.response}")
        assert_agi_next_build_handoff(agi_next_metadata, "approval-held-only AGI next build")

        completion = runtime.handle("harness completion")
        completion_metadata = completion.tool_results[0].metadata
        if completion_metadata.get("recent_failed_runs") != 0 or completion_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Harness completion approval-held-only counts drifted: {completion_metadata}")
        if completion_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Harness completion should review held-only approval rows: {completion_metadata}")
        if completion_metadata.get("completion_claim_ready") is not False:
            raise SystemExit(f"Harness completion should not claim ready while approval-held rows need review: {completion_metadata}")
        if "approval readiness 42" not in (completion_metadata.get("completion_proof_queue") or []):
            raise SystemExit(f"Harness completion proof queue should include held-only approval review: {completion_metadata}")
        if f"execution recovery packet {held_id}" in completion.response or f"execution recovery packet {held_id}" in " ".join(completion_metadata.get("completion_proof_queue") or []):
            raise SystemExit(f"Harness completion should not recover approval-held-only row #{held_id}: {completion_metadata}")
        if "recent failed/blocked runs: 0" not in completion.response or "recent approval-held runs: 1" not in completion.response:
            raise SystemExit(f"Harness completion should render approval-held-only counts: {completion.response}")
        assert_harness_completion_handoff(completion_metadata, "approval-held-only harness completion")

        audit = runtime.handle("completion audit")
        audit_metadata = audit.tool_results[0].metadata
        if audit_metadata.get("recent_failed_runs") != 0 or audit_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Completion audit approval-held-only counts drifted: {audit_metadata}")
        if audit_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Completion audit should review held-only approval rows: {audit_metadata}")
        if audit_metadata.get("completion_claim_ready") is not False:
            raise SystemExit(f"Completion audit should not claim ready while approval-held rows need review: {audit_metadata}")
        if "approval readiness 42" not in (audit_metadata.get("completion_proof_queue") or []):
            raise SystemExit(f"Completion audit proof queue should include held-only approval review: {audit_metadata}")
        if f"execution recovery packet {held_id}" in audit.response or f"execution recovery packet {held_id}" in " ".join(audit_metadata.get("completion_proof_queue") or []):
            raise SystemExit(f"Completion audit should not recover approval-held-only row #{held_id}: {audit_metadata}")
        if "recent failed/blocked tool runs: 0" not in audit.response or "recent approval-held tool runs: 1" not in audit.response:
            raise SystemExit(f"Completion audit should render approval-held-only counts: {audit.response}")
        assert_completion_audit_handoff(audit_metadata, "approval-held-only completion audit")

        mission = runtime.handle("execution mission control: summarize local harness proof state")
        mission_metadata = mission.tool_results[0].metadata
        if mission_metadata.get("mission_state") != "HOLD_FOR_APPROVAL_REVIEW":
            raise SystemExit(f"Execution mission should hold approval-held-only rows for approval review: {mission_metadata}")
        if mission_metadata.get("next_command") != "approval readiness 42":
            raise SystemExit(f"Execution mission should surface held-only approval review: {mission_metadata}")
        if mission_metadata.get("recent_failed_runs") != 0 or mission_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution mission approval-held-only counts drifted: {mission_metadata}")
        if f"execution recovery packet {held_id}" in mission.response or f"execution recovery packet {held_id}" in " ".join(mission_metadata.get("mission_command_queue") or []):
            raise SystemExit(f"Execution mission should not recover approval-held-only row #{held_id}: {mission_metadata}")
        if "recent failed/blocked runs: 0" not in mission.response or "recent approval-held runs: 1" not in mission.response:
            raise SystemExit(f"Execution mission should render approval-held-only counts: {mission.response}")
        assert_execution_mission_handoff(mission_metadata, "approval-held-only execution mission")

        runbook = runtime.handle("execution runbook: summarize local harness proof state")
        runbook_metadata = runbook.tool_results[0].metadata
        if runbook_metadata.get("runbook_state") != "HOLD_APPROVAL_REVIEW":
            raise SystemExit(f"Execution runbook should hold approval-held-only rows for approval review: {runbook_metadata}")
        if runbook_metadata.get("next_step") != "approval readiness 42":
            raise SystemExit(f"Execution runbook should surface held-only approval review: {runbook_metadata}")
        if runbook_metadata.get("recent_failed_runs") != 0 or runbook_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution runbook approval-held-only counts drifted: {runbook_metadata}")
        if f"execution recovery packet {held_id}" in runbook.response or f"execution recovery packet {held_id}" in " ".join(runbook_metadata.get("runbook_proof_queue") or []):
            raise SystemExit(f"Execution runbook should not recover approval-held-only row #{held_id}: {runbook_metadata}")
        if "recent failed/blocked runs: 0" not in runbook.response or "recent approval-held runs: 1" not in runbook.response:
            raise SystemExit(f"Execution runbook should render approval-held-only counts: {runbook.response}")
        assert_execution_runbook_handoff(runbook_metadata, "approval-held-only execution runbook")

        saved_case = runtime.handle("save execution case: summarize local harness proof state")
        if not saved_case.verified:
            raise SystemExit(f"Expected approval-held-only execution case save to run local-safe: {saved_case}")
        saved_case_id = saved_case.tool_results[0].metadata.get("case_id")
        gate = runtime.registry.get("execution_case_gate").handler({"case_id": saved_case_id})
        gate_metadata = gate.metadata
        if gate_metadata.get("recent_failed_runs") != 0 or gate_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution case gate approval-held-only counts drifted: {gate_metadata}")
        if gate_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Execution case gate should review held-only approval rows: {gate_metadata}")
        if f"execution recovery packet {held_id}" in gate.output or f"execution recovery packet {held_id}" in " ".join(gate_metadata.get("next_case_proof_commands") or []):
            raise SystemExit(f"Execution case gate should not recover approval-held-only row #{held_id}: {gate_metadata}")
        if "recent failed/blocked runs: 0" not in gate.output or "recent approval-held runs: 1" not in gate.output:
            raise SystemExit(f"Execution case gate should render approval-held-only counts: {gate.output}")
        assert_execution_case_gate_handoff(gate_metadata, "approval-held-only execution case gate")

        review = runtime.registry.get("execution_case_review_packet").handler({"case_id": saved_case_id})
        review_metadata = review.metadata
        if review_metadata.get("recent_failed_runs") != 0 or review_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution case review approval-held-only counts drifted: {review_metadata}")
        if review_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Execution case review should review held-only approval rows: {review_metadata}")
        if f"execution recovery packet {held_id}" in review.output or f"execution recovery packet {held_id}" in " ".join(review_metadata.get("next_case_proof_commands") or []):
            raise SystemExit(f"Execution case review should not recover approval-held-only row #{held_id}: {review_metadata}")
        if "recent failed/blocked runs: 0" not in review.output or "recent approval-held runs: 1" not in review.output:
            raise SystemExit(f"Execution case review should render approval-held-only counts: {review.output}")
        assert_execution_case_review_handoff(review_metadata, "approval-held-only execution case review")

        closure = runtime.registry.get("execution_case_closure_packet").handler({"case_id": saved_case_id})
        closure_metadata = closure.metadata
        if closure_metadata.get("recent_failed_runs") != 0 or closure_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution case closure approval-held-only counts drifted: {closure_metadata}")
        if closure_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Execution case closure should review held-only approval rows: {closure_metadata}")
        if "approval readiness 42" not in (closure_metadata.get("closure_proof_queue") or []):
            raise SystemExit(f"Execution case closure proof queue should include held-only approval review: {closure_metadata}")
        if f"execution recovery packet {held_id}" in closure.output or f"execution recovery packet {held_id}" in " ".join(closure_metadata.get("closure_proof_queue") or []):
            raise SystemExit(f"Execution case closure should not recover approval-held-only row #{held_id}: {closure_metadata}")
        if "recent failed/blocked runs: 0" not in closure.output or "recent approval-held runs: 1" not in closure.output:
            raise SystemExit(f"Execution case closure should render approval-held-only counts: {closure.output}")
        assert_execution_case_closure_handoff(closure_metadata, "approval-held-only execution case closure")

        timeline = runtime.registry.get("execution_case_timeline").handler({"case_id": saved_case_id})
        timeline_metadata = timeline.metadata
        if timeline_metadata.get("recent_failed_runs") != 0 or timeline_metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"Execution case timeline approval-held-only counts drifted: {timeline_metadata}")
        if timeline_metadata.get("approval_held_review_command") != "approval readiness 42":
            raise SystemExit(f"Execution case timeline should review held-only approval rows: {timeline_metadata}")
        if f"execution recovery packet {held_id}" in timeline.output:
            raise SystemExit(f"Execution case timeline should not recover approval-held-only row #{held_id}: {timeline.output}")
        if "recent failed/blocked runs: 0" not in timeline.output or "recent approval-held runs: 1" not in timeline.output:
            raise SystemExit(f"Execution case timeline should render approval-held-only counts: {timeline.output}")
        if "approval-held" not in timeline.output:
            raise SystemExit(f"Execution case timeline should label approval-held-only runtime rows: {timeline.output}")
        assert_execution_case_timeline_handoff(timeline_metadata, "approval-held-only execution case timeline")


def main() -> None:
    assert_harness_metadata_bool_is_exact()
    assert_harness_packet_aliases_use_default_request()
    assert_natural_approval_questions_route_to_rehearsal()
    assert_natural_risk_preflight_routes_before_broad_tools()
    assert_natural_action_readiness_and_governor_routes()
    assert_natural_verification_packet_routes()
    assert_natural_acceptance_gate_routes()
    assert_natural_completion_claim_routes()
    assert_natural_harness_completion_assessment_routes()
    assert_natural_completion_next_proof_routes()
    assert_natural_readiness_report_routes()
    assert_natural_harness_readiness_digest_routes()
    assert_natural_execution_prep_packet_routes()
    assert_natural_harness_control_packet_routes()
    assert_natural_execution_case_packet_routes()
    assert_natural_route_diagnosis_stays_read_only()
    assert_harness_latest_case_readiness_uses_exact_bools()
    assert_execution_proof_aliases_require_explicit_counts()
    assert_completion_proof_queue_uses_explicit_proof_sources()
    assert_completion_actionable_queue_preserves_storage_recheck()
    assert_completion_proof_surfaces_tolerate_malformed_rows()
    assert_harness_control_and_operations_tolerate_malformed_rows()
    assert_execution_proof_bundle_tolerates_malformed_recent_rows()
    assert_harness_operator_surfaces_separate_approval_held_runs()
    assert_agi_next_build_redacts_path_shaped_target_config()
    assert_agi_target_config_malformed_entries_fail_closed()
    for malformed_counter in ("not-a-number", True, float("inf"), None):
        if _metadata_int(malformed_counter, 7) != 7:
            raise SystemExit(f"harness metadata counters should default malformed values safely: {malformed_counter!r}")
    if _metadata_int("5") != 5:
        raise SystemExit("harness metadata counters should preserve valid numeric strings.")
    for truthy_readiness in ("true", "false", "yes", 1, [], ["ready"], None):
        if _metadata_bool(truthy_readiness):
            raise SystemExit(f"harness readiness booleans should reject non-bool truthy values: {truthy_readiness!r}")
    if _metadata_bool(True) is not True or _metadata_bool(False) is not False:
        raise SystemExit("harness readiness booleans should preserve exact bool values.")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("harness readiness booleans should use the explicit default only for non-bool values.")

    with TemporaryDirectory(prefix="jarvis-harness-diagnostic-action-") as temp:
        runtime = make_temp_runtime(Path(temp))
        diagnostic_run_ids = [
            runtime.store.log_tool_run(
                session_id="harness-diagnostic-action",
                tool_name="storage_status",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis storage status: ready",
                metadata={
                    "storage_metadata_only": True,
                    "storage_ready_for_completion_claim": True,
                },
            ),
            runtime.store.log_tool_run(
                session_id="harness-diagnostic-action",
                tool_name="storage_recovery_check",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis storage recovery check: blocked by fallback",
                metadata={
                    "storage_metadata_only": True,
                    "storage_recovery_required": True,
                    "storage_recovery_check_passed": False,
                },
            ),
            runtime.store.log_tool_run(
                session_id="harness-diagnostic-action",
                tool_name="readiness_report",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis readiness report: ready with attention",
                metadata={
                    "readiness_gate_state": "REVIEW_PENDING_APPROVALS",
                    "storage_ready_for_completion_claim": False,
                },
            ),
            runtime.store.log_tool_run(
                session_id="harness-diagnostic-action",
                tool_name="completion_claim_gate",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis completion claim gate: CLAIM_BLOCKED",
                metadata={
                    "verdict": "CLAIM_BLOCKED",
                    "completion_claim_ready": False,
                },
            ),
        ]
        learning_debt = _execution_learning_debt_snapshot(runtime.store.recent_tool_runs(limit=20))
        if learning_debt["state"] != "NO_RECENT_ACTION_RUNS":
            raise SystemExit(f"Harness learning debt should ignore successful diagnostic readiness/storage/completion runs: {learning_debt}")
        if learning_debt["blocks_completion_claim"] is not False:
            raise SystemExit(f"Harness learning debt should not block on successful diagnostic readiness/storage/completion runs: {learning_debt}")
        if learning_debt["recent_action_runs"] != 0:
            raise SystemExit(f"Harness learning debt should not count diagnostic readiness/storage/completion runs as action runs: {learning_debt}")
        if learning_debt["target_run_id"] is not None or learning_debt["target_tool_name"]:
            raise SystemExit(f"Harness learning debt should not target diagnostic runs {diagnostic_run_ids}: {learning_debt}")

    with TemporaryDirectory(prefix="jarvis-harness-meta-target-") as temp:
        runtime = make_temp_runtime(Path(temp))
        action_run_id = runtime.store.log_tool_run(
            session_id="harness-meta-target",
            tool_name="current_time",
            risk="READ_ONLY",
            ok=True,
            approved=False,
            output="Monday, June 08, 2026 at 06:21 PM",
            metadata={"route": "tools"},
        )
        runtime.store.log_tool_run(
            session_id="harness-meta-target",
            tool_name="completion_next_proof_packet",
            risk="READ_ONLY",
            ok=True,
            approved=False,
            output="Jarvis completion next proof packet",
            metadata={"next_proof_command": f"execution learning closure {action_run_id}"},
        )
        runtime.store.log_tool_run(
            session_id="harness-meta-target",
            tool_name="harness_readiness_digest",
            risk="READ_ONLY",
            ok=True,
            approved=False,
            output="Jarvis harness readiness digest",
            metadata={"source_tool": "completion_next_proof_packet"},
        )
        runtime.store.log_tool_run(
            session_id="harness-meta-target",
            tool_name="specialist_handoff_quality_gate",
            risk="READ_ONLY",
            ok=True,
            approved=False,
            output="Jarvis specialist handoff quality gate",
            metadata={"quality_gate_state": "HANDOFF_QUALITY_READY_FOR_PROPOSAL_GATE"},
        )
        runtime.store.log_tool_run(
            session_id="harness-meta-target",
            tool_name="specialist_proposal_gate",
            risk="READ_ONLY",
            ok=True,
            approved=False,
            output="Jarvis specialist proposal gate",
            metadata={"proposal_gate_state": "PROPOSAL_FALLBACK_NO_MODEL"},
        )
        runtime.store.log_tool_run(
            session_id="harness-meta-target",
            tool_name="continuation_packet",
            risk="READ_ONLY",
            ok=True,
            approved=False,
            output="Jarvis continuation packet with checkpoint recovery contract",
            metadata={"checkpoint_freshness": "missing"},
        )
        runtime.store.log_tool_run(
            session_id="harness-meta-target",
            tool_name="checkpoint_recovery_preview",
            risk="READ_ONLY",
            ok=True,
            approved=False,
            output="Jarvis checkpoint recovery preview",
            metadata={"checkpoint_freshness": "missing"},
        )
        learning_debt = _execution_learning_debt_snapshot(runtime.store.recent_tool_runs(limit=20))
        if learning_debt["target_run_id"] != action_run_id:
            raise SystemExit(f"Harness learning debt should ignore completion/readiness/specialist proof packets as targets: {learning_debt}")
        if learning_debt["target_tool_name"] != "current_time":
            raise SystemExit(f"Harness learning debt selected the wrong non-meta target: {learning_debt}")
        if learning_debt["recent_action_runs"] != 1:
            raise SystemExit(f"Harness learning debt should count only the real action row: {learning_debt}")
        if learning_debt["next_required_command"] != f"after-action learning packet {action_run_id}":
            raise SystemExit(f"Harness learning debt missed the evidence-first next command: {learning_debt}")
        if learning_debt["execution_learning_closure_command"] != f"execution learning closure {action_run_id}":
            raise SystemExit(f"Harness learning debt missed the stable closure command: {learning_debt}")
        actionable_commands = learning_debt.get("actionable_required_commands") or []
        if actionable_commands[:2] != [f"after-action learning packet {action_run_id}", f"execution learning closure {action_run_id}"]:
            raise SystemExit(f"Harness learning debt missed actionable evidence-first queue: {learning_debt}")
        if learning_debt.get("next_evidence_command") != f"after-action learning packet {action_run_id}":
            raise SystemExit(f"Harness learning debt missed actionable next evidence command: {learning_debt}")
        failed_action_run_id = runtime.store.log_tool_run(
            session_id="harness-meta-target",
            tool_name="checkpoint_recovery_execute",
            risk="READ_ONLY",
            ok=False,
            approved=False,
            output="Held for reviewed local-safe continuation evidence.",
            metadata={"route": "harness", "stop_condition": "reviewed_local_safe_step_required"},
        )
        runtime.store.log_tool_run(
            session_id="harness-meta-target",
            tool_name="execution_learning_closure_packet",
            risk="READ_ONLY",
            ok=True,
            approved=False,
            output="Learning closure ready.",
            metadata={
                "target_run_id": failed_action_run_id,
                "run_id": failed_action_run_id,
                "target_tool_name": "checkpoint_recovery_execute",
                "verdict": "LEARNING_CLOSURE_READY",
                "learning_closure_state": "LEARNING_CLOSURE_READY",
                "learning_closure_ready": True,
                "execution_learning_closure_handoff": {
                    "verdict": "LEARNING_CLOSURE_READY",
                    "learning_closure_ready": True,
                    "target": {"run_id": failed_action_run_id, "tool_name": "checkpoint_recovery_execute"},
                },
            },
        )
        learning_debt = _execution_learning_debt_snapshot(runtime.store.recent_tool_runs(limit=20))
        if learning_debt["target_run_id"] != failed_action_run_id:
            raise SystemExit(f"Harness learning debt should keep failed action as target after ready closure: {learning_debt}")
        if learning_debt["blocks_completion_claim"]:
            raise SystemExit(f"Harness learning debt should accept ready closure for failed target: {learning_debt}")
        if "failure_review" in learning_debt["missing"] or learning_debt["required_commands"]:
            raise SystemExit(f"Harness learning debt should not keep failure review proof after ready closure: {learning_debt}")
        if learning_debt["target_execution_learning_closure_packets"] != 1:
            raise SystemExit(f"Harness learning debt missed target closure packet count: {learning_debt}")
        if not learning_debt["target_failure_review_satisfied"]:
            raise SystemExit(f"Harness learning debt missed target failure-review satisfaction: {learning_debt}")

    with TemporaryDirectory(prefix="jarvis-harness-") as temp:
        runtime = make_temp_runtime(Path(temp))
        setup_cases = [
            "add task verify Jarvis harness direction priority high",
            "create goal Build Jarvis as an agent harness because AGI needs a reliable operating layer",
            "run command python3 --version",
        ]
        for case in setup_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1000])
            print()

        planner = RuleBasedPlanner()
        for case in ["priority goal please", "show priority goal", "show me priority goal"]:
            plan = planner.plan(case)
            if [(a.tool_name, a.args) for a in plan.actions] != [("priority_goal", {})]:
                raise SystemExit(f"planner missed priority-goal phone alias {case!r}: {plan.actions}")

        priority_cases = ["priority goal", "remember the goal"]
        for case in priority_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis priority goal",
                "Finish Jarvis V2 as an AI agent harness",
                "top build priority",
                "The model is the engine",
                "Priority build order",
                "does not override safety boundaries",
                "does not override the operator's explicit stop times",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Priority goal missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("priority") != "finish_jarvis_agent_harness":
                raise SystemExit(f"Priority goal missed metadata: {metadata}")
            if metadata.get("priority_build_order_count") != len(metadata.get("priority_build_order") or []):
                raise SystemExit(f"Priority goal missed build-order parity: {metadata}")
            if metadata.get("active_matching_goal_row_count") != len(metadata.get("active_matching_goal_rows") or []):
                raise SystemExit(f"Priority goal missed active-goal row parity: {metadata}")
            if metadata.get("priority_memory_row_count") != len(metadata.get("priority_memory_rows") or []):
                raise SystemExit(f"Priority goal missed memory row parity: {metadata}")
            if metadata.get("approval_gates_overridden") is not False or metadata.get("approval_granted") is not False:
                raise SystemExit(f"Priority goal should not override or grant approval: {metadata}")
            if metadata.get("draft_only") is not True or metadata.get("requires_manual_send") is not True or metadata.get("loads_without_execution") is not True:
                raise SystemExit(f"Priority goal missed review-only contract: {metadata}")
            if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                raise SystemExit(f"Priority goal missed stop-time override metadata: {metadata}")
            assert_priority_goal_handoff(metadata, "Priority goal")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Priority goal unsafe metadata {key}: {metadata}")

        cases = ["harness status", "agent harness", "jarvis agi direction"]
        for case in cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis agent harness status",
                "The AI model is the engine",
                "steering wheel",
                "AGI direction",
                "Harness components",
                "integration_enablement_gate",
                "integration_rehearsal_receipt",
                "integration_metadata_preview",
                "safety and approvals",
                "approval_readiness_packet",
                "execution_health_report",
                "autonomy loop",
                "learning loop",
                "Typed AGI harness contract",
                "Intelligence",
                "Engine",
                "Agents",
                "Tools+Memory",
                "Learning",
                "Model drafts do not authorize execution",
                "Internal workers are capacity, not companion personas",
                "AGI-direction gates still ahead",
                "approval-gated",
                "acceptance evidence",
                "Priority goals do not override the operator's explicit stop times",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Harness status missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("components") != 8 or metadata.get("tools", 0) <= 0:
                raise SystemExit(f"Harness status missed component/tool counts: {metadata}")
            component_rows = metadata.get("component_rows") or []
            if metadata.get("component_row_count") != len(component_rows) or metadata.get("components") != len(component_rows):
                raise SystemExit(f"Harness status missed component row parity: {metadata}")
            if metadata.get("strong_components") != len([row for row in component_rows if row.get("status") == "strong foundation"]):
                raise SystemExit(f"Harness status missed strong component parity: {metadata}")
            if metadata.get("partial_components") != len([row for row in component_rows if row.get("status") == "partial foundation"]):
                raise SystemExit(f"Harness status missed partial component parity: {metadata}")
            if metadata.get("missing_components") != len([row for row in component_rows if row.get("status") == "missing"]):
                raise SystemExit(f"Harness status missed missing component parity: {metadata}")
            if metadata.get("toolset_count_row_count") != len(metadata.get("toolset_count_rows") or []):
                raise SystemExit(f"Harness status missed toolset row parity: {metadata}")
            if metadata.get("risk_count_row_count") != len(metadata.get("risk_count_rows") or []):
                raise SystemExit(f"Harness status missed risk row parity: {metadata}")
            if sum(row.get("count") or 0 for row in metadata.get("risk_count_rows") or []) != metadata.get("tools"):
                raise SystemExit(f"Harness status risk rows should cover all tools: {metadata}")
            if metadata.get("best_next_command_count") != len(metadata.get("best_next_commands") or []):
                raise SystemExit(f"Harness status missed best-next command parity: {metadata}")
            assert_harness_layer_contract(metadata, "Harness status")
            if metadata.get("draft_only") is not True or metadata.get("requires_manual_send") is not True or metadata.get("loads_without_execution") is not True:
                raise SystemExit(f"Harness status missed review-only contract: {metadata}")
            if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(f"Harness status should not authorize execution or completion claims: {metadata}")
            if metadata.get("approval_granted") is not False:
                raise SystemExit(f"Harness status should not grant approval: {metadata}")
            if metadata.get("pending_approvals", 0) < 1:
                raise SystemExit(f"Harness status should report the queued approval from setup: {metadata}")
            if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                raise SystemExit(f"Harness status missed stop-time override metadata: {metadata}")
            assert_harness_status_handoff(metadata, "Harness status")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Harness status unsafe metadata {key}: {metadata}")

        execution_audit = runtime.handle("execution audit gate")
        print(f"[{'ok' if execution_audit.verified else 'blocked'}] execution audit gate")
        print(execution_audit.response[:2200])
        print()
        if not execution_audit.verified:
            raise SystemExit("Expected execution audit gate to run read-only.")
        for expected in [
            "Jarvis execution audit gate",
            "run-integrity stop-check",
            "Verdict:",
            "Inspected runs:",
            "Failed or blocked runs:",
            "Risky invoked runs without approval evidence:",
            "Problem runs:",
            "Required recovery path:",
            "execution recovery packet",
            "does not call models",
        ]:
            if expected not in execution_audit.response:
                raise SystemExit(f"Execution audit gate missing expected text: {expected}")
        audit_metadata = execution_audit.tool_results[0].metadata
        if audit_metadata.get("inspected_runs", 0) < 1 or audit_metadata.get("failed_or_blocked_runs", 0) < 1:
            raise SystemExit(f"Execution audit gate should see setup blocked run: {audit_metadata}")
        if audit_metadata.get("verdict") not in {"RECOVERY_REVIEW_REQUIRED", "RISKY_SUCCESS_WITHOUT_APPROVAL_EVIDENCE", "APPROVAL_LINK_MISSING", "AUDIT_GATE_CLEAR"}:
            raise SystemExit(f"Execution audit gate reported unexpected verdict: {audit_metadata}")
        for key in READ_ONLY_FLAGS:
            if audit_metadata.get(key) is not False:
                raise SystemExit(f"Execution audit gate unsafe metadata {key}: {audit_metadata}")

        readiness_matrix = runtime.handle("execution readiness matrix: use my computer to inspect the screen run a python script write a summary file and email me the result")
        print(f"[{'ok' if readiness_matrix.verified else 'blocked'}] execution readiness matrix")
        print(readiness_matrix.response[:2400])
        print()
        if not readiness_matrix.verified:
            raise SystemExit("Expected execution readiness matrix to run read-only.")
        for expected in [
            "Jarvis execution readiness matrix",
            "harness dashboard before action",
            "Verdict: HOLD_FOR_APPROVAL_REVIEW",
            "Readiness matrix:",
            "Approval:",
            "Planned route:",
            "Proof requirements:",
            "Stop conditions:",
            "does not call a model",
        ]:
            if expected not in readiness_matrix.response:
                raise SystemExit(f"Execution readiness matrix missing expected text: {expected}")
        matrix_metadata = readiness_matrix.tool_results[0].metadata
        if matrix_metadata.get("verdict") != "HOLD_FOR_APPROVAL_REVIEW":
            raise SystemExit(f"Execution readiness matrix should stop for setup approval review: {matrix_metadata}")
        if matrix_metadata.get("approval_required") is not True or "computer" not in matrix_metadata.get("matched_risks", []):
            raise SystemExit(f"Execution readiness matrix missed approval/risk metadata: {matrix_metadata}")
        if matrix_metadata.get("matrix_rows") != 8 or matrix_metadata.get("proof_requirements", 0) < 5:
            raise SystemExit(f"Execution readiness matrix missed proof matrix metadata: {matrix_metadata}")
        for key in READ_ONLY_FLAGS:
            if matrix_metadata.get(key) is not False:
                raise SystemExit(f"Execution readiness matrix unsafe metadata {key}: {matrix_metadata}")

        dispatch_result = runtime.handle("dispatch decision: use my computer to inspect the screen run a python script write a summary file and email me the result")
        print(f"[{'ok' if dispatch_result.verified else 'blocked'}] dispatch decision")
        print(dispatch_result.response[:2400])
        print()
        if not dispatch_result.verified:
            raise SystemExit("Expected dispatch decision packet to run read-only.")
        for expected in [
            "Jarvis dispatch decision packet",
            "final read-only go/no-go packet",
            "Dispatch decision: HOLD_REVIEW_PENDING_APPROVALS",
            "Can auto-run now: no",
            "Preflight commands:",
            "Proof contract after execution:",
            "acceptance gate:",
            "approval chain proof",
            "execution acceptance gate is run",
            "does not call a model",
        ]:
            if expected not in dispatch_result.response:
                raise SystemExit(f"Dispatch decision packet missing expected text: {expected}")
        dispatch_metadata = dispatch_result.tool_results[0].metadata
        if dispatch_metadata.get("decision") != "HOLD_REVIEW_PENDING_APPROVALS" or dispatch_metadata.get("can_auto_run") is not False:
            raise SystemExit(f"Dispatch decision should hold for pending approval review: {dispatch_metadata}")
        if dispatch_metadata.get("approval_required") is not True or "computer" not in dispatch_metadata.get("matched_risks", []):
            raise SystemExit(f"Dispatch decision missed approval/risk metadata: {dispatch_metadata}")
        if dispatch_metadata.get("proof_contract", 0) < 5 or dispatch_metadata.get("preflight_commands", 0) < 6:
            raise SystemExit(f"Dispatch decision missed proof/preflight metadata: {dispatch_metadata}")
        for key in READ_ONLY_FLAGS:
            if dispatch_metadata.get(key) is not False:
                raise SystemExit(f"Dispatch decision unsafe metadata {key}: {dispatch_metadata}")

        natural_dispatch = runtime.handle("use my computer to inspect the screen run a python script write a summary file and email me the result")
        print(f"[{'ok' if natural_dispatch.verified else 'blocked'}] natural risky dispatch")
        print(natural_dispatch.response[:2400])
        print()
        if not natural_dispatch.verified:
            raise SystemExit("Expected natural risky order to dispatch through read-only tooling.")
        if natural_dispatch.tool_results[0].tool_name != "dispatch_decision_packet":
            raise SystemExit("Natural risky order should route to dispatch_decision_packet.")
        natural_metadata = natural_dispatch.tool_results[0].metadata
        if natural_metadata.get("decision") != "HOLD_REVIEW_PENDING_APPROVALS":
            raise SystemExit(f"Natural risky order should hold while approval is pending: {natural_metadata}")
        if natural_metadata.get("can_auto_run") is not False or natural_metadata.get("approval_required") is not True:
            raise SystemExit(f"Natural risky order missed safe dispatch metadata: {natural_metadata}")
        if "computer" not in natural_metadata.get("matched_risks", []) or "shell/code" not in natural_metadata.get("matched_risks", []):
            raise SystemExit(f"Natural risky order missed risk signals: {natural_metadata}")
        for key in READ_ONLY_FLAGS:
            if natural_metadata.get(key) is not False:
                raise SystemExit(f"Natural risky dispatch unsafe metadata {key}: {natural_metadata}")

        argument_contract = runtime.handle("argument contract: run command python3 --version")
        print(f"[{'ok' if argument_contract.verified else 'blocked'}] argument contract exact command")
        print(argument_contract.response[:2200])
        print()
        if not argument_contract.verified:
            raise SystemExit("Argument contract should run read-only.")
        for expected in [
            "Jarvis argument contract packet",
            "EXACT_ARGUMENTS_READY",
            "run_shell_command",
            "Provided args: command='python3 --version'",
            "approval required",
            "Argument proof rules",
            "does not call a model",
        ]:
            if expected not in argument_contract.response:
                raise SystemExit(f"Argument contract missing expected text: {expected}")
        contract_metadata = argument_contract.tool_results[0].metadata
        if contract_metadata.get("planned_action_count") != 1 or contract_metadata.get("missing_argument_count") != 0:
            raise SystemExit(f"Argument contract missed exact planned args: {contract_metadata}")
        if contract_metadata.get("approval_required") is not True:
            raise SystemExit(f"Argument contract should preserve approval requirement: {contract_metadata}")
        for key in READ_ONLY_FLAGS:
            if contract_metadata.get(key) is not False:
                raise SystemExit(f"Argument contract unsafe metadata {key}: {contract_metadata}")

        natural_verification = runtime.handle("how would Jarvis verify: run command python3 --version")
        print(f"[{'ok' if natural_verification.verified else 'blocked'}] natural verification packet")
        print(natural_verification.response[:2200])
        print()
        if not natural_verification.verified:
            raise SystemExit("Natural verification packet should run read-only.")
        if natural_verification.tool_results[0].tool_name != "verification_packet":
            raise SystemExit("Natural verification question should route to verification_packet.")
        for expected in [
            "Jarvis verification packet",
            "Evidence requirements",
            "Failure signals",
            "Recovery if verification fails",
            "does not run the order",
        ]:
            if expected not in natural_verification.response:
                raise SystemExit(f"Natural verification packet missing expected text: {expected}")
        verification_metadata = natural_verification.tool_results[0].metadata
        if verification_metadata.get("approval_required") is not True or not str(verification_metadata.get("next_command", "")).strip():
            raise SystemExit(f"Natural verification packet missed approval/next-command metadata: {verification_metadata}")
        for key in READ_ONLY_FLAGS:
            if verification_metadata.get(key) is not False:
                raise SystemExit(f"Natural verification packet unsafe metadata {key}: {verification_metadata}")

        acceptance_gate = runtime.handle("acceptance gate: run command python3 --version; evidence approval packet viewed and recent tool run ok; tests smoke_test_harness passed; recovery stop on non-zero exit")
        print(f"[{'ok' if acceptance_gate.verified else 'blocked'}] execution acceptance gate")
        print(acceptance_gate.response[:2400])
        print()
        if not acceptance_gate.verified:
            raise SystemExit("Execution acceptance gate should run read-only.")
        for expected in [
            "Jarvis execution acceptance gate",
            "final read-only gate",
            "Verdict:",
            "Route summary:",
            "Acceptance receipts:",
            "Supplied proof:",
            "Blocking reasons:",
            "does not call a model",
        ]:
            if expected not in acceptance_gate.response:
                raise SystemExit(f"Execution acceptance gate missing expected text: {expected}")
        acceptance_metadata = acceptance_gate.tool_results[0].metadata
        if acceptance_metadata.get("approval_required") is not True or acceptance_metadata.get("has_evidence") is not True or acceptance_metadata.get("has_tests") is not True or acceptance_metadata.get("has_recovery") is not True:
            raise SystemExit(f"Execution acceptance gate missed proof metadata: {acceptance_metadata}")
        if acceptance_metadata.get("verdict") not in {"NOT_ACCEPTED", "READY_FOR_HUMAN_ACCEPTANCE_REVIEW"}:
            raise SystemExit(f"Execution acceptance gate reported unexpected verdict: {acceptance_metadata}")
        for key in READ_ONLY_FLAGS:
            if acceptance_metadata.get(key) is not False:
                raise SystemExit(f"Execution acceptance gate unsafe metadata {key}: {acceptance_metadata}")

        natural_acceptance = runtime.handle("can we call this done: run command python3 --version; evidence approval packet viewed and recent tool run ok; tests smoke_test_harness passed; recovery stop on non-zero exit")
        print(f"[{'ok' if natural_acceptance.verified else 'blocked'}] natural acceptance gate")
        print(natural_acceptance.response[:2200])
        print()
        if not natural_acceptance.verified:
            raise SystemExit("Natural acceptance gate should run read-only.")
        if natural_acceptance.tool_results[0].tool_name != "execution_acceptance_gate":
            raise SystemExit("Natural acceptance question should route to execution_acceptance_gate.")
        natural_acceptance_metadata = natural_acceptance.tool_results[0].metadata
        if natural_acceptance_metadata.get("approval_required") is not True or natural_acceptance_metadata.get("has_evidence") is not True or natural_acceptance_metadata.get("has_tests") is not True or natural_acceptance_metadata.get("has_recovery") is not True:
            raise SystemExit(f"Natural acceptance gate missed proof metadata: {natural_acceptance_metadata}")
        if natural_acceptance_metadata.get("verdict") not in {"NOT_ACCEPTED", "READY_FOR_HUMAN_ACCEPTANCE_REVIEW"}:
            raise SystemExit(f"Natural acceptance gate reported unexpected verdict: {natural_acceptance_metadata}")
        for key in READ_ONLY_FLAGS:
            if natural_acceptance_metadata.get(key) is not False:
                raise SystemExit(f"Natural acceptance gate unsafe metadata {key}: {natural_acceptance_metadata}")

        gap_argument_contract = runtime.handle("argument contract: use my computer to inspect the screen run a python script write a summary file and email me the result")
        print(f"[{'ok' if gap_argument_contract.verified else 'blocked'}] argument contract planner gap")
        print(gap_argument_contract.response[:2200])
        print()
        if not gap_argument_contract.verified:
            raise SystemExit("Planner-gap argument contract should run read-only.")
        for expected in [
            "Jarvis argument contract packet",
            "PLANNER_GAP",
            "no exact tool arguments",
            "planner gap",
            "computer",
            "shell/code",
        ]:
            if expected not in gap_argument_contract.response:
                raise SystemExit(f"Planner-gap argument contract missing expected text: {expected}")
        gap_contract_metadata = gap_argument_contract.tool_results[0].metadata
        if gap_contract_metadata.get("verdict") != "PLANNER_GAP" or gap_contract_metadata.get("planned_action_count") != 0:
            raise SystemExit(f"Planner-gap argument contract missed metadata: {gap_contract_metadata}")
        for key in READ_ONLY_FLAGS:
            if gap_contract_metadata.get(key) is not False:
                raise SystemExit(f"Planner-gap argument contract unsafe metadata {key}: {gap_contract_metadata}")

        planner_gap = runtime.handle("planner gap: use my computer to inspect the screen run a python script write a summary file and email me the result")
        print(f"[{'ok' if planner_gap.verified else 'blocked'}] planner gap")
        print(planner_gap.response[:2400])
        print()
        if not planner_gap.verified:
            raise SystemExit("Expected planner gap packet to run read-only.")
        for expected in [
            "Jarvis planner gap packet",
            "steering layer",
            "Classification: PLANNER_GAP",
            "execution governor:",
            "risky wording but no exact tool route",
            "Suggested follow-up commands:",
            "does not call a model",
        ]:
            if expected not in planner_gap.response:
                raise SystemExit(f"Planner gap packet missing expected text: {expected}")
        gap_metadata = planner_gap.tool_results[0].metadata
        if gap_metadata.get("classification") != "PLANNER_GAP":
            raise SystemExit(f"Planner gap should detect a missing exact route: {gap_metadata}")
        if gap_metadata.get("approval_required") is not True or "computer" not in gap_metadata.get("matched_risks", []):
            raise SystemExit(f"Planner gap missed approval/risk metadata: {gap_metadata}")
        if gap_metadata.get("planned_action_count") != 0 or gap_metadata.get("gap_signal_count", 0) < 1:
            raise SystemExit(f"Planner gap missed gap metadata: {gap_metadata}")
        if not str(gap_metadata.get("next_command", "")).startswith("execution governor: "):
            raise SystemExit(f"Planner gap should route back through the execution governor: {gap_metadata}")
        for key in READ_ONLY_FLAGS:
            if gap_metadata.get(key) is not False:
                raise SystemExit(f"Planner gap unsafe metadata {key}: {gap_metadata}")

        missing_cycle = runtime.registry.get("harness_cycle_preview").handler({})
        print(f"[{'ok' if missing_cycle.ok else 'blocked'}] harness cycle missing request")
        print(missing_cycle.output[:800])
        print()
        if missing_cycle.ok:
            raise SystemExit("Expected harness cycle missing-request path to refuse safely.")
        missing_cycle_metadata = missing_cycle.metadata
        assert_no_request_harness_cycle_handoff(missing_cycle_metadata, "Harness cycle missing request")

        cycle_cases = [
            "harness cycle: organize my desktop and summarize what changed",
            "agent harness preview: explain Jarvis memory",
        ]
        for case in cycle_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis harness cycle preview",
                "the model as the engine",
                "steering",
                "Likely route",
                "Cycle:",
                "execution governor:",
                "perceive",
                "route",
                "gate",
                "verify",
                "learn",
                "Execution boundary",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Harness cycle preview missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("cycle_stages") != 8:
                raise SystemExit(f"Harness cycle preview missed stage count: {metadata}")
            if metadata.get("primary_preview") != "execution governor":
                raise SystemExit(f"Harness cycle preview should start with the execution governor: {metadata}")
            assert_harness_cycle_handoff(metadata, "Harness cycle preview")
            if (
                metadata.get("draft_only") is not True
                or metadata.get("requires_manual_send") is not True
                or metadata.get("loads_without_execution") is not True
            ):
                raise SystemExit(f"Harness cycle preview missed review-only contract: {metadata}")
            if (
                metadata.get("authorizes_execution") is not False
                or metadata.get("authorizes_completion_claim") is not False
                or metadata.get("approval_granted") is not False
            ):
                raise SystemExit(f"Harness cycle preview should not grant authority: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Harness cycle preview unsafe metadata {key}: {metadata}")
            if case.startswith("harness cycle") and metadata.get("approval_required") is not True:
                raise SystemExit(f"Harness cycle preview should detect risky desktop wording: {metadata}")

        missing_lifecycle = runtime.registry.get("harness_lifecycle_state").handler({})
        print(f"[{'ok' if missing_lifecycle.ok else 'blocked'}] harness lifecycle missing request")
        print(missing_lifecycle.output[:800])
        print()
        if missing_lifecycle.ok:
            raise SystemExit("Expected harness lifecycle missing-request path to refuse safely.")
        missing_lifecycle_metadata = missing_lifecycle.metadata
        assert_no_request_harness_lifecycle_handoff(missing_lifecycle_metadata, "Harness lifecycle missing request")

        lifecycle_cases = [
            "harness lifecycle: inspect my screen and summarize what changed",
            "lifecycle state: explain Jarvis memory",
        ]
        for case in lifecycle_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis harness lifecycle state",
                "Route snapshot",
                "Lifecycle board",
                "perceive",
                "ground",
                "route",
                "plan",
                "gate",
                "act",
                "verify",
                "learn",
                "Immediate safe next commands",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Harness lifecycle state missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("lifecycle_stages") != 8:
                raise SystemExit(f"Harness lifecycle state missed stage count: {metadata}")
            if not str(metadata.get("likely_route", "")).strip():
                raise SystemExit(f"Harness lifecycle state missed route metadata: {metadata}")
            assert_harness_lifecycle_handoff(metadata, "Harness lifecycle state")
            if (
                metadata.get("draft_only") is not True
                or metadata.get("requires_manual_send") is not True
                or metadata.get("loads_without_execution") is not True
            ):
                raise SystemExit(f"Harness lifecycle state missed review-only contract: {metadata}")
            if (
                metadata.get("authorizes_execution") is not False
                or metadata.get("authorizes_completion_claim") is not False
                or metadata.get("approval_granted") is not False
            ):
                raise SystemExit(f"Harness lifecycle state should not grant authority: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Harness lifecycle state unsafe metadata {key}: {metadata}")
            if case.startswith("harness lifecycle") and "computer control" not in metadata.get("risk_signals", []):
                raise SystemExit(f"Harness lifecycle state should detect screen/computer wording: {metadata}")

        missing_control = runtime.registry.get("harness_control_surface").handler({})
        print(f"[{'ok' if missing_control.ok else 'blocked'}] harness control missing request")
        print(missing_control.output[:800])
        print()
        if missing_control.ok:
            raise SystemExit("Expected harness control missing-request path to refuse safely.")
        missing_control_metadata = missing_control.metadata
        assert_no_request_harness_control_handoff(missing_control_metadata, "Harness control missing request")

        control_surface_cases = [
            "harness control: use my computer to run a python script and email me the result",
            "control surface: explain Jarvis memory",
        ]
        for case in control_surface_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis harness control surface",
                "car dashboard",
                "engine",
                "steering",
                "pedals",
                "brakes",
                "dashboard",
                "Runtime blockers",
                "Required follow-up packets",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Harness control surface missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("control_groups") != 5 or metadata.get("ready_groups", 0) < 4:
                raise SystemExit(f"Harness control surface missed control group metadata: {metadata}")
            if not str(metadata.get("verdict", "")).strip() or not str(metadata.get("next_command", "")).strip():
                raise SystemExit(f"Harness control surface missed verdict/next command metadata: {metadata}")
            assert_harness_control_handoff(metadata, "Harness control surface")
            if (
                metadata.get("draft_only") is not True
                or metadata.get("requires_manual_send") is not True
                or metadata.get("loads_without_execution") is not True
            ):
                raise SystemExit(f"Harness control surface missed review-only contract: {metadata}")
            if (
                metadata.get("authorizes_execution") is not False
                or metadata.get("authorizes_completion_claim") is not False
                or metadata.get("approval_granted") is not False
            ):
                raise SystemExit(f"Harness control surface should not grant authority: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Harness control surface unsafe metadata {key}: {metadata}")
            if case.startswith("harness control"):
                if metadata.get("verdict") != "HOLD_FOR_APPROVAL_REVIEW" or metadata.get("can_auto_run") is not False:
                    raise SystemExit(f"Harness control surface should hold for pending approval review: {metadata}")
                if "computer control" not in metadata.get("risk_signals", []) or "shell/code" not in metadata.get("risk_signals", []):
                    raise SystemExit(f"Harness control surface missed risky command signals: {metadata}")
            else:
                if metadata.get("verdict") == "RECOVERY_REVIEW_FIRST":
                    if not str(metadata.get("next_command", "")).startswith("execution recovery packet "):
                        raise SystemExit(f"Harness control should keep recovery review as the active brake: {metadata}")
                    if "`execution governor: explain Jarvis memory`" not in result.response:
                        raise SystemExit("Harness control recovery view should still show governor-first follow-up routing.")
                else:
                    if metadata.get("verdict") != "READY_FOR_EXECUTION_GOVERNOR" or metadata.get("can_auto_run") is not True:
                        raise SystemExit(f"Low-risk harness control surface should enter through execution governor: {metadata}")
                    if not str(metadata.get("next_command", "")).startswith("execution governor: "):
                        raise SystemExit(f"Low-risk harness control surface missed governor-first next command: {metadata}")

        operations_cases = [
            "harness operations",
            "operator brief: keep building Jarvis as a safe agent harness",
        ]
        for case in operations_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis harness operations brief",
                "operator-layer packet",
                "Current command state",
                "Next safe move",
                "Harness controls available",
                "execution_health_report",
                "execution health report",
                "integration_execution_matrix",
                "integration execution matrix",
                "integration_adapter_manifest",
                "integration adapter manifest",
                "integration_adapter_probe",
                "integration adapter probe",
                "integration_adapter_acceptance",
                "integration adapter acceptance",
                "legacy_connector_migration_audit",
                "legacy connector migration audit",
                "integration_proof_bundle",
                "integration proof bundle",
                "integration_implementation_review",
                "integration implementation review",
                "Implementation review handoff",
                "Execution boundary",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Harness operations brief missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("next_move") != "review_pending_approval":
                raise SystemExit(f"Harness operations should review the setup pending approval first: {metadata}")
            if metadata.get("safe_to_continue") is not False or metadata.get("approval_review_required") is not True:
                raise SystemExit(f"Harness operations missed approval-review metadata: {metadata}")
            assert_harness_operations_handoff(metadata, f"Harness operations {case}")
            if (
                metadata.get("draft_only") is not True
                or metadata.get("requires_manual_send") is not True
                or metadata.get("loads_without_execution") is not True
            ):
                raise SystemExit(f"Harness operations missed review-only contract: {metadata}")
            if (
                metadata.get("authorizes_execution") is not False
                or metadata.get("authorizes_completion_claim") is not False
                or metadata.get("approval_granted") is not False
            ):
                raise SystemExit(f"Harness operations should not grant authority: {metadata}")
            if metadata.get("available_controls", 0) < 8 or metadata.get("missing_controls", 0) != 0:
                raise SystemExit(f"Harness operations missed control coverage: {metadata}")
            if "execution_health_report" not in metadata.get("available_control_tools", []):
                raise SystemExit(f"Harness operations missed execution health control: {metadata}")
            if "integration_execution_matrix" not in metadata.get("available_control_tools", []):
                raise SystemExit(f"Harness operations missed integration execution matrix control: {metadata}")
            if "integration_adapter_manifest" not in metadata.get("available_control_tools", []):
                raise SystemExit(f"Harness operations missed integration adapter manifest control: {metadata}")
            if "integration_adapter_probe" not in metadata.get("available_control_tools", []):
                raise SystemExit(f"Harness operations missed integration adapter probe control: {metadata}")
            if "integration_adapter_acceptance" not in metadata.get("available_control_tools", []):
                raise SystemExit(f"Harness operations missed integration adapter acceptance control: {metadata}")
            if "legacy_connector_migration_audit" not in metadata.get("available_control_tools", []):
                raise SystemExit(f"Harness operations missed legacy connector migration audit control: {metadata}")
            if "integration_dry_run_contract" not in metadata.get("available_control_tools", []):
                raise SystemExit(f"Harness operations missed integration dry-run contract control: {metadata}")
            if "integration_proof_bundle" not in metadata.get("available_control_tools", []):
                raise SystemExit(f"Harness operations missed integration proof bundle control: {metadata}")
            if "integration_implementation_review" not in metadata.get("available_control_tools", []):
                raise SystemExit(f"Harness operations missed integration implementation review control: {metadata}")
            if "integration dry run contract: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only" not in metadata.get("implementation_review_commands", []):
                raise SystemExit(f"Harness operations missed dry-run row-contract implementation review command: {metadata}")
            if "integration proof bundle: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed" not in metadata.get("implementation_review_commands", []):
                raise SystemExit(f"Harness operations missed proof-bundle implementation review command: {metadata}")
            if "legacy connector migration audit: browser calendar email" not in metadata.get("implementation_review_commands", []):
                raise SystemExit(f"Harness operations missed legacy connector migration audit implementation review command: {metadata}")
            if "integration implementation review: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed" not in metadata.get("implementation_review_commands", []):
                raise SystemExit(f"Harness operations missed implementation-review command: {metadata}")
            if metadata.get("implementation_review_command_count") != len(metadata.get("implementation_review_commands", [])):
                raise SystemExit(f"Harness operations missed implementation review command count: {metadata}")
            if "integration proof bundle: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed" not in metadata.get("recommended_control_commands", []):
                raise SystemExit(f"Harness operations missed proof-bundle recommended command: {metadata}")
            if "integration dry run contract: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only" not in metadata.get("recommended_control_commands", []):
                raise SystemExit(f"Harness operations missed dry-run recommended command: {metadata}")
            if "integration implementation review: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed" not in metadata.get("recommended_control_commands", []):
                raise SystemExit(f"Harness operations missed implementation-review recommended command: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Harness operations unsafe metadata {key}: {metadata}")

        gate_result = runtime.handle("agi gates")
        print(f"[{'ok' if gate_result.verified else 'blocked'}] agi gates")
        print(gate_result.response[:2400])
        print()
        if not gate_result.verified:
            raise SystemExit("Expected agi gates to run read-only.")
        for expected in [
            "Jarvis AGI-direction gate report",
            "does not claim Jarvis is AGI",
            "Gate status",
            "Evidence still missing",
            "multi-brain routing",
            "specialist_handoff_quality_gate",
            "specialist_proposal_gate",
            "specialist_action_proposal_contract",
            "specialist_proposal_completion_gate",
            "specialist_execution_handoff_packet",
            "specialist_post_run_closure_packet",
            "specialist_cycle_ledger",
            "Use the specialist execution handoff packet",
            "specialist post-run closure packet",
            "full specialist cycle ledger",
            "real speech input",
            "voice_confirmation_receipt",
            "voice_confirmation_audit_ledger",
            "voice_route_proof_bundle",
            "voice_runtime_bridge_packet",
            "voice_command_cockpit",
            "voice_action_audit_packet",
            "voice_execution_handoff_packet",
            "voice_post_run_closure_packet",
            "voice_cycle_ledger",
            "voice_stop_intent_packet",
            "Use the voice action audit",
            "voice confirmation audit ledger",
            "voice execution handoff packet",
            "voice post-run closure packet",
            "voice cycle ledger",
            "vision observe-act-verify",
            "observe_act_verify_proof_packet",
            "observe_act_verify_route_lock",
            "observe_act_verify_approval_bridge",
            "observe_act_verify_cockpit",
            "observe_act_verify_final_review",
            "observe_act_verify_action_audit",
            "observe_act_verify_execution_handoff",
            "observe_act_verify_post_run_closure",
            "observe-act-verify final review",
            "observe-act-verify action audit",
            "observe-act-verify execution handoff",
            "observe-act-verify post-run closure",
            "personal integrations",
            "integration_adapter_acceptance",
            "legacy_connector_migration_audit",
            "integration_proof_bundle",
            "integration_implementation_review",
            "integration_enablement_gate",
            "Use integration implementation review after the proof bundle",
            "long-running autonomy",
            "operator_timebox_contract",
            "operator timebox contract",
            "STOP_WINDOW_ACTIVE",
            "STOP_TIME_REACHED",
            "HELD_FOR_PARSEABLE_TIMEBOX",
            "checkpoint recovery follow-through packet",
            "autonomy_resume_gate",
            "autonomy resume gate",
            "autonomy_continuation_execution_packet",
            "autonomy continuation execution packet",
            "autonomy_step_closure_packet",
            "autonomy step closure packet",
            "autonomy_cycle_ledger",
            "autonomy cycle ledger",
            "evaluation and learning loop",
            "failure_patch_handoff_packet",
            "failure_patch_closeout_packet",
            "failure_learning_record_packet",
            "Evidence closure commands",
            "Focused verification",
            "Best next commands",
        ]:
            if expected not in gate_result.response:
                raise SystemExit(f"AGI gate report missing expected text: {expected}")
        gate_metadata = gate_result.tool_results[0].metadata
        if gate_metadata.get("gates") != 6:
            raise SystemExit(f"AGI gate report missed gate count: {gate_metadata}")
        if any(row.get("non_authorizing") is not True for row in gate_metadata.get("gate_rows") or []):
            raise SystemExit(f"AGI gate report rows should be non-authorizing: {gate_metadata}")
        if gate_metadata.get("draft_only") is not True or gate_metadata.get("requires_manual_send") is not True or gate_metadata.get("loads_without_execution") is not True:
            raise SystemExit(f"AGI gate report missed review-only contract: {gate_metadata}")
        if gate_metadata.get("authorizes_execution") is not False or gate_metadata.get("authorizes_completion_claim") is not False:
            raise SystemExit(f"AGI gate report should not authorize execution or completion claims: {gate_metadata}")
        if gate_metadata.get("approval_granted") is not False:
            raise SystemExit(f"AGI gate report should not grant approval: {gate_metadata}")
        assert_agi_gate_handoff(gate_metadata, "AGI gate report")
        gate_statuses = gate_metadata.get("gate_statuses") or {}
        present_by_gate = gate_metadata.get("present_evidence_by_gate") or {}
        next_by_gate = gate_metadata.get("next_moves_by_gate") or {}
        closure_by_gate = gate_metadata.get("evidence_closure_commands_by_gate") or {}
        verification_by_gate = gate_metadata.get("focused_verification_by_gate") or {}
        multi_brain_present = present_by_gate.get("multi-brain routing", [])
        if "specialist_handoff_quality_gate" not in multi_brain_present:
            raise SystemExit(f"AGI gate report missed specialist handoff quality gate evidence: {gate_metadata}")
        if "specialist_proposal_gate" not in multi_brain_present:
            raise SystemExit(f"AGI gate report missed specialist proposal gate evidence: {gate_metadata}")
        if "specialist_action_proposal_contract" not in multi_brain_present:
            raise SystemExit(f"AGI gate report missed specialist action proposal contract evidence: {gate_metadata}")
        if "specialist_tool_dry_run_packet" not in multi_brain_present:
            raise SystemExit(f"AGI gate report missed specialist tool dry-run evidence: {gate_metadata}")
        if "specialist_proposal_completion_gate" not in multi_brain_present:
            raise SystemExit(f"AGI gate report missed specialist proposal completion gate evidence: {gate_metadata}")
        if "specialist_execution_handoff_packet" not in multi_brain_present:
            raise SystemExit(f"AGI gate report missed specialist execution handoff evidence: {gate_metadata}")
        if "specialist_post_run_closure_packet" not in multi_brain_present:
            raise SystemExit(f"AGI gate report missed specialist post-run closure evidence: {gate_metadata}")
        if "specialist_cycle_ledger" not in multi_brain_present:
            raise SystemExit(f"AGI gate report missed specialist cycle ledger evidence: {gate_metadata}")
        multi_brain_next = next_by_gate.get("multi-brain routing", "")
        for expected_next in ["specialist execution handoff packet", "specialist post-run closure", "specialist proposal completion gate", "full specialist cycle ledger", "fresh-review boundary", "measured action proposal scorecard", "ToolRegistry", "PermissionPolicy", "verification packet", "runtime trace", "audit", "recovery", "learning", "completion-claim evidence"]:
            if expected_next not in multi_brain_next:
                raise SystemExit(f"AGI gate report missed multi-brain next move metadata '{expected_next}': {gate_metadata}")
        speech_present = present_by_gate.get("real speech input", [])
        if "voice_confirmation_receipt" not in speech_present:
            raise SystemExit(f"AGI gate report missed voice confirmation receipt evidence: {gate_metadata}")
        if "voice_confirmation_audit_ledger" not in speech_present:
            raise SystemExit(f"AGI gate report missed voice confirmation audit ledger evidence: {gate_metadata}")
        if "voice_route_proof_bundle" not in speech_present:
            raise SystemExit(f"AGI gate report missed voice route proof bundle evidence: {gate_metadata}")
        if "voice_runtime_bridge_packet" not in speech_present:
            raise SystemExit(f"AGI gate report missed voice runtime bridge evidence: {gate_metadata}")
        if "voice_command_cockpit" not in speech_present:
            raise SystemExit(f"AGI gate report missed voice command cockpit evidence: {gate_metadata}")
        if "voice_action_audit_packet" not in speech_present:
            raise SystemExit(f"AGI gate report missed voice action audit evidence: {gate_metadata}")
        if "voice_execution_handoff_packet" not in speech_present:
            raise SystemExit(f"AGI gate report missed voice execution handoff evidence: {gate_metadata}")
        if "voice_post_run_closure_packet" not in speech_present:
            raise SystemExit(f"AGI gate report missed voice post-run closure evidence: {gate_metadata}")
        if "voice_cycle_ledger" not in speech_present:
            raise SystemExit(f"AGI gate report missed voice cycle ledger evidence: {gate_metadata}")
        if "voice_stop_intent_packet" not in speech_present:
            raise SystemExit(f"AGI gate report missed voice stop intent evidence: {gate_metadata}")
        speech_next = next_by_gate.get("real speech input", "")
        for expected_next in ["voice stop intent packet", "stop/cancel/rerecord", "voice confirmation audit ledger", "supplied confirmation receipt id/nonce proof", "voice action audit", "voice execution handoff", "voice post-run closure", "voice post-run closure token boundary", "voice cycle ledger", "voice command cockpit", "command-intake-only bridge", "command-intake", "dispatch decision", "execution readiness matrix", "verification packet", "post-run proof queue", "execution audit", "execution health", "after-action learning"]:
            if expected_next not in speech_next:
                raise SystemExit(f"AGI gate report missed speech next move metadata '{expected_next}': {gate_metadata}")
        vision_present = present_by_gate.get("vision observe-act-verify", [])
        if "observe_act_verify_proof_packet" not in vision_present:
            raise SystemExit(f"AGI gate report missed observe-act-verify proof packet evidence: {gate_metadata}")
        if "observe_act_verify_route_lock" not in vision_present:
            raise SystemExit(f"AGI gate report missed observe-act-verify route-lock evidence: {gate_metadata}")
        if "observe_act_verify_approval_bridge" not in vision_present:
            raise SystemExit(f"AGI gate report missed observe-act-verify approval bridge evidence: {gate_metadata}")
        if "observe_act_verify_cockpit" not in vision_present:
            raise SystemExit(f"AGI gate report missed observe-act-verify cockpit evidence: {gate_metadata}")
        if "observe_act_verify_final_review" not in vision_present:
            raise SystemExit(f"AGI gate report missed observe-act-verify final review evidence: {gate_metadata}")
        if "observe_act_verify_action_audit" not in vision_present:
            raise SystemExit(f"AGI gate report missed observe-act-verify action audit evidence: {gate_metadata}")
        if "observe_act_verify_execution_handoff" not in vision_present:
            raise SystemExit(f"AGI gate report missed observe-act-verify execution handoff evidence: {gate_metadata}")
        if "observe_act_verify_post_run_closure" not in vision_present:
            raise SystemExit(f"AGI gate report missed observe-act-verify post-run closure evidence: {gate_metadata}")
        vision_next = next_by_gate.get("vision observe-act-verify", "")
        for expected_next in ["observe-act-verify action audit", "observe-act-verify final review", "execution handoff", "post-run closure", "exact primitive command", "natural-language route lock", "audit evidence", "execution-health evidence", "operator final-review evidence", "post-run proof queue"]:
            if expected_next not in vision_next:
                raise SystemExit(f"AGI gate report missed vision next move metadata '{expected_next}': {gate_metadata}")
        long_running_next = next_by_gate.get("long-running autonomy", "")
        long_running_present = present_by_gate.get("long-running autonomy", [])
        if "operator_timebox_contract" not in long_running_present:
            raise SystemExit(f"AGI gate report missed operator timebox contract evidence: {gate_metadata}")
        if "checkpoint_recovery_cockpit" not in long_running_present:
            raise SystemExit(f"AGI gate report missed checkpoint recovery cockpit evidence: {gate_metadata}")
        if "checkpoint_recovery_followthrough_packet" not in long_running_present:
            raise SystemExit(f"AGI gate report missed checkpoint recovery follow-through packet evidence: {gate_metadata}")
        if "autonomy_resume_gate" not in long_running_present:
            raise SystemExit(f"AGI gate report missed autonomy resume gate evidence: {gate_metadata}")
        if "autonomy_continuation_execution_packet" not in long_running_present:
            raise SystemExit(f"AGI gate report missed autonomy continuation execution evidence: {gate_metadata}")
        if "autonomy_step_closure_packet" not in long_running_present:
            raise SystemExit(f"AGI gate report missed autonomy step closure evidence: {gate_metadata}")
        if "autonomy_cycle_ledger" not in long_running_present:
            raise SystemExit(f"AGI gate report missed autonomy cycle ledger evidence: {gate_metadata}")
        for expected_next in ["operator timebox contract", "operator instruction supersession packet", "checkpoint recovery cockpit", "recovery follow-through packet", "autonomy resume gate", "autonomy continuation execution packet", "autonomy step closure packet", "autonomy cycle ledger", "latest-checkpoint path binding", "arbitrary checkpoint rejection", "hash-bound risky recovery approval boundary token", "prior-cycle ledger tokens as proof-only/non-authorizing evidence", "operator supersession token boundary rows", "autonomy cycle ledger token boundary rows", "full prior-step ledger", "one-step local-safe permission", "post-step proof queue", "risky next-step approval proof queue", "approval boundary rows", "post-step closure evidence", "readable post-step receipt/checkpoint file-hash binding", "approval packets"]:
            if expected_next not in long_running_next:
                raise SystemExit(f"AGI gate report missed long-running autonomy next move metadata '{expected_next}': {gate_metadata}")
        learning_present = present_by_gate.get("evaluation and learning loop", [])
        if "execution_learning_closure_packet" not in learning_present:
            raise SystemExit(f"AGI gate report missed execution learning closure evidence: {gate_metadata}")
        if "failure_learning_cockpit" not in learning_present:
            raise SystemExit(f"AGI gate report missed failure learning cockpit evidence: {gate_metadata}")
        if "failure_patch_receipt_packet" not in learning_present:
            raise SystemExit(f"AGI gate report missed failure patch receipt evidence: {gate_metadata}")
        if "failure_patch_completion_gate" not in learning_present:
            raise SystemExit(f"AGI gate report missed failure patch completion gate evidence: {gate_metadata}")
        if "failure_patch_handoff_packet" not in learning_present:
            raise SystemExit(f"AGI gate report missed failure patch handoff evidence: {gate_metadata}")
        if "failure_patch_closeout_packet" not in learning_present:
            raise SystemExit(f"AGI gate report missed failure patch closeout evidence: {gate_metadata}")
        if "failure_learning_record_packet" not in learning_present:
            raise SystemExit(f"AGI gate report missed failure learning record evidence: {gate_metadata}")
        if "failure_learning_closure_ledger" not in learning_present:
            raise SystemExit(f"AGI gate report missed failure learning closure ledger evidence: {gate_metadata}")
        if "execution learning closure packet" not in next_by_gate.get("evaluation and learning loop", ""):
            raise SystemExit(f"AGI gate report missed execution learning closure next move metadata: {gate_metadata}")
        if "failure learning cockpit" not in next_by_gate.get("evaluation and learning loop", ""):
            raise SystemExit(f"AGI gate report missed failure learning cockpit next move metadata: {gate_metadata}")
        if "failure patch receipt packet" not in next_by_gate.get("evaluation and learning loop", ""):
            raise SystemExit(f"AGI gate report missed failure patch receipt next move metadata: {gate_metadata}")
        if "failure patch completion gate" not in next_by_gate.get("evaluation and learning loop", ""):
            raise SystemExit(f"AGI gate report missed failure patch completion gate next move metadata: {gate_metadata}")
        if "failure patch handoff packet" not in next_by_gate.get("evaluation and learning loop", ""):
            raise SystemExit(f"AGI gate report missed failure patch handoff next move metadata: {gate_metadata}")
        if "failure patch closeout packet" not in next_by_gate.get("evaluation and learning loop", ""):
            raise SystemExit(f"AGI gate report missed failure patch closeout next move metadata: {gate_metadata}")
        if "failure learning record packet" not in next_by_gate.get("evaluation and learning loop", ""):
            raise SystemExit(f"AGI gate report missed failure learning record next move metadata: {gate_metadata}")
        if "failure learning closure ledger" not in next_by_gate.get("evaluation and learning loop", ""):
            raise SystemExit(f"AGI gate report missed failure learning closure ledger next move metadata: {gate_metadata}")
        if "non-authorizing closure token boundary rows" not in next_by_gate.get("evaluation and learning loop", ""):
            raise SystemExit(f"AGI gate report missed learning closure token boundary next move metadata: {gate_metadata}")
        personal_present = present_by_gate.get("personal integrations", [])
        if gate_statuses.get("personal integrations") != "strong prototype":
            raise SystemExit(f"AGI gate report missed personal integration gate status: {gate_metadata}")
        for expected_tool in ["integration_adapter_acceptance", "legacy_connector_migration_audit", "integration_adapter_manifest", "integration_adapter_probe", "integration_execution_matrix", "integration_enablement_gate", "integration_proof_bundle", "integration_implementation_review"]:
            if expected_tool not in personal_present:
                raise SystemExit(f"AGI gate report missed personal integration evidence {expected_tool}: {gate_metadata}")
        if "legacy connector migration audit before connector implementation review" not in next_by_gate.get("personal integrations", ""):
            raise SystemExit(f"AGI gate report missed legacy connector audit next move metadata: {gate_metadata}")
        if "integration implementation review after the proof bundle" not in next_by_gate.get("personal integrations", ""):
            raise SystemExit(f"AGI gate report missed personal integration next move metadata: {gate_metadata}")
        expected_personal_closure = [
            "agi next build move: personal integrations",
            "completion audit: improve AGI gate personal integrations",
            "evidence ledger",
            "completion claim gate: improve AGI gate personal integrations",
        ]
        if closure_by_gate.get("personal integrations") != expected_personal_closure:
            raise SystemExit(f"AGI gate report missed personal integration closure commands: {gate_metadata}")
        if "python3 -m jarvis_v2.scripts.smoke_test_personal" not in verification_by_gate.get("personal integrations", []):
            raise SystemExit(f"AGI gate report missed personal integration focused verification: {gate_metadata}")
        for key in READ_ONLY_FLAGS:
            if gate_metadata.get(key) is not False:
                raise SystemExit(f"AGI gate report unsafe metadata {key}: {gate_metadata}")

        agi_next_cases = ["agi next build move", "agi next build move: personal integrations"]
        for case in agi_next_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis AGI next build move",
                "read-only build-target packet",
                "Selected gate",
                "Build target",
                "Why this gate still matters",
                "Owning files to inspect first",
                "Target integrity",
                "file check: TARGETS_EXIST",
                "missing files: none",
                "Focused verification",
                "Acceptance checks",
                "Safety and recovery gates",
                "recovery closure state",
                "next recovery required",
                "learning debt state",
                "next actionable learning required",
                "ordered learning gate required",
                "Recovery and learning proof handoff",
                "recovery proof queue",
                "learning actionable queue",
                "learning proof queue",
                "Completion proof handoff",
                "Evidence closure plan",
                "completion audit:",
                "evidence ledger",
                "completion claim gate",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"AGI next build move missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("agi_gates") != 6 or metadata.get("agi_real_execution_gap_count") != 6:
                raise SystemExit(f"AGI next build move missed AGI gate metadata: {metadata}")
            assert_agi_next_build_handoff(metadata, f"AGI next build move '{case}'")
            if not metadata.get("selected_gate") or not metadata.get("likely_files") or not metadata.get("focused_tests") or not metadata.get("acceptance_checks"):
                raise SystemExit(f"AGI next build move missed target metadata: {metadata}")
            if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(f"AGI next build move should not authorize execution or completion claims: {metadata}")
            if metadata.get("target_file_integrity_status") != "TARGETS_EXIST":
                raise SystemExit(f"AGI next build move should prove target files exist: {metadata}")
            if metadata.get("target_files_exist") is not True or metadata.get("missing_target_files") != [] or metadata.get("missing_target_file_count") != 0:
                raise SystemExit(f"AGI next build move reported stale target files: {metadata}")
            if metadata.get("target_files_checked") != len(metadata.get("likely_files", [])):
                raise SystemExit(f"AGI next build move target file count diverged: {metadata}")
            if any(not row.get("exists") for row in metadata.get("target_file_rows", [])):
                raise SystemExit(f"AGI next build move target file rows include missing files: {metadata}")
            if "next learning proof:" in result.response or "next learning evidence proof:" in result.response:
                raise SystemExit(f"AGI next build move should use actionable learning required prose for '{case}'.")
            actionable_label = result.response.find("next actionable learning required")
            ordered_label = result.response.find("ordered learning gate required")
            if actionable_label == -1 or ordered_label == -1 or actionable_label > ordered_label:
                raise SystemExit(f"AGI next build move should show actionable learning required before ordered gate required for '{case}'.")
            actionable_queue_label = result.response.find("learning actionable queue")
            legacy_queue_label = result.response.find("learning proof queue")
            if actionable_queue_label == -1 or legacy_queue_label == -1 or actionable_queue_label > legacy_queue_label:
                raise SystemExit(f"AGI next build move should show actionable learning queue before legacy proof queue for '{case}'.")
            assert_execution_learning_actionable_aliases(metadata, f"AGI next build move '{case}'")
            actionable_learning_commands = metadata.get("execution_learning_actionable_required_commands") or []
            if metadata.get("execution_learning_blocks_completion_claim"):
                if not any(str(command).startswith("after-action learning packet") for command in actionable_learning_commands):
                    raise SystemExit(f"AGI next build move missed actionable after-action learning command: {metadata}")
                if not any(str(command).startswith("execution learning closure") for command in actionable_learning_commands):
                    raise SystemExit(f"AGI next build move missed actionable closure recheck command: {metadata}")
                after_index = next(index for index, command in enumerate(actionable_learning_commands) if str(command).startswith("after-action learning packet"))
                closure_index = next(index for index, command in enumerate(actionable_learning_commands) if str(command).startswith("execution learning closure"))
                if after_index > closure_index:
                    raise SystemExit(f"AGI next build move actionable learning queue is not evidence-first: {metadata}")
            if case.endswith("personal integrations") and metadata.get("selected_gate") != "personal integrations":
                raise SystemExit(f"AGI next build move did not honor focused gate: {metadata}")
            if case.endswith("personal integrations"):
                if metadata.get("agi_focus_selection_source") != "operator_requested_gate":
                    raise SystemExit(f"AGI next build move missed requested focus selection source: {metadata}")
                if metadata.get("agi_focus_deliberate_focus_override") is not True:
                    raise SystemExit(f"AGI next build move missed requested focus override flag: {metadata}")
            else:
                if metadata.get("agi_focus_selection_source") != "default_ranked_gate":
                    raise SystemExit(f"AGI next build move missed default focus selection source: {metadata}")
                if metadata.get("agi_focus_deliberate_focus_override") is not False:
                    raise SystemExit(f"AGI next build move default focus should not be a deliberate override: {metadata}")
            if metadata.get("selected_gate") == "vision observe-act-verify":
                joined_acceptance = " ".join(metadata.get("acceptance_checks", []))
                if "observe-act-verify proof packet" not in joined_acceptance or "observe-act-verify route lock" not in joined_acceptance or "approved-rerun proof" not in joined_acceptance or "observe-act-verify execution handoff" not in joined_acceptance or "observe-act-verify post-run closure" not in joined_acceptance:
                    raise SystemExit(f"AGI next build move missed vision proof-packet acceptance: {metadata}")
                if "python3 -m jarvis_v2.scripts.smoke_test_computer_plan" not in metadata.get("focused_tests", []):
                    raise SystemExit(f"AGI next build move missed vision proof focused test: {metadata}")
            if metadata.get("selected_gate") == "real speech input":
                joined_acceptance = " ".join(metadata.get("acceptance_checks", []))
                if "voice execution handoff packets" not in joined_acceptance or "post-run verification/audit/learning proof queue" not in joined_acceptance or "voice post-run closure packets" not in joined_acceptance or "voice post-run closure token boundary rows" not in joined_acceptance:
                    raise SystemExit(f"AGI next build move missed voice execution handoff acceptance: {metadata}")
                if "python3 -m jarvis_v2.scripts.smoke_test_voice" not in metadata.get("focused_tests", []):
                    raise SystemExit(f"AGI next build move missed voice focused test: {metadata}")
            if metadata.get("selected_gate") == "evaluation and learning loop":
                joined_acceptance = " ".join(metadata.get("acceptance_checks", []))
                if "execution learning closure packet" not in joined_acceptance or "repeated-failure promotion proof" not in joined_acceptance or "exact regression-test contract hash" not in joined_acceptance or "observed failures" not in joined_acceptance or "focused command" not in joined_acceptance or "failure patch receipt packet" not in joined_acceptance or "failure patch completion gate" not in joined_acceptance or "failure patch handoff packet" not in joined_acceptance or "failure patch closeout packet" not in joined_acceptance or "failure learning record packet" not in joined_acceptance or "failure learning closure ledger" not in joined_acceptance or "closure token boundary rows" not in joined_acceptance:
                    raise SystemExit(f"AGI next build move missed learning closure acceptance: {metadata}")
            if metadata.get("selected_gate") == "long-running autonomy":
                joined_acceptance = " ".join(metadata.get("acceptance_checks", []))
                if "checkpoint recovery follow-through packet" not in joined_acceptance or "normal follow-through resumes" not in joined_acceptance or "hash-bound risky recovery approval boundary token" not in joined_acceptance or "autonomy resume gate" not in joined_acceptance or "autonomy continuation execution packet" not in joined_acceptance or "post-step proof queue" not in joined_acceptance or "risky next-step approval proof queue" not in joined_acceptance or "approval boundary rows" not in joined_acceptance or "proof-only/non-authorizing evidence" not in joined_acceptance or "fresh cycle ledger token" not in joined_acceptance or "autonomy cycle ledger token boundary rows" not in joined_acceptance or "operator instruction supersession" not in joined_acceptance or "proof-only supersession token boundary" not in joined_acceptance or "operator instruction supersession token boundary travels through resume" not in joined_acceptance or "status APIs" not in joined_acceptance or "autonomy step closure packet" not in joined_acceptance or "receipt hash matching that file" not in joined_acceptance or "awake guard boundary proof" not in joined_acceptance or "OS wake locks" not in joined_acceptance:
                    raise SystemExit(f"AGI next build move missed long-running follow-through acceptance: {metadata}")
            if case.endswith("personal integrations"):
                joined_acceptance = " ".join(metadata.get("acceptance_checks", []))
                if "metadata row contract" not in joined_acceptance or "row limit" not in joined_acceptance or "promotion gate" not in joined_acceptance or "dry-run contract" not in joined_acceptance:
                    raise SystemExit(f"AGI next build move missed connector row-contract acceptance: {metadata}")
            selected_gate = metadata.get("selected_gate")
            expected_audit = f"completion audit: improve AGI gate {selected_gate}"
            expected_claim = f"completion claim gate: improve AGI gate {selected_gate}"
            if metadata.get("completion_audit_command") != expected_audit:
                raise SystemExit(f"AGI next build move missed completion audit handoff: {metadata}")
            if metadata.get("evidence_ledger_command") != "evidence ledger":
                raise SystemExit(f"AGI next build move missed evidence ledger handoff: {metadata}")
            if metadata.get("completion_claim_gate_command") != expected_claim:
                raise SystemExit(f"AGI next build move missed completion claim gate handoff: {metadata}")
            if metadata.get("proof_handoff_commands") != [expected_audit, "evidence ledger", expected_claim]:
                raise SystemExit(f"AGI next build move missed ordered proof handoff commands: {metadata}")
            if metadata.get("proof_handoff_command_count") != 3:
                raise SystemExit(f"AGI next build move missed proof handoff command count: {metadata}")
            expected_closure = [metadata.get("next_command"), expected_audit, "evidence ledger", expected_claim]
            if metadata.get("evidence_closure_commands") != expected_closure:
                raise SystemExit(f"AGI next build move missed evidence closure commands: {metadata}")
            if metadata.get("evidence_closure_command_count") != len(expected_closure):
                raise SystemExit(f"AGI next build move missed evidence closure command count: {metadata}")
            if metadata.get("focused_verification_commands") != metadata.get("focused_tests"):
                raise SystemExit(f"AGI next build move focused verification commands should mirror focused tests: {metadata}")
            if metadata.get("focused_verification_command_count") != len(metadata.get("focused_tests", [])):
                raise SystemExit(f"AGI next build move missed focused verification command count: {metadata}")
            acceptance_preview = metadata.get("acceptance_gap_preview") or []
            if metadata.get("acceptance_preview") != acceptance_preview:
                raise SystemExit(f"AGI next build move missed clearer acceptance preview alias: {metadata}")
            if metadata.get("acceptance_preview_count") != len(acceptance_preview):
                raise SystemExit(f"AGI next build move missed clearer acceptance preview count: {metadata}")
            if metadata.get("first_acceptance_check") != (acceptance_preview[0] if acceptance_preview else ""):
                raise SystemExit(f"AGI next build move missed clearer first acceptance check alias: {metadata}")
            if metadata.get("acceptance_gap_preview_count") != len(acceptance_preview):
                raise SystemExit(f"AGI next build move missed acceptance gap preview count: {metadata}")
            if acceptance_preview != metadata.get("acceptance_checks", [])[:3]:
                raise SystemExit(f"AGI next build move acceptance gap preview diverged from acceptance checks: {metadata}")
            if metadata.get("first_acceptance_gap") != (acceptance_preview[0] if acceptance_preview else ""):
                raise SystemExit(f"AGI next build move first acceptance gap diverged: {metadata}")
            if metadata.get("agi_next_acceptance_gap_preview") != acceptance_preview:
                raise SystemExit(f"AGI next build move selected acceptance preview alias diverged: {metadata}")
            if metadata.get("agi_next_acceptance_preview") != acceptance_preview:
                raise SystemExit(f"AGI next build move clearer selected acceptance preview alias diverged: {metadata}")
            if metadata.get("agi_next_acceptance_preview_count") != len(acceptance_preview):
                raise SystemExit(f"AGI next build move clearer selected acceptance preview count diverged: {metadata}")
            if metadata.get("agi_next_first_acceptance_check") != metadata.get("first_acceptance_check"):
                raise SystemExit(f"AGI next build move clearer selected first acceptance check alias diverged: {metadata}")
            if metadata.get("agi_next_acceptance_gap_preview_count") != len(acceptance_preview):
                raise SystemExit(f"AGI next build move selected acceptance preview count diverged: {metadata}")
            if metadata.get("agi_next_first_acceptance_gap") != metadata.get("first_acceptance_gap"):
                raise SystemExit(f"AGI next build move selected first acceptance gap alias diverged: {metadata}")
            if "execution_health_recovery_closure_missing_count" not in metadata:
                raise SystemExit(f"AGI next build move missed recovery closure proof metadata: {metadata}")
            assert_execution_health_recovery_proof_aliases(metadata, "AGI next build move")
            if metadata.get("execution_learning_state") in {None, ""}:
                raise SystemExit(f"AGI next build move missed execution learning state: {metadata}")
            if metadata.get("execution_learning_missing_count", 0) < 1:
                raise SystemExit(f"AGI next build move missed execution learning missing proof count: {metadata}")
            assert_execution_learning_proof_aliases(metadata, "AGI next build move")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"AGI next build move unsafe metadata {key}: {metadata}")

        completion_result = runtime.handle("harness completion")
        print(f"[{'ok' if completion_result.verified else 'blocked'}] harness completion")
        print(completion_result.response[:2200])
        print()
        if not completion_result.verified:
            raise SystemExit("Expected harness completion to run read-only.")
        for expected in [
            "Jarvis harness completion assessment",
            "not AGI itself",
            "Overall harness prototype",
            "component foundation",
            "AGI-direction gates",
            "execution readiness evidence",
            "Remaining real-execution gaps",
            "Safety floor",
            "Completion blockers",
            "Execution learning debt",
            "AGI real-execution gaps block completion claim",
            "learning evidence next required:",
            "learning actionable next required:",
            "learning actionable proof alias:",
            "learning closure command:",
            "Selected AGI next build target",
            "Next required commands",
            "Completion proof queue",
            "Completion claim ready: no",
            "after-action learning packet",
            "completion audit",
            "evidence ledger",
            "completion claim gate",
        ]:
            if expected not in completion_result.response:
                raise SystemExit(f"Harness completion missing expected text: {expected}")
        if "Next proof commands" in completion_result.response:
            raise SystemExit("Harness completion should render the operator queue as Next required commands.")
        if "next learning proof:" in completion_result.response or "next learning evidence proof:" in completion_result.response:
            raise SystemExit("Harness completion should use command-first learning next-required prose.")
        completion_metadata = completion_result.tool_results[0].metadata
        for key in ["overall_percent", "component_percent", "gate_percent", "execution_readiness_percent"]:
            if not isinstance(completion_metadata.get(key), int):
                raise SystemExit(f"Harness completion missed numeric metadata {key}: {completion_metadata}")
        if completion_metadata["overall_percent"] <= 0:
            raise SystemExit(f"Harness completion should be above zero: {completion_metadata}")
        if completion_metadata.get("completion_claim_ready") is not False:
            raise SystemExit(f"Harness completion should block final claim while gaps remain: {completion_metadata}")
        if not completion_metadata.get("completion_blockers") or not completion_metadata.get("completion_blocker_count"):
            raise SystemExit(f"Harness completion missed structured completion blockers: {completion_metadata}")
        if completion_metadata.get("completion_blockers_deduplicated") is not True:
            raise SystemExit(f"Harness completion should deduplicate completion blockers: {completion_metadata}")
        if completion_metadata.get("execution_learning_state") in {None, ""}:
            raise SystemExit(f"Harness completion missed execution learning state: {completion_metadata}")
        assert_execution_health_recovery_proof_aliases(completion_metadata, "Harness completion")
        assert_execution_health_verification_proof_aliases(completion_metadata, "Harness completion")
        if completion_metadata.get("execution_learning_blocks_completion_claim") and completion_metadata.get("execution_learning_missing_count", 0) < 1:
            raise SystemExit(f"Harness completion missed execution learning missing proof count: {completion_metadata}")
        if completion_metadata.get("execution_learning_blocks_completion_claim") and completion_metadata.get("execution_learning_target_run_id") is None:
            raise SystemExit(f"Harness completion missed execution learning target run: {completion_metadata}")
        learning_closure_command = completion_metadata.get("execution_learning_closure_command")
        if completion_metadata.get("execution_learning_blocks_completion_claim") and (not learning_closure_command or not str(learning_closure_command).startswith("execution learning closure")):
            raise SystemExit(f"Harness completion missed execution learning closure command: {completion_metadata}")
        if completion_metadata.get("execution_learning_blocks_completion_claim"):
            evidence_label = completion_result.response.find("learning evidence next required:")
            actionable_label = completion_result.response.find("learning actionable next required:")
            closure_label = completion_result.response.find("learning closure command:")
            if min(evidence_label, actionable_label, closure_label) < 0:
                raise SystemExit(f"Harness completion missed learning prose labels: {completion_result.response}")
            if not (evidence_label < closure_label and actionable_label < closure_label):
                raise SystemExit(f"Harness completion should show actionable learning evidence before closure: {completion_result.response}")
        learning_commands = completion_metadata.get("execution_learning_required_commands") or []
        if completion_metadata.get("execution_learning_blocks_completion_claim") and not any(str(command).startswith("after-action learning packet") for command in learning_commands):
            raise SystemExit(f"Harness completion missed after-action learning proof command: {completion_metadata}")
        if learning_commands and completion_metadata.get("execution_learning_next_required_command") != learning_commands[0]:
            raise SystemExit(f"Harness completion missed first learning proof command: {completion_metadata}")
        assert_execution_learning_proof_aliases(completion_metadata, "Harness completion")
        for command in ["completion audit", "evidence ledger", "completion claim gate"]:
            if command not in completion_metadata.get("next_proof_commands", []):
                raise SystemExit(f"Harness completion missed next proof command {command}: {completion_metadata}")
            if command not in completion_metadata.get("completion_proof_queue", []):
                raise SystemExit(f"Harness completion missed completion proof queue command {command}: {completion_metadata}")
        if learning_closure_command not in completion_metadata.get("next_proof_commands", []):
            raise SystemExit(f"Harness completion missed learning closure in next proof queue: {completion_metadata}")
        if learning_closure_command not in completion_metadata.get("completion_proof_queue", []):
            raise SystemExit(f"Harness completion missed learning closure in completion proof queue: {completion_metadata}")
        if learning_commands[0] not in completion_metadata.get("next_proof_commands", []):
            raise SystemExit(f"Harness completion missed learning command in next proof queue: {completion_metadata}")
        proof_queue = completion_metadata.get("completion_proof_queue") or []
        if not proof_queue or completion_metadata.get("completion_proof_queue_count") != len(proof_queue):
            raise SystemExit(f"Harness completion missed completion proof queue metadata: {completion_metadata}")
        if completion_metadata.get("execution_learning_blocks_completion_claim"):
            after_action_command = completion_metadata.get("execution_learning_after_action_learning_command")
            if after_action_command not in proof_queue:
                raise SystemExit(f"Harness completion proof queue missed after-action learning evidence command {after_action_command}: {completion_metadata}")
            if learning_closure_command in proof_queue and proof_queue.index(after_action_command) > proof_queue.index(learning_closure_command):
                raise SystemExit(f"Harness completion proof queue should put after-action evidence before learning closure: {completion_metadata}")
        for expected_command in completion_metadata.get("execution_health_recovery_closure_proof_queue") or []:
            if expected_command not in proof_queue:
                raise SystemExit(f"Harness completion proof queue missed explicit recovery proof command {expected_command}: {completion_metadata}")
        for expected_command in completion_metadata.get("execution_learning_proof_queue") or []:
            if expected_command not in proof_queue:
                raise SystemExit(f"Harness completion proof queue missed explicit learning proof command {expected_command}: {completion_metadata}")
        if completion_metadata.get("completion_next_proof_command") != proof_queue[0] or completion_metadata.get("next_completion_proof_command") != proof_queue[0]:
            raise SystemExit(f"Harness completion missed first completion proof command: {completion_metadata}")
        if completion_metadata.get("next_proof_command") != completion_metadata.get("next_proof_commands", [""])[0]:
            raise SystemExit(f"Harness completion missed first next proof command alias: {completion_metadata}")
        if completion_metadata.get("agi_real_execution_gap_count", 0) <= 0:
            raise SystemExit(f"Harness completion should expose AGI real-execution gaps: {completion_metadata}")
        if completion_metadata.get("agi_real_execution_blocks_completion_claim") is not True:
            raise SystemExit(f"Harness completion should mark AGI gaps as completion-claim blockers: {completion_metadata}")
        if completion_metadata.get("completion_claim_blocked_by_agi_gaps") is not True:
            raise SystemExit(f"Harness completion missed completion-claim AGI gap alias: {completion_metadata}")
        if completion_metadata.get("storage_runtime_fallback_active") is not False:
            raise SystemExit(f"Harness completion should report no storage fallback for temp runtime: {completion_metadata}")
        if completion_metadata.get("storage_readiness_blocks_completion_claim") is not False:
            raise SystemExit(f"Harness completion should not block on storage when primary temp storage is writable: {completion_metadata}")
        if completion_metadata.get("storage_readiness_next_commands") != [] or completion_metadata.get("storage_readiness_next_command_count") != 0:
            raise SystemExit(f"Harness completion should expose an empty storage recovery queue when fallback is inactive: {completion_metadata}")
        if completion_metadata.get("agi_real_execution_gap_count", 0) > 0 and completion_metadata.get("completion_claim_ready") is not False:
            raise SystemExit(f"Harness completion should not claim readiness while AGI gaps remain: {completion_metadata}")
        if not completion_metadata.get("agi_next_gate") or not completion_metadata.get("agi_next_build_command"):
            raise SystemExit(f"Harness completion missed selected AGI next build target: {completion_metadata}")
        if not completion_metadata.get("agi_next_target_title"):
            raise SystemExit(f"Harness completion missed selected AGI target title: {completion_metadata}")
        if completion_metadata.get("agi_next_evidence_closure_command_count") != len(completion_metadata.get("agi_next_evidence_closure_commands", [])):
            raise SystemExit(f"Harness completion missed selected AGI closure command count: {completion_metadata}")
        if completion_metadata.get("agi_next_focused_verification_command_count") != len(completion_metadata.get("agi_next_focused_verification_commands", [])):
            raise SystemExit(f"Harness completion missed selected AGI focused verification count: {completion_metadata}")
        if completion_metadata.get("agi_next_likely_file_count") != len(completion_metadata.get("agi_next_likely_files", [])):
            raise SystemExit(f"Harness completion missed selected AGI likely file count: {completion_metadata}")
        if completion_metadata.get("agi_next_acceptance_check_count") != len(completion_metadata.get("agi_next_acceptance_checks", [])):
            raise SystemExit(f"Harness completion missed selected AGI acceptance count: {completion_metadata}")
        assert_selected_agi_target_readiness(completion_metadata, "Harness completion")
        if bool(completion_metadata.get("agi_next_target_integrity_blocks_start")) is completion_metadata.get("agi_next_target_files_exist"):
            raise SystemExit(f"Harness completion target-integrity blocker should invert target_files_exist: {completion_metadata}")
        expected_agi_ready = bool(
            completion_metadata.get("agi_next_target_files_exist")
            and completion_metadata.get("agi_next_focused_verification_commands")
            and completion_metadata.get("agi_next_acceptance_checks")
        )
        if completion_metadata.get("agi_next_build_packet_ready_for_review") is not expected_agi_ready:
            raise SystemExit(f"Harness completion selected AGI readiness flag diverged: {completion_metadata}")
        assert_selected_agi_target_readiness(completion_metadata, "Harness completion")
        assert_harness_completion_handoff(completion_metadata, "Harness completion")
        if completion_metadata.get("agi_next_build_command") not in completion_metadata.get("completion_proof_queue", []):
            raise SystemExit(f"Harness completion proof queue missed selected AGI build command: {completion_metadata}")
        if completion_metadata.get("draft_only") is not True or completion_metadata.get("requires_manual_send") is not True or completion_metadata.get("loads_without_execution") is not True:
            raise SystemExit(f"Harness completion missed review-only contract: {completion_metadata}")
        if completion_metadata.get("authorizes_execution") is not False or completion_metadata.get("authorizes_completion_claim") is not False:
            raise SystemExit(f"Harness completion should not authorize execution or completion claims: {completion_metadata}")
        if completion_metadata.get("approval_granted") is not False:
            raise SystemExit(f"Harness completion should not grant approval: {completion_metadata}")
        for key in READ_ONLY_FLAGS:
            if completion_metadata.get(key) is not False:
                raise SystemExit(f"Harness completion unsafe metadata {key}: {completion_metadata}")

        audit_cases = [
            "completion audit",
            "completion audit: Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant",
            "what is missing before Jarvis is complete",
            "why isn't Jarvis done",
        ]
        for case in audit_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "completion_audit_packet":
                raise SystemExit(f"Expected '{case}' to route to completion_audit_packet.")
            for expected in [
                "Jarvis completion audit packet",
                "proof checklist",
                "Objective under audit",
                "Audit rule",
                "Requirement evidence",
                "AGI-direction gate coverage",
                "real-execution gaps still tracked",
                "selected next build gate",
                "selected next build command",
                "Evidence closure commands",
                "Focused verification",
                "command-first interface",
                "routing and tool orchestration",
                "planner gap",
                "argument contract",
                "approval gates and audit",
                "approval readiness",
                "verification and recovery",
                "learning loop",
                "Current-state blockers",
                "AGI real-execution gaps block completion claim",
                "recovery closure state",
                "recovery closure ready to retry",
                "recovery closure missing",
                "recovery next required",
                "recovery closure command queue",
                "execution learning debt state",
                "execution learning debt missing",
                "execution learning closure command",
                "learning next required",
                "execution learning command queue",
                "after-action learning packet",
                "Next verification commands",
                "Completion proof queue",
                "Completion verdict",
                "NOT COMPLETE",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Completion audit missing expected text for '{case}': {expected}")
            if "next required proof" in result.response:
                raise SystemExit(f"Completion audit should not use ambiguous recovery next-required/proof prose for '{case}'.")
            if "next learning proof:" in result.response or "next learning evidence proof:" in result.response:
                raise SystemExit(f"Completion audit should use command-first learning next-required prose for '{case}'.")
            metadata = result.tool_results[0].metadata
            if metadata.get("requirements") != 7:
                raise SystemExit(f"Completion audit missed requirement count: {metadata}")
            if metadata.get("agi_gates") != 6 or metadata.get("agi_real_execution_gap_count") != 6:
                raise SystemExit(f"Completion audit missed AGI gate metadata: {metadata}")
            if metadata.get("agi_real_execution_blocks_completion_claim") is not True:
                raise SystemExit(f"Completion audit should mark AGI gaps as completion-claim blockers: {metadata}")
            if metadata.get("completion_claim_blocked_by_agi_gaps") is not True:
                raise SystemExit(f"Completion audit missed completion-claim AGI gap alias: {metadata}")
            if metadata.get("agi_real_execution_gap_count", 0) > 0 and metadata.get("completion_claim_ready") is not False:
                raise SystemExit(f"Completion audit should not claim readiness while AGI gaps remain: {metadata}")
            if not metadata.get("agi_gate_statuses") or "personal integrations" not in metadata.get("agi_gate_statuses", {}):
                raise SystemExit(f"Completion audit missed AGI gate statuses: {metadata}")
            if "integration_adapter_acceptance" not in metadata.get("agi_missing_evidence_by_gate", {}).get("personal integrations", []) and metadata.get("agi_gate_statuses", {}).get("personal integrations") != "strong prototype":
                raise SystemExit(f"Completion audit missed personal integration adapter gate metadata: {metadata}")
            if "legacy_connector_migration_audit" not in metadata.get("agi_present_evidence_by_gate", {}).get("personal integrations", []):
                raise SystemExit(f"Completion audit missed legacy connector audit personal integration evidence: {metadata}")
            expected_personal_closure = [
                "agi next build move: personal integrations",
                "completion audit: improve AGI gate personal integrations",
                "evidence ledger",
                "completion claim gate: improve AGI gate personal integrations",
            ]
            audit_closure_by_gate = metadata.get("agi_evidence_closure_commands_by_gate") or {}
            if audit_closure_by_gate.get("personal integrations") != expected_personal_closure:
                raise SystemExit(f"Completion audit missed AGI closure commands: {metadata}")
            audit_verification_by_gate = metadata.get("agi_focused_verification_by_gate") or {}
            if "python3 -m jarvis_v2.scripts.smoke_test_personal" not in audit_verification_by_gate.get("personal integrations", []):
                raise SystemExit(f"Completion audit missed AGI focused verification commands: {metadata}")
            if not metadata.get("agi_next_gate") or not metadata.get("agi_next_build_command"):
                raise SystemExit(f"Completion audit missed selected AGI next build target: {metadata}")
            if metadata.get("agi_next_evidence_closure_command_count") != len(metadata.get("agi_next_evidence_closure_commands", [])):
                raise SystemExit(f"Completion audit missed selected AGI closure command count: {metadata}")
            assert_selected_agi_target_readiness(metadata, "Completion audit")
            if metadata.get("agi_next_build_command") not in proof_queue:
                raise SystemExit(f"Completion audit proof queue missed selected AGI build command: {metadata}")
            if metadata.get("completion_claim_ready") is not False:
                raise SystemExit(f"Completion audit should not claim completion with open task/approval setup: {metadata}")
            if metadata.get("pending_approvals", 0) < 1 or metadata.get("open_tasks", 0) < 1:
                raise SystemExit(f"Completion audit missed current blockers: {metadata}")
            proof_queue = metadata.get("completion_proof_queue") or []
            if not proof_queue or metadata.get("completion_proof_queue_count") != len(proof_queue):
                raise SystemExit(f"Completion audit missed completion proof queue metadata: {metadata}")
            for expected_command in metadata.get("execution_health_recovery_closure_proof_queue") or []:
                if expected_command not in proof_queue:
                    raise SystemExit(f"Completion audit proof queue missed explicit recovery proof command {expected_command}: {metadata}")
            for expected_command in metadata.get("execution_learning_proof_queue") or []:
                if expected_command not in proof_queue:
                    raise SystemExit(f"Completion audit proof queue missed explicit learning proof command {expected_command}: {metadata}")
            if metadata.get("completion_next_proof_command") != proof_queue[0] or metadata.get("next_completion_proof_command") != proof_queue[0]:
                raise SystemExit(f"Completion audit missed first completion proof command: {metadata}")
            if metadata.get("next_proof_command") != metadata.get("next_proof_commands", [""])[0]:
                raise SystemExit(f"Completion audit missed first next proof command alias: {metadata}")
            assert_completion_audit_handoff(metadata, "Completion audit")
            for command in ["harness completion", "agi gates", "completion audit", "evidence ledger", "completion claim gate"]:
                if command not in proof_queue:
                    raise SystemExit(f"Completion audit proof queue missed command {command}: {metadata}")
            if metadata.get("execution_health_recovery_closure_state") in {None, ""}:
                raise SystemExit(f"Completion audit missed recovery closure state: {metadata}")
            if "execution_health_recovery_closure_ready_to_retry" not in metadata:
                raise SystemExit(f"Completion audit missed recovery closure retry readiness: {metadata}")
            if "execution_health_recovery_closure_missing_count" not in metadata:
                raise SystemExit(f"Completion audit missed recovery closure missing count: {metadata}")
            closure_commands = metadata.get("execution_health_recovery_closure_required_commands") or []
            if metadata.get("execution_health_recovery_closure_blocks_completion_claim") and not closure_commands:
                raise SystemExit(f"Completion audit blocked completion without closure commands: {metadata}")
            if metadata.get("execution_health_recovery_closure_blocks_completion_claim") and metadata.get("execution_health_recovery_closure_target_run_id") is None:
                raise SystemExit(f"Completion audit missed recovery closure target run: {metadata}")
            if closure_commands and metadata.get("execution_health_recovery_closure_next_required_command") != closure_commands[0]:
                raise SystemExit(f"Completion audit missed first recovery proof command: {metadata}")
            assert_execution_health_recovery_proof_aliases(metadata, "Completion audit")
            assert_execution_health_verification_proof_aliases(metadata, "Completion audit")
            for expected_command in closure_commands:
                if expected_command not in proof_queue:
                    raise SystemExit(f"Completion audit proof queue missed recovery command {expected_command}: {metadata}")
            if metadata.get("execution_learning_state") in {None, ""}:
                raise SystemExit(f"Completion audit missed execution learning state: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") is not True:
                raise SystemExit(f"Completion audit should block on execution learning debt: {metadata}")
            if metadata.get("execution_learning_missing_count", 0) < 1:
                raise SystemExit(f"Completion audit missed execution learning missing proof count: {metadata}")
            if metadata.get("execution_learning_target_run_id") is None:
                raise SystemExit(f"Completion audit missed execution learning target run: {metadata}")
            learning_commands = metadata.get("execution_learning_required_commands") or []
            if metadata.get("execution_learning_blocks_completion_claim") and not any(str(command).startswith("after-action learning packet") for command in learning_commands):
                raise SystemExit(f"Completion audit missed after-action learning proof command: {metadata}")
            learning_closure_command = metadata.get("execution_learning_closure_command")
            if metadata.get("execution_learning_blocks_completion_claim") and (not learning_closure_command or not str(learning_closure_command).startswith("execution learning closure")):
                raise SystemExit(f"Completion audit missed execution learning closure command: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and learning_closure_command not in proof_queue:
                raise SystemExit(f"Completion audit proof queue missed learning closure command {learning_closure_command}: {metadata}")
            if learning_commands and metadata.get("execution_learning_next_required_command") != learning_commands[0]:
                raise SystemExit(f"Completion audit missed first learning proof command: {metadata}")
            assert_execution_learning_proof_aliases(metadata, "Completion audit")
            for expected_command in learning_commands:
                if expected_command not in proof_queue:
                    raise SystemExit(f"Completion audit proof queue missed learning command {expected_command}: {metadata}")
            if metadata.get("draft_only") is not True or metadata.get("requires_manual_send") is not True or metadata.get("loads_without_execution") is not True:
                raise SystemExit(f"Completion audit missed review-only contract: {metadata}")
            if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(f"Completion audit should not authorize execution or completion claims: {metadata}")
            if metadata.get("approval_granted") is not False:
                raise SystemExit(f"Completion audit should not grant approval: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Completion audit unsafe metadata {key}: {metadata}")

        ledger_cases = [
            "evidence ledger",
            "proof ledger: Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant",
            "what proof do we have for Jarvis completion",
            "show me the evidence for Jarvis completion",
        ]
        for case in ledger_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "evidence_ledger":
                raise SystemExit(f"Expected '{case}' to route to evidence_ledger.")
            for expected in [
                "Jarvis evidence ledger",
                "harness proof ledger",
                "Harness evidence lanes",
                "AGI-direction gates",
                "real-execution gaps still tracked",
                "selected next build gate",
                "selected next build command",
                "Evidence closure commands",
                "steering",
                "planner_gap_packet",
                "argument_contract_packet",
                "pedals",
                "brakes",
                "dashboard",
                "memory/state",
                "audit",
                "execution_health_report",
                "after_action_learning_packet",
                "Next required command",
                "dispatch decision: <next real order>",
                "learning review",
                "recovery",
                "learning",
                "Current audit state",
                "recent after-action learning packets",
                "recovery closure state",
                "recovery closure ready to retry",
                "recovery closure missing",
                "recovery next required",
                "recovery closure command queue",
                "execution learning debt state",
                "execution learning debt missing",
                "learning next required",
                "execution learning command queue",
                "Latest execution case proof state",
                "latest case: none saved",
                "case verdict: CASE_NOT_FOUND",
                "case closure verdict: CASE_CLOSURE_NOT_FOUND",
                "case closure ready: False",
                "case blocks completion claim: True",
                "mission command queue: 0 command(s)",
                "next case required command",
                "next case closure command",
                "Claim state",
                "read-only",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Evidence ledger missing expected text for '{case}': {expected}")
            if "next required proof" in result.response:
                raise SystemExit(f"Evidence ledger should not use ambiguous recovery next-required/proof prose for '{case}'.")
            if "Next proof command:" in result.response:
                raise SystemExit(f"Evidence ledger should render next required command, not next proof command for '{case}'.")
            if "next learning proof:" in result.response or "next learning evidence proof:" in result.response:
                raise SystemExit(f"Evidence ledger should use command-first learning next-required prose for '{case}'.")
            if "next case proof command" in result.response:
                raise SystemExit(f"Evidence ledger should render next case required command for '{case}'.")
            metadata = result.tool_results[0].metadata
            if metadata.get("lanes") != 8:
                raise SystemExit(f"Evidence ledger missed lane count: {metadata}")
            if metadata.get("agi_gates") != 6 or metadata.get("agi_real_execution_gap_count") != 6:
                raise SystemExit(f"Evidence ledger missed AGI gate metadata: {metadata}")
            if not metadata.get("agi_real_execution_gaps_by_gate") or "multi-brain routing" not in metadata.get("agi_real_execution_gaps_by_gate", {}):
                raise SystemExit(f"Evidence ledger missed AGI real-execution gap details: {metadata}")
            ledger_closure_by_gate = metadata.get("agi_evidence_closure_commands_by_gate") or {}
            if ledger_closure_by_gate.get("personal integrations") != expected_personal_closure:
                raise SystemExit(f"Evidence ledger missed AGI closure commands: {metadata}")
            if not metadata.get("agi_next_gate") or not metadata.get("agi_next_build_command"):
                raise SystemExit(f"Evidence ledger missed selected AGI next build target: {metadata}")
            if metadata.get("agi_next_evidence_closure_command_count") != len(metadata.get("agi_next_evidence_closure_commands", [])):
                raise SystemExit(f"Evidence ledger missed selected AGI closure command count: {metadata}")
            assert_selected_agi_target_readiness(metadata, "Evidence ledger")
            if metadata.get("completion_claim_ready") is not False:
                raise SystemExit(f"Evidence ledger should not claim readiness with setup blockers: {metadata}")
            if metadata.get("pending_approvals", 0) < 1 or metadata.get("open_tasks", 0) < 1:
                raise SystemExit(f"Evidence ledger missed current blockers: {metadata}")
            if not str(metadata.get("claim_state", "")).strip():
                raise SystemExit(f"Evidence ledger missed claim state: {metadata}")
            if "after_action_learning_runs" not in metadata:
                raise SystemExit(f"Evidence ledger missed after-action learning run count: {metadata}")
            if metadata.get("execution_health_recovery_closure_state") in {None, ""}:
                raise SystemExit(f"Evidence ledger missed recovery closure state: {metadata}")
            if "execution_health_recovery_closure_ready_to_retry" not in metadata:
                raise SystemExit(f"Evidence ledger missed recovery closure retry readiness: {metadata}")
            if "execution_health_recovery_closure_missing_count" not in metadata:
                raise SystemExit(f"Evidence ledger missed recovery closure missing count: {metadata}")
            closure_commands = metadata.get("execution_health_recovery_closure_required_commands") or []
            if metadata.get("execution_health_recovery_closure_blocks_completion_claim") and not closure_commands:
                raise SystemExit(f"Evidence ledger blocked completion without closure commands: {metadata}")
            if metadata.get("execution_health_recovery_closure_blocks_completion_claim") and metadata.get("execution_health_recovery_closure_target_run_id") is None:
                raise SystemExit(f"Evidence ledger missed recovery closure target run: {metadata}")
            if closure_commands and metadata.get("execution_health_recovery_closure_next_required_command") != closure_commands[0]:
                raise SystemExit(f"Evidence ledger missed first recovery proof command: {metadata}")
            assert_execution_health_recovery_proof_aliases(metadata, "Evidence ledger")
            if metadata.get("execution_learning_state") in {None, ""}:
                raise SystemExit(f"Evidence ledger missed execution learning state: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and metadata.get("execution_learning_missing_count", 0) < 1:
                raise SystemExit(f"Evidence ledger missed execution learning missing proof count: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and metadata.get("execution_learning_target_run_id") is None:
                raise SystemExit(f"Evidence ledger missed execution learning target run: {metadata}")
            learning_commands = metadata.get("execution_learning_required_commands") or []
            if metadata.get("execution_learning_blocks_completion_claim") and not any(str(command).startswith("after-action learning packet") for command in learning_commands):
                raise SystemExit(f"Evidence ledger missed after-action learning proof command: {metadata}")
            if learning_commands and metadata.get("execution_learning_next_required_command") != learning_commands[0]:
                raise SystemExit(f"Evidence ledger missed first learning proof command: {metadata}")
            assert_execution_learning_proof_aliases(metadata, "Evidence ledger")
            if metadata.get("latest_execution_case_found") is not False or metadata.get("latest_execution_case_verdict") != "CASE_NOT_FOUND":
                raise SystemExit(f"Evidence ledger missed no-case execution proof state: {metadata}")
            if metadata.get("latest_execution_case_closure_verdict") != "CASE_CLOSURE_NOT_FOUND" or metadata.get("latest_execution_case_closure_ready") is not False:
                raise SystemExit(f"Evidence ledger missed no-case closure state: {metadata}")
            if metadata.get("latest_execution_case_closure_blocks_completion_claim") is not True:
                raise SystemExit(f"Evidence ledger missed no-case closure completion block metadata: {metadata}")
            if metadata.get("latest_execution_case_next_closure_proof_command") != "save execution case: <next real order>":
                raise SystemExit(f"Evidence ledger missed no-case next closure command: {metadata}")
            if metadata.get("latest_execution_case_mission_command_count") != 0:
                raise SystemExit(f"Evidence ledger missed no-case mission queue count: {metadata}")
            next_proof_commands = metadata.get("next_proof_commands") or {}
            if next_proof_commands.get("steering") != "dispatch decision: <next real order>":
                raise SystemExit(f"Evidence ledger missed steering next proof command: {metadata}")
            if next_proof_commands.get("learning") != "learning review":
                raise SystemExit(f"Evidence ledger missed learning next proof command: {metadata}")
            assert_evidence_ledger_handoff(metadata, "Evidence ledger")
            if metadata.get("draft_only") is not True or metadata.get("requires_manual_send") is not True or metadata.get("loads_without_execution") is not True:
                raise SystemExit(f"Evidence ledger missed review-only contract: {metadata}")
            if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(f"Evidence ledger should not authorize execution or completion claims: {metadata}")
            if metadata.get("approval_granted") is not False:
                raise SystemExit(f"Evidence ledger should not grant approval: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Evidence ledger unsafe metadata {key}: {metadata}")

        claim_gate_cases = [
            "completion claim gate",
            "can Jarvis claim complete: Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant",
            "are we done with Jarvis",
            "is Jarvis finished",
            "did we finish Jarvis",
            "can you say Jarvis is complete",
            "should we mark Jarvis done",
            "can I claim Jarvis is finished",
            "should we call it done",
            "can this be marked complete",
            "completion claim please",
            "claim done please",
            "mark Jarvis done please",
        ]
        for case in claim_gate_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "completion_claim_gate":
                raise SystemExit(f"Expected '{case}' to route to completion_claim_gate.")
            for expected in [
                "Jarvis completion claim gate",
                "stop-check",
                "Evidence standard",
                "Requirement lanes",
                "Lane evidence details",
                "AGI-direction gate details",
                "real-execution gaps still tracked",
                "selected next build gate",
                "selected next build command",
                "AGI gate closure commands",
                "after_action_learning_packet",
                "execution_health_report",
                "available_without_recent_evidence",
                "next proof:",
                "approval readiness <id>",
                "verification packet: <next real order>",
                "Current proof state",
                "recent after-action learning packets",
                "recovery closure state",
                "recovery closure ready to retry",
                "recovery closure missing",
                "recovery next required",
                "recovery closure command queue",
                "execution learning debt state",
                "execution learning debt missing",
                "learning next required",
                "execution learning command queue",
                "Latest execution case proof state",
                "latest case: none saved",
                "case verdict: CASE_NOT_FOUND",
                "case closure verdict: CASE_CLOSURE_NOT_FOUND",
                "case closure ready: False",
                "case blocks completion claim: True",
                "mission command queue: 0 command(s)",
                "next case required command",
                "next case closure command",
                "Blockers",
                "Completion proof queue:",
                "Verdict: CLAIM_BLOCKED",
                "Safe claim:",
                "should not claim this is complete yet",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Completion claim gate missing expected text for '{case}': {expected}")
            if "next required proof" in result.response:
                raise SystemExit(f"Completion claim gate should not use ambiguous recovery next-required/proof prose for '{case}'.")
            if "next case proof command" in result.response:
                raise SystemExit(f"Completion claim gate should render next case required command for '{case}'.")
            if "next learning proof:" in result.response or "next learning evidence proof:" in result.response:
                raise SystemExit(f"Completion claim gate should use command-first learning next-required prose for '{case}'.")
            metadata = result.tool_results[0].metadata
            if metadata.get("allowed_to_claim") is not False or metadata.get("completion_claim_ready") is not False:
                raise SystemExit(f"Completion claim gate should block with setup blockers: {metadata}")
            if metadata.get("storage_runtime_fallback_active") is not False:
                raise SystemExit(f"Completion claim gate should report no storage fallback for temp runtime: {metadata}")
            if metadata.get("storage_readiness_blocks_completion_claim") is not False:
                raise SystemExit(f"Completion claim gate should not block on storage when primary temp storage is writable: {metadata}")
            if metadata.get("storage_readiness_next_commands") != [] or metadata.get("storage_readiness_next_command_count") != 0:
                raise SystemExit(f"Completion claim gate should expose an empty storage recovery queue when fallback is inactive: {metadata}")
            if metadata.get("blockers", 0) < 1 or metadata.get("pending_approvals", 0) < 1 or metadata.get("open_tasks", 0) < 1:
                raise SystemExit(f"Completion claim gate missed blockers: {metadata}")
            blocker_details = metadata.get("blocker_details", [])
            if len(blocker_details) != len(set(blocker_details)) or metadata.get("completion_blockers_deduplicated") is not True:
                raise SystemExit(f"Completion claim gate should deduplicate blocker details: {metadata}")
            proof_queue = metadata.get("completion_proof_queue") or []
            if not proof_queue or metadata.get("completion_proof_queue_count") != len(proof_queue):
                raise SystemExit(f"Completion claim gate missed completion proof queue metadata: {metadata}")
            if metadata.get("completion_next_proof_command") != proof_queue[0] or metadata.get("next_completion_proof_command") != proof_queue[0]:
                raise SystemExit(f"Completion claim gate missed first completion proof command: {metadata}")
            if metadata.get("pending_approvals"):
                approval_indexes = [
                    index
                    for index, command in enumerate(proof_queue)
                    if str(command).startswith("approval readiness ")
                ]
                approval_verification_indexes = [
                    index
                    for index, command in enumerate(proof_queue)
                    if str(command).startswith("verification receipt <approved run id from approval chain proof ")
                ]
                learning_indexes = [
                    index
                    for index, command in enumerate(proof_queue)
                    if str(command).startswith("after-action learning packet")
                ]
                if not approval_verification_indexes:
                    raise SystemExit(f"Completion claim gate proof queue missed approval-chain verification placeholder: {metadata}")
                if approval_indexes and learning_indexes and min(approval_indexes) > min(learning_indexes):
                    raise SystemExit(f"Completion claim gate should put approval review before learning evidence: {metadata}")
                if approval_indexes and min(approval_verification_indexes) < min(approval_indexes):
                    raise SystemExit(f"Completion claim gate should put approval verification after approval review: {metadata}")
                if learning_indexes and min(approval_verification_indexes) > min(learning_indexes):
                    raise SystemExit(f"Completion claim gate should complete approval proof before learning evidence: {metadata}")
            if metadata.get("agi_gates") != 6 or metadata.get("agi_real_execution_gap_count") != 6:
                raise SystemExit(f"Completion claim gate missed AGI gate metadata: {metadata}")
            claim_closure_by_gate = metadata.get("agi_evidence_closure_commands_by_gate") or {}
            if claim_closure_by_gate.get("personal integrations") != expected_personal_closure:
                raise SystemExit(f"Completion claim gate missed AGI closure commands: {metadata}")
            if not metadata.get("agi_next_moves_by_gate") or "personal integrations" not in metadata.get("agi_next_moves_by_gate", {}):
                raise SystemExit(f"Completion claim gate missed AGI next moves: {metadata}")
            if not metadata.get("agi_next_gate") or not metadata.get("agi_next_build_command"):
                raise SystemExit(f"Completion claim gate missed selected AGI next build target: {metadata}")
            if metadata.get("agi_next_evidence_closure_command_count") != len(metadata.get("agi_next_evidence_closure_commands", [])):
                raise SystemExit(f"Completion claim gate missed selected AGI closure command count: {metadata}")
            assert_selected_agi_target_readiness(metadata, "Completion claim gate")
            preflight_blockers = metadata.get("agi_next_implementation_preflight_blockers") or []
            if metadata.get("pending_approvals", 0) and "pending approvals must be reviewed or dismissed before starting a new AGI build slice" not in preflight_blockers:
                raise SystemExit(f"Completion claim gate missed pending-approval AGI implementation preflight blocker: {metadata}")
            if metadata.get("execution_health_recovery_closure_blocks_completion_claim") and "execution recovery closure proof debt must be closed first" not in preflight_blockers:
                raise SystemExit(f"Completion claim gate missed recovery AGI implementation preflight blocker: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and "execution learning proof debt must be closed first" not in preflight_blockers:
                raise SystemExit(f"Completion claim gate missed learning AGI implementation preflight blocker: {metadata}")
            if preflight_blockers and metadata.get("agi_next_implementation_preflight_ready") is not False:
                raise SystemExit(f"Completion claim gate should block AGI implementation preflight while blockers exist: {metadata}")
            if preflight_blockers and not metadata.get("agi_next_implementation_preflight_next_command"):
                raise SystemExit(f"Completion claim gate missed AGI implementation preflight next command: {metadata}")
            assert_completion_claim_handoff(metadata, "Completion claim gate")
            if metadata.get("agi_next_build_command") not in proof_queue:
                raise SystemExit(f"Completion claim gate proof queue missed selected AGI build command: {metadata}")
            if metadata.get("verdict") != "CLAIM_BLOCKED":
                raise SystemExit(f"Completion claim gate missed verdict: {metadata}")
            lane_status = metadata.get("lane_status") or {}
            if not lane_status or "command-first interface" not in lane_status:
                raise SystemExit(f"Completion claim gate missed lane status metadata: {metadata}")
            if not any(status == "available_without_recent_evidence" for status in lane_status.values()):
                raise SystemExit(f"Completion claim gate should distinguish available tools from recent evidence: {metadata}")
            if not metadata.get("lane_available_tools") or metadata.get("lane_evidence") is None:
                raise SystemExit(f"Completion claim gate missed lane evidence/tool metadata: {metadata}")
            next_proof_commands = metadata.get("next_proof_commands") or {}
            if next_proof_commands.get("approval gates and audit") != "approval readiness <id>":
                raise SystemExit(f"Completion claim gate missed approval next proof command: {metadata}")
            if next_proof_commands.get("verification and recovery") != "verification packet: <next real order>":
                raise SystemExit(f"Completion claim gate missed verification next proof command: {metadata}")
            if "after_action_learning_runs" not in metadata:
                raise SystemExit(f"Completion claim gate missed after-action learning run count: {metadata}")
            if metadata.get("execution_health_recovery_closure_state") in {None, ""}:
                raise SystemExit(f"Completion claim gate missed recovery closure state: {metadata}")
            if "execution_health_recovery_closure_ready_to_retry" not in metadata:
                raise SystemExit(f"Completion claim gate missed recovery closure retry readiness: {metadata}")
            if "execution_health_recovery_closure_missing_count" not in metadata:
                raise SystemExit(f"Completion claim gate missed recovery closure missing count: {metadata}")
            closure_commands = metadata.get("execution_health_recovery_closure_required_commands") or []
            if metadata.get("execution_health_recovery_closure_blocks_completion_claim") and not closure_commands:
                raise SystemExit(f"Completion claim gate blocked completion without closure commands: {metadata}")
            if metadata.get("execution_health_recovery_closure_blocks_completion_claim") and metadata.get("execution_health_recovery_closure_target_run_id") is None:
                raise SystemExit(f"Completion claim gate missed recovery closure target run: {metadata}")
            if closure_commands and metadata.get("execution_health_recovery_closure_next_required_command") != closure_commands[0]:
                raise SystemExit(f"Completion claim gate missed first recovery proof command: {metadata}")
            assert_execution_health_recovery_proof_aliases(metadata, "Completion claim gate")
            for expected_command in closure_commands:
                if expected_command not in proof_queue:
                    raise SystemExit(f"Completion claim gate proof queue missed recovery command {expected_command}: {metadata}")
            if metadata.get("execution_learning_state") in {None, ""}:
                raise SystemExit(f"Completion claim gate missed execution learning state: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and metadata.get("execution_learning_missing_count", 0) < 1:
                raise SystemExit(f"Completion claim gate missed execution learning missing proof count: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and metadata.get("execution_learning_target_run_id") is None:
                raise SystemExit(f"Completion claim gate missed execution learning target run: {metadata}")
            learning_commands = metadata.get("execution_learning_required_commands") or []
            if metadata.get("execution_learning_blocks_completion_claim") and not any(str(command).startswith("after-action learning packet") for command in learning_commands):
                raise SystemExit(f"Completion claim gate missed after-action learning proof command: {metadata}")
            learning_closure_command = metadata.get("execution_learning_closure_command")
            if metadata.get("execution_learning_blocks_completion_claim") and (not learning_closure_command or not str(learning_closure_command).startswith("execution learning closure")):
                raise SystemExit(f"Completion claim gate missed execution learning closure command: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and learning_closure_command not in proof_queue:
                raise SystemExit(f"Completion claim gate proof queue missed learning closure command {learning_closure_command}: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim"):
                after_action_command = metadata.get("execution_learning_after_action_learning_command")
                if after_action_command not in proof_queue:
                    raise SystemExit(f"Completion claim gate proof queue missed after-action learning evidence command {after_action_command}: {metadata}")
                if learning_closure_command in proof_queue and proof_queue.index(after_action_command) > proof_queue.index(learning_closure_command):
                    raise SystemExit(f"Completion claim gate proof queue should put after-action evidence before learning closure: {metadata}")
                for expected_text in [
                    "learning evidence next required:",
                    "learning actionable next required:",
                    "learning actionable proof alias:",
                    "execution learning closure command:",
                ]:
                    if expected_text not in result.response:
                        raise SystemExit(f"Completion claim gate missed learning prose label {expected_text}: {result.response}")
                evidence_label = result.response.find("learning evidence next required:")
                actionable_label = result.response.find("learning actionable next required:")
                closure_label = result.response.find("execution learning closure command:")
                if not (evidence_label < closure_label and actionable_label < closure_label):
                    raise SystemExit(f"Completion claim gate should show actionable learning evidence before closure: {result.response}")
            if learning_commands and metadata.get("execution_learning_next_required_command") != learning_commands[0]:
                raise SystemExit(f"Completion claim gate missed first learning proof command: {metadata}")
            actionable_learning_commands = metadata.get("execution_learning_actionable_required_commands") or []
            if metadata.get("execution_learning_blocks_completion_claim"):
                expected_after_action = metadata.get("execution_learning_after_action_learning_command")
                if not actionable_learning_commands or expected_after_action not in actionable_learning_commands:
                    raise SystemExit(f"Completion claim gate missed actionable after-action learning command: {metadata}")
                if learning_closure_command in actionable_learning_commands and actionable_learning_commands.index(expected_after_action) > actionable_learning_commands.index(learning_closure_command):
                    raise SystemExit(f"Completion claim gate should put after-action evidence before closure recheck: {metadata}")
                if metadata.get("execution_learning_next_evidence_command") != actionable_learning_commands[0]:
                    raise SystemExit(f"Completion claim gate missed next learning evidence command alias: {metadata}")
            assert_execution_learning_proof_aliases(metadata, "Completion claim gate")
            for expected_command in learning_commands:
                if expected_command not in proof_queue:
                    raise SystemExit(f"Completion claim gate proof queue missed learning command {expected_command}: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and not any("execution learning debt" in blocker for blocker in metadata.get("blocker_details", [])):
                raise SystemExit(f"Completion claim gate missed execution learning debt blocker: {metadata}")
            if metadata.get("latest_execution_case_found") is not False or metadata.get("latest_execution_case_verdict") != "CASE_NOT_FOUND":
                raise SystemExit(f"Completion claim gate missed no-case execution proof state: {metadata}")
            if metadata.get("latest_execution_case_closure_verdict") != "CASE_CLOSURE_NOT_FOUND" or metadata.get("latest_execution_case_closure_ready") is not False:
                raise SystemExit(f"Completion claim gate missed no-case closure state: {metadata}")
            if metadata.get("latest_execution_case_closure_blocks_completion_claim") is not True:
                raise SystemExit(f"Completion claim gate missed no-case closure completion block metadata: {metadata}")
            if metadata.get("latest_execution_case_next_closure_proof_command") != "save execution case: <next real order>":
                raise SystemExit(f"Completion claim gate missed no-case next closure command: {metadata}")
            if metadata.get("latest_execution_case_mission_command_count") != 0:
                raise SystemExit(f"Completion claim gate missed no-case mission queue count: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Completion claim gate unsafe metadata {key}: {metadata}")

        fallback_root = Path(temp) / "storage-fallback-runtime"
        fallback_root.mkdir()
        fallback_runtime = make_temp_runtime(fallback_root)
        fallback_runtime.storage_fallback = {
            "reason": "primary_storage_not_writable",
            "exception_type": "OperationalError",
            "db_path_display": "workspace-local fallback database",
            "vault_path_display": "workspace-local fallback notes",
        }
        fallback_completion = fallback_runtime.handle("harness completion")
        print(f"[{'ok' if fallback_completion.verified else 'blocked'}] harness completion with storage fallback")
        print(fallback_completion.response[:2200])
        print()
        if not fallback_completion.verified:
            raise SystemExit("Expected harness completion with storage fallback to run read-only.")
        fallback_completion_metadata = fallback_completion.tool_results[0].metadata
        for expected in [
            "Storage readiness",
            "runtime fallback active: yes",
            "blocks completion claim: yes",
            "workspace-local fallback database",
            "restart or reload Jarvis with the configured durable storage envs",
        ]:
            if expected not in fallback_completion.response:
                raise SystemExit(f"Harness completion fallback case missed text {expected}: {fallback_completion.response}")
        if fallback_completion_metadata.get("storage_runtime_fallback_active") is not True:
            raise SystemExit(f"Harness completion missed active storage fallback metadata: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("storage_readiness_blocks_completion_claim") is not True:
            raise SystemExit(f"Harness completion should block completion on storage fallback: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("storage_recovery_required") is not True:
            raise SystemExit(f"Harness completion missed explicit storage recovery requirement: {fallback_completion_metadata}")
        if "workspace-local fallback storage" not in str(fallback_completion_metadata.get("storage_recovery_reason") or ""):
            raise SystemExit(f"Harness completion missed explicit storage recovery reason: {fallback_completion_metadata}")
        assert_fallback_storage_recovery_mode(fallback_completion_metadata, "Harness completion")
        if fallback_completion_metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
            raise SystemExit(f"Harness completion missed explicit storage recovery check command: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
            raise SystemExit(f"Harness completion missed native storage recovery check command: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("storage_recovery_check_api") != "/api/storage-recovery-check":
            raise SystemExit(f"Harness completion missed read-only storage recovery check API: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
            raise SystemExit(f"Harness completion missed explicit storage recovery command: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("storage_runtime_fallback_reason") != "primary_storage_not_writable":
            raise SystemExit(f"Harness completion missed fallback reason: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("storage_runtime_fallback_exception_type") != "OperationalError":
            raise SystemExit(f"Harness completion missed fallback exception type: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("storage_runtime_fallback_db_path_display") != "workspace-local fallback database":
            raise SystemExit(f"Harness completion leaked/missed fallback database display: {fallback_completion_metadata}")
        fallback_commands = fallback_completion_metadata.get("storage_readiness_next_commands") or []
        if fallback_commands != EXPECTED_RESTART_ONLY_STORAGE_COMMANDS:
            raise SystemExit(f"Harness completion missed ordered storage recovery commands: {fallback_completion_metadata}")
        if fallback_commands[1] != fallback_completion_metadata.get("storage_recovery_check_tool_command"):
            raise SystemExit(f"Harness completion storage queue/native-check alias diverged: {fallback_completion_metadata}")
        if BOOTSTRAP_CHECK_COMMAND in fallback_commands or BOOTSTRAP_WRITE_COMMAND in fallback_commands:
            raise SystemExit(f"Harness completion restart-only storage queue should not require bootstrap commands: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("storage_readiness_next_command_count") != len(fallback_commands):
            raise SystemExit(f"Harness completion storage command count diverged: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("storage_readiness_next_required_command") != fallback_commands[0]:
            raise SystemExit(f"Harness completion missed storage next required alias: {fallback_completion_metadata}")
        if (fallback_completion_metadata.get("completion_proof_queue") or [])[: len(fallback_commands)] != fallback_commands:
            raise SystemExit(f"Harness completion should put storage recovery first in proof queue: {fallback_completion_metadata}")
        if fallback_completion_metadata.get("completion_next_proof_command") != fallback_commands[0]:
            raise SystemExit(f"Harness completion should make storage status the next proof command: {fallback_completion_metadata}")
        if not any("runtime is using workspace-local fallback storage" in blocker for blocker in fallback_completion_metadata.get("completion_blockers", [])):
            raise SystemExit(f"Harness completion missed storage blocker detail: {fallback_completion_metadata}")
        assert_harness_completion_handoff(fallback_completion_metadata, "Harness completion fallback")

        fallback_claim = fallback_runtime.handle("completion claim gate")
        print(f"[{'ok' if fallback_claim.verified else 'blocked'}] completion claim gate with storage fallback")
        print(fallback_claim.response[:2200])
        print()
        if not fallback_claim.verified:
            raise SystemExit("Expected completion claim gate with storage fallback to run read-only.")
        fallback_claim_metadata = fallback_claim.tool_results[0].metadata
        for expected in [
            "storage fallback active: yes",
            "storage readiness blocks completion claim: yes",
            "Restore durable primary storage before claiming completion",
            "storage status",
            "restart or reload Jarvis with the configured durable storage envs",
        ]:
            if expected not in fallback_claim.response:
                raise SystemExit(f"Completion claim fallback case missed text {expected}: {fallback_claim.response}")
        if fallback_claim_metadata.get("storage_runtime_fallback_active") is not True:
            raise SystemExit(f"Completion claim gate missed active storage fallback metadata: {fallback_claim_metadata}")
        if fallback_claim_metadata.get("storage_readiness_blocks_completion_claim") is not True:
            raise SystemExit(f"Completion claim gate should block completion on storage fallback: {fallback_claim_metadata}")
        if fallback_claim_metadata.get("storage_recovery_required") is not True:
            raise SystemExit(f"Completion claim gate missed explicit storage recovery requirement: {fallback_claim_metadata}")
        if fallback_claim_metadata.get("storage_recovery_reason") != fallback_completion_metadata.get("storage_recovery_reason"):
            raise SystemExit(f"Completion claim gate storage recovery reason diverged from harness completion: {fallback_claim_metadata}")
        assert_fallback_storage_recovery_mode(fallback_claim_metadata, "Completion claim gate")
        if fallback_claim_metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
            raise SystemExit(f"Completion claim gate storage recovery check command diverged: {fallback_claim_metadata}")
        if fallback_claim_metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
            raise SystemExit(f"Completion claim gate native storage recovery check command diverged: {fallback_claim_metadata}")
        if fallback_claim_metadata.get("storage_recovery_check_api") != "/api/storage-recovery-check":
            raise SystemExit(f"Completion claim gate read-only storage recovery check API diverged: {fallback_claim_metadata}")
        if fallback_claim_metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
            raise SystemExit(f"Completion claim gate storage recovery command diverged: {fallback_claim_metadata}")
        if fallback_claim_metadata.get("storage_readiness_next_commands") != fallback_commands:
            raise SystemExit(f"Completion claim gate storage recovery queue diverged: {fallback_claim_metadata}")
        if fallback_claim_metadata.get("storage_readiness_next_required_command") != fallback_commands[0]:
            raise SystemExit(f"Completion claim gate storage next required alias diverged: {fallback_claim_metadata}")
        fallback_proof_queue = fallback_claim_metadata.get("completion_proof_queue") or []
        if fallback_proof_queue[: len(fallback_commands)] != fallback_commands:
            raise SystemExit(f"Completion claim gate should put storage recovery first in proof queue: {fallback_claim_metadata}")
        for key in ["completion_next_proof_command", "next_completion_proof_command", "next_proof_command"]:
            if fallback_claim_metadata.get(key) != fallback_commands[0]:
                raise SystemExit(f"Completion claim gate should make storage status the first proof alias {key}: {fallback_claim_metadata}")
        if not any("runtime is using workspace-local fallback storage" in blocker for blocker in fallback_claim_metadata.get("blocker_details", [])):
            raise SystemExit(f"Completion claim gate missed storage blocker detail: {fallback_claim_metadata}")
        assert_execution_health_verification_proof_aliases(fallback_claim_metadata, "Completion claim gate fallback")
        assert_completion_claim_handoff(fallback_claim_metadata, "Completion claim gate fallback")

        fallback_next_proof = fallback_runtime.handle("completion next proof")
        print(f"[{'ok' if fallback_next_proof.verified else 'blocked'}] completion next proof with storage fallback")
        print(fallback_next_proof.response[:2200])
        print()
        if not fallback_next_proof.verified:
            raise SystemExit("Expected completion next proof with storage fallback to run read-only.")
        fallback_next_metadata = fallback_next_proof.tool_results[0].metadata
        if fallback_next_metadata.get("next_proof_command") != fallback_commands[0]:
            raise SystemExit(f"Completion next proof fallback should make storage status first: {fallback_next_metadata}")
        if "durable primary storage" not in str(fallback_next_metadata.get("command_reason") or ""):
            raise SystemExit(f"Completion next proof fallback should explain storage-specific command reason: {fallback_next_metadata}")
        if fallback_next_metadata.get("storage_readiness_next_commands") != fallback_commands:
            raise SystemExit(f"Completion next proof fallback missed storage recovery commands: {fallback_next_metadata}")
        if (fallback_next_metadata.get("completion_proof_queue") or [])[: len(fallback_commands)] != fallback_commands:
            raise SystemExit(f"Completion next proof fallback should preserve full storage proof queue prefix: {fallback_next_metadata}")
        if (fallback_next_metadata.get("proof_queue") or [])[: len(fallback_commands)] != fallback_commands:
            raise SystemExit(f"Completion next proof fallback should preserve full storage prefix in actionable proof queue: {fallback_next_metadata}")
        if (fallback_next_metadata.get("completion_actionable_proof_queue") or [])[: len(fallback_commands)] != fallback_commands:
            raise SystemExit(f"Completion next proof fallback should preserve full storage prefix in completion_actionable_proof_queue: {fallback_next_metadata}")
        if fallback_next_metadata.get("storage_readiness_next_proof_command") != fallback_commands[0]:
            raise SystemExit(f"Completion next proof fallback missed storage next proof alias: {fallback_next_metadata}")
        if fallback_next_metadata.get("storage_readiness_next_required_command") != fallback_commands[0]:
            raise SystemExit(f"Completion next proof fallback missed storage next required alias: {fallback_next_metadata}")
        if fallback_next_metadata.get("storage_recovery_required") is not True:
            raise SystemExit(f"Completion next proof fallback missed explicit storage recovery requirement: {fallback_next_metadata}")
        if fallback_next_metadata.get("storage_recovery_reason") != fallback_completion_metadata.get("storage_recovery_reason"):
            raise SystemExit(f"Completion next proof fallback storage recovery reason diverged: {fallback_next_metadata}")
        assert_fallback_storage_recovery_mode(fallback_next_metadata, "Completion next proof")
        if fallback_next_metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
            raise SystemExit(f"Completion next proof fallback storage recovery check command diverged: {fallback_next_metadata}")
        if fallback_next_metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
            raise SystemExit(f"Completion next proof fallback native storage recovery check command diverged: {fallback_next_metadata}")
        if fallback_next_metadata.get("storage_recovery_check_api") != "/api/storage-recovery-check":
            raise SystemExit(f"Completion next proof fallback read-only storage recovery check API diverged: {fallback_next_metadata}")
        if fallback_next_metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
            raise SystemExit(f"Completion next proof fallback storage recovery command diverged: {fallback_next_metadata}")
        fallback_handoff = fallback_runtime.handle("operator handoff")
        print(f"[{'ok' if fallback_handoff.verified else 'blocked'}] operator handoff with storage fallback")
        print(fallback_handoff.response[:2200])
        print()
        if not fallback_handoff.verified:
            raise SystemExit("Expected operator handoff with storage fallback to run read-only.")
        fallback_handoff_metadata = fallback_handoff.tool_results[0].metadata
        if fallback_handoff_metadata.get("next_operator_command") != fallback_commands[0]:
            raise SystemExit(f"Operator handoff fallback should make storage status first: {fallback_handoff_metadata}")
        if fallback_handoff_metadata.get("command_reason") != fallback_next_metadata.get("command_reason"):
            raise SystemExit(f"Operator handoff fallback should mirror storage command reason: {fallback_handoff_metadata}")
        if fallback_handoff_metadata.get("storage_recovery_required") is not True:
            raise SystemExit(f"Operator handoff fallback missed explicit storage recovery requirement: {fallback_handoff_metadata}")
        if fallback_handoff_metadata.get("storage_recovery_reason") != fallback_completion_metadata.get("storage_recovery_reason"):
            raise SystemExit(f"Operator handoff fallback storage recovery reason diverged: {fallback_handoff_metadata}")
        assert_fallback_storage_recovery_mode(fallback_handoff_metadata, "Operator handoff")
        if fallback_handoff_metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
            raise SystemExit(f"Operator handoff fallback storage recovery check command diverged: {fallback_handoff_metadata}")
        if fallback_handoff_metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
            raise SystemExit(f"Operator handoff fallback native storage recovery check command diverged: {fallback_handoff_metadata}")
        if fallback_handoff_metadata.get("storage_recovery_check_api") != "/api/storage-recovery-check":
            raise SystemExit(f"Operator handoff fallback read-only storage recovery check API diverged: {fallback_handoff_metadata}")
        if fallback_handoff_metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
            raise SystemExit(f"Operator handoff fallback storage recovery command diverged: {fallback_handoff_metadata}")
        if fallback_handoff_metadata.get("storage_readiness_next_required_command") != fallback_commands[0]:
            raise SystemExit(f"Operator handoff fallback missed storage next required alias: {fallback_handoff_metadata}")

        fallback_repair_root = Path(temp) / "storage-fallback-primary-diagnostics"
        fallback_repair_root.mkdir()
        fallback_repair_runtime = make_temp_runtime(fallback_repair_root)
        fallback_repair_runtime.storage_fallback = {
            "reason": "primary_storage_not_writable",
            "exception_type": "OperationalError",
            "db_path_display": "workspace-local fallback database",
            "vault_path_display": "workspace-local fallback notes",
            "primary_storage_diagnostics": {
                "available": False,
                "status": "needs attention",
                "issues": [
                    "database parent is not writable",
                    "database file is not writable",
                    "Obsidian vault is not writable",
                ],
                "metadata_only": True,
            },
        }
        repair_storage_commands = [
            "storage status",
            STORAGE_RECOVERY_CHECK_COMMAND,
            BOOTSTRAP_CHECK_COMMAND,
            BOOTSTRAP_WRITE_COMMAND,
        ]
        fallback_repair_completion = fallback_repair_runtime.handle("harness completion")
        if not fallback_repair_completion.verified:
            raise SystemExit("Expected harness completion with preserved primary diagnostics to run read-only.")
        repair_completion_metadata = fallback_repair_completion.tool_results[0].metadata
        expected_repair_storage_issues = [
            "database parent is not writable",
            "database file is not writable",
            "Obsidian vault is not writable",
            "runtime is using workspace-local fallback storage",
        ]
        if repair_completion_metadata.get("storage_recovery_mode") != "repair_configured_storage_then_restart_runtime":
            raise SystemExit(f"Harness completion should use repair mode when preserved primary storage is not ready: {repair_completion_metadata}")
        if repair_completion_metadata.get("storage_issues") != expected_repair_storage_issues:
            raise SystemExit(f"Harness completion should preserve configured storage issues under fallback: {repair_completion_metadata}")
        if repair_completion_metadata.get("storage_issue_count") != len(expected_repair_storage_issues):
            raise SystemExit(f"Harness completion storage issue count diverged: {repair_completion_metadata}")
        for expected_issue in expected_repair_storage_issues:
            if expected_issue not in str(repair_completion_metadata.get("storage_recovery_reason") or ""):
                raise SystemExit(f"Harness completion recovery reason missed storage issue {expected_issue}: {repair_completion_metadata}")
        if repair_completion_metadata.get("storage_readiness_next_commands") != repair_storage_commands:
            raise SystemExit(f"Harness completion repair-mode storage queue diverged: {repair_completion_metadata}")
        if repair_completion_metadata.get("storage_readiness_next_required_command") != repair_storage_commands[0]:
            raise SystemExit(f"Harness completion repair-mode storage next required alias diverged: {repair_completion_metadata}")
        if repair_completion_metadata.get("storage_recovery_next_operator_action") != "point Jarvis at writable durable storage, run the no-write storage check, then restart or reload Jarvis":
            raise SystemExit(f"Harness completion repair-mode operator action drifted: {repair_completion_metadata}")
        if not all(command in (repair_completion_metadata.get("completion_proof_queue") or []) for command in repair_storage_commands):
            raise SystemExit(f"Harness completion repair-mode proof queue missed storage commands: {repair_completion_metadata}")
        fallback_repair_digest = fallback_repair_runtime.handle("harness readiness digest")
        if not fallback_repair_digest.verified:
            raise SystemExit("Expected harness readiness digest with preserved primary diagnostics to run read-only.")
        repair_digest_metadata = fallback_repair_digest.tool_results[0].metadata
        if repair_digest_metadata.get("storage_recovery_mode") != "repair_configured_storage_then_restart_runtime":
            raise SystemExit(f"Harness readiness digest should preserve primary storage repair mode: {repair_digest_metadata}")
        if repair_digest_metadata.get("storage_issues") != expected_repair_storage_issues:
            raise SystemExit(f"Harness readiness digest should preserve configured storage issues under fallback: {repair_digest_metadata}")
        if repair_digest_metadata.get("storage_issue_count") != len(expected_repair_storage_issues):
            raise SystemExit(f"Harness readiness digest storage issue count diverged: {repair_digest_metadata}")
        if repair_digest_metadata.get("storage_readiness_next_commands") != repair_storage_commands:
            raise SystemExit(f"Harness readiness digest repair-mode storage queue diverged: {repair_digest_metadata}")
        if repair_digest_metadata.get("storage_readiness_next_required_command") != repair_storage_commands[0]:
            raise SystemExit(f"Harness readiness digest repair-mode storage next required alias diverged: {repair_digest_metadata}")
        if repair_digest_metadata.get("command_ladder_storage_prefix") != repair_storage_commands:
            raise SystemExit(f"Harness readiness digest command ladder should start with repair-mode storage queue: {repair_digest_metadata}")

        unhealthy_config_root = Path(temp) / "unhealthy-configured-storage"
        unhealthy_config_root.mkdir()
        unhealthy_runtime = make_temp_runtime(unhealthy_config_root)
        unhealthy_config = JarvisConfig(
            data_dir=unhealthy_config_root / "missing-data" / "child",
            db_path=unhealthy_config_root / "missing-data" / "child" / "jarvis.sqlite",
            obsidian_vault=unhealthy_config_root / "missing-vault" / "child",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        unhealthy_tools = make_harness_tools(
            unhealthy_runtime.store,
            unhealthy_runtime.vault,
            lambda: [],
            None,
            unhealthy_config,
        )
        unhealthy_completion = unhealthy_tools[8]({})
        if not unhealthy_completion.ok:
            raise SystemExit(f"Expected unhealthy configured storage harness completion to run read-only: {unhealthy_completion}")
        unhealthy_metadata = unhealthy_completion.metadata
        unhealthy_storage_commands = [
            "storage status",
            STORAGE_RECOVERY_CHECK_COMMAND,
            BOOTSTRAP_CHECK_COMMAND,
            BOOTSTRAP_WRITE_COMMAND,
        ]
        if unhealthy_metadata.get("storage_runtime_fallback_active") is not False:
            raise SystemExit(f"Harness completion should not report fallback for unhealthy configured storage: {unhealthy_metadata}")
        if unhealthy_metadata.get("storage_readiness_blocks_completion_claim") is not True:
            raise SystemExit(f"Harness completion should block unhealthy configured storage: {unhealthy_metadata}")
        if unhealthy_metadata.get("storage_recovery_required") is not True:
            raise SystemExit(f"Harness completion missed configured storage recovery requirement: {unhealthy_metadata}")
        if unhealthy_metadata.get("storage_recovery_mode") != "repair_configured_storage":
            raise SystemExit(f"Harness completion should use configured-storage repair mode: {unhealthy_metadata}")
        if unhealthy_metadata.get("storage_recovery_restart_required") is not False:
            raise SystemExit(f"Harness completion should not require restart before configured storage repair: {unhealthy_metadata}")
        if "writable durable storage" not in unhealthy_metadata.get("storage_recovery_next_operator_action", ""):
            raise SystemExit(f"Harness completion missed configured-storage operator action: {unhealthy_metadata}")
        if unhealthy_metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
            raise SystemExit(f"Harness completion missed native configured-storage check command: {unhealthy_metadata}")
        if unhealthy_metadata.get("storage_readiness_next_commands") != unhealthy_storage_commands:
            raise SystemExit(f"Harness completion configured-storage queue diverged: {unhealthy_metadata}")
        if unhealthy_metadata.get("storage_readiness_next_required_command") != unhealthy_storage_commands[0]:
            raise SystemExit(f"Harness completion configured-storage next required alias diverged: {unhealthy_metadata}")
        if (unhealthy_metadata.get("completion_proof_queue") or [])[: len(unhealthy_storage_commands)] != unhealthy_storage_commands:
            raise SystemExit(f"Harness completion should put configured-storage recovery first in proof queue: {unhealthy_metadata}")
        if unhealthy_metadata.get("completion_next_proof_command") != "storage status":
            raise SystemExit(f"Harness completion should make storage status the first configured-storage proof: {unhealthy_metadata}")

        next_proof_cases = [
            "completion next proof",
            "next proof command",
            "what proof should Jarvis do next",
            "next proof please",
            "what evidence should we gather next",
            "what verification should we run next",
            "what should we verify next for completion",
        ]
        for case in next_proof_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "completion_next_proof_packet":
                raise SystemExit(f"Expected '{case}' to route to completion_next_proof_packet.")
            for expected in [
                "Jarvis completion next proof packet",
                "Next required command:",
                "Command source:",
                "Execution case closure:",
                "Recovery and learning queues:",
                "learning actionable next required:",
                "learning actionable proof alias:",
                "learning closure command:",
                "Actionable completion proof queue:",
                "Ordered completion proof queue:",
                "After completing the next required command",
                "rerun `completion next proof`",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Completion next proof packet missed expected text for '{case}': {expected}")
            if "After completing the next proof command" in result.response:
                raise SystemExit(f"Completion next proof packet should use next required command follow-up prose for '{case}'.")
            if "Next proof command:" in result.response:
                raise SystemExit(f"Completion next proof packet should render next required command for '{case}'.")
            metadata = result.tool_results[0].metadata
            proof_queue = metadata.get("completion_proof_queue") or []
            if not proof_queue or metadata.get("completion_proof_queue_count") != len(proof_queue):
                raise SystemExit(f"Completion next proof packet missed proof queue metadata: {metadata}")
            actionable_queue = metadata.get("completion_actionable_proof_queue") or []
            if not actionable_queue or metadata.get("completion_actionable_proof_queue_count") != len(actionable_queue):
                raise SystemExit(f"Completion next proof packet missed actionable proof queue metadata: {metadata}")
            if metadata.get("proof_queue") != actionable_queue or metadata.get("proof_queue_count") != len(actionable_queue):
                raise SystemExit(f"Completion next proof packet missed generic actionable proof queue aliases: {metadata}")
            if metadata.get("completion_ordered_proof_queue") != proof_queue or metadata.get("completion_ordered_proof_queue_count") != len(proof_queue):
                raise SystemExit(f"Completion next proof packet missed ordered proof queue aliases: {metadata}")
            if metadata.get("completion_ordered_next_proof_command") != proof_queue[0]:
                raise SystemExit(f"Completion next proof packet missed ordered first proof command: {metadata}")
            if metadata.get("next_proof_command") != actionable_queue[0] or metadata.get("next_completion_proof_command") != actionable_queue[0] or metadata.get("completion_next_proof_command") != actionable_queue[0]:
                raise SystemExit(f"Completion next proof packet missed first proof command aliases: {metadata}")
            if metadata.get("completion_claim_ready") is not False or metadata.get("claim_gate_verdict") != "CLAIM_BLOCKED":
                raise SystemExit(f"Completion next proof packet should inherit blocked claim gate: {metadata}")
            if metadata.get("latest_execution_case_closure_verdict") != "CASE_CLOSURE_NOT_FOUND":
                raise SystemExit(f"Completion next proof packet missed latest case closure state: {metadata}")
            approval_verification_commands = [
                command
                for command in actionable_queue
                if str(command).startswith("verification receipt <approved run id from approval chain proof ")
            ]
            if metadata.get("pending_approvals") and not approval_verification_commands:
                raise SystemExit(f"Completion next proof packet missed approval-chain verification placeholder: {metadata}")
            if metadata.get("pending_approvals"):
                approval_indexes = [
                    index
                    for index, command in enumerate(actionable_queue)
                    if str(command).startswith("approval readiness ")
                ]
                learning_indexes = [
                    index
                    for index, command in enumerate(actionable_queue)
                    if str(command).startswith("after-action learning packet")
                ]
                verification_indexes = [
                    index
                    for index, command in enumerate(actionable_queue)
                    if str(command).startswith("verification receipt <approved run id from approval chain proof ")
                ]
                if approval_indexes and learning_indexes and min(approval_indexes) > min(learning_indexes):
                    raise SystemExit(f"Completion next proof packet should put approval review before learning evidence: {metadata}")
                if approval_indexes and verification_indexes and min(verification_indexes) < min(approval_indexes):
                    raise SystemExit(f"Completion next proof packet should put approval verification after approval review: {metadata}")
                if verification_indexes and learning_indexes and min(verification_indexes) > min(learning_indexes):
                    raise SystemExit(f"Completion next proof packet should complete approval proof before learning evidence: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and (not metadata.get("execution_learning_closure_command") or metadata.get("execution_learning_closure_command") not in proof_queue):
                raise SystemExit(f"Completion next proof packet missed learning closure command in proof queue: {metadata}")
            learning_actionable_label = result.response.find("learning actionable next required:")
            learning_closure_label = result.response.find("learning closure command:")
            if learning_actionable_label < 0 or learning_closure_label < 0 or learning_actionable_label > learning_closure_label:
                raise SystemExit(f"Completion next proof packet should show actionable learning evidence before closure: {metadata}")
            if metadata.get("completion_next_proof_uses_actionable_learning_prerequisite"):
                if metadata.get("next_proof_command") != metadata.get("execution_learning_actionable_next_proof_command"):
                    raise SystemExit(f"Completion next proof packet should route to actionable learning evidence first: {metadata}")
                if metadata.get("completion_ordered_next_proof_command") == metadata.get("next_proof_command"):
                    raise SystemExit(f"Completion next proof packet should keep ordered and actionable proofs distinct: {metadata}")
            if metadata.get("command_source") not in {"execution_learning_actionable", "execution_case_closure", "execution_recovery_closure", "execution_learning", "approval_queue", "task_completion", "completion_audit", "evidence_ledger", "agi_next_build_move", "completion_claim_gate"}:
                raise SystemExit(f"Completion next proof packet reported unknown command source: {metadata}")
            assert_selected_agi_target_readiness(metadata, "Completion next proof packet")
            assert_completion_next_proof_handoff(metadata, "Completion next proof packet")
            assert_storage_handoff_contract(metadata, "Completion next proof packet", "completion_next_proof_storage")
            assert_storage_handoff_contract(
                metadata.get("completion_next_proof_handoff") or {},
                "Completion next proof packet nested handoff",
                "completion_next_proof_storage",
            )
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Completion next proof packet unsafe metadata {key}: {metadata}")

        refresh_cases = [
            "completion proof refresh",
            "refresh proof lanes",
            "completion evidence refresh",
        ]
        for case in refresh_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "completion_proof_refresh_packet":
                raise SystemExit(f"Expected '{case}' to route to completion_proof_refresh_packet.")
            for expected in [
                "Jarvis completion proof refresh packet",
                "Lane refresh map:",
                "Current-evidence worklist:",
                "Refresh queues:",
                "Ordered refresh queue:",
                "Command readiness:",
                "placeholder commands",
                "resolved next proof:",
                "proof freshness:",
                "Placeholder resolution:",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Completion proof refresh packet missed expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            lane_rows = metadata.get("lane_rows") or []
            ordered_queue = metadata.get("ordered_refresh_queue") or []
            resolved_queue = metadata.get("resolved_refresh_queue") or []
            placeholder_rows = metadata.get("placeholder_resolution_rows") or []
            if not lane_rows or metadata.get("lane_count") != len(lane_rows):
                raise SystemExit(f"Completion proof refresh packet missed lane rows: {metadata}")
            if metadata.get("stale_lane_count", 0) < 1:
                raise SystemExit(f"Completion proof refresh packet should report stale lanes in setup fixture: {metadata}")
            stale_lane_names = metadata.get("stale_lane_names") or []
            current_worklist = metadata.get("lane_current_evidence_worklist") or []
            if metadata.get("lane_current_evidence_worklist_count") != len(current_worklist):
                raise SystemExit(f"Completion proof refresh packet missed current-evidence worklist count: {metadata}")
            if metadata.get("stale_lane_count") != len(stale_lane_names) or len(current_worklist) != len(stale_lane_names):
                raise SystemExit(f"Completion proof refresh packet should align stale lanes and current-evidence worklist: {metadata}")
            if not current_worklist or not all(row.get("needs_current_evidence") is True for row in current_worklist):
                raise SystemExit(f"Completion proof refresh packet missed actionable stale-lane rows: {metadata}")
            if any(not row.get("resolved_next_proof_command") for row in current_worklist):
                raise SystemExit(f"Completion proof refresh packet missed resolved proof commands: {metadata}")
            if not any(row.get("proof_freshness") == "available_without_recent_evidence" for row in current_worklist):
                raise SystemExit(f"Completion proof refresh packet missed available-without-recent-evidence freshness state: {metadata}")
            if metadata.get("available_without_recent_evidence_lane_count") != len(metadata.get("available_without_recent_evidence_lanes") or []):
                raise SystemExit(f"Completion proof refresh packet missed available-without-recent-evidence lane count: {metadata}")
            if metadata.get("partial_evidence_lane_count") != len(metadata.get("partial_evidence_lanes") or []):
                raise SystemExit(f"Completion proof refresh packet missed partial-evidence lane count: {metadata}")
            if metadata.get("missing_evidence_lane_count") != len(metadata.get("missing_evidence_lanes") or []):
                raise SystemExit(f"Completion proof refresh packet missed missing-evidence lane count: {metadata}")
            if not ordered_queue or metadata.get("ordered_refresh_queue_count") != len(ordered_queue):
                raise SystemExit(f"Completion proof refresh packet missed ordered queue: {metadata}")
            if not resolved_queue or metadata.get("resolved_refresh_queue_count") != len(resolved_queue):
                raise SystemExit(f"Completion proof refresh packet missed resolved queue: {metadata}")
            if metadata.get("first_refresh_command") != ordered_queue[0]:
                raise SystemExit(f"Completion proof refresh packet missed first command alias: {metadata}")
            if metadata.get("first_resolved_refresh_command") != resolved_queue[0]:
                raise SystemExit(f"Completion proof refresh packet missed first resolved command alias: {metadata}")
            if metadata.get("first_lane_refresh_command") != (metadata.get("lane_refresh_commands") or [""])[0]:
                raise SystemExit(f"Completion proof refresh packet missed first lane command alias: {metadata}")
            if not metadata.get("first_resolved_lane_refresh_command"):
                raise SystemExit(f"Completion proof refresh packet missed first resolved lane command alias: {metadata}")
            if metadata.get("lane_refresh_command_count") != len(metadata.get("lane_refresh_commands") or []):
                raise SystemExit(f"Completion proof refresh packet missed lane refresh command count: {metadata}")
            if metadata.get("placeholder_refresh_command_count") != len(metadata.get("placeholder_refresh_commands") or []):
                raise SystemExit(f"Completion proof refresh packet missed placeholder command count: {metadata}")
            if metadata.get("placeholder_resolution_count") != len(placeholder_rows):
                raise SystemExit(f"Completion proof refresh packet missed placeholder resolution count: {metadata}")
            if len(placeholder_rows) != metadata.get("placeholder_refresh_command_count"):
                raise SystemExit(f"Completion proof refresh packet should resolve every placeholder command: {metadata}")
            if any(not row.get("resolved_command") for row in placeholder_rows):
                raise SystemExit(f"Completion proof refresh packet returned unresolved placeholder rows: {metadata}")
            if metadata.get("unresolved_placeholder_count") != 0:
                raise SystemExit(f"Completion proof refresh packet should expose fallback commands for placeholders: {metadata}")
            if metadata.get("completion_claim_ready") is not False or metadata.get("claim_gate_verdict") != "CLAIM_BLOCKED":
                raise SystemExit(f"Completion proof refresh packet should inherit blocked claim gate: {metadata}")
            if "completion claim gate" not in ordered_queue:
                raise SystemExit(f"Completion proof refresh packet should end with claim refresh command: {metadata}")
            if metadata.get("agi_real_execution_gap_count") != 6:
                raise SystemExit(f"Completion proof refresh packet missed AGI gap count: {metadata}")
            claim_gate_metadata = metadata.get("claim_gate_metadata") or {}
            for key in [
                "latest_execution_case_review_state",
                "latest_execution_case_review_verdict",
                "latest_execution_case_review_next_safe_command",
                "latest_execution_case_review_approval_required",
                "latest_execution_case_review_checklist_items",
                "latest_execution_case_review_blocker_count",
                "latest_execution_case_review_handoff",
                "latest_execution_case_review_handoff_present",
                "latest_execution_case_evidence_preview_gate_count",
                "latest_execution_case_evidence_preview_ready_count",
                "latest_execution_case_evidence_preview_blocked_count",
                "latest_execution_case_evidence_preview_verdicts",
                "latest_execution_case_evidence_preview_latest_verdict",
                "latest_execution_case_evidence_preview_latest_event_id",
                "latest_execution_case_evidence_preview_latest_handoff",
                "latest_execution_case_evidence_preview_latest_handoff_present",
            ]:
                if metadata.get(key) != claim_gate_metadata.get(key):
                    raise SystemExit(f"Completion proof refresh packet latest case field {key} diverged from claim gate metadata: {metadata}")
                if metadata.get("completion_proof_refresh_handoff", {}).get(key) != metadata.get(key):
                    raise SystemExit(f"Completion proof refresh packet handoff latest case field {key} diverged: {metadata}")
            if metadata.get("latest_execution_case_review_handoff_present") and (metadata.get("latest_execution_case_review_handoff") or {}).get("source") != "execution_case_review_packet":
                raise SystemExit(f"Completion proof refresh packet missed latest case review handoff source: {metadata}")
            if metadata.get("latest_execution_case_evidence_preview_gate_count", 0) > 0:
                if metadata.get("latest_execution_case_evidence_preview_ready_count", 0) + metadata.get("latest_execution_case_evidence_preview_blocked_count", 0) != metadata.get("latest_execution_case_evidence_preview_gate_count"):
                    raise SystemExit(f"Completion proof refresh packet latest case evidence preflight counts diverged: {metadata}")
                if metadata.get("latest_execution_case_evidence_preview_latest_handoff_present") is not True:
                    raise SystemExit(f"Completion proof refresh packet missed latest case evidence preflight handoff flag: {metadata}")
                if (metadata.get("latest_execution_case_evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
                    raise SystemExit(f"Completion proof refresh packet missed latest case evidence preflight handoff source: {metadata}")
            assert_selected_agi_target_readiness(metadata, "Completion proof refresh packet")
            assert_completion_proof_refresh_handoff(metadata, "Completion proof refresh packet")
            assert_storage_handoff_contract(metadata, "Completion proof refresh packet", "completion_proof_refresh_storage")
            assert_storage_handoff_contract(
                metadata.get("completion_proof_refresh_handoff") or {},
                "Completion proof refresh packet nested handoff",
                "completion_proof_refresh_storage",
            )
            if not isinstance(metadata.get("claim_gate_metadata"), dict):
                raise SystemExit(f"Completion proof refresh packet missed nested claim metadata: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Completion proof refresh packet unsafe metadata {key}: {metadata}")

        operator_handoff_cases = [
            "operator handoff",
            "next operator handoff",
            "what should the operator do next",
        ]
        for case in operator_handoff_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "operator_handoff_packet":
                raise SystemExit(f"Expected '{case}' to route to operator_handoff_packet.")
            for expected in [
                "Jarvis operator handoff",
                "Next operator command:",
                "Command source:",
                "Command reason:",
                "First blocker:",
                "Queue position:",
                "Refresh command:",
                "Closure debt:",
                "latest case review state:",
                "latest case review verdict:",
                "latest case review next safe command:",
                "latest case evidence preflight:",
                "latest case evidence preflight verdict:",
                "latest execution case:",
                "recovery closure:",
                "execution learning:",
                "actionable learning next required:",
                "actionable learning proof alias:",
                "learning closure command:",
                "verification next required:",
                "storage proof alias:",
                "Storage readiness:",
                "AGI harness focus:",
                "selected target:",
                "target file integrity:",
                "ready for scoped implementation review:",
                "Operator contract:",
                "Completion proof queue:",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Operator handoff packet missed expected text for '{case}': {expected}")
            for stale_label in [
                "latest case next proof:",
                "recovery next proof:",
                "verification next proof:",
                "actionable learning next proof:",
                "learning next proof:",
                "- next proof:",
            ]:
                if stale_label in result.response:
                    raise SystemExit(f"Operator handoff packet should render command-first/proof-alias labels, not {stale_label!r}.")
            metadata = result.tool_results[0].metadata
            proof_queue = metadata.get("completion_proof_queue") or []
            if not proof_queue or metadata.get("completion_proof_queue_count") != len(proof_queue):
                raise SystemExit(f"Operator handoff packet missed proof queue metadata: {metadata}")
            if metadata.get("next_operator_command") != proof_queue[0]:
                raise SystemExit(f"Operator handoff packet missed first proof command: {metadata}")
            if metadata.get("completion_next_proof_command") != proof_queue[0] or metadata.get("next_proof_command") != proof_queue[0]:
                raise SystemExit(f"Operator handoff packet missed proof command aliases: {metadata}")
            if metadata.get("storage_readiness_blocks_completion_claim"):
                if metadata.get("storage_runtime_fallback_active") is not True:
                    raise SystemExit(f"Operator handoff packet missed active storage fallback: {metadata}")
                if not metadata.get("storage_readiness_next_commands"):
                    raise SystemExit(f"Operator handoff packet missed storage recovery commands: {metadata}")
                if metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
                    raise SystemExit(f"Operator handoff packet missed storage recovery check command: {metadata}")
                if metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
                    raise SystemExit(f"Operator handoff packet missed native storage recovery check command: {metadata}")
                if metadata.get("storage_recovery_check_api") != "/api/storage-recovery-check":
                    raise SystemExit(f"Operator handoff packet missed read-only storage recovery check API: {metadata}")
                if metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
                    raise SystemExit(f"Operator handoff packet missed storage recovery command: {metadata}")
                assert_fallback_storage_recovery_mode(metadata, "Operator handoff packet")
                storage_commands = metadata.get("storage_readiness_next_commands") or []
                if metadata.get("storage_readiness_next_proof_command") != storage_commands[0]:
                    raise SystemExit(f"Operator handoff packet missed storage next proof alias: {metadata}")
                storage_proof_queue = metadata.get("storage_readiness_proof_queue") or []
                storage_proof_preview = metadata.get("storage_readiness_proof_queue_preview") or []
                if storage_proof_preview != storage_proof_queue[:5]:
                    raise SystemExit(f"Operator handoff packet storage proof preview diverged: {metadata}")
                if metadata.get("storage_readiness_proof_queue_preview_count") != len(storage_proof_preview):
                    raise SystemExit(f"Operator handoff packet storage proof preview count diverged: {metadata}")
                if metadata.get("storage_readiness_proof_queue_remaining_count") != max(len(storage_proof_queue) - len(storage_proof_preview), 0):
                    raise SystemExit(f"Operator handoff packet storage proof remaining count diverged: {metadata}")
            if not metadata.get("command_reason") or not metadata.get("first_blocker"):
                raise SystemExit(f"Operator handoff packet missed reason/blocker metadata: {metadata}")
            if metadata.get("queue_position") != 1 or metadata.get("queue_total") != len(proof_queue):
                raise SystemExit(f"Operator handoff packet missed queue position metadata: {metadata}")
            if metadata.get("refresh_command") != "completion next proof":
                raise SystemExit(f"Operator handoff packet missed refresh command metadata: {metadata}")
            if not metadata.get("latest_execution_case_closure_verdict"):
                raise SystemExit(f"Operator handoff packet missed latest case closure metadata: {metadata}")
            if metadata.get("latest_execution_case_review_handoff_present") and (metadata.get("latest_execution_case_review_handoff") or {}).get("source") != "execution_case_review_packet":
                raise SystemExit(f"Operator handoff packet missed latest case review handoff source: {metadata}")
            if metadata.get("latest_execution_case_evidence_preview_gate_count", 0) > 0:
                if metadata.get("latest_execution_case_evidence_preview_ready_count", 0) + metadata.get("latest_execution_case_evidence_preview_blocked_count", 0) != metadata.get("latest_execution_case_evidence_preview_gate_count"):
                    raise SystemExit(f"Operator handoff packet latest case evidence preflight counts diverged: {metadata}")
                if metadata.get("latest_execution_case_evidence_preview_latest_handoff_present") is not True:
                    raise SystemExit(f"Operator handoff packet missed latest case evidence preflight handoff flag: {metadata}")
                if (metadata.get("latest_execution_case_evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
                    raise SystemExit(f"Operator handoff packet missed latest case evidence preflight handoff source: {metadata}")
            for key in [
                "latest_execution_case_review_state",
                "latest_execution_case_review_verdict",
                "latest_execution_case_review_next_safe_command",
                "latest_execution_case_review_approval_required",
                "latest_execution_case_review_checklist_items",
                "latest_execution_case_review_blocker_count",
                "latest_execution_case_review_handoff",
                "latest_execution_case_review_handoff_present",
                "latest_execution_case_evidence_preview_gate_count",
                "latest_execution_case_evidence_preview_ready_count",
                "latest_execution_case_evidence_preview_blocked_count",
                "latest_execution_case_evidence_preview_verdicts",
                "latest_execution_case_evidence_preview_latest_verdict",
                "latest_execution_case_evidence_preview_latest_event_id",
                "latest_execution_case_evidence_preview_latest_handoff",
                "latest_execution_case_evidence_preview_latest_handoff_present",
            ]:
                if metadata.get(key) != (metadata.get("completion_next_proof_handoff") or {}).get(key):
                    raise SystemExit(f"Operator handoff packet latest case review field {key} diverged from completion proof handoff: {metadata}")
            if not metadata.get("execution_health_recovery_closure_state"):
                raise SystemExit(f"Operator handoff packet missed recovery closure metadata: {metadata}")
            if not metadata.get("execution_learning_state"):
                raise SystemExit(f"Operator handoff packet missed execution learning metadata: {metadata}")
            if metadata.get("execution_learning_missing") and (not metadata.get("execution_learning_closure_command") or metadata.get("execution_learning_closure_command") not in proof_queue):
                raise SystemExit(f"Operator handoff packet missed learning closure command in proof queue: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") is not metadata.get("completion_next_proof_handoff", {}).get("execution_learning_blocks_completion_claim"):
                raise SystemExit(f"Operator handoff packet learning blocker diverged from completion proof handoff: {metadata}")
            assert_execution_learning_actionable_aliases(metadata, "Operator handoff packet")
            actionable_label = result.response.find("actionable learning next required:")
            closure_label = result.response.find("learning next required:")
            if actionable_label < 0 or closure_label < 0 or actionable_label > closure_label:
                raise SystemExit(f"Operator handoff packet should show actionable learning evidence before closure: {metadata}")
            if metadata.get("draft_only") is not True or metadata.get("requires_manual_send") is not True or metadata.get("loads_without_execution") is not True:
                raise SystemExit(f"Operator handoff packet missed draft-only contract: {metadata}")
            if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(f"Operator handoff packet should not authorize execution or completion claims: {metadata}")
            if metadata.get("approval_granted") is not False:
                raise SystemExit(f"Operator handoff packet should not grant approval: {metadata}")
            if metadata.get("source_tool") != "completion_next_proof_packet":
                raise SystemExit(f"Operator handoff packet missed source tool: {metadata}")
            assert_selected_agi_target_readiness(metadata, "Operator handoff packet")
            assert_operator_handoff(metadata, "Operator handoff packet")
            assert_storage_handoff_contract(metadata, "Operator handoff packet", "operator_handoff_storage")
            assert_storage_handoff_contract(
                metadata.get("operator_handoff") or {},
                "Operator handoff packet nested handoff",
                "operator_handoff_storage",
            )
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Operator handoff packet unsafe metadata {key}: {metadata}")

        readiness_digest_cases = [
            "harness readiness digest",
            "readiness digest",
            "what blocks Jarvis now",
        ]
        for case in readiness_digest_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "harness_readiness_digest":
                raise SystemExit(f"Expected '{case}' to route to harness_readiness_digest.")
            for expected in [
                "Jarvis harness readiness digest",
                "Readiness state:",
                "Readiness verdict:",
                "Completion claim ready:",
                "Top risk:",
                "Next command:",
                "Readiness rows:",
                "Main blockers:",
                "Recovery and learning:",
                "actionable learning next required:",
                "actionable learning proof alias:",
                "Storage readiness:",
                "next storage required:",
                "storage proof alias:",
                "AGI harness focus:",
                "selected target:",
                "target file integrity:",
                "ready for scoped implementation review:",
                "Command ladder:",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Harness readiness digest missed expected text for '{case}': {expected}")
            for stale_label in [
                "actionable learning next proof:",
                "next storage proof:",
            ]:
                if stale_label in result.response:
                    raise SystemExit(f"Harness readiness digest should render command-first/proof-alias labels, not {stale_label!r}.")
            metadata = result.tool_results[0].metadata
            command_ladder = metadata.get("command_ladder") or []
            proof_queue = metadata.get("completion_proof_queue") or []
            if metadata.get("source_tool") != "completion_next_proof_packet":
                raise SystemExit(f"Harness readiness digest missed source tool: {metadata}")
            if metadata.get("readiness_state") not in {"READY_FOR_HUMAN_COMPLETION_REVIEW", "PROOF_DEBT_BLOCKED", "NEXT_BUILD_SLICE_READY", "NEEDS_FRESH_COMPLETION_GATE"}:
                raise SystemExit(f"Harness readiness digest reported unknown readiness state: {metadata}")
            if metadata.get("readiness_verdict") != metadata.get("readiness_state"):
                raise SystemExit(f"Harness readiness digest readiness verdict alias diverged: {metadata}")
            if metadata.get("completion_next_proof_command") != metadata.get("next_proof_command"):
                raise SystemExit(f"Harness readiness digest completion next-proof alias diverged: {metadata}")
            if metadata.get("completion_claim_ready") is not False or metadata.get("allowed_to_claim") is not False:
                raise SystemExit(f"Harness readiness digest should inherit blocked claim gate: {metadata}")
            if metadata.get("storage_readiness_blocks_completion_claim"):
                if metadata.get("readiness_state") != "PROOF_DEBT_BLOCKED":
                    raise SystemExit(f"Harness readiness digest should classify storage fallback as proof debt: {metadata}")
                if "storage" not in str(metadata.get("top_risk", "")).lower():
                    raise SystemExit(f"Harness readiness digest top risk should name storage proof debt: {metadata}")
                if metadata.get("storage_runtime_fallback_active") is not True:
                    raise SystemExit(f"Harness readiness digest missed active storage fallback: {metadata}")
                if not metadata.get("storage_readiness_next_commands"):
                    raise SystemExit(f"Harness readiness digest missed storage recovery commands: {metadata}")
                storage_commands = metadata.get("storage_readiness_next_commands") or []
                if metadata.get("storage_readiness_next_proof_command") != storage_commands[0]:
                    raise SystemExit(f"Harness readiness digest missed storage next proof alias: {metadata}")
                if metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
                    raise SystemExit(f"Harness readiness digest missed storage recovery check command: {metadata}")
                if metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
                    raise SystemExit(f"Harness readiness digest missed native storage recovery check command: {metadata}")
                if metadata.get("storage_recovery_check_api") != "/api/storage-recovery-check":
                    raise SystemExit(f"Harness readiness digest missed read-only storage recovery check API: {metadata}")
                if metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
                    raise SystemExit(f"Harness readiness digest missed storage recovery command: {metadata}")
                assert_fallback_storage_recovery_mode(metadata, "Harness readiness digest")
                if storage_commands != EXPECTED_RESTART_ONLY_STORAGE_COMMANDS:
                    raise SystemExit(f"Harness readiness digest storage command aliases diverged from queue: {metadata}")
                storage_proof_queue = metadata.get("storage_readiness_proof_queue") or []
                storage_proof_preview = metadata.get("storage_readiness_proof_queue_preview") or []
                if storage_proof_preview != storage_proof_queue[:5]:
                    raise SystemExit(f"Harness readiness digest storage proof preview diverged: {metadata}")
                if metadata.get("storage_readiness_proof_queue_preview_count") != len(storage_proof_preview):
                    raise SystemExit(f"Harness readiness digest storage proof preview count diverged: {metadata}")
                if metadata.get("storage_readiness_proof_queue_remaining_count") != max(len(storage_proof_queue) - len(storage_proof_preview), 0):
                    raise SystemExit(f"Harness readiness digest storage proof remaining count diverged: {metadata}")
                if BOOTSTRAP_CHECK_COMMAND in storage_commands or BOOTSTRAP_WRITE_COMMAND in storage_commands:
                    raise SystemExit(f"Harness readiness digest restart-only queue should not require bootstrap commands: {metadata}")
                if command_ladder[: len(storage_commands)] != storage_commands:
                    raise SystemExit(f"Harness readiness digest command ladder should keep storage recovery commands first: {metadata}")
                if metadata.get("command_ladder_storage_prefix") != storage_commands:
                    raise SystemExit(f"Harness readiness digest missed storage prefix alias: {metadata}")
                if metadata.get("command_ladder_storage_prefix_count") != len(storage_commands):
                    raise SystemExit(f"Harness readiness digest missed storage prefix count: {metadata}")
            if not metadata.get("next_command") or metadata.get("next_proof_command") != metadata.get("next_command"):
                raise SystemExit(f"Harness readiness digest missed next command aliases: {metadata}")
            if not command_ladder or metadata.get("command_ladder_count") != len(command_ladder):
                raise SystemExit(f"Harness readiness digest missed command ladder metadata: {metadata}")
            command_ladder_preview = metadata.get("command_ladder_preview") or []
            if command_ladder_preview != command_ladder[:5]:
                raise SystemExit(f"Harness readiness digest command ladder preview diverged: {metadata}")
            if metadata.get("command_ladder_preview_count") != len(command_ladder_preview):
                raise SystemExit(f"Harness readiness digest command ladder preview count diverged: {metadata}")
            if metadata.get("command_ladder_remaining_count") != max(len(command_ladder) - len(command_ladder_preview), 0):
                raise SystemExit(f"Harness readiness digest command ladder remaining count diverged: {metadata}")
            if metadata.get("command_ladder_first_command") != command_ladder[0]:
                raise SystemExit(f"Harness readiness digest missed first command alias: {metadata}")
            if metadata.get("command_ladder_first_command") != metadata.get("next_command"):
                raise SystemExit(f"Harness readiness digest first command alias should match next command: {metadata}")
            if proof_queue and metadata.get("next_command") != proof_queue[0]:
                raise SystemExit(f"Harness readiness digest next command should follow completion proof queue: {metadata}")
            if metadata.get("completion_proof_queue_count") != len(proof_queue):
                raise SystemExit(f"Harness readiness digest missed proof queue count: {metadata}")
            proof_queue_preview = metadata.get("completion_proof_queue_preview") or []
            if proof_queue_preview != proof_queue[:5]:
                raise SystemExit(f"Harness readiness digest proof queue preview diverged: {metadata}")
            if metadata.get("completion_proof_queue_preview_count") != len(proof_queue_preview):
                raise SystemExit(f"Harness readiness digest proof queue preview count diverged: {metadata}")
            if metadata.get("completion_proof_queue_remaining_count") != max(len(proof_queue) - len(proof_queue_preview), 0):
                raise SystemExit(f"Harness readiness digest proof queue remaining count diverged: {metadata}")
            pending_approval_preview = metadata.get("pending_approval_preview") or []
            pending_approval_count = int(metadata.get("pending_approval_count") or 0)
            if metadata.get("approval_review_required") is not bool(pending_approval_count):
                raise SystemExit(f"Harness readiness digest approval review flag diverged from count: {metadata}")
            if metadata.get("pending_approval_preview_count") != len(pending_approval_preview):
                raise SystemExit(f"Harness readiness digest pending approval preview count diverged: {metadata}")
            if metadata.get("pending_approval_remaining_count") != max(pending_approval_count - len(pending_approval_preview), 0):
                raise SystemExit(f"Harness readiness digest pending approval remaining count diverged: {metadata}")
            if pending_approval_preview:
                first_approval = pending_approval_preview[0]
                first_id = first_approval.get("id")
                if metadata.get("pending_approval_first_id") != first_id:
                    raise SystemExit(f"Harness readiness digest first approval id diverged: {metadata}")
                if metadata.get("approval_review_first_readiness_command") != f"approval readiness {first_id}":
                    raise SystemExit(f"Harness readiness digest first approval readiness command diverged: {metadata}")
                if metadata.get("approval_review_first_proof_command") != f"approval chain proof {first_id}":
                    raise SystemExit(f"Harness readiness digest first approval proof command diverged: {metadata}")
                if metadata.get("approval_review_next_command") != metadata.get("approval_review_first_readiness_command"):
                    raise SystemExit(f"Harness readiness digest approval review next command diverged: {metadata}")
                if metadata.get("approval_review_row_next_command") != metadata.get("approval_review_next_command"):
                    raise SystemExit(f"Harness readiness digest approval review row-next command diverged: {metadata}")
            elif metadata.get("approval_review_next_command") or metadata.get("approval_review_row_next_command") != "none":
                raise SystemExit(f"Harness readiness digest should not advertise approval review work without pending approvals: {metadata}")
            latest_review_handoff = metadata.get("latest_execution_case_review_handoff") or {}
            if metadata.get("latest_execution_case_review_handoff_present"):
                expected_review_next = (
                    latest_review_handoff.get("next_command")
                    or latest_review_handoff.get("next_safe_command")
                    or metadata.get("latest_execution_case_review_next_safe_command")
                )
                latest_review_queue = [
                    str(command).strip()
                    for command in list(latest_review_handoff.get("mission_command_queue") or [])
                    if str(command).strip()
                ]
                latest_review_preview = metadata.get("latest_execution_case_review_mission_command_preview") or []
                if metadata.get("latest_execution_case_review_next_command") != expected_review_next:
                    raise SystemExit(f"Harness readiness digest latest review next command diverged from handoff: {metadata}")
                if metadata.get("latest_execution_case_review_initial_governor_command") != latest_review_handoff.get("initial_governor_command"):
                    raise SystemExit(f"Harness readiness digest latest review initial governor command diverged: {metadata}")
                if metadata.get("latest_execution_case_review_mission_command_queue") != latest_review_queue:
                    raise SystemExit(f"Harness readiness digest latest review mission queue diverged: {metadata}")
                if metadata.get("latest_execution_case_review_mission_command_count") != len(latest_review_queue):
                    raise SystemExit(f"Harness readiness digest latest review mission queue count diverged: {metadata}")
                if latest_review_preview != latest_review_queue[:5]:
                    raise SystemExit(f"Harness readiness digest latest review mission preview diverged: {metadata}")
                if metadata.get("latest_execution_case_review_mission_command_preview_count") != len(latest_review_preview):
                    raise SystemExit(f"Harness readiness digest latest review mission preview count diverged: {metadata}")
                if metadata.get("latest_execution_case_review_mission_command_remaining_count") != max(len(latest_review_queue) - len(latest_review_preview), 0):
                    raise SystemExit(f"Harness readiness digest latest review mission remaining count diverged: {metadata}")
                for key in [
                    "forecast_queue_before",
                    "forecast_queue_after_if_sent",
                    "forecast_queue_delta_if_sent",
                ]:
                    if metadata.get(f"latest_execution_case_review_{key}") != latest_review_handoff.get(key):
                        raise SystemExit(f"Harness readiness digest latest review forecast field {key} diverged: {metadata}")
            latest_case_closure_queue = metadata.get("latest_execution_case_closure_proof_queue") or []
            latest_case_closure_preview = metadata.get("latest_execution_case_closure_proof_queue_preview") or []
            if latest_case_closure_preview != latest_case_closure_queue[:5]:
                raise SystemExit(f"Harness readiness digest latest case closure preview diverged: {metadata}")
            if metadata.get("latest_execution_case_closure_proof_queue_preview_count") != len(latest_case_closure_preview):
                raise SystemExit(f"Harness readiness digest latest case closure preview count diverged: {metadata}")
            if metadata.get("latest_execution_case_closure_proof_queue_remaining_count") != max(len(latest_case_closure_queue) - len(latest_case_closure_preview), 0):
                raise SystemExit(f"Harness readiness digest latest case closure remaining count diverged: {metadata}")
            expected_latest_case_first_proof = latest_case_closure_preview[0] if latest_case_closure_preview else ""
            if metadata.get("latest_execution_case_closure_first_proof_command") != expected_latest_case_first_proof:
                raise SystemExit(f"Harness readiness digest latest case closure first proof diverged: {metadata}")
            if not metadata.get("readiness_rows") or len(metadata.get("readiness_rows")) < 6:
                raise SystemExit(f"Harness readiness digest missed readiness rows: {metadata}")
            readiness_rows = {row.get("name"): row for row in metadata.get("readiness_rows", [])}
            verification_row = readiness_rows.get("verification coverage") or {}
            verification_queue = metadata.get("execution_health_verification_proof_queue") or []
            verification_queue_preview = metadata.get("execution_health_verification_proof_queue_preview") or []
            expected_verification_gap_count = max(
                int(metadata.get("execution_health_verification_recent_tool_runs") or 0)
                - int(metadata.get("execution_health_verification_recent_verification_runs") or 0),
                0,
            )
            if metadata.get("execution_health_verification_gap_count") != expected_verification_gap_count:
                raise SystemExit(f"Harness readiness digest verification gap count diverged: {metadata}")
            if metadata.get("execution_health_verification_blocks_completion_claim") and not metadata.get("execution_health_verification_first_gap"):
                raise SystemExit(f"Harness readiness digest missed verification first gap: {metadata}")
            if verification_queue_preview != verification_queue[:3]:
                raise SystemExit(f"Harness readiness digest verification queue preview diverged: {metadata}")
            if metadata.get("execution_health_verification_proof_queue_preview_count") != len(verification_queue_preview):
                raise SystemExit(f"Harness readiness digest verification queue preview count diverged: {metadata}")
            if metadata.get("execution_health_verification_proof_queue_remaining_count") != max(len(verification_queue) - len(verification_queue_preview), 0):
                raise SystemExit(f"Harness readiness digest verification queue remaining count diverged: {metadata}")
            if metadata.get("execution_health_verification_blocks_completion_claim"):
                expected_verification_next = (
                    metadata.get("execution_health_verification_row_next_command")
                    or metadata.get("execution_health_verification_concrete_next_proof_command")
                    or metadata.get("execution_health_verification_next_proof_command")
                )
                if verification_row.get("state") != "blocked" or verification_row.get("next") != expected_verification_next:
                    raise SystemExit(f"Harness readiness digest verification row missed blocked proof command: {metadata}")
            elif verification_row.get("state") != "ready" or verification_row.get("next") != "none":
                raise SystemExit(f"Harness readiness digest verification row should not advertise stale proof work: {metadata}")
            if not metadata.get("blocker_details") or metadata.get("blocker_count") != len(metadata.get("blocker_details", [])):
                raise SystemExit(f"Harness readiness digest missed blocker details: {metadata}")
            if not metadata.get("execution_health_recovery_closure_state"):
                raise SystemExit(f"Harness readiness digest missed recovery closure state: {metadata}")
            if not metadata.get("execution_learning_state"):
                raise SystemExit(f"Harness readiness digest missed learning state: {metadata}")
            storage_issues = metadata.get("storage_issues") or []
            storage_issue_preview = metadata.get("storage_issue_preview") or []
            if storage_issue_preview != storage_issues[:3]:
                raise SystemExit(f"Harness readiness digest storage issue preview diverged: {metadata}")
            if metadata.get("storage_issue_preview_count") != len(storage_issue_preview):
                raise SystemExit(f"Harness readiness digest storage issue preview count diverged: {metadata}")
            if metadata.get("storage_first_issue") != (storage_issue_preview[0] if storage_issue_preview else ""):
                raise SystemExit(f"Harness readiness digest storage first issue diverged: {metadata}")
            assert_execution_learning_actionable_aliases(metadata, "Harness readiness digest")
            learning_actionable_next = metadata.get("execution_learning_actionable_next_proof_command")
            learning_ordered_next = metadata.get("execution_learning_next_proof_command")
            learning_row = readiness_rows.get("learning loop") or {}
            latest_case_queue = metadata.get("latest_execution_case_closure_proof_queue") or []
            if metadata.get("execution_learning_blocks_completion_claim") and learning_actionable_next:
                if learning_row.get("state") != "blocked" or learning_row.get("next") != learning_actionable_next:
                    raise SystemExit(f"Harness readiness digest learning row should show actionable evidence first: {metadata}")
            elif _metadata_bool(metadata.get("latest_execution_case_learning_blocks_completion_claim")) or (
                _metadata_bool(metadata.get("latest_execution_case_closure_blocks_completion_claim"))
                and any("learning" in str(command).lower() for command in latest_case_queue)
            ):
                latest_case_learning_next = (
                    metadata.get("latest_execution_case_learning_actionable_next_proof_command")
                    or metadata.get("latest_execution_case_learning_actionable_next_required_command")
                    or metadata.get("latest_execution_case_learning_next_proof_command")
                    or metadata.get("latest_execution_case_learning_next_required_command")
                    or metadata.get("latest_execution_case_next_closure_proof_command")
                )
                if learning_row.get("state") != "blocked" or learning_row.get("next") != latest_case_learning_next:
                    raise SystemExit(f"Harness readiness digest learning row missed latest case learning debt: {metadata}")
            elif learning_row.get("state") != "ready" or learning_row.get("next") != "none":
                raise SystemExit(f"Harness readiness digest learning row should not advertise stale proof work: {metadata}")
            agi_preflight_row = readiness_rows.get("AGI preflight") or {}
            preflight_blockers = metadata.get("agi_next_implementation_preflight_blockers") or []
            preflight_preview = metadata.get("agi_next_implementation_preflight_blocker_preview") or []
            expected_preflight_preview = preflight_blockers[:3]
            if preflight_preview != expected_preflight_preview:
                raise SystemExit(f"Harness readiness digest AGI preflight blocker preview diverged: {metadata}")
            if metadata.get("agi_next_implementation_preflight_blocker_preview_count") != len(preflight_preview):
                raise SystemExit(f"Harness readiness digest AGI preflight blocker preview count diverged: {metadata}")
            if metadata.get("agi_next_implementation_preflight_first_blocker") != (
                preflight_preview[0] if preflight_preview else ""
            ):
                raise SystemExit(f"Harness readiness digest AGI preflight first blocker diverged: {metadata}")
            agi_preflight_next = (
                metadata.get("agi_next_implementation_preflight_row_next_command")
                or metadata.get("agi_next_implementation_preflight_next_command")
                or metadata.get("agi_next_build_command")
                or metadata.get("agi_focus_canonical_selector_command")
                or "agi gates"
            )
            if metadata.get("agi_next_implementation_preflight_row_next_command") != agi_preflight_next:
                raise SystemExit(f"Harness readiness digest AGI preflight row-next alias diverged: {metadata}")
            if metadata.get("agi_next_implementation_preflight_ready"):
                if agi_preflight_row.get("state") != "ready" or agi_preflight_row.get("next") != agi_preflight_next:
                    raise SystemExit(f"Harness readiness digest AGI preflight row missed ready build command: {metadata}")
            elif agi_preflight_row.get("state") != "blocked" or agi_preflight_row.get("next") != agi_preflight_next:
                raise SystemExit(f"Harness readiness digest AGI preflight row missed blocked next command: {metadata}")
            if learning_actionable_next and learning_ordered_next and learning_actionable_next != learning_ordered_next:
                if learning_actionable_next not in command_ladder or learning_ordered_next not in command_ladder:
                    raise SystemExit(f"Harness readiness digest missed learning commands in command ladder: {metadata}")
                if command_ladder.index(learning_actionable_next) > command_ladder.index(learning_ordered_next):
                    raise SystemExit(f"Harness readiness digest should put actionable learning evidence before closure: {metadata}")
            agi_next_build_command = metadata.get("agi_next_build_command")
            if metadata.get("latest_execution_case_found") is True and latest_case_queue and agi_next_build_command in command_ladder:
                missing_latest_case_commands = [command for command in latest_case_queue if command not in command_ladder]
                if missing_latest_case_commands:
                    raise SystemExit(f"Harness readiness digest missed latest case closure commands in command ladder: {metadata}")
                agi_index = command_ladder.index(agi_next_build_command)
                if any(command_ladder.index(command) > agi_index for command in latest_case_queue):
                    raise SystemExit(f"Harness readiness digest should keep latest case closure proof before AGI build work: {metadata}")
            if metadata.get("latest_execution_case_evidence_preview_gate_count", 0) > 0:
                if metadata.get("latest_execution_case_evidence_preview_ready_count", 0) + metadata.get("latest_execution_case_evidence_preview_blocked_count", 0) != metadata.get("latest_execution_case_evidence_preview_gate_count"):
                    raise SystemExit(f"Harness readiness digest evidence preflight counts diverged: {metadata}")
                if metadata.get("latest_execution_case_evidence_preview_latest_handoff_present") is not True:
                    raise SystemExit(f"Harness readiness digest missed latest case evidence preflight handoff flag: {metadata}")
                if (metadata.get("latest_execution_case_evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
                    raise SystemExit(f"Harness readiness digest missed latest case evidence preflight handoff: {metadata}")
                latest_evidence_handoff = metadata.get("latest_execution_case_evidence_preview_latest_handoff") or {}
                if metadata.get("latest_execution_case_evidence_preview_next_command") != latest_evidence_handoff.get("next_command"):
                    raise SystemExit(f"Harness readiness digest evidence next command diverged from latest handoff: {metadata}")
                if metadata.get("latest_execution_case_evidence_preview_receipt_id") != latest_evidence_handoff.get("receipt_id"):
                    raise SystemExit(f"Harness readiness digest evidence receipt id diverged from latest handoff: {metadata}")
                if metadata.get("latest_execution_case_evidence_preview_receipt_kind") != latest_evidence_handoff.get("receipt_kind"):
                    raise SystemExit(f"Harness readiness digest evidence receipt kind diverged from latest handoff: {metadata}")
                if metadata.get("latest_execution_case_evidence_preview_receipt_target_status") != latest_evidence_handoff.get("receipt_target_status"):
                    raise SystemExit(f"Harness readiness digest evidence receipt target status diverged from latest handoff: {metadata}")
            assert_selected_agi_target_readiness(metadata, "Harness readiness digest")
            if metadata.get("draft_only") is not True or metadata.get("requires_manual_send") is not True or metadata.get("approval_granted") is not False:
                raise SystemExit(f"Harness readiness digest missed draft-only approval contract: {metadata}")
            if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(f"Harness readiness digest should not authorize execution or completion claims: {metadata}")
            assert_harness_readiness_handoff(metadata, "Harness readiness digest")
            assert_storage_handoff_contract(metadata, "Harness readiness digest", "harness_readiness_digest_storage")
            assert_storage_handoff_contract(
                metadata.get("harness_readiness_handoff") or {},
                "Harness readiness digest nested handoff",
                "harness_readiness_digest_storage",
            )
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Harness readiness digest unsafe metadata {key}: {metadata}")

        proof_bundle_cases = [
            (
                "execution proof bundle: use my computer to inspect the screen run a python script write a summary file and email me the result",
                True,
                False,
            ),
            (
                "can we trust this execution: use my computer to inspect the screen run a python script write a summary file and email me the result",
                True,
                False,
            ),
            (
                "what proof do we need before running this: use my computer to inspect the screen run a python script write a summary file and email me the result",
                True,
                False,
            ),
            (
                "proof bundle: explain Jarvis memory; verification answer cites memory limits; tests smoke_test_harness passed; evidence recent run ok; recovery say evidence is thin",
                False,
                True,
            ),
            (
                "show pre-trust proof for: explain Jarvis memory; verification answer cites memory limits; tests smoke_test_harness passed; evidence recent run ok; recovery say evidence is thin",
                False,
                True,
            ),
        ]
        for case, should_need_approval, should_have_receipts in proof_bundle_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "execution_proof_bundle":
                raise SystemExit(f"Expected '{case}' to route to execution_proof_bundle.")
            for expected in [
                "Jarvis execution proof bundle",
                "pre-trust packet",
                "Proof state: PROOF_BUNDLE_BLOCKED",
                "Can trust execution now: no",
                "Risk and approval",
                "Harness proof lanes",
                "execution_health_report",
                "Evidence gap summary",
                "Supplied proof receipts",
                "Current runtime evidence",
                "Blockers",
                "Next required commands",
                "Execution health recovery closure",
                "Execution learning debt",
                "learning evidence next required:",
                "learning actionable next required:",
                "learning actionable proof alias:",
                "learning closure command:",
                "learning next required",
                "actionable learning queue",
                "learning proof queue",
                "recovery:",
                "rollback or stop condition",
                "does not execute",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Execution proof bundle missing expected text for '{case}': {expected}")
            if "Next proof commands" in result.response:
                raise SystemExit(f"Execution proof bundle should render the operator queue as Next required commands for '{case}'.")
            if "next learning proof:" in result.response or "next learning evidence proof:" in result.response:
                raise SystemExit(f"Execution proof bundle should use command-first learning next-required prose for '{case}'.")
            metadata = result.tool_results[0].metadata
            if metadata.get("proof_lanes") != 8:
                raise SystemExit(f"Execution proof bundle missed lane count: {metadata}")
            if metadata.get("proof_requirements", 0) < 6:
                raise SystemExit(f"Execution proof bundle missed proof requirement count: {metadata}")
            if not metadata.get("next_proof_commands") or metadata.get("next_proof_command") != metadata["next_proof_commands"][0]:
                raise SystemExit(f"Execution proof bundle missed machine-readable next commands: {metadata}")
            if not str(metadata.get("next_proof_command", "")).startswith("execution governor: "):
                raise SystemExit(f"Execution proof bundle should route through the governor before lower-level proof packets: {metadata}")
            if not should_have_receipts and "verification" not in metadata.get("missing_requirement_names", []):
                raise SystemExit(f"Execution proof bundle should identify missing verification proof: {metadata}")
            if metadata.get("proof_state") != "PROOF_BUNDLE_BLOCKED" or metadata.get("can_trust_execution") is not False:
                raise SystemExit(f"Execution proof bundle should block in setup state: {metadata}")
            if metadata.get("approval_required") is not should_need_approval:
                raise SystemExit(f"Execution proof bundle missed approval requirement for '{case}': {metadata}")
            if metadata.get("supplied_verification") is not should_have_receipts or metadata.get("supplied_tests") is not should_have_receipts or metadata.get("supplied_evidence") is not should_have_receipts or metadata.get("supplied_recovery") is not should_have_receipts:
                raise SystemExit(f"Execution proof bundle missed supplied receipt flags for '{case}': {metadata}")
            if should_need_approval and not metadata.get("missing_receipts"):
                raise SystemExit(f"Execution proof bundle should list missing receipts for risky incomplete proof: {metadata}")
            if should_need_approval and "approval" not in metadata.get("missing_requirement_names", []):
                raise SystemExit(f"Execution proof bundle should identify missing approval proof: {metadata}")
            if metadata.get("pending_approvals", 0) < 1 or metadata.get("blockers", 0) < 1:
                raise SystemExit(f"Execution proof bundle should see setup blockers: {metadata}")
            if metadata.get("first_pending_approval_id") is None:
                raise SystemExit(f"Execution proof bundle should expose first pending approval id: {metadata}")
            if metadata.get("recent_failed_runs", 0) > 0:
                if metadata.get("first_failed_run_id") is None:
                    raise SystemExit(f"Execution proof bundle should expose first failed run id: {metadata}")
                if metadata.get("execution_health_recovery_closure_blocks_completion_claim") is not True:
                    raise SystemExit(f"Execution proof bundle should block on recovery closure debt: {metadata}")
                recovery_commands = metadata.get("execution_health_recovery_closure_required_commands") or []
                if not recovery_commands or metadata.get("execution_health_recovery_closure_next_required_command") != recovery_commands[0]:
                    raise SystemExit(f"Execution proof bundle missed recovery closure command queue: {metadata}")
                if not any(str(command).startswith("verification receipt") for command in recovery_commands):
                    raise SystemExit(f"Execution proof bundle missed verification receipt proof command: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and metadata.get("execution_learning_missing_count", 0) < 1:
                raise SystemExit(f"Execution proof bundle missed execution learning missing proof count: {metadata}")
            learning_closure_command = metadata.get("execution_learning_closure_command")
            if metadata.get("execution_learning_blocks_completion_claim") and (not learning_closure_command or not str(learning_closure_command).startswith("execution learning closure")):
                raise SystemExit(f"Execution proof bundle missed execution learning closure command: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim"):
                evidence_label = result.response.find("learning evidence next required:")
                actionable_label = result.response.find("learning actionable next required:")
                closure_label = result.response.find("learning closure command:")
                if min(evidence_label, actionable_label, closure_label) < 0:
                    raise SystemExit(f"Execution proof bundle missed learning prose labels: {result.response}")
                if not (evidence_label < closure_label and actionable_label < closure_label):
                    raise SystemExit(f"Execution proof bundle should show actionable learning evidence before closure: {result.response}")
            learning_commands = metadata.get("execution_learning_required_commands") or []
            if metadata.get("execution_learning_blocks_completion_claim") and not any(str(command).startswith("after-action learning packet") for command in learning_commands):
                raise SystemExit(f"Execution proof bundle missed after-action learning proof command: {metadata}")
            if learning_commands and metadata.get("execution_learning_next_required_command") != learning_commands[0]:
                raise SystemExit(f"Execution proof bundle missed first learning proof command: {metadata}")
            assert_execution_learning_proof_aliases(metadata, "Execution proof bundle")
            if metadata.get("execution_learning_blocks_completion_claim") and learning_closure_command not in metadata.get("next_proof_commands", []):
                raise SystemExit(f"Execution proof bundle missed learning closure in next command queue: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and not any("execution learning debt" in command or str(command).startswith("after-action learning packet") for command in metadata.get("next_proof_commands", [])):
                raise SystemExit(f"Execution proof bundle missed learning proof in next command queue: {metadata}")
            if metadata.get("draft_only") is not True or metadata.get("requires_manual_send") is not True or metadata.get("loads_without_execution") is not True:
                raise SystemExit(f"Execution proof bundle missed review-only contract: {metadata}")
            if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(f"Execution proof bundle should not authorize execution or completion claims: {metadata}")
            if metadata.get("approval_granted") is not False:
                raise SystemExit(f"Execution proof bundle should not grant approval: {metadata}")
            assert_execution_proof_handoff(metadata, "Execution proof bundle")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Execution proof bundle unsafe metadata {key}: {metadata}")

        proof_bundle_forecast = runtime.registry.get("execution_proof_bundle").handler(
            {
                "request": "run command python3 --version",
                "verification": "verification receipt 1",
                "tests": "smoke_test_harness passed",
                "evidence": "recent blocked run reviewed",
                "recovery": "stop on non-zero exit",
            }
        )
        print("[ok] direct execution_proof_bundle approval forecast")
        print(proof_bundle_forecast.output[:1400])
        print()
        metadata = proof_bundle_forecast.metadata
        assert_execution_proof_handoff(metadata, "Execution proof bundle approval forecast")
        if (
            metadata.get("forecast_queue_before") != 1
            or metadata.get("forecast_queue_after_if_sent") != 1
            or metadata.get("forecast_queue_delta_if_sent") != 0
            or metadata.get("forecast_new_approvals") != 0
            or metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Execution proof bundle missed approval reuse forecast: {metadata}")
        if "would queue new approvals if sent: 0" not in proof_bundle_forecast.output or "would reuse pending approval ids: 1" not in proof_bundle_forecast.output:
            raise SystemExit("Execution proof bundle did not render approval reuse forecast.")
        forecast = metadata.get("approval_queue_forecast") or []
        if not forecast or forecast[0].get("tool_name") != "run_shell_command" or forecast[0].get("existing_approval_id") != 1 or forecast[0].get("would_reuse_pending_approval") is not True:
            raise SystemExit(f"Execution proof bundle missed per-action approval reuse forecast: {metadata}")
        if metadata.get("planned_action_count") != 1 or metadata.get("planned_actions", [{}])[0].get("tool") != "run_shell_command":
            raise SystemExit(f"Execution proof bundle missed planned action forecast metadata: {metadata}")

        missing_receipt_bundle = runtime.handle(
            "proof bundle: explain Jarvis memory; verification verification receipt 99999 confirmed output; tests smoke_test_harness passed; evidence recent run ok; recovery stop on mismatch"
        )
        print(f"[{'ok' if missing_receipt_bundle.verified else 'blocked'}] proof bundle missing receipt target")
        print(missing_receipt_bundle.response[:2800])
        print()
        if not missing_receipt_bundle.verified:
            raise SystemExit("Expected proof bundle with missing receipt target to run read-only.")
        for expected in [
            "missing supplied verification targets: 99999",
            "supplied verification/tool-run receipt id(s) do not exist in audit log: 99999",
            "Proof state: PROOF_BUNDLE_BLOCKED",
        ]:
            if expected not in missing_receipt_bundle.response:
                raise SystemExit(f"Execution proof bundle missed missing receipt target text: {expected}")
        missing_receipt_bundle_metadata = missing_receipt_bundle.tool_results[0].metadata
        if missing_receipt_bundle_metadata.get("has_missing_supplied_verification_runs") is not True:
            raise SystemExit(f"Execution proof bundle missed missing receipt metadata: {missing_receipt_bundle_metadata}")
        if 99999 not in missing_receipt_bundle_metadata.get("missing_supplied_verification_run_ids", []):
            raise SystemExit(f"Execution proof bundle missed missing receipt id list: {missing_receipt_bundle_metadata}")
        if missing_receipt_bundle_metadata.get("can_trust_execution") is not False:
            raise SystemExit(f"Execution proof bundle should not trust missing receipt target: {missing_receipt_bundle_metadata}")
        for key in READ_ONLY_FLAGS:
            if missing_receipt_bundle_metadata.get(key) is not False:
                raise SystemExit(f"Execution proof bundle missing receipt unsafe metadata {key}: {missing_receipt_bundle_metadata}")

        mission_control_cases = [
            (
                "execution mission control: use my computer to inspect the screen run a python script write a summary file and email me the result",
                "HOLD_FOR_PENDING_APPROVAL_REVIEW",
                "NO_GO",
                True,
            ),
            (
                "mission rehearsal: explain Jarvis memory",
                "HOLD_FOR_PENDING_APPROVAL_REVIEW",
                "NO_GO",
                False,
            ),
        ]
        for case, expected_state, expected_go_no_go, should_need_approval in mission_control_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis execution mission control",
                "single read-only rehearsal",
                "Mission state:",
                "Go/no-go:",
                "Single next command:",
                "Route board:",
                "Lifecycle rehearsal:",
                "Mission packet sequence:",
                "Mission command queue:",
                "acceptance gate:",
                "execution audit gate",
                "execution learning closure",
                "after-action learning packet",
                "Operator checks:",
                "respect the operator's explicit stop times",
                "execution governor:",
                "Blockers:",
                "steering wheel and dashboard",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Execution mission control missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("mission_state") != expected_state or metadata.get("go_no_go") != expected_go_no_go:
                raise SystemExit(f"Execution mission control missed state for '{case}': {metadata}")
            if metadata.get("approval_required") is not should_need_approval:
                raise SystemExit(f"Execution mission control missed approval requirement for '{case}': {metadata}")
            if metadata.get("lifecycle_stages") != 9:
                raise SystemExit(f"Execution mission control missed lifecycle stage count: {metadata}")
            if "execution_readiness_matrix" not in metadata.get("route_packets", []) or "execution_proof_bundle" not in metadata.get("proof_packets", []):
                raise SystemExit(f"Execution mission control missed route/proof packet metadata: {metadata}")
            if "execution_recovery_packet" not in metadata.get("recovery_packets", []) or "execution_health_report" not in metadata.get("recovery_packets", []) or "failure_to_test_preview" not in metadata.get("recovery_packets", []) or "learning_review" not in metadata.get("learning_packets", []):
                raise SystemExit(f"Execution mission control missed recovery/learning packet metadata: {metadata}")
            command_queue = metadata.get("mission_command_queue") or []
            if metadata.get("mission_command_count") != len(command_queue) or len(command_queue) < 8:
                raise SystemExit(f"Execution mission control missed command queue count: {metadata}")
            learning_closure_command = metadata.get("learning_closure_command") or metadata.get("execution_learning_closure_command")
            if metadata.get("execution_learning_blocks_completion_claim") and (not learning_closure_command or not str(learning_closure_command).startswith("execution learning closure")):
                raise SystemExit(f"Execution mission control missed learning closure command: {metadata}")
            learning_command = metadata.get("learning_command")
            if metadata.get("execution_learning_blocks_completion_claim") and (not learning_command or not str(learning_command).startswith("after-action learning packet")):
                raise SystemExit(f"Execution mission control missed after-action learning command: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and learning_command != metadata.get("execution_learning_next_required_command"):
                raise SystemExit(f"Execution mission control after-action command should be the next active learning target: {metadata}")
            if (
                metadata.get("execution_learning_blocks_completion_claim")
                and learning_closure_command
                and learning_command
                and str(learning_closure_command).removeprefix("execution learning closure ")
                != str(learning_command).removeprefix("after-action learning packet ")
            ):
                raise SystemExit(f"Execution mission control learning commands should bind to the same target: {metadata}")
            expected_commands = [
                metadata.get("next_command"),
                metadata.get("cockpit_command"),
                metadata.get("governor_command"),
                metadata.get("dispatch_command"),
                metadata.get("runbook_command"),
                metadata.get("proof_bundle_command"),
                metadata.get("acceptance_command"),
                metadata.get("audit_command"),
                metadata.get("recovery_command"),
            ]
            if metadata.get("execution_learning_blocks_completion_claim"):
                expected_commands.extend([learning_command, learning_closure_command])
            for expected_command in expected_commands:
                if expected_command not in command_queue:
                    raise SystemExit(f"Execution mission control missed queued command {expected_command}: {metadata}")
            if (
                metadata.get("execution_learning_blocks_completion_claim")
                and learning_command in command_queue
                and learning_closure_command in command_queue
                and command_queue.index(learning_command) > command_queue.index(learning_closure_command)
            ):
                raise SystemExit(f"Execution mission control should place after-action learning before closure: {metadata}")
            if metadata.get("draft_only") is not True or metadata.get("requires_manual_send") is not True or metadata.get("loads_without_execution") is not True:
                raise SystemExit(f"Execution mission control missed review-only contract: {metadata}")
            if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(f"Execution mission control should not authorize execution or completion claims: {metadata}")
            if metadata.get("approval_granted") is not False:
                raise SystemExit(f"Execution mission control should not grant approval: {metadata}")
            assert_execution_mission_handoff(metadata, "Execution mission control")
            if metadata.get("pending_approvals", 0) < 1 or metadata.get("blockers", 0) < 1:
                raise SystemExit(f"Execution mission control should see setup blockers: {metadata}")
            if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                raise SystemExit(f"Execution mission control missed operator-limit metadata: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Execution mission control unsafe metadata {key}: {metadata}")

        handoff_cases = [
            (
                "execution case handoff: use my computer to inspect the screen run a python script write a summary file and email me the result",
                "CASE_HANDOFF_BLOCKED_UNTIL_REVIEW",
                True,
            ),
            (
                "cockpit to case: explain Jarvis memory",
                "CASE_HANDOFF_BLOCKED_UNTIL_REVIEW",
                False,
            ),
        ]
        for case, expected_verdict, should_need_approval in handoff_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis execution case handoff packet",
                "bridge from the command cockpit to a durable execution case",
                "Handoff verdict:",
                "Mission state:",
                "Go/no-go:",
                "Ready to save case:",
                "Next safe command:",
                "Cockpit-to-case path:",
                "command cockpit:",
                "save execution case:",
                "execution case latest",
                "execution case gate latest",
                "execution case review latest",
                "Mission command queue inherited for the case:",
                "Handoff proof queue:",
                "Case handoff rules:",
                "A saved case is not completion",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Execution case handoff missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("handoff_verdict") != expected_verdict:
                raise SystemExit(f"Execution case handoff missed verdict for '{case}': {metadata}")
            if metadata.get("ready_to_save_case") is not True:
                raise SystemExit(f"Execution case handoff should allow saving a durable case: {metadata}")
            if metadata.get("approval_required") is not should_need_approval:
                raise SystemExit(f"Execution case handoff missed approval requirement for '{case}': {metadata}")
            if not str(metadata.get("cockpit_command", "")).startswith("command cockpit: "):
                raise SystemExit(f"Execution case handoff missed cockpit command: {metadata}")
            if not str(metadata.get("save_case_command", "")).startswith("save execution case: "):
                raise SystemExit(f"Execution case handoff missed save case command: {metadata}")
            handoff_queue = metadata.get("handoff_proof_queue") or []
            if metadata.get("handoff_proof_queue_count") != len(handoff_queue) or metadata.get("proof_queue") != handoff_queue:
                raise SystemExit(f"Execution case handoff missed proof queue metadata: {metadata}")
            for expected_command in [
                metadata.get("cockpit_command"),
                metadata.get("save_case_command"),
                metadata.get("inspect_command"),
                metadata.get("gate_command"),
                metadata.get("review_command"),
                metadata.get("timeline_command"),
            ]:
                if expected_command not in handoff_queue:
                    raise SystemExit(f"Execution case handoff missed queued command {expected_command}: {metadata}")
            if metadata.get("mission_command_count") != len(metadata.get("mission_command_queue") or []):
                raise SystemExit(f"Execution case handoff missed mission queue count: {metadata}")
            mission_queue = metadata.get("mission_command_queue") or []
            if metadata.get("execution_learning_blocks_completion_claim"):
                learning_command = metadata.get("learning_command")
                learning_closure_command = metadata.get("learning_closure_command")
                if learning_command not in mission_queue or learning_closure_command not in mission_queue:
                    raise SystemExit(f"Execution case handoff missed inherited learning commands: {metadata}")
                if mission_queue.index(learning_command) > mission_queue.index(learning_closure_command):
                    raise SystemExit(f"Execution case handoff should inherit after-action learning before closure: {metadata}")
            if metadata.get("pending_approvals", 0) < 1 or metadata.get("blockers", 0) < 1:
                raise SystemExit(f"Execution case handoff should see setup blockers: {metadata}")
            if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                raise SystemExit(f"Execution case handoff missed operator-limit metadata: {metadata}")
            if (
                metadata.get("draft_only") is not True
                or metadata.get("requires_manual_send") is not True
                or metadata.get("loads_without_execution") is not True
            ):
                raise SystemExit(f"Execution case handoff missed review-only contract: {metadata}")
            if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(f"Execution case handoff should not authorize execution or completion claims: {metadata}")
            if metadata.get("approval_granted") is not False:
                raise SystemExit(f"Execution case handoff should not grant approval: {metadata}")
            assert_execution_case_handoff(metadata, "Execution case handoff")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Execution case handoff unsafe metadata {key}: {metadata}")

        missing_runbook = runtime.handle("execution runbook")
        print(f"[{'ok' if missing_runbook.verified else 'blocked'}] execution runbook missing request")
        print(missing_runbook.response[:800])
        print()
        if missing_runbook.verified:
            raise SystemExit("Expected execution runbook missing-request path to refuse safely.")
        missing_runbook_metadata = missing_runbook.tool_results[0].metadata
        assert_no_request_execution_runbook_handoff(missing_runbook_metadata, "Execution runbook missing request")

        runbook_cases = [
            (
                "execution runbook: use my computer to inspect the screen run a python script write a summary file and email me the result",
                "HOLD_REVIEW_PENDING_APPROVALS",
                True,
            ),
            (
                "runbook: explain Jarvis memory",
                "HOLD_REVIEW_PENDING_APPROVALS",
                False,
            ),
        ]
        for case, expected_state, should_need_approval in runbook_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis execution runbook",
                "operational runbook",
                "Runbook state:",
                "Next safe step:",
                "execution governor:",
                "Risk posture:",
                "Before acting:",
                "During execution:",
                "Respect the operator's explicit stop times",
                "After execution:",
                "Proof surfaces available:",
                "Blockers:",
                "steering artifact",
                "does not call models",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Execution runbook missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("runbook_state") != expected_state:
                raise SystemExit(f"Execution runbook missed state for '{case}': {metadata}")
            if metadata.get("approval_required") is not should_need_approval:
                raise SystemExit(f"Execution runbook missed approval requirement for '{case}': {metadata}")
            if metadata.get("before_steps", 0) < 4 or metadata.get("during_steps", 0) < 4 or metadata.get("after_steps", 0) < 5:
                raise SystemExit(f"Execution runbook missed staged step counts: {metadata}")
            if metadata.get("before_steps") != len(metadata.get("before_step_names", [])):
                raise SystemExit(f"Execution runbook before-step names/count diverged: {metadata}")
            if metadata.get("during_steps") != len(metadata.get("during_step_names", [])):
                raise SystemExit(f"Execution runbook during-step names/count diverged: {metadata}")
            if metadata.get("after_steps") != len(metadata.get("after_step_names", [])):
                raise SystemExit(f"Execution runbook after-step names/count diverged: {metadata}")
            if not metadata.get("blocker_details") or metadata.get("blocker_count") != metadata.get("blockers"):
                raise SystemExit(f"Execution runbook missed structured blocker details: {metadata}")
            runbook_queue = metadata.get("runbook_proof_queue") or []
            if not runbook_queue or metadata.get("runbook_proof_queue_count") != len(runbook_queue):
                raise SystemExit(f"Execution runbook missed structured proof queue: {metadata}")
            if metadata.get("runbook_next_proof_command") != runbook_queue[0]:
                raise SystemExit(f"Execution runbook missed next proof command alias: {metadata}")
            if str(metadata.get("next_step") or "").strip("`") not in runbook_queue:
                raise SystemExit(f"Execution runbook proof queue missed next step: {metadata}")
            learning_closure_command = metadata.get("execution_learning_closure_command")
            if metadata.get("execution_learning_blocks_completion_claim") and (not learning_closure_command or not str(learning_closure_command).startswith("execution learning closure")):
                raise SystemExit(f"Execution runbook missed learning closure command metadata: {metadata}")
            after_action_command = metadata.get("execution_learning_after_action_learning_command")
            if metadata.get("execution_learning_blocks_completion_claim") and after_action_command != metadata.get("execution_learning_next_required_command"):
                raise SystemExit(f"Execution runbook after-action command should be the next active learning target: {metadata}")
            if (
                metadata.get("execution_learning_blocks_completion_claim")
                and learning_closure_command
                and after_action_command
                and str(learning_closure_command).removeprefix("execution learning closure ")
                != str(after_action_command).removeprefix("after-action learning packet ")
            ):
                raise SystemExit(f"Execution runbook learning commands should bind to the same target: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and learning_closure_command not in runbook_queue:
                raise SystemExit(f"Execution runbook proof queue missed learning closure command: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and (not after_action_command or not str(after_action_command).startswith("after-action learning packet")):
                raise SystemExit(f"Execution runbook missed after-action learning command metadata: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and after_action_command != metadata.get("execution_learning_actionable_next_required_command"):
                raise SystemExit(f"Execution runbook after-action command should bind to the actionable learning target: {metadata}")
            if metadata.get("execution_learning_blocks_completion_claim") and after_action_command not in runbook_queue:
                raise SystemExit(f"Execution runbook proof queue missed after-action learning command: {metadata}")
            after_action_index = runbook_queue.index(after_action_command) if after_action_command in runbook_queue else -1
            learning_closure_index = runbook_queue.index(learning_closure_command) if learning_closure_command in runbook_queue else -1
            if after_action_index >= 0 and learning_closure_index >= 0 and after_action_index > learning_closure_index:
                raise SystemExit(f"Execution runbook should place after-action learning evidence before closure: {metadata}")
            response_after_action_index = result.response.find(f"`{after_action_command}`") if after_action_command else -1
            response_learning_closure_index = result.response.find(f"`{learning_closure_command}`") if learning_closure_command else -1
            if (
                metadata.get("execution_learning_blocks_completion_claim")
                and response_after_action_index >= 0
                and response_learning_closure_index >= 0
                and response_after_action_index > response_learning_closure_index
            ):
                raise SystemExit(f"Execution runbook visible prose should place after-action learning evidence before closure: {result.response}")
            if "execution_readiness_matrix" not in metadata.get("route_packets", []) or "execution_proof_bundle" not in metadata.get("proof_packets", []):
                raise SystemExit(f"Execution runbook missed route/proof packet metadata: {metadata}")
            if "execution_recovery_packet" not in metadata.get("recovery_packets", []) or "execution_health_report" not in metadata.get("recovery_packets", []) or "failure_to_test_preview" not in metadata.get("recovery_packets", []):
                raise SystemExit(f"Execution runbook missed recovery packet metadata: {metadata}")
            assert_execution_runbook_handoff(metadata, "Execution runbook")
            if metadata.get("pending_approvals", 0) < 1 or metadata.get("blockers", 0) < 1:
                raise SystemExit(f"Execution runbook should see setup blockers: {metadata}")
            if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                raise SystemExit(f"Execution runbook missed operator-limit metadata: {metadata}")
            if (
                metadata.get("draft_only") is not True
                or metadata.get("requires_manual_send") is not True
                or metadata.get("loads_without_execution") is not True
            ):
                raise SystemExit(f"Execution runbook missed review-only contract: {metadata}")
            if (
                metadata.get("authorizes_execution") is not False
                or metadata.get("authorizes_completion_claim") is not False
                or metadata.get("approval_granted") is not False
            ):
                raise SystemExit(f"Execution runbook should not grant authority: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Execution runbook unsafe metadata {key}: {metadata}")

        no_case_inspect = runtime.handle("execution case latest")
        print(f"[{'ok' if no_case_inspect.verified else 'blocked'}] execution case latest no case")
        print(no_case_inspect.response[:1200])
        print()
        if not no_case_inspect.verified:
            raise SystemExit("Expected execution case latest no-case path to run read-only.")
        no_case_inspect_metadata = no_case_inspect.tool_results[0].metadata
        if no_case_inspect_metadata.get("found") is not False or no_case_inspect_metadata.get("case_id") is not None:
            raise SystemExit(f"Execution case inspect no-case path missed metadata: {no_case_inspect_metadata}")
        assert_no_case_inspect_handoff(no_case_inspect_metadata, "Execution case inspect no case")
        if (
            no_case_inspect_metadata.get("draft_only") is not True
            or no_case_inspect_metadata.get("requires_manual_send") is not True
            or no_case_inspect_metadata.get("loads_without_execution") is not True
        ):
            raise SystemExit(f"Execution case inspect no-case missed review-only contract: {no_case_inspect_metadata}")
        if (
            no_case_inspect_metadata.get("authorizes_execution") is not False
            or no_case_inspect_metadata.get("authorizes_completion_claim") is not False
            or no_case_inspect_metadata.get("approval_granted") is not False
        ):
            raise SystemExit(f"Execution case inspect no-case should not grant authority: {no_case_inspect_metadata}")
        for key in READ_ONLY_FLAGS:
            if no_case_inspect_metadata.get(key) is not False:
                raise SystemExit(f"Execution case inspect no-case unsafe metadata {key}: {no_case_inspect_metadata}")

        no_case_evidence_packet = runtime.handle("case evidence packet latest: verification receipt 1 confirmed the output")
        print(f"[{'ok' if no_case_evidence_packet.verified else 'blocked'}] execution case evidence packet no case")
        print(no_case_evidence_packet.response[:1200])
        print()
        if not no_case_evidence_packet.verified:
            raise SystemExit("Expected execution case evidence packet no-case path to run read-only.")
        no_case_evidence_metadata = no_case_evidence_packet.tool_results[0].metadata
        if no_case_evidence_metadata.get("found") is not False or no_case_evidence_metadata.get("verdict") != "CASE_NOT_FOUND":
            raise SystemExit(f"Execution case evidence packet no-case path missed metadata: {no_case_evidence_metadata}")
        if no_case_evidence_metadata.get("reason") != "missing_case" or no_case_evidence_metadata.get("next_command") != "save execution case: <order>":
            raise SystemExit(f"Execution case evidence packet no-case path missed recovery command: {no_case_evidence_metadata}")
        assert_execution_case_evidence_handoff(no_case_evidence_metadata, "Execution case evidence packet no case")
        assert_case_evidence_preview_contract(no_case_evidence_metadata, "Execution case evidence packet no case")
        for key in READ_ONLY_FLAGS:
            if no_case_evidence_metadata.get(key) is not False:
                raise SystemExit(f"Execution case evidence packet no-case unsafe metadata {key}: {no_case_evidence_metadata}")

        no_case_gate = runtime.handle("execution case gate latest")
        print(f"[{'ok' if no_case_gate.verified else 'blocked'}] execution case gate no case")
        print(no_case_gate.response[:1200])
        print()
        if not no_case_gate.verified:
            raise SystemExit("Expected execution case gate no-case path to run read-only.")
        no_case_gate_metadata = no_case_gate.tool_results[0].metadata
        if no_case_gate_metadata.get("found") is not False or no_case_gate_metadata.get("verdict") != "CASE_NOT_FOUND":
            raise SystemExit(f"Execution case gate no-case path missed metadata: {no_case_gate_metadata}")
        assert_no_case_gate_handoff(no_case_gate_metadata, "Execution case gate no case")
        for key in READ_ONLY_FLAGS:
            if no_case_gate_metadata.get(key) is not False:
                raise SystemExit(f"Execution case gate no-case unsafe metadata {key}: {no_case_gate_metadata}")

        no_case_review = runtime.handle("execution case review latest")
        print(f"[{'ok' if no_case_review.verified else 'blocked'}] execution case review no case")
        print(no_case_review.response[:1200])
        print()
        if not no_case_review.verified:
            raise SystemExit("Expected execution case review no-case path to run read-only.")
        no_case_review_metadata = no_case_review.tool_results[0].metadata
        if no_case_review_metadata.get("found") is not False or no_case_review_metadata.get("review_state") != "NO_CASE":
            raise SystemExit(f"Execution case review no-case path missed metadata: {no_case_review_metadata}")
        assert_no_case_review_handoff(no_case_review_metadata, "Execution case review no case")
        for key in READ_ONLY_FLAGS:
            if no_case_review_metadata.get(key) is not False:
                raise SystemExit(f"Execution case review no-case unsafe metadata {key}: {no_case_review_metadata}")

        no_case_timeline = runtime.handle("execution case timeline latest")
        print(f"[{'ok' if no_case_timeline.verified else 'blocked'}] execution case timeline no case")
        print(no_case_timeline.response[:1200])
        print()
        if not no_case_timeline.verified:
            raise SystemExit("Expected execution case timeline no-case path to run read-only.")
        no_case_timeline_metadata = no_case_timeline.tool_results[0].metadata
        if no_case_timeline_metadata.get("found") is not False or no_case_timeline_metadata.get("verdict") != "CASE_NOT_FOUND":
            raise SystemExit(f"Execution case timeline no-case path missed metadata: {no_case_timeline_metadata}")
        assert_no_case_timeline_handoff(no_case_timeline_metadata, "Execution case timeline no case")
        for key in READ_ONLY_FLAGS:
            if no_case_timeline_metadata.get(key) is not False:
                raise SystemExit(f"Execution case timeline no-case unsafe metadata {key}: {no_case_timeline_metadata}")

        no_case_closure = runtime.handle("execution case closure latest")
        print(f"[{'ok' if no_case_closure.verified else 'blocked'}] execution case closure no case")
        print(no_case_closure.response[:1200])
        print()
        if not no_case_closure.verified:
            raise SystemExit("Expected execution case closure no-case path to run read-only.")
        no_case_closure_metadata = no_case_closure.tool_results[0].metadata
        if no_case_closure_metadata.get("found") is not False or no_case_closure_metadata.get("closure_verdict") != "CASE_CLOSURE_NOT_FOUND":
            raise SystemExit(f"Execution case closure no-case path missed metadata: {no_case_closure_metadata}")
        assert_no_case_closure_handoff(no_case_closure_metadata, "Execution case closure no case")
        for key in READ_ONLY_FLAGS:
            if no_case_closure_metadata.get(key) is not False:
                raise SystemExit(f"Execution case closure no-case unsafe metadata {key}: {no_case_closure_metadata}")

        bad_case_raw = "not-a-number-" + ("x" * 120)
        path_case_raw = "/\x55sers/example/Desktop/private-case-id"
        var_case_raw = "/var/folders/zc/jarvis/private-case-id"
        tmp_case_raw = "/tmp/jarvis/private-case-id"
        bad_case_tools = [
            "inspect_execution_case",
            "execution_case_evidence_packet",
            "append_execution_case_evidence",
            "execution_case_gate",
            "execution_case_review_packet",
            "execution_case_closure_packet",
            "execution_case_timeline",
        ]
        for tool_name in bad_case_tools:
            for raw_case_value, expected_raw_case_id in (
                (bad_case_raw, "not-a-number-"),
                (path_case_raw, "<local-path>"),
                (var_case_raw, "<local-path>"),
                (tmp_case_raw, "<local-path>"),
            ):
                result = runtime.registry.get(tool_name).handler({"case_id": raw_case_value})
                metadata = result.metadata
                if result.ok or metadata.get("reason") != "bad_case_id" or metadata.get("case_id") is not None:
                    raise SystemExit(f"{tool_name} should reject malformed case ids clearly: {metadata}")
                raw_case_id = metadata.get("raw_case_id")
                if not isinstance(raw_case_id, str) or not raw_case_id.startswith(expected_raw_case_id) or len(raw_case_id) > 80:
                    raise SystemExit(f"{tool_name} missed bounded raw case id metadata: {metadata}")
                for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
                    if fragment in str(raw_case_id):
                        raise SystemExit(f"{tool_name} leaked local path raw case id metadata: {metadata}")
                if "<local-path>" in str(raw_case_id) and str(raw_case_id) != "<local-path>":
                    raise SystemExit(f"{tool_name} leaked local path raw case id metadata: {metadata}")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"{tool_name} unsafe malformed-id metadata {key}: {metadata}")

            for bad_numeric_case_id in (0, -1):
                numeric_result = runtime.registry.get(tool_name).handler({"case_id": bad_numeric_case_id})
                numeric_metadata = numeric_result.metadata
                if numeric_result.ok or "positive number" not in numeric_result.output:
                    raise SystemExit(f"{tool_name} should reject non-positive case ids clearly: {numeric_result.output}")
                if numeric_metadata.get("reason") != "bad_case_id" or numeric_metadata.get("case_id") is not None:
                    raise SystemExit(f"{tool_name} should mark non-positive case ids as bad ids: {numeric_metadata}")
                if numeric_metadata.get("raw_case_id") != str(bad_numeric_case_id):
                    raise SystemExit(f"{tool_name} should preserve non-positive raw case id metadata: {numeric_metadata}")
                for key in READ_ONLY_FLAGS:
                    if numeric_metadata.get(key) is not False:
                        raise SystemExit(f"{tool_name} unsafe non-positive-id metadata {key}: {numeric_metadata}")

        saved_case = runtime.handle("save execution case: use my computer to inspect the screen run a python script write a summary file and email me the result")
        print(f"[{'ok' if saved_case.verified else 'blocked'}] save execution case")
        print(saved_case.response[:2600])
        print()
        if not saved_case.verified:
            raise SystemExit("Expected save execution case to run local-safe.")
        for expected in [
            "Execution case #",
            "Mission state: HOLD_FOR_PENDING_APPROVAL_REVIEW",
            "Go/no-go: NO_GO",
            "Next command:",
            "Initial governor command:",
            "Mission command queue:",
            "execution proof bundle:",
            "acceptance gate:",
            "execution audit gate",
            "does not execute tools",
            "approve requests",
            "queue approvals",
        ]:
            if expected not in saved_case.response:
                raise SystemExit(f"Save execution case missing expected text: {expected}")
        saved_metadata = saved_case.tool_results[0].metadata
        saved_case_id = saved_metadata.get("case_id")
        if not isinstance(saved_case_id, int) or saved_case_id < 1:
            raise SystemExit(f"Save execution case missed case id: {saved_metadata}")
        if saved_metadata.get("mission_state") != "HOLD_FOR_PENDING_APPROVAL_REVIEW" or saved_metadata.get("go_no_go") != "NO_GO":
            raise SystemExit(f"Save execution case missed hold/no-go metadata: {saved_metadata}")
        if saved_metadata.get("approval_required") is not True:
            raise SystemExit(f"Save execution case should preserve approval requirement: {saved_metadata}")
        if not str(saved_metadata.get("initial_governor_command", "")).startswith("execution governor: "):
            raise SystemExit(f"Save execution case missed governor checkpoint command: {saved_metadata}")
        saved_queue = saved_metadata.get("mission_command_queue") or []
        if saved_metadata.get("mission_command_count") != len(saved_queue) or len(saved_queue) < 8:
            raise SystemExit(f"Save execution case missed mission command queue: {saved_metadata}")
        saved_learning_closure_command = saved_metadata.get("learning_closure_command")
        if saved_metadata.get("execution_learning_blocks_completion_claim") and (not saved_learning_closure_command or not str(saved_learning_closure_command).startswith("execution learning closure")):
            raise SystemExit(f"Save execution case missed learning closure command: {saved_metadata}")
        saved_learning_command = saved_metadata.get("learning_command")
        if saved_metadata.get("execution_learning_blocks_completion_claim") and (not saved_learning_command or not str(saved_learning_command).startswith("after-action learning packet")):
            raise SystemExit(f"Save execution case missed after-action learning command: {saved_metadata}")
        saved_expected_commands = [
            saved_metadata.get("next_command"),
            saved_metadata.get("initial_governor_command"),
            saved_metadata.get("proof_bundle_command"),
            saved_metadata.get("acceptance_command"),
            saved_metadata.get("audit_command"),
            saved_metadata.get("recovery_command"),
        ]
        if saved_metadata.get("execution_learning_blocks_completion_claim"):
            saved_expected_commands.extend([saved_learning_command, saved_learning_closure_command])
        for expected_command in saved_expected_commands:
            if expected_command not in saved_queue:
                raise SystemExit(f"Save execution case missed queued command {expected_command}: {saved_metadata}")
        if (
            saved_metadata.get("execution_learning_blocks_completion_claim")
            and saved_learning_command in saved_queue
            and saved_learning_closure_command in saved_queue
            and saved_queue.index(saved_learning_command) > saved_queue.index(saved_learning_closure_command)
        ):
            raise SystemExit(f"Save execution case should preserve after-action learning before closure: {saved_metadata}")
        risk_signals = saved_metadata.get("risk_signals", [])
        if "computer control" not in risk_signals or "shell/code" not in risk_signals or "personal data" not in risk_signals:
            raise SystemExit(f"Save execution case missed risk signals: {saved_metadata}")
        note_path = Path(str(saved_metadata.get("note_path", "")))
        if not note_path.exists() or "Execution Cases" not in str(note_path):
            raise SystemExit(f"Save execution case did not write note path: {saved_metadata}")
        note_text = note_path.read_text(encoding="utf-8")
        if "## Mission Command Queue" not in note_text or str(saved_metadata.get("proof_bundle_command")) not in note_text:
            raise SystemExit(f"Save execution case note missed mission command queue: {saved_metadata}")
        if saved_metadata.get("execution_learning_blocks_completion_claim") and (str(saved_metadata.get("learning_closure_command")) not in note_text or str(saved_metadata.get("learning_command")) not in note_text):
            raise SystemExit(f"Save execution case note missed mission command queue: {saved_metadata}")
        assert_saved_execution_case_handoff(saved_metadata, "Save execution case")
        if saved_metadata.get("local_case_creation_only") is not True:
            raise SystemExit(f"Save execution case should declare local case creation only: {saved_metadata}")
        if saved_metadata.get("authorizes_execution") is not False or saved_metadata.get("authorizes_completion_claim") is not False:
            raise SystemExit(f"Save execution case should not authorize execution/completion: {saved_metadata}")
        if saved_metadata.get("approval_granted") is not False:
            raise SystemExit(f"Save execution case should not grant approval: {saved_metadata}")
        for key in [
            "calls_model",
            "calls_external_service",
            "executes_tools",
            "reads_personal_data",
            "reads_private_data",
            "executes_side_effect",
            "external_side_effect",
            "queues_approval",
            "requires_approval",
            "controls_computer",
            "speaks",
            "completes_tasks",
        ]:
            if saved_metadata.get(key) is not False:
                raise SystemExit(f"Save execution case unsafe metadata {key}: {saved_metadata}")
        for key in ["writes_files", "writes_database", "writes_notes"]:
            if saved_metadata.get(key) is not True:
                raise SystemExit(f"Save execution case missing local write metadata {key}: {saved_metadata}")

        inspected_case = runtime.handle("execution case latest")
        print(f"[{'ok' if inspected_case.verified else 'blocked'}] execution case latest")
        print(inspected_case.response[:2600])
        print()
        if not inspected_case.verified:
            raise SystemExit("Expected execution case latest to run read-only.")
        for expected in [
            f"Execution case #{saved_case_id}:",
            "use my computer to inspect the screen",
            "mission state: HOLD_FOR_PENDING_APPROVAL_REVIEW",
            "go/no-go: NO_GO",
            "next command:",
            "Mission command queue:",
            str(saved_metadata.get("proof_bundle_command")),
            str(saved_metadata.get("learning_closure_command")),
            str(saved_metadata.get("learning_command")),
            "Recovery closure debt:",
            "next required:",
            "Execution learning debt:",
            "Risky steps still require approval readiness, last-look approval packets, approval chain proof, and linked verification receipts",
            "This inspection is read-only",
        ]:
            if expected not in inspected_case.response:
                raise SystemExit(f"Inspect execution case missing expected text: {expected}")
        if "- next proof:" in inspected_case.response:
            raise SystemExit(f"Inspect execution case should render debt queue heads as next required, not next proof: {inspected_case.response}")
        inspect_metadata = inspected_case.tool_results[0].metadata
        if inspect_metadata.get("found") is not True or inspect_metadata.get("case_id") != saved_case_id:
            raise SystemExit(f"Inspect execution case missed saved id: {inspect_metadata}")
        if inspect_metadata.get("mission_command_queue") != saved_queue or inspect_metadata.get("mission_command_count") != len(saved_queue):
            raise SystemExit(f"Inspect execution case missed mission command queue metadata: {inspect_metadata}")
        if not inspect_metadata.get("execution_health_recovery_closure_state") or "execution_health_recovery_closure_blocks_completion_claim" not in inspect_metadata:
            raise SystemExit(f"Inspect execution case missed recovery closure metadata: {inspect_metadata}")
        if not inspect_metadata.get("execution_learning_state") or "execution_learning_blocks_completion_claim" not in inspect_metadata:
            raise SystemExit(f"Inspect execution case missed execution learning metadata: {inspect_metadata}")
        if inspect_metadata.get("execution_health_recovery_closure_blocks_completion_claim") and not inspect_metadata.get("execution_health_recovery_closure_required_commands"):
            raise SystemExit(f"Inspect execution case missed recovery closure command queue: {inspect_metadata}")
        if inspect_metadata.get("execution_learning_blocks_completion_claim") and not inspect_metadata.get("execution_learning_required_commands"):
            raise SystemExit(f"Inspect execution case missed execution learning command queue: {inspect_metadata}")
        assert_execution_health_recovery_proof_aliases(inspect_metadata, "Inspect execution case")
        assert_execution_learning_proof_aliases(inspect_metadata, "Inspect execution case")
        assert_inspect_execution_case_handoff(inspect_metadata, "Inspect execution case")
        if (
            inspect_metadata.get("draft_only") is not True
            or inspect_metadata.get("requires_manual_send") is not True
            or inspect_metadata.get("loads_without_execution") is not True
        ):
            raise SystemExit(f"Inspect execution case missed review-only contract: {inspect_metadata}")
        if (
            inspect_metadata.get("authorizes_execution") is not False
            or inspect_metadata.get("authorizes_completion_claim") is not False
            or inspect_metadata.get("approval_granted") is not False
        ):
            raise SystemExit(f"Inspect execution case should not grant authority: {inspect_metadata}")
        for key in READ_ONLY_FLAGS:
            if inspect_metadata.get(key) is not False:
                raise SystemExit(f"Inspect execution case unsafe metadata {key}: {inspect_metadata}")

        missing_summary_evidence_packet = runtime.registry.get("execution_case_evidence_packet").handler(
            {"case_id": "latest", "summary": ""}
        )
        print("[ok] direct case evidence packet missing summary")
        print(missing_summary_evidence_packet.output[:1200])
        print()
        missing_summary_metadata = missing_summary_evidence_packet.metadata
        if missing_summary_metadata.get("found") is not True or missing_summary_metadata.get("case_id") != saved_case_id:
            raise SystemExit(f"Case evidence packet missing-summary path missed saved case: {missing_summary_metadata}")
        if missing_summary_metadata.get("verdict") != "EVIDENCE_SUMMARY_REQUIRED" or missing_summary_metadata.get("ready_to_append") is not False:
            raise SystemExit(f"Case evidence packet missing-summary path missed verdict: {missing_summary_metadata}")
        if not str(missing_summary_metadata.get("next_command") or "").startswith(f"case evidence packet {saved_case_id}:"):
            raise SystemExit(f"Case evidence packet missing-summary path missed next command: {missing_summary_metadata}")
        assert_execution_case_evidence_handoff(missing_summary_metadata, "Case evidence packet missing summary")
        assert_case_evidence_preview_contract(missing_summary_metadata, "Case evidence packet missing summary")
        for key in READ_ONLY_FLAGS:
            if missing_summary_metadata.get(key) is not False:
                raise SystemExit(f"Case evidence packet missing-summary unsafe metadata {key}: {missing_summary_metadata}")

        valid_evidence_packet = runtime.handle("case evidence packet latest: approval packet 1 reviewed but not approved")
        print(f"[{'ok' if valid_evidence_packet.verified else 'blocked'}] case evidence packet valid approval")
        print(valid_evidence_packet.response[:2200])
        print()
        if not valid_evidence_packet.verified:
            raise SystemExit("Expected case evidence packet to run read-only.")
        for expected in [
            "Jarvis execution case evidence packet",
            "read-only last-look packet",
            "Verdict: EVIDENCE_READY_WITH_RECEIPT",
            "Ready to append: yes",
            "Receipt target check:",
            "status: found",
            "case evidence",
            "A saved evidence event is not a completion claim",
            "does not call models",
        ]:
            if expected not in valid_evidence_packet.response:
                raise SystemExit(f"Case evidence packet missed expected text: {expected}")
        valid_evidence_metadata = valid_evidence_packet.tool_results[0].metadata
        if valid_evidence_metadata.get("ready_to_append") is not True or valid_evidence_metadata.get("receipt_kind_normalized") != "approval_packet" or valid_evidence_metadata.get("receipt_id") != "1":
            raise SystemExit(f"Case evidence packet missed inferred approval receipt: {valid_evidence_metadata}")
        if valid_evidence_metadata.get("receipt_target_status") != "found" or valid_evidence_metadata.get("receipt_target_exists") is not True:
            raise SystemExit(f"Case evidence packet missed approval target existence: {valid_evidence_metadata}")
        if not str(valid_evidence_metadata.get("append_command", "")).startswith(f"case evidence {saved_case_id}: "):
            raise SystemExit(f"Case evidence packet missed append command: {valid_evidence_metadata}")
        assert_execution_case_evidence_handoff(valid_evidence_metadata, "Case evidence packet valid approval")
        assert_case_evidence_preview_contract(valid_evidence_metadata, "Case evidence packet valid approval")
        for key in READ_ONLY_FLAGS:
            if valid_evidence_metadata.get(key) is not False:
                raise SystemExit(f"Case evidence packet unsafe metadata {key}: {valid_evidence_metadata}")

        missing_evidence_packet = runtime.handle("case evidence packet latest: verification receipt 99999 confirmed the fake target")
        print(f"[{'ok' if missing_evidence_packet.verified else 'blocked'}] case evidence packet missing target")
        print(missing_evidence_packet.response[:2200])
        print()
        if not missing_evidence_packet.verified:
            raise SystemExit("Expected missing-target case evidence packet to run read-only.")
        missing_evidence_metadata = missing_evidence_packet.tool_results[0].metadata
        if missing_evidence_metadata.get("verdict") != "EVIDENCE_RECEIPT_TARGET_MISSING" or missing_evidence_metadata.get("ready_to_append") is not False:
            raise SystemExit(f"Case evidence packet missed missing target block: {missing_evidence_metadata}")
        if missing_evidence_metadata.get("receipt_target_status") != "missing" or missing_evidence_metadata.get("receipt_target_exists") is not False:
            raise SystemExit(f"Case evidence packet missed missing target metadata: {missing_evidence_metadata}")
        if missing_evidence_metadata.get("append_command"):
            raise SystemExit(f"Case evidence packet should not provide append command for missing target: {missing_evidence_metadata}")
        assert_execution_case_evidence_handoff(missing_evidence_metadata, "Case evidence packet missing target")
        assert_case_evidence_preview_contract(missing_evidence_metadata, "Case evidence packet missing target")
        for key in READ_ONLY_FLAGS:
            if missing_evidence_metadata.get(key) is not False:
                raise SystemExit(f"Case evidence packet missing target unsafe metadata {key}: {missing_evidence_metadata}")

        conflict_evidence_packet = runtime.registry.get("execution_case_evidence_packet").handler(
            {
                "case_id": "latest",
                "summary": "verification receipt 3 confirmed the output",
                "receipt_kind": "approval_packet",
                "receipt_id": "1",
            }
        )
        print("[ok] direct case evidence packet conflict")
        print(conflict_evidence_packet.output[:1800])
        print()
        conflict_metadata = conflict_evidence_packet.metadata
        if conflict_metadata.get("verdict") != "EVIDENCE_CONFLICT" or conflict_metadata.get("ready_to_append") is not False:
            raise SystemExit(f"Case evidence packet missed receipt conflict: {conflict_metadata}")
        if conflict_metadata.get("conflict_reason") != "receipt_kind_conflict":
            raise SystemExit(f"Case evidence packet missed conflict reason: {conflict_metadata}")
        assert_execution_case_evidence_handoff(conflict_metadata, "Case evidence packet conflict")
        assert_case_evidence_preview_contract(conflict_metadata, "Case evidence packet conflict")
        for key in READ_ONLY_FLAGS:
            if conflict_metadata.get(key) is not False:
                raise SystemExit(f"Case evidence packet conflict unsafe metadata {key}: {conflict_metadata}")

        case_gate_before_evidence = runtime.handle("execution case gate")
        print(f"[{'ok' if case_gate_before_evidence.verified else 'blocked'}] execution case gate before evidence")
        print(case_gate_before_evidence.response[:2600])
        print()
        if not case_gate_before_evidence.verified:
            raise SystemExit("Expected execution case gate to run read-only.")
        for expected in [
            "Jarvis execution case gate",
            f"Case: #{saved_case_id}",
            "Verdict: CASE_HELD_FOR_APPROVAL",
            "evidence events: 0",
            "Case proof gaps:",
            "mission: missing",
            "approval evidence: missing",
            "would queue new approvals if sent: 0",
            "Execution health recovery closure:",
            "Execution learning debt:",
            "no execution case evidence events are attached",
            "Mission command queue:",
            "This gate is read-only",
        ]:
            if expected not in case_gate_before_evidence.response:
                raise SystemExit(f"Execution case gate missing expected text before evidence: {expected}")
        gate_metadata = case_gate_before_evidence.tool_results[0].metadata
        if gate_metadata.get("case_id") != saved_case_id or gate_metadata.get("verdict") != "CASE_HELD_FOR_APPROVAL":
            raise SystemExit(f"Execution case gate missed saved id/verdict before evidence: {gate_metadata}")
        if gate_metadata.get("events") != 0 or gate_metadata.get("has_verification_evidence") is not False:
            raise SystemExit(f"Execution case gate missed empty evidence posture: {gate_metadata}")
        for proof_name in ["mission", "evidence", "approval evidence", "approval chain", "approved run verification", "verification"]:
            if proof_name not in gate_metadata.get("missing_case_proofs", []):
                raise SystemExit(f"Execution case gate missed missing proof `{proof_name}`: {gate_metadata}")
        if not gate_metadata.get("next_case_proof_commands") or gate_metadata.get("next_case_proof_command") != gate_metadata["next_case_proof_commands"][0]:
            raise SystemExit(f"Execution case gate missed machine-readable next proof command: {gate_metadata}")
        if gate_metadata.get("mission_command_queue") != saved_queue or gate_metadata.get("mission_command_count") != len(saved_queue):
            raise SystemExit(f"Execution case gate missed mission command queue metadata: {gate_metadata}")
        if (
            gate_metadata.get("forecast_queue_before") != 1
            or gate_metadata.get("forecast_queue_after_if_sent") != 1
            or gate_metadata.get("forecast_queue_delta_if_sent") != 0
            or gate_metadata.get("forecast_new_approvals") != 0
            or gate_metadata.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Execution case gate missed broad-case approval forecast: {gate_metadata}")
        if not gate_metadata.get("execution_health_recovery_closure_state") or "execution_health_recovery_closure_blocks_completion_claim" not in gate_metadata:
            raise SystemExit(f"Execution case gate missed recovery closure metadata: {gate_metadata}")
        if not gate_metadata.get("execution_learning_state") or "execution_learning_blocks_completion_claim" not in gate_metadata:
            raise SystemExit(f"Execution case gate missed execution learning metadata: {gate_metadata}")
        if gate_metadata.get("execution_health_recovery_closure_blocks_completion_claim") is True and "recovery closure" not in gate_metadata.get("missing_case_proofs", []):
            raise SystemExit(f"Execution case gate missed recovery closure proof gap: {gate_metadata}")
        if gate_metadata.get("execution_learning_blocks_completion_claim") is True and "execution learning" not in gate_metadata.get("missing_case_proofs", []):
            raise SystemExit(f"Execution case gate missed execution learning proof gap: {gate_metadata}")
        assert_execution_health_recovery_proof_aliases(gate_metadata, "Execution case gate")
        assert_execution_learning_proof_aliases(gate_metadata, "Execution case gate")
        for expected_command in gate_metadata.get("execution_health_recovery_closure_required_commands", []) + gate_metadata.get("execution_learning_required_commands", []):
            if expected_command and expected_command not in gate_metadata.get("next_case_proof_commands", []):
                raise SystemExit(f"Execution case gate missed debt proof command {expected_command}: {gate_metadata}")
        if gate_metadata.get("pending_approvals", 0) < 1 or gate_metadata.get("blockers", 0) < 3:
            raise SystemExit(f"Execution case gate should preserve approval/evidence blockers: {gate_metadata}")
        assert_execution_case_gate_handoff(gate_metadata, "Execution case gate")
        for key in READ_ONLY_FLAGS:
            if gate_metadata.get(key) is not False:
                raise SystemExit(f"Execution case gate unsafe metadata {key}: {gate_metadata}")

        appended_case = runtime.handle(f"case evidence {saved_case_id}: verification receipt 3 confirmed the route is still approval-held and no risky action ran")
        print(f"[{'ok' if appended_case.verified else 'blocked'}] case evidence latest")
        print(appended_case.response[:2200])
        print()
        if not appended_case.verified:
            raise SystemExit("Expected case evidence latest to run local-safe.")
        for expected in [
            f"Execution case #{saved_case_id} evidence event #",
            "Type: evidence",
            "verification receipt 3 confirmed",
            "does not execute tools",
            "queue approvals",
        ]:
            if expected not in appended_case.response:
                raise SystemExit(f"Append execution case evidence missing expected text: {expected}")
        append_metadata = appended_case.tool_results[0].metadata
        if append_metadata.get("case_id") != saved_case_id or not isinstance(append_metadata.get("event_id"), int):
            raise SystemExit(f"Append execution case evidence missed ids: {append_metadata}")
        if append_metadata.get("event_type") != "evidence" or "verification receipt 3" not in append_metadata.get("summary", ""):
            raise SystemExit(f"Append execution case evidence missed summary metadata: {append_metadata}")
        if append_metadata.get("receipt_kind") != "verification_receipt" or append_metadata.get("receipt_id") != "3":
            raise SystemExit(f"Append execution case evidence missed inferred verification receipt metadata: {append_metadata}")
        if append_metadata.get("inferred_receipt_kind") != "verification_receipt" or append_metadata.get("inferred_receipt_id") != "3":
            raise SystemExit(f"Append execution case evidence missed inferred receipt fields: {append_metadata}")
        if append_metadata.get("evidence_preview_verdict") != "EVIDENCE_READY_WITH_RECEIPT" or append_metadata.get("evidence_preview_ready_to_append") is not True:
            raise SystemExit(f"Append execution case evidence missed preview gate metadata: {append_metadata}")
        if (append_metadata.get("execution_case_evidence_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"Append execution case evidence missed preview handoff source: {append_metadata}")
        assert_append_case_evidence_contract(append_metadata, "Append execution case evidence")
        appended_note = Path(str(append_metadata.get("note_path", "")))
        if not appended_note.exists() or "Evidence Event" not in appended_note.read_text(encoding="utf-8"):
            raise SystemExit(f"Append execution case evidence did not append note event: {append_metadata}")
        for key in NON_WRITE_AUTHORITY_FLAGS:
            if append_metadata.get(key) is not False:
                raise SystemExit(f"Append execution case evidence unsafe metadata {key}: {append_metadata}")
        for key in ["writes_files", "writes_database", "writes_notes"]:
            if append_metadata.get(key) is not True:
                raise SystemExit(f"Append execution case evidence missing local write metadata {key}: {append_metadata}")

        inspected_case_with_event = runtime.handle("execution case latest")
        print(f"[{'ok' if inspected_case_with_event.verified else 'blocked'}] execution case latest with evidence")
        print(inspected_case_with_event.response[:2600])
        print()
        if "Evidence log:" not in inspected_case_with_event.response or "verification receipt 3 confirmed" not in inspected_case_with_event.response:
            raise SystemExit("Inspect execution case did not show appended evidence log.")
        if inspected_case_with_event.tool_results[0].metadata.get("events", 0) < 1:
            raise SystemExit(f"Inspect execution case missed event count: {inspected_case_with_event.tool_results[0].metadata}")

        case_gate_with_evidence = runtime.handle("case gate latest")
        print(f"[{'ok' if case_gate_with_evidence.verified else 'blocked'}] case gate latest with evidence")
        print(case_gate_with_evidence.response[:2600])
        print()
        if not case_gate_with_evidence.verified:
            raise SystemExit("Expected case gate latest to run read-only.")
        for expected in [
            "Jarvis execution case gate",
            "Verdict: CASE_HELD_FOR_APPROVAL",
            "evidence events: 1",
            "verification evidence: present",
            "approval evidence: missing",
            "approval chain: missing",
            "verified approved runs: none",
            "would queue new approvals if sent: 0",
            "would reuse pending approval ids: none",
            "Execution health recovery closure:",
            "Execution learning debt:",
            "Case proof gaps:",
            "approval evidence",
            "Mission command queue:",
            "Next safe command:",
        ]:
            if expected not in case_gate_with_evidence.response:
                raise SystemExit(f"Execution case gate missing expected text with evidence: {expected}")
        gate_with_evidence_metadata = case_gate_with_evidence.tool_results[0].metadata
        if gate_with_evidence_metadata.get("case_id") != saved_case_id or gate_with_evidence_metadata.get("verdict") != "CASE_HELD_FOR_APPROVAL":
            raise SystemExit(f"Execution case gate missed saved id/verdict with evidence: {gate_with_evidence_metadata}")
        if gate_with_evidence_metadata.get("events") < 1 or gate_with_evidence_metadata.get("has_verification_evidence") is not True:
            raise SystemExit(f"Execution case gate missed appended evidence posture: {gate_with_evidence_metadata}")
        if gate_with_evidence_metadata.get("evidence_preview_gate_count") != 1 or gate_with_evidence_metadata.get("evidence_preview_ready_count") != 1:
            raise SystemExit(f"Execution case gate missed saved evidence preview counts: {gate_with_evidence_metadata}")
        if gate_with_evidence_metadata.get("evidence_preview_latest_verdict") != "EVIDENCE_READY_WITH_RECEIPT":
            raise SystemExit(f"Execution case gate missed latest evidence preview verdict: {gate_with_evidence_metadata}")
        if gate_with_evidence_metadata.get("evidence_preview_verdicts") != ["EVIDENCE_READY_WITH_RECEIPT"]:
            raise SystemExit(f"Execution case gate missed evidence preview verdict list: {gate_with_evidence_metadata}")
        if (gate_with_evidence_metadata.get("evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"Execution case gate missed saved evidence preview handoff: {gate_with_evidence_metadata}")
        if gate_with_evidence_metadata.get("has_approval_evidence") is not False or gate_with_evidence_metadata.get("pending_approvals", 0) < 1:
            raise SystemExit(f"Execution case gate should remain approval-held: {gate_with_evidence_metadata}")
        if gate_with_evidence_metadata.get("has_approval_chain_evidence") is not False or gate_with_evidence_metadata.get("approval_chain_status") != "missing":
            raise SystemExit(f"Execution case gate should expose missing approval-chain proof: {gate_with_evidence_metadata}")
        if gate_with_evidence_metadata.get("has_verified_approval_run") is not False or gate_with_evidence_metadata.get("verified_approval_run_ids") != []:
            raise SystemExit(f"Execution case gate should expose missing verified-approved-run proof: {gate_with_evidence_metadata}")
        if "evidence" in gate_with_evidence_metadata.get("missing_case_proofs", []) or "verification" in gate_with_evidence_metadata.get("missing_case_proofs", []):
            raise SystemExit(f"Execution case gate should clear evidence/verification proof gaps after receipt evidence: {gate_with_evidence_metadata}")
        for proof_name in ["mission", "approval evidence", "approval chain", "approved run verification"]:
            if proof_name not in gate_with_evidence_metadata.get("missing_case_proofs", []):
                raise SystemExit(f"Execution case gate missed remaining proof gap `{proof_name}`: {gate_with_evidence_metadata}")
        if gate_with_evidence_metadata.get("mission_command_queue") != saved_queue or gate_with_evidence_metadata.get("mission_command_count") != len(saved_queue):
            raise SystemExit(f"Execution case gate with evidence missed mission command queue metadata: {gate_with_evidence_metadata}")
        if not gate_with_evidence_metadata.get("execution_health_recovery_closure_state") or "execution_health_recovery_closure_blocks_completion_claim" not in gate_with_evidence_metadata:
            raise SystemExit(f"Execution case gate with evidence missed recovery closure metadata: {gate_with_evidence_metadata}")
        if not gate_with_evidence_metadata.get("execution_learning_state") or "execution_learning_blocks_completion_claim" not in gate_with_evidence_metadata:
            raise SystemExit(f"Execution case gate with evidence missed execution learning metadata: {gate_with_evidence_metadata}")
        assert_execution_health_recovery_proof_aliases(gate_with_evidence_metadata, "Execution case gate with evidence")
        assert_execution_learning_proof_aliases(gate_with_evidence_metadata, "Execution case gate with evidence")
        for expected_command in gate_with_evidence_metadata.get("execution_health_recovery_closure_required_commands", []) + gate_with_evidence_metadata.get("execution_learning_required_commands", []):
            if expected_command and expected_command not in gate_with_evidence_metadata.get("next_case_proof_commands", []):
                raise SystemExit(f"Execution case gate with evidence missed debt proof command {expected_command}: {gate_with_evidence_metadata}")
        assert_execution_case_gate_handoff(gate_with_evidence_metadata, "Execution case gate with evidence")
        for key in READ_ONLY_FLAGS:
            if gate_with_evidence_metadata.get(key) is not False:
                raise SystemExit(f"Execution case gate with evidence unsafe metadata {key}: {gate_with_evidence_metadata}")

        ledger_with_case = runtime.handle("evidence ledger")
        print(f"[{'ok' if ledger_with_case.verified else 'blocked'}] evidence ledger with execution case")
        print(ledger_with_case.response[:3000])
        print()
        if not ledger_with_case.verified:
            raise SystemExit("Expected evidence ledger with case to run read-only.")
        for expected in [
            "Latest execution case proof state",
            f"latest case: #{saved_case_id}",
            "case verdict: CASE_HELD_FOR_APPROVAL",
            "case closure verdict: CASE_CLOSURE_BLOCKED",
            "case closure ready: False",
            "case blocks completion claim: True",
            f"mission command queue: {len(saved_queue)} command(s)",
            "evidence preflight events: 1",
            "evidence preflight ready events: 1",
            "latest evidence preflight verdict: EVIDENCE_READY_WITH_RECEIPT",
            "missing case proofs: mission, approval evidence, approval chain, approved run verification",
            "next case required command:",
            "next case closure command:",
            "case closure proof queue:",
            "case approval forecast: 0 new, 0 reused, queue after 1",
            "case recovery closure state:",
            "case execution learning state:",
        ]:
            if expected not in ledger_with_case.response:
                raise SystemExit(f"Evidence ledger with case missed expected text: {expected}")
        if "next case proof command:" in ledger_with_case.response:
            raise SystemExit("Evidence ledger with case should render next case required command.")
        ledger_with_case_metadata = ledger_with_case.tool_results[0].metadata
        if ledger_with_case_metadata.get("latest_execution_case_found") is not True or ledger_with_case_metadata.get("latest_execution_case_id") != saved_case_id:
            raise SystemExit(f"Evidence ledger with case missed latest case id: {ledger_with_case_metadata}")
        if ledger_with_case_metadata.get("latest_execution_case_verdict") != "CASE_HELD_FOR_APPROVAL" or ledger_with_case_metadata.get("latest_execution_case_ready") is not False:
            raise SystemExit(f"Evidence ledger with case missed verdict/readiness: {ledger_with_case_metadata}")
        if ledger_with_case_metadata.get("latest_execution_case_closure_verdict") != "CASE_CLOSURE_BLOCKED" or ledger_with_case_metadata.get("latest_execution_case_closure_ready") is not False:
            raise SystemExit(f"Evidence ledger with case missed closure verdict/readiness: {ledger_with_case_metadata}")
        if ledger_with_case_metadata.get("latest_execution_case_closure_blocks_completion_claim") is not True:
            raise SystemExit(f"Evidence ledger with case missed closure completion block: {ledger_with_case_metadata}")
        ledger_closure_queue = ledger_with_case_metadata.get("latest_execution_case_closure_proof_queue") or []
        if not ledger_closure_queue or ledger_with_case_metadata.get("latest_execution_case_closure_proof_queue_count") != len(ledger_closure_queue):
            raise SystemExit(f"Evidence ledger with case missed closure proof queue: {ledger_with_case_metadata}")
        if ledger_with_case_metadata.get("latest_execution_case_next_closure_proof_command") != ledger_closure_queue[0]:
            raise SystemExit(f"Evidence ledger with case missed next closure proof command: {ledger_with_case_metadata}")
        ledger_after_action_indexes = [
            index for index, command in enumerate(ledger_closure_queue)
            if str(command).startswith("after-action learning packet")
        ]
        ledger_closure_indexes = [
            index for index, command in enumerate(ledger_closure_queue)
            if str(command).startswith("execution learning closure")
        ]
        if ledger_after_action_indexes and ledger_closure_indexes and ledger_after_action_indexes[0] > ledger_closure_indexes[0]:
            raise SystemExit(f"Evidence ledger latest case closure queue should put after-action evidence before closure: {ledger_with_case_metadata}")
        if ledger_with_case_metadata.get("latest_execution_case_mission_command_count") != len(saved_queue):
            raise SystemExit(f"Evidence ledger with case missed mission queue count: {ledger_with_case_metadata}")
        if ledger_with_case_metadata.get("latest_execution_case_evidence_preview_gate_count") != 1 or ledger_with_case_metadata.get("latest_execution_case_evidence_preview_ready_count") != 1:
            raise SystemExit(f"Evidence ledger with case missed latest case evidence preview counts: {ledger_with_case_metadata}")
        if ledger_with_case_metadata.get("latest_execution_case_evidence_preview_latest_verdict") != "EVIDENCE_READY_WITH_RECEIPT":
            raise SystemExit(f"Evidence ledger with case missed latest case evidence preview verdict: {ledger_with_case_metadata}")
        if (ledger_with_case_metadata.get("latest_execution_case_evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"Evidence ledger with case missed latest case evidence preview handoff: {ledger_with_case_metadata}")
        if "approval chain" not in ledger_with_case_metadata.get("latest_execution_case_missing_proofs", []):
            raise SystemExit(f"Evidence ledger with case missed case proof gaps: {ledger_with_case_metadata}")
        if not ledger_with_case_metadata.get("latest_execution_case_next_proof_command"):
            raise SystemExit(f"Evidence ledger with case missed next proof command: {ledger_with_case_metadata}")
        if ledger_with_case_metadata.get("latest_execution_case_next_required_command") != ledger_with_case_metadata.get("latest_execution_case_next_proof_command"):
            raise SystemExit(f"Evidence ledger with case missed next required/proof command parity: {ledger_with_case_metadata}")
        if (
            ledger_with_case_metadata.get("latest_execution_case_forecast_queue_before") != 1
            or ledger_with_case_metadata.get("latest_execution_case_forecast_queue_after_if_sent") != 1
            or ledger_with_case_metadata.get("latest_execution_case_forecast_queue_delta_if_sent") != 0
            or ledger_with_case_metadata.get("latest_execution_case_forecast_new_approvals") != 0
            or ledger_with_case_metadata.get("latest_execution_case_forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Evidence ledger with case missed latest case approval forecast: {ledger_with_case_metadata}")
        if not ledger_with_case_metadata.get("latest_execution_case_recovery_closure_state"):
            raise SystemExit(f"Evidence ledger with case missed latest case recovery closure state: {ledger_with_case_metadata}")
        if not ledger_with_case_metadata.get("latest_execution_case_learning_state"):
            raise SystemExit(f"Evidence ledger with case missed latest case learning state: {ledger_with_case_metadata}")
        latest_recovery_queue = ledger_with_case_metadata.get("latest_execution_case_recovery_closure_proof_queue") or []
        latest_recovery_required = ledger_with_case_metadata.get("latest_execution_case_recovery_closure_required_commands") or []
        if latest_recovery_queue != latest_recovery_required:
            raise SystemExit(f"Evidence ledger with case missed latest case recovery proof queue alias: {ledger_with_case_metadata}")
        if ledger_with_case_metadata.get("latest_execution_case_recovery_closure_proof_queue_count") != len(latest_recovery_queue):
            raise SystemExit(f"Evidence ledger with case missed latest case recovery proof queue count: {ledger_with_case_metadata}")
        if latest_recovery_queue and ledger_with_case_metadata.get("latest_execution_case_recovery_closure_next_proof_command") != latest_recovery_queue[0]:
            raise SystemExit(f"Evidence ledger with case missed latest case recovery next proof command: {ledger_with_case_metadata}")
        latest_learning_queue = ledger_with_case_metadata.get("latest_execution_case_learning_proof_queue") or []
        latest_learning_required = ledger_with_case_metadata.get("latest_execution_case_learning_required_commands") or []
        if latest_learning_queue != latest_learning_required:
            raise SystemExit(f"Evidence ledger with case missed latest case learning proof queue alias: {ledger_with_case_metadata}")
        if ledger_with_case_metadata.get("latest_execution_case_learning_proof_queue_count") != len(latest_learning_queue):
            raise SystemExit(f"Evidence ledger with case missed latest case learning proof queue count: {ledger_with_case_metadata}")
        if latest_learning_queue and ledger_with_case_metadata.get("latest_execution_case_learning_next_proof_command") != latest_learning_queue[0]:
            raise SystemExit(f"Evidence ledger with case missed latest case learning next proof command: {ledger_with_case_metadata}")
        for expected_command in ledger_with_case_metadata.get("latest_execution_case_recovery_closure_required_commands", []) + ledger_with_case_metadata.get("latest_execution_case_learning_required_commands", []):
            if expected_command and expected_command not in ledger_with_case_metadata.get("latest_execution_case_next_proof_commands", []):
                raise SystemExit(f"Evidence ledger with case missed latest case debt proof command {expected_command}: {ledger_with_case_metadata}")

        claim_gate_with_case = runtime.handle("completion claim gate")
        print(f"[{'ok' if claim_gate_with_case.verified else 'blocked'}] completion claim gate with execution case")
        print(claim_gate_with_case.response[:3000])
        print()
        if not claim_gate_with_case.verified:
            raise SystemExit("Expected completion claim gate with case to run read-only.")
        for expected in [
            "Latest execution case proof state",
            f"latest case: #{saved_case_id}",
            "case verdict: CASE_HELD_FOR_APPROVAL",
            "case closure verdict: CASE_CLOSURE_BLOCKED",
            "case closure ready: False",
            "case blocks completion claim: True",
            f"mission command queue: {len(saved_queue)} command(s)",
            "evidence preflight events: 1",
            "evidence preflight ready events: 1",
            "latest evidence preflight verdict: EVIDENCE_READY_WITH_RECEIPT",
            f"latest execution case #{saved_case_id} has open closure proof debt",
            "case approval forecast: 0 new, 0 reused, queue after 1",
            "case recovery closure state:",
            "case execution learning state:",
            "Run `",
        ]:
            if expected not in claim_gate_with_case.response:
                raise SystemExit(f"Completion claim gate with case missed expected text: {expected}")
        claim_with_case_metadata = claim_gate_with_case.tool_results[0].metadata
        if claim_with_case_metadata.get("latest_execution_case_found") is not True or claim_with_case_metadata.get("latest_execution_case_id") != saved_case_id:
            raise SystemExit(f"Completion claim gate with case missed latest case id: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_verdict") != "CASE_HELD_FOR_APPROVAL" or claim_with_case_metadata.get("latest_execution_case_ready") is not False:
            raise SystemExit(f"Completion claim gate with case missed verdict/readiness: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_closure_verdict") != "CASE_CLOSURE_BLOCKED" or claim_with_case_metadata.get("latest_execution_case_closure_ready") is not False:
            raise SystemExit(f"Completion claim gate with case missed closure verdict/readiness: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_closure_blocks_completion_claim") is not True:
            raise SystemExit(f"Completion claim gate with case missed closure completion block: {claim_with_case_metadata}")
        claim_closure_queue = claim_with_case_metadata.get("latest_execution_case_closure_proof_queue") or []
        if not claim_closure_queue or claim_with_case_metadata.get("latest_execution_case_closure_proof_queue_count") != len(claim_closure_queue):
            raise SystemExit(f"Completion claim gate with case missed closure proof queue: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_next_closure_proof_command") != claim_closure_queue[0]:
            raise SystemExit(f"Completion claim gate with case missed next closure proof command: {claim_with_case_metadata}")
        claim_after_action_indexes = [
            index for index, command in enumerate(claim_closure_queue)
            if str(command).startswith("after-action learning packet")
        ]
        claim_closure_indexes = [
            index for index, command in enumerate(claim_closure_queue)
            if str(command).startswith("execution learning closure")
        ]
        if claim_after_action_indexes and claim_closure_indexes and claim_after_action_indexes[0] > claim_closure_indexes[0]:
            raise SystemExit(f"Completion claim gate latest case closure queue should put after-action evidence before closure: {claim_with_case_metadata}")
        completion_queue = claim_with_case_metadata.get("completion_proof_queue") or []
        if claim_after_action_indexes and claim_closure_indexes:
            after_command = claim_closure_queue[claim_after_action_indexes[0]]
            closure_command = claim_closure_queue[claim_closure_indexes[0]]
            if after_command in completion_queue and closure_command in completion_queue and completion_queue.index(after_command) > completion_queue.index(closure_command):
                raise SystemExit(f"Completion claim gate proof queue should put latest-case after-action evidence before closure: {claim_with_case_metadata}")
            if f"then recheck `{after_command}`" in claim_gate_with_case.response:
                raise SystemExit(f"Completion claim gate next-move prose should not recheck the after-action evidence command: {claim_gate_with_case.response}")
            if f"then recheck `{closure_command}`" not in claim_gate_with_case.response:
                raise SystemExit(f"Completion claim gate next-move prose missed closure recheck command {closure_command}: {claim_gate_with_case.response}")
        for expected_command in claim_closure_queue:
            if expected_command not in claim_with_case_metadata.get("completion_proof_queue", []):
                raise SystemExit(f"Completion claim gate proof queue missed closure command {expected_command}: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_mission_command_count") != len(saved_queue):
            raise SystemExit(f"Completion claim gate with case missed mission queue count: {claim_with_case_metadata}")
        claim_review_handoff = claim_with_case_metadata.get("latest_execution_case_review_handoff") or {}
        if not claim_with_case_metadata.get("latest_execution_case_review_state"):
            raise SystemExit(f"Completion claim gate with case missed latest case review state: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_review_state") != claim_review_handoff.get("review_state"):
            raise SystemExit(f"Completion claim gate with case review state diverged from handoff: {claim_with_case_metadata}")
        if not claim_with_case_metadata.get("latest_execution_case_review_verdict"):
            raise SystemExit(f"Completion claim gate with case missed latest case review verdict: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_review_verdict") != claim_review_handoff.get("verdict"):
            raise SystemExit(f"Completion claim gate with case review verdict diverged from handoff: {claim_with_case_metadata}")
        if not str(claim_with_case_metadata.get("latest_execution_case_review_next_safe_command") or ""):
            raise SystemExit(f"Completion claim gate with case missed latest case review next safe command: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_review_next_safe_command") != claim_review_handoff.get("next_safe_command"):
            raise SystemExit(f"Completion claim gate with case review next safe command diverged from handoff: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_review_handoff_present") is not True:
            raise SystemExit(f"Completion claim gate with case missed latest case review handoff flag: {claim_with_case_metadata}")
        if claim_review_handoff.get("source") != "execution_case_review_packet":
            raise SystemExit(f"Completion claim gate with case missed latest case review handoff: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_evidence_preview_gate_count") != 1 or claim_with_case_metadata.get("latest_execution_case_evidence_preview_ready_count") != 1:
            raise SystemExit(f"Completion claim gate with case missed latest case evidence preview counts: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_evidence_preview_latest_verdict") != "EVIDENCE_READY_WITH_RECEIPT":
            raise SystemExit(f"Completion claim gate with case missed latest case evidence preview verdict: {claim_with_case_metadata}")
        if (claim_with_case_metadata.get("latest_execution_case_evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"Completion claim gate with case missed latest case evidence preview handoff: {claim_with_case_metadata}")
        completion_next_with_case = runtime.handle("completion next proof")
        print(f"[{'ok' if completion_next_with_case.verified else 'blocked'}] completion next proof with execution case")
        print(completion_next_with_case.response[:3000])
        print()
        if not completion_next_with_case.verified:
            raise SystemExit("Expected completion next proof with case to run read-only.")
        completion_next_with_case_metadata = completion_next_with_case.tool_results[0].metadata
        for key in [
            "latest_execution_case_evidence_preview_gate_count",
            "latest_execution_case_evidence_preview_ready_count",
            "latest_execution_case_evidence_preview_blocked_count",
            "latest_execution_case_evidence_preview_verdicts",
            "latest_execution_case_evidence_preview_latest_verdict",
            "latest_execution_case_evidence_preview_latest_event_id",
            "latest_execution_case_evidence_preview_latest_handoff",
            "latest_execution_case_evidence_preview_latest_handoff_present",
            "latest_execution_case_review_state",
            "latest_execution_case_review_verdict",
            "latest_execution_case_review_next_safe_command",
            "latest_execution_case_review_approval_required",
            "latest_execution_case_review_checklist_items",
            "latest_execution_case_review_blocker_count",
            "latest_execution_case_review_handoff",
            "latest_execution_case_review_handoff_present",
        ]:
            if completion_next_with_case_metadata.get(key) != claim_with_case_metadata.get(key):
                raise SystemExit(f"Completion next proof with case missed latest case review/evidence field {key}: {completion_next_with_case_metadata}")
        if (completion_next_with_case_metadata.get("latest_execution_case_evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"Completion next proof with case missed latest case evidence preview handoff: {completion_next_with_case_metadata}")
        readiness_digest_with_case = runtime.handle("harness readiness digest")
        print(f"[{'ok' if readiness_digest_with_case.verified else 'blocked'}] harness readiness digest with execution case")
        print(readiness_digest_with_case.response[:3000])
        print()
        if not readiness_digest_with_case.verified:
            raise SystemExit("Expected harness readiness digest with case to run read-only.")
        readiness_digest_with_case_metadata = readiness_digest_with_case.tool_results[0].metadata
        if "latest execution case evidence preflight" not in readiness_digest_with_case.response:
            raise SystemExit(f"Harness readiness digest with case missed evidence preflight prose: {readiness_digest_with_case.response}")
        for key in [
            "latest_execution_case_evidence_preview_gate_count",
            "latest_execution_case_evidence_preview_ready_count",
            "latest_execution_case_evidence_preview_blocked_count",
            "latest_execution_case_evidence_preview_verdicts",
            "latest_execution_case_evidence_preview_latest_verdict",
            "latest_execution_case_evidence_preview_latest_event_id",
            "latest_execution_case_evidence_preview_latest_handoff",
            "latest_execution_case_evidence_preview_latest_handoff_present",
            "latest_execution_case_review_state",
            "latest_execution_case_review_verdict",
            "latest_execution_case_review_next_safe_command",
            "latest_execution_case_review_approval_required",
            "latest_execution_case_review_checklist_items",
            "latest_execution_case_review_blocker_count",
            "latest_execution_case_review_handoff",
            "latest_execution_case_review_handoff_present",
        ]:
            if readiness_digest_with_case_metadata.get(key) != completion_next_with_case_metadata.get(key):
                raise SystemExit(f"Harness readiness digest with case missed latest case review/evidence field {key}: {readiness_digest_with_case_metadata}")
        if (readiness_digest_with_case_metadata.get("latest_execution_case_evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"Harness readiness digest with case missed latest case evidence preview handoff: {readiness_digest_with_case_metadata}")
        readiness_blocker_alias = runtime.handle("what is blocking Jarvis?")
        print(f"[{'ok' if readiness_blocker_alias.verified else 'blocked'}] harness readiness blocker alias")
        print(readiness_blocker_alias.response[:3000])
        print()
        if not readiness_blocker_alias.verified:
            raise SystemExit("Expected harness readiness blocker alias to run read-only.")
        readiness_blocker_alias_metadata = readiness_blocker_alias.tool_results[0].metadata
        if readiness_blocker_alias.tool_results[0].tool_name != "harness_readiness_digest":
            raise SystemExit(f"Harness readiness blocker alias used wrong tool: {readiness_blocker_alias.tool_results[0].tool_name}")
        for key in [
            "latest_execution_case_evidence_preview_gate_count",
            "latest_execution_case_review_state",
            "latest_execution_case_review_verdict",
            "latest_execution_case_review_handoff_present",
        ]:
            if readiness_blocker_alias_metadata.get(key) != readiness_digest_with_case_metadata.get(key):
                raise SystemExit(f"Harness readiness blocker alias diverged on {key}: {readiness_blocker_alias_metadata}")
        if not any("latest execution case" in blocker for blocker in claim_with_case_metadata.get("blocker_details", [])):
            raise SystemExit(f"Completion claim gate with case missed case blocker: {claim_with_case_metadata}")
        if not claim_with_case_metadata.get("latest_execution_case_next_proof_command"):
            raise SystemExit(f"Completion claim gate with case missed next proof command: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_next_required_command") != claim_with_case_metadata.get("latest_execution_case_next_proof_command"):
            raise SystemExit(f"Completion claim gate with case missed next required/proof command parity: {claim_with_case_metadata}")
        if (
            claim_with_case_metadata.get("latest_execution_case_forecast_queue_before") != 1
            or claim_with_case_metadata.get("latest_execution_case_forecast_queue_after_if_sent") != 1
            or claim_with_case_metadata.get("latest_execution_case_forecast_queue_delta_if_sent") != 0
            or claim_with_case_metadata.get("latest_execution_case_forecast_new_approvals") != 0
            or claim_with_case_metadata.get("latest_execution_case_forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Completion claim gate with case missed latest case approval forecast: {claim_with_case_metadata}")
        if not claim_with_case_metadata.get("latest_execution_case_recovery_closure_state"):
            raise SystemExit(f"Completion claim gate with case missed latest case recovery closure state: {claim_with_case_metadata}")
        if not claim_with_case_metadata.get("latest_execution_case_learning_state"):
            raise SystemExit(f"Completion claim gate with case missed latest case learning state: {claim_with_case_metadata}")
        latest_recovery_queue = claim_with_case_metadata.get("latest_execution_case_recovery_closure_proof_queue") or []
        latest_recovery_required = claim_with_case_metadata.get("latest_execution_case_recovery_closure_required_commands") or []
        if latest_recovery_queue != latest_recovery_required:
            raise SystemExit(f"Completion claim gate with case missed latest case recovery proof queue alias: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_recovery_closure_proof_queue_count") != len(latest_recovery_queue):
            raise SystemExit(f"Completion claim gate with case missed latest case recovery proof queue count: {claim_with_case_metadata}")
        if latest_recovery_queue and claim_with_case_metadata.get("latest_execution_case_recovery_closure_next_proof_command") != latest_recovery_queue[0]:
            raise SystemExit(f"Completion claim gate with case missed latest case recovery next proof command: {claim_with_case_metadata}")
        latest_learning_queue = claim_with_case_metadata.get("latest_execution_case_learning_proof_queue") or []
        latest_learning_required = claim_with_case_metadata.get("latest_execution_case_learning_required_commands") or []
        if latest_learning_queue != latest_learning_required:
            raise SystemExit(f"Completion claim gate with case missed latest case learning proof queue alias: {claim_with_case_metadata}")
        if claim_with_case_metadata.get("latest_execution_case_learning_proof_queue_count") != len(latest_learning_queue):
            raise SystemExit(f"Completion claim gate with case missed latest case learning proof queue count: {claim_with_case_metadata}")
        if latest_learning_queue and claim_with_case_metadata.get("latest_execution_case_learning_next_proof_command") != latest_learning_queue[0]:
            raise SystemExit(f"Completion claim gate with case missed latest case learning next proof command: {claim_with_case_metadata}")
        for expected_command in claim_with_case_metadata.get("latest_execution_case_recovery_closure_required_commands", []) + claim_with_case_metadata.get("latest_execution_case_learning_required_commands", []):
            if expected_command and expected_command not in claim_with_case_metadata.get("latest_execution_case_next_proof_commands", []):
                raise SystemExit(f"Completion claim gate with case missed latest case debt proof command {expected_command}: {claim_with_case_metadata}")
        for key in READ_ONLY_FLAGS:
            if claim_with_case_metadata.get(key) is not False:
                raise SystemExit(f"Completion claim gate with case unsafe metadata {key}: {claim_with_case_metadata}")

        case_review = runtime.handle("execution case review latest")
        print(f"[{'ok' if case_review.verified else 'blocked'}] execution case review latest")
        print(case_review.response[:3000])
        print()
        if not case_review.verified:
            raise SystemExit("Expected execution case review latest to run read-only.")
        for expected in [
            "Jarvis execution case review packet",
            f"Case: #{saved_case_id}",
            "Review state: APPROVAL_REVIEW_REQUIRED",
            "Gate verdict: CASE_HELD_FOR_APPROVAL",
            "Evidence summary:",
            "verification evidence: present",
            "approval evidence: missing",
            "approval chain: missing",
            "verified approved runs: none",
            "Execution health recovery closure:",
            "Execution learning debt:",
            "Case proof gaps:",
            "Next required commands:",
            "Mission command queue:",
            "Recent case events:",
            "verification receipt 3 confirmed",
            "Review checklist:",
            "Decision guidance:",
            "Follow-up commands:",
            "does not call models",
        ]:
            if expected not in case_review.response:
                raise SystemExit(f"Execution case review packet missing expected text: {expected}")
        if "Next proof commands:" in case_review.response:
            raise SystemExit(f"Execution case review packet should render case gap handoff as next required commands: {case_review.response}")
        case_review_metadata = case_review.tool_results[0].metadata
        if case_review_metadata.get("case_id") != saved_case_id or case_review_metadata.get("review_state") != "APPROVAL_REVIEW_REQUIRED":
            raise SystemExit(f"Execution case review packet missed saved id/review state: {case_review_metadata}")
        if case_review_metadata.get("verdict") != "CASE_HELD_FOR_APPROVAL":
            raise SystemExit(f"Execution case review packet missed gate verdict: {case_review_metadata}")
        if case_review_metadata.get("events", 0) < 1 or case_review_metadata.get("has_verification_evidence") is not True:
            raise SystemExit(f"Execution case review packet missed verification evidence posture: {case_review_metadata}")
        if case_review_metadata.get("evidence_preview_gate_count") != 1 or case_review_metadata.get("evidence_preview_ready_count") != 1:
            raise SystemExit(f"Execution case review packet missed saved evidence preview counts: {case_review_metadata}")
        if case_review_metadata.get("evidence_preview_latest_verdict") != "EVIDENCE_READY_WITH_RECEIPT":
            raise SystemExit(f"Execution case review packet missed latest evidence preview verdict: {case_review_metadata}")
        if (case_review_metadata.get("evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"Execution case review packet missed saved evidence preview handoff: {case_review_metadata}")
        if case_review_metadata.get("has_approval_evidence") is not False or case_review_metadata.get("pending_approvals", 0) < 1:
            raise SystemExit(f"Execution case review packet should remain approval-held: {case_review_metadata}")
        if case_review_metadata.get("has_approval_chain_evidence") is not False or case_review_metadata.get("approval_chain_status") != "missing":
            raise SystemExit(f"Execution case review packet should expose missing approval-chain proof: {case_review_metadata}")
        if case_review_metadata.get("has_verified_approval_run") is not False or case_review_metadata.get("verified_approval_run_ids") != []:
            raise SystemExit(f"Execution case review packet should expose missing verified-approved-run proof: {case_review_metadata}")
        if (
            case_review_metadata.get("forecast_queue_before") != 1
            or case_review_metadata.get("forecast_queue_after_if_sent") != 1
            or case_review_metadata.get("forecast_queue_delta_if_sent") != 0
            or case_review_metadata.get("forecast_new_approvals") != 0
            or case_review_metadata.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Execution case review packet missed inherited approval forecast: {case_review_metadata}")
        if "approval chain" not in case_review_metadata.get("missing_case_proofs", []) or not case_review_metadata.get("next_case_proof_commands"):
            raise SystemExit(f"Execution case review packet missed proof-gap metadata: {case_review_metadata}")
        if case_review_metadata.get("mission_command_queue") != saved_queue or case_review_metadata.get("mission_command_count") != len(saved_queue):
            raise SystemExit(f"Execution case review packet missed mission command queue metadata: {case_review_metadata}")
        if case_review_metadata.get("execution_learning_state") != gate_with_evidence_metadata.get("execution_learning_state"):
            raise SystemExit(f"Execution case review packet missed inherited execution learning state: {case_review_metadata}")
        if case_review_metadata.get("execution_health_recovery_closure_state") != gate_with_evidence_metadata.get("execution_health_recovery_closure_state"):
            raise SystemExit(f"Execution case review packet missed inherited recovery closure state: {case_review_metadata}")
        assert_execution_health_recovery_proof_aliases(case_review_metadata, "Execution case review packet")
        assert_execution_learning_proof_aliases(case_review_metadata, "Execution case review packet")
        for expected_command in case_review_metadata.get("execution_health_recovery_closure_required_commands", []) + case_review_metadata.get("execution_learning_required_commands", []):
            if expected_command and expected_command not in case_review_metadata.get("next_case_proof_commands", []):
                raise SystemExit(f"Execution case review packet missed debt proof command {expected_command}: {case_review_metadata}")
        if case_review_metadata.get("checklist_items", 0) < 5:
            raise SystemExit(f"Execution case review packet missed checklist metadata: {case_review_metadata}")
        assert_execution_case_review_handoff(case_review_metadata, "Execution case review packet")
        for key in READ_ONLY_FLAGS:
            if case_review_metadata.get(key) is not False:
                raise SystemExit(f"Execution case review packet unsafe metadata {key}: {case_review_metadata}")

        case_closure = runtime.handle("execution case closure latest")
        print(f"[{'ok' if case_closure.verified else 'blocked'}] execution case closure latest")
        print(case_closure.response[:3000])
        print()
        if not case_closure.verified:
            raise SystemExit("Expected execution case closure latest to run read-only.")
        for expected in [
            "Jarvis execution case closure packet",
            f"Case: #{saved_case_id}",
            "Closure verdict: CASE_CLOSURE_BLOCKED",
            "Gate verdict: CASE_HELD_FOR_APPROVAL",
            "Review state: APPROVAL_REVIEW_REQUIRED",
            "Closure ready: no",
            "Blocks completion claim: yes",
            "Closure checklist:",
            "approval evidence: open",
            "approval chain: open",
            "approved run verification: open",
            "Proof debt:",
            "Recovery and learning:",
            "Approval and receipt closure:",
            "Mission command queue:",
            "Closure proof queue:",
            "Next closure command:",
            "does not call models",
        ]:
            if expected not in case_closure.response:
                raise SystemExit(f"Execution case closure packet missing expected text: {expected}")
        case_closure_metadata = case_closure.tool_results[0].metadata
        if case_closure_metadata.get("case_id") != saved_case_id or case_closure_metadata.get("closure_verdict") != "CASE_CLOSURE_BLOCKED":
            raise SystemExit(f"Execution case closure packet missed saved id/verdict: {case_closure_metadata}")
        if case_closure_metadata.get("gate_verdict") != "CASE_HELD_FOR_APPROVAL" or case_closure_metadata.get("review_state") != "APPROVAL_REVIEW_REQUIRED":
            raise SystemExit(f"Execution case closure packet missed inherited gate/review state: {case_closure_metadata}")
        if case_closure_metadata.get("case_closure_ready") is not False or case_closure_metadata.get("case_closure_blocks_completion_claim") is not True:
            raise SystemExit(f"Execution case closure packet missed closure blocking metadata: {case_closure_metadata}")
        if case_closure_metadata.get("evidence_preview_gate_count") != 1 or case_closure_metadata.get("evidence_preview_ready_count") != 1:
            raise SystemExit(f"Execution case closure packet missed saved evidence preview counts: {case_closure_metadata}")
        if case_closure_metadata.get("evidence_preview_latest_verdict") != "EVIDENCE_READY_WITH_RECEIPT":
            raise SystemExit(f"Execution case closure packet missed latest evidence preview verdict: {case_closure_metadata}")
        if (case_closure_metadata.get("evidence_preview_latest_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"Execution case closure packet missed saved evidence preview handoff: {case_closure_metadata}")
        for proof_name in ["mission", "approval evidence", "approval chain", "approved run verification"]:
            if proof_name not in case_closure_metadata.get("missing_case_proofs", []):
                raise SystemExit(f"Execution case closure packet missed proof gap `{proof_name}`: {case_closure_metadata}")
        if not case_closure_metadata.get("closure_proof_queue") or not case_closure_metadata.get("next_closure_proof_command"):
            raise SystemExit(f"Execution case closure packet missed proof queue: {case_closure_metadata}")
        if case_closure_metadata.get("next_closure_proof_command") != case_closure_metadata["closure_proof_queue"][0]:
            raise SystemExit(f"Execution case closure packet missed first proof command alias: {case_closure_metadata}")
        if case_closure_metadata.get("mission_command_queue") != saved_queue or case_closure_metadata.get("mission_command_count") != len(saved_queue):
            raise SystemExit(f"Execution case closure packet missed mission command queue metadata: {case_closure_metadata}")
        if case_closure_metadata.get("execution_learning_state") != gate_with_evidence_metadata.get("execution_learning_state"):
            raise SystemExit(f"Execution case closure packet missed inherited execution learning state: {case_closure_metadata}")
        if case_closure_metadata.get("execution_health_recovery_closure_state") != gate_with_evidence_metadata.get("execution_health_recovery_closure_state"):
            raise SystemExit(f"Execution case closure packet missed inherited recovery closure state: {case_closure_metadata}")
        assert_execution_health_recovery_proof_aliases(case_closure_metadata, "Execution case closure packet")
        assert_execution_learning_proof_aliases(case_closure_metadata, "Execution case closure packet")
        for expected_command in case_closure_metadata.get("execution_health_recovery_closure_required_commands", []) + case_closure_metadata.get("execution_learning_required_commands", []):
            if expected_command and expected_command not in case_closure_metadata.get("closure_proof_queue", []):
                raise SystemExit(f"Execution case closure packet missed debt proof command {expected_command}: {case_closure_metadata}")
        assert_execution_case_closure_handoff(case_closure_metadata, "Execution case closure packet")
        for key in READ_ONLY_FLAGS:
            if case_closure_metadata.get(key) is not False:
                raise SystemExit(f"Execution case closure packet unsafe metadata {key}: {case_closure_metadata}")

        case_timeline = runtime.handle("execution case timeline latest")
        print(f"[{'ok' if case_timeline.verified else 'blocked'}] execution case timeline latest")
        print(case_timeline.response[:3000])
        print()
        if not case_timeline.verified:
            raise SystemExit("Expected execution case timeline latest to run read-only.")
        for expected in [
            "Jarvis execution case timeline",
            f"Case: #{saved_case_id}",
            "Review state: APPROVAL_REVIEW_REQUIRED",
            "Gate verdict: CASE_HELD_FOR_APPROVAL",
            "Approval chain: missing",
            "Verified approved runs: none",
            "Approval forecast: 0 new, 0 reused, queue after 1",
            "Execution health recovery closure:",
            "Execution learning debt:",
            "Timeline:",
            "case saved",
            "Lifecycle checklist:",
            "Case proof gaps:",
            "Next required commands:",
            "Mission command queue:",
            "approval: missing",
            "verification: missing",
            "verification evidence present but not linked",
            "final review: missing",
            "verification receipt 3 confirmed",
            "Recent runtime context:",
            "Use this before claiming completion:",
            "does not call models",
        ]:
            if expected not in case_timeline.response:
                raise SystemExit(f"Execution case timeline missing expected text: {expected}")
        if "Next proof commands:" in case_timeline.response:
            raise SystemExit(f"Execution case timeline should render case gap handoff as next required commands: {case_timeline.response}")
        case_timeline_metadata = case_timeline.tool_results[0].metadata
        if case_timeline_metadata.get("case_id") != saved_case_id or case_timeline_metadata.get("review_state") != "APPROVAL_REVIEW_REQUIRED":
            raise SystemExit(f"Execution case timeline missed saved id/review state: {case_timeline_metadata}")
        if case_timeline_metadata.get("verdict") != "CASE_HELD_FOR_APPROVAL":
            raise SystemExit(f"Execution case timeline missed gate verdict: {case_timeline_metadata}")
        if case_timeline_metadata.get("events", 0) < 1 or case_timeline_metadata.get("recent_runs", 0) < 1:
            raise SystemExit(f"Execution case timeline missed event/run counts: {case_timeline_metadata}")
        if case_timeline_metadata.get("lifecycle_stages") != 6 or case_timeline_metadata.get("missing_stage_count", 0) < 3:
            raise SystemExit(f"Execution case timeline missed lifecycle stage metadata: {case_timeline_metadata}")
        if case_timeline_metadata.get("has_verification_evidence") is not True or case_timeline_metadata.get("has_approval_evidence") is not False:
            raise SystemExit(f"Execution case timeline missed proof posture metadata: {case_timeline_metadata}")
        if case_timeline_metadata.get("has_approval_chain_evidence") is not False or case_timeline_metadata.get("approval_chain_status") != "missing":
            raise SystemExit(f"Execution case timeline should expose missing approval-chain proof: {case_timeline_metadata}")
        if case_timeline_metadata.get("has_verified_approval_run") is not False or case_timeline_metadata.get("verified_approval_run_ids") != []:
            raise SystemExit(f"Execution case timeline should expose missing verified-approved-run proof: {case_timeline_metadata}")
        if (
            case_timeline_metadata.get("forecast_queue_before") != 1
            or case_timeline_metadata.get("forecast_queue_after_if_sent") != 1
            or case_timeline_metadata.get("forecast_queue_delta_if_sent") != 0
            or case_timeline_metadata.get("forecast_new_approvals") != 0
            or case_timeline_metadata.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"Execution case timeline missed inherited approval forecast: {case_timeline_metadata}")
        if "approval" not in case_timeline_metadata.get("missing_stages", []) or "verification" not in case_timeline_metadata.get("missing_stages", []) or "final review" not in case_timeline_metadata.get("missing_stages", []):
            raise SystemExit(f"Execution case timeline missed missing stage names: {case_timeline_metadata}")
        if "approval chain" not in case_timeline_metadata.get("missing_case_proofs", []) or not case_timeline_metadata.get("next_case_proof_commands"):
            raise SystemExit(f"Execution case timeline missed proof-gap metadata: {case_timeline_metadata}")
        if case_timeline_metadata.get("mission_command_queue") != saved_queue or case_timeline_metadata.get("mission_command_count") != len(saved_queue):
            raise SystemExit(f"Execution case timeline missed mission command queue metadata: {case_timeline_metadata}")
        if case_timeline_metadata.get("execution_learning_state") != gate_with_evidence_metadata.get("execution_learning_state"):
            raise SystemExit(f"Execution case timeline missed inherited execution learning state: {case_timeline_metadata}")
        if case_timeline_metadata.get("execution_health_recovery_closure_state") != gate_with_evidence_metadata.get("execution_health_recovery_closure_state"):
            raise SystemExit(f"Execution case timeline missed inherited recovery closure state: {case_timeline_metadata}")
        assert_execution_health_recovery_proof_aliases(case_timeline_metadata, "Execution case timeline")
        assert_execution_learning_proof_aliases(case_timeline_metadata, "Execution case timeline")
        for expected_command in case_timeline_metadata.get("execution_health_recovery_closure_required_commands", []) + case_timeline_metadata.get("execution_learning_required_commands", []):
            if expected_command and expected_command not in case_timeline_metadata.get("next_case_proof_commands", []):
                raise SystemExit(f"Execution case timeline missed debt proof command {expected_command}: {case_timeline_metadata}")
        assert_execution_case_timeline_handoff(case_timeline_metadata, "Execution case timeline")
        for key in READ_ONLY_FLAGS:
            if case_timeline_metadata.get(key) is not False:
                raise SystemExit(f"Execution case timeline unsafe metadata {key}: {case_timeline_metadata}")

        fake_approval_case = runtime.handle("case evidence latest: approval packet 1 reviewed but not approved")
        print(f"[{'ok' if fake_approval_case.verified else 'blocked'}] case evidence latest fake approval")
        print(fake_approval_case.response[:1800])
        print()
        if not fake_approval_case.verified:
            raise SystemExit("Expected fake approval case evidence to append locally.")
        fake_append_metadata = fake_approval_case.tool_results[0].metadata
        if fake_append_metadata.get("receipt_kind") != "approval_packet" or fake_append_metadata.get("receipt_id") != "1":
            raise SystemExit(f"Fake approval case evidence missed inferred approval receipt metadata: {fake_append_metadata}")
        fake_approval_gate = runtime.handle("execution case gate latest")
        print(f"[{'ok' if fake_approval_gate.verified else 'blocked'}] execution case gate fake approval")
        print(fake_approval_gate.response[:2800])
        print()
        if not fake_approval_gate.verified:
            raise SystemExit("Expected execution case gate after fake approval evidence to run read-only.")
        for expected in [
            "approval evidence: present",
            "approval chain: approval 1 pending",
            "verified approved runs: none",
            "risk signals require a proven approval chain",
        ]:
            if expected not in fake_approval_gate.response:
                raise SystemExit(f"Execution case gate should reject unproven approval evidence: {expected}")
        fake_approval_metadata = fake_approval_gate.tool_results[0].metadata
        if fake_approval_metadata.get("has_approval_evidence") is not True:
            raise SystemExit(f"Execution case gate should see textual approval evidence: {fake_approval_metadata}")
        if fake_approval_metadata.get("has_approval_chain_evidence") is not False:
            raise SystemExit(f"Execution case gate should reject approval text without a linked approved rerun: {fake_approval_metadata}")
        if fake_approval_metadata.get("approval_evidence_ids") != [1] or fake_approval_metadata.get("approval_linked_run_ids") != []:
            raise SystemExit(f"Execution case gate missed approval-chain metadata for fake evidence: {fake_approval_metadata}")
        if fake_approval_metadata.get("has_verified_approval_run") is not False or fake_approval_metadata.get("verified_approval_run_ids") != []:
            raise SystemExit(f"Execution case gate should not infer verified approved runs from fake evidence: {fake_approval_metadata}")
        for key in READ_ONLY_FLAGS:
            if fake_approval_metadata.get(key) is not False:
                raise SystemExit(f"Execution case gate fake approval unsafe metadata {key}: {fake_approval_metadata}")

        mixed_runtime_trace_case = runtime.handle(
            "case evidence latest: runtime trace receipt 99998 shows approval packet 1 was still pending"
        )
        print(f"[{'ok' if mixed_runtime_trace_case.verified else 'blocked'}] case evidence latest mixed runtime trace and approval")
        print(mixed_runtime_trace_case.response[:1800])
        print()
        if mixed_runtime_trace_case.verified:
            raise SystemExit("Expected mixed runtime-trace/approval evidence with a missing target to block before append.")
        mixed_runtime_trace_metadata = mixed_runtime_trace_case.tool_results[0].metadata
        if (
            mixed_runtime_trace_metadata.get("receipt_kind_normalized") != "runtime_trace_receipt"
            or mixed_runtime_trace_metadata.get("receipt_id") != "99998"
        ):
            raise SystemExit(
                f"Mixed runtime-trace evidence should infer the id attached to the runtime trace marker, not the approval id: {mixed_runtime_trace_metadata}"
            )
        if mixed_runtime_trace_metadata.get("receipt_target_status") != "missing" or mixed_runtime_trace_metadata.get("receipt_target_exists") is not False:
            raise SystemExit(f"Mixed runtime-trace append should preserve missing target status: {mixed_runtime_trace_metadata}")
        if mixed_runtime_trace_metadata.get("target_lookup_command") != "runtime trace receipt 99998":
            raise SystemExit(f"Mixed runtime-trace append missed runtime trace lookup command: {mixed_runtime_trace_metadata}")
        if (mixed_runtime_trace_metadata.get("execution_case_evidence_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"Mixed runtime-trace append missed evidence preview handoff: {mixed_runtime_trace_metadata}")
        assert_append_case_evidence_block_contract(mixed_runtime_trace_metadata, "Mixed runtime-trace append")
        for key in READ_ONLY_FLAGS:
            if mixed_runtime_trace_metadata.get(key) is not False:
                raise SystemExit(f"Mixed runtime-trace append unsafe metadata {key}: {mixed_runtime_trace_metadata}")

        conflict_kind = runtime.registry.get("append_execution_case_evidence").handler(
            {
                "case_id": saved_case_id,
                "summary": "verification receipt 3 confirmed the held route",
                "receipt_kind": "approval_packet",
                "receipt_id": "3",
            }
        )
        if conflict_kind.ok or conflict_kind.metadata.get("reason") != "receipt_conflict":
            raise SystemExit(f"Append execution case evidence should reject conflicting receipt kind: {conflict_kind.metadata}")
        if "receipt conflict" not in conflict_kind.output.lower() or "verification_receipt 3" not in conflict_kind.output:
            raise SystemExit(f"Append execution case evidence conflict output missed details: {conflict_kind.output}")
        assert_append_case_evidence_block_contract(conflict_kind.metadata, "Append execution case evidence conflict kind")
        for key in READ_ONLY_FLAGS:
            if conflict_kind.metadata.get(key) is not False:
                raise SystemExit(f"Append execution case evidence conflict kind unsafe metadata {key}: {conflict_kind.metadata}")

        conflict_id = runtime.registry.get("append_execution_case_evidence").handler(
            {
                "case_id": saved_case_id,
                "summary": "approval packet 1 reviewed but still pending",
                "receipt_kind": "approval_packet",
                "receipt_id": "2",
            }
        )
        if conflict_id.ok or conflict_id.metadata.get("reason") != "receipt_id_conflict":
            raise SystemExit(f"Append execution case evidence should reject conflicting receipt id: {conflict_id.metadata}")
        if "receipt id conflict" not in conflict_id.output.lower() or "approval_packet 1" not in conflict_id.output:
            raise SystemExit(f"Append execution case evidence id conflict output missed details: {conflict_id.output}")
        assert_append_case_evidence_block_contract(conflict_id.metadata, "Append execution case evidence conflict id")
        for key in READ_ONLY_FLAGS:
            if conflict_id.metadata.get(key) is not False:
                raise SystemExit(f"Append execution case evidence conflict id unsafe metadata {key}: {conflict_id.metadata}")

        missing_receipt_case = runtime.handle("case evidence latest: verification receipt 99999 confirmed the fake target")
        print(f"[{'ok' if missing_receipt_case.verified else 'blocked'}] case evidence latest missing receipt target")
        print(missing_receipt_case.response[:1600])
        print()
        if missing_receipt_case.verified:
            raise SystemExit("Expected missing-target receipt evidence to be blocked before append.")
        missing_receipt_metadata = missing_receipt_case.tool_results[0].metadata
        if missing_receipt_metadata.get("verdict") != "EVIDENCE_RECEIPT_TARGET_MISSING" or missing_receipt_metadata.get("ready_to_append") is not False:
            raise SystemExit(f"Missing-target append should preserve preview block metadata: {missing_receipt_metadata}")
        if missing_receipt_metadata.get("receipt_kind_normalized") != "verification_receipt" or missing_receipt_metadata.get("receipt_id") != "99999":
            raise SystemExit(f"Missing-target append missed inferred receipt metadata: {missing_receipt_metadata}")
        if missing_receipt_metadata.get("receipt_target_status") != "missing" or missing_receipt_metadata.get("receipt_target_exists") is not False:
            raise SystemExit(f"Missing-target append missed target status: {missing_receipt_metadata}")
        if (missing_receipt_metadata.get("execution_case_evidence_handoff") or {}).get("source") != "execution_case_evidence_packet":
            raise SystemExit(f"Missing-target append missed preview handoff source: {missing_receipt_metadata}")
        assert_append_case_evidence_block_contract(missing_receipt_metadata, "Missing-target append")
        for key in READ_ONLY_FLAGS:
            if missing_receipt_metadata.get(key) is not False:
                raise SystemExit(f"Missing-target append unsafe metadata {key}: {missing_receipt_metadata}")

        exact_command_case = runtime.handle("save execution case: run command python3 --version")
        print(f"[{'ok' if exact_command_case.verified else 'blocked'}] save exact command execution case")
        print(exact_command_case.response[:1600])
        print()
        if not exact_command_case.verified:
            raise SystemExit("Expected exact command execution case save to run local-safe.")
        exact_case_gate = runtime.handle("execution case gate latest")
        print(f"[{'ok' if exact_case_gate.verified else 'blocked'}] execution case gate exact command forecast")
        print(exact_case_gate.response[:2200])
        print()
        if not exact_case_gate.verified:
            raise SystemExit("Expected exact command execution case gate to run read-only.")
        exact_gate_metadata = exact_case_gate.tool_results[0].metadata
        if (
            exact_gate_metadata.get("forecast_queue_before") != 1
            or exact_gate_metadata.get("forecast_queue_after_if_sent") != 1
            or exact_gate_metadata.get("forecast_queue_delta_if_sent") != 0
            or exact_gate_metadata.get("forecast_new_approvals") != 0
            or exact_gate_metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Execution case gate missed exact-command approval reuse forecast: {exact_gate_metadata}")
        if "would queue new approvals if sent: 0" not in exact_case_gate.response or "would reuse pending approval ids: 1" not in exact_case_gate.response:
            raise SystemExit("Execution case gate did not render exact-command approval reuse forecast.")
        exact_case_forecast = exact_gate_metadata.get("approval_queue_forecast") or []
        if not exact_case_forecast or exact_case_forecast[0].get("tool_name") != "run_shell_command" or exact_case_forecast[0].get("existing_approval_id") != 1 or exact_case_forecast[0].get("would_reuse_pending_approval") is not True:
            raise SystemExit(f"Execution case gate missed per-action approval reuse forecast: {exact_gate_metadata}")
        exact_case_review = runtime.handle("execution case review latest")
        if "would queue new approvals if sent: 0" not in exact_case_review.response or "would reuse pending approval ids: 1" not in exact_case_review.response:
            raise SystemExit("Execution case review did not render inherited exact-command approval reuse forecast.")
        exact_review_metadata = exact_case_review.tool_results[0].metadata
        if (
            exact_review_metadata.get("forecast_queue_before") != 1
            or exact_review_metadata.get("forecast_queue_after_if_sent") != 1
            or exact_review_metadata.get("forecast_queue_delta_if_sent") != 0
            or exact_review_metadata.get("forecast_new_approvals") != 0
            or exact_review_metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Execution case review missed exact-command approval reuse forecast: {exact_review_metadata}")
        exact_case_timeline = runtime.handle("execution case timeline latest")
        if "Approval forecast: 0 new, 1 reused, queue after 1" not in exact_case_timeline.response:
            raise SystemExit("Execution case timeline did not render inherited exact-command approval reuse forecast.")
        exact_timeline_metadata = exact_case_timeline.tool_results[0].metadata
        if (
            exact_timeline_metadata.get("forecast_queue_before") != 1
            or exact_timeline_metadata.get("forecast_queue_after_if_sent") != 1
            or exact_timeline_metadata.get("forecast_queue_delta_if_sent") != 0
            or exact_timeline_metadata.get("forecast_new_approvals") != 0
            or exact_timeline_metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Execution case timeline missed exact-command approval reuse forecast: {exact_timeline_metadata}")
        exact_case_ledger = runtime.handle("evidence ledger")
        if "case approval forecast: 0 new, 1 reused, queue after 1" not in exact_case_ledger.response:
            raise SystemExit("Evidence ledger did not render inherited exact-command latest-case approval reuse forecast.")
        exact_ledger_metadata = exact_case_ledger.tool_results[0].metadata
        if (
            exact_ledger_metadata.get("latest_execution_case_forecast_queue_before") != 1
            or exact_ledger_metadata.get("latest_execution_case_forecast_queue_after_if_sent") != 1
            or exact_ledger_metadata.get("latest_execution_case_forecast_queue_delta_if_sent") != 0
            or exact_ledger_metadata.get("latest_execution_case_forecast_new_approvals") != 0
            or exact_ledger_metadata.get("latest_execution_case_forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Evidence ledger missed exact-command latest-case approval reuse forecast: {exact_ledger_metadata}")
        exact_case_claim = runtime.handle("completion claim gate")
        if "case approval forecast: 0 new, 1 reused, queue after 1" not in exact_case_claim.response:
            raise SystemExit("Completion claim gate did not render inherited exact-command latest-case approval reuse forecast.")
        exact_claim_metadata = exact_case_claim.tool_results[0].metadata
        if (
            exact_claim_metadata.get("latest_execution_case_forecast_queue_before") != 1
            or exact_claim_metadata.get("latest_execution_case_forecast_queue_after_if_sent") != 1
            or exact_claim_metadata.get("latest_execution_case_forecast_queue_delta_if_sent") != 0
            or exact_claim_metadata.get("latest_execution_case_forecast_new_approvals") != 0
            or exact_claim_metadata.get("latest_execution_case_forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"Completion claim gate missed exact-command latest-case approval reuse forecast: {exact_claim_metadata}")

        initial_evidence_case = runtime.registry.get("save_execution_case").handler(
            {
                "request": "summarize local harness smoke evidence",
                "initial_evidence": "verification smoke_test_harness confirmed the case save path records initial evidence",
            }
        )
        print("[ok] direct save execution case with initial evidence")
        print(initial_evidence_case.output[:1800])
        print()
        if not initial_evidence_case.ok:
            raise SystemExit(f"Expected initial-evidence case save to succeed: {initial_evidence_case.output}")
        initial_evidence_metadata = initial_evidence_case.metadata
        initial_evidence_case_id = initial_evidence_metadata.get("case_id")
        if initial_evidence_metadata.get("initial_evidence_supplied") is not True:
            raise SystemExit(f"Save execution case missed initial evidence supplied metadata: {initial_evidence_metadata}")
        if initial_evidence_metadata.get("initial_evidence_attached") is not True or initial_evidence_metadata.get("evidence_events_created") != 1:
            raise SystemExit(f"Save execution case did not attach initial evidence: {initial_evidence_metadata}")
        if not isinstance(initial_evidence_metadata.get("initial_evidence_event_id"), int):
            raise SystemExit(f"Save execution case missed initial evidence event id: {initial_evidence_metadata}")
        if initial_evidence_metadata.get("initial_evidence_receipt_target_status") != "not_referenced":
            raise SystemExit(f"Save execution case should mark receipt target not referenced: {initial_evidence_metadata}")
        assert_saved_execution_case_handoff(initial_evidence_metadata, "Save execution case with initial evidence")
        initial_note = Path(str(initial_evidence_metadata.get("note_path", ""))).read_text(encoding="utf-8")
        if "## Evidence Event" not in initial_note or "case save path records initial evidence" not in initial_note:
            raise SystemExit(f"Save execution case note missed initial evidence event: {initial_evidence_metadata}")
        initial_evidence_gate = runtime.registry.get("execution_case_gate").handler({"case_id": initial_evidence_case_id})
        if not initial_evidence_gate.ok:
            raise SystemExit(f"Expected initial-evidence case gate to run read-only: {initial_evidence_gate.output}")
        initial_evidence_gate_metadata = initial_evidence_gate.metadata
        if initial_evidence_gate_metadata.get("events") != 1:
            raise SystemExit(f"Execution case gate missed initial evidence event count: {initial_evidence_gate_metadata}")
        if initial_evidence_gate_metadata.get("has_verification_evidence") is not True:
            raise SystemExit(f"Execution case gate missed initial verification evidence: {initial_evidence_gate_metadata}")
        for key in READ_ONLY_FLAGS:
            if initial_evidence_gate_metadata.get(key) is not False:
                raise SystemExit(f"Execution case gate initial evidence unsafe metadata {key}: {initial_evidence_gate_metadata}")

        skipped_initial_evidence_case = runtime.registry.get("save_execution_case").handler(
            {
                "request": "summarize missing receipt handling",
                "initial_evidence": "verification receipt 999999 confirmed a fake target",
            }
        )
        print("[ok] direct save execution case skips missing initial evidence target")
        print(skipped_initial_evidence_case.output[:1800])
        print()
        if not skipped_initial_evidence_case.ok:
            raise SystemExit(f"Expected missing-target initial evidence case save to still save case: {skipped_initial_evidence_case.output}")
        skipped_initial_metadata = skipped_initial_evidence_case.metadata
        if skipped_initial_metadata.get("initial_evidence_supplied") is not True:
            raise SystemExit(f"Save execution case missed skipped initial evidence supplied metadata: {skipped_initial_metadata}")
        if skipped_initial_metadata.get("initial_evidence_attached") is not False or skipped_initial_metadata.get("evidence_events_created") != 0:
            raise SystemExit(f"Save execution case should skip missing-target initial evidence: {skipped_initial_metadata}")
        if skipped_initial_metadata.get("initial_evidence_receipt_target_status") != "missing":
            raise SystemExit(f"Save execution case missed missing initial receipt target status: {skipped_initial_metadata}")
        if skipped_initial_metadata.get("initial_evidence_skip_reason") != "receipt_target_missing":
            raise SystemExit(f"Save execution case missed missing-target skip reason: {skipped_initial_metadata}")
        assert_saved_execution_case_handoff(skipped_initial_metadata, "Save execution case skipped initial evidence")
        skipped_gate = runtime.registry.get("execution_case_gate").handler({"case_id": skipped_initial_metadata.get("case_id")})
        if skipped_gate.metadata.get("events") != 0:
            raise SystemExit(f"Execution case gate should not see skipped initial evidence event: {skipped_gate.metadata}")
        for key in READ_ONLY_FLAGS:
            if skipped_initial_metadata.get(key) is not False and key not in {"writes_files", "writes_database", "writes_notes"}:
                raise SystemExit(f"Save execution case skipped initial evidence unsafe metadata {key}: {skipped_initial_metadata}")
        for key in ["writes_files", "writes_database", "writes_notes"]:
            if skipped_initial_metadata.get(key) is not True:
                raise SystemExit(f"Save execution case skipped initial evidence missed write metadata {key}: {skipped_initial_metadata}")

        command_initial_evidence_case = runtime.handle(
            "save execution case: request: review local case command parser; evidence: verification smoke_test_harness confirmed command parser carries initial evidence"
        )
        print(f"[{'ok' if command_initial_evidence_case.verified else 'blocked'}] command save execution case with initial evidence")
        print(command_initial_evidence_case.response[:1800])
        print()
        if not command_initial_evidence_case.verified:
            raise SystemExit("Expected command initial-evidence case save to run local-safe.")
        command_initial_metadata = command_initial_evidence_case.tool_results[0].metadata
        if command_initial_metadata.get("request") != "review local case command parser":
            raise SystemExit(f"Command parser should keep request separate from initial evidence: {command_initial_metadata}")
        if command_initial_metadata.get("initial_evidence_attached") is not True or command_initial_metadata.get("evidence_events_created") != 1:
            raise SystemExit(f"Command parser did not carry initial evidence into save_execution_case: {command_initial_metadata}")
        assert_saved_execution_case_handoff(command_initial_metadata, "Command save execution case with initial evidence")
        command_initial_gate = runtime.handle("execution case gate latest")
        if command_initial_gate.tool_results[0].metadata.get("events") != 1:
            raise SystemExit(f"Latest command-parsed case missed initial evidence event: {command_initial_gate.tool_results[0].metadata}")
        if command_initial_gate.tool_results[0].metadata.get("has_verification_evidence") is not True:
            raise SystemExit(f"Latest command-parsed case missed verification evidence: {command_initial_gate.tool_results[0].metadata}")
        command_initial_gate_metadata = command_initial_gate.tool_results[0].metadata
        if command_initial_gate_metadata.get("go_no_go") == "PREFLIGHT_OK" and "mission" in command_initial_gate_metadata.get("missing_case_proofs", []):
            raise SystemExit(f"Local-safe PREFLIGHT_OK case should not keep mission proof open: {command_initial_gate_metadata}")
        command_initial_closure = runtime.handle("execution case closure latest")
        command_initial_closure_metadata = command_initial_closure.tool_results[0].metadata
        if (
            command_initial_closure_metadata.get("gate_verdict") == "CASE_READY_FOR_HUMAN_REVIEW"
            and "mission" in command_initial_closure_metadata.get("missing_case_proofs", [])
        ):
            raise SystemExit(f"Closure should not keep mission open after the gate is review-ready: {command_initial_closure_metadata}")

        doctrine_cases = ["harness doctrine", "what is an agent harness"]
        for case in doctrine_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            for expected in [
                "Jarvis agent harness doctrine",
                "A stronger AI brain is not enough by itself",
                "Runtime over brain alone",
                "Command-first operation",
                "Human-in-the-loop brakes",
                "Operator timeboxes outrank goals",
                "Lifecycle hooks",
                "Tool orchestration",
                "Progressive disclosure",
                "Coding-agent discipline",
                "Typed AGI harness contract",
                "Intelligence",
                "Engine",
                "Agents",
                "Tools+Memory",
                "Learning",
                "Model drafts do not authorize execution",
                "Internal workers are capacity, not companion personas",
                "Learning artifacts are evidence, not self-approval",
                "think before coding",
                "simple implementations",
                "surgical changes",
                "verified success criteria",
                "does not override approval gates",
                "does not override the operator's explicit stop times",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Harness doctrine missing expected text for '{case}': {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("principles") != 8 or metadata.get("priority") != "finish_jarvis_agent_harness":
                raise SystemExit(f"Harness doctrine missed metadata: {metadata}")
            if metadata.get("doctrine_row_count") != metadata.get("principles") or len(metadata.get("doctrine_rows") or []) != metadata.get("principles"):
                raise SystemExit(f"Harness doctrine missed doctrine row parity: {metadata}")
            if not metadata.get("build_filter") or metadata.get("build_filter_count") != len(metadata.get("build_filter") or []):
                raise SystemExit(f"Harness doctrine missed build-filter parity: {metadata}")
            assert_harness_layer_contract(metadata, "Harness doctrine")
            if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                raise SystemExit(f"Harness doctrine missed stop-time override metadata: {metadata}")
            if metadata.get("draft_only") is not True or metadata.get("requires_manual_send") is not True or metadata.get("loads_without_execution") is not True:
                raise SystemExit(f"Harness doctrine missed review-only contract: {metadata}")
            if metadata.get("authorizes_execution") is not False or metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(f"Harness doctrine should not authorize execution or completion claims: {metadata}")
            if metadata.get("approval_granted") is not False:
                raise SystemExit(f"Harness doctrine should not grant approval: {metadata}")
            assert_harness_doctrine_handoff(metadata, "Harness doctrine")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Harness doctrine unsafe metadata {key}: {metadata}")

        discipline_result = runtime.handle("coding discipline: improve Jarvis recovery tests")
        print(f"[ok={discipline_result.verified}] coding discipline")
        print(discipline_result.response[:1800])
        print()
        if not discipline_result.verified:
            raise SystemExit("Coding discipline packet should run read-only.")
        for expected in [
            "Jarvis coding discipline packet",
            "Think before coding",
            "Simplicity first",
            "Surgical changes",
            "Goal-driven execution",
            "authorizes edits no",
            "authorizes risky work no",
        ]:
            if expected not in discipline_result.response:
                raise SystemExit(f"Coding discipline packet missing expected text: {expected}")
        discipline_metadata = discipline_result.tool_results[0].metadata
        if discipline_metadata.get("discipline_row_count") != 4:
            raise SystemExit(f"Coding discipline packet should expose four discipline rows: {discipline_metadata}")
        if discipline_metadata.get("ready_rows") != 1 or discipline_metadata.get("review_rows") != 3:
            raise SystemExit(f"Coding discipline packet should expose ready/review row counts: {discipline_metadata}")
        if discipline_metadata.get("proof_queue_count") != len(discipline_metadata.get("proof_queue") or []):
            raise SystemExit(f"Coding discipline packet should expose proof queue parity: {discipline_metadata}")
        if discipline_metadata.get("next_proof_command") != "read owning files for: <target files>":
            raise SystemExit(f"Coding discipline packet should expose next proof command: {discipline_metadata}")
        if discipline_metadata.get("ready_for_implementation_review") is not False:
            raise SystemExit(f"Coding discipline packet should wait for verification and file scope before implementation review: {discipline_metadata}")
        if (
            discipline_metadata.get("authorizes_execution") is not False
            or discipline_metadata.get("authorizes_edits") is not False
            or discipline_metadata.get("authorizes_risky_work") is not False
            or discipline_metadata.get("authorizes_test_execution") is not False
            or discipline_metadata.get("authorizes_completion_claim") is not False
        ):
            raise SystemExit(f"Coding discipline packet should stay non-authorizing: {discipline_metadata}")
        if discipline_metadata.get("draft_only") is not True or discipline_metadata.get("requires_manual_send") is not True or discipline_metadata.get("loads_without_execution") is not True:
            raise SystemExit(f"Coding discipline packet missed review-only contract: {discipline_metadata}")
        if discipline_metadata.get("approval_granted") is not False:
            raise SystemExit(f"Coding discipline packet should not grant approval: {discipline_metadata}")
        assert_coding_discipline_handoff(discipline_metadata, "Coding discipline packet")
        for key in READ_ONLY_FLAGS:
            if discipline_metadata.get(key) is not False:
                raise SystemExit(f"Coding discipline packet unsafe metadata {key}: {discipline_metadata}")

        tools = runtime.handle("list tools core")
        print(f"[ok={tools.verified}] list tools core")
        print(tools.response[:1800])
        print()
        if not tools.verified or "priority_goal" not in tools.response or "harness_status" not in tools.response or "harness_doctrine" not in tools.response or "coding_discipline_packet" not in tools.response or "harness_completion_assessment" not in tools.response or "completion_audit_packet" not in tools.response or "evidence_ledger" not in tools.response or "completion_claim_gate" not in tools.response or "completion_next_proof_packet" not in tools.response or "operator_handoff_packet" not in tools.response or "harness_readiness_digest" not in tools.response or "execution_proof_bundle" not in tools.response or "execution_mission_control" not in tools.response or "execution_case_handoff_packet" not in tools.response or "save_execution_case" not in tools.response or "execution_case_evidence_packet" not in tools.response or "append_execution_case_evidence" not in tools.response or "inspect_execution_case" not in tools.response or "execution_case_gate" not in tools.response or "execution_case_review_packet" not in tools.response or "execution_case_closure_packet" not in tools.response or "execution_case_timeline" not in tools.response or "execution_runbook" not in tools.response or "harness_cycle_preview" not in tools.response or "harness_lifecycle_state" not in tools.response or "harness_control_surface" not in tools.response or "harness_operations_brief" not in tools.response or "agi_gate_report" not in tools.response or "agi_next_build_move" not in tools.response or "specialist_route_quality" not in tools.response or "specialist_execution_readiness" not in tools.response or "specialist_tool_dry_run_packet" not in tools.response or "specialist_execution_handoff_packet" not in tools.response or "specialist_post_run_closure_packet" not in tools.response or "specialist_cycle_ledger" not in tools.response:
            raise SystemExit("Core tool list should include harness tools.")
        for expected in [
            "completion_next_proof_packet [core, READ_ONLY]: Choose the next required command",
            "operator_handoff_packet [core, READ_ONLY]: Show the next operator-reviewable required command",
            "harness_readiness_digest [core, READ_ONLY]: Summarize Jarvis completion readiness, top blockers, next required command",
            "inspect_execution_case [core, READ_ONLY]: Inspect a saved Jarvis execution case file and its next required command plus proof aliases",
        ]:
            if expected not in tools.response:
                raise SystemExit(f"Core tool list missed command-first harness description: {expected}")
        if "next proof command" in tools.response:
            raise SystemExit("Core tool list should advertise next required commands, not next proof commands.")

        help_result = runtime.handle("help core")
        print(f"[ok={help_result.verified}] help core")
        print(help_result.response[:1200])
        print()
        if not help_result.verified or "priority goal" not in help_result.response or "harness status" not in help_result.response or "harness doctrine" not in help_result.response or "coding discipline" not in help_result.response or "harness completion" not in help_result.response or "completion audit" not in help_result.response or "evidence ledger" not in help_result.response or "completion claim gate" not in help_result.response or "completion next proof" not in help_result.response or "operator handoff" not in help_result.response or "harness readiness digest" not in help_result.response or "execution proof bundle" not in help_result.response or "execution mission control" not in help_result.response or "save execution case" not in help_result.response or "case evidence packet" not in help_result.response or "case evidence latest" not in help_result.response or "execution case latest" not in help_result.response or "execution case gate" not in help_result.response or "execution case review" not in help_result.response or "execution case closure" not in help_result.response or "execution case timeline" not in help_result.response or "execution runbook" not in help_result.response or "harness cycle" not in help_result.response or "harness lifecycle" not in help_result.response or "harness control" not in help_result.response or "harness operations" not in help_result.response or "agi gates" not in help_result.response or "agi next build move" not in help_result.response or "specialist route quality" not in help_result.response or "specialist execution readiness" not in help_result.response or "specialist tool dry run" not in help_result.response or "specialist execution handoff" not in help_result.response or "specialist post-run closure" not in help_result.response or "specialist cycle ledger" not in help_result.response:
            raise SystemExit("Core help should include priority goal, harness status, harness doctrine, harness completion, completion audit, evidence ledger, completion claim gate, execution proof bundle, execution mission control, execution runbook, AGI gates, AGI next build move, cycle preview, lifecycle state, control surface, and operations brief.")

    with TemporaryDirectory(prefix="jarvis-harness-recovery-closure-") as temp:
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
        cases = [
            ("harness cycle: remember that harness previews close recovery first", "harness_cycle_preview"),
            ("harness lifecycle: remember that harness previews close recovery first", "harness_lifecycle_state"),
            ("harness control: remember that harness previews close recovery first", "harness_control_surface"),
            ("harness operations: remember that harness previews close recovery first", "harness_operations_brief"),
        ]
        for command, tool_name in cases:
            result = runtime.handle(command)
            print(f"[ok={result.verified}] {command}")
            print(result.response[:1100])
            print()
            if not result.verified or result.tool_results[0].tool_name != tool_name:
                raise SystemExit(f"Recovery closure harness case did not route to {tool_name}: {result}")
            metadata = result.tool_results[0].metadata
            if metadata.get("safe_to_execute_now") is not False and metadata.get("can_auto_run") is not False:
                raise SystemExit(f"{tool_name} missed recovery safe-to-execute blocker: {metadata}")
            if metadata.get("recovery_closure_blocks_auto_execution") is not True:
                raise SystemExit(f"{tool_name} missed recovery closure metadata: {metadata}")
            if tool_name == "harness_cycle_preview":
                assert_harness_cycle_handoff(metadata, "Harness cycle recovery closure")
            if tool_name == "harness_lifecycle_state":
                assert_harness_lifecycle_handoff(metadata, "Harness lifecycle recovery closure")
            if tool_name == "harness_control_surface":
                assert_harness_control_handoff(metadata, "Harness control recovery closure")
            if tool_name == "harness_operations_brief":
                assert_harness_operations_handoff(metadata, "Harness operations recovery closure")
                if "can continue automatically: no, close recovery proof first" not in result.response:
                    raise SystemExit(f"{tool_name} should render recovery-aware auto-continue guidance: {result.response}")
            for expected in [f"verification receipt {run_id}", f"execution recovery packet {run_id}", f"execution learning closure {run_id}", f"after-action learning packet {run_id}"]:
                if expected not in metadata.get("recovery_closure_required_commands", []):
                    raise SystemExit(f"{tool_name} missed required closure command {expected!r}: {metadata}")
            closure_commands = metadata.get("recovery_closure_required_commands", [])
            learning_closure = f"execution learning closure {run_id}"
            after_action = f"after-action learning packet {run_id}"
            if closure_commands.index(after_action) > closure_commands.index(learning_closure):
                raise SystemExit(f"{tool_name} should require after-action learning before learning closure: {metadata}")
            if "Execution health recovery closure:" not in result.response:
                raise SystemExit(f"{tool_name} did not render recovery closure details.")


if __name__ == "__main__":
    main()
