from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from jarvis_v2.agent.failure_guidance import declare_failure_guidance
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.tools.audit import AFTER_ACTION_META_TOOLS
from jarvis_v2.tools.storage import (
    BOOTSTRAP_CHECK_COMMAND,
    BOOTSTRAP_WRITE_COMMAND,
    STORAGE_RECOVERY_CHECK_COMMAND,
    _configured_storage_diagnostics,
)


STORAGE_RECOVERY_CHECK_API = "/api/storage-recovery-check"
HARNESS_REQUEST_RECOVERY_ACTION = (
    "Provide the missing request, then retry through the normal policy."
)
HARNESS_CASE_ID_RECOVERY_ACTION = (
    "Use a positive execution case number or `latest`, then retry through the normal policy."
)
HARNESS_GATE_CONFIGURATION_RECOVERY_ACTION = (
    "Restore at least one AGI-direction gate, then retry the read-only build recommendation."
)
HARNESS_CASE_RECOVERY_ACTION = (
    "Create an execution case first, then retry the evidence append through the normal policy."
)
HARNESS_EVIDENCE_RECOVERY_ACTION = (
    "Correct the reported evidence input, then retry the append through the normal policy."
)


HARNESS_COMPONENTS = [
    {
        "name": "interface",
        "goal": "Natural command surface for text, speech, stop controls, and visible diagnostics.",
        "toolsets": {"conversation", "voice", "core"},
        "evidence": {"chat_loop_preview", "voice_command_lifecycle", "status_dashboard", "command_cockpit_packet"},
    },
    {
        "name": "brain routing",
        "goal": "Route chat, planning, memory, tool use, and future specialist models through inspectable logic.",
        "toolsets": {"core", "brain", "safety"},
        "evidence": {"architecture_map", "roadmap_report", "model_planner_prompt_preview", "specialist_router_contract", "specialist_orchestration_packet", "specialist_route_quality", "specialist_execution_readiness", "specialist_handoff_receipt", "specialist_handoff_quality_gate", "specialist_proposal_gate", "specialist_model_draft", "specialist_action_proposal_contract", "specialist_tool_dry_run_packet", "specialist_proposal_completion_gate", "specialist_execution_handoff_packet", "specialist_post_run_closure_packet", "specialist_cycle_ledger", "brain_think"},
    },
    {
        "name": "memory and state",
        "goal": "Durable context, Obsidian notes, preferences, people, goals, tasks, decisions, and resumable current state.",
        "toolsets": {"memory", "notes", "goals", "state", "continuity", "preferences", "people", "decisions"},
        "evidence": {"memory_tree_summary", "return_brief", "work_queue", "export_state_snapshot", "search_memory", "execution_case_handoff_packet", "save_execution_case", "execution_case_evidence_packet", "append_execution_case_evidence", "inspect_execution_case", "execution_case_gate", "execution_case_review_packet", "execution_case_closure_packet", "execution_case_timeline"},
    },
    {
        "name": "tool execution",
        "goal": "Explicit tools for local work, files, browser, Obsidian, scheduler, and future personal integrations, with acceptance evidence before enablement.",
        "toolsets": {"files", "browser", "personal", "scheduler", "computer", "code", "ingest", "utilities"},
        "evidence": {"tool_search", "risk_matrix", "execution_mission_control", "execution_case_handoff_packet", "save_execution_case", "execution_case_evidence_packet", "append_execution_case_evidence", "inspect_execution_case", "execution_case_gate", "execution_case_review_packet", "execution_case_closure_packet", "execution_case_timeline", "execution_runbook", "execution_proof_bundle", "integration_status", "integration_runbook", "integration_promotion_gate", "integration_implementation_spec", "integration_preflight_contract", "integration_enablement_gate", "integration_rehearsal_receipt", "integration_metadata_preview", "integration_proof_bundle", "integration_implementation_review", "integration_execution_matrix", "integration_adapter_manifest", "integration_adapter_probe", "integration_adapter_acceptance", "legacy_connector_migration_audit", "computer_task_plan"},
    },
    {
        "name": "safety and approvals",
        "goal": "Permission policy, pending approvals, readiness packets, last-look packets, audit records, and stop conditions.",
        "toolsets": {"safety", "approvals", "audit"},
        "evidence": {"safety_status", "approval_readiness_packet", "approval_execution_packet", "review_pending_approvals", "recent_tool_runs", "execution_audit_gate", "execution_recovery_packet", "execution_health_report", "recovery_closure_checklist", "after_action_learning_packet", "completion_next_proof_packet"},
    },
    {
        "name": "autonomy loop",
        "goal": "Plan, execute, verify, recover, and continue work without skipping user approvals for risky actions.",
        "toolsets": {"safety", "continuity", "focus", "proactive", "scheduler"},
        "evidence": {"autonomy_plan", "agent_loop_packet", "focus_brief", "next_action_packet", "mission_control", "execution_mission_control", "operator_timebox_contract", "autonomy_resume_gate", "autonomy_continuation_execution_packet", "autonomy_step_closure_packet", "autonomy_cycle_ledger"},
    },
    {
        "name": "learning loop",
        "goal": "Turn repeated feedback and workflows into reviewable memory, preferences, skills, tests, and notes.",
        "toolsets": {"learning", "feedback", "skills", "memory"},
        "evidence": {"learning_review", "session_learning_preview", "after_action_learning_packet", "feedback_actions", "failure_to_test_preview", "repeated_failure_clusters", "failure_promotion_packet", "failure_implementation_packet", "failure_apply_contract", "failure_learning_cockpit", "failure_patch_receipt_packet", "failure_patch_application_bridge", "failure_patch_completion_gate", "failure_patch_handoff_packet", "failure_patch_closeout_packet", "failure_learning_record_packet", "failure_learning_closure_ledger", "draft_skill_from_session"},
    },
    {
        "name": "observability and recovery",
        "goal": "Show status, readiness, diagnostics, recent runs, blockers, build delta, and resume plans.",
        "toolsets": {"audit", "continuity", "state", "safety", "system"},
        "evidence": {"readiness_report", "build_progress", "build_delta", "execution_mission_control", "execution_case_handoff_packet", "save_execution_case", "execution_case_evidence_packet", "append_execution_case_evidence", "inspect_execution_case", "execution_case_gate", "execution_case_review_packet", "execution_case_closure_packet", "execution_case_timeline", "execution_runbook", "execution_proof_bundle", "completion_next_proof_packet", "execution_health_report", "recovery_closure_checklist", "after_action_learning_packet", "work_block_checkpoint", "save_work_block_checkpoint", "checkpoint_recovery_preview", "checkpoint_recovery_execute", "checkpoint_recovery_receipt", "checkpoint_recovery_followthrough_packet", "autonomy_resume_gate", "autonomy_continuation_execution_packet", "autonomy_step_closure_packet", "autonomy_cycle_ledger", "operator_timebox_contract", "handoff_brief", "jarvis_doctor"},
    },
]


AGI_DIRECTION_GATES = [
    "multi-brain routing for chat, planning, code, vision, summarization, and reflection",
    "real speech input with transcript confirmation and microphone privacy boundaries",
    "vision-backed observe-act-verify with confidence and stop conditions",
    "approval-gated calendar, email, messages, contacts, and browser-account workflows",
    "long-running task execution with checkpoints, recovery, and user-visible audit",
    "stronger evaluation loops that turn failures into tests, preferences, or reviewed skills",
]


HARNESS_DOCTRINE = [
    {
        "name": "Runtime over brain alone",
        "principle": "The model is the engine; Jarvis must supply steering, pedals, brakes, dashboard, sensors, memory, tools, audit, recovery, and maintenance.",
        "jarvis_requirement": "Every stronger model route still needs explicit state, tool contracts, safety gates, observable progress, and verification receipts.",
    },
    {
        "name": "Command-first operation",
        "principle": "the operator should give natural orders by text or speech; Jarvis chooses the route automatically instead of making the operator pick modes.",
        "jarvis_requirement": "The GUI stays conversational while routing happens behind the scenes through previewable planner and harness packets.",
    },
    {
        "name": "Human-in-the-loop brakes",
        "principle": "Autonomy increases only where failure is bounded; risky work stops at approval readiness, a last-look approval packet, and approval chain proof.",
        "jarvis_requirement": "Shell/code, computer control, personal data, external side effects, destructive actions, and ambiguous risky work remain approval-gated.",
    },
    {
        "name": "Operator timeboxes outrank goals",
        "principle": "the operator's stop times, work windows, and pause instructions are hard operating limits, not suggestions.",
        "jarvis_requirement": "A priority goal can rank work, but it cannot justify continuing past an explicit stop time or ignoring a newer pause/stop instruction.",
    },
    {
        "name": "Lifecycle hooks",
        "principle": "A useful harness has repeatable before/during/after stages: perceive, ground, route, plan, gate, act, verify, learn.",
        "jarvis_requirement": "Every real action should leave enough evidence to resume, debug, recover, or promote a recurring failure into a test or skill.",
    },
    {
        "name": "Tool orchestration",
        "principle": "Tools are pedals and actuators; the harness decides when they can run, with what exact arguments, and how results are checked.",
        "jarvis_requirement": "ToolRegistry, PermissionPolicy, audit logs, readiness reports, approval readiness packets, last-look approval packets, and approval chain proof remain the execution spine.",
    },
    {
        "name": "Progressive disclosure",
        "principle": "The interface should show what matters now and keep deeper diagnostics available without crowding the main conversation.",
        "jarvis_requirement": "Keep the HUD compact, command-first, and scrollable for diagnostics, process maps, readiness, and recent run evidence.",
    },
    {
        "name": "Coding-agent discipline",
        "principle": "Jarvis should think before coding, prefer simple implementations, make surgical changes, and turn work into verified success criteria.",
        "jarvis_requirement": "Every build slice should surface assumptions, avoid speculative abstractions, touch only relevant lines, and prove success with focused tests before any completion claim.",
    },
]


AGI_HARNESS_LAYER_CONTRACT = [
    {
        "key": "intelligence",
        "title": "Intelligence",
        "contract": "Reason, draft, summarize, and propose routes from current context.",
        "evidence": {"chat_prompt_preview", "model_planner_prompt_preview", "brain_think", "agi_gate_report"},
        "boundary": "Model drafts do not authorize execution, approvals, personal-data reads, side effects, or completion claims.",
    },
    {
        "key": "engine",
        "title": "Engine",
        "contract": "Run the command-first runtime loop: perceive, ground, route, plan, gate, act, verify, recover.",
        "evidence": {
            "harness_cycle_preview",
            "harness_lifecycle_state",
            "execution_mission_control",
            "dispatch_decision_packet",
            "execution_proof_bundle",
        },
        "boundary": "The runtime, registry, permission policy, verifier, and audit trail own execution boundaries.",
    },
    {
        "key": "agents",
        "title": "Agents",
        "contract": "Use internal workers and specialist drafts as bounded capacity when parallel work is useful.",
        "evidence": {
            "subagent_fleet_status",
            "specialist_orchestration_packet",
            "specialist_action_proposal_contract",
            "specialist_tool_dry_run_packet",
            "specialist_cycle_ledger",
        },
        "boundary": "Internal workers are capacity, not companion personas, and cannot carry permission between reviews.",
    },
    {
        "key": "tools_memory",
        "title": "Tools+Memory",
        "contract": "Keep tools, memory, state, contacts, notes, and capability health typed, audited, and discoverable.",
        "evidence": {"capability_cockpit", "tool_detail", "memory_stats", "work_queue", "channel_health"},
        "boundary": "Tool calls and memory writes stay risk-classified, redacted, verified, and approval-gated when needed.",
    },
    {
        "key": "learning",
        "title": "Learning",
        "contract": "Turn repeated misses and useful patterns into reviewable tests, preferences, skills, and notes.",
        "evidence": {
            "learning_review",
            "after_action_learning_packet",
            "failure_learning_cockpit",
            "failure_learning_closure_ledger",
            "completion_claim_gate",
        },
        "boundary": "Learning artifacts are evidence, not self-approval or proof that Jarvis is finished.",
    },
]


def _harness_layer_contract_rows(tool_names: set[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    known_tools = tool_names or set()
    for layer in AGI_HARNESS_LAYER_CONTRACT:
        evidence_tools = sorted(str(name) for name in layer["evidence"])
        present_evidence = [name for name in evidence_tools if name in known_tools]
        missing_evidence = [name for name in evidence_tools if name not in known_tools]
        if not tool_names:
            status = "unmeasured"
        elif not missing_evidence:
            status = "ready"
        elif present_evidence:
            status = "partial"
        else:
            status = "missing"
        rows.append(
            {
                "key": layer["key"],
                "title": layer["title"],
                "contract": layer["contract"],
                "boundary": layer["boundary"],
                "status": status,
                "evidence_tools": evidence_tools,
                "evidence_tool_count": len(evidence_tools),
                "present_evidence": present_evidence,
                "present_evidence_count": len(present_evidence),
                "missing_evidence": missing_evidence,
                "missing_evidence_count": len(missing_evidence),
                "non_authorizing": True,
            }
        )
    return rows


def _harness_layer_contract_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "harness_layer_contract_rows": rows,
        "harness_layer_contract_row_count": len(rows),
        "harness_layer_contract_titles": [row["title"] for row in rows],
        "harness_layer_contract_ready_count": len([row for row in rows if row.get("status") == "ready"]),
        "harness_layer_contract_partial_count": len([row for row in rows if row.get("status") == "partial"]),
        "harness_layer_contract_missing_count": len([row for row in rows if row.get("status") == "missing"]),
        "harness_layer_contract_unmeasured_count": len([row for row in rows if row.get("status") == "unmeasured"]),
        "harness_layer_contract_non_authorizing": all(row.get("non_authorizing") is True for row in rows),
    }


LANE_PROOF_COMMANDS = {
    "steering": "dispatch decision: <next real order>",
    "pedals": "tool search: <capability>",
    "brakes": "approval readiness <id>",
    "dashboard": "harness completion",
    "memory/state": "work queue",
    "audit": "execution health report",
    "recovery": "checkpoint recovery: <objective>",
    "learning": "learning review",
    "command-first interface": "status dashboard",
    "routing and tool orchestration": "dispatch decision: <next real order>",
    "memory and state": "work queue",
    "approval gates and audit": "approval readiness <id>",
    "verification and recovery": "verification packet: <next real order>",
    "diagnostics and observability": "harness completion",
    "learning loop": "learning review",
}


RISK_KEYWORDS = {
    "computer control": ("computer", "screen", "click", "type", "mouse", "keyboard", "desktop", "window", "app"),
    "shell/code": ("run command", "terminal", "shell", "python", "script", "code", "execute", "install", "npm", "pip"),
    "files": ("write file", "edit file", "delete file", "move file", "rename", "folder", "directory"),
    "personal data": ("email", "calendar", "contact", "message", "gmail", "inbox", "private", "clipboard"),
    "external side effect": ("send", "post", "publish", "buy", "book", "schedule", "remind", "call", "text"),
    "destructive": ("delete", "remove", "erase", "wipe", "reset", "clear"),
}


def _split_csvish(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
    else:
        parts = [part.strip() for part in re.split(r"[,;\n]+", str(value or ""))]
    return [part for part in parts if part]


HARNESS_CYCLE_STAGES = [
    ("perceive", "Capture text/speech/vision input and preserve the original order."),
    ("ground", "Attach relevant memory, preferences, goals, current state, and recent approvals."),
    ("route", "Decide whether this is chat, read-only preview, local-safe work, or approval-gated execution."),
    ("plan", "Choose tools and produce exact arguments before anything touches the computer or outside world."),
    ("gate", "Auto-run only read-only/local-safe work; stop for approval on risky or personal actions."),
    ("act", "Run the approved/local-safe step through the tool registry and audit every result."),
    ("verify", "Check outcome evidence, surface failures, and choose recovery instead of silent retries."),
    ("learn", "Turn useful feedback, errors, and repeated workflow into memory, preferences, tests, or skills."),
]


AGI_GATE_EVIDENCE = [
    {
        "gate": "multi-brain routing",
        "evidence": {"chat_prompt_preview", "model_planner_prompt_preview", "specialist_router_contract", "specialist_orchestration_packet", "specialist_route_quality", "specialist_execution_readiness", "specialist_handoff_receipt", "specialist_handoff_quality_gate", "specialist_proposal_gate", "specialist_model_draft", "specialist_action_proposal_contract", "specialist_tool_dry_run_packet", "specialist_proposal_completion_gate", "specialist_execution_handoff_packet", "specialist_post_run_closure_packet", "specialist_cycle_ledger", "brain_think", "architecture_map"},
        "missing": "real specialist action execution is still disabled; bounded specialist drafts now require action-proposal contract proof with a measured action proposal scorecard, a specialist tool dry-run packet with verification scoring, proposal completion gate proof, execution handoff proof, specialist post-run closure packet proof with runtime_trace_sha256, verification_receipt_sha256, execution_audit_sha256, execution_recovery_sha256, after_action_learning_sha256, and completion_claim_sha256, plus a full specialist cycle ledger before any ToolRegistry/PermissionPolicy-reviewed action can be counted as closed or a fresh specialist review can begin",
        "next": "Use the specialist execution handoff packet after the specialist proposal completion gate, then the specialist post-run closure packet after runtime trace, verification receipt, audit, recovery, learning, completion-claim evidence, and post-run artifact hashes, then the full specialist cycle ledger so completed draft-only specialist output can enter only the normal runtime review path and cannot carry permission into the next review. The measured action proposal scorecard, ToolRegistry, exact arguments, PermissionPolicy, approval boundary, verification packet, execution receipt, audit, recovery, learning proof, completion-claim proof, post-run artifact hashes, and fresh-review boundary remain required.",
    },
    {
        "gate": "real speech input",
        "evidence": {"voice_setup_check", "voice_capture_privacy_packet", "voice_native_microphone_gate_packet", "voice_file_transcription_plan", "voice_audio_file_gate_packet", "voice_transcript_review", "voice_confirmation_packet", "voice_confirmation_receipt", "voice_confirmation_audit_ledger", "voice_route_gate_packet", "voice_route_proof_bundle", "voice_runtime_bridge_packet", "voice_command_cockpit", "voice_action_audit_packet", "voice_execution_handoff_packet", "voice_post_run_closure_packet", "voice_cycle_ledger", "voice_stop_intent_packet", "voice_command_lifecycle"},
        "missing": "native/offline microphone capture is still not enabled; browser push-to-talk and audio-file import remain transcript-first with visible privacy state, file-consent gates, supplied-receipt audit ledger, receipt-before-route gating, proof-bundle closure, command cockpit review, action-audit proof, execution-handoff packet, hash-bound post-run closure, proof-only voice post-run closure token boundary, full voice cycle ledger, stop-intent brake proof, and command-intake-only runtime bridge before routing",
        "next": "Use the voice stop intent packet for stop/cancel/rerecord speech. Use the voice confirmation audit ledger first so supplied privacy receipt proof and supplied confirmation receipt id/nonce proof bind before command-intake proof can count. Use the voice action audit packet, voice execution handoff packet, voice post-run closure packet, and then the voice cycle ledger after the voice command cockpit so confirmed speech has route gate, proof bundle, transcript hash match, command-intake-only bridge, dispatch decision, execution readiness matrix, approval boundary, verification packet, post-run proof queue, execution audit, execution health, after-action learning proof, valid post-run artifact sha256 fields, a proof-only voice post-run closure token boundary, and a full prior-command ledger before any next spoken order review.",
    },
    {
        "gate": "vision observe-act-verify",
        "evidence": {"computer_control_status", "computer_task_plan", "computer_action_packet", "approved_screen_observation_receipt", "screen_observation_freshness_packet", "screen_observation_confidence_packet", "screen_verification_contract", "observe_act_verify_proof_packet", "observe_act_verify_route_lock", "observe_act_verify_approval_bridge", "observe_act_verify_cockpit", "observe_act_verify_final_review", "observe_act_verify_action_audit", "observe_act_verify_execution_handoff", "observe_act_verify_post_run_closure", "observe_act_verify_cycle_ledger", "computer_control_readiness"},
        "missing": "actual approved screenshot observation is still required before real computer control; supplied observations now need approved screen-observation receipt metadata, explicit freshness proof for the one primitive, unified proof packet, route-lock review, approval-bridge receipt binding, cockpit API evidence, execution audit/health evidence, final human-review packet, action-audit proof, execution-handoff proof, post-run closure, and a full OAV cycle ledger before any next primitive review",
        "next": "Use the screen observation freshness packet after approved screen-observation receipt metadata, then observe-act-verify final review, observe-act-verify action audit, observe-act-verify execution handoff, observe-act-verify post-run closure, and the observe-act-verify cycle ledger so the exact primitive command stays behind a natural-language route lock and any approved primitive has fresh capture-age proof, audit evidence, execution-health evidence, operator final-review evidence, a post-run proof queue, matching verification receipt, execution audit, after-action learning proof, and a full prior-primitive ledger before Jarvis can even start a fresh OAV cockpit review for the next computer-control primitive.",
    },
    {
        "gate": "personal integrations",
        "evidence": {"integration_status", "integration_action_preview", "integration_scope_packet", "integration_dry_run_contract", "integration_runbook", "integration_promotion_gate", "integration_implementation_spec", "integration_preflight_contract", "integration_enablement_gate", "integration_rehearsal_receipt", "integration_metadata_preview", "integration_proof_bundle", "integration_implementation_review", "integration_boundary_contract", "integration_execution_matrix", "integration_adapter_manifest", "integration_adapter_probe", "integration_adapter_acceptance", "legacy_connector_migration_audit"},
        "missing": "real account connectors are still not enabled; legacy connector migration audit, execution matrix, adapter manifest, adapter probe, adapter acceptance, preflight, enablement, disabled-adapter rehearsal, metadata-only preview, proof-bundle receipts, and implementation-review status evidence must be satisfied before scoped data access, dry-run preview, or one-shot approval execution",
        "next": "Use legacy connector migration audit before connector implementation review. Use integration implementation review after the proof bundle: prove fake bounded metadata rows, metadata row schema/limit contract, disabled adapter boundary, exact metadata-only scope, adapter acceptance, enablement gate, rehearsal receipt, blocked private/full-content paths, side-effect blocks, tests, audit, status/API evidence, acceptance evidence, rollback, and verification before implementing one connector.",
    },
    {
        "gate": "long-running autonomy",
        "evidence": {"work_session_packet", "continuation_packet", "build_target_packet", "next_session_plan", "work_block_checkpoint", "save_work_block_checkpoint", "checkpoint_recovery_preview", "checkpoint_recovery_apply_packet", "checkpoint_recovery_execute", "checkpoint_recovery_receipt", "checkpoint_recovery_followthrough_packet", "checkpoint_recovery_cockpit", "autonomy_resume_gate", "autonomy_continuation_execution_packet", "autonomy_step_closure_packet", "autonomy_cycle_ledger", "operator_timebox_contract", "operator_instruction_supersession_packet"},
        "missing": "real risky-task recovery execution still requires explicit approval before shell/code or computer-control verification; risky checkpoint recovery steps now surface approval proof queues, non-authorizing boundary rows, and a hash-bound risky recovery approval boundary token before any risky recovery step can be recorded, while local-safe recovery execution emits the checkpoint recovery follow-through packet, autonomy resume gate, autonomy continuation execution packet, autonomy step closure packet, and autonomy cycle ledger so one normal local-safe step cannot roll into another without post-step verification, audit, health, checkpoint, learning evidence, recovery/post-step artifact hashes, readable post-step receipt/checkpoint file-hash binding, latest-checkpoint path binding, a hash-bound one-step execution contract token, a prior-cycle proof-only token boundary, a full prior-step ledger, a proof-only awake guard boundary, a proof-only operator supersession token boundary, an autonomy cycle ledger token boundary, and a risky next-step approval proof queue with boundary rows before any risky proposed continuation can be reviewed",
        "next": "Use the operator timebox contract, operator instruction supersession packet, checkpoint recovery cockpit, checkpoint recovery execute, recovery follow-through packet, autonomy resume gate, autonomy continuation execution packet, autonomy step closure packet, and autonomy cycle ledger before autonomous follow-through so STOP_WINDOW_ACTIVE, STOP_TIME_REACHED, HELD_FOR_PARSEABLE_TIMEBOX, checkpoint freshness, latest-checkpoint path binding, arbitrary checkpoint rejection, blockers, risky recovery approval proof queue, non-authorizing boundary rows, hash-bound risky recovery approval boundary token, proof queue, closure evidence, one-step local-safe permission, hash-bound one-step execution contract token, post-step proof queue, risky next-step approval proof queue, approval boundary rows, post-step closure evidence, recovery/post-step artifact hashes, readable post-step receipt/checkpoint file-hash binding, prior-cycle ledger tokens as proof-only/non-authorizing evidence, operator supersession token boundary rows, autonomy cycle ledger token boundary rows, awake guard tokens as proof-only/non-authorizing evidence, and a full prior-step ledger are visible; keep shell/code, OS wake locks, and computer-control recovery behind approval packets.",
    },
    {
        "gate": "evaluation and learning loop",
        "evidence": {"learning_review", "session_learning_preview", "after_action_learning_packet", "execution_learning_closure_packet", "feedback_actions", "failure_to_test_preview", "repeated_failure_clusters", "failure_promotion_packet", "failure_implementation_packet", "failure_apply_contract", "failure_learning_cockpit", "failure_patch_receipt_packet", "failure_patch_application_bridge", "failure_patch_completion_gate", "failure_patch_handoff_packet", "failure_patch_closeout_packet", "failure_learning_record_packet", "failure_learning_closure_ledger", "draft_skill_from_session"},
        "missing": "actual patch application still happens outside automatic Jarvis execution and must be bridged back to the reviewed apply contract before claiming a failure is fixed; learning closure now requires verification, recovery, after-action learning, repeated-failure promotion proof, an exact regression-test contract hash that binds observed failures, assertions, target file/test, focused command, and rollback, the failure learning cockpit, the failure patch receipt packet with patch_receipt_sha256 proof, the failure patch application bridge, the failure patch completion gate, the failure patch handoff packet, the failure patch closeout packet, the failure learning record packet with learning_record_sha256 proof, and proof-only failure learning closure token boundary rows before completion claims become durable learning records",
        "next": "Use the execution learning closure packet after after-action learning, then use the failure learning cockpit, failure apply contract, failure patch receipt packet, failure patch application bridge, failure patch completion gate, failure patch handoff packet, failure patch closeout packet, failure learning record packet, and failure learning closure ledger before claiming one reviewed regression was fixed and recorded with an exact regression-test contract hash, pre-patch failing-test proof, focused verification, compile receipts, rollback, patch_receipt_sha256, apply-contract evidence, applied-patch bridge evidence, completion-review handoff, evidence ledger, completion claim gate, post-claim review evidence, after-action learning, regression linkage, durable record evidence, learning_record_sha256, and non-authorizing closure token boundary rows.",
    },
]

AGI_GATE_BUILD_TARGETS = {
    "multi-brain routing": {
        "title": "Add measured specialist action proposal contracts without invoking uncontrolled model routes.",
        "files": [
            "jarvis_v2/tools/brain.py",
            "jarvis_v2/tools/model_status.py",
            "jarvis_v2/agent/planner.py",
            "jarvis_v2/scripts/smoke_test_harness.py",
            "jarvis_v2/scripts/smoke_test_model_status.py",
        ],
        "tests": [
            "python3 -m py_compile jarvis_v2/tools/brain.py jarvis_v2/tools/model_status.py jarvis_v2/agent/planner.py",
            "python3 -m jarvis_v2.scripts.smoke_test_harness",
            "python3 -m jarvis_v2.scripts.smoke_test_model_status",
        ],
        "acceptance": [
            "specialist route packets expose route, model target, fallback, proof target, and stop condition",
            "missing or unavailable specialist models degrade to preview/fallback without pretending execution happened",
            "handoff quality is measured or explicitly marked unmeasured before completion claims",
            "specialist proposal gate locks tool proposals until route quality, execution readiness, handoff receipt, verifier coverage, model readiness, approval boundary, ToolRegistry, and PermissionPolicy proof are present",
            "specialist action proposal contract keeps execution locked until a registered tool, exact arguments, risk preflight, approval boundary, and verification packet are proven",
            "specialist tool dry-run packet reviews ToolRegistry, exact arguments, PermissionPolicy risk, verification expectation, and approval boundary without executing the tool",
            "specialist execution handoff packet packages completed local-safe proposals for the normal runtime review path without executing, approving, queuing approvals, or bypassing post-run proof",
            "specialist execution handoff exposes a non-authorizing runtime-review contract so ready handoffs cannot authorize model calls, tool execution, approvals, personal-data reads, side effects, completion claims, or post-run proof bypass",
            "specialist post-run closure packet requires runtime trace, verification receipt, execution audit, recovery packet, after-action learning, and completion-claim evidence before a specialist handoff can count as closed",
            "specialist cycle ledger binds route, readiness, handoff, proposal, dry-run, execution handoff, post-run closure, and fresh-review boundary proof before starting the next specialist review",
            "specialist cycle ledger marks prior route, proposal, handoff, closure, and review-token proof as unable to authorize actions, model calls, tool execution, approvals, or fresh review",
        ],
    },
    "real speech input": {
        "title": "Make speech transcript confirmation auditable before any spoken order can act.",
        "files": [
            "jarvis_v2/tools/voice.py",
            "jarvis_v2/ui/status_server.py",
            "jarvis_v2/scripts/smoke_test_voice.py",
            "jarvis_v2/scripts/smoke_test_status_server.py",
        ],
        "tests": [
            "python3 -m py_compile jarvis_v2/tools/voice.py jarvis_v2/ui/status_server.py",
            "python3 -m jarvis_v2.scripts.smoke_test_voice",
            "python3 -m jarvis_v2.scripts.smoke_test_status_server",
        ],
        "acceptance": [
            "speech controls keep transcript confirmation visible before routing",
            "voice confirmation receipts and route gates expose transcript hash, planned action, confirmation state, and approval boundary before routing",
            "voice confirmation audit ledger requires supplied privacy receipt id, supplied confirmation receipt id, and supplied receipt nonce before command-intake proof can count",
            "voice route proof bundles join privacy receipt, confirmation receipt, route gate, transcript hash match, approval boundary, and required proof commands before routing",
            "voice runtime bridge packets allow confirmed transcripts to enter command intake only and block direct executable tool actions",
            "voice command cockpit consolidates receipt, route gate, proof bundle, and runtime bridge before command intake",
            "voice action audit packets prove confirmed speech cannot become direct execution before command intake, dispatch, readiness, verification, and approvals",
            "voice execution handoff packets package the exact confirmed transcript, pre-run command-intake proof chain, approval boundary, and post-run verification/audit/learning proof queue without executing speech",
            "voice post-run closure packets require verification receipt, execution health, execution audit, after-action learning evidence, and valid artifact sha256 fields before the next voice review",
            "voice post-run closure token boundary rows prove the completed spoken command is proof-only and cannot authorize routing, actions, model calls, tool execution, approvals, personal-data reads, side effects, receipt reuse, or the next voice review",
            "voice cycle ledgers bind privacy receipt, confirmation receipt, route gate, proof bundle, runtime bridge, cockpit, action audit, execution handoff, and hash-bound post-run closure into one prior-command proof chain",
            "voice cycle ledger preflight and fresh-review rows mark prior speech proof as unable to authorize actions, model calls, tool execution, approvals, routing, or transcript mutation",
            "voice stop intent packets prove stop/cancel/rerecord speech only stops capture or preserves a draft and never routes, deletes, rewrites, approves, or executes",
            "audio-file import has a metadata-only consent gate before any file read or transcription",
            "microphone/privacy state is explicit and read-only previews do not record audio",
            "stop speech remains separate from deleting or rewriting the message",
        ],
    },
    "vision observe-act-verify": {
        "title": "Strengthen screenshot confidence and observe-act-verify proof before computer control.",
        "files": [
            "jarvis_v2/tools/computer.py",
            "jarvis_v2/tools/safety.py",
            "jarvis_v2/ui/status_server.py",
            "jarvis_v2/scripts/smoke_test_computer_plan.py",
            "jarvis_v2/scripts/smoke_test_status_server.py",
        ],
        "tests": [
            "python3 -m py_compile jarvis_v2/tools/computer.py jarvis_v2/tools/safety.py jarvis_v2/ui/status_server.py",
            "python3 -m jarvis_v2.scripts.smoke_test_computer_plan",
            "python3 -m jarvis_v2.scripts.smoke_test_status_server",
        ],
        "acceptance": [
            "observation packets state expected screen evidence and confidence before action",
            "approved screen-observation receipt distinguishes source-only prototype evidence from durable screenshot adapter readiness",
            "observe-act-verify proof packet joins primitive action, fresh observation confidence, and verification evidence before approval review",
            "observe-act-verify route lock keeps natural-language desktop-control routing disabled until approval-chain and approved-rerun proof are present",
            "observe-act-verify approval bridge binds the route lock to an exact approval id, approved run id, and matching verification receipt before final route review",
            "observe-act-verify cockpit tool and API consolidate proof, route-lock, approval bridge, receipt binding, and remaining blockers without observing or controlling the screen",
            "observe-act-verify final review tool and API require cockpit, audit, execution-health, and operator review evidence before the operator decides on one primitive",
            "observe-act-verify action audit proves the exact primitive command, disabled computer control state, natural-language route lock, verification handoff, and approval decision boundary before any primitive can run",
            "observe-act-verify execution handoff packages the exact one-primitive approval handoff and post-run verification, health, audit, and after-action learning proof queue without executing computer control",
            "observe-act-verify post-run closure requires matching approved run, verification receipt, execution health, audit, and after-action learning before the next primitive review can begin",
            "observe-act-verify cycle ledger binds observation confidence, screen verification, proof packet, route lock, approval bridge, cockpit, final review, action audit, execution handoff, and post-run closure into one read-only prior-primitive proof chain",
            "observe-act-verify cycle ledger exposes fresh-review preflight and contract rows that keep prior primitive proof from authorizing actions, screenshots, computer control, approvals, route unlocks, or verification shortcuts",
            "computer control stays disabled until an exact approved primitive action is reviewed",
            "screen verification distinguishes observed evidence from assumed success",
        ],
    },
    "personal integrations": {
        "title": "Move one connector from disabled adapter proof toward scoped implementation without enabling accounts.",
        "files": [
            "jarvis_v2/tools/personal.py",
            "jarvis_v2/tools/registry.py",
            "jarvis_v2/agent/planner.py",
            "jarvis_v2/scripts/smoke_test_personal.py",
            "jarvis_v2/scripts/smoke_test_harness.py",
        ],
        "tests": [
            "python3 -m py_compile jarvis_v2/tools/personal.py jarvis_v2/tools/registry.py jarvis_v2/agent/planner.py",
            "python3 -m jarvis_v2.scripts.smoke_test_personal",
            "python3 -m jarvis_v2.scripts.smoke_test_harness",
        ],
        "acceptance": [
            "dry-run contract proves metadata row contract before metadata preview, promotion gate, or proof bundle",
            "proof bundle passes metadata preview, disabled adapter acceptance, metadata row contract, enablement gate, and rehearsal receipt before implementation review",
            "promotion gate carries metadata row contract proof, row limit, blocked payload fields, tests, audit, rollback, and approval boundaries before implementation spec",
            "implementation review requires proof bundle, metadata row contract, preflight row-contract proof, implementation spec, status/API smoke evidence, audit, verification, and rollback before code review",
            "adapter acceptance proves fake metadata rows, allowed row fields, row limit, and blocked full-content/side-effect paths",
            "enablement gate requires acceptance, audit, tests, rollback, and metadata-only scope",
            "route lock exposes a non-authorizing explicit route review contract so a ready connector route still cannot authorize account access, route unlock, natural-language routing, personal-data reads, side effects, or scope reuse",
            "natural-language connector execution remains disabled until explicit approval lanes exist",
        ],
    },
    "long-running autonomy": {
        "title": "Strengthen checkpoint recovery execution for reviewed local-safe continuation steps.",
        "files": [
            "jarvis_v2/tools/continuity.py",
            "jarvis_v2/tools/next_step.py",
            "jarvis_v2/ui/status_server.py",
            "jarvis_v2/scripts/smoke_test_next_step.py",
            "jarvis_v2/scripts/smoke_test_build_progress.py",
            "jarvis_v2/scripts/smoke_test_status_server.py",
        ],
        "tests": [
            "python3 -m py_compile jarvis_v2/tools/continuity.py jarvis_v2/tools/next_step.py jarvis_v2/ui/status_server.py",
            "python3 -m jarvis_v2.scripts.smoke_test_next_step",
            "python3 -m jarvis_v2.scripts.smoke_test_build_progress",
            "python3 -m jarvis_v2.scripts.smoke_test_status_server",
        ],
        "acceptance": [
            "operator timebox contract returns STOP_WINDOW_ACTIVE, STOP_TIME_REACHED, or HELD_FOR_PARSEABLE_TIMEBOX before autonomous continuation",
            "operator instruction supersession exposes a non-authorizing contract and proof-only supersession token boundary so newest stop/continue instructions override older goals and automations without authorizing execution, local-safe steps, risky work, approvals, recovery follow-through, goal override, or timebox override",
            "continuation packets include recovery queue, verification target, and stop condition",
            "local-safe recovery execution can be previewed and verified without approving risky commands",
            "checkpoint recovery execute emits a Recovery closure gate and checkpoint recovery follow-through packet with receipt, fresh checkpoint, verification evidence, stop condition, and approval boundary before normal follow-through resumes",
            "checkpoint recovery execute surfaces a risky recovery approval proof queue, non-authorizing boundary rows, and a hash-bound risky recovery approval boundary token before recording any risky recovery step",
            "autonomy resume gate binds operator timebox, checkpoint recovery cockpit, latest checkpoint path, and follow-through closure before normal local-safe continuation resumes",
            "autonomy continuation execution packet gates exactly one normal local-safe follow-through step with post-step proof queue, risky next-step approval proof queue, and risky-step approval boundary rows",
            "autonomy continuation execution packet emits a hash-bound one-step execution contract token that step closure, cycle ledger, and status APIs carry as non-reusable prior proof",
            "autonomy continuation execution packet treats prior cycle ledger tokens as proof-only/non-authorizing evidence and requires a fresh cycle ledger token after the next step",
            "autonomy step closure packet requires a readable post-step verification receipt, receipt hash matching that file, fresh checkpoint, checkpoint hash matching that file, execution health, execution audit, and after-action learning before another continuation review",
            "autonomy cycle ledger fresh-review contract keeps prior artifacts from authorizing local-safe steps, risky work, unreviewed follow-through, timebox reuse, checkpoint reuse, or token reuse",
            "autonomy cycle ledger token boundary rows prove the completed ledger token is proof-only and cannot authorize a new cycle, local-safe step, risky work, model/tool calls, personal-data reads, external side effects, or token reuse",
            "operator timebox receipt contract travels through resume, one-step continuation, step closure, and cycle ledger without authorizing execution, local-safe steps, risky work, approvals, timebox reuse, or reuse for the next step",
            "operator instruction supersession token boundary travels through resume, one-step continuation, step closure, cycle ledger, and status APIs without authorizing execution, local-safe steps, risky work, approvals, recovery follow-through, timebox override, goal override, model/tool calls, personal-data reads, external side effects, or reuse",
            "awake guard boundary proof travels through resume, one-step continuation, step closure, cycle ledger, and status APIs without authorizing OS wake locks, shell execution, computer control, approvals, or reuse for the next timebox",
            "status API mirrors prove local-safe and risky carried next-step approval boundaries through autonomy step closure and cycle ledger without queuing approvals",
            "failed or stale checkpoints route to recovery review before normal task follow-through",
        ],
    },
    "evaluation and learning loop": {
        "title": "Promote repeated failures into exact tests through a reviewed apply contract.",
        "files": [
            "jarvis_v2/tools/feedback.py",
            "jarvis_v2/tools/audit.py",
            "jarvis_v2/tools/registry.py",
            "jarvis_v2/agent/planner.py",
            "jarvis_v2/tools/help.py",
            "jarvis_v2/tools/capabilities.py",
            "jarvis_v2/tools/harness.py",
            "jarvis_v2/scripts/smoke_test_feedback.py",
            "jarvis_v2/scripts/smoke_test_learning_review.py",
            "jarvis_v2/scripts/smoke_test_audit.py",
        ],
        "tests": [
            "python3 -m py_compile jarvis_v2/tools/feedback.py jarvis_v2/tools/audit.py jarvis_v2/tools/registry.py jarvis_v2/agent/planner.py jarvis_v2/tools/help.py jarvis_v2/tools/capabilities.py jarvis_v2/tools/harness.py",
            "python3 -m jarvis_v2.scripts.smoke_test_feedback",
            "python3 -m jarvis_v2.scripts.smoke_test_learning_review",
            "python3 -m jarvis_v2.scripts.smoke_test_audit",
        ],
        "acceptance": [
            "execution learning closure packet blocks completion until verification, recovery, after-action learning, and repeated-failure promotion proof are present",
            "failure clusters include observed failure, expected behavior, target file/test, and rollback",
            "exact regression-test contract hash binds observed failures, assertions, expected behavior, target file/test, focused command, and rollback before patch receipt, application bridge, completion, closeout, and learning closure can be ready",
            "apply contracts stay read-only until a scoped patch is chosen",
            "failure patch receipt packet checks changed files, target smoke test evidence, focused verification, compile pass, rollback note, and patch_receipt_sha256 before completion review",
            "failure patch application bridge binds the applied patch to the reviewed apply contract before completion gates can count it",
            "failure patch review contract travels from application bridge through completion, handoff, closeout, durable learning record, and closure ledger without authorizing patch application, file writes, tool execution, approvals, completion claims, learning records, or reuse for the next patch",
            "failure patch completion gate blocks completion claims until cockpit, apply-contract review, application bridge, patch receipt, focused verification, compile pass, rollback, evidence ledger, and claim gate handoff are present",
            "failure patch handoff packet packages the ready completion gate, pre-claim proof chain, post-claim review queue, and rollback evidence without writing code or claiming the regression fixed",
            "failure patch closeout packet requires completion audit, evidence ledger, completion claim gate, and post-claim review evidence before learning can be recorded",
            "failure learning record packet requires closeout readiness, after-action learning, a record target, regression-test linkage, durable record evidence, and learning_record_sha256 before learning is treated as recorded",
            "failure learning closure ledger emits a proof-only closure token that cannot authorize patch application, file writes, tool execution, approvals, completion claims, learning records, or reuse for the next patch",
            "failure learning closure ledger exposes non-authorizing closure token boundary rows so prior learning closure proof cannot authorize patch application, file writes, tool execution, approvals, completion claims, learning records, or reuse for the next patch, learning record, or completion claim",
            "after-action learning cannot count as completion without verification evidence",
        ],
    },
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_agi_target_text(value: Any, limit: int = 500) -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        text = ""
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _safe_agi_target_list(values: list[Any], limit: int = 500) -> list[str]:
    safe_values: list[str] = []
    for value in values:
        text = _safe_agi_target_text(value, limit=limit)
        if text:
            safe_values.append(text)
    return safe_values


def _agi_target_value(target: Any, key: str, default: Any = None) -> Any:
    if not isinstance(target, dict):
        return default
    try:
        return target.get(key, default)
    except Exception:
        return default


def _agi_target_list_value(target: Any, key: str) -> list[Any]:
    value = _agi_target_value(target, key, [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return list(value)
    try:
        return list(value)
    except Exception:
        return [value]


def _target_file_integrity(paths: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw_path in paths:
        path = str(raw_path or "").strip()
        if not path:
            continue
        safe_path = _safe_agi_target_text(path, limit=260)
        path_obj = Path(path)
        if path_obj.is_absolute() or path.startswith("~"):
            exists = False
        else:
            exists = (PROJECT_ROOT / path).exists()
        rows.append({"path": safe_path, "exists": exists})
        if not exists:
            missing.append(safe_path)
    return {
        "rows": rows,
        "checked": len(rows),
        "missing": missing,
        "missing_count": len(missing),
        "all_exist": not missing,
        "status": "TARGETS_EXIST" if not missing else "TARGETS_STALE",
    }


def _agi_gate_summary(tool_names: set[str]) -> dict[str, Any]:
    gate_rows: list[dict[str, Any]] = []
    strong = 0
    partial = 0
    missing = 0
    gate_statuses: dict[str, str] = {}
    present_evidence_by_gate: dict[str, list[str]] = {}
    missing_evidence_by_gate: dict[str, list[str]] = {}
    next_moves_by_gate: dict[str, str] = {}
    evidence_closure_commands_by_gate: dict[str, list[str]] = {}
    focused_verification_by_gate: dict[str, list[str]] = {}
    real_execution_gaps_by_gate: dict[str, str] = {}
    for gate in AGI_GATE_EVIDENCE:
        present = sorted(gate["evidence"] & tool_names)
        missing_evidence = sorted(gate["evidence"] - tool_names)
        if len(present) >= 3:
            status = "strong prototype"
            strong += 1
        elif present:
            status = "partial prototype"
            partial += 1
        else:
            status = "missing"
            missing += 1
        gate_name = str(gate["gate"])
        gate_statuses[gate_name] = status
        present_evidence_by_gate[gate_name] = present
        missing_evidence_by_gate[gate_name] = missing_evidence
        next_moves_by_gate[gate_name] = str(gate["next"])
        target = AGI_GATE_BUILD_TARGETS.get(gate_name, {})
        closure_commands = [
            f"agi next build move: {gate_name}",
            f"completion audit: improve AGI gate {gate_name}",
            "evidence ledger",
            f"completion claim gate: improve AGI gate {gate_name}",
        ]
        verification_commands = _safe_agi_target_list(_agi_target_list_value(target, "tests"), limit=500)
        evidence_closure_commands_by_gate[gate_name] = closure_commands
        focused_verification_by_gate[gate_name] = verification_commands
        real_execution_gaps_by_gate[gate_name] = str(gate["missing"])
        gate_rows.append(
            {
                "gate": gate_name,
                "status": status,
                "present": present,
                "missing_evidence": missing_evidence,
                "real_execution_gap": str(gate["missing"]),
                "next": str(gate["next"]),
                "evidence_closure_commands": closure_commands,
                "focused_verification": verification_commands,
                "non_authorizing": True,
            }
        )
    return {
        "rows": gate_rows,
        "gates": len(AGI_GATE_EVIDENCE),
        "strong": strong,
        "partial": partial,
        "missing": missing,
        "gate_statuses": gate_statuses,
        "present_evidence_by_gate": present_evidence_by_gate,
        "missing_evidence_by_gate": missing_evidence_by_gate,
        "next_moves_by_gate": next_moves_by_gate,
        "evidence_closure_commands_by_gate": evidence_closure_commands_by_gate,
        "focused_verification_by_gate": focused_verification_by_gate,
        "real_execution_gaps_by_gate": real_execution_gaps_by_gate,
        "real_execution_gap_count": len(AGI_GATE_EVIDENCE),
    }


def _select_agi_next_gate(gate_summary: dict[str, Any], requested_gate: str = "") -> dict[str, Any]:
    return _select_agi_next_gate_with_context(gate_summary, requested_gate)["gate"]


def _select_agi_next_gate_with_context(gate_summary: dict[str, Any], requested_gate: str = "") -> dict[str, Any]:
    requested = str(requested_gate or "").strip().lower()
    rows = list(gate_summary.get("rows") or [])
    if requested:
        for gate in rows:
            gate_name = str(gate.get("gate") or "")
            if requested in gate_name.lower() or gate_name.lower() in requested:
                return {
                    "gate": gate,
                    "selection_source": "operator_requested_gate",
                    "selection_reason": f"matched requested AGI gate focus `{requested}`",
                    "deliberate_focus_override": True,
                }
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row.get("missing_evidence") else 1,
            0 if row.get("status") != "strong prototype" else 1,
            len(str(row.get("present") or "")),
        ),
    )
    gate = ranked[0] if ranked else {}
    return {
        "gate": gate,
        "selection_source": "default_ranked_gate_after_unmatched_request" if requested else "default_ranked_gate",
        "selection_reason": (
            f"requested AGI gate focus `{requested}` did not match a configured gate; using the highest-priority remaining real-execution gap"
            if requested
            else "selected the highest-priority remaining AGI real-execution gap"
        ),
        "deliberate_focus_override": False,
    }


def _selected_agi_target_readiness(gate_summary: dict[str, Any], requested_gate: str = "") -> dict[str, Any]:
    selection = _select_agi_next_gate_with_context(gate_summary, requested_gate)
    selected_gate = selection["gate"]
    gate_name = str(selected_gate.get("gate") or "")
    target = AGI_GATE_BUILD_TARGETS.get(gate_name, {}) if gate_name else {}
    raw_likely_files = _agi_target_list_value(target, "files")
    likely_files = _safe_agi_target_list(raw_likely_files, limit=260)
    acceptance_checks = _safe_agi_target_list(_agi_target_list_value(target, "acceptance"), limit=500)
    target_title = _safe_agi_target_text(_agi_target_value(target, "title", "") or selected_gate.get("next") or "", limit=500)
    acceptance_gap_preview = acceptance_checks[:3]
    first_acceptance_gap = acceptance_gap_preview[0] if acceptance_gap_preview else ""
    closure_commands = list(selected_gate.get("evidence_closure_commands") or [])
    verification_commands = list(selected_gate.get("focused_verification") or [])
    file_integrity = _target_file_integrity(raw_likely_files)
    build_ready = bool(file_integrity["all_exist"] and verification_commands and acceptance_checks)
    return {
        "gate": selected_gate,
        "gate_name": gate_name,
        "target": {
            "title": target_title,
            "files": likely_files,
            "tests": verification_commands,
            "acceptance": acceptance_checks,
        },
        "target_title": target_title,
        "closure_commands": closure_commands,
        "verification_commands": verification_commands,
        "likely_files": likely_files,
        "file_integrity": file_integrity,
        "acceptance_checks": acceptance_checks,
        "acceptance_preview": acceptance_gap_preview,
        "first_acceptance_check": first_acceptance_gap,
        "acceptance_gap_preview": acceptance_gap_preview,
        "first_acceptance_gap": first_acceptance_gap,
        "build_ready": build_ready,
        "selection_source": selection["selection_source"],
        "selection_reason": selection["selection_reason"],
        "canonical_selector_command": f"agi next build move: {gate_name}" if gate_name else "agi next build move",
        "deliberate_focus_override": selection["deliberate_focus_override"],
    }


def _agi_focus_selection_metadata(selected_agi_readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "agi_focus_selection_source": selected_agi_readiness.get("selection_source", ""),
        "agi_focus_selection_reason": selected_agi_readiness.get("selection_reason", ""),
        "agi_focus_canonical_selector_command": selected_agi_readiness.get("canonical_selector_command", ""),
        "agi_focus_deliberate_focus_override": bool(selected_agi_readiness.get("deliberate_focus_override")),
    }


def _agi_acceptance_gap_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    acceptance_checks = list(metadata.get("agi_next_acceptance_checks") or metadata.get("acceptance_checks") or [])
    acceptance_gap_preview = list(
        metadata.get("agi_next_acceptance_gap_preview")
        or metadata.get("acceptance_gap_preview")
        or acceptance_checks[:3]
    )
    first_acceptance_gap = str(
        metadata.get("agi_next_first_acceptance_gap")
        or metadata.get("first_acceptance_gap")
        or (acceptance_gap_preview[0] if acceptance_gap_preview else "")
    )
    return {
        "agi_next_acceptance_preview": acceptance_gap_preview,
        "agi_next_acceptance_preview_count": len(acceptance_gap_preview),
        "agi_next_first_acceptance_check": first_acceptance_gap,
        "agi_next_acceptance_gap_preview": acceptance_gap_preview,
        "agi_next_acceptance_gap_preview_count": len(acceptance_gap_preview),
        "agi_next_first_acceptance_gap": first_acceptance_gap,
    }


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "approves_request": False,
        "dismisses_request": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "edits_files": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "speaks": False,
        "completes_tasks": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _known_no_change_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
    action: str,
) -> dict[str, Any]:
    """Declare a rejected harness request that performed no action or write."""

    truth = dict(metadata)
    truth.update(
        {
            "state_changed": False,
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        truth,
        output=output,
        action=action,
    )


def _execution_case_evidence_preview_defaults() -> dict[str, Any]:
    return {
        "evidence_preview_gate_count": 0,
        "evidence_preview_ready_count": 0,
        "evidence_preview_blocked_count": 0,
        "evidence_preview_verdicts": [],
        "evidence_preview_latest_verdict": "",
        "evidence_preview_latest_event_id": None,
        "evidence_preview_latest_handoff": {},
        "evidence_preview_latest_handoff_present": False,
    }


def _execution_case_evidence_packet_defaults(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "found": False,
        "case_id": None,
        "request": "",
        "verdict": "CASE_NOT_FOUND",
        "ready_to_append": False,
        "event_type": "evidence",
        "summary": "",
        "explicit_receipt_kind": "",
        "explicit_receipt_id": "",
        "inferred_receipt_kind": "",
        "inferred_receipt_id": "",
        "receipt_kind": "",
        "receipt_kind_normalized": "",
        "receipt_id": "",
        "receipt_target_status": "not_required",
        "receipt_target_exists": False,
        "receipt_target_issue": "",
        "target_lookup_command": "",
        "conflict_reason": "",
        "append_command": "",
        "next_command": "save execution case: <order>",
        "reason": "missing_case",
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "writes_case": False,
        "writes_case_event": False,
        "writes_evidence": False,
    }
    defaults.update(overrides)
    return defaults


def _execution_case_no_case_gate_defaults() -> dict[str, Any]:
    return {
        "found": False,
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
        **_execution_case_evidence_preview_defaults(),
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
        "next_case_required_command": "save execution case: <order>",
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
    }


RECOVERY_CLOSURE_META_TOOLS = {
    *AFTER_ACTION_META_TOOLS,
    "action_readiness_packet",
    "action_rehearsal",
    "agent_loop_packet",
    "agent_loop_preview",
    "argument_contract_packet",
    "assistant_turn_rehearsal",
    "autonomy_plan",
    "command_diagnosis",
    "command_cockpit_packet",
    "command_intake_packet",
    "dispatch_decision_packet",
    "execution_acceptance_gate",
    "risk_preflight",
    "risky_request_lifecycle",
    "verification_packet",
    "execution_contract",
    "task_completion_packet",
    "execution_governor_packet",
    "execution_readiness_matrix",
    "planner_gap_packet",
}

DIAGNOSTIC_READ_ONLY_TOOLS = {
    "build_delta_report",
    "build_progress_report",
    "completion_audit",
    "completion_claim_gate",
    "completion_next_proof_packet",
    "evidence_ledger",
    "harness_completion_assessment",
    "harness_readiness_digest",
    "jarvis_doctor",
    "readiness_report",
    "storage_recovery_check",
    "storage_status",
}


_MISSING_ROW_VALUE = object()


def _row_value(row: Any, key: str, default: Any = _MISSING_ROW_VALUE) -> Any:
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        keys = row.keys()
        if key not in keys:
            return default
        return row[key]
    except Exception:
        return default


def _row_text(row: Any, key: str, default: str = "") -> str:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE or value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _row_bool(row: Any, key: str, default: bool = False) -> bool:
    value = _row_value(row, key)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return default


def _row_int(row: Any, key: str, default: int | None = None) -> int | None:
    value = _row_value(row, key)
    if isinstance(value, bool) or value is _MISSING_ROW_VALUE or value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _row_positive_int_text(row: Any, key: str = "id") -> str:
    value = _row_int(row, key)
    return str(value) if value is not None and value > 0 else ""


def _first_positive_int_text(rows: list[Any], key: str = "id") -> str:
    for row in rows:
        value = _row_positive_int_text(row, key)
        if value:
            return value
    return ""


def _readable_tool_run_rows(rows: list[Any]) -> tuple[list[Any], int]:
    readable: list[Any] = []
    unreadable = 0
    for row in rows:
        if _row_text(row, "tool_name"):
            readable.append(row)
        else:
            unreadable += 1
    return readable, unreadable


def _is_execution_action_row(row: Any) -> bool:
    tool_name = _row_text(row, "tool_name")
    if not tool_name:
        return False
    if tool_name in RECOVERY_CLOSURE_META_TOOLS:
        return False
    if tool_name in DIAGNOSTIC_READ_ONLY_TOOLS and _row_bool(row, "ok") and _row_text(row, "risk") == "READ_ONLY":
        return False
    return True


def _row_metadata(row: Any) -> dict[str, Any]:
    try:
        metadata = _row_value(row, "metadata")
        if metadata is _MISSING_ROW_VALUE:
            return {}
        if isinstance(metadata, dict):
            return dict(metadata)
        parsed = json.loads(metadata or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


APPROVAL_HOLD_METADATA_KEYS = (
    "requires_confirmation",
    "requires_approval",
    "approval_required",
)

APPROVAL_HOLD_VALUES = {
    "approval_required",
    "approval_gate",
    "approval_gated",
    "approval_held",
    "approval_hold",
    "explicit_approval_required",
    "confirmation_required",
    "requires_confirmation",
    "requires_approval",
}


def _metadata_truthy_loose(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return False


def _normalized_metadata_token(value: Any) -> str:
    try:
        text = "" if value is None else str(value)
    except Exception:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().casefold()).strip("_")


def _is_approval_hold_metadata(metadata: dict[str, Any]) -> bool:
    if any(_metadata_truthy_loose(metadata.get(key)) for key in APPROVAL_HOLD_METADATA_KEYS):
        return True
    for key in (
        "failure_kind",
        "failure_stage",
        "stage",
        "guard_reason",
        "reason",
        "status",
        "send_status",
        "call_status",
    ):
        token = _normalized_metadata_token(metadata.get(key))
        if token in APPROVAL_HOLD_VALUES:
            return True
        if "approval" in token and any(
            marker in token for marker in ("required", "requires", "gate", "gated", "hold", "held")
        ):
            return True
    return False


def _is_approval_held_tool_run(row: Any) -> bool:
    if _row_bool(row, "ok", default=True):
        return False
    return _is_approval_hold_metadata(_row_metadata(row))


def _recent_tool_run_attention_buckets(rows: list[Any]) -> tuple[list[Any], list[Any]]:
    failed: list[Any] = []
    approval_held: list[Any] = []
    for row in rows:
        if _row_bool(row, "ok", default=True):
            continue
        if _is_approval_held_tool_run(row):
            approval_held.append(row)
        else:
            failed.append(row)
    return failed, approval_held


def _approval_review_command(rows: list[Any]) -> str:
    for row in rows:
        approval_id = _row_int(row, "approval_id")
        metadata = _row_metadata(row)
        for candidate in (
            approval_id,
            _metadata_int(metadata.get("approval_id"), 0),
            _metadata_int(metadata.get("approved_approval_id"), 0),
            *_ids_from_text(
                _row_text(row, "output"),
                ("approval #", "approval id", "approval packet", "queued as approval"),
            ),
        ):
            if candidate:
                return f"approval readiness {candidate}"
    return "approval history"


def _ids_from_text(value: Any, markers: tuple[str, ...]) -> list[int]:
    text = str(value or "")
    ids: list[int] = []
    seen: set[int] = set()
    for marker in markers:
        marker_pattern = re.escape(marker).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"\b{marker_pattern}\s*#?\s*(\d+)\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            candidate = int(match.group(1))
            if candidate not in seen:
                ids.append(candidate)
                seen.add(candidate)
    return ids


def _append_unique(items: list[str], candidates: list[str]) -> None:
    for candidate in candidates:
        if candidate and candidate not in items:
            items.append(candidate)


def _completion_proof_queue_from_sources(
    *,
    recovery_proof_queue: list[str],
    execution_health_verification_proof_queue: list[str] | None = None,
    learning_closure_command: str,
    learning_proof_queue: list[str],
    learning_actionable_proof_queue: list[str],
    pending_approval_ids: list[int],
    open_task_ids: list[int],
    baseline_commands: list[str],
    selected_agi_closure_commands: list[str],
) -> list[str]:
    proof_queue: list[str] = []
    if pending_approval_ids:
        approval_id = pending_approval_ids[0]
        _append_unique(
            proof_queue,
            [
                f"approval readiness {approval_id}",
                f"approval packet {approval_id}",
                f"approval chain proof {approval_id}",
                f"verification receipt <approved run id from approval chain proof {approval_id}>",
            ],
        )
    if open_task_ids:
        _append_unique(proof_queue, [f"task completion packet {open_task_ids[0]}"])
    _append_unique(proof_queue, list(recovery_proof_queue))
    _append_unique(proof_queue, list(execution_health_verification_proof_queue or []))
    learning_commands = list(learning_actionable_proof_queue or [])
    if learning_commands:
        _append_unique(proof_queue, learning_commands)
    elif learning_closure_command:
        _append_unique(proof_queue, [learning_closure_command])
        _append_unique(proof_queue, list(learning_proof_queue))
    else:
        _append_unique(proof_queue, list(learning_proof_queue))
    _append_unique(proof_queue, baseline_commands)
    _append_unique(proof_queue, selected_agi_closure_commands)
    return proof_queue


def _completion_actionable_queue_from_ordered(
    *,
    completion_queue: list[str],
    learning_closure_command: str,
    learning_actionable_queue: list[str],
) -> tuple[list[str], bool]:
    learning_prerequisites_inserted = bool(
        learning_actionable_queue
        and learning_closure_command
        and learning_closure_command in completion_queue
        and learning_actionable_queue[0] != learning_closure_command
    )
    if not learning_prerequisites_inserted:
        return list(completion_queue), False

    completion_actionable_queue: list[str] = []
    for command in completion_queue:
        if command == learning_closure_command:
            _append_unique(completion_actionable_queue, learning_actionable_queue)
        elif command:
            completion_actionable_queue.append(command)
    return completion_actionable_queue, True


def _agi_implementation_preflight_metadata(
    *,
    build_packet_ready: bool,
    pending_approvals: list[Any],
    recovery_closure: dict[str, Any],
    learning_debt: dict[str, Any],
    build_command: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not build_packet_ready:
        blockers.append("target files, focused verification, and acceptance checks must all be present")
    if pending_approvals:
        blockers.append("pending approvals must be reviewed or dismissed before starting a new AGI build slice")
    if recovery_closure.get("blocks_completion_claim"):
        blockers.append("execution recovery closure proof debt must be closed first")
    if learning_debt.get("blocks_completion_claim"):
        blockers.append("execution learning proof debt must be closed first")

    next_command = build_command
    if not build_packet_ready:
        next_command = "agi gates"
    elif pending_approvals:
        next_command = "approval review"
    elif recovery_closure.get("blocks_completion_claim"):
        next_command = (
            str(recovery_closure.get("next_proof_command") or "")
            or str(recovery_closure.get("next_required_command") or "")
            or _recovery_closure_checklist_command(recovery_closure)
            or "recovery closure checklist"
        )
    elif learning_debt.get("blocks_completion_claim"):
        next_command = (
            str(learning_debt.get("next_evidence_command") or "")
            or str(learning_debt.get("next_proof_command") or "")
            or str(learning_debt.get("next_required_command") or "")
            or "execution learning closure"
        )

    return {
        "agi_next_implementation_preflight_ready": not blockers,
        "agi_next_implementation_preflight_blockers": blockers,
        "agi_next_implementation_preflight_blocker_count": len(blockers),
        "agi_next_implementation_preflight_next_command": "" if not blockers else next_command,
    }


def _execution_health_verification_snapshot(recent_runs: list[Any]) -> dict[str, Any]:
    action_runs = [row for row in recent_runs if _is_execution_action_row(row)]
    verification_tool_names = {
        "verification_receipt",
        "runtime_trace_receipt",
        "verification_packet",
        "execution_audit_gate",
        "execution_recovery_packet",
        "execution_health_report",
        "recovery_closure_checklist",
        "after_action_learning_packet",
        "execution_acceptance_gate",
        "completion_audit_packet",
        "task_completion_packet",
        "evidence_ledger",
    }
    verification_runs = [
        row
        for row in recent_runs
        if _row_text(row, "tool_name") in verification_tool_names
    ]
    if not recent_runs:
        state = "no_recent_tool_runs"
    elif verification_runs:
        state = "present"
    else:
        state = "missing"
    proof_queue = ["verification receipt latest"] if state in {"missing", "no_recent_tool_runs"} else []
    proof_queue_preview = proof_queue[:3]
    gap_count = max(len(recent_runs) - len(verification_runs), 0)
    first_gap = ""
    if state == "no_recent_tool_runs":
        first_gap = "no recent tool runs have been recorded"
    elif proof_queue:
        first_gap = (
            "recent tool activity has no verification receipt"
            if gap_count
            else "verification receipt is required before completion"
        )
    return {
        "state": state,
        "recent_tool_runs": len(recent_runs),
        "recent_action_runs": len(action_runs),
        "recent_verification_runs": len(verification_runs),
        "gap_count": gap_count,
        "first_gap": first_gap,
        "proof_queue": proof_queue,
        "proof_queue_count": len(proof_queue),
        "proof_queue_preview": proof_queue_preview,
        "proof_queue_preview_count": len(proof_queue_preview),
        "proof_queue_remaining_count": max(len(proof_queue) - len(proof_queue_preview), 0),
        "next_proof_command": proof_queue[0] if proof_queue else "",
        "next_required_command": proof_queue[0] if proof_queue else "",
        "blocks_completion_claim": bool(proof_queue),
    }


def _execution_health_recovery_closure_snapshot(store: MemoryStore, recent_runs: list[Any]) -> dict[str, Any]:
    risky_levels = {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
    action_runs = [
        row
        for row in recent_runs
        if _is_execution_action_row(row)
    ]
    failed_action_runs, approval_held_action_runs = _recent_tool_run_attention_buckets(action_runs)
    recovery_context_action_runs = [row for row in action_runs if row not in approval_held_action_runs]
    newest_problem = None
    for row in recovery_context_action_runs:
        risk = _row_text(row, "risk")
        ok = _row_bool(row, "ok")
        approved = _row_bool(row, "approved")
        approval_id = _row_int(row, "approval_id")
        if not ok or (risk in risky_levels and (not approved or approval_id is None)):
            newest_problem = row
            break

    target = newest_problem or (recovery_context_action_runs[0] if recovery_context_action_runs else (action_runs[0] if action_runs else None))
    target_run_id = _row_int(target, "id") if target is not None else None
    target_tool_name = _row_text(target, "tool_name") if target is not None else ""
    target_ok = _row_bool(target, "ok") if target is not None else None
    target_risk = _row_text(target, "risk") if target is not None else ""
    target_approved = _row_bool(target, "approved") if target is not None else None
    target_approval_id = _row_int(target, "approval_id") if target is not None else None
    approval_problem = bool(
        target is not None
        and target_risk in risky_levels
        and (not target_approved or target_approval_id is None)
    )

    target_verification_runs = [
        row
        for row in recent_runs
        if _row_text(row, "tool_name") == "verification_receipt" and _row_metadata(row).get("run_id") == target_run_id
    ]
    target_recovery_runs = [
        row
        for row in recent_runs
        if _row_text(row, "tool_name") == "execution_recovery_packet" and _row_metadata(row).get("run_id") == target_run_id
    ]
    target_learning_runs = [
        row
        for row in recent_runs
        if _row_text(row, "tool_name") == "after_action_learning_packet" and _row_metadata(row).get("run_id") == target_run_id
    ]

    output_approval_ids = _ids_from_text(
        _row_text(target, "output") if target is not None else "",
        ("approval #", "approval id", "approval packet", "queued as approval"),
    )
    approval_proof_ids: list[int] = []
    for candidate in ([target_approval_id] if target_approval_id is not None else []) + output_approval_ids:
        if candidate not in approval_proof_ids:
            approval_proof_ids.append(candidate)

    missing: list[str] = []
    commands: list[str] = []
    if target_run_id is not None and newest_problem is not None:
        if not target_verification_runs:
            missing.append("target_verification_receipt")
            commands.append(f"verification receipt {target_run_id}")
        if not target_recovery_runs:
            missing.append("target_recovery_packet")
            commands.append(f"execution recovery packet {target_run_id}")
        if not target_learning_runs:
            missing.append("target_after_action_learning_packet")
            commands.append(f"after-action learning packet {target_run_id}")
            commands.append(f"execution learning closure {target_run_id}")
        if approval_problem:
            missing.append("approval_chain_proof")
            if approval_proof_ids:
                first = approval_proof_ids[0]
                _append_unique(
                    commands,
                    [
                        f"approval readiness {first}",
                        f"approval packet {first}",
                        f"approval chain proof {first}",
                        f"verification receipt <approved run id from approval chain proof {first}>",
                    ],
                )
            else:
                commands.append("approval history")

    if target_run_id is None:
        state = "no_recent_execution"
    elif newest_problem is None:
        state = "not_needed"
    elif missing:
        state = "blocked_missing_" + "_and_".join(missing)
    else:
        state = "ready_for_operator_retry_review"

    return {
        "target_run_id": target_run_id,
        "target_tool_name": target_tool_name,
        "target_ok": target_ok,
        "target_risk": target_risk,
        "target_approved": target_approved,
        "target_approval_id": target_approval_id,
        "newest_problem_found": newest_problem is not None,
        "approval_problem": approval_problem,
        "approval_proof_ids": approval_proof_ids,
        "target_verification_receipts": len(target_verification_runs),
        "target_recovery_packets": len(target_recovery_runs),
        "target_after_action_learning_packets": len(target_learning_runs),
        "failed_action_runs": len(failed_action_runs),
        "approval_held_action_runs": len(approval_held_action_runs),
        "state": state,
        "missing": missing,
        "missing_count": len(missing),
        "required_commands": commands,
        "next_required_command": commands[0] if commands else "",
        "proof_queue": commands,
        "proof_queue_count": len(commands),
        "next_proof_command": commands[0] if commands else "",
        "ready_to_retry": state == "ready_for_operator_retry_review",
        "blocks_auto_execution": state not in {"no_recent_execution", "not_needed", "ready_for_operator_retry_review"},
        "blocks_completion_claim": state not in {"no_recent_execution", "not_needed", "ready_for_operator_retry_review"},
    }


def _recovery_closure_checklist_command(recovery_closure: dict[str, Any]) -> str:
    return "recovery closure checklist" if recovery_closure.get("blocks_completion_claim") else ""


def _execution_proof_handoff_aliases(execution_proof_handoff: dict[str, Any]) -> dict[str, Any]:
    recovery_queue = list(execution_proof_handoff.get("execution_health_recovery_closure_proof_queue") or [])
    learning_queue = list(execution_proof_handoff.get("execution_learning_proof_queue") or [])
    actionable_learning_queue = list(execution_proof_handoff.get("execution_learning_actionable_proof_queue") or [])
    return {
        "execution_proof_recovery_closure_state": execution_proof_handoff.get("execution_health_recovery_closure_state"),
        "execution_proof_recovery_closure_ready_to_retry": execution_proof_handoff.get("execution_health_recovery_closure_ready_to_retry"),
        "execution_proof_recovery_closure_missing": list(execution_proof_handoff.get("execution_health_recovery_closure_missing") or []),
        "execution_proof_recovery_closure_missing_count": execution_proof_handoff.get("execution_health_recovery_closure_missing_count"),
        "execution_proof_recovery_closure_proof_queue": recovery_queue,
        "execution_proof_recovery_closure_proof_queue_count": execution_proof_handoff.get("execution_health_recovery_closure_proof_queue_count"),
        "execution_proof_recovery_closure_next_required_command": execution_proof_handoff.get("execution_health_recovery_closure_next_required_command"),
        "execution_proof_recovery_closure_next_proof_command": execution_proof_handoff.get("execution_health_recovery_closure_next_proof_command"),
        "execution_proof_recovery_closure_blocks_completion_claim": execution_proof_handoff.get("execution_health_recovery_closure_blocks_completion_claim"),
        "execution_proof_learning_state": execution_proof_handoff.get("execution_learning_state"),
        "execution_proof_learning_blocks_completion_claim": execution_proof_handoff.get("execution_learning_blocks_completion_claim"),
        "execution_proof_learning_missing": list(execution_proof_handoff.get("execution_learning_missing") or []),
        "execution_proof_learning_missing_count": execution_proof_handoff.get("execution_learning_missing_count"),
        "execution_proof_learning_proof_queue": learning_queue,
        "execution_proof_learning_proof_queue_count": execution_proof_handoff.get("execution_learning_proof_queue_count"),
        "execution_proof_learning_next_required_command": execution_proof_handoff.get("execution_learning_next_required_command"),
        "execution_proof_learning_next_proof_command": execution_proof_handoff.get("execution_learning_next_proof_command"),
        "execution_proof_learning_actionable_proof_queue": actionable_learning_queue,
        "execution_proof_learning_actionable_proof_queue_count": execution_proof_handoff.get("execution_learning_actionable_proof_queue_count"),
        "execution_proof_learning_actionable_next_required_command": execution_proof_handoff.get("execution_learning_actionable_next_required_command"),
        "execution_proof_learning_actionable_next_proof_command": execution_proof_handoff.get("execution_learning_actionable_next_proof_command"),
    }


def _execution_learning_debt_snapshot(recent_runs: list[Any]) -> dict[str, Any]:
    action_runs = [
        row
        for row in recent_runs
        if _is_execution_action_row(row)
    ]
    failed_or_blocked, approval_held_action_runs = _recent_tool_run_attention_buckets(action_runs)
    learning_context_action_runs = [row for row in action_runs if row not in approval_held_action_runs]
    after_action_runs = [row for row in recent_runs if _row_text(row, "tool_name") == "after_action_learning_packet"]
    verification_runs = [
        row
        for row in recent_runs
        if _row_text(row, "tool_name") in {"verification_receipt", "runtime_trace_receipt", "execution_audit_gate"}
    ]
    recovery_runs = [row for row in recent_runs if _row_text(row, "tool_name") == "execution_recovery_packet"]
    learning_closure_runs = [row for row in recent_runs if _row_text(row, "tool_name") == "execution_learning_closure_packet"]
    target = failed_or_blocked[0] if failed_or_blocked else (learning_context_action_runs[0] if learning_context_action_runs else None)
    target_run_id = _row_int(target, "id") if target is not None else None
    target_tool_name = _row_text(target, "tool_name") if target is not None else ""
    target_learning_runs = [
        row
        for row in after_action_runs
        if _row_metadata(row).get("run_id") == target_run_id
    ]
    target_learning_closure_runs: list[Any] = []
    if target_run_id is not None:
        for row in learning_closure_runs:
            metadata = _row_metadata(row)
            handoff = metadata.get("execution_learning_closure_handoff")
            if not isinstance(handoff, dict):
                handoff = {}
            handoff_target = handoff.get("target")
            if not isinstance(handoff_target, dict):
                handoff_target = {}
            closure_target_run_id = (
                metadata.get("target_run_id")
                or metadata.get("run_id")
                or handoff_target.get("run_id")
            )
            if _metadata_int(closure_target_run_id, -1) != target_run_id:
                continue
            closure_ready = (
                metadata.get("learning_closure_ready") is True
                or metadata.get("execution_learning_closure_ready") is True
                or handoff.get("learning_closure_ready") is True
                or metadata.get("learning_closure_state") == "LEARNING_CLOSURE_READY"
                or metadata.get("verdict") == "LEARNING_CLOSURE_READY"
                or handoff.get("verdict") == "LEARNING_CLOSURE_READY"
            )
            if closure_ready:
                target_learning_closure_runs.append(row)
    target_learning_satisfied = bool(target_learning_runs or target_learning_closure_runs)
    target_failure_review_satisfied = bool(target_learning_closure_runs)

    missing: list[str] = []
    commands: list[str] = []
    if target_run_id is not None and not target_learning_satisfied:
        missing.append("target_after_action_learning_packet")
        commands.append(f"after-action learning packet {target_run_id}")
        commands.append(f"execution learning closure {target_run_id}")
    if failed_or_blocked and not target_failure_review_satisfied:
        missing.append("failure_review")
        commands.append(f"execution recovery packet {target_run_id}")
        commands.append(f"failure to test preview: run #{target_run_id} {target_tool_name} failed or blocked")
    if failed_or_blocked and not target_learning_satisfied:
        state = "LEARNING_DEBT_AFTER_FAILURE"
    elif missing:
        state = "LEARNING_REVIEW_REQUIRED"
    elif learning_context_action_runs:
        state = "LEARNING_LOOP_HAS_RECENT_ACTION_CONTEXT"
    elif approval_held_action_runs:
        state = "LEARNING_LOOP_HAS_APPROVAL_HELD_CONTEXT"
    else:
        state = "NO_RECENT_ACTION_RUNS"
    learning_closure_command = f"execution learning closure {target_run_id}" if target_run_id is not None else "execution learning closure"
    learning_evidence_command = f"after-action learning packet {target_run_id}" if target_run_id is not None else "after-action learning packet"
    actionable_commands: list[str] = []
    for command in commands:
        if command == learning_closure_command and "target_after_action_learning_packet" in missing:
            _append_unique(actionable_commands, [learning_evidence_command])
        elif command != learning_closure_command:
            _append_unique(actionable_commands, [command])
    if commands:
        _append_unique(actionable_commands, [learning_closure_command])

    return {
        "state": state,
        "blocks_completion_claim": state in {"LEARNING_DEBT_AFTER_FAILURE", "LEARNING_REVIEW_REQUIRED"},
        "recent_action_runs": len(action_runs),
        "failed_or_blocked_action_runs": len(failed_or_blocked),
        "approval_held_action_runs": len(approval_held_action_runs),
        "learning_context_action_runs": len(learning_context_action_runs),
        "recent_verification_runs": len(verification_runs),
        "recent_recovery_runs": len(recovery_runs),
        "recent_after_action_learning_runs": len(after_action_runs),
        "recent_execution_learning_closure_runs": len(learning_closure_runs),
        "target_run_id": target_run_id,
        "target_tool_name": target_tool_name,
        "target_after_action_learning_packets": len(target_learning_runs),
        "target_execution_learning_closure_packets": len(target_learning_closure_runs),
        "target_learning_satisfied": target_learning_satisfied,
        "target_failure_review_satisfied": target_failure_review_satisfied,
        "learning_closure_command": learning_closure_command,
        "execution_learning_closure_command": learning_closure_command,
        "learning_evidence_command": learning_evidence_command,
        "after_action_learning_command": learning_evidence_command,
        "missing": missing,
        "missing_count": len(missing),
        "required_commands": commands,
        "next_required_command": commands[0] if commands else "",
        "proof_queue": commands,
        "proof_queue_count": len(commands),
        "next_proof_command": commands[0] if commands else "",
        "actionable_required_commands": actionable_commands,
        "actionable_required_command_count": len(actionable_commands),
        "actionable_next_required_command": actionable_commands[0] if actionable_commands else "",
        "actionable_proof_queue": actionable_commands,
        "actionable_proof_queue_count": len(actionable_commands),
        "actionable_next_proof_command": actionable_commands[0] if actionable_commands else "",
        "next_evidence_command": actionable_commands[0] if actionable_commands else "",
    }


def _risk_signals(request: str) -> list[str]:
    text = request.lower()
    signals = []
    for label, keywords in RISK_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            signals.append(label)
    return signals


def _execution_health_recovery_closure(store: MemoryStore, *, limit: int = 20) -> dict[str, Any]:
    from jarvis_v2.tools import autonomy

    recovery_closure = autonomy._execution_health_recovery_closure_snapshot(store.recent_tool_runs(limit=limit))
    proof_queue = list(recovery_closure["proof_queue"])
    recovery_closure["proof_queue"] = proof_queue
    recovery_closure.setdefault("failed_action_runs", 0)
    recovery_closure.setdefault("approval_held_action_runs", 0)
    return recovery_closure


def _execution_health_recovery_metadata(recovery_closure: dict[str, Any]) -> dict[str, Any]:
    return {
        "recovery_closure_state": recovery_closure["state"],
        "recovery_closure_ready_to_retry": recovery_closure["ready_to_retry"],
        "recovery_closure_missing": recovery_closure["missing"],
        "recovery_closure_missing_count": recovery_closure["missing_count"],
        "recovery_closure_required_commands": recovery_closure["required_commands"],
        "recovery_closure_next_required_command": recovery_closure["next_required_command"],
        "recovery_closure_blocks_auto_execution": recovery_closure["blocks_auto_execution"],
        "recovery_closure_target_run_id": recovery_closure["target_run_id"],
        "recovery_closure_target_tool_name": recovery_closure["target_tool_name"],
        "recovery_closure_failed_action_runs": recovery_closure.get("failed_action_runs", 0),
        "recovery_closure_approval_held_action_runs": recovery_closure.get("approval_held_action_runs", 0),
        "recovery_closure_proof_queue": recovery_closure["proof_queue"],
        "recovery_closure_proof_queue_count": recovery_closure["proof_queue_count"],
        "recovery_closure_next_proof_command": recovery_closure["next_proof_command"],
    }


def _execution_health_recovery_lines(recovery_closure: dict[str, Any]) -> list[str]:
    if not recovery_closure["blocks_auto_execution"]:
        return []
    commands = recovery_closure["required_commands"]
    return [
        "",
        "Execution health recovery closure:",
        f"- state: {recovery_closure['state']}",
        f"- ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
        f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
        f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
        f"- recovery next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- recovery next required: none",
        f"- command queue: {', '.join(f'`{command}`' for command in commands) if commands else 'none'}",
    ]


LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")


def _short(value: Any, limit: int = 240) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _short_raw(value: Any, limit: int = 240) -> str:
    text = "" if value is None else str(value).strip()
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _metadata_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if value is True:
        return True
    if value is False:
        return False
    return default


def _execution_case_metadata(case: Any) -> dict[str, Any]:
    try:
        if "metadata" not in case.keys():
            return {}
        raw = case["metadata"]
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _approval_id_from_text(value: str) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    for marker in ("approval #", "approval id", "approval"):
        text = text.replace(marker, " ")
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    if not digits:
        return None
    try:
        return int(digits[0])
    except ValueError:
        return None


def _approval_any_status(store: MemoryStore, approval_id: int):
    for status in ("pending", "approved", "dismissed"):
        row = store.get_pending_approval(approval_id, status=status)
        if row is not None:
            return row
    return None


def _approval_ids_from_case_events(events: list[Any]) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    pattern = re.compile(
        r"\b(approval\s+packet|approval\s+receipt|approval\s+#|approval\s+id|approved\s+approval|approved\s+request|last-look\s+approval|approval)\s*#?\s*(\d+)\b",
        re.IGNORECASE,
    )
    for event in events:
        try:
            receipt_kind = str(event["receipt_kind"] or "").strip().lower()
            receipt_id = str(event["receipt_id"] or "").strip()
            text = f"{event['event_type']} {event['summary']} {event['receipt_kind']} {event['receipt_id']}"
        except Exception:
            receipt_kind = ""
            receipt_id = ""
            text = str(event)

        if receipt_kind in {"approval", "approval_packet", "approval receipt", "approval_receipt"} and receipt_id:
            approval_id = _approval_id_from_text(receipt_id)
            if approval_id is not None and approval_id not in seen:
                ids.append(approval_id)
                seen.add(approval_id)
            continue

        matched = False
        for match in pattern.finditer(text):
            matched = True
            approval_id = int(match.group(2))
            if approval_id not in seen:
                ids.append(approval_id)
                seen.add(approval_id)
        if matched:
            continue

        lowered = text.lower()
        if not any(marker in lowered for marker in ("approval packet", "approval #", "approval id", "approval receipt", "approved approval", "approved request", "last-look approval")):
            continue
        approval_id = _approval_id_from_text(text)
        if approval_id is not None and approval_id not in seen:
            ids.append(approval_id)
            seen.add(approval_id)
    return ids


def _approval_chain_state_for_events(store: MemoryStore, events: list[Any]) -> tuple[str, list[int], list[int]]:
    approval_ids = _approval_ids_from_case_events(events)
    if not approval_ids:
        return "missing", [], []

    linked_runs = store.approved_tool_runs_for_approvals(approval_ids, limit=20)
    linked_run_ids = [int(row["id"]) for row in linked_runs]
    successful_linked_runs = [row for row in linked_runs if bool(row["ok"])]
    if successful_linked_runs:
        return "proven", approval_ids, linked_run_ids

    states = []
    for approval_id in approval_ids:
        row = _approval_any_status(store, approval_id)
        if row is None:
            states.append(f"approval {approval_id} not found")
        elif str(row["status"]).lower() != "approved":
            states.append(f"approval {approval_id} {row['status']}")
        else:
            states.append(f"approval {approval_id} approved without successful linked rerun")
    return "; ".join(states) if states else "unproven", approval_ids, linked_run_ids


def _verification_receipt_ids_from_case_events(events: list[Any]) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    text_markers = ("verification receipt", "runtime trace receipt", "tool run", "run id", "run #")
    receipt_kinds = {"verification", "verification_receipt", "runtime trace", "runtime_trace", "runtime_trace_receipt", "tool run", "tool_run"}
    for event in events:
        try:
            receipt_kind = str(event["receipt_kind"] or "").strip().lower()
            receipt_id = str(event["receipt_id"] or "").strip()
            text = f"{event['event_type']} {event['summary']} {event['receipt_kind']} {event['receipt_id']}"
        except Exception:
            receipt_kind = ""
            receipt_id = ""
            text = str(event)

        candidate: int | None = None
        if receipt_kind in receipt_kinds and receipt_id:
            candidate = _approval_id_from_text(receipt_id)
        else:
            lowered = text.lower()
            if not any(marker in lowered for marker in text_markers):
                continue
            candidate = _approval_id_from_text(text)

        if candidate is not None and candidate not in seen:
            ids.append(candidate)
            seen.add(candidate)
    return ids


def _verification_receipt_refs_from_case_events(events: list[Any]) -> list[tuple[str, int]]:
    refs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    receipt_kinds = {
        "verification": "tool_run",
        "verification_receipt": "tool_run",
        "tool run": "tool_run",
        "tool_run": "tool_run",
        "runtime trace": "runtime_trace_receipt",
        "runtime_trace": "runtime_trace_receipt",
        "runtime_trace_receipt": "runtime_trace_receipt",
    }
    for event in events:
        try:
            receipt_kind = str(event["receipt_kind"] or "").strip().lower()
            receipt_id = str(event["receipt_id"] or "").strip()
            text = f"{event['event_type']} {event['summary']} {event['receipt_kind']} {event['receipt_id']}"
        except Exception:
            receipt_kind = ""
            receipt_id = ""
            text = str(event)

        normalized_kind = receipt_kinds.get(receipt_kind)
        if normalized_kind and receipt_id:
            candidate = _approval_id_from_text(receipt_id)
            if candidate is not None:
                ref = (normalized_kind, candidate)
                if ref not in seen:
                    refs.append(ref)
                    seen.add(ref)
                continue

        for ref in _verification_receipt_refs_from_text(text):
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)
    return refs


def _verification_receipt_ids_from_text(value: str) -> list[int]:
    text = str(value or "")
    ids: list[int] = []
    seen: set[int] = set()
    pattern = re.compile(
        r"\b(?:verification\s+receipt|runtime\s+trace\s+receipt|tool\s+run|run\s+id|run\s+#)\s*#?\s*(\d+)\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        candidate = int(match.group(1))
        if candidate not in seen:
            ids.append(candidate)
            seen.add(candidate)
    return ids


def _verification_receipt_refs_from_text(value: str) -> list[tuple[str, int]]:
    text = str(value or "")
    refs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    pattern = re.compile(
        r"\b(runtime\s+trace\s+receipt|verification\s+receipt|tool\s+run|run\s+id|run\s+#)\s*#?\s*(\d+)\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        marker = " ".join(match.group(1).lower().split())
        receipt_id = int(match.group(2))
        if marker == "runtime trace receipt":
            kind = "runtime_trace_receipt"
        else:
            kind = "tool_run"
        ref = (kind, receipt_id)
        if ref not in seen:
            refs.append(ref)
            seen.add(ref)
    if not refs:
        for run_id in _verification_receipt_ids_from_text(text):
            ref = ("tool_run", run_id)
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)
    return refs


def _missing_runtime_trace_message_ids(store: MemoryStore, message_ids: list[int]) -> list[int]:
    missing: list[int] = []
    seen: set[int] = set()
    for message_id in message_ids:
        if message_id in seen:
            continue
        seen.add(message_id)
        row = store.get_message(message_id)
        if row is None:
            missing.append(message_id)
            continue
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        if row["role"] != "assistant" or not isinstance(metadata.get("runtime_trace"), dict):
            missing.append(message_id)
    return missing


def _approved_run_ids_from_runtime_trace_messages(store: MemoryStore, message_ids: list[int], approval_id: int | None) -> list[int]:
    run_ids: list[int] = []
    seen: set[int] = set()
    if approval_id is None:
        return run_ids
    for message_id in message_ids:
        row = store.get_message(message_id)
        if row is None:
            continue
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            continue
        trace = metadata.get("runtime_trace")
        if not isinstance(trace, dict):
            continue
        results = trace.get("tool_results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            rerun = result.get("approved_rerun_result")
            if not isinstance(rerun, dict):
                continue
            if rerun.get("approved_approval_id") != approval_id or not rerun.get("ok"):
                continue
            run_id = rerun.get("logged_tool_run_id")
            if isinstance(run_id, int) and run_id not in seen:
                run_ids.append(run_id)
                seen.add(run_id)
    return run_ids


def _receipt_reference_from_text(value: str) -> tuple[str, str]:
    text = str(value or "")
    marker_kinds = (
        ("execution_recovery_packet", r"execution\s+recovery\s+packet|execution\s+recovery"),
        ("verification_receipt", r"verification\s+receipt"),
        ("runtime_trace_receipt", r"runtime\s+trace\s+receipt|runtime\s+trace"),
        ("approval_packet", r"approval\s+packet|approval\s+receipt|approval"),
        ("tool_run", r"tool\s+run|run\s+id|run\s+#"),
    )
    for receipt_kind, marker_pattern in marker_kinds:
        pattern = re.compile(rf"\b(?:{marker_pattern})\s*#?\s*(\d+)\b", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return receipt_kind, match.group(1)
    return "", ""


def _missing_tool_run_ids(store: MemoryStore, run_ids: list[int]) -> list[int]:
    missing: list[int] = []
    seen: set[int] = set()
    for run_id in run_ids:
        if run_id in seen:
            continue
        seen.add(run_id)
        if store.get_tool_run(run_id) is None:
            missing.append(run_id)
    return missing


def _normalized_receipt_kind(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "verification": "verification_receipt",
        "verification_receipt": "verification_receipt",
        "runtime_trace": "runtime_trace_receipt",
        "runtime_trace_receipt": "runtime_trace_receipt",
        "execution_recovery": "execution_recovery_packet",
        "execution_recovery_packet": "execution_recovery_packet",
        "approval": "approval_packet",
        "approval_receipt": "approval_packet",
        "approval_packet": "approval_packet",
        "tool_run": "tool_run",
        "run": "tool_run",
    }
    return aliases.get(text, text)


def make_harness_tools(
    store: MemoryStore,
    vault: ObsidianVault,
    list_tools: Callable[[], list[Any]],
    storage_fallback: Callable[[], dict[str, Any] | None] | None = None,
    config: JarvisConfig | None = None,
):
    def _first_present(args: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
        for name in names:
            if name in args and args[name] is not None:
                return args[name]
        return default

    def _resolve_execution_case_arg(tool_name: str, args: dict[str, Any]) -> tuple[Any | None, ToolResult | None]:
        raw_id = _first_present(args, ("case_id", "id"), "latest")
        if isinstance(raw_id, str) and raw_id.strip().lower() in {"", "latest", "last", "newest"}:
            cases = store.list_execution_cases(limit=1)
            return (cases[0] if cases else None), None
        try:
            case_id = int(raw_id)
        except (TypeError, ValueError):
            output = (
                "Execution case id must be a number or `latest`. "
                f"{HARNESS_CASE_ID_RECOVERY_ACTION}"
            )
            return None, ToolResult(
                tool_name,
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(reason="bad_case_id", case_id=None, raw_case_id=_short_raw(raw_id, limit=80)),
                    output=output,
                    action=HARNESS_CASE_ID_RECOVERY_ACTION,
                ),
            )
        if case_id <= 0:
            output = (
                "Execution case id must be a positive number or `latest`. "
                f"{HARNESS_CASE_ID_RECOVERY_ACTION}"
            )
            return None, ToolResult(
                tool_name,
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(reason="bad_case_id", case_id=None, raw_case_id=_short_raw(raw_id, limit=80)),
                    output=output,
                    action=HARNESS_CASE_ID_RECOVERY_ACTION,
                ),
            )
        return store.get_execution_case(case_id), None

    def _storage_readiness_snapshot() -> dict[str, Any]:
        fallback: dict[str, Any] | None = None
        if storage_fallback is not None:
            try:
                fallback = storage_fallback()
            except Exception:
                fallback = None
        fallback_active = bool(fallback)
        diagnostics, _diagnostics_source = _configured_storage_diagnostics(config, fallback)
        configured_ready = bool(diagnostics.get("available"))
        configured_issues = [str(issue) for issue in list(diagnostics.get("issues") or [])]
        if fallback_active and configured_ready:
            recovery_mode = "restart_runtime_to_configured_storage"
            recovery_next_operator_action = (
                "restart or reload Jarvis with the configured durable storage envs, then run `storage status`"
            )
        elif fallback_active:
            recovery_mode = "repair_configured_storage_then_restart_runtime"
            recovery_next_operator_action = (
                "point Jarvis at writable durable storage, run the no-write storage check, then restart or reload Jarvis"
            )
        elif not configured_ready:
            recovery_mode = "repair_configured_storage"
            recovery_next_operator_action = (
                "point Jarvis at writable durable storage and run the no-write storage check"
            )
        else:
            recovery_mode = "none"
            recovery_next_operator_action = ""
        if fallback_active and configured_ready:
            next_commands = [
                "storage status",
                STORAGE_RECOVERY_CHECK_COMMAND,
                "storage status",
            ]
        elif fallback_active:
            next_commands = [
                "storage status",
                STORAGE_RECOVERY_CHECK_COMMAND,
                BOOTSTRAP_CHECK_COMMAND,
                BOOTSTRAP_WRITE_COMMAND,
            ]
        elif not configured_ready:
            next_commands = [
                "storage status",
                STORAGE_RECOVERY_CHECK_COMMAND,
                BOOTSTRAP_CHECK_COMMAND,
                BOOTSTRAP_WRITE_COMMAND,
            ]
        else:
            next_commands = []
        if fallback_active and configured_ready:
            blocker = "runtime is using workspace-local fallback storage"
        elif fallback_active:
            blocker_parts = configured_issues or ["primary storage is not writable"]
            blocker = "; ".join([*blocker_parts, "runtime is using workspace-local fallback storage"])
        elif not configured_ready:
            issues = configured_issues or ["configured storage needs attention"]
            blocker = "; ".join(str(issue) for issue in issues)
        else:
            blocker = ""
        recovery_required = fallback_active or not configured_ready
        return {
            "active": fallback_active,
            "reason": str((fallback or {}).get("reason") or ""),
            "exception_type": str((fallback or {}).get("exception_type") or ""),
            "db_path_display": str((fallback or {}).get("db_path_display") or ""),
            "vault_path_display": str((fallback or {}).get("vault_path_display") or ""),
            "blocks_completion_claim": recovery_required,
            "recovery_required": recovery_required,
            "recovery_reason": blocker,
            "recovery_mode": recovery_mode,
            "recovery_next_operator_action": recovery_next_operator_action,
            "recovery_restart_required": fallback_active,
            "recovery_check_tool_command": STORAGE_RECOVERY_CHECK_COMMAND if recovery_required else "",
            "recovery_check_command": BOOTSTRAP_CHECK_COMMAND if recovery_required else "",
            "recovery_command": BOOTSTRAP_WRITE_COMMAND if recovery_required else "",
            "next_commands": next_commands,
            "blocker": blocker,
            "issues": configured_issues + (["runtime is using workspace-local fallback storage"] if fallback_active else []),
            "issue_count": len(configured_issues) + (1 if fallback_active else 0),
        }

    def _storage_handoff_from_metadata(metadata: dict[str, Any], *, source: str) -> dict[str, Any]:
        next_commands = [str(command) for command in list(metadata.get("storage_readiness_next_commands") or [])]
        proof_queue = [str(command) for command in list(metadata.get("storage_readiness_proof_queue") or next_commands)]
        proof_queue_preview = [str(command) for command in list(metadata.get("storage_readiness_proof_queue_preview") or proof_queue[:5])]
        proof_queue_remaining_count = int(
            metadata.get(
                "storage_readiness_proof_queue_remaining_count",
                max(len(proof_queue) - len(proof_queue_preview), 0),
            )
        )
        issues = [str(issue) for issue in list(metadata.get("storage_issues") or [])]
        next_proof_command = str(
            metadata.get("storage_readiness_next_proof_command")
            or (proof_queue[0] if proof_queue else "")
        )
        first_proof_command = str(
            metadata.get("storage_readiness_first_proof_command")
            or (proof_queue[0] if proof_queue else "")
        )
        next_required_command = str(
            metadata.get("storage_readiness_next_required_command")
            or next_proof_command
            or (next_commands[0] if next_commands else "")
        )
        recovery_check_tool_command = str(metadata.get("storage_recovery_check_tool_command") or "")
        recovery_check_command = str(metadata.get("storage_recovery_check_command") or "")
        fallback_command = next_required_command or next_proof_command or recovery_check_tool_command or recovery_check_command
        return _safe_metadata(
            source=source,
            handoff_ready=True,
            storage_handoff_ready=True,
            ready_for_operator=True,
            state_changed=False,
            changed=[],
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
            storage_status=str(metadata.get("storage_status") or "unknown"),
            storage_available=bool(metadata.get("storage_available", False)),
            storage_workspace_local_notes=bool(metadata.get("storage_workspace_local_notes", False)),
            storage_runtime_fallback_active=_metadata_bool(metadata.get("storage_runtime_fallback_active")),
            storage_runtime_fallback_reason=str(metadata.get("storage_runtime_fallback_reason") or ""),
            storage_runtime_fallback_exception_type=str(metadata.get("storage_runtime_fallback_exception_type") or ""),
            storage_runtime_fallback_db_path_display=str(metadata.get("storage_runtime_fallback_db_path_display") or ""),
            storage_runtime_fallback_vault_path_display=str(metadata.get("storage_runtime_fallback_vault_path_display") or ""),
            storage_readiness_blocks_completion_claim=_metadata_bool(metadata.get("storage_readiness_blocks_completion_claim")),
            storage_readiness_blocker=str(metadata.get("storage_readiness_blocker") or ""),
            storage_readiness_next_commands=next_commands,
            storage_readiness_next_command_count=len(next_commands),
            storage_readiness_next_required_command=next_required_command,
            storage_readiness_next_proof_command=next_proof_command,
            storage_readiness_proof_queue=proof_queue,
            storage_readiness_proof_queue_count=len(proof_queue),
            storage_readiness_proof_queue_preview=proof_queue_preview,
            storage_readiness_proof_queue_preview_count=int(
                metadata.get("storage_readiness_proof_queue_preview_count", len(proof_queue_preview))
            ),
            storage_readiness_proof_queue_remaining_count=proof_queue_remaining_count,
            storage_readiness_first_proof_command=first_proof_command,
            storage_recovery_required=_metadata_bool(metadata.get("storage_recovery_required")),
            storage_recovery_reason=str(metadata.get("storage_recovery_reason") or ""),
            storage_recovery_mode=str(metadata.get("storage_recovery_mode") or "none"),
            storage_recovery_next_operator_action=str(metadata.get("storage_recovery_next_operator_action") or ""),
            storage_recovery_restart_required=_metadata_bool(metadata.get("storage_recovery_restart_required")),
            storage_recovery_check_tool_command=recovery_check_tool_command,
            storage_recovery_check_command=recovery_check_command,
            storage_recovery_check_api=str(metadata.get("storage_recovery_check_api") or STORAGE_RECOVERY_CHECK_API),
            storage_recovery_command=str(metadata.get("storage_recovery_command") or ""),
            storage_issues=issues,
            storage_issue_count=int(metadata.get("storage_issue_count") or len(issues)),
            next_commands=proof_queue or next_commands,
            next_command=fallback_command,
            next_proof_command=next_proof_command or recovery_check_tool_command or recovery_check_command,
            boundaries={
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
            },
        )

    def priority_goal(_: dict[str, Any]) -> ToolResult:
        active_goals = store.list_goals(status="active", limit=20)
        priority_memories = []
        for query in [
            '"finish Jarvis"',
            '"agent harness"',
            '"priority goal"',
            '"AGI-like personal assistant"',
        ]:
            try:
                priority_memories.extend(store.search_memories(query, limit=5))
            except Exception:
                continue

        seen_memory_ids: set[int] = set()
        unique_memories = []
        for memory in priority_memories:
            memory_id = int(memory["id"])
            if memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(memory_id)
            unique_memories.append(memory)

        matching_goals = [
            goal
            for goal in active_goals
            if any(
                marker in f"{goal['title']} {goal['purpose']}".lower()
                for marker in ("jarvis", "harness", "agi")
            )
        ]
        priority_build_order = [
            "Keep the command-first GUI and conversation loop usable.",
            "Make routing, memory, tools, approvals, audit, and recovery dependable.",
            "Add real-execution abilities only behind exact arguments, last-look approval packets, approval chain proof, verification, and rollback/recovery evidence.",
            "Turn repeated feedback and failures into tests, preferences, skills, or documented harness improvements.",
        ]
        matching_goal_rows = [
            {
                "id": goal["id"],
                "title": str(goal["title"])[:160],
                "purpose": str(goal["purpose"] or "")[:240],
                "status": goal["status"],
                "non_authorizing": True,
            }
            for goal in matching_goals[:5]
        ]
        priority_memory_rows = [
            {
                "id": memory["id"],
                "category": memory["category"],
                "title": str(memory["title"] or "")[:160],
                "body_preview_chars": min(len(str(memory["body"] or "")), 160),
                "non_authorizing": True,
            }
            for memory in unique_memories[:5]
        ]
        priority_metadata = {
            "active_matching_goals": len(matching_goals),
            "active_matching_goal_rows": matching_goal_rows,
            "active_matching_goal_row_count": len(matching_goal_rows),
            "priority_memories": len(unique_memories),
            "priority_memory_rows": priority_memory_rows,
            "priority_memory_row_count": len(priority_memory_rows),
            "priority": "finish_jarvis_agent_harness",
            "priority_objective": "Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant.",
            "priority_build_order": priority_build_order,
            "priority_build_order_count": len(priority_build_order),
            "approval_gates_overridden": False,
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "operator_timeboxes_override_priority": True,
            "stop_times_override_priority": True,
        }
        priority_goal_handoff = _safe_metadata(source="priority_goal", **priority_metadata)

        lines = [
            "Jarvis priority goal:",
            "- Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant.",
            "- Until Jarvis is finished, treat this as the top build priority over side quests.",
            "",
            "Working definition:",
            "- The model is the engine; Jarvis must also provide steering, pedals, brakes, dashboard, memory, tools, approvals, audit, recovery, and verification.",
            "",
            "Priority build order:",
            *[f"{index}. {item}" for index, item in enumerate(priority_build_order, start=1)],
            "",
            "Current local evidence:",
            f"- active Jarvis/harness/AGI goals: {len(matching_goals)}",
            f"- priority memories found: {len(unique_memories)}",
        ]
        if matching_goals:
            lines.append("")
            lines.append("Active matching goals:")
            for goal in matching_goals[:5]:
                lines.append(f"- #{goal['id']} {goal['title']}: {goal['purpose'] or 'no purpose recorded'}")
        if unique_memories:
            lines.append("")
            lines.append("Recent matching memory:")
            for memory in unique_memories[:5]:
                lines.append(f"- [{memory['category']}] {memory['title']}: {memory['body'][:160]}")

        lines.extend(
            [
                "",
                "Guardrail:",
                "- This priority does not override safety boundaries. Computer control, shell/code execution, personal data, external side effects, destructive actions, and risky actions stay approval-gated.",
                "- This priority also does not override the operator's explicit stop times, work windows, pause commands, or newer instructions.",
            ]
        )

        return ToolResult(
            "priority_goal",
            True,
            "\n".join(lines),
            _safe_metadata(
                **priority_metadata,
                priority_goal_handoff=priority_goal_handoff,
            ),
        )

    def harness_status(_: dict[str, Any]) -> ToolResult:
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        toolset_counts = Counter(tool.toolset for tool in tools)
        risk_counts = Counter(tool.risk.name for tool in tools)
        pending = store.list_pending_approvals(limit=10)
        jobs = store.list_jobs()
        enabled_jobs = [job for job in jobs if job["enabled"]]
        active_goals = store.list_goals(status="active", limit=10)
        open_tasks = store.list_tasks(status="open", limit=10)
        recent_runs = store.recent_tool_runs(limit=10)

        lines = [
            "Jarvis agent harness status:",
            "",
            "Definition:",
            "- Jarvis is the harness around AI brains: interface, state, tools, approvals, audit, recovery, verification, and learning.",
            "- The AI model is the engine; the harness is the steering wheel, pedals, brakes, dashboard, sensors, maintenance system, and safety cage.",
            "- The AGI direction is to make this harness capable enough to host stronger brains and reliably pursue the operator's goals in the real world.",
            "",
            "Current harness signals:",
            f"- registered tools: {len(tools)}",
            f"- toolsets: {len(toolset_counts)}",
            f"- risk mix: " + ", ".join(f"{risk}={count}" for risk, count in sorted(risk_counts.items())),
            f"- pending approvals: {len(pending)}",
            f"- enabled scheduled jobs: {len(enabled_jobs)} / {len(jobs)}",
            f"- active goals: {len(active_goals)}",
            f"- open tasks: {len(open_tasks)}",
            f"- recent audited tool runs: {len(recent_runs)}",
            "",
            "Harness components:",
        ]

        ready_components = 0
        partial_components = 0
        missing_components = 0
        component_rows: list[dict[str, Any]] = []
        for component in HARNESS_COMPONENTS:
            present_toolsets = sorted(component["toolsets"] & set(toolset_counts))
            present_evidence = sorted(component["evidence"] & tool_names)
            if len(present_evidence) >= 3 and present_toolsets:
                status = "strong foundation"
                ready_components += 1
            elif present_evidence or present_toolsets:
                status = "partial foundation"
                partial_components += 1
            else:
                status = "missing"
                missing_components += 1
            component_rows.append(
                {
                    "name": component["name"],
                    "status": status,
                    "goal": component["goal"],
                    "present_toolsets": present_toolsets,
                    "present_toolset_count": len(present_toolsets),
                    "present_evidence": present_evidence,
                    "present_evidence_count": len(present_evidence),
                    "non_authorizing": True,
                }
            )
            lines.extend(
                [
                    f"- {component['name']}: {status}",
                    f"  Goal: {component['goal']}",
                    f"  Evidence: {', '.join(present_evidence) if present_evidence else 'none yet'}",
                ]
            )

        layer_contract_rows = _harness_layer_contract_rows(tool_names)
        lines.extend(
            [
                "",
                "Typed AGI harness contract:",
            ]
        )
        for row in layer_contract_rows:
            lines.extend(
                [
                    f"- {row['title']}: {row['status']}",
                    f"  Contract: {row['contract']}",
                    f"  Boundary: {row['boundary']}",
                    f"  Evidence: {', '.join(row['present_evidence']) if row['present_evidence'] else 'none yet'}",
                ]
            )

        lines.extend(
            [
                "",
                "AGI-direction gates still ahead:",
                *[f"- {gate}" for gate in AGI_DIRECTION_GATES],
                "",
                "Build rule:",
                "- Improve the harness before increasing autonomy: every new ability needs routing, state, safety boundary, audit, verification, and recovery.",
                "- Read-only/local-safe work may run automatically; shell/code, personal data, computer control, destructive changes, reminders, and external side effects remain approval-gated.",
                "- Priority goals do not override the operator's explicit stop times, work windows, pause commands, or newer instructions.",
                "",
                "Best next commands:",
                "- `architecture map` to inspect the six assistant layers",
                "- `roadmap` to choose the next build phase",
                "- `readiness report` to check setup and blockers",
                "- `work queue` or `focus brief` to select safe work",
            ]
        )

        best_next_commands = [
            "architecture map",
            "roadmap",
            "readiness report",
            "work queue",
            "focus brief",
        ]
        risk_count_rows = [
            {
                "risk": risk,
                "count": count,
                "non_authorizing": True,
            }
            for risk, count in sorted(risk_counts.items())
        ]
        toolset_count_rows = [
            {
                "toolset": toolset,
                "count": count,
                "non_authorizing": True,
            }
            for toolset, count in sorted(toolset_counts.items())
        ]
        layer_contract_metadata = _harness_layer_contract_metadata(layer_contract_rows)
        harness_status_metadata = _safe_metadata(
            components=len(HARNESS_COMPONENTS),
            component_rows=component_rows,
            component_row_count=len(component_rows),
            strong_components=ready_components,
            partial_components=partial_components,
            missing_components=missing_components,
            tools=len(tools),
            toolsets=len(toolset_counts),
            toolset_count_rows=toolset_count_rows,
            toolset_count_row_count=len(toolset_count_rows),
            risk_count_rows=risk_count_rows,
            risk_count_row_count=len(risk_count_rows),
            pending_approvals=len(pending),
            enabled_jobs=len(enabled_jobs),
            jobs=len(jobs),
            active_goals=len(active_goals),
            open_tasks=len(open_tasks),
            recent_tool_runs=len(recent_runs),
            best_next_commands=best_next_commands,
            best_next_command_count=len(best_next_commands),
            **layer_contract_metadata,
            draft_only=True,
            requires_manual_send=True,
            loads_without_execution=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
            operator_timeboxes_override_priority=True,
            stop_times_override_priority=True,
        )
        harness_status_handoff = _safe_metadata(
            source="harness_status",
            **harness_status_metadata,
        )

        return ToolResult(
            "harness_status",
            True,
            "\n".join(lines),
            _safe_metadata(
                **harness_status_metadata,
                harness_status_handoff=harness_status_handoff,
            ),
        )

    def harness_cycle_preview(args: dict[str, Any]) -> ToolResult:
        request = str(args.get("request") or args.get("goal") or "").strip()
        if not request:
            output = (
                "Tell Jarvis the order to preview. "
                "Try `harness cycle: organize my desktop and summarize what changed`. "
                f"{HARNESS_REQUEST_RECOVERY_ACTION}"
            )
            harness_cycle_handoff = _safe_metadata(
                source="harness_cycle_preview",
                reason="missing_request",
                found=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
            )
            return ToolResult(
                "harness_cycle_preview",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(
                        source="harness_cycle_preview",
                        reason="missing_request",
                        found=False,
                        draft_only=True,
                        requires_manual_send=True,
                        loads_without_execution=True,
                        authorizes_execution=False,
                        authorizes_completion_claim=False,
                        approval_granted=False,
                        harness_cycle_handoff=harness_cycle_handoff,
                    ),
                    output=output,
                    action=HARNESS_REQUEST_RECOVERY_ACTION,
                ),
            )

        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=10)
        recovery_closure = _execution_health_recovery_closure(store)
        risks = _risk_signals(request)
        approval_required = bool(risks)
        likely_route = "approval-gated execution preview" if approval_required else "chat/read-only or local-safe execution"
        primary_preview = "execution governor"
        fallback_preview = "action rehearsal" if approval_required else "command diagnosis"

        evidence_tools = [
            name
            for name in [
                "command_diagnosis",
                "chat_loop_preview",
                "execution_governor_packet",
                "command_cockpit_packet",
                "dispatch_decision_packet",
                "action_rehearsal",
                "risk_preflight",
                "action_readiness_packet",
                "approval_execution_packet",
                "recent_tool_runs",
                "learning_review",
            ]
            if name in tool_names
        ]

        lines = [
            "Jarvis harness cycle preview:",
            f"Order: {request}",
            "",
            "Harness principle:",
            "- Treat the model as the engine; the harness provides steering, pedals, brakes, dashboard, audit, recovery, and verification.",
            "- This preview is the steering layer: it explains how Jarvis would move from an order to safe action without running anything.",
            "",
            "Likely route:",
            f"- route: {likely_route}",
            f"- approval required before real execution: {'yes' if approval_required else 'no, unless the planner later finds risky tool use'}",
            f"- risk signals: {', '.join(risks) if risks else 'none obvious from wording'}",
            f"- pending approvals already visible: {len(pending)}",
            f"- recovery closure blocks auto-run: {'yes' if recovery_closure['blocks_auto_execution'] else 'no'}",
            "",
            "Cycle:",
        ]
        for index, (name, description) in enumerate(HARNESS_CYCLE_STAGES, start=1):
            lines.append(f"{index}. {name}: {description}")

        lines.extend(
            [
                "",
                "Preview commands before acting:",
                f"- `{primary_preview}: {request}`",
                f"- `{fallback_preview}: {request}`",
                f"- `action readiness: {request}`",
                "- `approval review` before approving anything already queued",
                "",
                "Execution boundary:",
                "- Read-only/local-safe steps may run automatically.",
                "- Shell/code, personal data, external side effects, destructive changes, reminders, and computer control must stop at approval.",
                "- Approval should include exact planned arguments, last-look packet, audit record, and verification target.",
                "",
                "Harness evidence available now:",
                f"- {', '.join(evidence_tools) if evidence_tools else 'no preview/audit tools found'}",
            ]
        )
        lines.extend(_execution_health_recovery_lines(recovery_closure))
        recovery_metadata = _execution_health_recovery_metadata(recovery_closure)
        harness_cycle_handoff = _safe_metadata(
            source="harness_cycle_preview",
            request=request,
            request_chars=len(request),
            likely_route=likely_route,
            primary_preview=primary_preview,
            fallback_preview=fallback_preview,
            safe_to_execute_now=not approval_required and not pending and not recovery_closure["blocks_auto_execution"],
            risk_signals=risks,
            risk_signal_count=len(risks),
            approval_required=approval_required,
            pending_approvals=len(pending),
            pending_approval_review_required=bool(pending),
            cycle_stages=len(HARNESS_CYCLE_STAGES),
            evidence_tools=evidence_tools,
            evidence_tool_count=len(evidence_tools),
            draft_only=True,
            requires_manual_send=True,
            loads_without_execution=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
            **recovery_metadata,
        )

        return ToolResult(
            "harness_cycle_preview",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                request_chars=len(request),
                likely_route=likely_route,
                primary_preview=primary_preview,
                fallback_preview=fallback_preview,
                safe_to_execute_now=not approval_required and not pending and not recovery_closure["blocks_auto_execution"],
                risk_signals=risks,
                risk_signal_count=len(risks),
                approval_required=approval_required,
                pending_approvals=len(pending),
                pending_approval_review_required=bool(pending),
                cycle_stages=len(HARNESS_CYCLE_STAGES),
                evidence_tools=evidence_tools,
                evidence_tool_count=len(evidence_tools),
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                **recovery_metadata,
                harness_cycle_handoff=harness_cycle_handoff,
            ),
        )

    def harness_lifecycle_state(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"), limit=500)
        if not request:
            output = (
                "Tell Jarvis the order to map. "
                "Try `harness lifecycle: inspect my screen and summarize what changed`. "
                f"{HARNESS_REQUEST_RECOVERY_ACTION}"
            )
            harness_lifecycle_handoff = _safe_metadata(
                source="harness_lifecycle_state",
                reason="missing_request",
                found=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
            )
            return ToolResult(
                "harness_lifecycle_state",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(
                        source="harness_lifecycle_state",
                        reason="missing_request",
                        found=False,
                        draft_only=True,
                        requires_manual_send=True,
                        loads_without_execution=True,
                        authorizes_execution=False,
                        authorizes_completion_claim=False,
                        approval_granted=False,
                        harness_lifecycle_handoff=harness_lifecycle_handoff,
                    ),
                    output=output,
                    action=HARNESS_REQUEST_RECOVERY_ACTION,
                ),
            )

        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=10)
        recovery_closure = _execution_health_recovery_closure(store)
        risks = _risk_signals(request)
        risky = bool(risks)
        route = "approval-gated" if risky else "low-risk routing candidate"
        stage_tools = {
            "perceive": ["chat_loop_preview", "voice_transcript_review", "screen_verification_contract"],
            "ground": ["chat_context", "memory_tree_summary", "return_brief", "priority_goal"],
            "route": ["command_diagnosis", "specialist_router_contract", "specialist_orchestration_packet", "specialist_route_quality", "specialist_execution_readiness", "specialist_handoff_receipt", "specialist_model_draft", "specialist_action_proposal_contract", "specialist_tool_dry_run_packet", "specialist_proposal_completion_gate", "specialist_execution_handoff_packet", "specialist_post_run_closure_packet", "specialist_cycle_ledger", "risk_preflight"],
            "plan": ["execution_governor_packet", "dispatch_decision_packet", "execution_contract", "execution_readiness_matrix", "action_rehearsal", "autonomy_plan"],
            "gate": ["action_readiness_packet", "approval_execution_packet", "safety_status"],
            "act": ["recent_tool_runs", "checkpoint_recovery_execute"],
            "verify": ["verification_packet", "verification_receipt", "screen_verification_contract"],
            "learn": ["after_action_learning_packet", "learning_review", "failure_to_test_preview", "failure_promotion_packet", "failure_implementation_packet", "failure_apply_contract", "failure_patch_receipt_packet", "failure_patch_application_bridge", "failure_patch_completion_gate", "failure_patch_handoff_packet", "failure_patch_closeout_packet", "failure_learning_record_packet", "failure_learning_closure_ledger"],
        }
        stage_states: list[dict[str, Any]] = []
        for name, description in HARNESS_CYCLE_STAGES:
            evidence = [tool_name for tool_name in stage_tools.get(name, []) if tool_name in tool_names]
            if name == "act":
                status = "held for approval" if risky else "ready only after route chooses a safe tool"
            elif name == "gate" and pending:
                status = "blocked by pending approval review"
            elif name in {"verify", "learn"}:
                status = "prepared after action evidence exists"
            elif evidence:
                status = "ready"
            else:
                status = "needs implementation"
            stage_states.append(
                {
                    "name": name,
                    "description": description,
                    "status": status,
                    "evidence": evidence,
                }
            )

        lines = [
            "Jarvis harness lifecycle state:",
            f"Order: {request}",
            "",
            "Route snapshot:",
            f"- likely route: {route}",
            f"- risk signals: {', '.join(risks) if risks else 'none obvious from wording'}",
            f"- pending approvals visible: {len(pending)}",
            f"- recovery closure blocks auto-run: {'yes' if recovery_closure['blocks_auto_execution'] else 'no'}",
            "- safety rule: read-only/local-safe tools may run automatically; risky actions stop for last-look approval.",
            "",
            "Lifecycle board:",
        ]
        for index, stage in enumerate(stage_states, start=1):
            lines.extend(
                [
                    f"{index}. {stage['name']} - {stage['status']}",
                    f"   purpose: {stage['description']}",
                    f"   evidence tools: {', '.join(stage['evidence']) if stage['evidence'] else 'none yet'}",
                ]
            )

        lines.extend(
            [
                "",
                "Immediate safe next commands:",
                f"- `command diagnosis: {request}`",
                f"- `specialist router contract: {request}`",
                f"- `execution contract: {request}`",
                f"- `execution readiness matrix: {request}`",
                f"- `verification packet: {request}`",
            ]
        )
        if pending:
            first = pending[0]
            lines.append(f"- `approval readiness {first['id']}` to check queue position and staleness")
            lines.append(f"- `approval packet {first['id']}` only after readiness points to the last-look review")

        lines.extend(
            [
                "",
                "Boundary:",
                "- This lifecycle state is read-only. It does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        lines.extend(_execution_health_recovery_lines(recovery_closure))
        recovery_metadata = _execution_health_recovery_metadata(recovery_closure)
        lifecycle_metadata = {
            "request": request,
            "request_chars": len(request),
            "likely_route": route,
            "safe_to_execute_now": not risky and not pending and not recovery_closure["blocks_auto_execution"],
            "approval_required": risky,
            "risk_signals": risks,
            "risk_signal_count": len(risks),
            "pending_approvals": len(pending),
            "pending_approval_review_required": bool(pending),
            "lifecycle_stages": len(stage_states),
            "ready_stages": len([stage for stage in stage_states if stage["status"] == "ready"]),
            "held_stages": len([stage for stage in stage_states if "held" in stage["status"] or "blocked" in stage["status"]]),
            "stage_states": stage_states,
            "stage_state_count": len(stage_states),
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            **recovery_metadata,
        }
        harness_lifecycle_handoff = _safe_metadata(
            source="harness_lifecycle_state",
            **lifecycle_metadata,
        )

        return ToolResult(
            "harness_lifecycle_state",
            True,
            "\n".join(lines),
            _safe_metadata(
                **lifecycle_metadata,
                harness_lifecycle_handoff=harness_lifecycle_handoff,
            ),
        )

    def harness_control_surface(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("goal"), limit=500)
        if not request:
            output = (
                "Tell Jarvis the order to inspect. "
                "Try `harness control: organize my downloads and summarize what changed`. "
                f"{HARNESS_REQUEST_RECOVERY_ACTION}"
            )
            harness_control_handoff = _safe_metadata(
                source="harness_control_surface",
                reason="missing_request",
                found=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
            )
            return ToolResult(
                "harness_control_surface",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(
                        source="harness_control_surface",
                        reason="missing_request",
                        found=False,
                        draft_only=True,
                        requires_manual_send=True,
                        loads_without_execution=True,
                        authorizes_execution=False,
                        authorizes_completion_claim=False,
                        approval_granted=False,
                        harness_control_handoff=harness_control_handoff,
                    ),
                    output=output,
                    action=HARNESS_REQUEST_RECOVERY_ACTION,
                ),
            )

        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=10)
        recent_runs = store.recent_tool_runs(limit=20)
        readable_recent_runs, unreadable_recent_run_rows = _readable_tool_run_rows(recent_runs)
        recovery_closure = _execution_health_recovery_closure_snapshot(store, readable_recent_runs)
        recent_failures, recent_approval_held = _recent_tool_run_attention_buckets(readable_recent_runs)
        first_pending_approval_id = _first_positive_int_text(pending)
        first_failed_run_id = _first_positive_int_text(recent_failures)
        approval_hold_command = _approval_review_command(recent_approval_held)
        risks = _risk_signals(request)
        risky = bool(risks)

        control_groups = [
            {
                "name": "engine",
                "purpose": "Choose the reasoning route without pretending the model already acted.",
                "evidence": {"chat_prompt_preview", "model_planner_prompt_preview", "specialist_router_contract", "specialist_orchestration_packet", "specialist_route_quality", "specialist_execution_readiness", "specialist_action_proposal_contract", "specialist_tool_dry_run_packet", "specialist_proposal_completion_gate", "specialist_execution_handoff_packet", "specialist_post_run_closure_packet", "specialist_cycle_ledger", "brain_think"},
                "ready_if": "brain route can be previewed",
            },
            {
                "name": "steering",
                "purpose": "Convert the natural order into route, exact arguments, and planner-gap checks.",
                "evidence": {"command_intake_packet", "execution_governor_packet", "dispatch_decision_packet", "planner_gap_packet", "execution_contract", "argument_contract_packet"},
                "ready_if": "command can be routed before execution",
            },
            {
                "name": "pedals",
                "purpose": "Expose the tool actuators, but only after route and permission checks.",
                "evidence": {"tool_search", "tool_detail", "risk_matrix", "execution_mission_control", "computer_task_plan", "integration_action_preview"},
                "ready_if": "candidate tools and risks are visible",
            },
            {
                "name": "brakes",
                "purpose": "Stop risky work at approval, ambiguity, failure, or stale state.",
                "evidence": {"safety_status", "approval_readiness_packet", "approval_execution_packet", "review_pending_approvals", "execution_audit_gate", "execution_recovery_packet", "execution_health_report", "recovery_closure_checklist", "after_action_learning_packet"},
                "ready_if": "approval and recovery gates are visible",
            },
            {
                "name": "dashboard",
                "purpose": "Show current state, diagnostics, and proof before claiming completion.",
                "evidence": {"readiness_report", "harness_operations_brief", "execution_governor_packet", "execution_readiness_matrix", "verification_packet", "runtime_trace_receipt", "evidence_ledger"},
                "ready_if": "proof and diagnostics are visible",
            },
        ]

        group_states: list[dict[str, Any]] = []
        ready_groups = 0
        partial_groups = 0
        missing_groups = 0
        for group in control_groups:
            present = sorted(group["evidence"] & tool_names)
            if len(present) >= 2:
                status = "ready"
                ready_groups += 1
            elif present:
                status = "partial"
                partial_groups += 1
            else:
                status = "missing"
                missing_groups += 1
            group_states.append(
                {
                    "name": group["name"],
                    "status": status,
                    "purpose": group["purpose"],
                    "ready_if": group["ready_if"],
                    "evidence": present,
                }
            )

        if pending and risky:
            verdict = "HOLD_FOR_APPROVAL_REVIEW"
            next_command = f"approval readiness {first_pending_approval_id}" if first_pending_approval_id else "pending approvals"
            can_auto_run = False
        elif unreadable_recent_run_rows:
            verdict = "AUDIT_REVIEW_REQUIRED"
            next_command = "recent tool runs"
            can_auto_run = False
        elif recovery_closure["blocks_auto_execution"]:
            verdict = "RECOVERY_CLOSURE_REQUIRED"
            next_command = recovery_closure["required_commands"][0] if recovery_closure["required_commands"] else "execution health report"
            can_auto_run = False
        elif recent_failures:
            verdict = "RECOVERY_REVIEW_FIRST"
            next_command = f"execution recovery packet {first_failed_run_id}" if first_failed_run_id else "execution health report"
            can_auto_run = False
        elif recent_approval_held:
            verdict = "APPROVAL_REVIEW_REQUIRED"
            next_command = approval_hold_command
            can_auto_run = False
        elif risky:
            verdict = "APPROVAL_GATED_ROUTE_REQUIRED"
            next_command = f"execution governor: {request}"
            can_auto_run = False
        else:
            verdict = "READY_FOR_EXECUTION_GOVERNOR"
            next_command = f"execution governor: {request}"
            can_auto_run = True

        lines = [
            "Jarvis harness control surface:",
            "This is the car dashboard for an agent harness: engine, steering, pedals, brakes, and proof before Jarvis acts.",
            "",
            f"Order: {request}",
            f"Verdict: {verdict}",
            f"Next command: `{next_command}`",
            f"Can auto-run now: {'yes, if command intake keeps it read-only/local-safe' if can_auto_run else 'no'}",
            "",
            "Control surface:",
        ]
        for group in group_states:
            lines.extend(
                [
                    f"- {group['name']}: {group['status']}",
                    f"  purpose: {group['purpose']}",
                    f"  ready if: {group['ready_if']}",
                    f"  evidence: {', '.join(group['evidence']) if group['evidence'] else 'none'}",
                ]
            )

        lines.extend(
            [
                "",
                "Runtime blockers:",
                f"- risk signals: {', '.join(risks) if risks else 'none obvious from wording'}",
                f"- pending approvals: {len(pending)}",
                f"- unreadable recent tool run rows: {unreadable_recent_run_rows}",
                f"- recent failed/blocked runs: {len(recent_failures)}",
                f"- recent approval-held runs: {len(recent_approval_held)}",
                f"- recovery closure: {recovery_closure['state']}",
                "",
                "Required follow-up packets:",
                f"- `execution governor: {request}`",
                f"- `command intake: {request}`",
                f"- `dispatch decision: {request}`",
                f"- `execution readiness matrix: {request}`",
                f"- `verification packet: {request}`",
                "- `execution audit gate` before claiming completion",
                "",
                "Boundary:",
                "- This control surface is read-only. It does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        lines.extend(_execution_health_recovery_lines(recovery_closure))
        recovery_metadata = _execution_health_recovery_metadata(recovery_closure)
        control_metadata = {
            "request": request,
            "request_chars": len(request),
            "verdict": verdict,
            "next_command": next_command,
            "can_auto_run": can_auto_run,
            "risk_signals": risks,
            "risk_signal_count": len(risks),
            "approval_required": risky,
            "approval_review_required": bool((pending and risky) or recent_approval_held),
            "pending_approvals": len(pending),
            "readable_recent_tool_runs": len(readable_recent_runs),
            "unreadable_recent_tool_run_rows": unreadable_recent_run_rows,
            "recent_failed_runs": len(recent_failures),
            "recent_approval_held_runs": len(recent_approval_held),
            "control_groups": len(group_states),
            "ready_groups": ready_groups,
            "partial_groups": partial_groups,
            "missing_groups": missing_groups,
            "group_states": group_states,
            "group_state_count": len(group_states),
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            **recovery_metadata,
        }
        harness_control_handoff = _safe_metadata(
            source="harness_control_surface",
            **control_metadata,
        )

        return ToolResult(
            "harness_control_surface",
            True,
            "\n".join(lines),
            _safe_metadata(
                **control_metadata,
                harness_control_handoff=harness_control_handoff,
            ),
        )

    def harness_operations_brief(args: dict[str, Any]) -> ToolResult:
        objective = _short(
            args.get("objective") or args.get("goal") or args.get("request") or (
                "Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant."
            ),
            limit=500,
        )
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=10)
        open_tasks = store.list_tasks(status="open", limit=10)
        active_goals = store.list_goals(status="active", limit=10)
        recent_runs = store.recent_tool_runs(limit=20)
        readable_recent_runs, unreadable_recent_run_rows = _readable_tool_run_rows(recent_runs)
        recovery_closure = _execution_health_recovery_closure_snapshot(store, readable_recent_runs)
        recent_failures, recent_approval_held = _recent_tool_run_attention_buckets(readable_recent_runs)
        first_pending_approval_id = _first_positive_int_text(pending)
        first_failed_run_id = _first_positive_int_text(recent_failures)
        approval_hold_command = _approval_review_command(recent_approval_held)

        if pending:
            next_move = "review_pending_approval"
            next_command = f"approval readiness {first_pending_approval_id}" if first_pending_approval_id else "pending approvals"
            stop_condition = "Do not approve or rerun the queued action unless approval readiness and the exact last-look packet are explicitly trusted by the operator."
            safe_to_continue = False
        elif unreadable_recent_run_rows:
            next_move = "review_unreadable_recent_runs"
            next_command = "recent tool runs"
            stop_condition = "Stop until unreadable audit rows are reviewed; do not continue from a possibly corrupted execution history."
            safe_to_continue = False
        elif recent_failures:
            next_move = "recover_failed_run"
            next_command = f"execution recovery packet {first_failed_run_id}" if first_failed_run_id else "execution health report"
            stop_condition = "Stop if the failed run involved shell/code, computer control, personal data, files, destructive work, or outside-world effects."
            safe_to_continue = True
        elif recent_approval_held:
            next_move = "review_approval_held_run"
            next_command = approval_hold_command
            stop_condition = "Review the queued approval packet before treating the held action as failed or rerunning it."
            safe_to_continue = False
        elif open_tasks:
            next_move = "work_next_open_task"
            next_command = "build target packet"
            stop_condition = "Stop before risky execution and produce exact route, approval, audit, verification, and recovery evidence first."
            safe_to_continue = True
        else:
            next_move = "strengthen_harness_evidence"
            next_command = "completion audit"
            stop_condition = "Stop before claiming completion unless the evidence ledger and completion claim gate are clear."
            safe_to_continue = True

        required_tools = [
            "priority_goal",
            "harness_status",
            "harness_cycle_preview",
            "harness_lifecycle_state",
            "harness_control_surface",
            "execution_governor_packet",
            "dispatch_decision_packet",
            "execution_readiness_matrix",
            "approval_execution_packet",
            "verification_packet",
            "execution_recovery_packet",
            "execution_health_report",
            "integration_execution_matrix",
            "integration_adapter_manifest",
            "integration_adapter_probe",
            "integration_adapter_acceptance",
            "legacy_connector_migration_audit",
            "integration_dry_run_contract",
            "integration_proof_bundle",
            "integration_implementation_review",
            "evidence_ledger",
            "completion_claim_gate",
        ]
        available_tools = [name for name in required_tools if name in tool_names]
        missing_tools = [name for name in required_tools if name not in tool_names]
        implementation_review_commands = [
            "integration execution matrix: email",
            "integration adapter manifest: email",
            "integration adapter acceptance: email",
            "legacy connector migration audit: browser calendar email",
            "integration dry run contract: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only",
            "integration proof bundle: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed",
            "integration implementation review: email -> search mailbox metadata; target selected source; time selected window; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed",
        ]
        recommended_control_commands = [next_command, *implementation_review_commands, "execution health report", "evidence ledger", "completion claim gate"]
        auto_execution_allowed = safe_to_continue and not recovery_closure["blocks_auto_execution"]
        auto_continue_status = (
            "yes"
            if auto_execution_allowed
            else "no, review approval first"
            if not safe_to_continue
            else "no, close recovery proof first"
        )

        lines = [
            "Jarvis harness operations brief:",
            "This is the operator-layer packet for deciding the next safe build move without executing it.",
            "",
            f"Objective: {objective}",
            "",
            "Current command state:",
            f"- pending approvals: {len(pending)}",
            f"- open tasks: {len(open_tasks)}",
            f"- active goals: {len(active_goals)}",
            f"- unreadable recent tool run rows: {unreadable_recent_run_rows}",
            f"- recent failed/blocked runs: {len(recent_failures)}",
            f"- recent approval-held runs: {len(recent_approval_held)}",
            "",
            "Next safe move:",
            f"- move: {next_move}",
            f"- command: `{next_command}`",
            f"- can continue automatically: {auto_continue_status}",
            f"- stop condition: {stop_condition}",
            "",
            "Harness controls available:",
            f"- present: {', '.join(available_tools) if available_tools else 'none'}",
            f"- missing: {', '.join(missing_tools) if missing_tools else 'none'}",
            "",
            "Implementation review handoff:",
            *[f"- `{command}`" for command in implementation_review_commands],
            "",
            "Execution boundary:",
            "- Read-only/local-safe build work may proceed automatically.",
            "- Shell/code, computer control, personal data, external effects, destructive actions, and risky work remain approval-gated.",
            "- Every real action should leave route, argument, approval, audit, verification, recovery, and learning evidence.",
            "- Use `execution health report` to summarize failed runs, repeated failures, recovery coverage, and the next safe audit command.",
            "- Use `integration execution matrix` before enabling personal connectors so disabled adapters, proof gates, and approval lanes stay visible.",
            "- Use `integration adapter manifest` before implementing connector adapters so fixtures, blocked-path tests, audit fields, and enablement gates are explicit.",
            "- Use `integration adapter probe` to prove metadata-only fake rows work and full-content/side-effect paths stay blocked before real adapters exist.",
            "- Use `integration adapter acceptance` to run the happy-path and blocked-path adapter proof cases before any connector enablement gate.",
            "- Use `legacy connector migration audit` to review browser, calendar, and email legacy surfaces as read-only migration evidence before connector implementation review.",
            "- Use `integration proof bundle` to tie metadata preview, adapter acceptance, metadata row contract, enablement, rehearsal, audit, verification, rollback, and stop proof into one implementation-review packet.",
            "- Use `integration implementation review` to require proof bundle, metadata row contract, preflight row-contract proof, implementation spec, status/API evidence, smoke, audit, verification, rollback, and stop proof before connector code review.",
            "",
            "Boundary:",
            "- This operations brief is read-only. It does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
        ]
        lines.extend(_execution_health_recovery_lines(recovery_closure))
        recovery_metadata = _execution_health_recovery_metadata(recovery_closure)
        operations_metadata = {
            "objective": objective,
            "objective_chars": len(objective),
            "next_move": next_move,
            "next_command": next_command,
            "safe_to_continue": safe_to_continue,
            "safe_to_execute_now": auto_execution_allowed,
            "can_auto_run": auto_execution_allowed,
            "auto_continue_status": auto_continue_status,
            "stop_condition": stop_condition,
            "pending_approvals": len(pending),
            "open_tasks": len(open_tasks),
            "active_goals": len(active_goals),
            "readable_recent_tool_runs": len(readable_recent_runs),
            "unreadable_recent_tool_run_rows": unreadable_recent_run_rows,
            "recent_failed_runs": len(recent_failures),
            "recent_approval_held_runs": len(recent_approval_held),
            "required_controls": len(required_tools),
            "available_controls": len(available_tools),
            "missing_controls": len(missing_tools),
            "available_control_tools": available_tools,
            "available_control_tool_count": len(available_tools),
            "missing_control_tools": missing_tools,
            "missing_control_tool_count": len(missing_tools),
            "recommended_control_commands": recommended_control_commands,
            "recommended_control_command_count": len(recommended_control_commands),
            "implementation_review_commands": implementation_review_commands,
            "implementation_review_command_count": len(implementation_review_commands),
            "approval_review_required": bool(pending or recent_approval_held),
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            **recovery_metadata,
        }
        harness_operations_handoff = _safe_metadata(
            source="harness_operations_brief",
            **operations_metadata,
        )

        return ToolResult(
            "harness_operations_brief",
            True,
            "\n".join(lines),
            _safe_metadata(
                **operations_metadata,
                harness_operations_handoff=harness_operations_handoff,
            ),
        )

    def agi_gate_report(_: dict[str, Any]) -> ToolResult:
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=10)
        active_goals = store.list_goals(status="active", limit=10)
        open_tasks = store.list_tasks(status="open", limit=10)

        gate_summary = _agi_gate_summary(tool_names)
        selected_agi_readiness = _selected_agi_target_readiness(gate_summary)
        selected_agi_gate_name = selected_agi_readiness["gate_name"]
        selected_agi_target_title = str(selected_agi_readiness["target_title"] or "")
        selected_agi_closure_commands = list(selected_agi_readiness["closure_commands"] or [])
        selected_agi_verification_commands = list(selected_agi_readiness["verification_commands"] or [])
        selected_agi_likely_files = list(selected_agi_readiness["likely_files"] or [])
        selected_agi_acceptance_checks = list(selected_agi_readiness["acceptance_checks"] or [])
        selected_agi_acceptance_gap_preview = selected_agi_acceptance_checks[:3]
        selected_agi_first_acceptance_gap = selected_agi_acceptance_gap_preview[0] if selected_agi_acceptance_gap_preview else ""
        selected_agi_file_integrity = selected_agi_readiness["file_integrity"]
        selected_agi_build_ready = bool(selected_agi_readiness["build_ready"])
        lines = [
            "Jarvis AGI-direction gate report:",
            "",
            "Purpose:",
            "- This does not claim Jarvis is AGI. It measures whether the harness is becoming capable enough to host stronger AI brains safely.",
            "- Each gate must have interface, state, routing, safety, audit, verification, and recovery before real autonomy increases.",
            "",
            "Gate status:",
        ]
        for gate in gate_summary["rows"]:
            lines.extend(
                [
                    f"- {gate['gate']}: {gate['status']}",
                    f"  Evidence: {', '.join(gate['present']) if gate['present'] else 'none yet'}",
                    f"  Evidence still missing: {', '.join(gate['missing_evidence']) if gate['missing_evidence'] else 'none from registry'}",
                    f"  Missing: {gate['real_execution_gap']}",
                    f"  Next safe build move: {gate['next']}",
                    f"  Evidence closure commands: {', '.join(f'`{command}`' for command in gate['evidence_closure_commands'])}",
                    f"  Focused verification: {', '.join(f'`{command}`' for command in gate['focused_verification']) if gate['focused_verification'] else 'none configured'}",
                ]
            )

        lines.extend(
            [
                "",
                "Selected next build target:",
                f"- gate: {selected_agi_gate_name or 'none'}",
                f"- target: {selected_agi_target_title or 'none'}",
                f"- next build command: `{selected_agi_closure_commands[0]}`" if selected_agi_closure_commands else "- next build command: none",
                f"- target file integrity: {selected_agi_file_integrity['status']}",
                f"- likely files: {', '.join(selected_agi_likely_files) if selected_agi_likely_files else 'none'}",
                f"- acceptance checks: {len(selected_agi_acceptance_checks)}",
                f"- focused verification commands: {len(selected_agi_verification_commands)}",
                f"- ready for scoped implementation review: {'yes' if selected_agi_build_ready else 'no'}",
            ]
        )

        lines.extend(
            [
                "",
                "Current operating constraints:",
                f"- pending approvals: {len(pending)}",
                f"- active goals: {len(active_goals)}",
                f"- open tasks: {len(open_tasks)}",
                "- AGI-direction work must stay preview-first until every risky action has exact arguments, approval, audit, verification, and recovery.",
                "",
                "Best next commands:",
                "- `harness cycle: <order>`",
                "- `action readiness: <order>`",
                "- `roadmap`",
                "- `learning review`",
            ]
        )
        gate_metadata = {
            "gates": len(AGI_GATE_EVIDENCE),
            "gate_rows": gate_summary["rows"],
            "gate_row_count": len(gate_summary["rows"]),
            "strong_gates": gate_summary["strong"],
            "partial_gates": gate_summary["partial"],
            "missing_gates": gate_summary["missing"],
            "pending_approvals": len(pending),
            "active_goals": len(active_goals),
            "open_tasks": len(open_tasks),
            "gate_statuses": gate_summary["gate_statuses"],
            "present_evidence_by_gate": gate_summary["present_evidence_by_gate"],
            "missing_evidence_by_gate": gate_summary["missing_evidence_by_gate"],
            "next_moves_by_gate": gate_summary["next_moves_by_gate"],
            "evidence_closure_commands_by_gate": gate_summary["evidence_closure_commands_by_gate"],
            "focused_verification_by_gate": gate_summary["focused_verification_by_gate"],
            "real_execution_gaps_by_gate": gate_summary["real_execution_gaps_by_gate"],
            "real_execution_gap_count": gate_summary["real_execution_gap_count"],
            "agi_next_gate": selected_agi_gate_name,
            "agi_next_target_title": selected_agi_target_title,
            "agi_next_build_command": selected_agi_closure_commands[0] if selected_agi_closure_commands else "",
            "agi_next_evidence_closure_commands": selected_agi_closure_commands,
            "agi_next_evidence_closure_command_count": len(selected_agi_closure_commands),
            "agi_next_focused_verification_commands": selected_agi_verification_commands,
            "agi_next_focused_verification_command_count": len(selected_agi_verification_commands),
            "agi_next_likely_files": selected_agi_likely_files,
            "agi_next_likely_file_count": len(selected_agi_likely_files),
            "agi_next_target_file_integrity_status": selected_agi_file_integrity["status"],
            "agi_next_target_files_checked": selected_agi_file_integrity["checked"],
            "agi_next_target_files_exist": selected_agi_file_integrity["all_exist"],
            "agi_next_missing_target_files": selected_agi_file_integrity["missing"],
            "agi_next_missing_target_file_count": selected_agi_file_integrity["missing_count"],
            "agi_next_target_integrity_blocks_start": not selected_agi_file_integrity["all_exist"],
            "agi_next_acceptance_checks": selected_agi_acceptance_checks,
            "agi_next_acceptance_check_count": len(selected_agi_acceptance_checks),
            "agi_next_acceptance_preview": selected_agi_acceptance_gap_preview,
            "agi_next_acceptance_preview_count": len(selected_agi_acceptance_gap_preview),
            "agi_next_first_acceptance_check": selected_agi_first_acceptance_gap,
            "agi_next_acceptance_gap_preview": selected_agi_acceptance_gap_preview,
            "agi_next_acceptance_gap_preview_count": len(selected_agi_acceptance_gap_preview),
            "agi_next_first_acceptance_gap": selected_agi_first_acceptance_gap,
            "agi_next_build_packet_ready_for_review": selected_agi_build_ready,
            **_agi_focus_selection_metadata(selected_agi_readiness),
            "best_next_commands": [
                "harness cycle: <order>",
                "action readiness: <order>",
                "roadmap",
                "learning review",
            ],
            "best_next_command_count": 4,
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        }
        agi_gate_handoff = _safe_metadata(
            source="agi_gate_report",
            **gate_metadata,
        )

        return ToolResult(
            "agi_gate_report",
            True,
            "\n".join(lines),
            _safe_metadata(
                **gate_metadata,
                agi_gate_handoff=agi_gate_handoff,
            ),
        )

    def agi_next_build_move(args: dict[str, Any]) -> ToolResult:
        requested_gate = _short(args.get("gate") or args.get("focus") or args.get("target"), limit=120).lower()
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=10)
        recent_runs = store.recent_tool_runs(limit=30)
        failed_runs, approval_held_runs = _recent_tool_run_attention_buckets(recent_runs)
        approval_held_review_command = _approval_review_command(approval_held_runs) if approval_held_runs else ""
        recovery_closure = _execution_health_recovery_closure_snapshot(store, recent_runs)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        gate_summary = _agi_gate_summary(tool_names)

        selection = _select_agi_next_gate_with_context(gate_summary, requested_gate)
        selected = selection["gate"]

        if not selected:
            output = (
                "No AGI-direction gates are configured. "
                f"{HARNESS_GATE_CONFIGURATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "agi_next_build_move",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(reason="no_gates", selected_gate=None),
                    output=output,
                    action=HARNESS_GATE_CONFIGURATION_RECOVERY_ACTION,
                ),
            )

        target = AGI_GATE_BUILD_TARGETS.get(selected["gate"], {})
        target_title = _safe_agi_target_text(_agi_target_value(target, "title", "") or selected["next"], limit=500)
        raw_likely_files = _agi_target_list_value(target, "files")
        likely_files = _safe_agi_target_list(raw_likely_files, limit=260)
        focused_tests = _safe_agi_target_list(_agi_target_list_value(target, "tests"), limit=500)
        acceptance_checks = _safe_agi_target_list(_agi_target_list_value(target, "acceptance"), limit=500)
        acceptance_gap_preview = acceptance_checks[:3]
        first_acceptance_gap = acceptance_gap_preview[0] if acceptance_gap_preview else ""
        file_integrity = _target_file_integrity(raw_likely_files)
        build_packet_ready = bool(file_integrity["all_exist"] and focused_tests and acceptance_checks)
        next_command = f"agi next build move: {selected['gate']}"
        implementation_preflight = _agi_implementation_preflight_metadata(
            build_packet_ready=build_packet_ready,
            pending_approvals=pending,
            recovery_closure=recovery_closure,
            learning_debt=learning_debt,
            build_command=next_command,
        )
        implementation_preflight_blockers = list(implementation_preflight["agi_next_implementation_preflight_blockers"])
        implementation_preflight_ready = bool(implementation_preflight["agi_next_implementation_preflight_ready"])
        implementation_preflight_next_command = str(implementation_preflight["agi_next_implementation_preflight_next_command"] or "")
        completion_audit_command = f"completion audit: improve AGI gate {selected['gate']}"
        evidence_ledger_command = "evidence ledger"
        completion_claim_gate_command = f"completion claim gate: improve AGI gate {selected['gate']}"
        proof_handoff_commands = [
            completion_audit_command,
            evidence_ledger_command,
            completion_claim_gate_command,
        ]
        evidence_closure_commands = [next_command, *proof_handoff_commands]

        lines = [
            "Jarvis AGI next build move:",
            "This is a read-only build-target packet. It selects one real-execution gate gap without editing files, running tests, approving requests, reading personal data, controlling the computer, or queuing approvals.",
            "",
            f"Selected gate: {selected['gate']}",
            f"Gate status: {selected['status']}",
            f"Build target: {target_title}",
            "",
            "Why this gate still matters:",
            f"- real-execution gap: {selected['real_execution_gap']}",
            f"- registry evidence still missing: {', '.join(selected['missing_evidence']) if selected['missing_evidence'] else 'none from registry'}",
            f"- present evidence: {', '.join(selected['present']) if selected['present'] else 'none'}",
            "",
            "Owning files to inspect first:",
        ]
        lines.extend(f"- `{path}`" for path in likely_files)
        lines.extend(
            [
                "",
                "Target integrity:",
                f"- file check: {file_integrity['status']}",
                f"- files checked: {file_integrity['checked']}",
                f"- missing files: {', '.join(file_integrity['missing']) if file_integrity['missing'] else 'none'}",
                "",
                "Build packet readiness:",
                f"- owning files ready: {'yes' if file_integrity['all_exist'] else 'no'}",
                f"- focused verification commands: {len(focused_tests)}",
                f"- acceptance checks: {len(acceptance_checks)}",
                f"- ready for scoped implementation review: {'yes' if build_packet_ready else 'no'}",
                f"- implementation preflight ready: {'yes' if implementation_preflight_ready else 'no'}",
                f"- implementation preflight blockers: {', '.join(implementation_preflight_blockers) if implementation_preflight_blockers else 'none'}",
                f"- next preflight command: `{implementation_preflight_next_command}`",
            ]
        )
        lines.extend(["", "Focused verification:"])
        lines.extend(f"- `{command}`" for command in focused_tests)
        lines.extend(["", "Acceptance checks:"])
        lines.extend(f"- {check}" for check in acceptance_checks)
        lines.extend(
            [
                "",
                "Safety and recovery gates:",
                f"- pending approvals visible: {len(pending)}",
                f"- recent failed/blocked runs: {len(failed_runs)}",
                f"- recent approval-held runs: {len(approval_held_runs)}",
                f"- approval-held review: `{approval_held_review_command}`" if approval_held_review_command else "- approval-held review: none",
                f"- recovery closure state: {recovery_closure['state']}",
                f"- recovery closure missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- next recovery required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next recovery required: none",
                f"- learning debt state: {learning_debt['state']}",
                f"- learning debt missing: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
                f"- next actionable learning required: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- next actionable learning required: none",
                f"- ordered learning gate required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- ordered learning gate required: none",
                "- keep shell/code, computer control, personal data, external side effects, and destructive actions approval-gated",
                "- after any implementation, run the focused smoke test and `completion claim gate` before claiming the gate improved",
                "",
                "Recovery and learning proof handoff:",
                f"- recovery proof queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands'][:6]) if recovery_closure['required_commands'] else 'none'}",
                f"- learning actionable queue: {', '.join(f'`{command}`' for command in learning_debt['actionable_proof_queue'][:6]) if learning_debt['actionable_proof_queue'] else 'none'}",
                f"- learning proof queue: {', '.join(f'`{command}`' for command in learning_debt['required_commands'][:6]) if learning_debt['required_commands'] else 'none'}",
                "",
                "Completion proof handoff:",
                f"- audit: `{completion_audit_command}`",
                f"- evidence: `{evidence_ledger_command}`",
                f"- claim gate: `{completion_claim_gate_command}`",
                "",
                "Evidence closure plan:",
                f"- selected gate packet: `{next_command}`",
                *[f"- proof command: `{command}`" for command in proof_handoff_commands],
                f"- focused verification: {', '.join(f'`{command}`' for command in focused_tests) if focused_tests else 'none configured'}",
                "",
                "Next command:",
                f"- `{next_command}`",
                "",
                "Boundary:",
                "- This packet is planning only. It does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        build_metadata = {
            "selected_gate": selected["gate"],
            "gate_status": selected["status"],
            "target_title": target_title,
            "agi_next_gate": selected["gate"],
            "agi_next_target_title": target_title,
            "agi_next_build_command": next_command,
            "agi_next_likely_files": likely_files,
            "agi_next_likely_file_count": len(likely_files),
            "agi_next_target_file_integrity_status": file_integrity["status"],
            "agi_next_target_files_checked": file_integrity["checked"],
            "agi_next_target_files_exist": file_integrity["all_exist"],
            "agi_next_missing_target_files": file_integrity["missing"],
            "agi_next_missing_target_file_count": file_integrity["missing_count"],
            "agi_next_target_integrity_blocks_start": not file_integrity["all_exist"],
            "agi_next_focused_verification_commands": focused_tests,
            "agi_next_focused_verification_command_count": len(focused_tests),
            "agi_next_acceptance_checks": acceptance_checks,
            "agi_next_acceptance_check_count": len(acceptance_checks),
            "agi_next_acceptance_preview": acceptance_gap_preview,
            "agi_next_acceptance_preview_count": len(acceptance_gap_preview),
            "agi_next_first_acceptance_check": first_acceptance_gap,
            "agi_next_acceptance_gap_preview": acceptance_gap_preview,
            "agi_next_acceptance_gap_preview_count": len(acceptance_gap_preview),
            "agi_next_first_acceptance_gap": first_acceptance_gap,
            "agi_next_evidence_closure_commands": evidence_closure_commands,
            "agi_next_evidence_closure_command_count": len(evidence_closure_commands),
            "agi_next_build_packet_ready_for_review": build_packet_ready,
            **implementation_preflight,
            "present_evidence": selected["present"],
            "present_evidence_count": len(selected["present"]),
            "missing_evidence": selected["missing_evidence"],
            "missing_evidence_count": len(selected["missing_evidence"]),
            "real_execution_gap": selected["real_execution_gap"],
            "next_safe_build_move": selected["next"],
            "likely_files": likely_files,
            "likely_file_count": len(likely_files),
            "target_file_integrity_status": file_integrity["status"],
            "target_files_checked": file_integrity["checked"],
            "target_files_exist": file_integrity["all_exist"],
            "missing_target_files": file_integrity["missing"],
            "missing_target_file_count": file_integrity["missing_count"],
            "target_file_rows": file_integrity["rows"],
            "target_integrity_blocks_start": not file_integrity["all_exist"],
            "focused_tests": focused_tests,
            "focused_test_count": len(focused_tests),
            "acceptance_checks": acceptance_checks,
            "acceptance_check_count": len(acceptance_checks),
            "acceptance_preview": acceptance_gap_preview,
            "acceptance_preview_count": len(acceptance_gap_preview),
            "first_acceptance_check": first_acceptance_gap,
            "acceptance_gap_preview": acceptance_gap_preview,
            "acceptance_gap_preview_count": len(acceptance_gap_preview),
            "first_acceptance_gap": first_acceptance_gap,
            "build_packet_ready_for_review": build_packet_ready,
            "implementation_preflight_ready": implementation_preflight_ready,
            "implementation_preflight_blockers": implementation_preflight_blockers,
            "implementation_preflight_blocker_count": len(implementation_preflight_blockers),
            "implementation_preflight_next_command": implementation_preflight_next_command,
            "agi_focus_selection_source": selection["selection_source"],
            "agi_focus_selection_reason": selection["selection_reason"],
            "agi_focus_canonical_selector_command": f"agi next build move: {selected['gate']}",
            "agi_focus_deliberate_focus_override": selection["deliberate_focus_override"],
            "pending_approvals": len(pending),
            "recent_failed_runs": len(failed_runs),
            "recent_approval_held_runs": len(approval_held_runs),
            "approval_held_review_command": approval_held_review_command,
            "execution_health_recovery_closure_state": recovery_closure["state"],
            "execution_health_recovery_closure_ready_to_retry": recovery_closure["ready_to_retry"],
            "execution_health_recovery_closure_missing": recovery_closure["missing"],
            "execution_health_recovery_closure_missing_count": recovery_closure["missing_count"],
            "execution_health_recovery_closure_required_commands": recovery_closure["required_commands"],
            "execution_health_recovery_closure_next_required_command": recovery_closure["next_required_command"],
            "execution_health_recovery_closure_checklist_command": _recovery_closure_checklist_command(recovery_closure),
            "execution_health_recovery_closure_should_open_checklist": bool(_recovery_closure_checklist_command(recovery_closure)),
            "execution_health_recovery_closure_proof_queue": recovery_closure["proof_queue"],
            "execution_health_recovery_closure_proof_queue_count": recovery_closure["proof_queue_count"],
            "execution_health_recovery_closure_next_proof_command": recovery_closure["next_proof_command"],
            "execution_health_recovery_closure_blocks_completion_claim": recovery_closure["blocks_completion_claim"],
            "execution_health_recovery_closure_target_run_id": recovery_closure["target_run_id"],
            "execution_health_recovery_closure_target_tool_name": recovery_closure["target_tool_name"],
            "execution_health_recovery_closure_target_verification_receipts": recovery_closure["target_verification_receipts"],
            "execution_health_recovery_closure_target_recovery_packets": recovery_closure["target_recovery_packets"],
            "execution_health_recovery_closure_target_after_action_learning_packets": recovery_closure["target_after_action_learning_packets"],
            "execution_learning_state": learning_debt["state"],
            "execution_learning_blocks_completion_claim": learning_debt["blocks_completion_claim"],
            "execution_learning_recent_action_runs": learning_debt["recent_action_runs"],
            "execution_learning_failed_or_blocked_action_runs": learning_debt["failed_or_blocked_action_runs"],
            "execution_learning_recent_verification_runs": learning_debt["recent_verification_runs"],
            "execution_learning_recent_recovery_runs": learning_debt["recent_recovery_runs"],
            "execution_learning_recent_after_action_learning_runs": learning_debt["recent_after_action_learning_runs"],
            "execution_learning_target_run_id": learning_debt["target_run_id"],
            "execution_learning_target_tool_name": learning_debt["target_tool_name"],
            "execution_learning_target_after_action_learning_packets": learning_debt["target_after_action_learning_packets"],
            "execution_learning_missing": learning_debt["missing"],
            "execution_learning_missing_count": learning_debt["missing_count"],
            "execution_learning_required_commands": learning_debt["required_commands"],
            "execution_learning_next_required_command": learning_debt["next_required_command"],
            "execution_learning_proof_queue": learning_debt["proof_queue"],
            "execution_learning_proof_queue_count": learning_debt["proof_queue_count"],
            "execution_learning_next_proof_command": learning_debt["next_proof_command"],
            "execution_learning_evidence_command": learning_debt["learning_evidence_command"],
            "execution_learning_after_action_learning_command": learning_debt["after_action_learning_command"],
            "execution_learning_actionable_required_commands": learning_debt["actionable_required_commands"],
            "execution_learning_actionable_required_command_count": learning_debt["actionable_required_command_count"],
            "execution_learning_actionable_proof_queue": learning_debt["actionable_proof_queue"],
            "execution_learning_actionable_proof_queue_count": learning_debt["actionable_proof_queue_count"],
            "execution_learning_actionable_next_required_command": learning_debt["actionable_next_required_command"],
            "execution_learning_actionable_next_proof_command": learning_debt["actionable_next_proof_command"],
            "execution_learning_next_evidence_command": learning_debt["next_evidence_command"],
            "agi_gates": gate_summary["gates"],
            "agi_gate_statuses": gate_summary["gate_statuses"],
            "agi_real_execution_gap_count": gate_summary["real_execution_gap_count"],
            "real_execution_gap_count": gate_summary["real_execution_gap_count"],
            "selected_real_execution_gap": selected["real_execution_gap"],
            "selected_real_execution_gap_gate": selected["gate"],
            "selected_real_execution_gap_detail": selected["real_execution_gap"],
            "completion_audit_command": completion_audit_command,
            "evidence_ledger_command": evidence_ledger_command,
            "completion_claim_gate_command": completion_claim_gate_command,
            "proof_handoff_commands": proof_handoff_commands,
            "proof_handoff_command_count": len(proof_handoff_commands),
            "evidence_closure_commands": evidence_closure_commands,
            "evidence_closure_command_count": len(evidence_closure_commands),
            "focused_verification_commands": focused_tests,
            "focused_verification_command_count": len(focused_tests),
            "next_command": next_command,
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        }
        agi_next_build_handoff = _safe_metadata(
            source="agi_next_build_move",
            **build_metadata,
        )

        return ToolResult(
            "agi_next_build_move",
            True,
            "\n".join(lines),
            _safe_metadata(
                **build_metadata,
                agi_next_build_handoff=agi_next_build_handoff,
            ),
        )

    def harness_completion_assessment(_: dict[str, Any]) -> ToolResult:
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        toolset_counts = Counter(tool.toolset for tool in tools)
        pending = store.list_pending_approvals(limit=10)
        recent_runs = store.recent_tool_runs(limit=50)
        recovery_closure = _execution_health_recovery_closure_snapshot(store, recent_runs)
        execution_health_verification = _execution_health_verification_snapshot(recent_runs)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        open_tasks = store.list_tasks(status="open", limit=10)
        active_goals = store.list_goals(status="active", limit=10)
        agi_summary = _agi_gate_summary(tool_names)
        selected_agi_readiness = _selected_agi_target_readiness(agi_summary)
        selected_agi_gate = selected_agi_readiness["gate"]
        selected_agi_gate_name = selected_agi_readiness["gate_name"]
        selected_agi_closure_commands = selected_agi_readiness["closure_commands"]
        selected_agi_verification_commands = selected_agi_readiness["verification_commands"]
        selected_agi_target = selected_agi_readiness["target"]
        selected_agi_likely_files = selected_agi_readiness["likely_files"]
        selected_agi_acceptance_checks = selected_agi_readiness["acceptance_checks"]
        selected_agi_file_integrity = selected_agi_readiness["file_integrity"]
        selected_agi_build_ready = selected_agi_readiness["build_ready"]
        storage_readiness = _storage_readiness_snapshot()
        agi_real_execution_blocks_completion_claim = agi_summary["real_execution_gap_count"] > 0
        failed_runs, approval_held_runs = _recent_tool_run_attention_buckets(recent_runs)
        approval_held_review_command = _approval_review_command(approval_held_runs) if approval_held_runs else ""

        component_scores: list[tuple[str, float, str]] = []
        for component in HARNESS_COMPONENTS:
            present_toolsets = bool(component["toolsets"] & set(toolset_counts))
            present_evidence = sorted(component["evidence"] & tool_names)
            if len(present_evidence) >= 3 and present_toolsets:
                component_scores.append((component["name"], 0.75, "strong foundation"))
            elif present_evidence or present_toolsets:
                component_scores.append((component["name"], 0.35, "partial foundation"))
            else:
                component_scores.append((component["name"], 0.0, "missing"))

        gate_scores: list[tuple[str, float, str]] = []
        for gate in AGI_GATE_EVIDENCE:
            present = sorted(gate["evidence"] & tool_names)
            if len(present) >= 3:
                gate_scores.append((gate["gate"], 0.65, "strong prototype with real-execution gap"))
            elif present:
                gate_scores.append((gate["gate"], 0.30, "partial prototype"))
            else:
                gate_scores.append((gate["gate"], 0.0, "missing"))

        component_percent = round(sum(score for _, score, _ in component_scores) / len(component_scores) * 100)
        gate_percent = round(sum(score for _, score, _ in gate_scores) / len(gate_scores) * 100)
        execution_readiness_percent = round(
            (
                (0.25 if "approval_execution_packet" in tool_names else 0)
                + (0.25 if {"checkpoint_recovery_receipt", "checkpoint_recovery_followthrough_packet"} <= tool_names else 0)
                + (0.25 if {"recent_tool_runs", "runtime_trace_receipt"} <= tool_names else 0)
                + (0.25 if any(row["ok"] for row in recent_runs) else 0)
            )
            * 100
        )
        overall_percent = round(component_percent * 0.45 + gate_percent * 0.35 + execution_readiness_percent * 0.20)
        incomplete_components = [name for name, score, _ in component_scores if score < 0.75]
        completion_blockers: list[str] = []
        if incomplete_components:
            completion_blockers.append("component foundations still need direct evidence: " + ", ".join(incomplete_components))
        if agi_summary["real_execution_gap_count"]:
            completion_blockers.append(f"{agi_summary['real_execution_gap_count']} AGI-direction real-execution gate gap(s) remain")
        if storage_readiness["blocks_completion_claim"]:
            completion_blockers.append(str(storage_readiness["blocker"]))
        if pending:
            completion_blockers.append(f"{len(pending)} pending approval(s) must be reviewed or resolved")
        if open_tasks:
            completion_blockers.append(f"{len(open_tasks)} open task(s) remain")
        if failed_runs:
            completion_blockers.append(f"{len(failed_runs)} recent failed or blocked tool run(s) need recovery review")
        if approval_held_runs:
            completion_blockers.append(
                f"{len(approval_held_runs)} recent approval-held tool run(s) need approval review"
                + (f" via `{approval_held_review_command}`" if approval_held_review_command else "")
            )
        if recovery_closure["blocks_completion_claim"]:
            completion_blockers.append(
                "execution health recovery closure is incomplete: "
                + ", ".join(recovery_closure["missing"] or [str(recovery_closure["state"])])
            )
        if execution_health_verification["blocks_completion_claim"]:
            completion_blockers.append("execution health verification coverage is missing")
        if learning_debt["blocks_completion_claim"]:
            completion_blockers.append(
                "execution learning debt is incomplete: "
                + ", ".join(learning_debt["missing"] or [str(learning_debt["state"])])
            )
        learning_closure_command = (
            f"execution learning closure {learning_debt['target_run_id']}"
            if learning_debt["blocks_completion_claim"] and learning_debt["target_run_id"] is not None
            else "execution learning closure"
            if learning_debt["blocks_completion_claim"]
            else ""
        )
        learning_actionable_commands = list(learning_debt["actionable_required_commands"])
        next_proof_commands = _completion_proof_queue_from_sources(
            recovery_proof_queue=list(recovery_closure["proof_queue"]),
            execution_health_verification_proof_queue=list(execution_health_verification["proof_queue"]),
            learning_closure_command=learning_closure_command,
            learning_proof_queue=list(learning_debt["proof_queue"]),
            learning_actionable_proof_queue=learning_actionable_commands,
            pending_approval_ids=[pending[0]["id"]] if pending else [],
            open_task_ids=[open_tasks[0]["id"]] if open_tasks else [],
            baseline_commands=[*storage_readiness["next_commands"], "completion audit", "evidence ledger", "completion claim gate"],
            selected_agi_closure_commands=selected_agi_closure_commands,
        )
        if storage_readiness["next_commands"]:
            storage_first_queue = list(storage_readiness["next_commands"])
            _append_unique(storage_first_queue, next_proof_commands)
            next_proof_commands = storage_first_queue
        if approval_held_review_command:
            _append_unique(next_proof_commands, [approval_held_review_command])
        completion_proof_queue = list(next_proof_commands)
        completion_claim_ready = not completion_blockers

        lines = [
            "Jarvis harness completion assessment:",
            "This estimates the Jarvis V2 agent-harness prototype, not AGI itself.",
            "",
            f"Overall harness prototype: {overall_percent}%",
            f"- component foundation: {component_percent}%",
            f"- AGI-direction gates: {gate_percent}%",
            f"- execution readiness evidence: {execution_readiness_percent}%",
            "",
            "What is strong now:",
        ]
        strong_items = [name for name, score, _ in component_scores if score >= 0.75]
        if strong_items:
            lines.extend(f"- {name}" for name in strong_items)
        else:
            lines.append("- no component has reached strong foundation yet")

        lines.extend(["", "Remaining real-execution gaps:"])
        for name, score, status in gate_scores:
            if score < 1.0:
                lines.append(f"- {name}: {status}")

        lines.extend(
            [
                "",
                "Safety floor:",
                f"- pending approvals visible: {len(pending)}",
                f"- recent failed/blocked runs: {len(failed_runs)}",
                f"- recent approval-held runs: {len(approval_held_runs)}",
                f"- approval-held review: `{approval_held_review_command}`" if approval_held_review_command else "- approval-held review: none",
                "- More autonomy should only increase after exact arguments, approval gates, audit receipts, verification, and recovery are proven.",
                "",
            "Completion blockers:",
        ]
        )
        if completion_blockers:
            lines.extend(f"- {blocker}" for blocker in completion_blockers)
        else:
            lines.append("- none found by this read-only assessment")
        lines.extend(
            [
                "",
                "Execution learning debt:",
                f"- AGI real-execution gaps block completion claim: {'yes' if agi_real_execution_blocks_completion_claim else 'no'}",
                "Storage readiness:",
                f"- runtime fallback active: {'yes' if storage_readiness['active'] else 'no'}",
                f"- blocks completion claim: {'yes' if storage_readiness['blocks_completion_claim'] else 'no'}",
                f"- reason: {storage_readiness['reason'] or 'none'}",
                f"- database route: {storage_readiness['db_path_display'] or 'configured storage'}",
                f"- notes route: {storage_readiness['vault_path_display'] or 'configured vault'}",
                f"- next operator action: {storage_readiness['recovery_next_operator_action'] or 'none'}",
                f"- next commands: {', '.join(f'`{command}`' for command in storage_readiness['next_commands']) if storage_readiness['next_commands'] else 'none'}",
                "",
                "Execution health recovery closure:",
                f"- state: {recovery_closure['state']}",
                f"- checklist overview: `{_recovery_closure_checklist_command(recovery_closure)}`" if _recovery_closure_checklist_command(recovery_closure) else "- checklist overview: none",
                f"- recovery next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- recovery next required: none",
                f"- command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                "",
                "Execution health verification coverage:",
                f"- state: {execution_health_verification['state']}",
                f"- recent verification packets: {execution_health_verification['recent_verification_runs']}",
                f"- next required: `{execution_health_verification['next_required_command']}`" if execution_health_verification["next_required_command"] else "- next required: none",
                f"- proof queue: {', '.join(f'`{command}`' for command in execution_health_verification['proof_queue']) if execution_health_verification['proof_queue'] else 'none'}",
                "",
                "Execution learning debt:",
                f"- state: {learning_debt['state']}",
                f"- recent action runs: {learning_debt['recent_action_runs']}",
                f"- failed or blocked action runs: {learning_debt['failed_or_blocked_action_runs']}",
                f"- recent after-action learning packets: {learning_debt['recent_after_action_learning_runs']}",
                f"- missing learning proof: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
                f"- learning evidence next required: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- learning evidence next required: none",
                f"- learning actionable next required: `{learning_actionable_commands[0]}`" if learning_actionable_commands else "- learning actionable next required: none",
                f"- learning actionable proof alias: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- learning actionable proof alias: none",
                f"- learning closure command: `{learning_closure_command}`" if learning_closure_command else "- learning closure command: none",
                f"- learning next required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- learning next required: none",
                f"- actionable learning queue: {', '.join(f'`{command}`' for command in learning_actionable_commands) if learning_actionable_commands else 'none'}",
                "",
                "Next required commands:",
                *[f"- `{command}`" for command in next_proof_commands],
                "",
                "Selected AGI next build target:",
                f"- gate: {selected_agi_gate_name or 'none'}",
                f"- target: {selected_agi_target.get('title') or selected_agi_gate.get('next') or 'none'}",
                f"- command: `{selected_agi_closure_commands[0]}`" if selected_agi_closure_commands else "- command: none",
                f"- target integrity: {selected_agi_file_integrity['status']}",
                f"- owning files checked: {selected_agi_file_integrity['checked']}",
                f"- focused verification commands: {len(selected_agi_verification_commands)}",
                f"- acceptance checks: {len(selected_agi_acceptance_checks)}",
                f"- ready for scoped implementation review: {'yes' if selected_agi_build_ready else 'no'}",
                f"- closure queue: {', '.join(f'`{command}`' for command in selected_agi_closure_commands) if selected_agi_closure_commands else 'none'}",
                f"- focused verification: {', '.join(f'`{command}`' for command in selected_agi_verification_commands) if selected_agi_verification_commands else 'none configured'}",
                "",
                "Completion proof queue:",
                *[f"- `{command}`" for command in completion_proof_queue],
                "",
                f"Completion claim ready: {'yes' if completion_claim_ready else 'no'}",
                "Rule: run `completion audit`, `evidence ledger`, and `completion claim gate` before any final completion claim.",
            ]
        )

        completion_metadata = {
            "overall_percent": overall_percent,
            "component_percent": component_percent,
            "gate_percent": gate_percent,
            "execution_readiness_percent": execution_readiness_percent,
            "pending_approvals": len(pending),
            "open_tasks": len(open_tasks),
            "active_goals": len(active_goals),
            "recent_failed_runs": len(failed_runs),
            "recent_approval_held_runs": len(approval_held_runs),
            "approval_held_review_command": approval_held_review_command,
            "components": len(component_scores),
            "incomplete_components": incomplete_components,
            "incomplete_component_count": len(incomplete_components),
            "gates": len(gate_scores),
            "agi_gates": agi_summary["gates"],
            "agi_strong_gates": agi_summary["strong"],
            "agi_partial_gates": agi_summary["partial"],
            "agi_missing_gates": agi_summary["missing"],
            "agi_real_execution_gap_count": agi_summary["real_execution_gap_count"],
            "agi_real_execution_blocks_completion_claim": agi_real_execution_blocks_completion_claim,
            "completion_claim_blocked_by_agi_gaps": agi_real_execution_blocks_completion_claim,
            "storage_runtime_fallback_active": storage_readiness["active"],
            "storage_runtime_fallback_reason": storage_readiness["reason"],
            "storage_runtime_fallback_exception_type": storage_readiness["exception_type"],
            "storage_runtime_fallback_db_path_display": storage_readiness["db_path_display"],
            "storage_runtime_fallback_vault_path_display": storage_readiness["vault_path_display"],
            "storage_readiness_blocks_completion_claim": storage_readiness["blocks_completion_claim"],
            "storage_recovery_required": storage_readiness["recovery_required"],
            "storage_recovery_reason": storage_readiness["recovery_reason"],
            "storage_issues": list(storage_readiness.get("issues") or []),
            "storage_issue_count": int(storage_readiness.get("issue_count") or 0),
            "storage_recovery_mode": storage_readiness["recovery_mode"],
            "storage_recovery_next_operator_action": storage_readiness["recovery_next_operator_action"],
            "storage_recovery_restart_required": storage_readiness["recovery_restart_required"],
            "storage_recovery_check_command": storage_readiness["recovery_check_command"],
            "storage_recovery_check_tool_command": storage_readiness["recovery_check_tool_command"],
            "storage_recovery_check_api": STORAGE_RECOVERY_CHECK_API,
            "storage_recovery_command": storage_readiness["recovery_command"],
            "storage_readiness_blocker": storage_readiness["blocker"],
            "storage_readiness_next_commands": storage_readiness["next_commands"],
            "storage_readiness_next_command_count": len(storage_readiness["next_commands"]),
            "storage_readiness_next_required_command": storage_readiness["next_commands"][0] if storage_readiness["blocks_completion_claim"] and storage_readiness["next_commands"] else "",
            "storage_readiness_proof_queue": list(storage_readiness["next_commands"]) if storage_readiness["blocks_completion_claim"] else [],
            "storage_readiness_proof_queue_count": len(storage_readiness["next_commands"]) if storage_readiness["blocks_completion_claim"] else 0,
            "storage_readiness_first_proof_command": storage_readiness["next_commands"][0] if storage_readiness["blocks_completion_claim"] and storage_readiness["next_commands"] else "",
            "agi_next_gate": selected_agi_gate_name,
            "agi_next_target_title": selected_agi_target.get("title") or selected_agi_gate.get("next") or "",
            "agi_next_build_command": selected_agi_closure_commands[0] if selected_agi_closure_commands else "",
            "agi_next_evidence_closure_commands": selected_agi_closure_commands,
            "agi_next_evidence_closure_command_count": len(selected_agi_closure_commands),
            "agi_next_focused_verification_commands": selected_agi_verification_commands,
            "agi_next_focused_verification_command_count": len(selected_agi_verification_commands),
            "agi_next_likely_files": selected_agi_likely_files,
            "agi_next_likely_file_count": len(selected_agi_likely_files),
            "agi_next_target_file_integrity_status": selected_agi_file_integrity["status"],
            "agi_next_target_files_checked": selected_agi_file_integrity["checked"],
            "agi_next_target_files_exist": selected_agi_file_integrity["all_exist"],
            "agi_next_missing_target_files": selected_agi_file_integrity["missing"],
            "agi_next_missing_target_file_count": selected_agi_file_integrity["missing_count"],
            "agi_next_target_integrity_blocks_start": not selected_agi_file_integrity["all_exist"],
            "agi_next_acceptance_checks": selected_agi_acceptance_checks,
            "agi_next_acceptance_check_count": len(selected_agi_acceptance_checks),
            **_agi_acceptance_gap_metadata({"agi_next_acceptance_checks": selected_agi_acceptance_checks}),
            "agi_next_build_packet_ready_for_review": selected_agi_build_ready,
            **_agi_focus_selection_metadata(selected_agi_readiness),
            "completion_blockers": completion_blockers,
            "completion_blocker_count": len(completion_blockers),
            "completion_blockers_deduplicated": len(completion_blockers) == len(set(completion_blockers)),
            "completion_claim_ready": completion_claim_ready,
            "next_proof_commands": next_proof_commands,
            "next_proof_command": next_proof_commands[0] if next_proof_commands else "",
            "completion_proof_queue": completion_proof_queue,
            "completion_proof_queue_count": len(completion_proof_queue),
            "completion_next_proof_command": completion_proof_queue[0] if completion_proof_queue else "",
            "next_completion_proof_command": completion_proof_queue[0] if completion_proof_queue else "",
            "execution_health_recovery_closure_state": recovery_closure["state"],
            "execution_health_recovery_closure_ready_to_retry": recovery_closure["ready_to_retry"],
            "execution_health_recovery_closure_missing": recovery_closure["missing"],
            "execution_health_recovery_closure_missing_count": recovery_closure["missing_count"],
            "execution_health_recovery_closure_required_commands": recovery_closure["required_commands"],
            "execution_health_recovery_closure_next_required_command": recovery_closure["next_required_command"],
            "execution_health_recovery_closure_checklist_command": _recovery_closure_checklist_command(recovery_closure),
            "execution_health_recovery_closure_should_open_checklist": bool(_recovery_closure_checklist_command(recovery_closure)),
            "execution_health_recovery_closure_proof_queue": recovery_closure["proof_queue"],
            "execution_health_recovery_closure_proof_queue_count": recovery_closure["proof_queue_count"],
            "execution_health_recovery_closure_next_proof_command": recovery_closure["next_proof_command"],
            "execution_health_recovery_closure_blocks_completion_claim": recovery_closure["blocks_completion_claim"],
            "execution_health_recovery_closure_target_run_id": recovery_closure["target_run_id"],
            "execution_health_recovery_closure_target_tool_name": recovery_closure["target_tool_name"],
            "execution_health_recovery_closure_target_verification_receipts": recovery_closure["target_verification_receipts"],
            "execution_health_recovery_closure_target_recovery_packets": recovery_closure["target_recovery_packets"],
            "execution_health_recovery_closure_target_after_action_learning_packets": recovery_closure["target_after_action_learning_packets"],
            "execution_health_verification_coverage_state": execution_health_verification["state"],
            "execution_health_verification_recent_tool_runs": execution_health_verification["recent_tool_runs"],
            "execution_health_verification_recent_action_runs": execution_health_verification["recent_action_runs"],
            "execution_health_verification_recent_verification_runs": execution_health_verification["recent_verification_runs"],
            "execution_health_verification_proof_queue": execution_health_verification["proof_queue"],
            "execution_health_verification_proof_queue_count": execution_health_verification["proof_queue_count"],
            "execution_health_verification_next_required_command": execution_health_verification["next_required_command"],
            "execution_health_verification_next_proof_command": execution_health_verification["next_proof_command"],
            "execution_health_verification_blocks_completion_claim": execution_health_verification["blocks_completion_claim"],
            "execution_learning_state": learning_debt["state"],
            "execution_learning_blocks_completion_claim": learning_debt["blocks_completion_claim"],
            "execution_learning_recent_action_runs": learning_debt["recent_action_runs"],
            "execution_learning_failed_or_blocked_action_runs": learning_debt["failed_or_blocked_action_runs"],
            "execution_learning_recent_verification_runs": learning_debt["recent_verification_runs"],
            "execution_learning_recent_recovery_runs": learning_debt["recent_recovery_runs"],
            "execution_learning_recent_after_action_learning_runs": learning_debt["recent_after_action_learning_runs"],
            "execution_learning_target_run_id": learning_debt["target_run_id"],
            "execution_learning_target_tool_name": learning_debt["target_tool_name"],
            "execution_learning_target_after_action_learning_packets": learning_debt["target_after_action_learning_packets"],
            "execution_learning_missing": learning_debt["missing"],
            "execution_learning_missing_count": learning_debt["missing_count"],
            "execution_learning_closure_command": learning_closure_command,
            "execution_learning_evidence_command": learning_debt["learning_evidence_command"],
            "execution_learning_after_action_learning_command": learning_debt["after_action_learning_command"],
            "execution_learning_required_commands": learning_debt["required_commands"],
            "execution_learning_next_required_command": learning_debt["next_required_command"],
            "execution_learning_proof_queue": learning_debt["proof_queue"],
            "execution_learning_proof_queue_count": learning_debt["proof_queue_count"],
            "execution_learning_next_proof_command": learning_debt["next_proof_command"],
            "execution_learning_actionable_required_commands": learning_actionable_commands,
            "execution_learning_actionable_required_command_count": len(learning_actionable_commands),
            "execution_learning_actionable_proof_queue": learning_actionable_commands,
            "execution_learning_actionable_proof_queue_count": len(learning_actionable_commands),
            "execution_learning_actionable_next_required_command": learning_debt["actionable_next_required_command"],
            "execution_learning_actionable_next_proof_command": learning_debt["actionable_next_proof_command"],
            "execution_learning_next_evidence_command": learning_debt["next_evidence_command"],
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        }
        harness_completion_handoff = _safe_metadata(
            source="harness_completion_assessment",
            **completion_metadata,
        )

        return ToolResult(
            "harness_completion_assessment",
            True,
            "\n".join(lines),
            _safe_metadata(
                **completion_metadata,
                harness_completion_handoff=harness_completion_handoff,
            ),
        )

    def completion_audit_packet(args: dict[str, Any]) -> ToolResult:
        objective = _short(
            args.get("objective") or args.get("goal") or args.get("request") or (
                "Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant."
            ),
            limit=500,
        )
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=10)
        recent_runs = store.recent_tool_runs(limit=50)
        failed_runs, approval_held_runs = _recent_tool_run_attention_buckets(recent_runs)
        approval_held_review_command = _approval_review_command(approval_held_runs) if approval_held_runs else ""
        recovery_closure = _execution_health_recovery_closure_snapshot(store, recent_runs)
        execution_health_verification = _execution_health_verification_snapshot(recent_runs)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        open_tasks = store.list_tasks(status="open", limit=10)
        active_goals = store.list_goals(status="active", limit=10)
        agi_summary = _agi_gate_summary(tool_names)
        selected_agi_readiness = _selected_agi_target_readiness(agi_summary)
        selected_agi_gate_name = selected_agi_readiness["gate_name"]
        selected_agi_closure_commands = selected_agi_readiness["closure_commands"]
        selected_agi_verification_commands = selected_agi_readiness["verification_commands"]
        selected_agi_likely_files = selected_agi_readiness["likely_files"]
        selected_agi_file_integrity = selected_agi_readiness["file_integrity"]
        selected_agi_acceptance_checks = selected_agi_readiness["acceptance_checks"]
        selected_agi_build_ready = selected_agi_readiness["build_ready"]
        selected_agi_target_title = selected_agi_readiness["target_title"]
        agi_real_execution_blocks_completion_claim = agi_summary["real_execution_gap_count"] > 0
        learning_closure_command = (
            f"execution learning closure {learning_debt['target_run_id']}"
            if learning_debt["blocks_completion_claim"] and learning_debt["target_run_id"] is not None
            else "execution learning closure"
            if learning_debt["blocks_completion_claim"]
            else ""
        )
        learning_actionable_commands = list(learning_debt["actionable_required_commands"])
        completion_proof_queue = _completion_proof_queue_from_sources(
            recovery_proof_queue=list(recovery_closure["proof_queue"]),
            execution_health_verification_proof_queue=list(execution_health_verification["proof_queue"]),
            learning_closure_command=learning_closure_command,
            learning_proof_queue=list(learning_debt["proof_queue"]),
            learning_actionable_proof_queue=learning_actionable_commands,
            pending_approval_ids=[pending[0]["id"]] if pending else [],
            open_task_ids=[open_tasks[0]["id"]] if open_tasks else [],
            baseline_commands=["harness completion", "agi gates", "completion audit", "evidence ledger", "completion claim gate"],
            selected_agi_closure_commands=selected_agi_closure_commands,
        )
        if approval_held_review_command:
            _append_unique(completion_proof_queue, [approval_held_review_command])

        requirement_groups = [
            {
                "name": "command-first interface",
                "prove_with": {"status_dashboard", "chat_loop_preview", "voice_command_lifecycle"},
                "evidence_source": "`status dashboard`, `/api/status`, `chat loop preview`, voice lifecycle tools",
                "missing_if": "no rendered GUI/status route or no speech/text lifecycle preview",
            },
            {
                "name": "routing and tool orchestration",
                "prove_with": {"execution_governor_packet", "dispatch_decision_packet", "planner_gap_packet", "execution_contract", "argument_contract_packet", "execution_mission_control", "save_execution_case", "append_execution_case_evidence", "inspect_execution_case", "execution_case_gate", "execution_case_review_packet", "execution_case_timeline", "execution_runbook", "execution_proof_bundle", "execution_acceptance_gate", "execution_readiness_matrix", "verification_packet", "action_rehearsal", "risk_matrix", "specialist_orchestration_packet", "specialist_route_quality", "specialist_execution_readiness", "specialist_handoff_receipt", "specialist_model_draft", "specialist_action_proposal_contract", "specialist_tool_dry_run_packet", "specialist_proposal_completion_gate", "specialist_execution_handoff_packet", "specialist_post_run_closure_packet", "specialist_cycle_ledger"},
                "evidence_source": "`execution governor`, `dispatch decision`, `planner gap`, `execution contract`, `argument contract`, `execution mission control`, `save execution case`, `case evidence latest`, `execution case latest`, `execution case gate`, `execution case review`, `execution case timeline`, `execution runbook`, `acceptance gate`, `execution readiness matrix`, `verification packet`, `action rehearsal`, `risk matrix`, `specialist handoff receipt`",
                "missing_if": "orders cannot be converted into exact route, planner-gap, tool-argument, risk, and verification packets",
            },
            {
                "name": "memory and state",
                "prove_with": {"memory_tree_summary", "export_state_snapshot", "work_queue", "return_brief"},
                "evidence_source": "`memory tree`, `export state snapshot`, `work queue`, `return brief`",
                "missing_if": "Jarvis cannot show durable memory/state needed to resume work",
            },
            {
                "name": "approval gates and audit",
                "prove_with": {"approval_readiness_packet", "approval_execution_packet", "review_pending_approvals", "approval_history", "recent_tool_runs", "runtime_trace_receipt", "execution_audit_gate", "execution_health_report", "recovery_closure_checklist"},
                "evidence_source": "`approval review`, `approval readiness #ID`, `approval packet #ID`, `approval chain proof #ID`, `approval history`, `recent tool runs`, `runtime trace receipt`, `execution audit gate`, `execution health report`",
                "missing_if": "risky actions lack readiness review, last-look packet, approval chain proof, one-shot approval, or audit evidence",
            },
            {
                "name": "verification and recovery",
                "prove_with": {"verification_packet", "verification_receipt", "execution_recovery_packet", "execution_health_report", "recovery_closure_checklist", "checkpoint_recovery_preview", "checkpoint_recovery_execute", "checkpoint_recovery_receipt", "checkpoint_recovery_followthrough_packet", "autonomy_resume_gate", "autonomy_continuation_execution_packet", "autonomy_step_closure_packet", "autonomy_cycle_ledger"},
                "evidence_source": "`verification packet`, `verification receipt`, `execution recovery packet`, `execution health report`, checkpoint recovery preview/apply/receipt/follow-through tools",
                "missing_if": "Jarvis cannot identify proof, failure signals, recovery path, or recovery-closure state",
            },
            {
                "name": "diagnostics and observability",
                "prove_with": {"readiness_report", "build_progress_report", "build_delta_report", "harness_completion_assessment"},
                "evidence_source": "`readiness report`, `build progress`, `build delta`, `harness completion`",
                "missing_if": "status, blockers, recent runs, and completion estimates are not visible",
            },
            {
                "name": "learning loop",
                "prove_with": {"learning_review", "after_action_learning_packet", "feedback_actions", "failure_promotion_packet", "failure_implementation_packet", "failure_apply_contract", "failure_patch_receipt_packet", "failure_patch_application_bridge", "failure_patch_completion_gate", "failure_patch_handoff_packet", "failure_patch_closeout_packet", "failure_learning_record_packet", "failure_learning_closure_ledger", "draft_skill_from_session"},
                "evidence_source": "`learning review`, `feedback actions`, `failure promotion packet`, skill drafting",
                "missing_if": "failures and corrections cannot become tests, preferences, or skills",
            },
        ]

        lines = [
            "Jarvis completion audit packet:",
            "This is a proof checklist, not a completion claim. It is read-only and does not mark the goal done.",
            "",
            f"Objective under audit: {objective}",
            "",
            "Audit rule:",
            "- Treat completion as unproven until every requirement has direct current evidence.",
            "- Strong evidence means a current tool/API/test/runtime output proves the behavior at the right scope.",
            "- Weak evidence means a nearby feature exists but the full behavior is still unproven.",
            "- Missing evidence means keep building before claiming completion.",
            "",
            "Requirement evidence:",
        ]
        strong = 0
        partial = 0
        missing = 0
        for group in requirement_groups:
            present = sorted(group["prove_with"] & tool_names)
            if len(present) >= 3:
                status = "strong evidence"
                strong += 1
            elif present:
                status = "partial evidence"
                partial += 1
            else:
                status = "missing evidence"
                missing += 1
            lines.extend(
                [
                    f"- {group['name']}: {status}",
                    f"  Evidence to inspect: {group['evidence_source']}",
                    f"  Current matching tools: {', '.join(present) if present else 'none'}",
                    f"  Missing if: {group['missing_if']}",
                ]
            )

        lines.extend(
            [
                "",
                "AGI-direction gate coverage:",
                f"- strong prototype gates: {agi_summary['strong']} / {agi_summary['gates']}",
                f"- partial prototype gates: {agi_summary['partial']}",
                f"- missing gates: {agi_summary['missing']}",
                f"- real-execution gaps still tracked: {agi_summary['real_execution_gap_count']}",
                f"- selected next build gate: {selected_agi_gate_name or 'none'}",
                f"- selected target: {selected_agi_target_title or 'none'}",
                f"- selected next build command: `{selected_agi_closure_commands[0]}`" if selected_agi_closure_commands else "- selected next build command: none",
                f"- selected closure queue: {', '.join(f'`{command}`' for command in selected_agi_closure_commands) if selected_agi_closure_commands else 'none'}",
                f"- selected focused verification: {', '.join(f'`{command}`' for command in selected_agi_verification_commands) if selected_agi_verification_commands else 'none configured'}",
                f"- selected target integrity: {selected_agi_file_integrity['status']}",
                f"- selected target files checked: {selected_agi_file_integrity['checked']}",
                f"- selected acceptance checks: {len(selected_agi_acceptance_checks)}",
                f"- selected target ready for scoped implementation review: {'yes' if selected_agi_build_ready else 'no'}",
            ]
        )
        for gate in agi_summary["rows"]:
            lines.extend(
                [
                    f"- {gate['gate']}: {gate['status']}",
                    f"  Evidence still missing: {', '.join(gate['missing_evidence']) if gate['missing_evidence'] else 'none from registry'}",
                    f"  Real-execution gap: {gate['real_execution_gap']}",
                    f"  Next safe build move: {gate['next']}",
                    f"  Evidence closure commands: {', '.join(f'`{command}`' for command in gate['evidence_closure_commands'])}",
                    f"  Focused verification: {', '.join(f'`{command}`' for command in gate['focused_verification']) if gate['focused_verification'] else 'none configured'}",
                ]
            )

        lines.extend(
            [
                "",
                "Current-state blockers:",
                f"- pending approvals: {len(pending)}",
                f"- open tasks: {len(open_tasks)}",
                f"- active goals: {len(active_goals)}",
                f"- recent failed/blocked tool runs: {len(failed_runs)}",
                f"- recent approval-held tool runs: {len(approval_held_runs)}",
                f"- approval-held review: `{approval_held_review_command}`" if approval_held_review_command else "- approval-held review: none",
                f"- AGI real-execution gaps block completion claim: {'yes' if agi_real_execution_blocks_completion_claim else 'no'}",
                f"- recovery closure state: {recovery_closure['state']}",
                f"- recovery closure ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
                f"- recovery closure missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- recovery closure checklist overview: `{_recovery_closure_checklist_command(recovery_closure)}`" if _recovery_closure_checklist_command(recovery_closure) else "- recovery closure checklist overview: none",
                f"- recovery next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- recovery next required: none",
                f"- recovery closure command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                f"- execution learning debt state: {learning_debt['state']}",
                f"- execution learning debt missing: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
                f"- learning evidence next required: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- learning evidence next required: none",
                f"- learning actionable next required: `{learning_actionable_commands[0]}`" if learning_actionable_commands else "- learning actionable next required: none",
                f"- learning actionable proof alias: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- learning actionable proof alias: none",
                f"- execution learning closure command: `{learning_closure_command}`" if learning_closure_command else "- execution learning closure command: none",
                f"- learning next required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- learning next required: none",
                f"- execution learning command queue: {', '.join(f'`{command}`' for command in learning_debt['required_commands']) if learning_debt['required_commands'] else 'none'}",
                f"- execution learning actionable queue: {', '.join(f'`{command}`' for command in learning_actionable_commands) if learning_actionable_commands else 'none'}",
            ]
        )
        if pending:
            first = pending[0]
            lines.append(f"- first approval blocker: approval #{first['id']} {first['tool_name']} -> `approval readiness {first['id']}` then `approval packet {first['id']}` then `approval chain proof {first['id']}`")
        if open_tasks:
            lines.append(f"- first open task: #{open_tasks[0]['id']} {_short(open_tasks[0]['body'])}")
        if recovery_closure["required_commands"]:
            lines.append("- recovery closure queue:")
            lines.extend(f"  - `{command}`" for command in recovery_closure["required_commands"])
        if learning_debt["required_commands"]:
            lines.append("- execution learning debt queue:")
            lines.extend(f"  - `{command}`" for command in (learning_actionable_commands or learning_debt["required_commands"]))

        lines.extend(
            [
                "",
                "Next verification commands:",
                "- `harness completion` for the current prototype estimate.",
                "- `agi gates` for remaining AGI-direction gates.",
                "- `dispatch decision: <next real order>`, `execution readiness matrix: <next real order>`, and `verification packet: <next real order>` before trusting execution completion.",
                "- `readiness report`, `build progress`, and `recent tool runs` for runtime evidence.",
                "- `smoke_test_status_server` and the closest feature smoke test after code changes.",
                "",
                "Completion proof queue:",
                *[f"- `{command}`" for command in completion_proof_queue],
                "",
                "Completion verdict:",
            ]
        )
        if (
            missing
            or partial
            or pending
            or open_tasks
            or agi_real_execution_blocks_completion_claim
            or approval_held_runs
            or recovery_closure["blocks_completion_claim"]
            or learning_debt["blocks_completion_claim"]
        ):
            lines.append("- NOT COMPLETE: evidence is still partial, missing, or blocked by current work state.")
        else:
            lines.append("- AUDIT LOOKS CLEAR: run broad verification before any final completion claim.")

        lines.extend(
            [
                "",
                "Boundary:",
                "- This audit packet does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        audit_metadata = {
            "objective": objective,
            "requirements": len(requirement_groups),
            "strong_evidence": strong,
            "partial_evidence": partial,
            "missing_evidence": missing,
            "pending_approvals": len(pending),
            "open_tasks": len(open_tasks),
            "active_goals": len(active_goals),
            "recent_failed_runs": len(failed_runs),
            "recent_approval_held_runs": len(approval_held_runs),
            "approval_held_review_command": approval_held_review_command,
            "next_proof_commands": completion_proof_queue,
            "next_proof_command": completion_proof_queue[0] if completion_proof_queue else "",
            "completion_proof_queue": completion_proof_queue,
            "completion_proof_queue_count": len(completion_proof_queue),
            "completion_next_proof_command": completion_proof_queue[0] if completion_proof_queue else "",
            "next_completion_proof_command": completion_proof_queue[0] if completion_proof_queue else "",
            "execution_health_recovery_closure_state": recovery_closure["state"],
            "execution_health_recovery_closure_ready_to_retry": recovery_closure["ready_to_retry"],
            "execution_health_recovery_closure_missing": recovery_closure["missing"],
            "execution_health_recovery_closure_missing_count": recovery_closure["missing_count"],
            "execution_health_recovery_closure_required_commands": recovery_closure["required_commands"],
            "execution_health_recovery_closure_next_required_command": recovery_closure["next_required_command"],
            "execution_health_recovery_closure_checklist_command": _recovery_closure_checklist_command(recovery_closure),
            "execution_health_recovery_closure_should_open_checklist": bool(_recovery_closure_checklist_command(recovery_closure)),
            "execution_health_recovery_closure_proof_queue": recovery_closure["proof_queue"],
            "execution_health_recovery_closure_proof_queue_count": recovery_closure["proof_queue_count"],
            "execution_health_recovery_closure_next_proof_command": recovery_closure["next_proof_command"],
            "execution_health_recovery_closure_blocks_completion_claim": recovery_closure["blocks_completion_claim"],
            "execution_health_recovery_closure_target_run_id": recovery_closure["target_run_id"],
            "execution_health_recovery_closure_target_tool_name": recovery_closure["target_tool_name"],
            "execution_health_recovery_closure_target_verification_receipts": recovery_closure["target_verification_receipts"],
            "execution_health_recovery_closure_target_recovery_packets": recovery_closure["target_recovery_packets"],
            "execution_health_recovery_closure_target_after_action_learning_packets": recovery_closure["target_after_action_learning_packets"],
            "execution_health_verification_coverage_state": execution_health_verification["state"],
            "execution_health_verification_recent_tool_runs": execution_health_verification["recent_tool_runs"],
            "execution_health_verification_recent_action_runs": execution_health_verification["recent_action_runs"],
            "execution_health_verification_recent_verification_runs": execution_health_verification["recent_verification_runs"],
            "execution_health_verification_proof_queue": execution_health_verification["proof_queue"],
            "execution_health_verification_proof_queue_count": execution_health_verification["proof_queue_count"],
            "execution_health_verification_next_required_command": execution_health_verification["next_required_command"],
            "execution_health_verification_next_proof_command": execution_health_verification["next_proof_command"],
            "execution_health_verification_blocks_completion_claim": execution_health_verification["blocks_completion_claim"],
            "execution_learning_state": learning_debt["state"],
            "execution_learning_blocks_completion_claim": learning_debt["blocks_completion_claim"],
            "execution_learning_recent_action_runs": learning_debt["recent_action_runs"],
            "execution_learning_failed_or_blocked_action_runs": learning_debt["failed_or_blocked_action_runs"],
            "execution_learning_recent_verification_runs": learning_debt["recent_verification_runs"],
            "execution_learning_recent_recovery_runs": learning_debt["recent_recovery_runs"],
            "execution_learning_recent_after_action_learning_runs": learning_debt["recent_after_action_learning_runs"],
            "execution_learning_target_run_id": learning_debt["target_run_id"],
            "execution_learning_target_tool_name": learning_debt["target_tool_name"],
            "execution_learning_target_after_action_learning_packets": learning_debt["target_after_action_learning_packets"],
            "execution_learning_missing": learning_debt["missing"],
            "execution_learning_missing_count": learning_debt["missing_count"],
            "execution_learning_closure_command": learning_closure_command,
            "execution_learning_evidence_command": learning_debt["learning_evidence_command"],
            "execution_learning_after_action_learning_command": learning_debt["after_action_learning_command"],
            "execution_learning_required_commands": learning_debt["required_commands"],
            "execution_learning_next_required_command": learning_debt["next_required_command"],
            "execution_learning_proof_queue": learning_debt["proof_queue"],
            "execution_learning_proof_queue_count": learning_debt["proof_queue_count"],
            "execution_learning_next_proof_command": learning_debt["next_proof_command"],
            "execution_learning_actionable_required_commands": learning_actionable_commands,
            "execution_learning_actionable_required_command_count": len(learning_actionable_commands),
            "execution_learning_actionable_proof_queue": learning_actionable_commands,
            "execution_learning_actionable_proof_queue_count": len(learning_actionable_commands),
            "execution_learning_actionable_next_required_command": learning_debt["actionable_next_required_command"],
            "execution_learning_actionable_next_proof_command": learning_debt["actionable_next_proof_command"],
            "execution_learning_next_evidence_command": learning_debt["next_evidence_command"],
            "agi_gates": agi_summary["gates"],
            "agi_strong_gates": agi_summary["strong"],
            "agi_partial_gates": agi_summary["partial"],
            "agi_missing_gates": agi_summary["missing"],
            "agi_gate_statuses": agi_summary["gate_statuses"],
            "agi_present_evidence_by_gate": agi_summary["present_evidence_by_gate"],
            "agi_missing_evidence_by_gate": agi_summary["missing_evidence_by_gate"],
            "agi_real_execution_gaps_by_gate": agi_summary["real_execution_gaps_by_gate"],
            "agi_real_execution_gap_count": agi_summary["real_execution_gap_count"],
            "agi_real_execution_blocks_completion_claim": agi_real_execution_blocks_completion_claim,
            "completion_claim_blocked_by_agi_gaps": agi_real_execution_blocks_completion_claim,
            "agi_next_moves_by_gate": agi_summary["next_moves_by_gate"],
            "agi_evidence_closure_commands_by_gate": agi_summary["evidence_closure_commands_by_gate"],
            "agi_focused_verification_by_gate": agi_summary["focused_verification_by_gate"],
            "agi_next_gate": selected_agi_gate_name,
            "agi_next_target_title": selected_agi_target_title,
            "agi_next_build_command": selected_agi_closure_commands[0] if selected_agi_closure_commands else "",
            "agi_next_evidence_closure_commands": selected_agi_closure_commands,
            "agi_next_evidence_closure_command_count": len(selected_agi_closure_commands),
            "agi_next_focused_verification_commands": selected_agi_verification_commands,
            "agi_next_focused_verification_command_count": len(selected_agi_verification_commands),
            "agi_next_likely_files": selected_agi_likely_files,
            "agi_next_likely_file_count": len(selected_agi_likely_files),
            "agi_next_target_file_integrity_status": selected_agi_file_integrity["status"],
            "agi_next_target_files_checked": selected_agi_file_integrity["checked"],
            "agi_next_target_files_exist": selected_agi_file_integrity["all_exist"],
            "agi_next_missing_target_files": selected_agi_file_integrity["missing"],
            "agi_next_missing_target_file_count": selected_agi_file_integrity["missing_count"],
            "agi_next_target_integrity_blocks_start": not selected_agi_file_integrity["all_exist"],
            "agi_next_acceptance_checks": selected_agi_acceptance_checks,
            "agi_next_acceptance_check_count": len(selected_agi_acceptance_checks),
            **_agi_acceptance_gap_metadata({"agi_next_acceptance_checks": selected_agi_acceptance_checks}),
            "agi_next_build_packet_ready_for_review": selected_agi_build_ready,
            **_agi_focus_selection_metadata(selected_agi_readiness),
            "completion_claim_ready": not (
                missing
                or partial
                or pending
                or open_tasks
                or agi_real_execution_blocks_completion_claim
                or approval_held_runs
                or recovery_closure["blocks_completion_claim"]
                or learning_debt["blocks_completion_claim"]
            ),
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        }
        completion_audit_handoff = _safe_metadata(
            source="completion_audit_packet",
            **audit_metadata,
        )

        return ToolResult(
            "completion_audit_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                **audit_metadata,
                completion_audit_handoff=completion_audit_handoff,
            ),
        )

    def _latest_execution_case_proof_state() -> dict[str, Any]:
        cases = store.list_execution_cases(limit=1)
        if not cases:
            return {
                "found": False,
                "case_id": None,
                "verdict": "CASE_NOT_FOUND",
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
                "missing_case_proofs": [],
                "next_case_proof_commands": [],
                "next_case_required_command": "save execution case: <next real order>",
                "next_case_proof_command": "save execution case: <next real order>",
                "recovery_closure_state": "no_case",
                "recovery_closure_missing": [],
                "recovery_closure_next_required_command": "",
                "recovery_closure_required_commands": [],
                "recovery_closure_proof_queue": [],
                "recovery_closure_proof_queue_count": 0,
                "recovery_closure_next_proof_command": "",
                "recovery_closure_blocks_completion_claim": False,
                "learning_state": "no_case",
                "learning_missing": [],
                "learning_next_required_command": "",
                "learning_required_commands": [],
                "learning_proof_queue": [],
                "learning_proof_queue_count": 0,
                "learning_next_proof_command": "",
                "learning_actionable_required_commands": [],
                "learning_actionable_proof_queue": [],
                "learning_actionable_proof_queue_count": 0,
                "learning_actionable_next_required_command": "",
                "learning_actionable_next_proof_command": "",
                "learning_next_evidence_command": "",
                "learning_blocks_completion_claim": False,
                "approval_queue_forecast": [],
                "forecast_new_approvals": 0,
                "forecast_reused_approval_ids": [],
                "forecast_queue_before": 0,
                "forecast_queue_after_if_sent": 0,
                "forecast_queue_delta_if_sent": 0,
                "ready_for_human_review": False,
                "review_state": "NO_CASE",
                "review_verdict": "CASE_NOT_FOUND",
                "review_next_safe_command": "",
                "review_approval_required": False,
                "review_checklist_items": 0,
                "review_blocker_count": 0,
                "review_handoff": {},
                "review_handoff_present": False,
                "closure_verdict": "CASE_CLOSURE_NOT_FOUND",
                "case_closure_ready": False,
                "case_closure_blocks_completion_claim": True,
                "closure_proof_queue": ["save execution case: <next real order>"],
                "closure_proof_queue_count": 1,
                "next_closure_proof_command": "save execution case: <next real order>",
            }

        case = cases[0]
        case_id = int(case["id"])
        try:
            gate = execution_case_gate({"case_id": case_id})
            metadata = gate.metadata if gate.ok else {}
        except Exception:
            metadata = {}
        try:
            closure = execution_case_closure_packet({"case_id": case_id})
            closure_metadata = closure.metadata if closure.ok else {}
        except Exception:
            closure_metadata = {}
        try:
            review = execution_case_review_packet({"case_id": case_id})
            review_metadata = review.metadata if review.ok else {}
        except Exception:
            review_metadata = {}

        case_metadata = _execution_case_metadata(case)
        mission_command_queue = list(
            metadata.get("mission_command_queue")
            or case_metadata.get("mission_command_queue")
            or []
        )
        verdict = str(metadata.get("verdict") or "CASE_UNKNOWN")
        next_commands = list(metadata.get("next_case_proof_commands") or [])
        next_command = str(
            metadata.get("next_case_proof_command")
            or (next_commands[0] if next_commands else "")
            or metadata.get("next_safe_command")
            or case["next_command"]
            or "execution case gate"
        )
        next_required_command = str(
            metadata.get("next_case_required_command")
            or metadata.get("next_case_proof_command")
            or (next_commands[0] if next_commands else "")
            or metadata.get("next_safe_command")
            or case["next_command"]
            or "execution case gate"
        )
        missing_case_proofs = list(metadata.get("missing_case_proofs") or [])
        recovery_closure_state = str(metadata.get("execution_health_recovery_closure_state") or "unknown")
        recovery_closure_missing = list(metadata.get("execution_health_recovery_closure_missing") or [])
        recovery_closure_next_required_command = str(metadata.get("execution_health_recovery_closure_next_required_command") or "")
        recovery_closure_required_commands = list(metadata.get("execution_health_recovery_closure_required_commands") or [])
        recovery_closure_proof_queue = list(metadata.get("execution_health_recovery_closure_proof_queue") or [])
        recovery_closure_next_proof_command = str(metadata.get("execution_health_recovery_closure_next_proof_command") or "")
        recovery_closure_blocks_completion_claim = _metadata_bool(metadata.get("execution_health_recovery_closure_blocks_completion_claim"))
        learning_state = str(metadata.get("execution_learning_state") or "unknown")
        learning_missing = list(metadata.get("execution_learning_missing") or [])
        learning_next_required_command = str(metadata.get("execution_learning_next_required_command") or "")
        learning_required_commands = list(metadata.get("execution_learning_required_commands") or [])
        learning_closure_command = str(metadata.get("execution_learning_closure_command") or "")
        learning_actionable_commands = list(metadata.get("execution_learning_actionable_required_commands") or [])
        if not learning_actionable_commands and learning_required_commands:
            learning_target_run_id = metadata.get("execution_learning_target_run_id")
            learning_closure_command = (
                f"execution learning closure {learning_target_run_id}"
                if learning_target_run_id is not None
                else "execution learning closure"
            )
            learning_evidence_command = (
                f"after-action learning packet {learning_target_run_id}"
                if learning_target_run_id is not None
                else "after-action learning packet"
            )
            for command in learning_required_commands:
                if command == learning_closure_command and "target_after_action_learning_packet" in learning_missing:
                    _append_unique(learning_actionable_commands, [learning_evidence_command])
                elif command != learning_closure_command:
                    _append_unique(learning_actionable_commands, [command])
            _append_unique(learning_actionable_commands, [learning_closure_command])
        learning_next_evidence_command = str(
            metadata.get("execution_learning_next_evidence_command")
            or metadata.get("execution_learning_actionable_next_required_command")
            or (learning_actionable_commands[0] if learning_actionable_commands else "")
        )
        learning_blocks_completion_claim = _metadata_bool(metadata.get("execution_learning_blocks_completion_claim"))
        approval_queue_forecast = list(metadata.get("approval_queue_forecast") or [])
        forecast_new_approvals = _metadata_int(metadata.get("forecast_new_approvals"))
        forecast_reused_approval_ids = list(metadata.get("forecast_reused_approval_ids") or [])
        forecast_queue_before = _metadata_int(metadata.get("forecast_queue_before"))
        forecast_queue_after_if_sent = _metadata_int(metadata.get("forecast_queue_after_if_sent"), forecast_queue_before)
        forecast_queue_delta_if_sent = _metadata_int(metadata.get("forecast_queue_delta_if_sent"), forecast_new_approvals)
        closure_verdict = str(closure_metadata.get("closure_verdict") or "CASE_CLOSURE_UNKNOWN")
        case_closure_ready = _metadata_bool(closure_metadata.get("case_closure_ready"))
        case_closure_blocks_completion_claim = _metadata_bool(
            closure_metadata.get("case_closure_blocks_completion_claim"),
            default=not case_closure_ready,
        )
        closure_proof_queue = list(closure_metadata.get("closure_proof_queue") or [])
        if learning_actionable_commands:
            learning_related_commands = {
                str(command)
                for command in [
                    *learning_required_commands,
                    *learning_actionable_commands,
                    learning_next_required_command,
                    learning_next_evidence_command,
                ]
                if str(command or "").strip()
            }
            learning_anchor_indexes = [
                index
                for index, command in enumerate(closure_proof_queue)
                if str(command) in learning_related_commands
            ]
            insertion_index = min(learning_anchor_indexes) if learning_anchor_indexes else len(closure_proof_queue)
            closure_proof_queue = [
                command
                for command in closure_proof_queue
                if str(command) not in learning_related_commands
            ]
            for offset, command in enumerate(learning_actionable_commands):
                if command and command not in closure_proof_queue:
                    closure_proof_queue.insert(insertion_index + offset, command)
        next_closure_proof_command = str(
            closure_metadata.get("next_closure_proof_command")
            or (closure_proof_queue[0] if closure_proof_queue else "")
            or next_command
        )
        if closure_proof_queue:
            next_closure_proof_command = str(closure_proof_queue[0])
        evidence_preview_source = closure_metadata if "evidence_preview_gate_count" in closure_metadata else metadata
        evidence_preview_latest_handoff = evidence_preview_source.get("evidence_preview_latest_handoff")
        evidence_preview_latest_handoff = (
            evidence_preview_latest_handoff if isinstance(evidence_preview_latest_handoff, dict) else {}
        )
        review_handoff = review_metadata.get("execution_case_review_handoff")
        review_handoff = review_handoff if isinstance(review_handoff, dict) else {}
        review_state = str(review_metadata.get("review_state") or "")
        review_verdict = str(review_metadata.get("verdict") or "")
        review_next_safe_command = str(review_metadata.get("next_safe_command") or "")

        return {
            "found": True,
            "case_id": case_id,
            "request": case["request"],
            "verdict": verdict,
            "mission_state": metadata.get("mission_state", case["mission_state"]),
            "go_no_go": metadata.get("go_no_go", case["go_no_go"]),
            "events": metadata.get("events", 0),
            "pending_approvals": metadata.get("pending_approvals", 0),
            "blockers": metadata.get("blockers", 0),
            "missing_case_proofs": missing_case_proofs,
            "next_case_proof_commands": next_commands,
            "next_case_required_command": next_required_command,
            "next_case_proof_command": next_command,
            "recovery_closure_state": recovery_closure_state,
            "recovery_closure_missing": recovery_closure_missing,
            "recovery_closure_next_required_command": recovery_closure_next_required_command,
            "recovery_closure_required_commands": recovery_closure_required_commands,
            "recovery_closure_proof_queue": recovery_closure_proof_queue,
            "recovery_closure_proof_queue_count": metadata.get("execution_health_recovery_closure_proof_queue_count"),
            "recovery_closure_next_proof_command": recovery_closure_next_proof_command,
            "recovery_closure_blocks_completion_claim": recovery_closure_blocks_completion_claim,
            "learning_state": learning_state,
            "learning_missing": learning_missing,
            "learning_next_required_command": learning_next_required_command,
            "learning_required_commands": learning_required_commands,
            "learning_closure_command": learning_closure_command,
            "learning_proof_queue": learning_required_commands,
            "learning_proof_queue_count": len(learning_required_commands),
            "learning_next_proof_command": learning_next_required_command,
            "learning_actionable_required_commands": learning_actionable_commands,
            "learning_actionable_proof_queue": learning_actionable_commands,
            "learning_actionable_proof_queue_count": len(learning_actionable_commands),
            "learning_actionable_next_required_command": learning_next_evidence_command,
            "learning_actionable_next_proof_command": learning_next_evidence_command,
            "learning_next_evidence_command": learning_next_evidence_command,
            "learning_blocks_completion_claim": learning_blocks_completion_claim,
            "approval_queue_forecast": approval_queue_forecast,
            "forecast_new_approvals": forecast_new_approvals,
            "forecast_reused_approval_ids": forecast_reused_approval_ids,
            "forecast_queue_before": forecast_queue_before,
            "forecast_queue_after_if_sent": forecast_queue_after_if_sent,
            "forecast_queue_delta_if_sent": forecast_queue_delta_if_sent,
            "mission_command_queue": mission_command_queue,
            "mission_command_count": len(mission_command_queue),
            "evidence_preview_gate_count": _metadata_int(evidence_preview_source.get("evidence_preview_gate_count")),
            "evidence_preview_ready_count": _metadata_int(evidence_preview_source.get("evidence_preview_ready_count")),
            "evidence_preview_blocked_count": _metadata_int(evidence_preview_source.get("evidence_preview_blocked_count")),
            "evidence_preview_verdicts": list(evidence_preview_source.get("evidence_preview_verdicts") or []),
            "evidence_preview_latest_verdict": str(evidence_preview_source.get("evidence_preview_latest_verdict") or ""),
            "evidence_preview_latest_event_id": evidence_preview_source.get("evidence_preview_latest_event_id"),
            "evidence_preview_latest_handoff": evidence_preview_latest_handoff,
            "evidence_preview_latest_handoff_present": _metadata_bool(evidence_preview_source.get("evidence_preview_latest_handoff_present")),
            "ready_for_human_review": case_closure_ready,
            "review_state": review_state,
            "review_verdict": review_verdict,
            "review_next_safe_command": review_next_safe_command,
            "review_approval_required": _metadata_bool(review_metadata.get("approval_required")),
            "review_checklist_items": _metadata_int(review_metadata.get("checklist_items")),
            "review_blocker_count": _metadata_int(review_metadata.get("blockers")),
            "review_handoff": review_handoff,
            "review_handoff_present": bool(review_handoff),
            "closure_verdict": closure_verdict,
            "case_closure_ready": case_closure_ready,
            "case_closure_blocks_completion_claim": case_closure_blocks_completion_claim,
            "closure_proof_queue": closure_proof_queue,
            "closure_proof_queue_count": len(closure_proof_queue),
            "next_closure_proof_command": next_closure_proof_command,
        }

    def evidence_ledger(args: dict[str, Any]) -> ToolResult:
        objective = _short(
            args.get("objective") or args.get("goal") or args.get("request") or (
                "Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant."
            ),
            limit=500,
        )
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=20)
        recent_runs = store.recent_tool_runs(limit=30)
        readable_recent_runs, unreadable_recent_run_rows = _readable_tool_run_rows(recent_runs)
        recent_run_names = {_row_text(row, "tool_name") for row in readable_recent_runs if _row_bool(row, "ok")}
        open_tasks = store.list_tasks(status="open", limit=20)
        active_goals = store.list_goals(status="active", limit=20)
        first_pending_approval_id = _first_positive_int_text(pending)
        first_open_task_id = _first_positive_int_text(open_tasks)
        agi_summary = _agi_gate_summary(tool_names)
        selected_agi_readiness = _selected_agi_target_readiness(agi_summary)
        selected_agi_gate_name = selected_agi_readiness["gate_name"]
        selected_agi_closure_commands = selected_agi_readiness["closure_commands"]
        selected_agi_verification_commands = selected_agi_readiness["verification_commands"]
        selected_agi_likely_files = selected_agi_readiness["likely_files"]
        selected_agi_file_integrity = selected_agi_readiness["file_integrity"]
        selected_agi_acceptance_checks = selected_agi_readiness["acceptance_checks"]
        selected_agi_build_ready = selected_agi_readiness["build_ready"]
        selected_agi_target_title = selected_agi_readiness["target_title"]

        evidence_requirements = [
            ("steering", {"harness_cycle_preview", "execution_governor_packet", "dispatch_decision_packet", "planner_gap_packet", "execution_contract", "argument_contract_packet", "execution_readiness_matrix", "execution_mission_control", "save_execution_case", "append_execution_case_evidence", "inspect_execution_case", "execution_case_gate", "execution_case_review_packet", "execution_case_closure_packet", "execution_case_timeline", "execution_runbook", "execution_proof_bundle", "action_readiness_packet", "risk_preflight"}),
            ("pedals", {"tool_search", "tool_detail", "risk_matrix", "capability_map"}),
            ("brakes", {"approval_readiness_packet", "approval_execution_packet", "review_pending_approvals", "approval_history", "safety_status"}),
            ("dashboard", {"status_dashboard", "readiness_report", "jarvis_doctor", "harness_completion_assessment"}),
            ("memory/state", {"memory_tree_summary", "export_state_snapshot", "return_brief", "work_queue"}),
            ("audit", {"recent_tool_runs", "verification_receipt", "runtime_trace_receipt", "execution_audit_gate", "execution_recovery_packet", "execution_health_report", "recovery_closure_checklist", "after_action_learning_packet", "execution_acceptance_gate", "execution_mission_control", "save_execution_case", "append_execution_case_evidence", "inspect_execution_case", "execution_case_gate", "execution_case_review_packet", "execution_case_closure_packet", "execution_case_timeline", "execution_runbook", "execution_proof_bundle", "completion_audit_packet", "task_completion_packet"}),
            ("recovery", {"work_block_checkpoint", "checkpoint_recovery_preview", "checkpoint_recovery_execute", "checkpoint_recovery_receipt", "checkpoint_recovery_followthrough_packet", "autonomy_resume_gate", "autonomy_continuation_execution_packet", "autonomy_step_closure_packet", "autonomy_cycle_ledger"}),
            ("learning", {"learning_review", "after_action_learning_packet", "feedback_actions", "failure_promotion_packet", "failure_implementation_packet", "failure_apply_contract", "failure_patch_receipt_packet", "failure_patch_application_bridge", "failure_patch_completion_gate", "failure_patch_handoff_packet", "failure_patch_closeout_packet", "failure_learning_record_packet", "failure_learning_closure_ledger", "draft_skill_from_session"}),
        ]

        ledger_rows = []
        strong = 0
        partial = 0
        missing = 0
        next_proof_commands = {}
        for name, required_tools in evidence_requirements:
            present = sorted(required_tools & tool_names)
            if len(present) >= 3:
                status = "strong"
                strong += 1
            elif present:
                status = "partial"
                partial += 1
            else:
                status = "missing"
                missing += 1
            next_proof_commands[name] = LANE_PROOF_COMMANDS.get(name, "completion audit")
            ledger_rows.append((name, status, present, sorted(required_tools - tool_names)))

        recent_successes = [row for row in readable_recent_runs if _row_bool(row, "ok")]
        recent_failures = [row for row in readable_recent_runs if not _row_bool(row, "ok")]
        verification_runs = [
            row
            for row in readable_recent_runs
            if _row_text(row, "tool_name") in {"verification_receipt", "runtime_trace_receipt", "verification_packet", "execution_audit_gate", "execution_recovery_packet", "execution_health_report", "recovery_closure_checklist", "after_action_learning_packet", "completion_audit_packet", "task_completion_packet"}
        ]
        after_action_learning_runs = [
            row
            for row in readable_recent_runs
            if _row_text(row, "tool_name") == "after_action_learning_packet"
        ]
        evidence_completion_runs = [
            row
            for row in readable_recent_runs
            if _row_text(row, "tool_name") in {"complete_task_with_evidence", "checkpoint_recovery_execute", "checkpoint_recovery_receipt", "checkpoint_recovery_followthrough_packet", "autonomy_resume_gate", "autonomy_continuation_execution_packet", "autonomy_step_closure_packet", "autonomy_cycle_ledger"}
        ]
        latest_case = _latest_execution_case_proof_state()
        recovery_closure = _execution_health_recovery_closure_snapshot(store, readable_recent_runs)
        execution_health_verification = _execution_health_verification_snapshot(readable_recent_runs)
        learning_debt = _execution_learning_debt_snapshot(readable_recent_runs)
        learning_actionable_commands = list(learning_debt["actionable_required_commands"])

        if pending:
            claim_state = "BLOCKED_BY_PENDING_APPROVAL"
        elif open_tasks:
            claim_state = "OPEN_WORK_REMAINS"
        elif latest_case["found"] and latest_case["case_closure_blocks_completion_claim"]:
            claim_state = "EXECUTION_CASE_CLOSURE_BLOCKED"
        elif recovery_closure["blocks_completion_claim"]:
            claim_state = "EXECUTION_HEALTH_RECOVERY_CLOSURE_BLOCKED"
        elif learning_debt["blocks_completion_claim"]:
            claim_state = "EXECUTION_LEARNING_DEBT_BLOCKED"
        elif missing or partial:
            claim_state = "NEEDS_MORE_REQUIREMENT_EVIDENCE"
        elif agi_summary["real_execution_gap_count"]:
            claim_state = "AGI_GATE_GAPS_REMAIN"
        elif not verification_runs:
            claim_state = "NEEDS_RECENT_VERIFICATION_EVIDENCE"
        else:
            claim_state = "READY_FOR_HUMAN_COMPLETION_REVIEW"

        lines = [
            "Jarvis evidence ledger:",
            "This is the harness proof ledger. It does not claim completion or change state.",
            "",
            f"Objective: {objective}",
            "",
            "Harness evidence lanes:",
        ]
        for name, status, present, missing_tools in ledger_rows:
            lines.extend(
                [
                    f"- {name}: {status}",
                    f"  Present: {', '.join(present) if present else 'none'}",
                    f"  Missing: {', '.join(missing_tools) if missing_tools else 'none'}",
                    f"  Next required command: `{next_proof_commands[name]}`",
                ]
            )

        lines.extend(
            [
                "",
                "AGI-direction gates:",
                f"- strong prototype gates: {agi_summary['strong']} / {agi_summary['gates']}",
                f"- partial prototype gates: {agi_summary['partial']}",
                f"- missing gates: {agi_summary['missing']}",
                f"- real-execution gaps still tracked: {agi_summary['real_execution_gap_count']}",
                f"- selected next build gate: {selected_agi_gate_name or 'none'}",
                f"- selected target: {selected_agi_target_title or 'none'}",
                f"- selected next build command: `{selected_agi_closure_commands[0]}`" if selected_agi_closure_commands else "- selected next build command: none",
                f"- selected closure queue: {', '.join(f'`{command}`' for command in selected_agi_closure_commands) if selected_agi_closure_commands else 'none'}",
                f"- selected focused verification: {', '.join(f'`{command}`' for command in selected_agi_verification_commands) if selected_agi_verification_commands else 'none configured'}",
                f"- selected target integrity: {selected_agi_file_integrity['status']}",
                f"- selected target files checked: {selected_agi_file_integrity['checked']}",
                f"- selected acceptance checks: {len(selected_agi_acceptance_checks)}",
                f"- selected target ready for scoped implementation review: {'yes' if selected_agi_build_ready else 'no'}",
            ]
        )
        for gate in agi_summary["rows"]:
            lines.extend(
                [
                    f"- {gate['gate']}: {gate['status']}",
                    f"  Present: {', '.join(gate['present']) if gate['present'] else 'none'}",
                    f"  Missing evidence: {', '.join(gate['missing_evidence']) if gate['missing_evidence'] else 'none from registry'}",
                    f"  Gap: {gate['real_execution_gap']}",
                    f"  Evidence closure commands: {', '.join(f'`{command}`' for command in gate['evidence_closure_commands'])}",
                ]
            )

        lines.extend(
            [
                "",
                "Current audit state:",
                f"- pending approvals: {len(pending)}",
                f"- open tasks: {len(open_tasks)}",
                f"- active goals: {len(active_goals)}",
                f"- recent tool runs inspected: {len(recent_runs)}",
                f"- unreadable recent tool run rows: {unreadable_recent_run_rows}",
                f"- recent successful runs: {len(recent_successes)}",
                f"- recent failed/blocked runs: {len(recent_failures)}",
                f"- recent verification/audit packets: {len(verification_runs)}",
                f"- recent after-action learning packets: {len(after_action_learning_runs)}",
                f"- recent evidence-completion runs: {len(evidence_completion_runs)}",
                f"- recovery closure state: {recovery_closure['state']}",
                f"- recovery closure ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
                f"- recovery closure missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- recovery closure checklist overview: `{_recovery_closure_checklist_command(recovery_closure)}`" if _recovery_closure_checklist_command(recovery_closure) else "- recovery closure checklist overview: none",
                f"- recovery next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- recovery next required: none",
                f"- recovery closure command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                f"- execution learning debt state: {learning_debt['state']}",
                f"- execution learning debt missing: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
                f"- learning next required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- learning next required: none",
                f"- learning evidence next required: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- learning evidence next required: none",
                f"- execution learning command queue: {', '.join(f'`{command}`' for command in learning_debt['required_commands']) if learning_debt['required_commands'] else 'none'}",
                f"- execution learning actionable queue: {', '.join(f'`{command}`' for command in learning_actionable_commands) if learning_actionable_commands else 'none'}",
            ]
        )

        lines.extend(
            [
                "",
                "Latest execution case proof state:",
                f"- latest case: #{latest_case['case_id']}" if latest_case["found"] else "- latest case: none saved",
                f"- case verdict: {latest_case['verdict']}",
                f"- case closure verdict: {latest_case['closure_verdict']}",
                f"- case closure ready: {latest_case['case_closure_ready']}",
                f"- case blocks completion claim: {latest_case['case_closure_blocks_completion_claim']}",
                f"- case ready for human review: {latest_case['ready_for_human_review']}",
                f"- mission command queue: {latest_case['mission_command_count']} command(s)",
                f"- evidence preflight events: {latest_case['evidence_preview_gate_count']}",
                f"- evidence preflight ready events: {latest_case['evidence_preview_ready_count']}",
                f"- latest evidence preflight verdict: {latest_case['evidence_preview_latest_verdict'] or 'none'}",
                f"- missing case proofs: {', '.join(latest_case['missing_case_proofs']) if latest_case['missing_case_proofs'] else 'none'}",
                f"- next case required command: `{latest_case['next_case_required_command']}`",
                f"- next case closure command: `{latest_case['next_closure_proof_command']}`",
                f"- case closure proof queue: {', '.join(f'`{command}`' for command in latest_case['closure_proof_queue']) if latest_case['closure_proof_queue'] else 'none'}",
                f"- case approval forecast: {latest_case['forecast_new_approvals']} new, {len(latest_case['forecast_reused_approval_ids'])} reused, queue after {latest_case['forecast_queue_after_if_sent']}",
                f"- case recovery closure state: {latest_case['recovery_closure_state']}",
                f"- case recovery closure missing: {', '.join(latest_case['recovery_closure_missing']) if latest_case['recovery_closure_missing'] else 'none'}",
                f"- case recovery closure next: `{latest_case['recovery_closure_next_required_command']}`" if latest_case["recovery_closure_next_required_command"] else "- case recovery closure next: none",
                f"- case execution learning state: {latest_case['learning_state']}",
                f"- case execution learning missing: {', '.join(latest_case['learning_missing']) if latest_case['learning_missing'] else 'none'}",
                f"- case execution learning next: `{latest_case['learning_next_required_command']}`" if latest_case["learning_next_required_command"] else "- case execution learning next: none",
            ]
        )

        if pending:
            if first_pending_approval_id:
                first_tool_name = next((_row_text(row, "tool_name", "unknown") for row in pending if _row_positive_int_text(row) == first_pending_approval_id), "unknown")
                lines.append(f"- first approval blocker: approval #{first_pending_approval_id} {first_tool_name} -> `approval readiness {first_pending_approval_id}` then `approval packet {first_pending_approval_id}` then `approval chain proof {first_pending_approval_id}`")
            else:
                lines.append("- first approval blocker: pending approval row missing a readable id -> `approval review`")
        if open_tasks:
            if first_open_task_id:
                first_task_body = next((_row_text(row, "body") for row in open_tasks if _row_positive_int_text(row) == first_open_task_id), "")
                lines.append(f"- first open task: #{first_open_task_id} {_short(first_task_body)}")
            else:
                lines.append("- first open task: unreadable task row -> `task board`")
        if verification_runs:
            first = verification_runs[0]
            lines.append(f"- newest verification evidence: run #{_row_positive_int_text(first) or 'latest'} {_row_text(first, 'tool_name', 'unknown')}")
        if evidence_completion_runs:
            first = evidence_completion_runs[0]
            lines.append(f"- newest completion evidence: run #{_row_positive_int_text(first) or 'latest'} {_row_text(first, 'tool_name', 'unknown')}")

        lines.extend(
            [
                "",
                f"Claim state: {claim_state}",
                "",
                "How to use this ledger:",
                "- If a lane is partial or missing, build the missing harness capability before claiming that lane.",
                "- If a task is done, use `task completion packet <id>` and then `complete task <id> with evidence: ...`.",
                "- If a tool ran, use `verification receipt <run id>` before treating it as proof.",
                "- If a risky action is involved, approval evidence must be linked before Jarvis counts it as complete.",
                "",
                "Boundary:",
                "- This ledger is read-only. It does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, speak, complete tasks, or queue approvals.",
            ]
        )

        ledger_metadata = dict(
                objective=objective,
                lanes=len(evidence_requirements),
                strong_evidence=strong,
                partial_evidence=partial,
                missing_evidence=missing,
                pending_approvals=len(pending),
                open_tasks=len(open_tasks),
                active_goals=len(active_goals),
                recent_runs=len(recent_runs),
                readable_recent_runs=len(readable_recent_runs),
                unreadable_recent_tool_run_rows=unreadable_recent_run_rows,
                recent_successes=len(recent_successes),
                recent_failures=len(recent_failures),
                verification_runs=len(verification_runs),
                after_action_learning_runs=len(after_action_learning_runs),
                evidence_completion_runs=len(evidence_completion_runs),
                execution_health_recovery_closure_state=recovery_closure["state"],
                execution_health_recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                execution_health_recovery_closure_missing=recovery_closure["missing"],
                execution_health_recovery_closure_missing_count=recovery_closure["missing_count"],
                execution_health_recovery_closure_required_commands=recovery_closure["required_commands"],
                execution_health_recovery_closure_next_required_command=recovery_closure["next_required_command"],
                execution_health_recovery_closure_checklist_command=_recovery_closure_checklist_command(recovery_closure),
                execution_health_recovery_closure_should_open_checklist=bool(_recovery_closure_checklist_command(recovery_closure)),
                execution_health_recovery_closure_proof_queue=recovery_closure["proof_queue"],
                execution_health_recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
                execution_health_recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
                execution_health_recovery_closure_blocks_completion_claim=recovery_closure["blocks_completion_claim"],
                execution_health_recovery_closure_target_run_id=recovery_closure["target_run_id"],
                execution_health_recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                execution_health_recovery_closure_target_verification_receipts=recovery_closure["target_verification_receipts"],
                execution_health_recovery_closure_target_recovery_packets=recovery_closure["target_recovery_packets"],
                execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure["target_after_action_learning_packets"],
                execution_health_verification_coverage_state=execution_health_verification["state"],
                execution_health_verification_recent_tool_runs=execution_health_verification["recent_tool_runs"],
                execution_health_verification_recent_action_runs=execution_health_verification["recent_action_runs"],
                execution_health_verification_recent_verification_runs=execution_health_verification["recent_verification_runs"],
                execution_health_verification_gap_count=execution_health_verification["gap_count"],
                execution_health_verification_first_gap=execution_health_verification["first_gap"],
                execution_health_verification_proof_queue=execution_health_verification["proof_queue"],
                execution_health_verification_proof_queue_count=execution_health_verification["proof_queue_count"],
                execution_health_verification_proof_queue_preview=execution_health_verification["proof_queue_preview"],
                execution_health_verification_proof_queue_preview_count=execution_health_verification["proof_queue_preview_count"],
                execution_health_verification_proof_queue_remaining_count=execution_health_verification["proof_queue_remaining_count"],
                execution_health_verification_next_required_command=execution_health_verification["next_required_command"],
                execution_health_verification_next_proof_command=execution_health_verification["next_proof_command"],
                execution_health_verification_blocks_completion_claim=execution_health_verification["blocks_completion_claim"],
                execution_learning_state=learning_debt["state"],
                execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
                execution_learning_recent_action_runs=learning_debt["recent_action_runs"],
                execution_learning_failed_or_blocked_action_runs=learning_debt["failed_or_blocked_action_runs"],
                execution_learning_recent_verification_runs=learning_debt["recent_verification_runs"],
                execution_learning_recent_recovery_runs=learning_debt["recent_recovery_runs"],
                execution_learning_recent_after_action_learning_runs=learning_debt["recent_after_action_learning_runs"],
                execution_learning_target_run_id=learning_debt["target_run_id"],
                execution_learning_target_tool_name=learning_debt["target_tool_name"],
                execution_learning_target_after_action_learning_packets=learning_debt["target_after_action_learning_packets"],
                execution_learning_missing=learning_debt["missing"],
                execution_learning_missing_count=learning_debt["missing_count"],
                execution_learning_required_commands=learning_debt["required_commands"],
                execution_learning_next_required_command=learning_debt["next_required_command"],
                execution_learning_proof_queue=learning_debt["proof_queue"],
                execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
                execution_learning_next_proof_command=learning_debt["next_proof_command"],
                execution_learning_evidence_command=learning_debt["learning_evidence_command"],
                execution_learning_after_action_learning_command=learning_debt["after_action_learning_command"],
                execution_learning_actionable_required_commands=learning_actionable_commands,
                execution_learning_actionable_required_command_count=len(learning_actionable_commands),
                execution_learning_actionable_proof_queue=learning_actionable_commands,
                execution_learning_actionable_proof_queue_count=len(learning_actionable_commands),
                execution_learning_actionable_next_required_command=learning_debt["actionable_next_required_command"],
                execution_learning_actionable_next_proof_command=learning_debt["actionable_next_proof_command"],
                execution_learning_next_evidence_command=learning_debt["next_evidence_command"],
                latest_execution_case_found=latest_case["found"],
                latest_execution_case_id=latest_case["case_id"],
                latest_execution_case_verdict=latest_case["verdict"],
                latest_execution_case_ready=latest_case["ready_for_human_review"],
                latest_execution_case_review_state=latest_case["review_state"],
                latest_execution_case_review_verdict=latest_case["review_verdict"],
                latest_execution_case_review_next_safe_command=latest_case["review_next_safe_command"],
                latest_execution_case_review_approval_required=latest_case["review_approval_required"],
                latest_execution_case_review_checklist_items=latest_case["review_checklist_items"],
                latest_execution_case_review_blocker_count=latest_case["review_blocker_count"],
                latest_execution_case_review_handoff=latest_case["review_handoff"],
                latest_execution_case_review_handoff_present=latest_case["review_handoff_present"],
                latest_execution_case_closure_verdict=latest_case["closure_verdict"],
                latest_execution_case_closure_ready=latest_case["case_closure_ready"],
                latest_execution_case_closure_blocks_completion_claim=latest_case["case_closure_blocks_completion_claim"],
                latest_execution_case_closure_proof_queue=latest_case["closure_proof_queue"],
                latest_execution_case_closure_proof_queue_count=latest_case["closure_proof_queue_count"],
                latest_execution_case_next_closure_proof_command=latest_case["next_closure_proof_command"],
                latest_execution_case_missing_proofs=latest_case["missing_case_proofs"],
                latest_execution_case_next_required_command=latest_case["next_case_required_command"],
                latest_execution_case_next_proof_command=latest_case["next_case_proof_command"],
                latest_execution_case_next_proof_commands=latest_case["next_case_proof_commands"],
                latest_execution_case_mission_command_queue=latest_case["mission_command_queue"],
                latest_execution_case_mission_command_count=latest_case["mission_command_count"],
                latest_execution_case_evidence_preview_gate_count=latest_case["evidence_preview_gate_count"],
                latest_execution_case_evidence_preview_ready_count=latest_case["evidence_preview_ready_count"],
                latest_execution_case_evidence_preview_blocked_count=latest_case["evidence_preview_blocked_count"],
                latest_execution_case_evidence_preview_verdicts=latest_case["evidence_preview_verdicts"],
                latest_execution_case_evidence_preview_latest_verdict=latest_case["evidence_preview_latest_verdict"],
                latest_execution_case_evidence_preview_latest_event_id=latest_case["evidence_preview_latest_event_id"],
                latest_execution_case_evidence_preview_latest_handoff=latest_case["evidence_preview_latest_handoff"],
                latest_execution_case_evidence_preview_latest_handoff_present=latest_case["evidence_preview_latest_handoff_present"],
                latest_execution_case_evidence_preview_next_command=str(
                    (latest_case["evidence_preview_latest_handoff"] or {}).get("next_command")
                    or (latest_case["evidence_preview_latest_handoff"] or {}).get("append_command")
                    or ""
                ),
                latest_execution_case_evidence_preview_receipt_id=str(
                    (latest_case["evidence_preview_latest_handoff"] or {}).get("receipt_id")
                    or (latest_case["evidence_preview_latest_handoff"] or {}).get("inferred_receipt_id")
                    or ""
                ),
                latest_execution_case_evidence_preview_receipt_kind=str(
                    (latest_case["evidence_preview_latest_handoff"] or {}).get("receipt_kind")
                    or (latest_case["evidence_preview_latest_handoff"] or {}).get("inferred_receipt_kind")
                    or ""
                ),
                latest_execution_case_evidence_preview_receipt_target_status=str(
                    (latest_case["evidence_preview_latest_handoff"] or {}).get("receipt_target_status") or ""
                ),
                latest_execution_case_approval_queue_forecast=latest_case["approval_queue_forecast"],
                latest_execution_case_forecast_new_approvals=latest_case["forecast_new_approvals"],
                latest_execution_case_forecast_reused_approval_ids=latest_case["forecast_reused_approval_ids"],
                latest_execution_case_forecast_queue_before=latest_case["forecast_queue_before"],
                latest_execution_case_forecast_queue_after_if_sent=latest_case["forecast_queue_after_if_sent"],
                latest_execution_case_forecast_queue_delta_if_sent=latest_case["forecast_queue_delta_if_sent"],
                latest_execution_case_recovery_closure_state=latest_case["recovery_closure_state"],
                latest_execution_case_recovery_closure_missing=latest_case["recovery_closure_missing"],
                latest_execution_case_recovery_closure_next_required_command=latest_case["recovery_closure_next_required_command"],
                latest_execution_case_recovery_closure_required_commands=latest_case["recovery_closure_required_commands"],
                latest_execution_case_recovery_closure_proof_queue=latest_case["recovery_closure_proof_queue"],
                latest_execution_case_recovery_closure_proof_queue_count=latest_case["recovery_closure_proof_queue_count"],
                latest_execution_case_recovery_closure_next_proof_command=latest_case["recovery_closure_next_proof_command"],
                latest_execution_case_recovery_closure_blocks_completion_claim=latest_case["recovery_closure_blocks_completion_claim"],
                latest_execution_case_learning_state=latest_case["learning_state"],
                latest_execution_case_learning_missing=latest_case["learning_missing"],
                latest_execution_case_learning_next_required_command=latest_case["learning_next_required_command"],
                latest_execution_case_learning_required_commands=latest_case["learning_required_commands"],
                latest_execution_case_learning_proof_queue=latest_case["learning_proof_queue"],
                latest_execution_case_learning_proof_queue_count=latest_case["learning_proof_queue_count"],
                latest_execution_case_learning_next_proof_command=latest_case["learning_next_proof_command"],
                latest_execution_case_learning_blocks_completion_claim=latest_case["learning_blocks_completion_claim"],
                next_proof_commands=next_proof_commands,
                agi_gates=agi_summary["gates"],
                agi_strong_gates=agi_summary["strong"],
                agi_partial_gates=agi_summary["partial"],
                agi_missing_gates=agi_summary["missing"],
                agi_gate_statuses=agi_summary["gate_statuses"],
            agi_missing_evidence_by_gate=agi_summary["missing_evidence_by_gate"],
            agi_real_execution_gaps_by_gate=agi_summary["real_execution_gaps_by_gate"],
            agi_real_execution_gap_count=agi_summary["real_execution_gap_count"],
            real_execution_gap_count=agi_summary["real_execution_gap_count"],
            selected_real_execution_gap=selected_agi_gate_name,
            selected_real_execution_gap_gate=selected_agi_gate_name,
            selected_real_execution_gap_detail=agi_summary["real_execution_gaps_by_gate"].get(selected_agi_gate_name, ""),
            agi_next_moves_by_gate=agi_summary["next_moves_by_gate"],
                agi_evidence_closure_commands_by_gate=agi_summary["evidence_closure_commands_by_gate"],
                agi_focused_verification_by_gate=agi_summary["focused_verification_by_gate"],
                agi_next_gate=selected_agi_gate_name,
                agi_next_target_title=selected_agi_target_title,
                agi_next_build_command=selected_agi_closure_commands[0] if selected_agi_closure_commands else "",
                agi_next_evidence_closure_commands=selected_agi_closure_commands,
                agi_next_evidence_closure_command_count=len(selected_agi_closure_commands),
                agi_next_focused_verification_commands=selected_agi_verification_commands,
                agi_next_focused_verification_command_count=len(selected_agi_verification_commands),
                agi_next_likely_files=selected_agi_likely_files,
                agi_next_likely_file_count=len(selected_agi_likely_files),
                agi_next_target_file_integrity_status=selected_agi_file_integrity["status"],
                agi_next_target_files_checked=selected_agi_file_integrity["checked"],
                agi_next_target_files_exist=selected_agi_file_integrity["all_exist"],
                agi_next_missing_target_files=selected_agi_file_integrity["missing"],
                agi_next_missing_target_file_count=selected_agi_file_integrity["missing_count"],
                agi_next_target_integrity_blocks_start=not selected_agi_file_integrity["all_exist"],
                agi_next_acceptance_checks=selected_agi_acceptance_checks,
                agi_next_acceptance_check_count=len(selected_agi_acceptance_checks),
                **_agi_acceptance_gap_metadata({"agi_next_acceptance_checks": selected_agi_acceptance_checks}),
                agi_next_build_packet_ready_for_review=selected_agi_build_ready,
                **_agi_focus_selection_metadata(selected_agi_readiness),
                claim_state=claim_state,
                completion_claim_ready=claim_state == "READY_FOR_HUMAN_COMPLETION_REVIEW",
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
        )
        evidence_ledger_handoff = _safe_metadata(
            source="evidence_ledger",
            **ledger_metadata,
        )

        return ToolResult(
            "evidence_ledger",
            True,
            "\n".join(lines),
            _safe_metadata(
                **ledger_metadata,
                evidence_ledger_handoff=evidence_ledger_handoff,
            ),
        )

    def completion_claim_gate(args: dict[str, Any]) -> ToolResult:
        objective = _short(
            args.get("objective") or args.get("goal") or args.get("request") or (
                "Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant."
            ),
            limit=500,
        )
        proposed_claim = _short(args.get("claim") or args.get("message") or args.get("statement"), limit=500)
        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=20)
        recent_runs = store.recent_tool_runs(limit=30)
        readable_recent_runs, unreadable_recent_run_rows = _readable_tool_run_rows(recent_runs)
        recent_run_names = {_row_text(row, "tool_name") for row in readable_recent_runs if _row_bool(row, "ok")}
        open_tasks = store.list_tasks(status="open", limit=20)
        active_goals = store.list_goals(status="active", limit=20)
        first_pending_approval_id = _first_positive_int_text(pending)
        first_open_task_id = _first_positive_int_text(open_tasks)
        agi_summary = _agi_gate_summary(tool_names)
        selected_agi_readiness = _selected_agi_target_readiness(agi_summary)
        selected_agi_gate_name = selected_agi_readiness["gate_name"]
        selected_agi_closure_commands = selected_agi_readiness["closure_commands"]
        selected_agi_verification_commands = selected_agi_readiness["verification_commands"]
        selected_agi_likely_files = selected_agi_readiness["likely_files"]
        selected_agi_file_integrity = selected_agi_readiness["file_integrity"]
        selected_agi_acceptance_checks = selected_agi_readiness["acceptance_checks"]
        selected_agi_build_ready = selected_agi_readiness["build_ready"]
        selected_agi_target_title = selected_agi_readiness["target_title"]
        storage_readiness = _storage_readiness_snapshot()

        requirements = [
            ("command-first interface", {"status_dashboard", "chat_loop_preview", "voice_command_lifecycle"}),
            ("routing and tool orchestration", {"execution_governor_packet", "dispatch_decision_packet", "planner_gap_packet", "execution_contract", "argument_contract_packet", "execution_mission_control", "save_execution_case", "append_execution_case_evidence", "inspect_execution_case", "execution_case_gate", "execution_case_review_packet", "execution_case_closure_packet", "execution_case_timeline", "execution_runbook", "execution_acceptance_gate", "execution_readiness_matrix", "verification_packet", "action_rehearsal", "risk_matrix", "specialist_orchestration_packet", "specialist_route_quality", "specialist_execution_readiness", "specialist_handoff_receipt", "specialist_model_draft", "specialist_action_proposal_contract", "specialist_tool_dry_run_packet", "specialist_proposal_completion_gate", "specialist_execution_handoff_packet", "specialist_post_run_closure_packet", "specialist_cycle_ledger"}),
            ("memory and state", {"memory_tree_summary", "export_state_snapshot", "work_queue", "return_brief"}),
            ("approval gates and audit", {"approval_readiness_packet", "approval_execution_packet", "review_pending_approvals", "approval_history", "recent_tool_runs", "runtime_trace_receipt", "execution_audit_gate", "execution_recovery_packet", "execution_health_report", "recovery_closure_checklist", "after_action_learning_packet", "execution_acceptance_gate"}),
            ("verification and recovery", {"verification_packet", "verification_receipt", "checkpoint_recovery_preview", "checkpoint_recovery_receipt", "checkpoint_recovery_followthrough_packet", "autonomy_resume_gate", "autonomy_continuation_execution_packet", "autonomy_step_closure_packet", "autonomy_cycle_ledger"}),
            ("diagnostics and observability", {"readiness_report", "build_progress_report", "build_delta_report", "harness_completion_assessment"}),
            ("learning loop", {"learning_review", "after_action_learning_packet", "feedback_actions", "failure_promotion_packet", "failure_implementation_packet", "failure_apply_contract", "failure_patch_receipt_packet", "failure_patch_application_bridge", "failure_patch_completion_gate", "failure_patch_handoff_packet", "failure_patch_closeout_packet", "failure_learning_record_packet", "failure_learning_closure_ledger", "draft_skill_from_session"}),
        ]
        missing_lanes = []
        partial_lanes = []
        strong_lanes = []
        unavailable_lanes = []
        lane_rows = []
        next_proof_commands = {}
        for name, required in requirements:
            available = sorted(required & tool_names)
            evidence = sorted(required & recent_run_names)
            if len(evidence) >= 3:
                strong_lanes.append(name)
                status = "strong"
            elif evidence:
                partial_lanes.append(name)
                status = "partial"
            elif available:
                partial_lanes.append(name)
                status = "available_without_recent_evidence"
            else:
                missing_lanes.append(name)
                unavailable_lanes.append(name)
                status = "missing"
            next_proof_commands[name] = LANE_PROOF_COMMANDS.get(name, "completion audit")
            lane_rows.append((name, status, available, evidence))

        verification_runs = [
            row
            for row in readable_recent_runs
            if _row_text(row, "tool_name") in {"verification_receipt", "runtime_trace_receipt", "verification_packet", "execution_audit_gate", "execution_recovery_packet", "execution_health_report", "recovery_closure_checklist", "after_action_learning_packet", "execution_acceptance_gate", "completion_audit_packet", "task_completion_packet", "evidence_ledger"}
        ]
        after_action_learning_runs = [
            row
            for row in readable_recent_runs
            if _row_text(row, "tool_name") == "after_action_learning_packet"
        ]
        failed_runs = [row for row in readable_recent_runs if not _row_bool(row, "ok")]
        first_failed_run_id = _first_positive_int_text(failed_runs)
        evidence_completion_runs = [
            row
            for row in readable_recent_runs
            if _row_text(row, "tool_name") in {"complete_task_with_evidence", "checkpoint_recovery_receipt", "checkpoint_recovery_followthrough_packet", "autonomy_resume_gate", "autonomy_continuation_execution_packet", "autonomy_step_closure_packet", "autonomy_cycle_ledger"}
        ]
        latest_case = _latest_execution_case_proof_state()
        recovery_closure = _execution_health_recovery_closure_snapshot(store, readable_recent_runs)
        execution_health_verification = _execution_health_verification_snapshot(readable_recent_runs)
        learning_debt = _execution_learning_debt_snapshot(readable_recent_runs)

        blockers = []

        def add_blocker(blocker: str) -> None:
            if blocker and blocker not in blockers:
                blockers.append(blocker)

        if pending:
            add_blocker(f"{len(pending)} pending approval(s) must be reviewed or resolved")
        if unreadable_recent_run_rows:
            add_blocker(f"{unreadable_recent_run_rows} unreadable recent tool run row(s) need storage/audit review")
        if open_tasks:
            add_blocker(f"{len(open_tasks)} open task(s) remain")
        if latest_case["found"] and latest_case["case_closure_blocks_completion_claim"]:
            add_blocker(
                f"latest execution case #{latest_case['case_id']} has open closure proof debt ({latest_case['closure_verdict']})"
            )
        if missing_lanes:
            add_blocker("missing requirement evidence: " + ", ".join(missing_lanes))
        if partial_lanes:
            add_blocker("partial requirement evidence: " + ", ".join(partial_lanes))
        if agi_summary["real_execution_gap_count"]:
            add_blocker(f"{agi_summary['real_execution_gap_count']} AGI-direction real-execution gate gap(s) remain")
        if storage_readiness["blocks_completion_claim"]:
            add_blocker(str(storage_readiness["blocker"]))
        if recovery_closure["blocks_completion_claim"]:
            add_blocker(
                "execution health recovery closure is incomplete: "
                + ", ".join(recovery_closure["missing"] or [str(recovery_closure["state"])])
            )
        if execution_health_verification["blocks_completion_claim"]:
            add_blocker("execution health verification coverage is missing")
        if learning_debt["blocks_completion_claim"]:
            add_blocker(
                "execution learning debt is incomplete: "
                + ", ".join(learning_debt["missing"] or [str(learning_debt["state"])])
            )
        if not verification_runs:
            add_blocker("no recent verification/audit packet is available")
        if failed_runs:
            add_blocker(f"{len(failed_runs)} recent failed or blocked run(s) need review")

        completion_proof_queue: list[str] = []

        def append_proof_command(command: str) -> None:
            command = str(command or "").strip()
            if command and command not in completion_proof_queue:
                completion_proof_queue.append(command)

        if pending:
            if first_pending_approval_id:
                append_proof_command(f"approval readiness {first_pending_approval_id}")
                append_proof_command(f"approval packet {first_pending_approval_id}")
                append_proof_command(f"approval chain proof {first_pending_approval_id}")
                append_proof_command(f"verification receipt <approved run id from approval chain proof {first_pending_approval_id}>")
            else:
                append_proof_command("approval review")
        if open_tasks:
            append_proof_command(f"task completion packet {first_open_task_id}" if first_open_task_id else "task board")
        if latest_case["found"] and latest_case["case_closure_blocks_completion_claim"]:
            append_proof_command(str(latest_case["next_closure_proof_command"]))
            append_proof_command(str(latest_case["next_case_proof_command"]))
            for command in latest_case["closure_proof_queue"]:
                append_proof_command(str(command))
            case_proof_commands = latest_case["next_case_proof_commands"]
            if isinstance(case_proof_commands, dict):
                case_proof_commands = case_proof_commands.values()
            for command in case_proof_commands:
                append_proof_command(str(command))
            for command in latest_case["recovery_closure_required_commands"]:
                append_proof_command(command)
            for command in latest_case["learning_required_commands"]:
                append_proof_command(command)
        for command in recovery_closure["required_commands"]:
            append_proof_command(command)
        learning_closure_command = (
            f"execution learning closure {learning_debt['target_run_id']}"
            if learning_debt["blocks_completion_claim"] and learning_debt["target_run_id"] is not None
            else "execution learning closure"
            if learning_debt["blocks_completion_claim"]
            else ""
        )
        if learning_closure_command:
            append_proof_command(learning_closure_command)
        for command in learning_debt["required_commands"]:
            append_proof_command(command)
        learning_actionable_commands = list(learning_debt["actionable_required_commands"])
        if learning_actionable_commands:
            learning_command_set = {command for command in learning_actionable_commands if command}
            if learning_closure_command:
                learning_command_set.add(learning_closure_command)
            anchor_indexes = [
                index
                for index, command in enumerate(completion_proof_queue)
                if command in learning_command_set
            ]
            insertion_index = min(anchor_indexes) if anchor_indexes else len(completion_proof_queue)
            completion_proof_queue = [
                command
                for command in completion_proof_queue
                if command not in learning_command_set
            ]
            for offset, command in enumerate(learning_actionable_commands):
                if command and command not in completion_proof_queue:
                    completion_proof_queue.insert(insertion_index + offset, command)
        if failed_runs:
            append_proof_command(f"verification receipt {first_failed_run_id}" if first_failed_run_id else "verification receipt latest")
            append_proof_command("execution health report")
        if not verification_runs:
            append_proof_command("verification receipt latest")
            append_proof_command("completion audit")
            append_proof_command("evidence ledger")
        if agi_summary["real_execution_gap_count"]:
            append_proof_command("agi gates")
            for command in selected_agi_closure_commands:
                append_proof_command(command)
            for gate in agi_summary["rows"]:
                for command in gate["evidence_closure_commands"]:
                    append_proof_command(command)
        if storage_readiness["next_commands"]:
            storage_prefix = [
                str(command).strip()
                for command in storage_readiness["next_commands"]
                if str(command).strip()
            ]
            prefixed_commands = set(storage_prefix)
            completion_proof_queue = [
                command
                for command in completion_proof_queue
                if command not in prefixed_commands
            ]
            completion_proof_queue = storage_prefix + completion_proof_queue

        allowed_to_claim = not blockers
        verdict = "CLAIM_REVIEW_READY" if allowed_to_claim else "CLAIM_BLOCKED"
        safe_claim = (
            "I can say this is ready for human completion review, backed by recent audit evidence."
            if allowed_to_claim
            else "I should not claim this is complete yet; the evidence gate is still blocking completion."
        )

        lines = [
            "Jarvis completion claim gate:",
            "This is a stop-check before Jarvis says an objective is done. It is read-only and does not mark anything complete.",
            "",
            f"Objective: {objective}",
        ]
        if proposed_claim:
            lines.append(f"Proposed claim: {proposed_claim}")
        lines.extend(
            [
                "",
                "Evidence standard:",
                "- A completion claim needs clear requirement evidence, no pending approvals, no open work for the objective, recent verification/audit packets, and no unreviewed failed runs.",
                "- The gate blocks optimistic summaries when proof is incomplete.",
                "",
                "Requirement lanes:",
                f"- strong: {len(strong_lanes)} ({', '.join(strong_lanes) if strong_lanes else 'none'})",
                f"- partial: {len(partial_lanes)} ({', '.join(partial_lanes) if partial_lanes else 'none'})",
                f"- missing: {len(missing_lanes)} ({', '.join(missing_lanes) if missing_lanes else 'none'})",
                "",
                "Lane evidence details:",
                *[
                    f"- {name}: {status}; evidence runs: {', '.join(evidence) if evidence else 'none'}; available tools: {', '.join(available[:5]) if available else 'none'}; next proof: `{next_proof_commands[name]}`"
                    for name, status, available, evidence in lane_rows
                ],
                "",
                "AGI-direction gate details:",
                f"- strong prototype gates: {agi_summary['strong']} / {agi_summary['gates']}",
                f"- partial prototype gates: {agi_summary['partial']}",
                f"- missing gates: {agi_summary['missing']}",
                f"- real-execution gaps still tracked: {agi_summary['real_execution_gap_count']}",
                f"- selected next build gate: {selected_agi_gate_name or 'none'}",
                f"- selected target: {selected_agi_target_title or 'none'}",
                f"- selected next build command: `{selected_agi_closure_commands[0]}`" if selected_agi_closure_commands else "- selected next build command: none",
                f"- selected closure queue: {', '.join(f'`{command}`' for command in selected_agi_closure_commands) if selected_agi_closure_commands else 'none'}",
                f"- selected focused verification: {', '.join(f'`{command}`' for command in selected_agi_verification_commands) if selected_agi_verification_commands else 'none configured'}",
                f"- selected target integrity: {selected_agi_file_integrity['status']}",
                f"- selected target files checked: {selected_agi_file_integrity['checked']}",
                f"- selected acceptance checks: {len(selected_agi_acceptance_checks)}",
                f"- selected target ready for scoped implementation review: {'yes' if selected_agi_build_ready else 'no'}",
                *[
                    f"- {gate['gate']}: {gate['status']}; missing evidence: {', '.join(gate['missing_evidence']) if gate['missing_evidence'] else 'none from registry'}; gap: {gate['real_execution_gap']}"
                    for gate in agi_summary["rows"]
                ],
                "",
                "AGI gate closure commands:",
                *[
                    f"- {gate['gate']}: {', '.join(f'`{command}`' for command in gate['evidence_closure_commands'])}"
                    for gate in agi_summary["rows"]
                ],
                "",
                "Current proof state:",
                f"- pending approvals: {len(pending)}",
                f"- open tasks: {len(open_tasks)}",
                f"- active goals: {len(active_goals)}",
                f"- recent verification/audit packets: {len(verification_runs)}",
                f"- recent after-action learning packets: {len(after_action_learning_runs)}",
                f"- recent evidence-completion runs: {len(evidence_completion_runs)}",
                f"- recent failed/blocked runs: {len(failed_runs)}",
                "- execution health surface: `execution health report` (`execution_health_report`)",
                f"- recovery closure state: {recovery_closure['state']}",
                f"- recovery closure ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
                f"- recovery closure missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- recovery next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- recovery next required: none",
                f"- recovery closure command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                f"- execution learning debt state: {learning_debt['state']}",
                f"- execution learning debt missing: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
                f"- learning evidence next required: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- learning evidence next required: none",
                f"- learning actionable next required: `{learning_actionable_commands[0]}`" if learning_actionable_commands else "- learning actionable next required: none",
                f"- learning actionable proof alias: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- learning actionable proof alias: none",
                f"- execution learning closure command: `{learning_closure_command}`" if learning_closure_command else "- execution learning closure command: none",
                f"- learning next required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- learning next required: none",
                f"- execution learning command queue: {', '.join(f'`{command}`' for command in learning_debt['required_commands']) if learning_debt['required_commands'] else 'none'}",
                f"- execution learning actionable queue: {', '.join(f'`{command}`' for command in learning_actionable_commands) if learning_actionable_commands else 'none'}",
                f"- storage fallback active: {'yes' if storage_readiness['active'] else 'no'}",
                f"- storage readiness blocks completion claim: {'yes' if storage_readiness['blocks_completion_claim'] else 'no'}",
                f"- storage readiness reason: {storage_readiness['reason'] or 'none'}",
                f"- storage database route: {storage_readiness['db_path_display'] or 'configured storage'}",
                f"- storage notes route: {storage_readiness['vault_path_display'] or 'configured vault'}",
                f"- storage next operator action: {storage_readiness['recovery_next_operator_action'] or 'none'}",
                f"- storage recovery commands: {', '.join(f'`{command}`' for command in storage_readiness['next_commands']) if storage_readiness['next_commands'] else 'none'}",
                f"- completion proof queue: {', '.join(f'`{command}`' for command in completion_proof_queue) if completion_proof_queue else 'none'}",
                "",
                "Completion proof queue:",
                *(
                    [f"- `{command}`" for command in completion_proof_queue]
                    if completion_proof_queue
                    else ["- none; this gate found no ordered proof commands"]
                ),
                "",
                "Latest execution case proof state:",
                f"- latest case: #{latest_case['case_id']}" if latest_case["found"] else "- latest case: none saved",
                f"- case verdict: {latest_case['verdict']}",
                f"- case closure verdict: {latest_case['closure_verdict']}",
                f"- case closure ready: {latest_case['case_closure_ready']}",
                f"- case blocks completion claim: {latest_case['case_closure_blocks_completion_claim']}",
                f"- case ready for human review: {latest_case['ready_for_human_review']}",
                f"- mission command queue: {latest_case['mission_command_count']} command(s)",
                f"- evidence preflight events: {latest_case['evidence_preview_gate_count']}",
                f"- evidence preflight ready events: {latest_case['evidence_preview_ready_count']}",
                f"- latest evidence preflight verdict: {latest_case['evidence_preview_latest_verdict'] or 'none'}",
                f"- missing case proofs: {', '.join(latest_case['missing_case_proofs']) if latest_case['missing_case_proofs'] else 'none'}",
                f"- next case required command: `{latest_case['next_case_required_command']}`",
                f"- next case closure command: `{latest_case['next_closure_proof_command']}`",
                f"- case closure proof queue: {', '.join(f'`{command}`' for command in latest_case['closure_proof_queue']) if latest_case['closure_proof_queue'] else 'none'}",
                f"- case approval forecast: {latest_case['forecast_new_approvals']} new, {len(latest_case['forecast_reused_approval_ids'])} reused, queue after {latest_case['forecast_queue_after_if_sent']}",
                f"- case recovery closure state: {latest_case['recovery_closure_state']}",
                f"- case recovery closure missing: {', '.join(latest_case['recovery_closure_missing']) if latest_case['recovery_closure_missing'] else 'none'}",
                f"- case recovery closure next: `{latest_case['recovery_closure_next_required_command']}`" if latest_case["recovery_closure_next_required_command"] else "- case recovery closure next: none",
                f"- case execution learning state: {latest_case['learning_state']}",
                f"- case execution learning missing: {', '.join(latest_case['learning_missing']) if latest_case['learning_missing'] else 'none'}",
                f"- case execution learning next: `{latest_case['learning_next_required_command']}`" if latest_case["learning_next_required_command"] else "- case execution learning next: none",
                f"- case execution learning evidence next: `{latest_case['learning_next_evidence_command']}`" if latest_case["learning_next_evidence_command"] else "- case execution learning evidence next: none",
                f"- case execution learning actionable queue: {', '.join(f'`{command}`' for command in latest_case['learning_actionable_required_commands']) if latest_case['learning_actionable_required_commands'] else 'none'}",
                "",
                "Blockers:",
            ]
        )
        if blockers:
            lines.extend(f"- {blocker}" for blocker in blockers)
        else:
            lines.append("- none found by this read-only gate")
        lines.extend(
            [
                "",
                f"Verdict: {verdict}",
                f"Safe claim: {safe_claim}",
                "",
                "Next moves:",
            ]
        )
        if pending:
            if first_pending_approval_id:
                lines.append(f"- Review `approval readiness {first_pending_approval_id}` before the last-look `approval packet {first_pending_approval_id}`, then run `approval chain proof {first_pending_approval_id}` before deciding.")
            else:
                lines.append("- Review `approval review` before deciding; at least one pending approval row is missing a readable id.")
        if open_tasks:
            if first_open_task_id:
                lines.append(f"- Inspect `task completion packet {first_open_task_id}` before closing task #{first_open_task_id}.")
            else:
                lines.append("- Inspect `task board` before closing open work; at least one open task row is missing a readable id.")
        if latest_case["found"] and latest_case["case_closure_blocks_completion_claim"]:
            if latest_case["learning_next_evidence_command"]:
                learning_recheck_command = (
                    latest_case["learning_closure_command"]
                    or latest_case["learning_next_required_command"]
                    or latest_case["next_case_proof_command"]
                    or latest_case["next_closure_proof_command"]
                )
                lines.append(f"- Run `{latest_case['learning_next_evidence_command']}` to produce the latest execution case learning evidence, then recheck `{learning_recheck_command}`.")
            else:
                lines.append(f"- Run `{latest_case['next_closure_proof_command']}` to close the latest execution case proof debt.")
        if not verification_runs:
            lines.append("- Run `completion audit` or `evidence ledger` after the next verified work slice.")
        if failed_runs:
            lines.append(f"- Run `verification receipt {first_failed_run_id}` or `recent tool runs` to inspect the newest failed/blocked run." if first_failed_run_id else "- Run `verification receipt latest` or `recent tool runs` to inspect the newest failed/blocked run.")
            lines.append("- Run `execution health report` to summarize repeated failures, recovery coverage, and the next safe audit command.")
        if recovery_closure["required_commands"]:
            lines.append("- Close the recovery proof queue before retrying or claiming completion:")
            lines.extend(f"  - `{command}`" for command in recovery_closure["required_commands"])
        if learning_debt["required_commands"]:
            lines.append("- Close the execution learning debt queue before claiming completion:")
            lines.extend(f"  - `{command}`" for command in (learning_actionable_commands or learning_debt["required_commands"]))
        if storage_readiness["next_commands"]:
            lines.append("- Restore durable primary storage before claiming completion:")
            lines.extend(f"  - `{command}`" for command in storage_readiness["next_commands"])
        if completion_proof_queue:
            lines.append("- Follow the ordered completion proof queue:")
            lines.extend(f"  - `{command}`" for command in completion_proof_queue)
        if agi_summary["real_execution_gap_count"]:
            lines.append(f"- Run `{selected_agi_closure_commands[0]}` as the selected AGI next build move." if selected_agi_closure_commands else "- Run `agi gates` and pick the next safe build move for the first gate with remaining real-execution risk.")
        if not blockers:
            lines.append("- Ask the operator for final review before marking the broader goal complete.")
        lines.extend(
            [
                "",
                "Boundary:",
                "- This claim gate does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, mark goals done, or queue approvals.",
            ]
        )
        next_proof_command = ""
        if pending:
            next_proof_command = f"approval readiness {first_pending_approval_id}" if first_pending_approval_id else "approval review"
        elif completion_proof_queue:
            next_proof_command = str(completion_proof_queue[0])
        elif next_proof_commands:
            next_proof_command = str(next(iter(next_proof_commands.values()), ""))
        selected_agi_implementation_preflight = _agi_implementation_preflight_metadata(
            build_packet_ready=bool(selected_agi_build_ready),
            pending_approvals=pending,
            recovery_closure=recovery_closure,
            learning_debt=learning_debt,
            build_command=selected_agi_closure_commands[0] if selected_agi_closure_commands else "agi gates",
        )
        latest_case_evidence_preview_handoff = dict(
            latest_case["evidence_preview_latest_handoff"] or {}
        )
        latest_case_evidence_preview_next_command = str(
            latest_case_evidence_preview_handoff.get("next_command")
            or latest_case_evidence_preview_handoff.get("append_command")
            or ""
        )
        latest_case_evidence_preview_receipt_id = str(
            latest_case_evidence_preview_handoff.get("receipt_id")
            or latest_case_evidence_preview_handoff.get("inferred_receipt_id")
            or ""
        )
        latest_case_evidence_preview_receipt_kind = str(
            latest_case_evidence_preview_handoff.get("receipt_kind")
            or latest_case_evidence_preview_handoff.get("inferred_receipt_kind")
            or ""
        )
        latest_case_evidence_preview_receipt_target_status = str(
            latest_case_evidence_preview_handoff.get("receipt_target_status") or ""
        )
        completion_claim_handoff = _safe_metadata(
            source="completion_claim_gate",
            completion_claim_handoff_ready=True,
            handoff_ready=True,
            objective=objective,
            proposed_claim=proposed_claim,
            verdict=verdict,
            allowed_to_claim=allowed_to_claim,
            completion_claim_ready=allowed_to_claim,
            blocker_count=len(blockers),
            blockers=len(blockers),
            completion_blocker_count=len(blockers),
            blocker_details=blockers,
            completion_blockers_deduplicated=True,
            safe_claim=safe_claim,
            completion_proof_queue=completion_proof_queue,
            completion_proof_queue_count=len(completion_proof_queue),
            completion_next_proof_command=completion_proof_queue[0] if completion_proof_queue else "",
            next_completion_proof_command=completion_proof_queue[0] if completion_proof_queue else "",
            next_proof_commands=next_proof_commands,
            next_proof_command_count=len(next_proof_commands),
            next_proof_command=next_proof_command,
            pending_approvals=len(pending),
            open_tasks=len(open_tasks),
            active_goals=len(active_goals),
            recent_tool_runs=len(recent_runs),
            readable_recent_tool_runs=len(readable_recent_runs),
            unreadable_recent_tool_run_rows=unreadable_recent_run_rows,
            execution_health_verification_coverage_state=execution_health_verification["state"],
            execution_health_verification_recent_tool_runs=execution_health_verification["recent_tool_runs"],
            execution_health_verification_recent_action_runs=execution_health_verification["recent_action_runs"],
            execution_health_verification_recent_verification_runs=execution_health_verification["recent_verification_runs"],
            execution_health_verification_gap_count=execution_health_verification["gap_count"],
            execution_health_verification_first_gap=execution_health_verification["first_gap"],
            execution_health_verification_proof_queue=execution_health_verification["proof_queue"],
            execution_health_verification_proof_queue_count=execution_health_verification["proof_queue_count"],
            execution_health_verification_proof_queue_preview=execution_health_verification["proof_queue_preview"],
            execution_health_verification_proof_queue_preview_count=execution_health_verification["proof_queue_preview_count"],
            execution_health_verification_proof_queue_remaining_count=execution_health_verification["proof_queue_remaining_count"],
            execution_health_verification_next_required_command=execution_health_verification["next_required_command"],
            execution_health_verification_next_proof_command=execution_health_verification["next_proof_command"],
            execution_health_verification_blocks_completion_claim=execution_health_verification["blocks_completion_claim"],
            execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
            execution_health_recovery_closure_blocks_completion_claim=recovery_closure["blocks_completion_claim"],
            storage_runtime_fallback_active=storage_readiness["active"],
            storage_runtime_fallback_reason=storage_readiness["reason"],
            storage_runtime_fallback_exception_type=storage_readiness["exception_type"],
            storage_runtime_fallback_db_path_display=storage_readiness["db_path_display"],
            storage_runtime_fallback_vault_path_display=storage_readiness["vault_path_display"],
            storage_readiness_blocks_completion_claim=storage_readiness["blocks_completion_claim"],
            storage_recovery_required=storage_readiness["recovery_required"],
            storage_recovery_reason=storage_readiness["recovery_reason"],
            storage_issues=list(storage_readiness.get("issues") or []),
            storage_issue_count=int(storage_readiness.get("issue_count") or 0),
            storage_recovery_mode=storage_readiness["recovery_mode"],
            storage_recovery_next_operator_action=storage_readiness["recovery_next_operator_action"],
            storage_recovery_restart_required=storage_readiness["recovery_restart_required"],
            storage_recovery_check_command=storage_readiness["recovery_check_command"],
            storage_recovery_check_tool_command=storage_readiness["recovery_check_tool_command"],
            storage_recovery_check_api=STORAGE_RECOVERY_CHECK_API,
            storage_recovery_command=storage_readiness["recovery_command"],
            storage_readiness_blocker=storage_readiness["blocker"],
            storage_readiness_next_commands=storage_readiness["next_commands"],
            storage_readiness_next_command_count=len(storage_readiness["next_commands"]),
            storage_readiness_next_required_command=storage_readiness["next_commands"][0] if storage_readiness["blocks_completion_claim"] and storage_readiness["next_commands"] else "",
            storage_readiness_next_proof_command=storage_readiness["next_commands"][0] if storage_readiness["blocks_completion_claim"] and storage_readiness["next_commands"] else "",
            storage_readiness_proof_queue=list(storage_readiness["next_commands"]) if storage_readiness["blocks_completion_claim"] else [],
            storage_readiness_proof_queue_count=len(storage_readiness["next_commands"]) if storage_readiness["blocks_completion_claim"] else 0,
            storage_readiness_first_proof_command=storage_readiness["next_commands"][0] if storage_readiness["blocks_completion_claim"] and storage_readiness["next_commands"] else "",
            latest_execution_case_ready=latest_case["ready_for_human_review"],
            latest_execution_case_verdict=latest_case["verdict"],
            latest_execution_case_review_state=latest_case["review_state"],
            latest_execution_case_review_verdict=latest_case["review_verdict"],
            latest_execution_case_review_next_safe_command=latest_case["review_next_safe_command"],
            latest_execution_case_review_approval_required=latest_case["review_approval_required"],
            latest_execution_case_review_checklist_items=latest_case["review_checklist_items"],
            latest_execution_case_review_blocker_count=latest_case["review_blocker_count"],
            latest_execution_case_review_handoff=latest_case["review_handoff"],
            latest_execution_case_review_handoff_present=latest_case["review_handoff_present"],
            latest_execution_case_closure_verdict=latest_case["closure_verdict"],
            latest_execution_case_closure_ready=latest_case["case_closure_ready"],
            latest_execution_case_closure_blocks_completion_claim=latest_case["case_closure_blocks_completion_claim"],
            latest_execution_case_evidence_preview_gate_count=latest_case["evidence_preview_gate_count"],
            latest_execution_case_evidence_preview_ready_count=latest_case["evidence_preview_ready_count"],
            latest_execution_case_evidence_preview_blocked_count=latest_case["evidence_preview_blocked_count"],
            latest_execution_case_evidence_preview_verdicts=latest_case["evidence_preview_verdicts"],
            latest_execution_case_evidence_preview_latest_verdict=latest_case["evidence_preview_latest_verdict"],
            latest_execution_case_evidence_preview_latest_event_id=latest_case["evidence_preview_latest_event_id"],
            latest_execution_case_evidence_preview_latest_handoff=latest_case["evidence_preview_latest_handoff"],
            latest_execution_case_evidence_preview_latest_handoff_present=latest_case["evidence_preview_latest_handoff_present"],
            latest_execution_case_evidence_preview_next_command=latest_case_evidence_preview_next_command,
            latest_execution_case_evidence_preview_receipt_id=latest_case_evidence_preview_receipt_id,
            latest_execution_case_evidence_preview_receipt_kind=latest_case_evidence_preview_receipt_kind,
            latest_execution_case_evidence_preview_receipt_target_status=latest_case_evidence_preview_receipt_target_status,
            agi_real_execution_gap_count=agi_summary["real_execution_gap_count"],
            real_execution_gap_count=agi_summary["real_execution_gap_count"],
            selected_real_execution_gap=selected_agi_gate_name,
            selected_real_execution_gap_gate=selected_agi_gate_name,
            selected_real_execution_gap_detail=agi_summary["real_execution_gaps_by_gate"].get(selected_agi_gate_name, ""),
            agi_next_gate=selected_agi_gate_name,
            agi_next_target_title=selected_agi_target_title,
            agi_next_build_command=selected_agi_closure_commands[0] if selected_agi_closure_commands else "",
            agi_next_evidence_closure_commands=selected_agi_closure_commands,
            agi_next_evidence_closure_command_count=len(selected_agi_closure_commands),
            agi_next_focused_verification_commands=selected_agi_verification_commands,
            agi_next_focused_verification_command_count=len(selected_agi_verification_commands),
            agi_next_likely_files=selected_agi_likely_files,
            agi_next_likely_file_count=len(selected_agi_likely_files),
            agi_next_target_file_integrity_status=selected_agi_file_integrity["status"],
            agi_next_target_files_checked=selected_agi_file_integrity["checked"],
            agi_next_target_files_exist=selected_agi_file_integrity["all_exist"],
            agi_next_missing_target_files=selected_agi_file_integrity["missing"],
            agi_next_missing_target_file_count=selected_agi_file_integrity["missing_count"],
            agi_next_target_integrity_blocks_start=not selected_agi_file_integrity["all_exist"],
            agi_next_acceptance_checks=selected_agi_acceptance_checks,
            agi_next_acceptance_check_count=len(selected_agi_acceptance_checks),
            **_agi_acceptance_gap_metadata({"agi_next_acceptance_checks": selected_agi_acceptance_checks}),
            agi_next_build_packet_ready_for_review=selected_agi_build_ready,
            **selected_agi_implementation_preflight,
            **_agi_focus_selection_metadata(selected_agi_readiness),
        )

        return ToolResult(
            "completion_claim_gate",
            True,
            "\n".join(lines),
            _safe_metadata(
                objective=objective,
                proposed_claim=proposed_claim,
                verdict=verdict,
                allowed_to_claim=allowed_to_claim,
                blockers=len(blockers),
                completion_blocker_count=len(blockers),
                blocker_details=blockers,
                completion_blockers_deduplicated=True,
                safe_claim=safe_claim,
                completion_proof_queue=completion_proof_queue,
                completion_proof_queue_count=len(completion_proof_queue),
                next_completion_proof_command=completion_proof_queue[0] if completion_proof_queue else "",
                completion_next_proof_command=completion_proof_queue[0] if completion_proof_queue else "",
                next_proof_command=next_proof_command,
                next_proof_command_count=len(next_proof_commands),
                pending_approvals=len(pending),
                open_tasks=len(open_tasks),
                active_goals=len(active_goals),
                recent_tool_runs=len(recent_runs),
                readable_recent_tool_runs=len(readable_recent_runs),
                unreadable_recent_tool_run_rows=unreadable_recent_run_rows,
                strong_evidence=len(strong_lanes),
                partial_evidence=len(partial_lanes),
                missing_evidence=len(missing_lanes),
                unavailable_evidence=len(unavailable_lanes),
                lane_status={name: status for name, status, _available, _evidence in lane_rows},
                lane_evidence={name: evidence for name, _status, _available, evidence in lane_rows},
                lane_available_tools={name: available for name, _status, available, _evidence in lane_rows},
                next_proof_commands=next_proof_commands,
                storage_runtime_fallback_active=storage_readiness["active"],
                storage_runtime_fallback_reason=storage_readiness["reason"],
                storage_runtime_fallback_exception_type=storage_readiness["exception_type"],
                storage_runtime_fallback_db_path_display=storage_readiness["db_path_display"],
                storage_runtime_fallback_vault_path_display=storage_readiness["vault_path_display"],
                storage_readiness_blocks_completion_claim=storage_readiness["blocks_completion_claim"],
                storage_recovery_required=storage_readiness["recovery_required"],
                storage_recovery_reason=storage_readiness["recovery_reason"],
                storage_issues=list(storage_readiness.get("issues") or []),
                storage_issue_count=int(storage_readiness.get("issue_count") or 0),
                storage_recovery_mode=storage_readiness["recovery_mode"],
                storage_recovery_next_operator_action=storage_readiness["recovery_next_operator_action"],
                storage_recovery_restart_required=storage_readiness["recovery_restart_required"],
                storage_recovery_check_command=storage_readiness["recovery_check_command"],
                storage_recovery_check_tool_command=storage_readiness["recovery_check_tool_command"],
                storage_recovery_check_api=STORAGE_RECOVERY_CHECK_API,
                storage_recovery_command=storage_readiness["recovery_command"],
                storage_readiness_blocker=storage_readiness["blocker"],
                storage_readiness_next_commands=storage_readiness["next_commands"],
                storage_readiness_next_command_count=len(storage_readiness["next_commands"]),
                storage_readiness_next_required_command=storage_readiness["next_commands"][0] if storage_readiness["blocks_completion_claim"] and storage_readiness["next_commands"] else "",
                storage_readiness_next_proof_command=storage_readiness["next_commands"][0] if storage_readiness["blocks_completion_claim"] and storage_readiness["next_commands"] else "",
                storage_readiness_proof_queue=list(storage_readiness["next_commands"]) if storage_readiness["blocks_completion_claim"] else [],
                storage_readiness_proof_queue_count=len(storage_readiness["next_commands"]) if storage_readiness["blocks_completion_claim"] else 0,
                storage_readiness_first_proof_command=storage_readiness["next_commands"][0] if storage_readiness["blocks_completion_claim"] and storage_readiness["next_commands"] else "",
                latest_execution_case_found=latest_case["found"],
                latest_execution_case_id=latest_case["case_id"],
                latest_execution_case_verdict=latest_case["verdict"],
                latest_execution_case_ready=latest_case["ready_for_human_review"],
                latest_execution_case_review_state=latest_case["review_state"],
                latest_execution_case_review_verdict=latest_case["review_verdict"],
                latest_execution_case_review_next_safe_command=latest_case["review_next_safe_command"],
                latest_execution_case_review_approval_required=latest_case["review_approval_required"],
                latest_execution_case_review_checklist_items=latest_case["review_checklist_items"],
                latest_execution_case_review_blocker_count=latest_case["review_blocker_count"],
                latest_execution_case_review_handoff=latest_case["review_handoff"],
                latest_execution_case_review_handoff_present=latest_case["review_handoff_present"],
                latest_execution_case_closure_verdict=latest_case["closure_verdict"],
                latest_execution_case_closure_ready=latest_case["case_closure_ready"],
                latest_execution_case_closure_blocks_completion_claim=latest_case["case_closure_blocks_completion_claim"],
                latest_execution_case_closure_proof_queue=latest_case["closure_proof_queue"],
                latest_execution_case_closure_proof_queue_count=latest_case["closure_proof_queue_count"],
                latest_execution_case_next_closure_proof_command=latest_case["next_closure_proof_command"],
                latest_execution_case_missing_proofs=latest_case["missing_case_proofs"],
                latest_execution_case_next_required_command=latest_case["next_case_required_command"],
                latest_execution_case_next_proof_command=latest_case["next_case_proof_command"],
                latest_execution_case_next_proof_commands=latest_case["next_case_proof_commands"],
                latest_execution_case_mission_command_queue=latest_case["mission_command_queue"],
                latest_execution_case_mission_command_count=latest_case["mission_command_count"],
                latest_execution_case_evidence_preview_gate_count=latest_case["evidence_preview_gate_count"],
                latest_execution_case_evidence_preview_ready_count=latest_case["evidence_preview_ready_count"],
                latest_execution_case_evidence_preview_blocked_count=latest_case["evidence_preview_blocked_count"],
                latest_execution_case_evidence_preview_verdicts=latest_case["evidence_preview_verdicts"],
                latest_execution_case_evidence_preview_latest_verdict=latest_case["evidence_preview_latest_verdict"],
                latest_execution_case_evidence_preview_latest_event_id=latest_case["evidence_preview_latest_event_id"],
                latest_execution_case_evidence_preview_latest_handoff=latest_case["evidence_preview_latest_handoff"],
                latest_execution_case_evidence_preview_latest_handoff_present=latest_case["evidence_preview_latest_handoff_present"],
                latest_execution_case_evidence_preview_next_command=latest_case_evidence_preview_next_command,
                latest_execution_case_evidence_preview_receipt_id=latest_case_evidence_preview_receipt_id,
                latest_execution_case_evidence_preview_receipt_kind=latest_case_evidence_preview_receipt_kind,
                latest_execution_case_evidence_preview_receipt_target_status=latest_case_evidence_preview_receipt_target_status,
                latest_execution_case_approval_queue_forecast=latest_case["approval_queue_forecast"],
                latest_execution_case_forecast_new_approvals=latest_case["forecast_new_approvals"],
                latest_execution_case_forecast_reused_approval_ids=latest_case["forecast_reused_approval_ids"],
                latest_execution_case_forecast_queue_before=latest_case["forecast_queue_before"],
                latest_execution_case_forecast_queue_after_if_sent=latest_case["forecast_queue_after_if_sent"],
                latest_execution_case_forecast_queue_delta_if_sent=latest_case["forecast_queue_delta_if_sent"],
                latest_execution_case_recovery_closure_state=latest_case["recovery_closure_state"],
                latest_execution_case_recovery_closure_missing=latest_case["recovery_closure_missing"],
                latest_execution_case_recovery_closure_next_required_command=latest_case["recovery_closure_next_required_command"],
                latest_execution_case_recovery_closure_required_commands=latest_case["recovery_closure_required_commands"],
                latest_execution_case_recovery_closure_proof_queue=latest_case["recovery_closure_proof_queue"],
                latest_execution_case_recovery_closure_proof_queue_count=latest_case["recovery_closure_proof_queue_count"],
                latest_execution_case_recovery_closure_next_proof_command=latest_case["recovery_closure_next_proof_command"],
                latest_execution_case_recovery_closure_blocks_completion_claim=latest_case["recovery_closure_blocks_completion_claim"],
                latest_execution_case_learning_state=latest_case["learning_state"],
                latest_execution_case_learning_missing=latest_case["learning_missing"],
                latest_execution_case_learning_next_required_command=latest_case["learning_next_required_command"],
                latest_execution_case_learning_required_commands=latest_case["learning_required_commands"],
                latest_execution_case_learning_proof_queue=latest_case["learning_proof_queue"],
                latest_execution_case_learning_proof_queue_count=latest_case["learning_proof_queue_count"],
                latest_execution_case_learning_next_proof_command=latest_case["learning_next_proof_command"],
                latest_execution_case_learning_actionable_required_commands=latest_case["learning_actionable_required_commands"],
                latest_execution_case_learning_actionable_proof_queue=latest_case["learning_actionable_proof_queue"],
                latest_execution_case_learning_actionable_proof_queue_count=latest_case["learning_actionable_proof_queue_count"],
                latest_execution_case_learning_actionable_next_required_command=latest_case["learning_actionable_next_required_command"],
                latest_execution_case_learning_actionable_next_proof_command=latest_case["learning_actionable_next_proof_command"],
                latest_execution_case_learning_next_evidence_command=latest_case["learning_next_evidence_command"],
                latest_execution_case_learning_blocks_completion_claim=latest_case["learning_blocks_completion_claim"],
                agi_gates=agi_summary["gates"],
                agi_strong_gates=agi_summary["strong"],
                agi_partial_gates=agi_summary["partial"],
                agi_missing_gates=agi_summary["missing"],
                agi_gate_statuses=agi_summary["gate_statuses"],
                agi_missing_evidence_by_gate=agi_summary["missing_evidence_by_gate"],
                agi_real_execution_gaps_by_gate=agi_summary["real_execution_gaps_by_gate"],
                agi_real_execution_gap_count=agi_summary["real_execution_gap_count"],
                real_execution_gap_count=agi_summary["real_execution_gap_count"],
                selected_real_execution_gap=selected_agi_gate_name,
                selected_real_execution_gap_gate=selected_agi_gate_name,
                selected_real_execution_gap_detail=agi_summary["real_execution_gaps_by_gate"].get(selected_agi_gate_name, ""),
                agi_next_moves_by_gate=agi_summary["next_moves_by_gate"],
                agi_evidence_closure_commands_by_gate=agi_summary["evidence_closure_commands_by_gate"],
                agi_focused_verification_by_gate=agi_summary["focused_verification_by_gate"],
                agi_next_gate=selected_agi_gate_name,
                agi_next_target_title=selected_agi_target_title,
                agi_next_build_command=selected_agi_closure_commands[0] if selected_agi_closure_commands else "",
                agi_next_evidence_closure_commands=selected_agi_closure_commands,
                agi_next_evidence_closure_command_count=len(selected_agi_closure_commands),
                agi_next_focused_verification_commands=selected_agi_verification_commands,
                agi_next_focused_verification_command_count=len(selected_agi_verification_commands),
                agi_next_likely_files=selected_agi_likely_files,
                agi_next_likely_file_count=len(selected_agi_likely_files),
                agi_next_target_file_integrity_status=selected_agi_file_integrity["status"],
                agi_next_target_files_checked=selected_agi_file_integrity["checked"],
                agi_next_target_files_exist=selected_agi_file_integrity["all_exist"],
                agi_next_missing_target_files=selected_agi_file_integrity["missing"],
                agi_next_missing_target_file_count=selected_agi_file_integrity["missing_count"],
                agi_next_target_integrity_blocks_start=not selected_agi_file_integrity["all_exist"],
                agi_next_acceptance_checks=selected_agi_acceptance_checks,
                agi_next_acceptance_check_count=len(selected_agi_acceptance_checks),
                **_agi_acceptance_gap_metadata({"agi_next_acceptance_checks": selected_agi_acceptance_checks}),
                agi_next_build_packet_ready_for_review=selected_agi_build_ready,
                **selected_agi_implementation_preflight,
                **_agi_focus_selection_metadata(selected_agi_readiness),
                verification_runs=len(verification_runs),
                after_action_learning_runs=len(after_action_learning_runs),
                evidence_completion_runs=len(evidence_completion_runs),
                recent_failed_runs=len(failed_runs),
                execution_health_recovery_closure_state=recovery_closure["state"],
                execution_health_recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                execution_health_recovery_closure_missing=recovery_closure["missing"],
                execution_health_recovery_closure_missing_count=recovery_closure["missing_count"],
                execution_health_recovery_closure_required_commands=recovery_closure["required_commands"],
                execution_health_recovery_closure_next_required_command=recovery_closure["next_required_command"],
                execution_health_recovery_closure_checklist_command=_recovery_closure_checklist_command(recovery_closure),
                execution_health_recovery_closure_should_open_checklist=bool(_recovery_closure_checklist_command(recovery_closure)),
                execution_health_recovery_closure_proof_queue=recovery_closure["proof_queue"],
                execution_health_recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
                execution_health_recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
                execution_health_recovery_closure_blocks_completion_claim=recovery_closure["blocks_completion_claim"],
                execution_health_recovery_closure_target_run_id=recovery_closure["target_run_id"],
                execution_health_recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                execution_health_recovery_closure_target_verification_receipts=recovery_closure["target_verification_receipts"],
                execution_health_recovery_closure_target_recovery_packets=recovery_closure["target_recovery_packets"],
                execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure["target_after_action_learning_packets"],
                execution_health_verification_coverage_state=execution_health_verification["state"],
                execution_health_verification_recent_tool_runs=execution_health_verification["recent_tool_runs"],
                execution_health_verification_recent_action_runs=execution_health_verification["recent_action_runs"],
                execution_health_verification_recent_verification_runs=execution_health_verification["recent_verification_runs"],
                execution_health_verification_gap_count=execution_health_verification["gap_count"],
                execution_health_verification_first_gap=execution_health_verification["first_gap"],
                execution_health_verification_proof_queue=execution_health_verification["proof_queue"],
                execution_health_verification_proof_queue_count=execution_health_verification["proof_queue_count"],
                execution_health_verification_proof_queue_preview=execution_health_verification["proof_queue_preview"],
                execution_health_verification_proof_queue_preview_count=execution_health_verification["proof_queue_preview_count"],
                execution_health_verification_proof_queue_remaining_count=execution_health_verification["proof_queue_remaining_count"],
                execution_health_verification_next_required_command=execution_health_verification["next_required_command"],
                execution_health_verification_next_proof_command=execution_health_verification["next_proof_command"],
                execution_health_verification_blocks_completion_claim=execution_health_verification["blocks_completion_claim"],
                execution_learning_state=learning_debt["state"],
                execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
                execution_learning_recent_action_runs=learning_debt["recent_action_runs"],
                execution_learning_failed_or_blocked_action_runs=learning_debt["failed_or_blocked_action_runs"],
                execution_learning_recent_verification_runs=learning_debt["recent_verification_runs"],
                execution_learning_recent_recovery_runs=learning_debt["recent_recovery_runs"],
                execution_learning_recent_after_action_learning_runs=learning_debt["recent_after_action_learning_runs"],
                execution_learning_target_run_id=learning_debt["target_run_id"],
                execution_learning_target_tool_name=learning_debt["target_tool_name"],
                execution_learning_target_after_action_learning_packets=learning_debt["target_after_action_learning_packets"],
                execution_learning_missing=learning_debt["missing"],
                execution_learning_missing_count=learning_debt["missing_count"],
                execution_learning_closure_command=learning_closure_command,
                execution_learning_evidence_command=learning_debt["learning_evidence_command"],
                execution_learning_after_action_learning_command=learning_debt["after_action_learning_command"],
                execution_learning_required_commands=learning_debt["required_commands"],
                execution_learning_next_required_command=learning_debt["next_required_command"],
                execution_learning_proof_queue=learning_debt["proof_queue"],
                execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
                execution_learning_next_proof_command=learning_debt["next_proof_command"],
                execution_learning_actionable_required_commands=learning_actionable_commands,
                execution_learning_actionable_required_command_count=len(learning_actionable_commands),
                execution_learning_actionable_proof_queue=learning_actionable_commands,
                execution_learning_actionable_proof_queue_count=len(learning_actionable_commands),
                execution_learning_actionable_next_required_command=learning_debt["actionable_next_required_command"],
                execution_learning_actionable_next_proof_command=learning_debt["actionable_next_proof_command"],
                execution_learning_next_evidence_command=learning_debt["next_evidence_command"],
                completion_claim_ready=allowed_to_claim,
                completion_claim_handoff_ready=True,
                completion_claim_handoff=completion_claim_handoff,
            ),
        )

    def completion_next_proof_packet(args: dict[str, Any]) -> ToolResult:
        objective = _short(
            args.get("objective") or args.get("goal") or args.get("request") or (
                "Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant."
            ),
            limit=500,
        )
        claim_gate = completion_claim_gate({"objective": objective})
        gate_metadata = dict(claim_gate.metadata)
        completion_queue = list(gate_metadata.get("completion_proof_queue") or [])
        ordered_next_command = str(
            gate_metadata.get("next_completion_proof_command")
            or gate_metadata.get("completion_next_proof_command")
            or (completion_queue[0] if completion_queue else "")
        )
        claim_state = str(gate_metadata.get("verdict") or "UNKNOWN")
        blockers = list(gate_metadata.get("blocker_details") or [])
        latest_case_closure_verdict = str(gate_metadata.get("latest_execution_case_closure_verdict") or "")
        latest_case_closure_ready = _metadata_bool(gate_metadata.get("latest_execution_case_closure_ready"))
        latest_case_closure_blocks = _metadata_bool(gate_metadata.get("latest_execution_case_closure_blocks_completion_claim"))
        recovery_queue = list(gate_metadata.get("execution_health_recovery_closure_proof_queue") or [])
        verification_queue = list(gate_metadata.get("execution_health_verification_proof_queue") or [])
        learning_queue = list(gate_metadata.get("execution_learning_proof_queue") or [])
        learning_actionable_queue = list(gate_metadata.get("execution_learning_actionable_proof_queue") or [])
        learning_actionable_next_required = str(gate_metadata.get("execution_learning_actionable_next_required_command") or "")
        learning_actionable_next = str(
            gate_metadata.get("execution_learning_actionable_next_proof_command")
            or (learning_actionable_queue[0] if learning_actionable_queue else "")
        )
        learning_closure_command = str(gate_metadata.get("execution_learning_closure_command") or "")
        case_closure_queue = list(gate_metadata.get("latest_execution_case_closure_proof_queue") or [])
        case_closure_queue_preview = case_closure_queue[:5]
        case_closure_queue_remaining_count = max(
            len(case_closure_queue) - len(case_closure_queue_preview),
            0,
        )
        case_closure_first_proof_command = (
            str(case_closure_queue_preview[0])
            if case_closure_queue_preview
            else ""
        )
        case_mission_queue = list(gate_metadata.get("latest_execution_case_mission_command_queue") or [])
        case_next_proof_commands = list(gate_metadata.get("latest_execution_case_next_proof_commands") or [])

        completion_actionable_queue, learning_prerequisites_inserted = _completion_actionable_queue_from_ordered(
            completion_queue=completion_queue,
            learning_closure_command=learning_closure_command,
            learning_actionable_queue=learning_actionable_queue,
        )
        uses_actionable_learning_prerequisite = bool(
            learning_prerequisites_inserted
            and ordered_next_command == learning_closure_command
        )

        next_command = completion_actionable_queue[0] if completion_actionable_queue else ordered_next_command
        if not next_command:
            next_command = "completion claim gate"
        command_source = "completion_claim_gate"
        if uses_actionable_learning_prerequisite and next_command in learning_actionable_queue:
            command_source = "execution_learning_actionable"
        elif next_command in case_closure_queue:
            command_source = "execution_case_closure"
        elif next_command in verification_queue:
            command_source = "execution_health_verification"
        elif next_command in recovery_queue:
            command_source = "execution_recovery_closure"
        elif next_command in learning_queue:
            command_source = "execution_learning"
        elif next_command.startswith("approval "):
            command_source = "approval_queue"
        elif next_command.startswith("task completion packet"):
            command_source = "task_completion"
        elif next_command == "completion audit":
            command_source = "completion_audit"
        elif next_command == "evidence ledger":
            command_source = "evidence_ledger"
        elif next_command.startswith("agi next build move"):
            command_source = "agi_next_build_move"
        command_source_reasons = {
            "execution_learning_actionable": "execution learning closure has missing prerequisite proof, so the actionable evidence command comes first",
            "execution_case_closure": "latest saved execution case still has closure proof debt",
            "execution_health_verification": "recent tool activity needs explicit verification coverage before completion claims",
            "execution_recovery_closure": "recent failed or blocked execution needs recovery closure proof",
            "execution_learning": "recent failed or blocked execution needs after-action learning proof",
            "approval_queue": "pending risky approval needs readiness, last-look packet, and chain proof before retry",
            "task_completion": "open task needs an evidence-backed completion packet before closing",
            "completion_audit": "completion audit must be reviewed before any broad completion claim",
            "evidence_ledger": "proof lanes need ledger review before claiming completion",
            "agi_next_build_move": "next AGI harness build target needs a focused proof slice",
            "completion_claim_gate": "completion gate needs a fresh review or human confirmation",
        }
        command_reason = command_source_reasons.get(command_source, "completion proof queue selected this as the next ordered proof")
        storage_commands = list(gate_metadata.get("storage_readiness_next_commands") or [])
        storage_blocks_completion = _metadata_bool(gate_metadata.get("storage_readiness_blocks_completion_claim"))
        storage_blocker = str(gate_metadata.get("storage_readiness_blocker") or "")
        storage_issues = list(gate_metadata.get("storage_issues") or [])
        storage_issue_count = int(gate_metadata.get("storage_issue_count") or len(storage_issues))
        storage_readiness_next_proof_command = (
            str(storage_commands[0])
            if storage_blocks_completion and storage_commands
            else ""
        )
        storage_readiness_next_required_command = (
            str(gate_metadata.get("storage_readiness_next_required_command") or storage_readiness_next_proof_command)
            if storage_blocks_completion
            else ""
        )
        storage_readiness_proof_queue = list(storage_commands) if storage_blocks_completion else []
        storage_readiness_proof_queue_preview = storage_readiness_proof_queue[:5]
        storage_readiness_proof_queue_remaining_count = max(
            len(storage_readiness_proof_queue) - len(storage_readiness_proof_queue_preview),
            0,
        )
        storage_readiness_first_proof_command = (
            str(storage_readiness_proof_queue[0])
            if storage_readiness_proof_queue
            else ""
        )
        storage_handoff = _storage_handoff_from_metadata(
            {
                **gate_metadata,
                "storage_readiness_next_required_command": storage_readiness_next_required_command,
                "storage_readiness_next_proof_command": storage_readiness_next_proof_command,
                "storage_readiness_proof_queue": storage_readiness_proof_queue,
                "storage_readiness_proof_queue_count": len(storage_readiness_proof_queue),
                "storage_readiness_proof_queue_preview": storage_readiness_proof_queue_preview,
                "storage_readiness_proof_queue_preview_count": len(storage_readiness_proof_queue_preview),
                "storage_readiness_proof_queue_remaining_count": storage_readiness_proof_queue_remaining_count,
                "storage_readiness_first_proof_command": storage_readiness_first_proof_command,
            },
            source="completion_next_proof_storage",
        )
        real_execution_gap_count = int(gate_metadata.get("agi_real_execution_gap_count") or 0)
        selected_real_execution_gap = str(
            gate_metadata.get("selected_real_execution_gap")
            or gate_metadata.get("real_execution_gap")
            or gate_metadata.get("agi_next_gate")
            or ""
        )
        selected_real_execution_gap_gate = str(
            gate_metadata.get("selected_real_execution_gap_gate")
            or gate_metadata.get("agi_next_gate")
            or selected_real_execution_gap
            or ""
        )
        selected_real_execution_gap_detail = str(
            gate_metadata.get("selected_real_execution_gap_detail")
            or (gate_metadata.get("agi_real_execution_gaps_by_gate") or {}).get(selected_real_execution_gap_gate)
            or gate_metadata.get("real_execution_gap")
            or ""
        )
        if storage_blocks_completion and storage_commands and next_command == storage_commands[0]:
            command_reason = (
                "durable primary storage must be checked first because the runtime is using workspace-local fallback storage"
            )
            if storage_blocker:
                command_reason = f"{command_reason}: {storage_blocker}"
        blocker_summary = blockers[0] if blockers else "no blocker details from completion claim gate"
        queue_position = completion_actionable_queue.index(next_command) + 1 if next_command in completion_actionable_queue else 0
        ordered_queue_position = completion_queue.index(ordered_next_command) + 1 if ordered_next_command in completion_queue else 0

        lines = [
            "Jarvis completion next proof packet:",
            "This read-only packet turns the completion gate into one command-first proof step. It does not run the command.",
            "",
            f"Objective: {objective}",
            f"Claim gate verdict: {claim_state}",
            f"Completion claim ready: {'yes' if _metadata_bool(gate_metadata.get('allowed_to_claim')) else 'no'}",
            f"Next required command: `{next_command}`",
            f"Command source: {command_source}",
            f"Command reason: {command_reason}",
            f"First blocker: {blocker_summary}",
            f"Queue position: {queue_position} of {len(completion_actionable_queue)}",
        "",
            "Why this proof comes next:",
        ]
        if uses_actionable_learning_prerequisite:
            lines[11:11] = [
                f"Ordered completion proof: `{ordered_next_command}`",
                f"Next actionable proof: `{next_command}`",
                f"Ordered queue position: {ordered_queue_position} of {len(completion_queue)}",
            ]
        if blockers:
            lines.extend(f"- {blocker}" for blocker in blockers[:6])
        else:
            lines.append("- no blockers found by the completion claim gate; ask the operator for human review before claiming broad completion")
        lines.extend(
            [
                "",
                "Execution case closure:",
                f"- review state: {gate_metadata.get('latest_execution_case_review_state') or 'unknown'}",
                f"- review verdict: {gate_metadata.get('latest_execution_case_review_verdict') or 'unknown'}",
                f"- review next safe command: `{gate_metadata.get('latest_execution_case_review_next_safe_command')}`" if gate_metadata.get("latest_execution_case_review_next_safe_command") else "- review next safe command: none",
                f"- verdict: {latest_case_closure_verdict or 'none'}",
                f"- ready: {'yes' if latest_case_closure_ready else 'no'}",
                f"- blocks completion claim: {'yes' if latest_case_closure_blocks else 'no'}",
                f"- proof queue: {', '.join(f'`{command}`' for command in case_closure_queue) if case_closure_queue else 'none'}",
                f"- mission queue: {', '.join(f'`{command}`' for command in case_mission_queue) if case_mission_queue else 'none'}",
                "",
                "Recovery and learning queues:",
                f"- verification coverage: {gate_metadata.get('execution_health_verification_coverage_state') or 'unknown'}",
                f"- verification proof queue: {', '.join(f'`{command}`' for command in verification_queue) if verification_queue else 'none'}",
                f"- recovery proof queue: {', '.join(f'`{command}`' for command in recovery_queue) if recovery_queue else 'none'}",
                f"- learning actionable next required: `{learning_actionable_next_required}`" if learning_actionable_next_required else "- learning actionable next required: none",
                f"- learning actionable proof alias: `{learning_actionable_next}`" if learning_actionable_next else "- learning actionable proof alias: none",
                f"- learning closure command: `{learning_closure_command}`" if learning_closure_command else "- learning closure command: none",
                f"- learning proof queue: {', '.join(f'`{command}`' for command in learning_queue) if learning_queue else 'none'}",
                f"- learning actionable proof queue: {', '.join(f'`{command}`' for command in learning_actionable_queue) if learning_actionable_queue else 'none'}",
                "",
                "Actionable completion proof queue:",
            ]
        )
        if completion_actionable_queue:
            lines.extend(f"- `{command}`" for command in completion_actionable_queue)
        else:
            lines.append("- none; run `completion claim gate` for a fresh check")
        lines.extend(["", "Ordered completion proof queue:"])
        if completion_queue:
            lines.extend(f"- `{command}`" for command in completion_queue)
        else:
            lines.append("- none; run `completion claim gate` for a fresh check")
        lines.extend(
            [
                "",
                "Follow-up:",
                "- After completing the next required command, rerun `completion next proof` to refresh the ordered queue.",
                "- Do not claim Jarvis is complete until `completion claim gate` is clear and the operator has reviewed the result.",
                "",
                "Boundary:",
                "- This packet does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, write notes, write the database, control the computer, call external services, speak, complete tasks, mark goals done, or queue approvals.",
            ]
        )
        completion_next_proof_handoff = _safe_metadata(
            source="completion_next_proof_packet",
            completion_next_proof_handoff_ready=True,
            handoff_ready=True,
            objective=objective,
            claim_gate_verdict=claim_state,
            completion_claim_ready=_metadata_bool(gate_metadata.get("allowed_to_claim")),
            allowed_to_claim=_metadata_bool(gate_metadata.get("allowed_to_claim")),
            blockers=gate_metadata.get("blockers"),
            blocker_details=blockers,
            completion_proof_queue=completion_queue,
            completion_proof_queue_count=len(completion_queue),
            completion_ordered_proof_queue=completion_queue,
            completion_ordered_proof_queue_count=len(completion_queue),
            completion_ordered_next_proof_command=ordered_next_command,
            completion_actionable_proof_queue=completion_actionable_queue,
            completion_actionable_proof_queue_count=len(completion_actionable_queue),
            completion_actionable_next_proof_command=next_command,
            completion_actionable_queue_includes_learning_prerequisites=learning_prerequisites_inserted,
            completion_next_proof_uses_actionable_learning_prerequisite=uses_actionable_learning_prerequisite,
            proof_queue=completion_actionable_queue,
            proof_queue_count=len(completion_actionable_queue),
            next_completion_proof_command=next_command,
            completion_next_proof_command=next_command,
            next_proof_command=next_command,
            command_source=command_source,
            command_reason=command_reason,
            first_blocker=blocker_summary,
            queue_position=queue_position,
            queue_total=len(completion_actionable_queue),
            ordered_queue_position=ordered_queue_position,
            ordered_queue_total=len(completion_queue),
            refresh_command="completion next proof",
            latest_execution_case_found=gate_metadata.get("latest_execution_case_found"),
            latest_execution_case_id=gate_metadata.get("latest_execution_case_id"),
            latest_execution_case_review_state=gate_metadata.get("latest_execution_case_review_state", ""),
            latest_execution_case_review_verdict=gate_metadata.get("latest_execution_case_review_verdict", ""),
            latest_execution_case_review_next_safe_command=gate_metadata.get("latest_execution_case_review_next_safe_command", ""),
            latest_execution_case_review_approval_required=gate_metadata.get("latest_execution_case_review_approval_required", False),
            latest_execution_case_review_checklist_items=gate_metadata.get("latest_execution_case_review_checklist_items", 0),
            latest_execution_case_review_blocker_count=gate_metadata.get("latest_execution_case_review_blocker_count", 0),
            latest_execution_case_review_handoff=gate_metadata.get("latest_execution_case_review_handoff", {}),
            latest_execution_case_review_handoff_present=gate_metadata.get("latest_execution_case_review_handoff_present", False),
            latest_execution_case_closure_verdict=latest_case_closure_verdict,
            latest_execution_case_closure_ready=latest_case_closure_ready,
            latest_execution_case_closure_blocks_completion_claim=latest_case_closure_blocks,
            latest_execution_case_closure_proof_queue=case_closure_queue,
            latest_execution_case_closure_proof_queue_count=len(case_closure_queue),
            latest_execution_case_closure_proof_queue_preview=case_closure_queue_preview,
            latest_execution_case_closure_proof_queue_preview_count=len(case_closure_queue_preview),
            latest_execution_case_closure_proof_queue_remaining_count=case_closure_queue_remaining_count,
            latest_execution_case_closure_first_proof_command=case_closure_first_proof_command,
            latest_execution_case_next_closure_proof_command=gate_metadata.get("latest_execution_case_next_closure_proof_command"),
            latest_execution_case_next_required_command=gate_metadata.get(
                "latest_execution_case_next_required_command",
                gate_metadata.get("latest_execution_case_next_proof_command"),
            ),
            latest_execution_case_next_proof_command=gate_metadata.get("latest_execution_case_next_proof_command"),
            latest_execution_case_next_proof_commands=case_next_proof_commands,
            latest_execution_case_mission_command_queue=case_mission_queue,
            latest_execution_case_mission_command_count=len(case_mission_queue),
            latest_execution_case_evidence_preview_gate_count=gate_metadata.get("latest_execution_case_evidence_preview_gate_count", 0),
            latest_execution_case_evidence_preview_ready_count=gate_metadata.get("latest_execution_case_evidence_preview_ready_count", 0),
            latest_execution_case_evidence_preview_blocked_count=gate_metadata.get("latest_execution_case_evidence_preview_blocked_count", 0),
            latest_execution_case_evidence_preview_verdicts=gate_metadata.get("latest_execution_case_evidence_preview_verdicts", []),
            latest_execution_case_evidence_preview_latest_verdict=gate_metadata.get("latest_execution_case_evidence_preview_latest_verdict", ""),
            latest_execution_case_evidence_preview_latest_event_id=gate_metadata.get("latest_execution_case_evidence_preview_latest_event_id"),
            latest_execution_case_evidence_preview_latest_handoff=gate_metadata.get("latest_execution_case_evidence_preview_latest_handoff", {}),
            latest_execution_case_evidence_preview_latest_handoff_present=gate_metadata.get("latest_execution_case_evidence_preview_latest_handoff_present", False),
            latest_execution_case_evidence_preview_next_command=gate_metadata.get("latest_execution_case_evidence_preview_next_command", ""),
            latest_execution_case_evidence_preview_receipt_id=gate_metadata.get("latest_execution_case_evidence_preview_receipt_id", ""),
            latest_execution_case_evidence_preview_receipt_kind=gate_metadata.get("latest_execution_case_evidence_preview_receipt_kind", ""),
            latest_execution_case_evidence_preview_receipt_target_status=gate_metadata.get("latest_execution_case_evidence_preview_receipt_target_status", ""),
            execution_health_recovery_closure_state=gate_metadata.get("execution_health_recovery_closure_state"),
            execution_health_recovery_closure_missing=gate_metadata.get("execution_health_recovery_closure_missing"),
            execution_health_recovery_closure_required_commands=gate_metadata.get("execution_health_recovery_closure_required_commands", []),
            execution_health_recovery_closure_next_required_command=gate_metadata.get(
                "execution_health_recovery_closure_next_required_command",
                gate_metadata.get("execution_health_recovery_closure_next_proof_command", ""),
            ),
            execution_health_recovery_closure_proof_queue=recovery_queue,
            execution_health_recovery_closure_proof_queue_count=len(recovery_queue),
            execution_health_recovery_closure_next_proof_command=gate_metadata.get("execution_health_recovery_closure_next_proof_command"),
            execution_health_recovery_closure_blocks_completion_claim=gate_metadata.get("execution_health_recovery_closure_blocks_completion_claim"),
            execution_health_verification_coverage_state=gate_metadata.get("execution_health_verification_coverage_state", ""),
            execution_health_verification_recent_tool_runs=gate_metadata.get("execution_health_verification_recent_tool_runs", 0),
            execution_health_verification_recent_action_runs=gate_metadata.get("execution_health_verification_recent_action_runs", 0),
            execution_health_verification_recent_verification_runs=gate_metadata.get("execution_health_verification_recent_verification_runs", 0),
            execution_health_verification_gap_count=gate_metadata.get("execution_health_verification_gap_count", 0),
            execution_health_verification_first_gap=gate_metadata.get("execution_health_verification_first_gap", ""),
            execution_health_verification_proof_queue=verification_queue,
            execution_health_verification_proof_queue_count=len(verification_queue),
            execution_health_verification_proof_queue_preview=gate_metadata.get("execution_health_verification_proof_queue_preview", verification_queue[:3]),
            execution_health_verification_proof_queue_preview_count=gate_metadata.get("execution_health_verification_proof_queue_preview_count", len(verification_queue[:3])),
            execution_health_verification_proof_queue_remaining_count=gate_metadata.get("execution_health_verification_proof_queue_remaining_count", max(len(verification_queue) - len(verification_queue[:3]), 0)),
            execution_health_verification_next_required_command=gate_metadata.get("execution_health_verification_next_required_command", gate_metadata.get("execution_health_verification_next_proof_command", "")),
            execution_health_verification_next_proof_command=gate_metadata.get("execution_health_verification_next_proof_command", ""),
            execution_health_verification_blocks_completion_claim=gate_metadata.get("execution_health_verification_blocks_completion_claim", False),
            execution_learning_state=gate_metadata.get("execution_learning_state"),
            execution_learning_missing=gate_metadata.get("execution_learning_missing"),
            execution_learning_closure_command=gate_metadata.get("execution_learning_closure_command"),
            execution_learning_required_commands=gate_metadata.get("execution_learning_required_commands", []),
            execution_learning_next_required_command=gate_metadata.get(
                "execution_learning_next_required_command",
                gate_metadata.get("execution_learning_next_proof_command", ""),
            ),
            execution_learning_proof_queue=learning_queue,
            execution_learning_proof_queue_count=len(learning_queue),
            execution_learning_next_proof_command=gate_metadata.get("execution_learning_next_proof_command"),
            execution_learning_actionable_required_commands=gate_metadata.get("execution_learning_actionable_required_commands", []),
            execution_learning_actionable_required_command_count=gate_metadata.get("execution_learning_actionable_required_command_count", 0),
            execution_learning_actionable_proof_queue=learning_actionable_queue,
            execution_learning_actionable_proof_queue_count=len(learning_actionable_queue),
            execution_learning_actionable_next_required_command=gate_metadata.get("execution_learning_actionable_next_required_command"),
            execution_learning_actionable_next_proof_command=gate_metadata.get("execution_learning_actionable_next_proof_command"),
            execution_learning_next_evidence_command=gate_metadata.get("execution_learning_next_evidence_command"),
            execution_learning_blocks_completion_claim=gate_metadata.get("execution_learning_blocks_completion_claim"),
            storage_runtime_fallback_active=_metadata_bool(gate_metadata.get("storage_runtime_fallback_active")),
            storage_runtime_fallback_reason=gate_metadata.get("storage_runtime_fallback_reason") or "",
            storage_runtime_fallback_exception_type=gate_metadata.get("storage_runtime_fallback_exception_type") or "",
            storage_runtime_fallback_db_path_display=gate_metadata.get("storage_runtime_fallback_db_path_display") or "",
            storage_runtime_fallback_vault_path_display=gate_metadata.get("storage_runtime_fallback_vault_path_display") or "",
            storage_readiness_blocks_completion_claim=_metadata_bool(gate_metadata.get("storage_readiness_blocks_completion_claim")),
            storage_recovery_required=_metadata_bool(gate_metadata.get("storage_recovery_required")),
            storage_recovery_reason=gate_metadata.get("storage_recovery_reason") or "",
            storage_issues=storage_issues,
            storage_issue_count=storage_issue_count,
            storage_recovery_mode=gate_metadata.get("storage_recovery_mode") or "none",
            storage_recovery_next_operator_action=gate_metadata.get("storage_recovery_next_operator_action") or "",
            storage_recovery_restart_required=_metadata_bool(gate_metadata.get("storage_recovery_restart_required")),
            storage_recovery_check_command=gate_metadata.get("storage_recovery_check_command") or "",
            storage_recovery_check_tool_command=gate_metadata.get("storage_recovery_check_tool_command") or "",
            storage_recovery_check_api=gate_metadata.get("storage_recovery_check_api") or STORAGE_RECOVERY_CHECK_API,
            storage_recovery_command=gate_metadata.get("storage_recovery_command") or "",
            storage_readiness_blocker=gate_metadata.get("storage_readiness_blocker") or "",
            storage_readiness_next_commands=list(gate_metadata.get("storage_readiness_next_commands") or []),
            storage_readiness_next_command_count=len(list(gate_metadata.get("storage_readiness_next_commands") or [])),
            storage_readiness_next_required_command=storage_readiness_next_required_command,
            storage_readiness_next_proof_command=storage_readiness_next_proof_command,
            storage_readiness_proof_queue=storage_readiness_proof_queue,
            storage_readiness_proof_queue_count=len(storage_readiness_proof_queue),
            storage_readiness_proof_queue_preview=storage_readiness_proof_queue_preview,
            storage_readiness_proof_queue_preview_count=len(storage_readiness_proof_queue_preview),
            storage_readiness_proof_queue_remaining_count=storage_readiness_proof_queue_remaining_count,
            storage_readiness_first_proof_command=storage_readiness_first_proof_command,
            storage_handoff=storage_handoff,
            agi_next_gate=gate_metadata.get("agi_next_gate"),
            agi_next_target_title=gate_metadata.get("agi_next_target_title"),
            agi_next_build_command=gate_metadata.get("agi_next_build_command"),
            agi_next_evidence_closure_commands=gate_metadata.get("agi_next_evidence_closure_commands", []),
            agi_next_evidence_closure_command_count=gate_metadata.get("agi_next_evidence_closure_command_count", 0),
            agi_next_focused_verification_commands=gate_metadata.get("agi_next_focused_verification_commands", []),
            agi_next_focused_verification_command_count=gate_metadata.get("agi_next_focused_verification_command_count", 0),
            agi_next_likely_files=gate_metadata.get("agi_next_likely_files", []),
            agi_next_likely_file_count=gate_metadata.get("agi_next_likely_file_count", 0),
            agi_next_target_file_integrity_status=gate_metadata.get("agi_next_target_file_integrity_status", ""),
            agi_next_target_files_checked=gate_metadata.get("agi_next_target_files_checked", 0),
            agi_next_target_files_exist=gate_metadata.get("agi_next_target_files_exist", False),
            agi_next_missing_target_files=gate_metadata.get("agi_next_missing_target_files", []),
            agi_next_missing_target_file_count=gate_metadata.get("agi_next_missing_target_file_count", 0),
            agi_next_target_integrity_blocks_start=gate_metadata.get("agi_next_target_integrity_blocks_start", True),
            agi_next_acceptance_checks=gate_metadata.get("agi_next_acceptance_checks", []),
            agi_next_acceptance_check_count=gate_metadata.get("agi_next_acceptance_check_count", 0),
            **_agi_acceptance_gap_metadata(gate_metadata),
            agi_next_build_packet_ready_for_review=gate_metadata.get("agi_next_build_packet_ready_for_review", False),
            agi_next_implementation_preflight_ready=gate_metadata.get("agi_next_implementation_preflight_ready", False),
            agi_next_implementation_preflight_blockers=gate_metadata.get("agi_next_implementation_preflight_blockers", []),
            agi_next_implementation_preflight_blocker_count=gate_metadata.get("agi_next_implementation_preflight_blocker_count", 0),
            agi_next_implementation_preflight_next_command=gate_metadata.get("agi_next_implementation_preflight_next_command", ""),
            agi_focus_selection_source=gate_metadata.get("agi_focus_selection_source", ""),
            agi_focus_selection_reason=gate_metadata.get("agi_focus_selection_reason", ""),
            agi_focus_canonical_selector_command=gate_metadata.get("agi_focus_canonical_selector_command", ""),
            agi_focus_deliberate_focus_override=gate_metadata.get("agi_focus_deliberate_focus_override", False),
            agi_real_execution_gap_count=gate_metadata.get("agi_real_execution_gap_count"),
            real_execution_gap_count=real_execution_gap_count,
            selected_real_execution_gap=selected_real_execution_gap,
            selected_real_execution_gap_gate=selected_real_execution_gap_gate,
            selected_real_execution_gap_detail=selected_real_execution_gap_detail,
            draft_only=True,
            requires_manual_send=True,
            loads_without_execution=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
        )

        return ToolResult(
            "completion_next_proof_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                objective=objective,
                claim_gate_verdict=claim_state,
                completion_claim_ready=_metadata_bool(gate_metadata.get("allowed_to_claim")),
                allowed_to_claim=_metadata_bool(gate_metadata.get("allowed_to_claim")),
                blockers=gate_metadata.get("blockers"),
                blocker_details=blockers,
                completion_proof_queue=completion_queue,
                completion_proof_queue_count=len(completion_queue),
                completion_ordered_proof_queue=completion_queue,
                completion_ordered_proof_queue_count=len(completion_queue),
                completion_ordered_next_proof_command=ordered_next_command,
                completion_actionable_proof_queue=completion_actionable_queue,
                completion_actionable_proof_queue_count=len(completion_actionable_queue),
                completion_actionable_next_proof_command=next_command,
                completion_actionable_queue_includes_learning_prerequisites=learning_prerequisites_inserted,
                completion_next_proof_uses_actionable_learning_prerequisite=uses_actionable_learning_prerequisite,
                proof_queue=completion_actionable_queue,
                proof_queue_count=len(completion_actionable_queue),
                next_completion_proof_command=next_command,
                completion_next_proof_command=next_command,
                next_proof_command=next_command,
                command_source=command_source,
                command_reason=command_reason,
                first_blocker=blocker_summary,
                queue_position=queue_position,
                queue_total=len(completion_actionable_queue),
                ordered_queue_position=ordered_queue_position,
                ordered_queue_total=len(completion_queue),
                refresh_command="completion next proof",
                latest_execution_case_found=gate_metadata.get("latest_execution_case_found"),
                latest_execution_case_id=gate_metadata.get("latest_execution_case_id"),
                latest_execution_case_review_state=gate_metadata.get("latest_execution_case_review_state", ""),
                latest_execution_case_review_verdict=gate_metadata.get("latest_execution_case_review_verdict", ""),
                latest_execution_case_review_next_safe_command=gate_metadata.get("latest_execution_case_review_next_safe_command", ""),
                latest_execution_case_review_approval_required=gate_metadata.get("latest_execution_case_review_approval_required", False),
                latest_execution_case_review_checklist_items=gate_metadata.get("latest_execution_case_review_checklist_items", 0),
                latest_execution_case_review_blocker_count=gate_metadata.get("latest_execution_case_review_blocker_count", 0),
                latest_execution_case_review_handoff=gate_metadata.get("latest_execution_case_review_handoff", {}),
                latest_execution_case_review_handoff_present=gate_metadata.get("latest_execution_case_review_handoff_present", False),
                latest_execution_case_closure_verdict=latest_case_closure_verdict,
                latest_execution_case_closure_ready=latest_case_closure_ready,
                latest_execution_case_closure_blocks_completion_claim=latest_case_closure_blocks,
                latest_execution_case_closure_proof_queue=case_closure_queue,
                latest_execution_case_closure_proof_queue_count=len(case_closure_queue),
                latest_execution_case_closure_proof_queue_preview=case_closure_queue_preview,
                latest_execution_case_closure_proof_queue_preview_count=len(case_closure_queue_preview),
                latest_execution_case_closure_proof_queue_remaining_count=case_closure_queue_remaining_count,
                latest_execution_case_closure_first_proof_command=case_closure_first_proof_command,
                latest_execution_case_next_closure_proof_command=gate_metadata.get("latest_execution_case_next_closure_proof_command"),
                latest_execution_case_next_required_command=gate_metadata.get(
                    "latest_execution_case_next_required_command",
                    gate_metadata.get("latest_execution_case_next_proof_command"),
                ),
                latest_execution_case_next_proof_command=gate_metadata.get("latest_execution_case_next_proof_command"),
                latest_execution_case_next_proof_commands=case_next_proof_commands,
                latest_execution_case_mission_command_queue=case_mission_queue,
                latest_execution_case_mission_command_count=len(case_mission_queue),
                latest_execution_case_evidence_preview_gate_count=gate_metadata.get("latest_execution_case_evidence_preview_gate_count", 0),
                latest_execution_case_evidence_preview_ready_count=gate_metadata.get("latest_execution_case_evidence_preview_ready_count", 0),
                latest_execution_case_evidence_preview_blocked_count=gate_metadata.get("latest_execution_case_evidence_preview_blocked_count", 0),
                latest_execution_case_evidence_preview_verdicts=gate_metadata.get("latest_execution_case_evidence_preview_verdicts", []),
                latest_execution_case_evidence_preview_latest_verdict=gate_metadata.get("latest_execution_case_evidence_preview_latest_verdict", ""),
                latest_execution_case_evidence_preview_latest_event_id=gate_metadata.get("latest_execution_case_evidence_preview_latest_event_id"),
                latest_execution_case_evidence_preview_latest_handoff=gate_metadata.get("latest_execution_case_evidence_preview_latest_handoff", {}),
                latest_execution_case_evidence_preview_latest_handoff_present=gate_metadata.get("latest_execution_case_evidence_preview_latest_handoff_present", False),
                latest_execution_case_evidence_preview_next_command=gate_metadata.get("latest_execution_case_evidence_preview_next_command", ""),
                latest_execution_case_evidence_preview_receipt_id=gate_metadata.get("latest_execution_case_evidence_preview_receipt_id", ""),
                latest_execution_case_evidence_preview_receipt_kind=gate_metadata.get("latest_execution_case_evidence_preview_receipt_kind", ""),
                latest_execution_case_evidence_preview_receipt_target_status=gate_metadata.get("latest_execution_case_evidence_preview_receipt_target_status", ""),
                execution_health_recovery_closure_state=gate_metadata.get("execution_health_recovery_closure_state"),
                execution_health_recovery_closure_missing=gate_metadata.get("execution_health_recovery_closure_missing"),
                execution_health_recovery_closure_required_commands=gate_metadata.get("execution_health_recovery_closure_required_commands", []),
                execution_health_recovery_closure_next_required_command=gate_metadata.get(
                    "execution_health_recovery_closure_next_required_command",
                    gate_metadata.get("execution_health_recovery_closure_next_proof_command", ""),
                ),
                execution_health_recovery_closure_proof_queue=recovery_queue,
                execution_health_recovery_closure_proof_queue_count=len(recovery_queue),
                execution_health_recovery_closure_next_proof_command=gate_metadata.get("execution_health_recovery_closure_next_proof_command"),
                execution_health_recovery_closure_blocks_completion_claim=gate_metadata.get("execution_health_recovery_closure_blocks_completion_claim"),
                execution_health_verification_coverage_state=gate_metadata.get("execution_health_verification_coverage_state", ""),
                execution_health_verification_recent_tool_runs=gate_metadata.get("execution_health_verification_recent_tool_runs", 0),
                execution_health_verification_recent_action_runs=gate_metadata.get("execution_health_verification_recent_action_runs", 0),
                execution_health_verification_recent_verification_runs=gate_metadata.get("execution_health_verification_recent_verification_runs", 0),
                execution_health_verification_gap_count=gate_metadata.get("execution_health_verification_gap_count", 0),
                execution_health_verification_first_gap=gate_metadata.get("execution_health_verification_first_gap", ""),
                execution_health_verification_proof_queue=verification_queue,
                execution_health_verification_proof_queue_count=len(verification_queue),
                execution_health_verification_proof_queue_preview=gate_metadata.get("execution_health_verification_proof_queue_preview", verification_queue[:3]),
                execution_health_verification_proof_queue_preview_count=gate_metadata.get("execution_health_verification_proof_queue_preview_count", len(verification_queue[:3])),
                execution_health_verification_proof_queue_remaining_count=gate_metadata.get("execution_health_verification_proof_queue_remaining_count", max(len(verification_queue) - len(verification_queue[:3]), 0)),
                execution_health_verification_next_required_command=gate_metadata.get("execution_health_verification_next_required_command", gate_metadata.get("execution_health_verification_next_proof_command", "")),
                execution_health_verification_next_proof_command=gate_metadata.get("execution_health_verification_next_proof_command", ""),
                execution_health_verification_blocks_completion_claim=gate_metadata.get("execution_health_verification_blocks_completion_claim", False),
                execution_learning_state=gate_metadata.get("execution_learning_state"),
                execution_learning_missing=gate_metadata.get("execution_learning_missing"),
                execution_learning_closure_command=gate_metadata.get("execution_learning_closure_command"),
                execution_learning_required_commands=gate_metadata.get("execution_learning_required_commands", []),
                execution_learning_next_required_command=gate_metadata.get(
                    "execution_learning_next_required_command",
                    gate_metadata.get("execution_learning_next_proof_command", ""),
                ),
                execution_learning_proof_queue=learning_queue,
                execution_learning_proof_queue_count=len(learning_queue),
                execution_learning_next_proof_command=gate_metadata.get("execution_learning_next_proof_command"),
                execution_learning_actionable_required_commands=gate_metadata.get("execution_learning_actionable_required_commands", []),
                execution_learning_actionable_required_command_count=gate_metadata.get("execution_learning_actionable_required_command_count", 0),
                execution_learning_actionable_proof_queue=learning_actionable_queue,
                execution_learning_actionable_proof_queue_count=len(learning_actionable_queue),
                execution_learning_actionable_next_required_command=gate_metadata.get("execution_learning_actionable_next_required_command"),
                execution_learning_actionable_next_proof_command=gate_metadata.get("execution_learning_actionable_next_proof_command"),
                execution_learning_next_evidence_command=gate_metadata.get("execution_learning_next_evidence_command"),
                execution_learning_blocks_completion_claim=gate_metadata.get("execution_learning_blocks_completion_claim"),
                storage_runtime_fallback_active=_metadata_bool(gate_metadata.get("storage_runtime_fallback_active")),
                storage_runtime_fallback_reason=gate_metadata.get("storage_runtime_fallback_reason") or "",
                storage_runtime_fallback_exception_type=gate_metadata.get("storage_runtime_fallback_exception_type") or "",
                storage_runtime_fallback_db_path_display=gate_metadata.get("storage_runtime_fallback_db_path_display") or "",
                storage_runtime_fallback_vault_path_display=gate_metadata.get("storage_runtime_fallback_vault_path_display") or "",
                storage_readiness_blocks_completion_claim=_metadata_bool(gate_metadata.get("storage_readiness_blocks_completion_claim")),
                storage_recovery_required=_metadata_bool(gate_metadata.get("storage_recovery_required")),
                storage_recovery_reason=gate_metadata.get("storage_recovery_reason") or "",
                storage_issues=storage_issues,
                storage_issue_count=storage_issue_count,
                storage_recovery_mode=gate_metadata.get("storage_recovery_mode") or "none",
                storage_recovery_next_operator_action=gate_metadata.get("storage_recovery_next_operator_action") or "",
                storage_recovery_restart_required=_metadata_bool(gate_metadata.get("storage_recovery_restart_required")),
                storage_recovery_check_command=gate_metadata.get("storage_recovery_check_command") or "",
                storage_recovery_check_tool_command=gate_metadata.get("storage_recovery_check_tool_command") or "",
                storage_recovery_check_api=gate_metadata.get("storage_recovery_check_api") or STORAGE_RECOVERY_CHECK_API,
                storage_recovery_command=gate_metadata.get("storage_recovery_command") or "",
                storage_readiness_blocker=gate_metadata.get("storage_readiness_blocker") or "",
                storage_readiness_next_commands=list(gate_metadata.get("storage_readiness_next_commands") or []),
                storage_readiness_next_command_count=len(list(gate_metadata.get("storage_readiness_next_commands") or [])),
                storage_readiness_next_required_command=storage_readiness_next_required_command,
                storage_readiness_next_proof_command=storage_readiness_next_proof_command,
                storage_readiness_proof_queue=storage_readiness_proof_queue,
                storage_readiness_proof_queue_count=len(storage_readiness_proof_queue),
                storage_readiness_proof_queue_preview=storage_readiness_proof_queue_preview,
                storage_readiness_proof_queue_preview_count=len(storage_readiness_proof_queue_preview),
                storage_readiness_proof_queue_remaining_count=storage_readiness_proof_queue_remaining_count,
                storage_readiness_first_proof_command=storage_readiness_first_proof_command,
                storage_handoff=storage_handoff,
                agi_next_gate=gate_metadata.get("agi_next_gate"),
                agi_next_target_title=gate_metadata.get("agi_next_target_title"),
                agi_next_build_command=gate_metadata.get("agi_next_build_command"),
                agi_next_evidence_closure_commands=gate_metadata.get("agi_next_evidence_closure_commands", []),
                agi_next_evidence_closure_command_count=gate_metadata.get("agi_next_evidence_closure_command_count", 0),
                agi_next_focused_verification_commands=gate_metadata.get("agi_next_focused_verification_commands", []),
                agi_next_focused_verification_command_count=gate_metadata.get("agi_next_focused_verification_command_count", 0),
                agi_next_likely_files=gate_metadata.get("agi_next_likely_files", []),
                agi_next_likely_file_count=gate_metadata.get("agi_next_likely_file_count", 0),
                agi_next_target_file_integrity_status=gate_metadata.get("agi_next_target_file_integrity_status", ""),
                agi_next_target_files_checked=gate_metadata.get("agi_next_target_files_checked", 0),
                agi_next_target_files_exist=gate_metadata.get("agi_next_target_files_exist", False),
                agi_next_missing_target_files=gate_metadata.get("agi_next_missing_target_files", []),
                agi_next_missing_target_file_count=gate_metadata.get("agi_next_missing_target_file_count", 0),
                agi_next_target_integrity_blocks_start=gate_metadata.get("agi_next_target_integrity_blocks_start", True),
                agi_next_acceptance_checks=gate_metadata.get("agi_next_acceptance_checks", []),
                agi_next_acceptance_check_count=gate_metadata.get("agi_next_acceptance_check_count", 0),
                **_agi_acceptance_gap_metadata(gate_metadata),
                agi_next_build_packet_ready_for_review=gate_metadata.get("agi_next_build_packet_ready_for_review", False),
                agi_next_implementation_preflight_ready=gate_metadata.get("agi_next_implementation_preflight_ready", False),
                agi_next_implementation_preflight_blockers=gate_metadata.get("agi_next_implementation_preflight_blockers", []),
                agi_next_implementation_preflight_blocker_count=gate_metadata.get("agi_next_implementation_preflight_blocker_count", 0),
                agi_next_implementation_preflight_next_command=gate_metadata.get("agi_next_implementation_preflight_next_command", ""),
                agi_focus_selection_source=gate_metadata.get("agi_focus_selection_source", ""),
                agi_focus_selection_reason=gate_metadata.get("agi_focus_selection_reason", ""),
                agi_focus_canonical_selector_command=gate_metadata.get("agi_focus_canonical_selector_command", ""),
                agi_focus_deliberate_focus_override=gate_metadata.get("agi_focus_deliberate_focus_override", False),
                agi_real_execution_gap_count=gate_metadata.get("agi_real_execution_gap_count"),
                real_execution_gap_count=real_execution_gap_count,
                selected_real_execution_gap=selected_real_execution_gap,
                selected_real_execution_gap_gate=selected_real_execution_gap_gate,
                selected_real_execution_gap_detail=selected_real_execution_gap_detail,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                completion_next_proof_handoff_ready=True,
                completion_next_proof_handoff=completion_next_proof_handoff,
            ),
        )

    def completion_proof_refresh_packet(args: dict[str, Any]) -> ToolResult:
        objective = _short(
            args.get("objective") or args.get("goal") or args.get("request") or (
                "Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant."
            ),
            limit=500,
        )
        claim_gate = completion_claim_gate({"objective": objective})
        gate_metadata = dict(claim_gate.metadata)
        lane_status = dict(gate_metadata.get("lane_status") or {})
        lane_evidence = dict(gate_metadata.get("lane_evidence") or {})
        lane_available_tools = dict(gate_metadata.get("lane_available_tools") or {})
        next_proof_commands = dict(gate_metadata.get("next_proof_commands") or {})
        completion_queue = list(gate_metadata.get("completion_proof_queue") or [])
        agi_commands = list(gate_metadata.get("agi_next_evidence_closure_commands") or [])
        storage_readiness_next_commands = list(gate_metadata.get("storage_readiness_next_commands") or [])
        storage_runtime_fallback_active = _metadata_bool(gate_metadata.get("storage_runtime_fallback_active"))
        storage_readiness_blocks_completion_claim = _metadata_bool(gate_metadata.get("storage_readiness_blocks_completion_claim"))
        storage_recovery_required = _metadata_bool(gate_metadata.get("storage_recovery_required"))
        storage_recovery_reason = str(gate_metadata.get("storage_recovery_reason") or "").strip()
        storage_recovery_mode = str(gate_metadata.get("storage_recovery_mode") or "none").strip()
        storage_recovery_next_operator_action = str(gate_metadata.get("storage_recovery_next_operator_action") or "").strip()
        storage_recovery_restart_required = _metadata_bool(gate_metadata.get("storage_recovery_restart_required"))
        storage_recovery_check_tool_command = str(gate_metadata.get("storage_recovery_check_tool_command") or "").strip()
        storage_recovery_check_command = str(gate_metadata.get("storage_recovery_check_command") or "").strip()
        storage_recovery_check_api = str(gate_metadata.get("storage_recovery_check_api") or STORAGE_RECOVERY_CHECK_API).strip()
        storage_recovery_command = str(gate_metadata.get("storage_recovery_command") or "").strip()
        storage_issues = list(gate_metadata.get("storage_issues") or [])
        storage_issue_count = int(gate_metadata.get("storage_issue_count") or len(storage_issues))
        storage_runtime_fallback_reason = str(gate_metadata.get("storage_runtime_fallback_reason") or "").strip()
        storage_runtime_fallback_exception_type = str(gate_metadata.get("storage_runtime_fallback_exception_type") or "").strip()
        storage_runtime_fallback_db_path_display = str(gate_metadata.get("storage_runtime_fallback_db_path_display") or "").strip()
        storage_runtime_fallback_vault_path_display = str(gate_metadata.get("storage_runtime_fallback_vault_path_display") or "").strip()
        storage_readiness_blocker = str(gate_metadata.get("storage_readiness_blocker") or "").strip()
        storage_readiness_proof_queue = list(storage_readiness_next_commands) if storage_readiness_blocks_completion_claim else []
        storage_readiness_proof_queue_preview = storage_readiness_proof_queue[:5]
        storage_readiness_proof_queue_remaining_count = max(
            len(storage_readiness_proof_queue) - len(storage_readiness_proof_queue_preview),
            0,
        )
        storage_readiness_next_proof_command = (
            str(storage_readiness_proof_queue[0]) if storage_readiness_proof_queue else ""
        )
        storage_readiness_next_required_command = (
            str(gate_metadata.get("storage_readiness_next_required_command") or storage_readiness_next_proof_command)
            if storage_readiness_blocks_completion_claim
            else ""
        )
        storage_readiness_first_proof_command = storage_readiness_next_proof_command
        storage_handoff = _storage_handoff_from_metadata(
            {
                **gate_metadata,
                "storage_readiness_next_required_command": storage_readiness_next_required_command,
                "storage_readiness_next_proof_command": storage_readiness_next_proof_command,
                "storage_readiness_proof_queue": storage_readiness_proof_queue,
                "storage_readiness_proof_queue_count": len(storage_readiness_proof_queue),
                "storage_readiness_proof_queue_preview": storage_readiness_proof_queue_preview,
                "storage_readiness_proof_queue_preview_count": len(storage_readiness_proof_queue_preview),
                "storage_readiness_proof_queue_remaining_count": storage_readiness_proof_queue_remaining_count,
                "storage_readiness_first_proof_command": storage_readiness_first_proof_command,
            },
            source="completion_proof_refresh_storage",
        )
        execution_learning_proof_queue = list(gate_metadata.get("execution_learning_proof_queue") or [])
        execution_learning_state = str(gate_metadata.get("execution_learning_state") or "NO_RECENT_ACTION_RUNS")
        execution_learning_missing = list(gate_metadata.get("execution_learning_missing") or [])
        execution_learning_next_proof_command = str(gate_metadata.get("execution_learning_next_proof_command") or "").strip()
        execution_learning_blocks_completion_claim = _metadata_bool(gate_metadata.get("execution_learning_blocks_completion_claim"))
        execution_learning_target_run_id = gate_metadata.get("execution_learning_target_run_id")
        execution_learning_target_tool_name = str(gate_metadata.get("execution_learning_target_tool_name") or "").strip()
        latest_case_projection = {
            "latest_execution_case_review_state": gate_metadata.get("latest_execution_case_review_state", ""),
            "latest_execution_case_review_verdict": gate_metadata.get("latest_execution_case_review_verdict", ""),
            "latest_execution_case_review_next_safe_command": gate_metadata.get("latest_execution_case_review_next_safe_command", ""),
            "latest_execution_case_review_approval_required": gate_metadata.get("latest_execution_case_review_approval_required", False),
            "latest_execution_case_review_checklist_items": gate_metadata.get("latest_execution_case_review_checklist_items", 0),
            "latest_execution_case_review_blocker_count": gate_metadata.get("latest_execution_case_review_blocker_count", 0),
            "latest_execution_case_review_handoff": gate_metadata.get("latest_execution_case_review_handoff", {}),
            "latest_execution_case_review_handoff_present": gate_metadata.get("latest_execution_case_review_handoff_present", False),
            "latest_execution_case_evidence_preview_gate_count": gate_metadata.get("latest_execution_case_evidence_preview_gate_count", 0),
            "latest_execution_case_evidence_preview_ready_count": gate_metadata.get("latest_execution_case_evidence_preview_ready_count", 0),
            "latest_execution_case_evidence_preview_blocked_count": gate_metadata.get("latest_execution_case_evidence_preview_blocked_count", 0),
            "latest_execution_case_evidence_preview_verdicts": gate_metadata.get("latest_execution_case_evidence_preview_verdicts", []),
            "latest_execution_case_evidence_preview_latest_verdict": gate_metadata.get("latest_execution_case_evidence_preview_latest_verdict", ""),
            "latest_execution_case_evidence_preview_latest_event_id": gate_metadata.get("latest_execution_case_evidence_preview_latest_event_id"),
            "latest_execution_case_evidence_preview_latest_handoff": gate_metadata.get("latest_execution_case_evidence_preview_latest_handoff", {}),
            "latest_execution_case_evidence_preview_latest_handoff_present": gate_metadata.get("latest_execution_case_evidence_preview_latest_handoff_present", False),
            "latest_execution_case_evidence_preview_next_command": gate_metadata.get("latest_execution_case_evidence_preview_next_command", ""),
            "latest_execution_case_evidence_preview_receipt_id": gate_metadata.get("latest_execution_case_evidence_preview_receipt_id", ""),
            "latest_execution_case_evidence_preview_receipt_kind": gate_metadata.get("latest_execution_case_evidence_preview_receipt_kind", ""),
            "latest_execution_case_evidence_preview_receipt_target_status": gate_metadata.get("latest_execution_case_evidence_preview_receipt_target_status", ""),
        }

        lane_rows: list[dict[str, Any]] = []
        lane_refresh_commands: list[str] = []
        for lane, status in lane_status.items():
            command = str(next_proof_commands.get(lane) or "completion audit").strip()
            evidence = list(lane_evidence.get(lane) or [])
            available = list(lane_available_tools.get(lane) or [])
            stale = status != "strong"
            if stale and command and command not in lane_refresh_commands:
                lane_refresh_commands.append(command)
            lane_rows.append(
                {
                    "lane": lane,
                    "status": status,
                    "stale": stale,
                    "next_proof_command": command,
                    "evidence_runs": evidence,
                    "evidence_run_count": len(evidence),
                    "available_tools": available,
                    "available_tool_count": len(available),
                    "command_has_placeholder": "<" in command or ">" in command,
                }
            )

        ordered_refresh_queue: list[str] = []

        def append_refresh(command: str) -> None:
            command = str(command or "").strip()
            if command and command not in ordered_refresh_queue:
                ordered_refresh_queue.append(command)

        for command in lane_refresh_commands:
            append_refresh(command)
        for command in agi_commands:
            append_refresh(command)
        for command in completion_queue:
            append_refresh(command)
        append_refresh("completion claim gate")

        placeholder_commands = [
            command for command in ordered_refresh_queue if "<" in command or ">" in command
        ]
        concrete_commands = [
            command for command in ordered_refresh_queue if command not in placeholder_commands
        ]
        pending = store.list_pending_approvals(limit=10)
        latest_case = _latest_execution_case_proof_state()
        latest_request = str(latest_case.get("request") or objective).strip()
        latest_case_id = latest_case.get("case_id")
        latest_approval_id = pending[0]["id"] if pending else None
        recovery_proof_queue = list(
            gate_metadata.get("execution_health_recovery_closure_proof_queue")
            or latest_case.get("recovery_closure_proof_queue")
            or []
        )
        verification_fallback = next(
            (command for command in recovery_proof_queue if str(command).startswith("verification receipt ")),
            "verification receipt latest",
        )
        recovery_fallback = next(
            (command for command in recovery_proof_queue if str(command).startswith("execution recovery packet ")),
            "execution recovery packet",
        )
        approval_chain_fallback = (
            f"approval chain proof {latest_approval_id}"
            if latest_approval_id is not None
            else "approval history"
        )
        placeholder_resolution_rows: list[dict[str, Any]] = []
        for command in placeholder_commands:
            resolved_command = ""
            resolution_state = "unresolved"
            reason = "Placeholder needs a real target before it can be run."
            if command.startswith("approval readiness <id>"):
                if pending:
                    approval_id = latest_approval_id
                    resolved_command = f"approval readiness {approval_id}"
                    resolution_state = "actionable"
                    reason = f"Use pending approval #{approval_id}."
                else:
                    resolved_command = "approval history"
                    resolution_state = "fallback"
                    reason = "No pending approval id is available; inspect approval history instead."
            elif command.startswith("verification packet: <next real order>"):
                resolved_command = f"verification packet: {latest_request}"
                resolution_state = "actionable" if latest_request else "fallback"
                reason = (
                    f"Use latest execution case #{latest_case['case_id']} request."
                    if latest_case.get("found")
                    else "No saved execution case exists; use the current objective as the verification target."
                )
            elif command.startswith("dispatch decision: <next real order>"):
                resolved_command = f"dispatch decision: {latest_request}"
                resolution_state = "actionable" if latest_request else "fallback"
                reason = (
                    f"Use latest execution case #{latest_case['case_id']} request."
                    if latest_case.get("found")
                    else "No saved execution case exists; use the current objective as the dispatch target."
                )
            elif re.match(r"^case evidence \d+: verification receipt <id>", command):
                resolved_command = command.replace("verification receipt <id>", verification_fallback)
                resolution_state = "actionable" if verification_fallback != "verification receipt latest" else "fallback"
                reason = f"Use `{verification_fallback}` as the current verification receipt target."
            elif re.match(r"^case evidence \d+: approval readiness <id>", command):
                if latest_approval_id is not None:
                    resolved_command = command.replace("<id>", str(latest_approval_id))
                    resolution_state = "actionable"
                    reason = f"Use pending approval #{latest_approval_id} across the approval proof chain."
                else:
                    resolved_command = "approval history"
                    resolution_state = "fallback"
                    reason = "No pending approval id is available; inspect approval history first."
            elif command.startswith("approval chain proof <approval id>"):
                resolved_command = approval_chain_fallback
                resolution_state = "actionable" if latest_approval_id is not None else "fallback"
                reason = (
                    f"Use pending approval #{latest_approval_id} for chain proof."
                    if latest_approval_id is not None
                    else "No approval id is available; inspect approval history first."
                )
            elif re.match(r"^case evidence \d+: verification receipt <approved run id>", command):
                resolved_command = approval_chain_fallback
                resolution_state = "fallback"
                reason = "The approved run id is not known until approval chain proof is reviewed."
            elif re.match(r"^case evidence \d+: execution recovery packet <id>", command):
                resolved_command = command.replace("execution recovery packet <id>", recovery_fallback)
                resolution_state = "actionable" if recovery_fallback != "execution recovery packet" else "fallback"
                reason = f"Use `{recovery_fallback}` as the current recovery packet target."
            elif "verification receipt <approved run id from approval chain proof" in command:
                match = re.search(r"approval chain proof (\d+)", command)
                approval_id = match.group(1) if match else latest_approval_id
                resolved_command = f"approval chain proof {approval_id}" if approval_id else "approval history"
                resolution_state = "fallback"
                reason = "The approved run id is produced only after approval chain proof is reviewed."
            placeholder_resolution_rows.append(
                {
                    "placeholder_command": command,
                    "resolved_command": resolved_command,
                    "resolution_state": resolution_state,
                    "actionable": resolution_state == "actionable",
                    "reason": reason,
                    "latest_execution_case_id": latest_case_id,
                    "pending_approval_count": len(pending),
                }
            )
        placeholder_resolution_by_command = {
            row["placeholder_command"]: row["resolved_command"]
            for row in placeholder_resolution_rows
            if row["resolved_command"]
        }
        for row in lane_rows:
            command = str(row["next_proof_command"] or "")
            resolved = placeholder_resolution_by_command.get(command, command)
            row["resolved_next_proof_command"] = resolved
            row["current_evidence_ready"] = row["status"] == "strong"
            row["needs_current_evidence"] = row["status"] != "strong"
            row["refresh_actionable"] = bool(resolved) and not row["current_evidence_ready"]
            if row["status"] == "missing":
                row["proof_freshness"] = "missing_tool_surface"
            elif row["status"] == "available_without_recent_evidence":
                row["proof_freshness"] = "available_without_recent_evidence"
            elif row["status"] == "partial":
                row["proof_freshness"] = "partial_recent_evidence"
            else:
                row["proof_freshness"] = "current"
        resolved_refresh_queue = [
            placeholder_resolution_by_command.get(command, command)
            for command in ordered_refresh_queue
        ]
        actionable_placeholder_count = sum(1 for row in placeholder_resolution_rows if row["actionable"])
        unresolved_placeholder_count = sum(
            1 for row in placeholder_resolution_rows
            if row["resolution_state"] == "unresolved"
        )
        stale_lanes = [row for row in lane_rows if row["stale"]]
        stale_lane_names = [str(row["lane"]) for row in stale_lanes]
        available_without_recent_evidence_lanes = [
            str(row["lane"]) for row in lane_rows if row["status"] == "available_without_recent_evidence"
        ]
        partial_evidence_lanes = [
            str(row["lane"]) for row in lane_rows if row["status"] == "partial"
        ]
        missing_evidence_lanes = [
            str(row["lane"]) for row in lane_rows if row["status"] == "missing"
        ]
        lane_current_evidence_worklist = [
            row for row in lane_rows if row["needs_current_evidence"]
        ]
        first_command = ordered_refresh_queue[0] if ordered_refresh_queue else "completion claim gate"
        first_resolved_command = resolved_refresh_queue[0] if resolved_refresh_queue else first_command
        first_lane_refresh_command = lane_refresh_commands[0] if lane_refresh_commands else ""
        first_resolved_lane_refresh_command = (
            placeholder_resolution_by_command.get(first_lane_refresh_command, first_lane_refresh_command)
            if first_lane_refresh_command
            else ""
        )
        claim_verdict = str(gate_metadata.get("verdict") or "UNKNOWN")

        lines = [
            "Jarvis completion proof refresh packet:",
            "This is read-only. It refreshes the proof map across completion lanes without running proof commands or weakening the completion claim gate.",
            "",
            f"Objective: {objective}",
            f"Claim gate verdict: {claim_verdict}",
            f"Completion claim ready: {'yes' if gate_metadata.get('allowed_to_claim') else 'no'}",
            f"Stale proof lanes: {len(stale_lanes)}",
            f"Lanes needing current evidence: {len(lane_current_evidence_worklist)}",
            f"First refresh command: `{first_command}`",
            f"First resolved command: `{first_resolved_command}`",
            f"First lane proof command: `{first_lane_refresh_command}`" if first_lane_refresh_command else "First lane proof command: none",
            f"First resolved lane proof command: `{first_resolved_lane_refresh_command}`" if first_resolved_lane_refresh_command else "First resolved lane proof command: none",
            "",
            "Lane refresh map:",
        ]
        for row in lane_rows:
            lines.extend(
                [
                    f"- {row['lane']}: {row['status']}",
                    f"  next proof: `{row['next_proof_command']}`",
                    f"  resolved next proof: `{row['resolved_next_proof_command']}`",
                    f"  proof freshness: {row['proof_freshness']}",
                    f"  evidence runs: {', '.join(row['evidence_runs']) if row['evidence_runs'] else 'none'}",
                    f"  available tools: {', '.join(row['available_tools'][:6]) if row['available_tools'] else 'none'}",
                ]
            )
        lines.extend(
            [
                "",
                "Current-evidence worklist:",
            ]
        )
        if lane_current_evidence_worklist:
            lines.extend(
                f"- {row['lane']}: `{row['resolved_next_proof_command']}` ({row['proof_freshness']})"
                for row in lane_current_evidence_worklist
            )
        else:
            lines.append("- none; lane evidence is current enough for the completion gate")
        lines.extend(
            [
                "",
                "Refresh queues:",
                f"- lane refresh commands: {', '.join(f'`{command}`' for command in lane_refresh_commands) if lane_refresh_commands else 'none'}",
                f"- AGI selected closure commands: {', '.join(f'`{command}`' for command in agi_commands) if agi_commands else 'none'}",
                f"- completion gate queue: {', '.join(f'`{command}`' for command in completion_queue) if completion_queue else 'none'}",
                "",
                "Ordered refresh queue:",
            ]
        )
        lines.extend(f"- `{command}`" for command in ordered_refresh_queue)
        lines.extend(
            [
                "",
                "Storage recovery:",
                f"- runtime fallback active: {'yes' if storage_runtime_fallback_active else 'no'}",
                f"- blocks completion claim: {'yes' if storage_readiness_blocks_completion_claim else 'no'}",
                f"- recovery required: {'yes' if storage_recovery_required else 'no'}",
                f"- recovery reason: {storage_recovery_reason or 'none'}",
                f"- recovery mode: {storage_recovery_mode or 'none'}",
                f"- restart required: {'yes' if storage_recovery_restart_required else 'no'}",
                f"- next operator action: {storage_recovery_next_operator_action or 'none'}",
                f"- recovery queue: {', '.join(f'`{command}`' for command in storage_readiness_next_commands) if storage_readiness_next_commands else 'none'}",
            ]
        )
        lines.extend(
            [
                "",
                "Execution learning:",
                f"- state: {execution_learning_state}",
                f"- blocks completion claim: {'yes' if execution_learning_blocks_completion_claim else 'no'}",
                f"- target run: #{execution_learning_target_run_id} `{execution_learning_target_tool_name}`" if execution_learning_target_run_id is not None else "- target run: none",
                f"- missing: {', '.join(str(item) for item in execution_learning_missing) if execution_learning_missing else 'none'}",
                f"- learning proof queue: {', '.join(f'`{command}`' for command in execution_learning_proof_queue) if execution_learning_proof_queue else 'none'}",
                "",
                "Latest execution case review:",
                f"- review state: {latest_case_projection['latest_execution_case_review_state'] or 'none'}",
                f"- review verdict: {latest_case_projection['latest_execution_case_review_verdict'] or 'none'}",
                f"- review next safe command: `{latest_case_projection['latest_execution_case_review_next_safe_command']}`" if latest_case_projection["latest_execution_case_review_next_safe_command"] else "- review next safe command: none",
                f"- evidence preflight events: {latest_case_projection['latest_execution_case_evidence_preview_gate_count']}",
                f"- evidence preflight ready events: {latest_case_projection['latest_execution_case_evidence_preview_ready_count']}",
                f"- evidence preflight blocked events: {latest_case_projection['latest_execution_case_evidence_preview_blocked_count']}",
                f"- evidence preflight latest verdict: {latest_case_projection['latest_execution_case_evidence_preview_latest_verdict'] or 'none'}",
            ]
        )
        lines.extend(
            [
                "",
                "Command readiness:",
                f"- concrete commands: {len(concrete_commands)}",
                f"- placeholder commands needing a real target/id/order: {len(placeholder_commands)}",
            ]
        )
        if placeholder_commands:
            lines.extend(f"- placeholder: `{command}`" for command in placeholder_commands)
        if placeholder_resolution_rows:
            lines.append("")
            lines.append("Placeholder resolution:")
            lines.extend(
                f"- `{row['placeholder_command']}` -> `{row['resolved_command']}` ({row['resolution_state']})"
                for row in placeholder_resolution_rows
            )
        lines.extend(
            [
                "",
                "Boundary:",
                "- This packet does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, write notes, write the database, control the computer, call external services, speak, complete tasks, mark goals done, or queue approvals.",
            ]
        )
        completion_proof_refresh_handoff = _safe_metadata(
            source="completion_proof_refresh_packet",
            completion_proof_refresh_handoff_ready=True,
            handoff_ready=True,
            objective=objective,
            claim_gate_verdict=claim_verdict,
            completion_claim_ready=_metadata_bool(gate_metadata.get("allowed_to_claim")),
            allowed_to_claim=_metadata_bool(gate_metadata.get("allowed_to_claim")),
            stale_lane_count=len(stale_lanes),
            stale_lane_names=stale_lane_names,
            available_without_recent_evidence_lanes=available_without_recent_evidence_lanes,
            available_without_recent_evidence_lane_count=len(available_without_recent_evidence_lanes),
            partial_evidence_lanes=partial_evidence_lanes,
            partial_evidence_lane_count=len(partial_evidence_lanes),
            missing_evidence_lanes=missing_evidence_lanes,
            missing_evidence_lane_count=len(missing_evidence_lanes),
            lane_current_evidence_worklist=lane_current_evidence_worklist,
            lane_current_evidence_worklist_count=len(lane_current_evidence_worklist),
            lane_count=len(lane_rows),
            lane_rows=lane_rows,
            lane_status=lane_status,
            lane_refresh_commands=lane_refresh_commands,
            lane_refresh_command_count=len(lane_refresh_commands),
            ordered_refresh_queue=ordered_refresh_queue,
            ordered_refresh_queue_count=len(ordered_refresh_queue),
            resolved_refresh_queue=resolved_refresh_queue,
            resolved_refresh_queue_count=len(resolved_refresh_queue),
            first_refresh_command=first_command,
            first_resolved_refresh_command=first_resolved_command,
            first_lane_refresh_command=first_lane_refresh_command,
            first_resolved_lane_refresh_command=first_resolved_lane_refresh_command,
            concrete_refresh_commands=concrete_commands,
            concrete_refresh_command_count=len(concrete_commands),
            placeholder_refresh_commands=placeholder_commands,
            placeholder_refresh_command_count=len(placeholder_commands),
            placeholder_resolution_rows=placeholder_resolution_rows,
            placeholder_resolution_count=len(placeholder_resolution_rows),
            actionable_placeholder_count=actionable_placeholder_count,
            unresolved_placeholder_count=unresolved_placeholder_count,
            completion_proof_queue=completion_queue,
            completion_proof_queue_count=len(completion_queue),
            storage_runtime_fallback_reason=storage_runtime_fallback_reason,
            storage_runtime_fallback_exception_type=storage_runtime_fallback_exception_type,
            storage_runtime_fallback_db_path_display=storage_runtime_fallback_db_path_display,
            storage_runtime_fallback_vault_path_display=storage_runtime_fallback_vault_path_display,
            storage_runtime_fallback_active=storage_runtime_fallback_active,
            storage_readiness_blocks_completion_claim=storage_readiness_blocks_completion_claim,
            storage_readiness_blocker=storage_readiness_blocker,
            storage_recovery_required=storage_recovery_required,
            storage_recovery_reason=storage_recovery_reason,
            storage_recovery_mode=storage_recovery_mode,
            storage_recovery_next_operator_action=storage_recovery_next_operator_action,
            storage_recovery_restart_required=storage_recovery_restart_required,
            storage_issues=storage_issues,
            storage_issue_count=storage_issue_count,
            storage_readiness_next_commands=storage_readiness_next_commands,
            storage_readiness_next_command_count=len(storage_readiness_next_commands),
            storage_readiness_next_required_command=storage_readiness_next_required_command,
            storage_readiness_next_proof_command=storage_readiness_next_proof_command,
            storage_readiness_proof_queue=storage_readiness_proof_queue,
            storage_readiness_proof_queue_count=len(storage_readiness_proof_queue),
            storage_readiness_proof_queue_preview=storage_readiness_proof_queue_preview,
            storage_readiness_proof_queue_preview_count=len(storage_readiness_proof_queue_preview),
            storage_readiness_proof_queue_remaining_count=storage_readiness_proof_queue_remaining_count,
            storage_readiness_first_proof_command=storage_readiness_first_proof_command,
            storage_recovery_check_tool_command=storage_recovery_check_tool_command,
            storage_recovery_check_command=storage_recovery_check_command,
            storage_recovery_check_api=storage_recovery_check_api,
            storage_recovery_command=storage_recovery_command,
            storage_handoff=storage_handoff,
            execution_learning_state=execution_learning_state,
            execution_learning_missing=execution_learning_missing,
            execution_learning_missing_count=len(execution_learning_missing),
            execution_learning_target_run_id=execution_learning_target_run_id,
            execution_learning_target_tool_name=execution_learning_target_tool_name,
            execution_learning_proof_queue=execution_learning_proof_queue,
            execution_learning_proof_queue_count=len(execution_learning_proof_queue),
            execution_learning_next_proof_command=execution_learning_next_proof_command,
            execution_learning_blocks_completion_claim=execution_learning_blocks_completion_claim,
            **latest_case_projection,
            next_proof_commands=next_proof_commands,
            agi_next_gate=gate_metadata.get("agi_next_gate"),
            agi_real_execution_gap_count=gate_metadata.get("agi_real_execution_gap_count"),
            agi_next_target_title=gate_metadata.get("agi_next_target_title"),
            agi_next_build_command=gate_metadata.get("agi_next_build_command"),
            agi_next_evidence_closure_commands=agi_commands,
            agi_next_evidence_closure_command_count=len(agi_commands),
            agi_next_focused_verification_commands=gate_metadata.get("agi_next_focused_verification_commands", []),
            agi_next_focused_verification_command_count=gate_metadata.get("agi_next_focused_verification_command_count", 0),
            agi_next_likely_files=gate_metadata.get("agi_next_likely_files", []),
            agi_next_likely_file_count=gate_metadata.get("agi_next_likely_file_count", 0),
            agi_next_target_file_integrity_status=gate_metadata.get("agi_next_target_file_integrity_status", ""),
            agi_next_target_files_checked=gate_metadata.get("agi_next_target_files_checked", 0),
            agi_next_target_files_exist=gate_metadata.get("agi_next_target_files_exist", False),
            agi_next_missing_target_files=gate_metadata.get("agi_next_missing_target_files", []),
            agi_next_missing_target_file_count=gate_metadata.get("agi_next_missing_target_file_count", 0),
            agi_next_target_integrity_blocks_start=gate_metadata.get("agi_next_target_integrity_blocks_start", True),
            agi_next_acceptance_checks=gate_metadata.get("agi_next_acceptance_checks", []),
            agi_next_acceptance_check_count=gate_metadata.get("agi_next_acceptance_check_count", 0),
            **_agi_acceptance_gap_metadata(gate_metadata),
            agi_next_build_packet_ready_for_review=gate_metadata.get("agi_next_build_packet_ready_for_review", False),
            agi_next_implementation_preflight_ready=gate_metadata.get("agi_next_implementation_preflight_ready", False),
            agi_next_implementation_preflight_blockers=gate_metadata.get("agi_next_implementation_preflight_blockers", []),
            agi_next_implementation_preflight_blocker_count=gate_metadata.get("agi_next_implementation_preflight_blocker_count", 0),
            agi_next_implementation_preflight_next_command=gate_metadata.get("agi_next_implementation_preflight_next_command", ""),
            agi_focus_selection_source=gate_metadata.get("agi_focus_selection_source", ""),
            agi_focus_selection_reason=gate_metadata.get("agi_focus_selection_reason", ""),
            agi_focus_canonical_selector_command=gate_metadata.get("agi_focus_canonical_selector_command", ""),
            agi_focus_deliberate_focus_override=gate_metadata.get("agi_focus_deliberate_focus_override", False),
            blocker_details=gate_metadata.get("blocker_details", []),
            completion_blocker_count=gate_metadata.get("completion_blocker_count", 0),
            refresh_after_command="completion proof refresh",
            draft_only=True,
            requires_manual_send=True,
            loads_without_execution=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
        )

        return ToolResult(
            "completion_proof_refresh_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                objective=objective,
                claim_gate_verdict=claim_verdict,
                completion_claim_ready=_metadata_bool(gate_metadata.get("allowed_to_claim")),
                allowed_to_claim=_metadata_bool(gate_metadata.get("allowed_to_claim")),
                stale_lane_count=len(stale_lanes),
                stale_lane_names=stale_lane_names,
                available_without_recent_evidence_lanes=available_without_recent_evidence_lanes,
                available_without_recent_evidence_lane_count=len(available_without_recent_evidence_lanes),
                partial_evidence_lanes=partial_evidence_lanes,
                partial_evidence_lane_count=len(partial_evidence_lanes),
                missing_evidence_lanes=missing_evidence_lanes,
                missing_evidence_lane_count=len(missing_evidence_lanes),
                lane_current_evidence_worklist=lane_current_evidence_worklist,
                lane_current_evidence_worklist_count=len(lane_current_evidence_worklist),
                lane_count=len(lane_rows),
                lane_rows=lane_rows,
                lane_status=lane_status,
                lane_refresh_commands=lane_refresh_commands,
                lane_refresh_command_count=len(lane_refresh_commands),
                agi_next_gate=gate_metadata.get("agi_next_gate"),
                agi_real_execution_gap_count=gate_metadata.get("agi_real_execution_gap_count"),
                agi_next_target_title=gate_metadata.get("agi_next_target_title"),
                agi_next_build_command=gate_metadata.get("agi_next_build_command"),
                agi_next_evidence_closure_commands=agi_commands,
                agi_next_evidence_closure_command_count=len(agi_commands),
                agi_next_focused_verification_commands=gate_metadata.get("agi_next_focused_verification_commands", []),
                agi_next_focused_verification_command_count=gate_metadata.get("agi_next_focused_verification_command_count", 0),
                agi_next_likely_files=gate_metadata.get("agi_next_likely_files", []),
                agi_next_likely_file_count=gate_metadata.get("agi_next_likely_file_count", 0),
                agi_next_target_file_integrity_status=gate_metadata.get("agi_next_target_file_integrity_status", ""),
                agi_next_target_files_checked=gate_metadata.get("agi_next_target_files_checked", 0),
                agi_next_target_files_exist=gate_metadata.get("agi_next_target_files_exist", False),
                agi_next_missing_target_files=gate_metadata.get("agi_next_missing_target_files", []),
                agi_next_missing_target_file_count=gate_metadata.get("agi_next_missing_target_file_count", 0),
                agi_next_target_integrity_blocks_start=gate_metadata.get("agi_next_target_integrity_blocks_start", True),
                agi_next_acceptance_checks=gate_metadata.get("agi_next_acceptance_checks", []),
                agi_next_acceptance_check_count=gate_metadata.get("agi_next_acceptance_check_count", 0),
                **_agi_acceptance_gap_metadata(gate_metadata),
                agi_next_build_packet_ready_for_review=gate_metadata.get("agi_next_build_packet_ready_for_review", False),
                agi_next_implementation_preflight_ready=gate_metadata.get("agi_next_implementation_preflight_ready", False),
                agi_next_implementation_preflight_blockers=gate_metadata.get("agi_next_implementation_preflight_blockers", []),
                agi_next_implementation_preflight_blocker_count=gate_metadata.get("agi_next_implementation_preflight_blocker_count", 0),
                agi_next_implementation_preflight_next_command=gate_metadata.get("agi_next_implementation_preflight_next_command", ""),
                agi_focus_selection_source=gate_metadata.get("agi_focus_selection_source", ""),
                agi_focus_selection_reason=gate_metadata.get("agi_focus_selection_reason", ""),
                agi_focus_canonical_selector_command=gate_metadata.get("agi_focus_canonical_selector_command", ""),
                agi_focus_deliberate_focus_override=gate_metadata.get("agi_focus_deliberate_focus_override", False),
                completion_proof_queue=completion_queue,
                completion_proof_queue_count=len(completion_queue),
                storage_runtime_fallback_active=storage_runtime_fallback_active,
                storage_runtime_fallback_reason=storage_runtime_fallback_reason,
                storage_runtime_fallback_exception_type=storage_runtime_fallback_exception_type,
                storage_runtime_fallback_db_path_display=storage_runtime_fallback_db_path_display,
                storage_runtime_fallback_vault_path_display=storage_runtime_fallback_vault_path_display,
                storage_readiness_blocks_completion_claim=storage_readiness_blocks_completion_claim,
                storage_readiness_blocker=storage_readiness_blocker,
                storage_recovery_required=storage_recovery_required,
                storage_recovery_reason=storage_recovery_reason,
                storage_recovery_mode=storage_recovery_mode,
                storage_recovery_next_operator_action=storage_recovery_next_operator_action,
                storage_recovery_restart_required=storage_recovery_restart_required,
                storage_issues=storage_issues,
                storage_issue_count=storage_issue_count,
                storage_readiness_next_commands=storage_readiness_next_commands,
                storage_readiness_next_command_count=len(storage_readiness_next_commands),
                storage_readiness_next_proof_command=storage_readiness_next_proof_command,
                storage_readiness_proof_queue=storage_readiness_proof_queue,
                storage_readiness_proof_queue_count=len(storage_readiness_proof_queue),
                storage_readiness_proof_queue_preview=storage_readiness_proof_queue_preview,
                storage_readiness_proof_queue_preview_count=len(storage_readiness_proof_queue_preview),
                storage_readiness_proof_queue_remaining_count=storage_readiness_proof_queue_remaining_count,
                storage_readiness_first_proof_command=storage_readiness_first_proof_command,
                storage_recovery_check_tool_command=storage_recovery_check_tool_command,
                storage_recovery_check_command=storage_recovery_check_command,
                storage_recovery_check_api=storage_recovery_check_api,
                storage_recovery_command=storage_recovery_command,
                storage_handoff=storage_handoff,
                execution_learning_state=execution_learning_state,
                execution_learning_missing=execution_learning_missing,
                execution_learning_missing_count=len(execution_learning_missing),
                execution_learning_target_run_id=execution_learning_target_run_id,
                execution_learning_target_tool_name=execution_learning_target_tool_name,
                execution_learning_proof_queue=execution_learning_proof_queue,
                execution_learning_proof_queue_count=len(execution_learning_proof_queue),
                execution_learning_next_proof_command=execution_learning_next_proof_command,
                execution_learning_blocks_completion_claim=execution_learning_blocks_completion_claim,
                **latest_case_projection,
                ordered_refresh_queue=ordered_refresh_queue,
                ordered_refresh_queue_count=len(ordered_refresh_queue),
                resolved_refresh_queue=resolved_refresh_queue,
                resolved_refresh_queue_count=len(resolved_refresh_queue),
                first_refresh_command=first_command,
                first_resolved_refresh_command=first_resolved_command,
                first_lane_refresh_command=first_lane_refresh_command,
                first_resolved_lane_refresh_command=first_resolved_lane_refresh_command,
                concrete_refresh_commands=concrete_commands,
                concrete_refresh_command_count=len(concrete_commands),
                placeholder_refresh_commands=placeholder_commands,
                placeholder_refresh_command_count=len(placeholder_commands),
                placeholder_resolution_rows=placeholder_resolution_rows,
                placeholder_resolution_count=len(placeholder_resolution_rows),
                actionable_placeholder_count=actionable_placeholder_count,
                unresolved_placeholder_count=unresolved_placeholder_count,
                next_proof_commands=next_proof_commands,
                blocker_details=gate_metadata.get("blocker_details", []),
                completion_blocker_count=gate_metadata.get("completion_blocker_count", 0),
                refresh_after_command="completion proof refresh",
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                claim_gate_metadata=gate_metadata,
                completion_proof_refresh_handoff_ready=True,
                completion_proof_refresh_handoff=completion_proof_refresh_handoff,
            ),
        )

    def operator_handoff_packet(args: dict[str, Any]) -> ToolResult:
        objective = _short(
            args.get("objective") or args.get("goal") or args.get("request") or (
                "Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant."
            ),
            limit=500,
        )
        proof = completion_next_proof_packet({"objective": objective})
        metadata = dict(proof.metadata)
        completion_next_proof_handoff = dict(metadata.get("completion_next_proof_handoff") or {})
        command = str(metadata.get("completion_next_proof_command") or metadata.get("next_proof_command") or "").strip()
        proof_queue = list(metadata.get("completion_actionable_proof_queue") or metadata.get("proof_queue") or metadata.get("completion_proof_queue") or [])
        ordered_proof_queue = list(metadata.get("completion_proof_queue") or [])
        command_source = str(metadata.get("command_source") or "unknown")
        command_reason = str(metadata.get("command_reason") or "completion proof queue selected this as the next ordered proof")
        first_blocker = str(metadata.get("first_blocker") or "none")
        queue_position = _metadata_int(
            metadata.get("queue_position"),
            proof_queue.index(command) + 1 if command in proof_queue else 0,
        )
        queue_total = _metadata_int(metadata.get("queue_total"), len(proof_queue))
        refresh_command = str(metadata.get("refresh_command") or "completion next proof")
        latest_case_verdict = str(metadata.get("latest_execution_case_closure_verdict") or "none")
        latest_case_closure_ready = _metadata_bool(metadata.get("latest_execution_case_closure_ready"))
        latest_case_closure_blocks_completion_claim = _metadata_bool(metadata.get("latest_execution_case_closure_blocks_completion_claim"))
        latest_case_closure_queue = list(metadata.get("latest_execution_case_closure_proof_queue") or [])
        latest_case_review_state = str(metadata.get("latest_execution_case_review_state") or "")
        latest_case_review_verdict = str(metadata.get("latest_execution_case_review_verdict") or "")
        latest_case_review_next_safe_command = str(metadata.get("latest_execution_case_review_next_safe_command") or "")
        latest_case_review_handoff = dict(metadata.get("latest_execution_case_review_handoff") or {})
        latest_case_review_handoff_present = _metadata_bool(metadata.get("latest_execution_case_review_handoff_present"))
        latest_case_evidence_preview_gate_count = int(metadata.get("latest_execution_case_evidence_preview_gate_count") or 0)
        latest_case_evidence_preview_ready_count = int(metadata.get("latest_execution_case_evidence_preview_ready_count") or 0)
        latest_case_evidence_preview_blocked_count = int(metadata.get("latest_execution_case_evidence_preview_blocked_count") or 0)
        latest_case_evidence_preview_verdicts = list(metadata.get("latest_execution_case_evidence_preview_verdicts") or [])
        latest_case_evidence_preview_latest_verdict = str(metadata.get("latest_execution_case_evidence_preview_latest_verdict") or "")
        latest_case_evidence_preview_latest_event_id = metadata.get("latest_execution_case_evidence_preview_latest_event_id")
        latest_case_evidence_preview_latest_handoff = dict(metadata.get("latest_execution_case_evidence_preview_latest_handoff") or {})
        latest_case_evidence_preview_latest_handoff_present = _metadata_bool(metadata.get("latest_execution_case_evidence_preview_latest_handoff_present"))
        latest_case_evidence_preview_next_command = str(
            latest_case_evidence_preview_latest_handoff.get("next_command")
            or latest_case_evidence_preview_latest_handoff.get("append_command")
            or ""
        )
        latest_case_evidence_preview_receipt_id = str(
            latest_case_evidence_preview_latest_handoff.get("receipt_id")
            or latest_case_evidence_preview_latest_handoff.get("inferred_receipt_id")
            or ""
        )
        latest_case_evidence_preview_receipt_kind = str(
            latest_case_evidence_preview_latest_handoff.get("receipt_kind")
            or latest_case_evidence_preview_latest_handoff.get("inferred_receipt_kind")
            or ""
        )
        latest_case_evidence_preview_receipt_target_status = str(
            latest_case_evidence_preview_latest_handoff.get("receipt_target_status") or ""
        )
        latest_case_next = str(
            metadata.get("latest_execution_case_next_closure_proof_command")
            or (latest_case_closure_queue[0] if latest_case_closure_queue else "")
        )
        if latest_case_next and not latest_case_closure_queue:
            latest_case_closure_queue = [latest_case_next]
        latest_case_closure_queue_preview = latest_case_closure_queue[:5]
        latest_case_closure_queue_remaining_count = max(
            len(latest_case_closure_queue) - len(latest_case_closure_queue_preview),
            0,
        )
        latest_case_closure_first_proof_command = (
            str(latest_case_closure_queue_preview[0])
            if latest_case_closure_queue_preview
            else ""
        )
        recovery_state = str(metadata.get("execution_health_recovery_closure_state") or "unknown")
        recovery_missing = list(metadata.get("execution_health_recovery_closure_missing") or [])
        recovery_required_commands = list(metadata.get("execution_health_recovery_closure_required_commands") or [])
        recovery_next_required = str(
            metadata.get("execution_health_recovery_closure_next_required_command")
            or (recovery_required_commands[0] if recovery_required_commands else "")
            or metadata.get("execution_health_recovery_closure_next_proof_command")
            or ""
        )
        recovery_next = str(metadata.get("execution_health_recovery_closure_next_proof_command") or "")
        recovery_queue = list(metadata.get("execution_health_recovery_closure_proof_queue") or [])
        recovery_blocks_completion_claim = _metadata_bool(metadata.get("execution_health_recovery_closure_blocks_completion_claim"))
        verification_state = str(metadata.get("execution_health_verification_coverage_state") or "unknown")
        verification_queue = list(metadata.get("execution_health_verification_proof_queue") or [])
        verification_next = str(
            metadata.get("execution_health_verification_next_proof_command")
            or (verification_queue[0] if verification_queue else "")
        )
        verification_next_required = str(
            metadata.get("execution_health_verification_next_required_command")
            or verification_next
        )
        verification_concrete_next = next(
            (
                str(command).strip()
                for command in proof_queue
                if re.match(r"^verification receipt \d+$", str(command).strip())
            ),
            "",
        )
        verification_row_next = verification_concrete_next or verification_next
        verification_blocks_completion_claim = _metadata_bool(metadata.get("execution_health_verification_blocks_completion_claim"))
        verification_recent_tool_runs = int(metadata.get("execution_health_verification_recent_tool_runs") or 0)
        verification_recent_action_runs = int(metadata.get("execution_health_verification_recent_action_runs") or 0)
        verification_recent_verification_runs = int(metadata.get("execution_health_verification_recent_verification_runs") or 0)
        verification_queue_preview = verification_queue[:3]
        verification_queue_remaining_count = max(len(verification_queue) - len(verification_queue_preview), 0)
        verification_gap_count = max(verification_recent_tool_runs - verification_recent_verification_runs, 0)
        verification_first_gap = ""
        if verification_state == "no_recent_tool_runs":
            verification_first_gap = "no recent tool runs have been recorded"
        elif verification_blocks_completion_claim:
            verification_first_gap = (
                "recent tool activity has no verification receipt"
                if verification_gap_count
                else "verification receipt is required before completion"
            )
        learning_state = str(metadata.get("execution_learning_state") or "unknown")
        learning_missing = list(metadata.get("execution_learning_missing") or [])
        learning_closure_command = str(metadata.get("execution_learning_closure_command") or "")
        learning_required_commands = list(metadata.get("execution_learning_required_commands") or [])
        learning_next_required = str(
            metadata.get("execution_learning_next_required_command")
            or (learning_required_commands[0] if learning_required_commands else "")
            or metadata.get("execution_learning_next_proof_command")
            or ""
        )
        learning_next = str(metadata.get("execution_learning_next_proof_command") or "")
        learning_queue = list(metadata.get("execution_learning_proof_queue") or [])
        learning_blocks_completion_claim = _metadata_bool(metadata.get("execution_learning_blocks_completion_claim"))
        learning_actionable_required_commands = list(metadata.get("execution_learning_actionable_required_commands") or [])
        learning_actionable_queue = list(metadata.get("execution_learning_actionable_proof_queue") or [])
        learning_actionable_next_required = str(metadata.get("execution_learning_actionable_next_required_command") or "")
        learning_actionable_next = str(
            metadata.get("execution_learning_actionable_next_proof_command")
            or (learning_actionable_queue[0] if learning_actionable_queue else "")
        )
        storage_runtime_fallback_active = _metadata_bool(metadata.get("storage_runtime_fallback_active"))
        storage_readiness_blocks_completion_claim = _metadata_bool(metadata.get("storage_readiness_blocks_completion_claim"))
        storage_readiness_next_commands = list(metadata.get("storage_readiness_next_commands") or [])
        storage_readiness_proof_queue = list(metadata.get("storage_readiness_proof_queue") or storage_readiness_next_commands)
        storage_readiness_proof_queue_preview = list(
            metadata.get("storage_readiness_proof_queue_preview")
            or storage_readiness_proof_queue[:5]
        )
        storage_readiness_proof_queue_remaining_count = int(
            metadata.get(
                "storage_readiness_proof_queue_remaining_count",
                max(len(storage_readiness_proof_queue) - len(storage_readiness_proof_queue_preview), 0),
            )
        )
        storage_recovery_mode = str(metadata.get("storage_recovery_mode") or "none")
        storage_recovery_next_operator_action = str(metadata.get("storage_recovery_next_operator_action") or "")
        storage_recovery_restart_required = _metadata_bool(metadata.get("storage_recovery_restart_required"))
        storage_readiness_next_proof_command = str(
            metadata.get("storage_readiness_next_proof_command")
            or (
                storage_readiness_next_commands[0]
                if storage_readiness_blocks_completion_claim and storage_readiness_next_commands
                else ""
            )
        )
        storage_readiness_next_required_command = str(
            metadata.get("storage_readiness_next_required_command")
            or storage_readiness_next_proof_command
        )
        storage_readiness_first_proof_command = str(
            metadata.get("storage_readiness_first_proof_command")
            or (storage_readiness_proof_queue[0] if storage_readiness_proof_queue else "")
        )
        storage_readiness_blocker = str(metadata.get("storage_readiness_blocker") or "")
        storage_issues = list(metadata.get("storage_issues") or [])
        storage_issue_count = int(metadata.get("storage_issue_count") or len(storage_issues))
        storage_issue_preview = [str(issue) for issue in storage_issues[:3]]
        storage_first_issue = storage_issue_preview[0] if storage_issue_preview else ""
        storage_handoff = _storage_handoff_from_metadata(
            {
                **metadata,
                "storage_readiness_next_commands": storage_readiness_next_commands,
                "storage_readiness_next_command_count": len(storage_readiness_next_commands),
                "storage_readiness_next_required_command": storage_readiness_next_required_command,
                "storage_readiness_next_proof_command": storage_readiness_next_proof_command,
                "storage_readiness_proof_queue": storage_readiness_proof_queue,
                "storage_readiness_proof_queue_count": len(storage_readiness_proof_queue),
                "storage_readiness_proof_queue_preview": storage_readiness_proof_queue_preview,
                "storage_readiness_proof_queue_preview_count": int(
                    metadata.get("storage_readiness_proof_queue_preview_count", len(storage_readiness_proof_queue_preview))
                ),
                "storage_readiness_proof_queue_remaining_count": storage_readiness_proof_queue_remaining_count,
                "storage_readiness_first_proof_command": storage_readiness_first_proof_command,
            },
            source="operator_handoff_storage",
        )
        agi_next_gate = str(metadata.get("agi_next_gate") or "")
        agi_next_target_title = str(metadata.get("agi_next_target_title") or "")
        agi_next_build_command = str(metadata.get("agi_next_build_command") or "")
        agi_next_likely_files = list(metadata.get("agi_next_likely_files") or [])
        agi_next_acceptance_checks = list(metadata.get("agi_next_acceptance_checks") or [])
        agi_next_focused_verification_commands = list(metadata.get("agi_next_focused_verification_commands") or [])
        agi_next_target_file_integrity_status = str(metadata.get("agi_next_target_file_integrity_status") or "unknown")
        agi_next_missing_target_files = list(metadata.get("agi_next_missing_target_files") or [])
        agi_next_build_packet_ready = _metadata_bool(metadata.get("agi_next_build_packet_ready_for_review"))
        agi_focus_selection_source = str(metadata.get("agi_focus_selection_source") or "")
        agi_focus_selection_reason = str(metadata.get("agi_focus_selection_reason") or "")
        agi_focus_canonical_selector_command = str(metadata.get("agi_focus_canonical_selector_command") or "")
        agi_focus_deliberate_focus_override = _metadata_bool(metadata.get("agi_focus_deliberate_focus_override"))
        agi_next_implementation_preflight_ready = _metadata_bool(metadata.get("agi_next_implementation_preflight_ready"))
        agi_next_implementation_preflight_blockers = list(metadata.get("agi_next_implementation_preflight_blockers") or [])
        agi_next_implementation_preflight_blocker_preview = agi_next_implementation_preflight_blockers[:3]
        agi_next_implementation_preflight_first_blocker = (
            str(agi_next_implementation_preflight_blocker_preview[0])
            if agi_next_implementation_preflight_blocker_preview
            else ""
        )
        agi_next_implementation_preflight_next_command = str(
            metadata.get("agi_next_implementation_preflight_next_command") or ""
        )
        agi_next_implementation_preflight_row_next_command = (
            agi_next_implementation_preflight_next_command
            or agi_next_build_command
            or agi_focus_canonical_selector_command
            or "agi gates"
        )
        real_execution_gap_count = int(
            metadata.get("real_execution_gap_count")
            or metadata.get("agi_real_execution_gap_count")
            or 0
        )
        selected_real_execution_gap = str(
            metadata.get("selected_real_execution_gap")
            or metadata.get("real_execution_gap")
            or metadata.get("agi_next_gate")
            or ""
        )
        selected_real_execution_gap_gate = str(
            metadata.get("selected_real_execution_gap_gate")
            or metadata.get("agi_next_gate")
            or selected_real_execution_gap
            or ""
        )
        selected_real_execution_gap_detail = str(
            metadata.get("selected_real_execution_gap_detail")
            or (metadata.get("agi_real_execution_gaps_by_gate") or {}).get(selected_real_execution_gap_gate)
            or metadata.get("real_execution_gap")
            or ""
        )
        operator_handoff = {
            "source": "operator_handoff_packet",
            "operator_handoff_ready": True,
            "handoff_ready": True,
            "source_tool": proof.tool_name,
            "objective": objective,
            "next_operator_command": command,
            "next_proof_command": command,
            "completion_next_proof_command": command,
            "command_source": command_source,
            "command_reason": command_reason,
            "first_blocker": first_blocker,
            "queue_position": queue_position,
            "queue_total": queue_total,
            "refresh_command": refresh_command,
            "claim_gate_verdict": metadata.get("claim_gate_verdict"),
            "completion_claim_ready": _metadata_bool(metadata.get("allowed_to_claim")),
            "allowed_to_claim": _metadata_bool(metadata.get("allowed_to_claim")),
            "latest_execution_case_review_state": latest_case_review_state,
            "latest_execution_case_review_verdict": latest_case_review_verdict,
            "latest_execution_case_review_next_safe_command": latest_case_review_next_safe_command,
            "latest_execution_case_review_approval_required": _metadata_bool(metadata.get("latest_execution_case_review_approval_required")),
            "latest_execution_case_review_checklist_items": metadata.get("latest_execution_case_review_checklist_items", 0),
            "latest_execution_case_review_blocker_count": metadata.get("latest_execution_case_review_blocker_count", 0),
            "latest_execution_case_review_handoff": latest_case_review_handoff,
            "latest_execution_case_review_handoff_present": latest_case_review_handoff_present,
            "latest_execution_case_evidence_preview_gate_count": latest_case_evidence_preview_gate_count,
            "latest_execution_case_evidence_preview_ready_count": latest_case_evidence_preview_ready_count,
            "latest_execution_case_evidence_preview_blocked_count": latest_case_evidence_preview_blocked_count,
            "latest_execution_case_evidence_preview_verdicts": latest_case_evidence_preview_verdicts,
            "latest_execution_case_evidence_preview_latest_verdict": latest_case_evidence_preview_latest_verdict,
            "latest_execution_case_evidence_preview_latest_event_id": latest_case_evidence_preview_latest_event_id,
            "latest_execution_case_evidence_preview_latest_handoff": latest_case_evidence_preview_latest_handoff,
            "latest_execution_case_evidence_preview_latest_handoff_present": latest_case_evidence_preview_latest_handoff_present,
            "latest_execution_case_evidence_preview_next_command": latest_case_evidence_preview_next_command,
            "latest_execution_case_evidence_preview_receipt_id": latest_case_evidence_preview_receipt_id,
            "latest_execution_case_evidence_preview_receipt_kind": latest_case_evidence_preview_receipt_kind,
            "latest_execution_case_evidence_preview_receipt_target_status": latest_case_evidence_preview_receipt_target_status,
            "latest_execution_case_closure_verdict": latest_case_verdict,
            "latest_execution_case_closure_ready": latest_case_closure_ready,
            "latest_execution_case_closure_blocks_completion_claim": latest_case_closure_blocks_completion_claim,
            "latest_execution_case_closure_proof_queue": latest_case_closure_queue,
            "latest_execution_case_closure_proof_queue_count": len(latest_case_closure_queue),
            "latest_execution_case_closure_proof_queue_preview": latest_case_closure_queue_preview,
            "latest_execution_case_closure_proof_queue_preview_count": len(latest_case_closure_queue_preview),
            "latest_execution_case_closure_proof_queue_remaining_count": latest_case_closure_queue_remaining_count,
            "latest_execution_case_closure_first_proof_command": latest_case_closure_first_proof_command,
            "latest_execution_case_next_closure_proof_command": latest_case_next,
            "execution_health_recovery_closure_state": recovery_state,
            "execution_health_recovery_closure_missing": recovery_missing,
            "execution_health_recovery_closure_missing_count": len(recovery_missing),
            "execution_health_recovery_closure_required_commands": recovery_required_commands,
            "execution_health_recovery_closure_next_required_command": recovery_next_required,
            "execution_health_recovery_closure_next_proof_command": recovery_next,
            "execution_health_recovery_closure_proof_queue": recovery_queue,
            "execution_health_recovery_closure_proof_queue_count": len(recovery_queue),
            "execution_health_recovery_closure_blocks_completion_claim": recovery_blocks_completion_claim,
            "execution_health_verification_coverage_state": verification_state,
            "execution_health_verification_recent_tool_runs": metadata.get("execution_health_verification_recent_tool_runs", 0),
            "execution_health_verification_recent_action_runs": metadata.get("execution_health_verification_recent_action_runs", 0),
            "execution_health_verification_recent_verification_runs": metadata.get("execution_health_verification_recent_verification_runs", 0),
            "execution_health_verification_proof_queue": verification_queue,
            "execution_health_verification_proof_queue_count": len(verification_queue),
            "execution_health_verification_next_required_command": verification_next_required,
            "execution_health_verification_next_proof_command": verification_next,
            "execution_health_verification_blocks_completion_claim": verification_blocks_completion_claim,
            "execution_learning_state": learning_state,
            "execution_learning_missing": learning_missing,
            "execution_learning_missing_count": len(learning_missing),
            "execution_learning_closure_command": learning_closure_command,
            "execution_learning_required_commands": learning_required_commands,
            "execution_learning_next_required_command": learning_next_required,
            "execution_learning_next_proof_command": learning_next,
            "execution_learning_proof_queue": learning_queue,
            "execution_learning_proof_queue_count": len(learning_queue),
            "execution_learning_blocks_completion_claim": learning_blocks_completion_claim,
            "execution_learning_actionable_required_commands": learning_actionable_required_commands,
            "execution_learning_actionable_required_command_count": len(learning_actionable_required_commands),
            "execution_learning_actionable_proof_queue": learning_actionable_queue,
            "execution_learning_actionable_proof_queue_count": len(learning_actionable_queue),
            "execution_learning_actionable_next_required_command": learning_actionable_next_required,
            "execution_learning_actionable_next_proof_command": learning_actionable_next,
            "execution_learning_next_evidence_command": metadata.get("execution_learning_next_evidence_command") or "",
            "storage_runtime_fallback_active": storage_runtime_fallback_active,
            "storage_runtime_fallback_reason": metadata.get("storage_runtime_fallback_reason") or "",
            "storage_runtime_fallback_exception_type": metadata.get("storage_runtime_fallback_exception_type") or "",
            "storage_runtime_fallback_db_path_display": metadata.get("storage_runtime_fallback_db_path_display") or "",
            "storage_runtime_fallback_vault_path_display": metadata.get("storage_runtime_fallback_vault_path_display") or "",
            "storage_readiness_blocks_completion_claim": storage_readiness_blocks_completion_claim,
            "storage_recovery_required": _metadata_bool(metadata.get("storage_recovery_required")),
            "storage_recovery_reason": metadata.get("storage_recovery_reason") or "",
            "storage_issues": storage_issues,
            "storage_issue_count": storage_issue_count,
            "storage_issue_preview": storage_issue_preview,
            "storage_issue_preview_count": len(storage_issue_preview),
            "storage_first_issue": storage_first_issue,
            "storage_recovery_mode": storage_recovery_mode,
            "storage_recovery_next_operator_action": storage_recovery_next_operator_action,
            "storage_recovery_restart_required": storage_recovery_restart_required,
            "storage_recovery_check_command": metadata.get("storage_recovery_check_command") or "",
            "storage_recovery_check_tool_command": metadata.get("storage_recovery_check_tool_command") or "",
            "storage_recovery_check_api": metadata.get("storage_recovery_check_api") or STORAGE_RECOVERY_CHECK_API,
            "storage_recovery_command": metadata.get("storage_recovery_command") or "",
            "storage_readiness_blocker": storage_readiness_blocker,
            "storage_readiness_next_commands": storage_readiness_next_commands,
            "storage_readiness_next_command_count": len(storage_readiness_next_commands),
            "storage_readiness_next_required_command": storage_readiness_next_required_command,
            "storage_readiness_next_proof_command": storage_readiness_next_proof_command,
            "storage_readiness_proof_queue": storage_readiness_proof_queue,
            "storage_readiness_proof_queue_count": len(storage_readiness_proof_queue),
            "storage_readiness_proof_queue_preview": storage_readiness_proof_queue_preview,
            "storage_readiness_proof_queue_preview_count": len(storage_readiness_proof_queue_preview),
            "storage_readiness_proof_queue_remaining_count": storage_readiness_proof_queue_remaining_count,
            "storage_readiness_first_proof_command": storage_readiness_first_proof_command,
            "storage_handoff": storage_handoff,
            "agi_next_gate": agi_next_gate,
            "agi_next_target_title": agi_next_target_title,
            "agi_next_build_command": agi_next_build_command,
            "agi_next_evidence_closure_commands": metadata.get("agi_next_evidence_closure_commands"),
            "agi_next_evidence_closure_command_count": metadata.get("agi_next_evidence_closure_command_count"),
            "agi_next_focused_verification_commands": agi_next_focused_verification_commands,
            "agi_next_likely_files": agi_next_likely_files,
            "agi_next_likely_file_count": len(agi_next_likely_files),
            "agi_next_missing_target_files": agi_next_missing_target_files,
            "agi_next_missing_target_file_count": len(agi_next_missing_target_files),
            "agi_next_target_files_checked": metadata.get("agi_next_target_files_checked"),
            "agi_next_target_files_exist": metadata.get("agi_next_target_files_exist"),
            "agi_next_target_integrity_blocks_start": metadata.get("agi_next_target_integrity_blocks_start"),
            "agi_next_acceptance_checks": agi_next_acceptance_checks,
            "agi_next_acceptance_check_count": len(agi_next_acceptance_checks),
            **_agi_acceptance_gap_metadata({"agi_next_acceptance_checks": agi_next_acceptance_checks}),
            "agi_next_focused_verification_command_count": len(agi_next_focused_verification_commands),
            "agi_next_target_file_integrity_status": agi_next_target_file_integrity_status,
            "agi_next_build_packet_ready_for_review": agi_next_build_packet_ready,
            "agi_next_implementation_preflight_ready": agi_next_implementation_preflight_ready,
            "agi_next_implementation_preflight_blockers": agi_next_implementation_preflight_blockers,
            "agi_next_implementation_preflight_blocker_count": len(agi_next_implementation_preflight_blockers),
            "agi_next_implementation_preflight_next_command": agi_next_implementation_preflight_next_command,
            "agi_focus_selection_source": agi_focus_selection_source,
            "agi_focus_selection_reason": agi_focus_selection_reason,
            "agi_focus_canonical_selector_command": agi_focus_canonical_selector_command,
            "agi_focus_deliberate_focus_override": agi_focus_deliberate_focus_override,
            "agi_real_execution_gap_count": metadata.get("agi_real_execution_gap_count"),
            "real_execution_gap_count": real_execution_gap_count,
            "selected_real_execution_gap": selected_real_execution_gap,
            "selected_real_execution_gap_gate": selected_real_execution_gap_gate,
            "selected_real_execution_gap_detail": selected_real_execution_gap_detail,
            "proof_queue": proof_queue,
            "proof_queue_count": len(proof_queue),
            "completion_proof_queue": ordered_proof_queue,
            "completion_proof_queue_count": len(ordered_proof_queue),
            "completion_actionable_proof_queue": proof_queue,
            "completion_actionable_proof_queue_count": len(proof_queue),
            "completion_actionable_next_proof_command": metadata.get("completion_actionable_next_proof_command"),
            "completion_actionable_queue_includes_learning_prerequisites": metadata.get("completion_actionable_queue_includes_learning_prerequisites"),
            "completion_next_proof_handoff": completion_next_proof_handoff,
            "completion_next_proof_handoff_present": bool(completion_next_proof_handoff),
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "calls_external_service": False,
            "executes_tools": False,
            "reads_personal_data": False,
            "reads_private_data": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "queues_approval": False,
            "requires_approval": False,
            "controls_computer": False,
            "speaks": False,
            "completes_tasks": False,
        }

        lines = [
            "Jarvis operator handoff:",
            "This read-only handoff turns the completion proof queue into one operator-reviewable command. It does not load, run, approve, queue, or complete anything.",
            "",
            f"Objective: {objective}",
            f"Next operator command: `{command}`" if command else "Next operator command: none",
            f"Command source: {command_source}",
            f"Command reason: {command_reason}",
            f"First blocker: {first_blocker}",
            f"Queue position: {queue_position} of {queue_total}",
            f"Claim gate verdict: {metadata.get('claim_gate_verdict') or 'unknown'}",
            f"Completion claim ready: {'yes' if metadata.get('allowed_to_claim') else 'no'}",
            f"Refresh command: `{refresh_command}`",
            "",
            "Closure debt:",
            f"- latest case review state: {latest_case_review_state or 'unknown'}",
            f"- latest case review verdict: {latest_case_review_verdict or 'unknown'}",
            f"- latest case review next safe command: `{latest_case_review_next_safe_command}`" if latest_case_review_next_safe_command else "- latest case review next safe command: none",
            f"- latest case evidence preflight: {latest_case_evidence_preview_ready_count}/{latest_case_evidence_preview_gate_count} ready, {latest_case_evidence_preview_blocked_count} blocked",
            f"- latest case evidence preflight verdict: {latest_case_evidence_preview_latest_verdict or 'none'}",
            f"- latest execution case: {latest_case_verdict}",
            f"- latest case next required: `{latest_case_next}`" if latest_case_next else "- latest case next required: none",
            f"- latest case proof alias: `{latest_case_next}`" if latest_case_next else "- latest case proof alias: none",
            f"- recovery closure: {recovery_state}",
            f"- recovery missing: {', '.join(recovery_missing) if recovery_missing else 'none'}",
            f"- recovery next required: `{recovery_next_required}`" if recovery_next_required else "- recovery next required: none",
            f"- recovery proof alias: `{recovery_next}`" if recovery_next else "- recovery proof alias: none",
            f"- verification coverage: {verification_state}",
            f"- verification blocks completion claim: {'yes' if verification_blocks_completion_claim else 'no'}",
            f"- verification next required: `{verification_next_required}`" if verification_next_required else "- verification next required: none",
            f"- verification proof alias: `{verification_next}`" if verification_next else "- verification proof alias: none",
            f"- execution learning: {learning_state}",
            f"- learning blocks completion claim: {'yes' if learning_blocks_completion_claim else 'no'}",
            f"- learning missing: {', '.join(learning_missing) if learning_missing else 'none'}",
            f"- actionable learning next required: `{learning_actionable_next_required}`" if learning_actionable_next_required else "- actionable learning next required: none",
            f"- actionable learning proof alias: `{learning_actionable_next}`" if learning_actionable_next else "- actionable learning proof alias: none",
            f"- learning closure command: `{learning_closure_command}`" if learning_closure_command else "- learning closure command: none",
            f"- learning next required: `{learning_next_required}`" if learning_next_required else "- learning next required: none",
            f"- learning proof alias: `{learning_next}`" if learning_next else "- learning proof alias: none",
            "",
            "Storage readiness:",
            f"- runtime fallback active: {'yes' if storage_runtime_fallback_active else 'no'}",
            f"- blocks completion claim: {'yes' if storage_readiness_blocks_completion_claim else 'no'}",
            f"- blocker: {storage_readiness_blocker or 'none'}",
            f"- recovery mode: {storage_recovery_mode or 'none'}",
            f"- restart required: {'yes' if storage_recovery_restart_required else 'no'}",
            f"- next operator action: {storage_recovery_next_operator_action or 'none'}",
            f"- next required: `{storage_readiness_next_required_command}`" if storage_readiness_next_required_command else "- next required: none",
            f"- storage proof alias: `{storage_readiness_next_proof_command}`" if storage_readiness_next_proof_command else "- storage proof alias: none",
            f"- proof queue: {', '.join(f'`{command}`' for command in storage_readiness_proof_queue) if storage_readiness_proof_queue else 'none'}",
            f"- recovery commands: {', '.join(f'`{command}`' for command in storage_readiness_next_commands) if storage_readiness_next_commands else 'none'}",
            "",
            "AGI harness focus:",
            f"- next gate: {agi_next_gate or 'none'}",
            f"- selected target: {agi_next_target_title or 'none'}",
            f"- next build command: `{agi_next_build_command}`" if agi_next_build_command else "- next build command: none",
            f"- selection source: {agi_focus_selection_source or 'unknown'}",
            f"- selection reason: {agi_focus_selection_reason or 'none'}",
            f"- canonical selector command: `{agi_focus_canonical_selector_command}`" if agi_focus_canonical_selector_command else "- canonical selector command: none",
            f"- deliberate focus override: {'yes' if agi_focus_deliberate_focus_override else 'no'}",
            f"- target file integrity: {agi_next_target_file_integrity_status}",
            f"- likely files: {', '.join(agi_next_likely_files) if agi_next_likely_files else 'none'}",
            f"- missing target files: {', '.join(agi_next_missing_target_files) if agi_next_missing_target_files else 'none'}",
            f"- acceptance checks: {len(agi_next_acceptance_checks)}",
            f"- focused verification commands: {len(agi_next_focused_verification_commands)}",
            f"- ready for scoped implementation review: {'yes' if agi_next_build_packet_ready else 'no'}",
            f"- implementation preflight ready: {'yes' if agi_next_implementation_preflight_ready else 'no'}",
            f"- implementation preflight blockers: {', '.join(agi_next_implementation_preflight_blockers) if agi_next_implementation_preflight_blockers else 'none'}",
            f"- implementation preflight next command: `{agi_next_implementation_preflight_next_command}`" if agi_next_implementation_preflight_next_command else "- implementation preflight next command: none",
            "",
            "Operator contract:",
            "- Review the command before sending it.",
            "- Loading or reading this handoff grants no approval.",
            f"- After completing the command and its proof, rerun `{refresh_command}` or `operator handoff`.",
            "",
            "Completion proof queue:",
        ]
        if proof_queue:
            lines.extend(f"- `{item}`" for item in proof_queue[:10])
            if len(proof_queue) > 10:
                lines.append(f"- ... {len(proof_queue) - 10} more command(s)")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Boundary:",
                "- This handoff does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, write notes, write the database, control the computer, call external services, speak, complete tasks, mark goals done, or queue approvals.",
            ]
        )
        return ToolResult(
            "operator_handoff_packet",
            proof.ok,
            "\n".join(lines),
            _safe_metadata(
                objective=objective,
                next_operator_command=command,
                next_proof_command=command,
                completion_next_proof_command=command,
                command_source=command_source,
                command_reason=command_reason,
                first_blocker=first_blocker,
                queue_position=queue_position,
                queue_total=queue_total,
                refresh_command=refresh_command,
                claim_gate_verdict=metadata.get("claim_gate_verdict"),
                completion_claim_ready=_metadata_bool(metadata.get("allowed_to_claim")),
                allowed_to_claim=_metadata_bool(metadata.get("allowed_to_claim")),
                latest_execution_case_review_state=latest_case_review_state,
                latest_execution_case_review_verdict=latest_case_review_verdict,
                latest_execution_case_review_next_safe_command=latest_case_review_next_safe_command,
                latest_execution_case_review_approval_required=_metadata_bool(metadata.get("latest_execution_case_review_approval_required")),
                latest_execution_case_review_checklist_items=metadata.get("latest_execution_case_review_checklist_items", 0),
                latest_execution_case_review_blocker_count=metadata.get("latest_execution_case_review_blocker_count", 0),
                latest_execution_case_review_handoff=latest_case_review_handoff,
                latest_execution_case_review_handoff_present=latest_case_review_handoff_present,
                latest_execution_case_evidence_preview_gate_count=latest_case_evidence_preview_gate_count,
                latest_execution_case_evidence_preview_ready_count=latest_case_evidence_preview_ready_count,
                latest_execution_case_evidence_preview_blocked_count=latest_case_evidence_preview_blocked_count,
                latest_execution_case_evidence_preview_verdicts=latest_case_evidence_preview_verdicts,
                latest_execution_case_evidence_preview_latest_verdict=latest_case_evidence_preview_latest_verdict,
                latest_execution_case_evidence_preview_latest_event_id=latest_case_evidence_preview_latest_event_id,
                latest_execution_case_evidence_preview_latest_handoff=latest_case_evidence_preview_latest_handoff,
                latest_execution_case_evidence_preview_latest_handoff_present=latest_case_evidence_preview_latest_handoff_present,
                latest_execution_case_evidence_preview_next_command=latest_case_evidence_preview_next_command,
                latest_execution_case_evidence_preview_receipt_id=latest_case_evidence_preview_receipt_id,
                latest_execution_case_evidence_preview_receipt_kind=latest_case_evidence_preview_receipt_kind,
                latest_execution_case_evidence_preview_receipt_target_status=latest_case_evidence_preview_receipt_target_status,
                latest_execution_case_closure_verdict=latest_case_verdict,
                latest_execution_case_closure_ready=latest_case_closure_ready,
                latest_execution_case_closure_blocks_completion_claim=latest_case_closure_blocks_completion_claim,
                latest_execution_case_closure_proof_queue=latest_case_closure_queue,
                latest_execution_case_closure_proof_queue_count=len(latest_case_closure_queue),
                latest_execution_case_closure_proof_queue_preview=latest_case_closure_queue_preview,
                latest_execution_case_closure_proof_queue_preview_count=len(latest_case_closure_queue_preview),
                latest_execution_case_closure_proof_queue_remaining_count=latest_case_closure_queue_remaining_count,
                latest_execution_case_closure_first_proof_command=latest_case_closure_first_proof_command,
                latest_execution_case_next_closure_proof_command=latest_case_next,
                execution_health_recovery_closure_state=recovery_state,
                execution_health_recovery_closure_missing=recovery_missing,
                execution_health_recovery_closure_missing_count=len(recovery_missing),
                execution_health_recovery_closure_required_commands=recovery_required_commands,
                execution_health_recovery_closure_next_required_command=recovery_next_required,
                execution_health_recovery_closure_next_proof_command=recovery_next,
                execution_health_recovery_closure_proof_queue=recovery_queue,
                execution_health_recovery_closure_proof_queue_count=len(recovery_queue),
                execution_health_recovery_closure_blocks_completion_claim=recovery_blocks_completion_claim,
                execution_health_verification_coverage_state=verification_state,
                execution_health_verification_recent_tool_runs=metadata.get("execution_health_verification_recent_tool_runs", 0),
                execution_health_verification_recent_action_runs=metadata.get("execution_health_verification_recent_action_runs", 0),
                execution_health_verification_recent_verification_runs=metadata.get("execution_health_verification_recent_verification_runs", 0),
                execution_health_verification_gap_count=metadata.get("execution_health_verification_gap_count", 0),
                execution_health_verification_first_gap=metadata.get("execution_health_verification_first_gap", ""),
                execution_health_verification_proof_queue=verification_queue,
                execution_health_verification_proof_queue_count=len(verification_queue),
                execution_health_verification_proof_queue_preview=metadata.get("execution_health_verification_proof_queue_preview", verification_queue[:3]),
                execution_health_verification_proof_queue_preview_count=metadata.get("execution_health_verification_proof_queue_preview_count", len(verification_queue[:3])),
                execution_health_verification_proof_queue_remaining_count=metadata.get("execution_health_verification_proof_queue_remaining_count", max(len(verification_queue) - len(verification_queue[:3]), 0)),
                execution_health_verification_next_proof_command=verification_next,
                execution_health_verification_blocks_completion_claim=verification_blocks_completion_claim,
                execution_learning_state=learning_state,
                execution_learning_missing=learning_missing,
                execution_learning_missing_count=len(learning_missing),
                execution_learning_closure_command=learning_closure_command,
                execution_learning_required_commands=learning_required_commands,
                execution_learning_next_required_command=learning_next_required,
                execution_learning_next_proof_command=learning_next,
                execution_learning_proof_queue=learning_queue,
                execution_learning_proof_queue_count=len(learning_queue),
                execution_learning_blocks_completion_claim=learning_blocks_completion_claim,
                execution_learning_actionable_required_commands=learning_actionable_required_commands,
                execution_learning_actionable_required_command_count=len(learning_actionable_required_commands),
                execution_learning_actionable_proof_queue=learning_actionable_queue,
                execution_learning_actionable_proof_queue_count=len(learning_actionable_queue),
                execution_learning_actionable_next_required_command=learning_actionable_next_required,
                execution_learning_actionable_next_proof_command=learning_actionable_next,
                execution_learning_next_evidence_command=metadata.get("execution_learning_next_evidence_command") or "",
                storage_runtime_fallback_active=storage_runtime_fallback_active,
                storage_runtime_fallback_reason=metadata.get("storage_runtime_fallback_reason") or "",
                storage_runtime_fallback_exception_type=metadata.get("storage_runtime_fallback_exception_type") or "",
                storage_runtime_fallback_db_path_display=metadata.get("storage_runtime_fallback_db_path_display") or "",
                storage_runtime_fallback_vault_path_display=metadata.get("storage_runtime_fallback_vault_path_display") or "",
                storage_readiness_blocks_completion_claim=storage_readiness_blocks_completion_claim,
                storage_recovery_required=_metadata_bool(metadata.get("storage_recovery_required")),
                storage_recovery_reason=metadata.get("storage_recovery_reason") or "",
                storage_issues=storage_issues,
                storage_issue_count=storage_issue_count,
                storage_recovery_mode=storage_recovery_mode,
                storage_recovery_next_operator_action=storage_recovery_next_operator_action,
                storage_recovery_restart_required=storage_recovery_restart_required,
                storage_recovery_check_command=metadata.get("storage_recovery_check_command") or "",
                storage_recovery_check_tool_command=metadata.get("storage_recovery_check_tool_command") or "",
                storage_recovery_check_api=metadata.get("storage_recovery_check_api") or STORAGE_RECOVERY_CHECK_API,
                storage_recovery_command=metadata.get("storage_recovery_command") or "",
                storage_readiness_blocker=storage_readiness_blocker,
                storage_readiness_next_commands=storage_readiness_next_commands,
                storage_readiness_next_command_count=len(storage_readiness_next_commands),
                storage_readiness_next_required_command=storage_readiness_next_required_command,
                storage_readiness_next_proof_command=storage_readiness_next_proof_command,
                storage_readiness_proof_queue=storage_readiness_proof_queue,
                storage_readiness_proof_queue_count=len(storage_readiness_proof_queue),
                storage_readiness_proof_queue_preview=storage_readiness_proof_queue_preview,
                storage_readiness_proof_queue_preview_count=len(storage_readiness_proof_queue_preview),
                storage_readiness_proof_queue_remaining_count=storage_readiness_proof_queue_remaining_count,
                storage_readiness_first_proof_command=storage_readiness_first_proof_command,
                storage_handoff=storage_handoff,
                agi_next_gate=agi_next_gate,
                agi_next_target_title=agi_next_target_title,
                agi_next_build_command=agi_next_build_command,
                agi_next_evidence_closure_commands=metadata.get("agi_next_evidence_closure_commands"),
                agi_next_evidence_closure_command_count=metadata.get("agi_next_evidence_closure_command_count"),
                agi_next_focused_verification_commands=agi_next_focused_verification_commands,
                agi_next_focused_verification_command_count=len(agi_next_focused_verification_commands),
                agi_next_likely_files=agi_next_likely_files,
                agi_next_likely_file_count=len(agi_next_likely_files),
                agi_next_target_file_integrity_status=agi_next_target_file_integrity_status,
                agi_next_target_files_checked=metadata.get("agi_next_target_files_checked"),
                agi_next_target_files_exist=metadata.get("agi_next_target_files_exist"),
                agi_next_missing_target_files=agi_next_missing_target_files,
                agi_next_missing_target_file_count=len(agi_next_missing_target_files),
                agi_next_target_integrity_blocks_start=metadata.get("agi_next_target_integrity_blocks_start"),
                agi_next_acceptance_checks=agi_next_acceptance_checks,
                agi_next_acceptance_check_count=len(agi_next_acceptance_checks),
                **_agi_acceptance_gap_metadata({"agi_next_acceptance_checks": agi_next_acceptance_checks}),
                agi_next_build_packet_ready_for_review=agi_next_build_packet_ready,
                agi_next_implementation_preflight_ready=agi_next_implementation_preflight_ready,
                agi_next_implementation_preflight_blockers=agi_next_implementation_preflight_blockers,
                agi_next_implementation_preflight_blocker_count=len(agi_next_implementation_preflight_blockers),
                agi_next_implementation_preflight_next_command=agi_next_implementation_preflight_next_command,
                agi_focus_selection_source=agi_focus_selection_source,
                agi_focus_selection_reason=agi_focus_selection_reason,
                agi_focus_canonical_selector_command=agi_focus_canonical_selector_command,
                agi_focus_deliberate_focus_override=agi_focus_deliberate_focus_override,
                agi_real_execution_gap_count=metadata.get("agi_real_execution_gap_count"),
                real_execution_gap_count=real_execution_gap_count,
                selected_real_execution_gap=selected_real_execution_gap,
                selected_real_execution_gap_gate=selected_real_execution_gap_gate,
                selected_real_execution_gap_detail=selected_real_execution_gap_detail,
                proof_queue=proof_queue,
                proof_queue_count=len(proof_queue),
                completion_proof_queue=ordered_proof_queue,
                completion_proof_queue_count=len(ordered_proof_queue),
                completion_actionable_proof_queue=proof_queue,
                completion_actionable_proof_queue_count=len(proof_queue),
                completion_actionable_next_proof_command=metadata.get("completion_actionable_next_proof_command"),
                completion_actionable_queue_includes_learning_prerequisites=metadata.get("completion_actionable_queue_includes_learning_prerequisites"),
                completion_next_proof_handoff=completion_next_proof_handoff,
                completion_next_proof_handoff_present=bool(completion_next_proof_handoff),
                operator_handoff=operator_handoff,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                operator_handoff_ready=True,
                source_tool=proof.tool_name,
            ),
        )

    def harness_readiness_digest(args: dict[str, Any]) -> ToolResult:
        objective = _short(
            args.get("objective") or args.get("goal") or args.get("request") or (
                "Finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant."
            ),
            limit=500,
        )
        proof = completion_next_proof_packet({"objective": objective})
        metadata = dict(proof.metadata)
        completion_next_proof_handoff = dict(metadata.get("completion_next_proof_handoff") or {})
        proof_queue = list(metadata.get("completion_proof_queue") or [])
        pending_approvals = store.list_pending_approvals(limit=100)
        approval_review_preview: list[dict[str, Any]] = []
        approval_review_preview_unreadable_rows = 0
        for row in pending_approvals[:3]:
            approval_id = _row_positive_int_text(row)
            if not approval_id:
                approval_review_preview_unreadable_rows += 1
                continue
            approval_review_preview.append(
                {
                    "id": int(approval_id),
                    "tool_name": _row_text(row, "tool_name", "unknown"),
                    "status": _row_text(row, "status", "unknown"),
                    "created_at": _row_text(row, "created_at", "unknown"),
                    "readiness_command": f"approval readiness {approval_id}",
                    "packet_command": f"approval packet {approval_id}",
                    "proof_command": f"approval chain proof {approval_id}",
                }
            )
        approval_review_first_id = approval_review_preview[0]["id"] if approval_review_preview else None
        approval_review_first_readiness_command = approval_review_preview[0]["readiness_command"] if approval_review_preview else ""
        approval_review_first_proof_command = approval_review_preview[0]["proof_command"] if approval_review_preview else ""
        approval_review_next_command = approval_review_first_readiness_command or ("approval review" if pending_approvals else "")
        approval_review_row_next_command = approval_review_next_command or "none"
        blockers = list(metadata.get("blocker_details") or [])
        next_command = str(metadata.get("next_proof_command") or metadata.get("completion_next_proof_command") or "")
        command_source = str(metadata.get("command_source") or "unknown")
        command_reason = str(metadata.get("command_reason") or "unknown")
        recovery_state = str(metadata.get("execution_health_recovery_closure_state") or "unknown")
        recovery_missing = list(metadata.get("execution_health_recovery_closure_missing") or [])
        recovery_queue = list(metadata.get("execution_health_recovery_closure_proof_queue") or [])
        recovery_required_commands = list(metadata.get("execution_health_recovery_closure_required_commands") or [])
        recovery_next_required = str(
            metadata.get("execution_health_recovery_closure_next_required_command")
            or (recovery_required_commands[0] if recovery_required_commands else "")
            or metadata.get("execution_health_recovery_closure_next_proof_command")
            or ""
        )
        verification_state = str(metadata.get("execution_health_verification_coverage_state") or "unknown")
        verification_queue = list(metadata.get("execution_health_verification_proof_queue") or [])
        verification_next = str(
            metadata.get("execution_health_verification_next_proof_command")
            or (verification_queue[0] if verification_queue else "")
        )
        verification_next_required = str(
            metadata.get("execution_health_verification_next_required_command")
            or verification_next
        )
        verification_concrete_next = next(
            (
                str(command).strip()
                for command in proof_queue
                if re.match(r"^verification receipt \d+$", str(command).strip())
            ),
            "",
        )
        verification_row_next = verification_concrete_next or verification_next
        verification_blocks_completion_claim = _metadata_bool(metadata.get("execution_health_verification_blocks_completion_claim"))
        verification_recent_tool_runs = int(metadata.get("execution_health_verification_recent_tool_runs") or 0)
        verification_recent_action_runs = int(metadata.get("execution_health_verification_recent_action_runs") or 0)
        verification_recent_verification_runs = int(metadata.get("execution_health_verification_recent_verification_runs") or 0)
        verification_queue_preview = list(metadata.get("execution_health_verification_proof_queue_preview") or verification_queue[:3])
        verification_queue_remaining_count = int(
            metadata.get(
                "execution_health_verification_proof_queue_remaining_count",
                max(len(verification_queue) - len(verification_queue_preview), 0),
            )
        )
        verification_gap_count = int(
            metadata.get(
                "execution_health_verification_gap_count",
                max(verification_recent_tool_runs - verification_recent_verification_runs, 0),
            )
        )
        verification_first_gap = str(metadata.get("execution_health_verification_first_gap") or "")
        if not verification_first_gap:
            if verification_state == "no_recent_tool_runs":
                verification_first_gap = "no recent tool runs have been recorded"
            elif verification_blocks_completion_claim:
                verification_first_gap = (
                    "recent tool activity has no verification receipt"
                    if verification_gap_count
                    else "verification receipt is required before completion"
                )
        learning_state = str(metadata.get("execution_learning_state") or "unknown")
        learning_missing = list(metadata.get("execution_learning_missing") or [])
        learning_queue = list(metadata.get("execution_learning_proof_queue") or [])
        learning_required_commands = list(metadata.get("execution_learning_required_commands") or [])
        learning_next_required = str(
            metadata.get("execution_learning_next_required_command")
            or (learning_required_commands[0] if learning_required_commands else "")
            or metadata.get("execution_learning_next_proof_command")
            or ""
        )
        learning_actionable_queue = list(metadata.get("execution_learning_actionable_proof_queue") or [])
        learning_actionable_next = str(metadata.get("execution_learning_actionable_next_proof_command") or "").strip()
        learning_actionable_required_commands = list(metadata.get("execution_learning_actionable_required_commands") or [])
        learning_actionable_next_required = str(
            metadata.get("execution_learning_actionable_next_required_command")
            or (learning_actionable_required_commands[0] if learning_actionable_required_commands else "")
            or learning_actionable_next
        ).strip()
        storage_runtime_fallback_active = _metadata_bool(metadata.get("storage_runtime_fallback_active"))
        storage_readiness_blocks_completion_claim = _metadata_bool(metadata.get("storage_readiness_blocks_completion_claim"))
        storage_readiness_next_commands = list(metadata.get("storage_readiness_next_commands") or [])
        storage_readiness_proof_queue = list(metadata.get("storage_readiness_proof_queue") or storage_readiness_next_commands)
        storage_readiness_proof_queue_preview = list(
            metadata.get("storage_readiness_proof_queue_preview")
            or storage_readiness_proof_queue[:5]
        )
        storage_readiness_proof_queue_remaining_count = int(
            metadata.get(
                "storage_readiness_proof_queue_remaining_count",
                max(len(storage_readiness_proof_queue) - len(storage_readiness_proof_queue_preview), 0),
            )
        )
        storage_recovery_mode = str(metadata.get("storage_recovery_mode") or "none")
        storage_recovery_next_operator_action = str(metadata.get("storage_recovery_next_operator_action") or "")
        storage_recovery_restart_required = _metadata_bool(metadata.get("storage_recovery_restart_required"))
        storage_readiness_next_proof_command = str(
            metadata.get("storage_readiness_next_proof_command")
            or (
                storage_readiness_next_commands[0]
                if storage_readiness_blocks_completion_claim and storage_readiness_next_commands
                else ""
            )
        )
        storage_readiness_next_required_command = str(
            metadata.get("storage_readiness_next_required_command")
            or storage_readiness_next_proof_command
        )
        storage_readiness_first_proof_command = str(
            metadata.get("storage_readiness_first_proof_command")
            or (storage_readiness_proof_queue[0] if storage_readiness_proof_queue else "")
        )
        storage_readiness_blocker = str(metadata.get("storage_readiness_blocker") or "")
        storage_issues = list(metadata.get("storage_issues") or [])
        storage_issue_count = int(metadata.get("storage_issue_count") or len(storage_issues))
        storage_issue_preview = [str(issue) for issue in storage_issues[:3]]
        storage_first_issue = storage_issue_preview[0] if storage_issue_preview else ""
        agi_next_gate = str(metadata.get("agi_next_gate") or "")
        agi_next_target_title = str(metadata.get("agi_next_target_title") or "")
        agi_next_build_command = str(metadata.get("agi_next_build_command") or "")
        agi_next_likely_files = list(metadata.get("agi_next_likely_files") or [])
        agi_next_acceptance_checks = list(metadata.get("agi_next_acceptance_checks") or [])
        agi_next_focused_verification_commands = list(metadata.get("agi_next_focused_verification_commands") or [])
        agi_next_target_file_integrity_status = str(metadata.get("agi_next_target_file_integrity_status") or "unknown")
        agi_next_missing_target_files = list(metadata.get("agi_next_missing_target_files") or [])
        agi_next_build_packet_ready = _metadata_bool(metadata.get("agi_next_build_packet_ready_for_review"))
        agi_next_implementation_preflight_ready = _metadata_bool(metadata.get("agi_next_implementation_preflight_ready"))
        agi_next_implementation_preflight_blockers = list(metadata.get("agi_next_implementation_preflight_blockers") or [])
        agi_next_implementation_preflight_blocker_preview = agi_next_implementation_preflight_blockers[:3]
        agi_next_implementation_preflight_first_blocker = (
            str(agi_next_implementation_preflight_blocker_preview[0])
            if agi_next_implementation_preflight_blocker_preview
            else ""
        )
        agi_next_implementation_preflight_next_command = str(
            metadata.get("agi_next_implementation_preflight_next_command") or ""
        )
        agi_focus_selection_source = str(metadata.get("agi_focus_selection_source") or "")
        agi_focus_selection_reason = str(metadata.get("agi_focus_selection_reason") or "")
        agi_focus_canonical_selector_command = str(metadata.get("agi_focus_canonical_selector_command") or "")
        agi_focus_deliberate_focus_override = _metadata_bool(metadata.get("agi_focus_deliberate_focus_override"))
        agi_next_implementation_preflight_row_next_command = (
            agi_next_implementation_preflight_next_command
            or agi_next_build_command
            or agi_focus_canonical_selector_command
            or "agi gates"
        )
        real_execution_gap_count = int(
            metadata.get("real_execution_gap_count")
            or metadata.get("agi_real_execution_gap_count")
            or 0
        )
        selected_real_execution_gap = str(
            metadata.get("selected_real_execution_gap")
            or metadata.get("real_execution_gap")
            or metadata.get("agi_next_gate")
            or ""
        )
        selected_real_execution_gap_gate = str(
            metadata.get("selected_real_execution_gap_gate")
            or metadata.get("agi_next_gate")
            or selected_real_execution_gap
            or ""
        )
        selected_real_execution_gap_detail = str(
            metadata.get("selected_real_execution_gap_detail")
            or (metadata.get("agi_real_execution_gaps_by_gate") or {}).get(selected_real_execution_gap_gate)
            or metadata.get("real_execution_gap")
            or ""
        )
        latest_case_found = _metadata_bool(metadata.get("latest_execution_case_found"))
        latest_case_verdict = str(metadata.get("latest_execution_case_closure_verdict") or "none")
        latest_case_review_state = str(metadata.get("latest_execution_case_review_state") or "")
        latest_case_review_verdict = str(metadata.get("latest_execution_case_review_verdict") or "")
        latest_case_review_next_safe_command = str(metadata.get("latest_execution_case_review_next_safe_command") or "")
        latest_case_review_approval_required = _metadata_bool(metadata.get("latest_execution_case_review_approval_required"))
        latest_case_review_checklist_items = int(metadata.get("latest_execution_case_review_checklist_items") or 0)
        latest_case_review_blocker_count = int(metadata.get("latest_execution_case_review_blocker_count") or 0)
        latest_case_review_handoff = dict(metadata.get("latest_execution_case_review_handoff") or {})
        latest_case_review_handoff_present = _metadata_bool(metadata.get("latest_execution_case_review_handoff_present"))
        latest_case_review_next_command = str(
            latest_case_review_handoff.get("next_command")
            or latest_case_review_handoff.get("next_safe_command")
            or latest_case_review_next_safe_command
            or ""
        )
        latest_case_review_initial_governor_command = str(
            latest_case_review_handoff.get("initial_governor_command") or ""
        )
        latest_case_review_mission_command_queue = [
            str(command).strip()
            for command in list(latest_case_review_handoff.get("mission_command_queue") or [])
            if str(command).strip()
        ]
        latest_case_review_mission_command_preview = latest_case_review_mission_command_queue[:5]
        latest_case_review_mission_command_remaining_count = max(
            len(latest_case_review_mission_command_queue) - len(latest_case_review_mission_command_preview),
            0,
        )
        latest_case_review_forecast_queue_before = latest_case_review_handoff.get("forecast_queue_before")
        latest_case_review_forecast_queue_after_if_sent = latest_case_review_handoff.get("forecast_queue_after_if_sent")
        latest_case_review_forecast_queue_delta_if_sent = latest_case_review_handoff.get("forecast_queue_delta_if_sent")
        latest_case_closure_ready = _metadata_bool(metadata.get("latest_execution_case_closure_ready"))
        latest_case_closure_blocks_completion_claim = _metadata_bool(metadata.get("latest_execution_case_closure_blocks_completion_claim"))
        latest_case_closure_queue = list(metadata.get("latest_execution_case_closure_proof_queue") or [])
        latest_case_next_closure_proof_command = str(
            metadata.get("latest_execution_case_next_closure_proof_command")
            or (latest_case_closure_queue[0] if latest_case_closure_queue else "")
        )
        latest_case_closure_queue_preview = latest_case_closure_queue[:5]
        latest_case_closure_queue_remaining_count = max(
            len(latest_case_closure_queue) - len(latest_case_closure_queue_preview),
            0,
        )
        latest_case_closure_first_proof_command = (
            str(latest_case_closure_queue_preview[0])
            if latest_case_closure_queue_preview
            else ""
        )
        latest_case_concrete_next_closure_proof_command = ""
        if latest_case_next_closure_proof_command:
            for index, command in enumerate(proof_queue):
                if str(command).strip() != latest_case_next_closure_proof_command:
                    continue
                if index + 1 < len(proof_queue):
                    candidate = str(proof_queue[index + 1]).strip()
                    if candidate and candidate != latest_case_next_closure_proof_command:
                        latest_case_concrete_next_closure_proof_command = candidate
                break
        latest_case_row_next_closure_command = (
            latest_case_concrete_next_closure_proof_command
            or latest_case_next_closure_proof_command
            or "save execution case: <next real order>"
        )
        latest_case_next_required_command = str(
            metadata.get("latest_execution_case_next_required_command")
            or metadata.get("latest_execution_case_next_proof_command")
            or latest_case_next_closure_proof_command
            or ""
        )
        latest_case_learning_blocks_completion_claim = _metadata_bool(
            metadata.get("latest_execution_case_learning_blocks_completion_claim")
        )
        latest_case_learning_actionable_next = str(
            metadata.get("latest_execution_case_learning_actionable_next_proof_command")
            or metadata.get("latest_execution_case_learning_actionable_next_required_command")
            or ""
        )
        latest_case_learning_next = str(
            latest_case_learning_actionable_next
            or metadata.get("latest_execution_case_learning_next_proof_command")
            or metadata.get("latest_execution_case_learning_next_required_command")
            or latest_case_next_closure_proof_command
            or ""
        )
        latest_case_closure_learning_queue = [
            str(command).strip()
            for command in latest_case_closure_queue
            if "learning" in str(command).lower()
        ]
        latest_case_closure_learning_blocks_completion_claim = bool(
            latest_case_closure_blocks_completion_claim
            and (latest_case_closure_learning_queue or "learning" in latest_case_next_closure_proof_command.lower())
        )
        latest_case_evidence_preview_gate_count = int(metadata.get("latest_execution_case_evidence_preview_gate_count") or 0)
        latest_case_evidence_preview_ready_count = int(metadata.get("latest_execution_case_evidence_preview_ready_count") or 0)
        latest_case_evidence_preview_blocked_count = int(metadata.get("latest_execution_case_evidence_preview_blocked_count") or 0)
        latest_case_evidence_preview_verdicts = list(metadata.get("latest_execution_case_evidence_preview_verdicts") or [])
        latest_case_evidence_preview_latest_verdict = str(
            metadata.get("latest_execution_case_evidence_preview_latest_verdict") or ""
        )
        latest_case_evidence_preview_latest_event_id = metadata.get("latest_execution_case_evidence_preview_latest_event_id")
        latest_case_evidence_preview_latest_handoff = dict(
            metadata.get("latest_execution_case_evidence_preview_latest_handoff") or {}
        )
        latest_case_evidence_preview_latest_handoff_present = _metadata_bool(
            metadata.get("latest_execution_case_evidence_preview_latest_handoff_present")
        )
        latest_case_evidence_preview_next_command = str(
            latest_case_evidence_preview_latest_handoff.get("next_command")
            or latest_case_evidence_preview_latest_handoff.get("append_command")
            or ""
        )
        latest_case_evidence_preview_receipt_id = str(
            latest_case_evidence_preview_latest_handoff.get("receipt_id")
            or latest_case_evidence_preview_latest_handoff.get("inferred_receipt_id")
            or ""
        )
        latest_case_evidence_preview_receipt_kind = str(
            latest_case_evidence_preview_latest_handoff.get("receipt_kind")
            or latest_case_evidence_preview_latest_handoff.get("inferred_receipt_kind")
            or ""
        )
        latest_case_evidence_preview_receipt_target_status = str(
            latest_case_evidence_preview_latest_handoff.get("receipt_target_status") or ""
        )
        completion_ready = _metadata_bool(metadata.get("completion_claim_ready")) or _metadata_bool(metadata.get("allowed_to_claim"))

        if completion_ready:
            readiness_state = "READY_FOR_HUMAN_COMPLETION_REVIEW"
            top_risk = "completion claim needs human review before marking the goal done"
        elif storage_readiness_blocks_completion_claim:
            readiness_state = "PROOF_DEBT_BLOCKED"
            top_risk = "durable primary storage must be restored before autonomy or completion claims"
        elif verification_blocks_completion_claim:
            readiness_state = "PROOF_DEBT_BLOCKED"
            top_risk = "recent tool activity needs verification coverage before autonomy or completion claims"
        elif command_source in {"approval_queue", "execution_recovery_closure", "execution_learning", "execution_case_closure"}:
            readiness_state = "PROOF_DEBT_BLOCKED"
            top_risk = "proof debt must be closed before more autonomy or completion claims"
        elif agi_next_build_command:
            readiness_state = "NEXT_BUILD_SLICE_READY"
            top_risk = "next AGI harness gate needs implementation and focused verification"
        else:
            readiness_state = "NEEDS_FRESH_COMPLETION_GATE"
            top_risk = "readiness state is stale or lacks a concrete proof command"

        learning_loop_blocked = (
            _metadata_bool(metadata.get("execution_learning_blocks_completion_claim"))
            or latest_case_learning_blocks_completion_claim
            or latest_case_closure_learning_blocks_completion_claim
        )
        learning_loop_next = (
            learning_actionable_next
            or learning_next_required
            or (latest_case_learning_next if learning_loop_blocked else "")
            or ("execution learning closure" if learning_loop_blocked else "none")
        )
        readiness_rows = [
            ("completion claim", "ready" if completion_ready else "blocked", "completion claim gate"),
            ("next proof", "ready" if next_command else "missing", next_command or "completion next proof"),
            ("execution case", "saved" if latest_case_found else "missing", latest_case_row_next_closure_command),
            ("recovery closure", "ready" if not metadata.get("execution_health_recovery_closure_blocks_completion_claim") else "blocked", recovery_next_required or "execution health report"),
            (
                "verification coverage",
                "ready" if not verification_blocks_completion_claim else "blocked",
                verification_row_next if verification_blocks_completion_claim else "none",
            ),
            (
                "learning loop",
                "blocked" if learning_loop_blocked else "ready",
                learning_loop_next,
            ),
            ("storage readiness", "blocked" if storage_readiness_blocks_completion_claim else "ready", storage_readiness_next_commands[0] if storage_readiness_next_commands else "storage status"),
            (
                "AGI preflight",
                "ready" if agi_next_implementation_preflight_ready else "blocked",
                agi_next_implementation_preflight_row_next_command,
            ),
            ("AGI gate", "tracked" if agi_next_gate else "unknown", agi_next_build_command or "agi gates"),
        ]

        command_ladder: list[str] = []
        if storage_readiness_blocks_completion_claim and storage_readiness_next_commands:
            command_ladder.extend(str(command).strip() for command in storage_readiness_next_commands if str(command).strip())
        elif next_command:
            command_ladder.append(next_command.strip())
        prioritized_commands = [next_command]
        prioritized_commands.extend(str(command) for command in proof_queue)
        prioritized_commands.extend(
            [
                recovery_next_required,
                str(metadata.get("execution_health_recovery_closure_next_proof_command") or ""),
                verification_next,
                learning_next_required,
                learning_actionable_next,
                str(metadata.get("execution_learning_next_proof_command") or ""),
                agi_next_build_command,
                "completion next proof",
                "completion claim gate",
            ]
        )
        for command in prioritized_commands:
            command = command.strip()
            if command and command not in command_ladder:
                command_ladder.append(command)
        command_ladder_first_command = command_ladder[0] if command_ladder else ""
        command_ladder_storage_prefix = (
            list(storage_readiness_next_commands)
            if storage_readiness_blocks_completion_claim and storage_readiness_next_commands
            else []
        )
        command_ladder_preview = command_ladder[:5]
        command_ladder_remaining_count = max(len(command_ladder) - len(command_ladder_preview), 0)
        completion_proof_queue_preview = proof_queue[:5]
        completion_proof_queue_remaining_count = max(len(proof_queue) - len(completion_proof_queue_preview), 0)
        storage_handoff = _storage_handoff_from_metadata(
            {
                **metadata,
                "storage_readiness_next_commands": storage_readiness_next_commands,
                "storage_readiness_next_command_count": len(storage_readiness_next_commands),
                "storage_readiness_next_required_command": storage_readiness_next_required_command,
                "storage_readiness_next_proof_command": storage_readiness_next_proof_command,
                "storage_readiness_proof_queue": storage_readiness_proof_queue,
                "storage_readiness_proof_queue_count": len(storage_readiness_proof_queue),
                "storage_readiness_proof_queue_preview": storage_readiness_proof_queue_preview,
                "storage_readiness_proof_queue_preview_count": len(storage_readiness_proof_queue_preview),
                "storage_readiness_proof_queue_remaining_count": storage_readiness_proof_queue_remaining_count,
                "storage_readiness_first_proof_command": storage_readiness_first_proof_command,
            },
            source="harness_readiness_digest_storage",
        )
        readiness_row_dicts = [
            {"name": name, "state": state, "next": next_step}
            for name, state, next_step in readiness_rows
        ]
        harness_readiness_handoff = {
            "source": "harness_readiness_digest",
            "harness_readiness_handoff_ready": True,
            "handoff_ready": True,
            "source_tool": proof.tool_name,
            "objective": objective,
            "readiness_state": readiness_state,
            "readiness_verdict": readiness_state,
            "completion_claim_ready": completion_ready,
            "allowed_to_claim": completion_ready,
            "top_risk": top_risk,
            "next_command": next_command,
            "next_proof_command": next_command,
            "completion_next_proof_command": next_command,
            "command_source": command_source,
            "command_reason": command_reason,
            "blocker_details": blockers,
            "blocker_count": len(blockers),
            "readiness_rows": readiness_row_dicts,
            "readiness_row_count": len(readiness_row_dicts),
            "command_ladder": command_ladder,
            "command_ladder_count": len(command_ladder),
            "command_ladder_first_command": command_ladder_first_command,
            "command_ladder_preview": command_ladder_preview,
            "command_ladder_preview_count": len(command_ladder_preview),
            "command_ladder_remaining_count": command_ladder_remaining_count,
            "command_ladder_storage_prefix": command_ladder_storage_prefix,
            "command_ladder_storage_prefix_count": len(command_ladder_storage_prefix),
            "completion_proof_queue": proof_queue,
            "completion_proof_queue_count": len(proof_queue),
            "completion_proof_queue_preview": completion_proof_queue_preview,
            "completion_proof_queue_preview_count": len(completion_proof_queue_preview),
            "completion_proof_queue_remaining_count": completion_proof_queue_remaining_count,
            "approval_review_required": bool(pending_approvals),
            "pending_approval_count": len(pending_approvals),
            "pending_approval_preview": approval_review_preview,
            "pending_approval_preview_count": len(approval_review_preview),
            "pending_approval_preview_unreadable_rows": approval_review_preview_unreadable_rows,
            "pending_approval_remaining_count": max(len(pending_approvals) - len(approval_review_preview), 0),
            "pending_approval_first_id": approval_review_first_id,
            "approval_review_first_readiness_command": approval_review_first_readiness_command,
            "approval_review_first_proof_command": approval_review_first_proof_command,
            "approval_review_next_command": approval_review_next_command,
            "approval_review_row_next_command": approval_review_row_next_command,
            "completion_next_proof_handoff": completion_next_proof_handoff,
            "completion_next_proof_handoff_present": bool(completion_next_proof_handoff),
            "latest_execution_case_found": latest_case_found,
            "latest_execution_case_review_state": latest_case_review_state,
            "latest_execution_case_review_verdict": latest_case_review_verdict,
            "latest_execution_case_review_next_safe_command": latest_case_review_next_safe_command,
            "latest_execution_case_review_next_command": latest_case_review_next_command,
            "latest_execution_case_review_initial_governor_command": latest_case_review_initial_governor_command,
            "latest_execution_case_review_mission_command_queue": latest_case_review_mission_command_queue,
            "latest_execution_case_review_mission_command_count": len(latest_case_review_mission_command_queue),
            "latest_execution_case_review_mission_command_preview": latest_case_review_mission_command_preview,
            "latest_execution_case_review_mission_command_preview_count": len(latest_case_review_mission_command_preview),
            "latest_execution_case_review_mission_command_remaining_count": latest_case_review_mission_command_remaining_count,
            "latest_execution_case_review_forecast_queue_before": latest_case_review_forecast_queue_before,
            "latest_execution_case_review_forecast_queue_after_if_sent": latest_case_review_forecast_queue_after_if_sent,
            "latest_execution_case_review_forecast_queue_delta_if_sent": latest_case_review_forecast_queue_delta_if_sent,
            "latest_execution_case_review_approval_required": latest_case_review_approval_required,
            "latest_execution_case_review_checklist_items": latest_case_review_checklist_items,
            "latest_execution_case_review_blocker_count": latest_case_review_blocker_count,
            "latest_execution_case_review_handoff": latest_case_review_handoff,
            "latest_execution_case_review_handoff_present": latest_case_review_handoff_present,
            "latest_execution_case_closure_verdict": latest_case_verdict,
            "latest_execution_case_closure_ready": latest_case_closure_ready,
            "latest_execution_case_closure_blocks_completion_claim": latest_case_closure_blocks_completion_claim,
            "latest_execution_case_closure_proof_queue": latest_case_closure_queue,
            "latest_execution_case_closure_proof_queue_count": len(latest_case_closure_queue),
            "latest_execution_case_closure_proof_queue_preview": latest_case_closure_queue_preview,
            "latest_execution_case_closure_proof_queue_preview_count": len(latest_case_closure_queue_preview),
            "latest_execution_case_closure_proof_queue_remaining_count": latest_case_closure_queue_remaining_count,
            "latest_execution_case_closure_first_proof_command": latest_case_closure_first_proof_command,
            "latest_execution_case_next_closure_proof_command": latest_case_next_closure_proof_command,
            "latest_execution_case_concrete_next_closure_proof_command": latest_case_concrete_next_closure_proof_command,
            "latest_execution_case_row_next_closure_command": latest_case_row_next_closure_command,
            "latest_execution_case_next_required_command": latest_case_next_required_command,
            "latest_execution_case_next_proof_command": metadata.get("latest_execution_case_next_proof_command") or latest_case_next_required_command,
            "latest_execution_case_next_proof_commands": metadata.get("latest_execution_case_next_proof_commands") or [],
            "latest_execution_case_learning_state": metadata.get("latest_execution_case_learning_state") or "",
            "latest_execution_case_learning_missing": metadata.get("latest_execution_case_learning_missing") or [],
            "latest_execution_case_learning_next_required_command": metadata.get("latest_execution_case_learning_next_required_command") or "",
            "latest_execution_case_learning_required_commands": metadata.get("latest_execution_case_learning_required_commands") or [],
            "latest_execution_case_learning_proof_queue": metadata.get("latest_execution_case_learning_proof_queue") or [],
            "latest_execution_case_learning_proof_queue_count": len(metadata.get("latest_execution_case_learning_proof_queue") or []),
            "latest_execution_case_learning_next_proof_command": metadata.get("latest_execution_case_learning_next_proof_command") or "",
            "latest_execution_case_learning_actionable_required_commands": metadata.get("latest_execution_case_learning_actionable_required_commands") or [],
            "latest_execution_case_learning_actionable_proof_queue": metadata.get("latest_execution_case_learning_actionable_proof_queue") or [],
            "latest_execution_case_learning_actionable_proof_queue_count": len(metadata.get("latest_execution_case_learning_actionable_proof_queue") or []),
            "latest_execution_case_learning_actionable_next_required_command": metadata.get("latest_execution_case_learning_actionable_next_required_command") or "",
            "latest_execution_case_learning_actionable_next_proof_command": latest_case_learning_actionable_next,
            "latest_execution_case_learning_next_evidence_command": metadata.get("latest_execution_case_learning_next_evidence_command") or "",
            "latest_execution_case_learning_blocks_completion_claim": latest_case_learning_blocks_completion_claim,
            "latest_execution_case_evidence_preview_gate_count": latest_case_evidence_preview_gate_count,
            "latest_execution_case_evidence_preview_ready_count": latest_case_evidence_preview_ready_count,
            "latest_execution_case_evidence_preview_blocked_count": latest_case_evidence_preview_blocked_count,
            "latest_execution_case_evidence_preview_verdicts": latest_case_evidence_preview_verdicts,
            "latest_execution_case_evidence_preview_latest_verdict": latest_case_evidence_preview_latest_verdict,
            "latest_execution_case_evidence_preview_latest_event_id": latest_case_evidence_preview_latest_event_id,
            "latest_execution_case_evidence_preview_latest_handoff": latest_case_evidence_preview_latest_handoff,
            "latest_execution_case_evidence_preview_latest_handoff_present": latest_case_evidence_preview_latest_handoff_present,
            "latest_execution_case_evidence_preview_next_command": latest_case_evidence_preview_next_command,
            "latest_execution_case_evidence_preview_receipt_id": latest_case_evidence_preview_receipt_id,
            "latest_execution_case_evidence_preview_receipt_kind": latest_case_evidence_preview_receipt_kind,
            "latest_execution_case_evidence_preview_receipt_target_status": latest_case_evidence_preview_receipt_target_status,
            "execution_health_recovery_closure_state": recovery_state,
            "execution_health_recovery_closure_missing": recovery_missing,
            "execution_health_recovery_closure_missing_count": len(recovery_missing),
            "execution_health_recovery_closure_required_commands": recovery_required_commands,
            "execution_health_recovery_closure_next_required_command": recovery_next_required,
            "execution_health_recovery_closure_next_proof_command": metadata.get("execution_health_recovery_closure_next_proof_command"),
            "execution_health_recovery_closure_proof_queue": recovery_queue,
            "execution_health_recovery_closure_proof_queue_count": len(recovery_queue),
            "execution_health_recovery_closure_blocks_completion_claim": metadata.get("execution_health_recovery_closure_blocks_completion_claim"),
            "execution_health_verification_coverage_state": verification_state,
            "execution_health_verification_recent_tool_runs": verification_recent_tool_runs,
            "execution_health_verification_recent_action_runs": verification_recent_action_runs,
            "execution_health_verification_recent_verification_runs": verification_recent_verification_runs,
            "execution_health_verification_gap_count": verification_gap_count,
            "execution_health_verification_first_gap": verification_first_gap,
            "execution_health_verification_proof_queue": verification_queue,
            "execution_health_verification_proof_queue_count": len(verification_queue),
            "execution_health_verification_proof_queue_preview": verification_queue_preview,
            "execution_health_verification_proof_queue_preview_count": len(verification_queue_preview),
            "execution_health_verification_proof_queue_remaining_count": verification_queue_remaining_count,
            "execution_health_verification_next_required_command": verification_next_required,
            "execution_health_verification_next_proof_command": verification_next,
            "execution_health_verification_concrete_next_proof_command": verification_concrete_next,
            "execution_health_verification_row_next_command": verification_row_next,
            "execution_health_verification_blocks_completion_claim": verification_blocks_completion_claim,
            "execution_learning_state": learning_state,
            "execution_learning_missing": learning_missing,
            "execution_learning_missing_count": len(learning_missing),
            "execution_learning_required_commands": learning_required_commands,
            "execution_learning_next_required_command": learning_next_required,
            "execution_learning_next_proof_command": metadata.get("execution_learning_next_proof_command"),
            "execution_learning_proof_queue": learning_queue,
            "execution_learning_proof_queue_count": len(learning_queue),
            "execution_learning_evidence_command": metadata.get("execution_learning_evidence_command"),
            "execution_learning_after_action_learning_command": metadata.get("execution_learning_after_action_learning_command"),
            "execution_learning_actionable_required_commands": learning_actionable_required_commands,
            "execution_learning_actionable_required_command_count": len(learning_actionable_required_commands),
            "execution_learning_actionable_proof_queue": learning_actionable_queue,
            "execution_learning_actionable_proof_queue_count": len(learning_actionable_queue),
            "execution_learning_actionable_next_required_command": metadata.get("execution_learning_actionable_next_required_command") or "",
            "execution_learning_actionable_next_proof_command": learning_actionable_next,
            "execution_learning_next_evidence_command": metadata.get("execution_learning_next_evidence_command") or "",
            "execution_learning_blocks_completion_claim": metadata.get("execution_learning_blocks_completion_claim"),
            "storage_runtime_fallback_active": storage_runtime_fallback_active,
            "storage_runtime_fallback_reason": metadata.get("storage_runtime_fallback_reason"),
            "storage_runtime_fallback_exception_type": metadata.get("storage_runtime_fallback_exception_type"),
            "storage_runtime_fallback_db_path_display": metadata.get("storage_runtime_fallback_db_path_display"),
            "storage_runtime_fallback_vault_path_display": metadata.get("storage_runtime_fallback_vault_path_display"),
            "storage_readiness_blocks_completion_claim": storage_readiness_blocks_completion_claim,
            "storage_recovery_required": _metadata_bool(metadata.get("storage_recovery_required")),
            "storage_recovery_reason": metadata.get("storage_recovery_reason") or "",
            "storage_issues": storage_issues,
            "storage_issue_count": storage_issue_count,
            "storage_recovery_mode": storage_recovery_mode,
            "storage_recovery_next_operator_action": storage_recovery_next_operator_action,
            "storage_recovery_restart_required": storage_recovery_restart_required,
            "storage_recovery_check_command": metadata.get("storage_recovery_check_command") or "",
            "storage_recovery_check_tool_command": metadata.get("storage_recovery_check_tool_command") or "",
            "storage_recovery_check_api": metadata.get("storage_recovery_check_api") or STORAGE_RECOVERY_CHECK_API,
            "storage_recovery_command": metadata.get("storage_recovery_command") or "",
            "storage_readiness_blocker": storage_readiness_blocker,
            "storage_readiness_next_commands": storage_readiness_next_commands,
            "storage_readiness_next_command_count": len(storage_readiness_next_commands),
            "storage_readiness_next_required_command": storage_readiness_next_required_command,
            "storage_readiness_next_proof_command": storage_readiness_next_proof_command,
            "storage_readiness_proof_queue": storage_readiness_proof_queue,
            "storage_readiness_proof_queue_count": len(storage_readiness_proof_queue),
            "storage_readiness_proof_queue_preview": storage_readiness_proof_queue_preview,
            "storage_readiness_proof_queue_preview_count": len(storage_readiness_proof_queue_preview),
            "storage_readiness_proof_queue_remaining_count": storage_readiness_proof_queue_remaining_count,
            "storage_readiness_first_proof_command": storage_readiness_first_proof_command,
            "storage_handoff": storage_handoff,
            "agi_next_gate": agi_next_gate,
            "agi_next_target_title": agi_next_target_title,
            "agi_next_build_command": agi_next_build_command,
            "agi_next_evidence_closure_commands": metadata.get("agi_next_evidence_closure_commands"),
            "agi_next_evidence_closure_command_count": metadata.get("agi_next_evidence_closure_command_count"),
            "agi_next_focused_verification_commands": agi_next_focused_verification_commands,
            "agi_next_likely_files": agi_next_likely_files,
            "agi_next_likely_file_count": len(agi_next_likely_files),
            "agi_next_missing_target_files": agi_next_missing_target_files,
            "agi_next_missing_target_file_count": len(agi_next_missing_target_files),
            "agi_next_target_files_checked": metadata.get("agi_next_target_files_checked"),
            "agi_next_target_files_exist": metadata.get("agi_next_target_files_exist"),
            "agi_next_target_integrity_blocks_start": metadata.get("agi_next_target_integrity_blocks_start"),
            "agi_next_acceptance_checks": agi_next_acceptance_checks,
            "agi_next_acceptance_check_count": len(agi_next_acceptance_checks),
            **_agi_acceptance_gap_metadata({"agi_next_acceptance_checks": agi_next_acceptance_checks}),
            "agi_next_focused_verification_command_count": len(agi_next_focused_verification_commands),
            "agi_next_target_file_integrity_status": agi_next_target_file_integrity_status,
            "agi_next_build_packet_ready_for_review": agi_next_build_packet_ready,
            "agi_next_implementation_preflight_ready": agi_next_implementation_preflight_ready,
            "agi_next_implementation_preflight_blockers": agi_next_implementation_preflight_blockers,
            "agi_next_implementation_preflight_blocker_count": len(agi_next_implementation_preflight_blockers),
            "agi_next_implementation_preflight_blocker_preview": agi_next_implementation_preflight_blocker_preview,
            "agi_next_implementation_preflight_blocker_preview_count": len(agi_next_implementation_preflight_blocker_preview),
            "agi_next_implementation_preflight_first_blocker": agi_next_implementation_preflight_first_blocker,
            "agi_next_implementation_preflight_next_command": agi_next_implementation_preflight_next_command,
            "agi_next_implementation_preflight_row_next_command": agi_next_implementation_preflight_row_next_command,
            "agi_focus_selection_source": agi_focus_selection_source,
            "agi_focus_selection_reason": agi_focus_selection_reason,
            "agi_focus_canonical_selector_command": agi_focus_canonical_selector_command,
            "agi_focus_deliberate_focus_override": agi_focus_deliberate_focus_override,
            "agi_real_execution_gap_count": metadata.get("agi_real_execution_gap_count"),
            "real_execution_gap_count": real_execution_gap_count,
            "selected_real_execution_gap": selected_real_execution_gap,
            "selected_real_execution_gap_gate": selected_real_execution_gap_gate,
            "selected_real_execution_gap_detail": selected_real_execution_gap_detail,
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "calls_external_service": False,
            "executes_tools": False,
            "reads_personal_data": False,
            "reads_private_data": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "queues_approval": False,
            "requires_approval": False,
            "controls_computer": False,
            "speaks": False,
            "completes_tasks": False,
        }

        lines = [
            "Jarvis harness readiness digest:",
            "This is a compact read-only triage packet for Jarvis completion. It does not run proof commands or mark the goal done.",
            "",
            f"Objective: {objective}",
            f"Readiness state: {readiness_state}",
            f"Readiness verdict: {readiness_state}",
            f"Completion claim ready: {'yes' if completion_ready else 'no'}",
            f"Top risk: {top_risk}",
            f"Next command: `{next_command}`" if next_command else "Next command: none",
            f"Command source: {command_source}",
            f"Command reason: {command_reason}",
            "",
            "Readiness rows:",
        ]
        lines.extend(f"- {name}: {state}; next: `{next_step}`" for name, state, next_step in readiness_rows)
        lines.extend(
            [
                "",
                "Main blockers:",
            ]
        )
        if blockers:
            lines.extend(f"- {blocker}" for blocker in blockers[:8])
        else:
            lines.append("- none from the completion gate")
        lines.extend(
            [
                "",
                "Recovery and learning:",
                f"- recovery closure: {recovery_state}; missing: {', '.join(recovery_missing) if recovery_missing else 'none'}",
                f"- verification coverage: {verification_state}; queue: {', '.join(f'`{command}`' for command in verification_queue) if verification_queue else 'none'}",
                f"- verification first gap: {verification_first_gap or 'none'}",
                f"- verification queue preview: {len(verification_queue_preview)} shown, {verification_queue_remaining_count} remaining",
                f"- execution learning: {learning_state}; missing: {', '.join(learning_missing) if learning_missing else 'none'}",
                f"- actionable learning next required: `{learning_actionable_next_required}`" if learning_actionable_next_required else "- actionable learning next required: none",
                f"- actionable learning proof alias: `{learning_actionable_next}`" if learning_actionable_next else "- actionable learning proof alias: none",
                f"- actionable learning queue: {', '.join(f'`{command}`' for command in learning_actionable_queue) if learning_actionable_queue else 'none'}",
                f"- latest execution case review: {latest_case_review_state or 'unknown'}; verdict: {latest_case_review_verdict or 'unknown'}",
                f"- latest execution case review next safe command: `{latest_case_review_next_safe_command}`" if latest_case_review_next_safe_command else "- latest execution case review next safe command: none",
                f"- latest execution case review next command: `{latest_case_review_next_command}`" if latest_case_review_next_command else "- latest execution case review next command: none",
                f"- latest execution case review mission preview: {len(latest_case_review_mission_command_preview)} shown, {latest_case_review_mission_command_remaining_count} remaining",
                f"- latest execution case review approval forecast: {latest_case_review_forecast_queue_before} -> {latest_case_review_forecast_queue_after_if_sent} (delta {latest_case_review_forecast_queue_delta_if_sent})",
                f"- latest execution case closure: {latest_case_verdict}",
                f"- latest execution case evidence preflight: {latest_case_evidence_preview_ready_count}/{latest_case_evidence_preview_gate_count} ready; latest: {latest_case_evidence_preview_latest_verdict or 'none'}",
                f"- latest execution case evidence next command: `{latest_case_evidence_preview_next_command}`" if latest_case_evidence_preview_next_command else "- latest execution case evidence next command: none",
                f"- latest execution case evidence receipt: {latest_case_evidence_preview_receipt_kind or 'none'} {latest_case_evidence_preview_receipt_id or ''}".rstrip(),
                "",
                "Storage readiness:",
                f"- runtime fallback active: {'yes' if storage_runtime_fallback_active else 'no'}",
                f"- blocks completion claim: {'yes' if storage_readiness_blocks_completion_claim else 'no'}",
                f"- blocker: {storage_readiness_blocker or 'none'}",
                f"- first issue: {storage_first_issue or 'none'}",
                f"- issue preview: {', '.join(storage_issue_preview) if storage_issue_preview else 'none'}",
                f"- recovery mode: {storage_recovery_mode or 'none'}",
                f"- restart required: {'yes' if storage_recovery_restart_required else 'no'}",
                f"- next operator action: {storage_recovery_next_operator_action or 'none'}",
                f"- next storage required: `{storage_readiness_next_required_command}`" if storage_readiness_next_required_command else "- next storage required: none",
                f"- storage proof alias: `{storage_readiness_next_proof_command}`" if storage_readiness_next_proof_command else "- storage proof alias: none",
                f"- storage proof queue: {', '.join(f'`{command}`' for command in storage_readiness_proof_queue) if storage_readiness_proof_queue else 'none'}",
                f"- recovery commands: {', '.join(f'`{command}`' for command in storage_readiness_next_commands) if storage_readiness_next_commands else 'none'}",
                "",
                "AGI harness focus:",
                f"- next gate: {agi_next_gate or 'none'}",
                f"- selected target: {agi_next_target_title or 'none'}",
                f"- next build command: `{agi_next_build_command}`" if agi_next_build_command else "- next build command: none",
                f"- selection source: {agi_focus_selection_source or 'unknown'}",
                f"- selection reason: {agi_focus_selection_reason or 'none'}",
                f"- canonical selector command: `{agi_focus_canonical_selector_command}`" if agi_focus_canonical_selector_command else "- canonical selector command: none",
                f"- deliberate focus override: {'yes' if agi_focus_deliberate_focus_override else 'no'}",
                f"- target file integrity: {agi_next_target_file_integrity_status}",
                f"- likely files: {', '.join(agi_next_likely_files) if agi_next_likely_files else 'none'}",
                f"- missing target files: {', '.join(agi_next_missing_target_files) if agi_next_missing_target_files else 'none'}",
                f"- acceptance checks: {len(agi_next_acceptance_checks)}",
                f"- focused verification commands: {len(agi_next_focused_verification_commands)}",
                f"- ready for scoped implementation review: {'yes' if agi_next_build_packet_ready else 'no'}",
                f"- implementation preflight ready: {'yes' if agi_next_implementation_preflight_ready else 'no'}",
                f"- implementation preflight blockers: {', '.join(agi_next_implementation_preflight_blockers) if agi_next_implementation_preflight_blockers else 'none'}",
                f"- implementation preflight first blocker: {agi_next_implementation_preflight_first_blocker or 'none'}",
                f"- implementation preflight blocker preview: {', '.join(agi_next_implementation_preflight_blocker_preview) if agi_next_implementation_preflight_blocker_preview else 'none'}",
                f"- implementation preflight next command: `{agi_next_implementation_preflight_next_command}`" if agi_next_implementation_preflight_next_command else "- implementation preflight next command: none",
                f"- pending approval preview: {len(approval_review_preview)} shown, {max(len(pending_approvals) - len(approval_review_preview), 0)} remaining",
                f"- first approval readiness command: `{approval_review_first_readiness_command}`" if approval_review_first_readiness_command else "- first approval readiness command: none",
                f"- approval review next command: `{approval_review_next_command}`" if approval_review_next_command else "- approval review next command: none",
                "",
                "Command ladder:",
            ]
        )
        lines.extend(f"- `{command}`" for command in command_ladder[:12])
        if len(command_ladder) > 12:
            lines.append(f"- ... {len(command_ladder) - 12} more command(s)")
        lines.extend(
            [
                "",
                "Stop rules:",
                "- Do not claim Jarvis is complete while this digest reports blocked proof debt, pending approvals, open tasks, failed runs, or AGI gate gaps.",
                "- Do not use this digest as approval for shell/code, computer control, personal data, external side effects, destructive actions, or risky actions.",
                "- the operator's explicit stop times, work windows, pause commands, and newer instructions override the goal and this digest.",
                "",
                "Boundary:",
                "- This digest does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, write notes, write the database, control the computer, call external services, speak, complete tasks, mark goals done, or queue approvals.",
            ]
        )

        return ToolResult(
            "harness_readiness_digest",
            proof.ok,
            "\n".join(lines),
            _safe_metadata(
                objective=objective,
                readiness_state=readiness_state,
                readiness_verdict=readiness_state,
                completion_claim_ready=completion_ready,
                allowed_to_claim=completion_ready,
                top_risk=top_risk,
                next_command=next_command,
                next_proof_command=next_command,
                completion_next_proof_command=next_command,
                command_source=command_source,
                command_reason=command_reason,
                blockers=metadata.get("blockers"),
                blocker_details=blockers,
                blocker_count=len(blockers),
                readiness_rows=readiness_row_dicts,
                readiness_row_count=len(readiness_row_dicts),
                command_ladder=command_ladder,
                command_ladder_count=len(command_ladder),
                command_ladder_first_command=command_ladder_first_command,
                command_ladder_preview=command_ladder_preview,
                command_ladder_preview_count=len(command_ladder_preview),
                command_ladder_remaining_count=command_ladder_remaining_count,
                command_ladder_storage_prefix=command_ladder_storage_prefix,
                command_ladder_storage_prefix_count=len(command_ladder_storage_prefix),
                completion_proof_queue=proof_queue,
                completion_proof_queue_count=len(proof_queue),
                completion_proof_queue_preview=completion_proof_queue_preview,
                completion_proof_queue_preview_count=len(completion_proof_queue_preview),
                completion_proof_queue_remaining_count=completion_proof_queue_remaining_count,
                approval_review_required=bool(pending_approvals),
                pending_approval_count=len(pending_approvals),
                pending_approval_preview=approval_review_preview,
                pending_approval_preview_count=len(approval_review_preview),
                pending_approval_preview_unreadable_rows=approval_review_preview_unreadable_rows,
                pending_approval_remaining_count=max(len(pending_approvals) - len(approval_review_preview), 0),
                pending_approval_first_id=approval_review_first_id,
                approval_review_first_readiness_command=approval_review_first_readiness_command,
                approval_review_first_proof_command=approval_review_first_proof_command,
                approval_review_next_command=approval_review_next_command,
                approval_review_row_next_command=approval_review_row_next_command,
                completion_next_proof_handoff=completion_next_proof_handoff,
                completion_next_proof_handoff_present=bool(completion_next_proof_handoff),
                latest_execution_case_found=latest_case_found,
                latest_execution_case_review_state=latest_case_review_state,
                latest_execution_case_review_verdict=latest_case_review_verdict,
                latest_execution_case_review_next_safe_command=latest_case_review_next_safe_command,
                latest_execution_case_review_next_command=latest_case_review_next_command,
                latest_execution_case_review_initial_governor_command=latest_case_review_initial_governor_command,
                latest_execution_case_review_mission_command_queue=latest_case_review_mission_command_queue,
                latest_execution_case_review_mission_command_count=len(latest_case_review_mission_command_queue),
                latest_execution_case_review_mission_command_preview=latest_case_review_mission_command_preview,
                latest_execution_case_review_mission_command_preview_count=len(latest_case_review_mission_command_preview),
                latest_execution_case_review_mission_command_remaining_count=latest_case_review_mission_command_remaining_count,
                latest_execution_case_review_forecast_queue_before=latest_case_review_forecast_queue_before,
                latest_execution_case_review_forecast_queue_after_if_sent=latest_case_review_forecast_queue_after_if_sent,
                latest_execution_case_review_forecast_queue_delta_if_sent=latest_case_review_forecast_queue_delta_if_sent,
                latest_execution_case_review_approval_required=latest_case_review_approval_required,
                latest_execution_case_review_checklist_items=latest_case_review_checklist_items,
                latest_execution_case_review_blocker_count=latest_case_review_blocker_count,
                latest_execution_case_review_handoff=latest_case_review_handoff,
                latest_execution_case_review_handoff_present=latest_case_review_handoff_present,
                latest_execution_case_closure_verdict=latest_case_verdict,
                latest_execution_case_closure_ready=latest_case_closure_ready,
                latest_execution_case_closure_blocks_completion_claim=latest_case_closure_blocks_completion_claim,
                latest_execution_case_closure_proof_queue=latest_case_closure_queue,
                latest_execution_case_closure_proof_queue_count=len(latest_case_closure_queue),
                latest_execution_case_closure_proof_queue_preview=latest_case_closure_queue_preview,
                latest_execution_case_closure_proof_queue_preview_count=len(latest_case_closure_queue_preview),
                latest_execution_case_closure_proof_queue_remaining_count=latest_case_closure_queue_remaining_count,
                latest_execution_case_closure_first_proof_command=latest_case_closure_first_proof_command,
                latest_execution_case_next_closure_proof_command=latest_case_next_closure_proof_command,
                latest_execution_case_concrete_next_closure_proof_command=latest_case_concrete_next_closure_proof_command,
                latest_execution_case_row_next_closure_command=latest_case_row_next_closure_command,
                latest_execution_case_learning_state=metadata.get("latest_execution_case_learning_state") or "",
                latest_execution_case_learning_missing=metadata.get("latest_execution_case_learning_missing") or [],
                latest_execution_case_learning_next_required_command=metadata.get("latest_execution_case_learning_next_required_command") or "",
                latest_execution_case_learning_required_commands=metadata.get("latest_execution_case_learning_required_commands") or [],
                latest_execution_case_learning_proof_queue=metadata.get("latest_execution_case_learning_proof_queue") or [],
                latest_execution_case_learning_proof_queue_count=len(metadata.get("latest_execution_case_learning_proof_queue") or []),
                latest_execution_case_learning_next_proof_command=metadata.get("latest_execution_case_learning_next_proof_command") or "",
                latest_execution_case_learning_actionable_required_commands=metadata.get("latest_execution_case_learning_actionable_required_commands") or [],
                latest_execution_case_learning_actionable_proof_queue=metadata.get("latest_execution_case_learning_actionable_proof_queue") or [],
                latest_execution_case_learning_actionable_proof_queue_count=len(metadata.get("latest_execution_case_learning_actionable_proof_queue") or []),
                latest_execution_case_learning_actionable_next_required_command=metadata.get("latest_execution_case_learning_actionable_next_required_command") or "",
                latest_execution_case_learning_actionable_next_proof_command=latest_case_learning_actionable_next,
                latest_execution_case_learning_next_evidence_command=metadata.get("latest_execution_case_learning_next_evidence_command") or "",
                latest_execution_case_learning_blocks_completion_claim=latest_case_learning_blocks_completion_claim,
                latest_execution_case_evidence_preview_gate_count=latest_case_evidence_preview_gate_count,
                latest_execution_case_evidence_preview_ready_count=latest_case_evidence_preview_ready_count,
                latest_execution_case_evidence_preview_blocked_count=latest_case_evidence_preview_blocked_count,
                latest_execution_case_evidence_preview_verdicts=latest_case_evidence_preview_verdicts,
                latest_execution_case_evidence_preview_latest_verdict=latest_case_evidence_preview_latest_verdict,
                latest_execution_case_evidence_preview_latest_event_id=latest_case_evidence_preview_latest_event_id,
                latest_execution_case_evidence_preview_latest_handoff=latest_case_evidence_preview_latest_handoff,
                latest_execution_case_evidence_preview_latest_handoff_present=latest_case_evidence_preview_latest_handoff_present,
                latest_execution_case_evidence_preview_next_command=latest_case_evidence_preview_next_command,
                latest_execution_case_evidence_preview_receipt_id=latest_case_evidence_preview_receipt_id,
                latest_execution_case_evidence_preview_receipt_kind=latest_case_evidence_preview_receipt_kind,
                latest_execution_case_evidence_preview_receipt_target_status=latest_case_evidence_preview_receipt_target_status,
                execution_health_recovery_closure_state=recovery_state,
                execution_health_recovery_closure_missing=recovery_missing,
                execution_health_recovery_closure_missing_count=len(recovery_missing),
                execution_health_recovery_closure_required_commands=recovery_required_commands,
                execution_health_recovery_closure_next_required_command=recovery_next_required,
                execution_health_recovery_closure_next_proof_command=metadata.get("execution_health_recovery_closure_next_proof_command"),
                execution_health_recovery_closure_proof_queue=recovery_queue,
                execution_health_recovery_closure_proof_queue_count=len(recovery_queue),
                execution_health_recovery_closure_blocks_completion_claim=metadata.get("execution_health_recovery_closure_blocks_completion_claim"),
                execution_health_verification_coverage_state=verification_state,
                execution_health_verification_recent_tool_runs=verification_recent_tool_runs,
                execution_health_verification_recent_action_runs=verification_recent_action_runs,
                execution_health_verification_recent_verification_runs=verification_recent_verification_runs,
                execution_health_verification_gap_count=verification_gap_count,
                execution_health_verification_first_gap=verification_first_gap,
                execution_health_verification_proof_queue=verification_queue,
                execution_health_verification_proof_queue_count=len(verification_queue),
                execution_health_verification_proof_queue_preview=verification_queue_preview,
                execution_health_verification_proof_queue_preview_count=len(verification_queue_preview),
                execution_health_verification_proof_queue_remaining_count=verification_queue_remaining_count,
                execution_health_verification_next_required_command=verification_next_required,
                execution_health_verification_next_proof_command=verification_next,
                execution_health_verification_concrete_next_proof_command=verification_concrete_next,
                execution_health_verification_row_next_command=verification_row_next,
                execution_health_verification_blocks_completion_claim=verification_blocks_completion_claim,
                execution_learning_state=learning_state,
                execution_learning_missing=learning_missing,
                execution_learning_missing_count=len(learning_missing),
                execution_learning_required_commands=learning_required_commands,
                execution_learning_next_required_command=learning_next_required,
                execution_learning_next_proof_command=metadata.get("execution_learning_next_proof_command"),
                execution_learning_proof_queue=learning_queue,
                execution_learning_proof_queue_count=len(learning_queue),
                execution_learning_evidence_command=metadata.get("execution_learning_evidence_command"),
                execution_learning_after_action_learning_command=metadata.get("execution_learning_after_action_learning_command"),
                execution_learning_actionable_required_commands=learning_actionable_required_commands,
                execution_learning_actionable_required_command_count=len(learning_actionable_required_commands),
                execution_learning_actionable_proof_queue=learning_actionable_queue,
                execution_learning_actionable_proof_queue_count=len(learning_actionable_queue),
                execution_learning_actionable_next_required_command=metadata.get("execution_learning_actionable_next_required_command") or "",
                execution_learning_actionable_next_proof_command=learning_actionable_next,
                execution_learning_next_evidence_command=metadata.get("execution_learning_next_evidence_command") or "",
                execution_learning_blocks_completion_claim=metadata.get("execution_learning_blocks_completion_claim"),
                storage_runtime_fallback_active=storage_runtime_fallback_active,
                storage_runtime_fallback_reason=metadata.get("storage_runtime_fallback_reason"),
                storage_runtime_fallback_exception_type=metadata.get("storage_runtime_fallback_exception_type"),
                storage_runtime_fallback_db_path_display=metadata.get("storage_runtime_fallback_db_path_display"),
                storage_runtime_fallback_vault_path_display=metadata.get("storage_runtime_fallback_vault_path_display"),
                storage_readiness_blocks_completion_claim=storage_readiness_blocks_completion_claim,
                storage_recovery_required=_metadata_bool(metadata.get("storage_recovery_required")),
                storage_recovery_reason=metadata.get("storage_recovery_reason") or "",
                storage_issues=storage_issues,
                storage_issue_count=storage_issue_count,
                storage_issue_preview=storage_issue_preview,
                storage_issue_preview_count=len(storage_issue_preview),
                storage_first_issue=storage_first_issue,
                storage_recovery_mode=storage_recovery_mode,
                storage_recovery_next_operator_action=storage_recovery_next_operator_action,
                storage_recovery_restart_required=storage_recovery_restart_required,
                storage_recovery_check_command=metadata.get("storage_recovery_check_command") or "",
                storage_recovery_check_tool_command=metadata.get("storage_recovery_check_tool_command") or "",
                storage_recovery_check_api=metadata.get("storage_recovery_check_api") or STORAGE_RECOVERY_CHECK_API,
                storage_recovery_command=metadata.get("storage_recovery_command") or "",
                storage_readiness_blocker=storage_readiness_blocker,
                storage_readiness_next_commands=storage_readiness_next_commands,
                storage_readiness_next_command_count=len(storage_readiness_next_commands),
                storage_readiness_next_required_command=storage_readiness_next_required_command,
                storage_readiness_next_proof_command=storage_readiness_next_proof_command,
                storage_readiness_proof_queue=storage_readiness_proof_queue,
                storage_readiness_proof_queue_count=len(storage_readiness_proof_queue),
                storage_readiness_proof_queue_preview=storage_readiness_proof_queue_preview,
                storage_readiness_proof_queue_preview_count=len(storage_readiness_proof_queue_preview),
                storage_readiness_proof_queue_remaining_count=storage_readiness_proof_queue_remaining_count,
                storage_readiness_first_proof_command=storage_readiness_first_proof_command,
                storage_handoff=storage_handoff,
                agi_next_gate=agi_next_gate,
                agi_next_target_title=agi_next_target_title,
                agi_next_build_command=agi_next_build_command,
                agi_next_evidence_closure_commands=metadata.get("agi_next_evidence_closure_commands"),
                agi_next_evidence_closure_command_count=metadata.get("agi_next_evidence_closure_command_count"),
                agi_next_focused_verification_commands=agi_next_focused_verification_commands,
                agi_next_focused_verification_command_count=len(agi_next_focused_verification_commands),
                agi_next_likely_files=agi_next_likely_files,
                agi_next_likely_file_count=len(agi_next_likely_files),
                agi_next_target_file_integrity_status=agi_next_target_file_integrity_status,
                agi_next_target_files_checked=metadata.get("agi_next_target_files_checked"),
                agi_next_target_files_exist=metadata.get("agi_next_target_files_exist"),
                agi_next_missing_target_files=agi_next_missing_target_files,
                agi_next_missing_target_file_count=len(agi_next_missing_target_files),
                agi_next_target_integrity_blocks_start=metadata.get("agi_next_target_integrity_blocks_start"),
                agi_next_acceptance_checks=agi_next_acceptance_checks,
                agi_next_acceptance_check_count=len(agi_next_acceptance_checks),
                **_agi_acceptance_gap_metadata({"agi_next_acceptance_checks": agi_next_acceptance_checks}),
                agi_next_build_packet_ready_for_review=agi_next_build_packet_ready,
                agi_next_implementation_preflight_ready=agi_next_implementation_preflight_ready,
                agi_next_implementation_preflight_blockers=agi_next_implementation_preflight_blockers,
                agi_next_implementation_preflight_blocker_count=len(agi_next_implementation_preflight_blockers),
                agi_next_implementation_preflight_blocker_preview=agi_next_implementation_preflight_blocker_preview,
                agi_next_implementation_preflight_blocker_preview_count=len(agi_next_implementation_preflight_blocker_preview),
                agi_next_implementation_preflight_first_blocker=agi_next_implementation_preflight_first_blocker,
                agi_next_implementation_preflight_next_command=agi_next_implementation_preflight_next_command,
                agi_next_implementation_preflight_row_next_command=agi_next_implementation_preflight_row_next_command,
                agi_focus_selection_source=agi_focus_selection_source,
                agi_focus_selection_reason=agi_focus_selection_reason,
                agi_focus_canonical_selector_command=agi_focus_canonical_selector_command,
                agi_focus_deliberate_focus_override=agi_focus_deliberate_focus_override,
                agi_real_execution_gap_count=metadata.get("agi_real_execution_gap_count"),
                real_execution_gap_count=real_execution_gap_count,
                selected_real_execution_gap=selected_real_execution_gap,
                selected_real_execution_gap_gate=selected_real_execution_gap_gate,
                selected_real_execution_gap_detail=selected_real_execution_gap_detail,
                harness_readiness_handoff_ready=True,
                harness_readiness_handoff=harness_readiness_handoff,
                source_tool="completion_next_proof_packet",
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
            ),
        )

    def execution_proof_bundle(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("objective"), limit=600)
        supplied_evidence = _short(args.get("evidence") or args.get("proof"), limit=400)
        supplied_tests = _short(args.get("tests") or args.get("test_plan") or args.get("verification_tests"), limit=400)
        supplied_verification = _short(args.get("verification") or args.get("verify") or args.get("expected"), limit=400)
        supplied_recovery = _short(args.get("recovery") or args.get("rollback") or args.get("stop"), limit=400)
        supplied_approval = _short(args.get("approval") or args.get("approval_id") or args.get("approval_receipt"), limit=180)

        if not request:
            output = (
                "Request is required. "
                "Try `execution proof bundle: organize my downloads and summarize what changed`. "
                f"{HARNESS_REQUEST_RECOVERY_ACTION}"
            )
            return ToolResult(
                "execution_proof_bundle",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(reason="missing_request"),
                    output=output,
                    action=HARNESS_REQUEST_RECOVERY_ACTION,
                ),
            )

        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        tools_by_name = {tool.name: tool for tool in tools}
        pending = store.list_pending_approvals(limit=10)
        recent_runs = store.recent_tool_runs(limit=50)
        readable_recent_runs, unreadable_recent_run_rows = _readable_tool_run_rows(recent_runs)
        recovery_closure = _execution_health_recovery_closure_snapshot(store, readable_recent_runs)
        learning_debt = _execution_learning_debt_snapshot(readable_recent_runs)
        open_tasks = store.list_tasks(status="open", limit=10)
        risk_signals = _risk_signals(request)
        approval_required = bool(risk_signals)
        from jarvis_v2.agent.planner import RuleBasedPlanner

        plan = RuleBasedPlanner().plan(request, allow_risky_natural_dispatch=False)
        planned_actions: list[dict[str, Any]] = []
        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                requires_approval = True
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
            approval_required = approval_required or requires_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "args": {str(key): _short(value, limit=180) for key, value in action.args.items()},
                    "reason": _short(action.reason, limit=180),
                }
            )
        approval_queue_forecast = []
        for action, item in zip(plan.actions, planned_actions):
            if not item["requires_approval"]:
                continue
            existing = store.find_matching_pending_approval(request, action.tool_name, action.args)
            existing_id = int(existing["id"]) if existing is not None else None
            approval_queue_forecast.append(
                {
                    "tool_name": action.tool_name,
                    "risk": item["risk"],
                    "existing_approval_id": existing_id,
                    "would_queue_new_approval": existing_id is None,
                    "would_reuse_pending_approval": existing_id is not None,
                    "planned_arg_keys": sorted(str(key) for key in action.args),
                }
            )
        forecast_new_approvals = sum(1 for item in approval_queue_forecast if item["would_queue_new_approval"])
        forecast_reused_approval_ids = [
            item["existing_approval_id"]
            for item in approval_queue_forecast
            if item["existing_approval_id"] is not None
        ]
        forecast_queue_before = len(pending)
        forecast_queue_after_if_sent = forecast_queue_before + forecast_new_approvals
        supplied_approval_id = _approval_id_from_text(supplied_approval)
        approval_chain_status = "not supplied"
        approval_chain_valid = False
        approval_linked_run_ids: list[int] = []
        supplied_verification_refs = _verification_receipt_refs_from_text(supplied_verification)
        supplied_verification_run_ids = [receipt_id for kind, receipt_id in supplied_verification_refs if kind == "tool_run"]
        supplied_runtime_trace_message_ids = [receipt_id for kind, receipt_id in supplied_verification_refs if kind == "runtime_trace_receipt"]
        missing_supplied_verification_run_ids = _missing_tool_run_ids(store, supplied_verification_run_ids)
        missing_supplied_runtime_trace_message_ids = _missing_runtime_trace_message_ids(store, supplied_runtime_trace_message_ids)
        verified_approval_run_ids: list[int] = []
        runtime_trace_verified_approval_run_ids: list[int] = []
        if supplied_approval_id is not None:
            approval_row = _approval_any_status(store, supplied_approval_id)
            if approval_row is None:
                approval_chain_status = "approval not found"
            else:
                linked_runs = store.approved_tool_runs_for_approvals([supplied_approval_id], limit=10)
                approval_linked_run_ids = [int(row["id"]) for row in linked_runs]
                successful_linked_runs = [row for row in linked_runs if bool(row["ok"])]
                status = str(approval_row["status"]).lower()
                if status != "approved":
                    approval_chain_status = f"approval {status}"
                elif not linked_runs:
                    approval_chain_status = "approved without linked rerun"
                elif not successful_linked_runs:
                    approval_chain_status = "linked rerun failed"
                else:
                    approval_chain_status = "approved linked rerun proven"
                    approval_chain_valid = True
                runtime_trace_verified_approval_run_ids = _approved_run_ids_from_runtime_trace_messages(
                    store,
                    supplied_runtime_trace_message_ids,
                    supplied_approval_id,
                )
                verified_approval_run_ids = sorted(
                    (set(approval_linked_run_ids) & set(supplied_verification_run_ids))
                    | (set(approval_linked_run_ids) & set(runtime_trace_verified_approval_run_ids))
                )

        proof_lanes = [
            ("route", {"execution_governor_packet", "dispatch_decision_packet", "planner_gap_packet", "execution_contract"}),
            ("arguments", {"argument_contract_packet", "execution_readiness_matrix"}),
            ("risk gate", {"risk_preflight", "action_rehearsal", "approval_readiness_packet", "approval_execution_packet", "review_pending_approvals"}),
            ("verification", {"verification_packet", "verification_receipt", "runtime_trace_receipt"}),
            ("acceptance", {"execution_acceptance_gate", "task_completion_packet", "completion_audit_packet"}),
            ("audit", {"recent_tool_runs", "execution_audit_gate", "execution_recovery_packet", "execution_health_report", "evidence_ledger"}),
            ("recovery", {"checkpoint_recovery_preview", "checkpoint_recovery_receipt", "checkpoint_recovery_followthrough_packet", "autonomy_resume_gate", "autonomy_continuation_execution_packet", "autonomy_step_closure_packet", "autonomy_cycle_ledger", "work_block_checkpoint"}),
            ("learning", {"after_action_learning_packet", "failure_to_test_preview", "failure_promotion_packet", "learning_review"}),
        ]

        lane_rows = []
        missing_lanes = 0
        partial_lanes = 0
        strong_lanes = 0
        for name, required in proof_lanes:
            present = sorted(required & tool_names)
            if len(present) >= 2:
                status = "strong"
                strong_lanes += 1
            elif present:
                status = "partial"
                partial_lanes += 1
            else:
                status = "missing"
                missing_lanes += 1
            lane_rows.append((name, status, present, sorted(required - tool_names)))
        proof_lane_rows = [
            {
                "name": name,
                "status": status,
                "present": present,
                "present_count": len(present),
                "missing": missing,
                "missing_count": len(missing),
            }
            for name, status, present, missing in lane_rows
        ]

        missing_receipts = []
        if approval_required and not supplied_approval:
            missing_receipts.append("approval receipt for risky execution")
        if not supplied_verification:
            missing_receipts.append("verification target")
        if not supplied_tests:
            missing_receipts.append("test or smoke-test evidence")
        if not supplied_evidence:
            missing_receipts.append("outcome evidence")
        if not supplied_recovery:
            missing_receipts.append("recovery or stop-condition evidence")

        failed_runs = [row for row in readable_recent_runs if not _row_bool(row, "ok")]
        verification_runs = [
            row
            for row in readable_recent_runs
            if _row_text(row, "tool_name") in {"verification_receipt", "runtime_trace_receipt", "verification_packet", "execution_acceptance_gate", "execution_proof_bundle"}
        ]

        blockers = []
        if unreadable_recent_run_rows:
            blockers.append("recent tool-run audit includes unreadable row(s); review recent tool runs first")
        if pending:
            blockers.append(f"{len(pending)} pending approval(s) should be reviewed before new risky execution")
        if approval_required and not supplied_approval:
            blockers.append("risky request has no linked approval receipt")
        if approval_required and supplied_approval and supplied_approval_id is None:
            blockers.append("supplied approval receipt must include a concrete approval id")
        if approval_required and supplied_approval_id is not None and not approval_chain_valid:
            blockers.append(f"supplied approval receipt is not valid execution proof: {approval_chain_status}")
        if approval_required and approval_chain_valid and not (
            supplied_verification_run_ids or supplied_runtime_trace_message_ids
        ):
            blockers.append("risky proof needs a concrete verification/runtime-trace receipt id for the approved rerun")
        if approval_required and approval_chain_valid and supplied_verification_run_ids and not verified_approval_run_ids:
            blockers.append("supplied verification receipt does not match any linked approved rerun")
        if missing_supplied_verification_run_ids:
            blockers.append(
                "supplied verification/tool-run receipt id(s) do not exist in audit log: "
                + ", ".join(str(item) for item in missing_supplied_verification_run_ids)
            )
        if missing_supplied_runtime_trace_message_ids:
            blockers.append(
                "supplied runtime-trace receipt message id(s) do not exist or lack trace metadata: "
                + ", ".join(str(item) for item in missing_supplied_runtime_trace_message_ids)
            )
        if missing_receipts:
            blockers.append("missing supplied proof: " + ", ".join(missing_receipts))
        if missing_lanes:
            blockers.append("missing harness proof lanes: " + ", ".join(name for name, status, _, _ in lane_rows if status == "missing"))
        if failed_runs:
            blockers.append(f"{len(failed_runs)} recent failed or blocked run(s) need review")

        if blockers:
            proof_state = "PROOF_BUNDLE_BLOCKED"
            can_trust_execution = False
        elif partial_lanes:
            proof_state = "PROOF_BUNDLE_NEEDS_MORE_EVIDENCE"
            can_trust_execution = False
        else:
            proof_state = "PROOF_BUNDLE_READY_FOR_HUMAN_REVIEW"
            can_trust_execution = True

        first_pending_approval_id_text = _first_positive_int_text(pending)
        first_failed_run_id_text = _first_positive_int_text(failed_runs)
        first_pending_approval_id = int(first_pending_approval_id_text) if first_pending_approval_id_text else None
        first_failed_run_id = int(first_failed_run_id_text) if first_failed_run_id_text else None
        proof_requirements = [
            ("route", "execution governor", f"execution governor: {request}", True),
            ("arguments", "argument contract", f"argument contract: {request}", True),
            ("verification", "verification packet or receipt", f"verification packet: {request}", bool(supplied_verification)),
            ("tests", "focused or smoke-test evidence", "run focused smoke tests for the changed behavior", bool(supplied_tests)),
            ("evidence", "bounded outcome evidence", "attach a verification receipt or runtime trace receipt", bool(supplied_evidence)),
            ("recovery", "rollback or stop condition", "name the stop condition before accepting execution", bool(supplied_recovery)),
        ]
        if approval_required:
            proof_requirements.insert(
                2,
                (
                    "approval",
                    "approval readiness and last-look chain",
                    f"approval readiness {first_pending_approval_id or '<approval id>'} then approval packet {first_pending_approval_id or '<approval id>'} then approval chain proof {first_pending_approval_id or '<approval id>'}",
                    bool(supplied_approval) and approval_chain_valid,
                ),
            )
        proof_requirement_rows = [
            {
                "name": name,
                "label": label,
                "command": command,
                "present": present,
            }
            for name, label, command, present in proof_requirements
        ]
        missing_requirement_names = [name for name, _label, _command, present in proof_requirements if not present]
        next_proof_commands = [
            f"execution governor: {request}",
            f"dispatch decision: {request}",
            f"argument contract: {request}",
            f"verification packet: {request}",
            f"acceptance gate: {request}; evidence <receipt>; tests <smoke>; recovery <stop condition>",
            "execution audit gate",
        ]
        if first_pending_approval_id is not None:
            next_proof_commands.append(f"approval readiness {first_pending_approval_id}")
            next_proof_commands.append(f"approval packet {first_pending_approval_id}")
        if first_failed_run_id is not None:
            next_proof_commands.append(f"execution recovery packet {first_failed_run_id}")
            next_proof_commands.append(f"verification receipt {first_failed_run_id}")
        _append_unique(next_proof_commands, list(recovery_closure["required_commands"]))
        learning_closure_command = (
            f"execution learning closure {learning_debt['target_run_id']}"
            if learning_debt["blocks_completion_claim"] and learning_debt["target_run_id"] is not None
            else "execution learning closure"
            if learning_debt["blocks_completion_claim"]
            else ""
        )
        if learning_closure_command:
            _append_unique(next_proof_commands, [learning_closure_command])
        _append_unique(next_proof_commands, list(learning_debt["required_commands"]))

        lines = [
            "Jarvis execution proof bundle:",
            "This is the pre-trust packet for a real action. It is read-only and does not execute, approve, dismiss, write, control the computer, read private data, call external services, speak, complete tasks, or queue approvals.",
            "",
            f"Request: {request}",
            f"Proof state: {proof_state}",
            f"Can trust execution now: {'yes, after human review' if can_trust_execution else 'no'}",
            "",
            "Risk and approval:",
        ]
        if risk_signals:
            lines.extend(f"- {signal}: approval-gated" for signal in risk_signals)
        else:
            lines.append("- no obvious risky wording; ToolRegistry risk still decides real execution")
        lines.extend(
            [
                f"- approval required: {'yes' if approval_required else 'no'}",
                f"- supplied approval receipt: {supplied_approval or 'none'}",
                f"- approval chain status: {approval_chain_status}",
                f"- linked approved runs: {', '.join(str(item) for item in approval_linked_run_ids) if approval_linked_run_ids else 'none'}",
                f"- supplied verification/tool-run ids: {', '.join(str(item) for item in supplied_verification_run_ids) if supplied_verification_run_ids else 'none'}",
                f"- supplied runtime-trace message ids: {', '.join(str(item) for item in supplied_runtime_trace_message_ids) if supplied_runtime_trace_message_ids else 'none'}",
                f"- missing supplied verification targets: {', '.join(str(item) for item in missing_supplied_verification_run_ids) if missing_supplied_verification_run_ids else 'none'}",
                f"- missing supplied runtime-trace targets: {', '.join(str(item) for item in missing_supplied_runtime_trace_message_ids) if missing_supplied_runtime_trace_message_ids else 'none'}",
                f"- verified approved runs: {', '.join(str(item) for item in verified_approval_run_ids) if verified_approval_run_ids else 'none'}",
                f"- would queue new approvals if sent: {forecast_new_approvals}",
                f"- would reuse pending approval ids: {', '.join(str(item) for item in forecast_reused_approval_ids) if forecast_reused_approval_ids else 'none'}",
                f"- forecast queue after if sent: {forecast_queue_after_if_sent}",
                "",
                "Harness proof lanes:",
            ]
        )
        for name, status, present, missing in lane_rows:
            lines.extend(
                [
                    f"- {name}: {status}",
                    f"  present: {', '.join(present) if present else 'none'}",
                    f"  missing: {', '.join(missing) if missing else 'none'}",
                ]
            )

        lines.extend(["", "Evidence gap summary:"])
        for name, label, command, present in proof_requirements:
            state = "present" if present else "missing"
            lines.append(f"- {name}: {state} | {label}")
            if not present:
                lines.append(f"  next: `{command}`")

        lines.extend(
            [
                "",
                "Supplied proof receipts:",
                f"- verification: {supplied_verification or 'missing'}",
                f"- tests: {supplied_tests or 'missing'}",
                f"- evidence: {supplied_evidence or 'missing'}",
                f"- recovery/stop condition: {supplied_recovery or 'missing'}",
                "",
                "Current runtime evidence:",
                f"- pending approvals: {len(pending)}",
                f"- open tasks: {len(open_tasks)}",
                f"- recent runs inspected: {len(recent_runs)}",
                f"- readable recent runs: {len(readable_recent_runs)}",
                f"- unreadable recent tool run rows: {unreadable_recent_run_rows}",
                f"- recent verification/audit packets: {len(verification_runs)}",
                f"- recent failed/blocked runs: {len(failed_runs)}",
                f"- recovery closure state: {recovery_closure['state']}",
                f"- recovery closure missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- execution learning debt state: {learning_debt['state']}",
                f"- execution learning debt missing: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
                "",
                "Blockers:",
            ]
        )
        if blockers:
            lines.extend(f"- {blocker}" for blocker in blockers)
        else:
            lines.append("- none found by this read-only bundle")

        lines.extend(
            [
                "",
                "Next required commands:",
            ]
        )
        lines.extend(f"- `{command}`" for command in next_proof_commands)
        lines.extend(
            [
                "",
                "Execution health recovery closure:",
                f"- state: {recovery_closure['state']}",
                f"- ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
                f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
                f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- recovery next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- recovery next required: none",
                f"- command queue: {', '.join(f'`{command}`' for command in recovery_closure['required_commands']) if recovery_closure['required_commands'] else 'none'}",
                "",
                "Execution learning debt:",
                f"- state: {learning_debt['state']}",
                f"- blocks completion claim: {'yes' if learning_debt['blocks_completion_claim'] else 'no'}",
                f"- target run: #{learning_debt['target_run_id']} `{learning_debt['target_tool_name']}`" if learning_debt["target_run_id"] is not None else "- target run: none",
                f"- missing: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
                f"- learning evidence next required: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- learning evidence next required: none",
                f"- learning actionable next required: `{learning_debt['actionable_next_required_command']}`" if learning_debt["actionable_next_required_command"] else "- learning actionable next required: none",
                f"- learning actionable proof alias: `{learning_debt['actionable_next_proof_command']}`" if learning_debt["actionable_next_proof_command"] else "- learning actionable proof alias: none",
                f"- learning closure command: `{learning_closure_command}`" if learning_closure_command else "- learning closure command: none",
                f"- learning next required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- learning next required: none",
                f"- actionable learning queue: {', '.join(f'`{command}`' for command in learning_debt['actionable_required_commands']) if learning_debt['actionable_required_commands'] else 'none'}",
                f"- learning proof queue: {', '.join(f'`{command}`' for command in learning_debt['required_commands']) if learning_debt['required_commands'] else 'none'}",
            ]
        )
        execution_proof_handoff = {
            "source": "execution_proof_bundle",
            "execution_proof_handoff_ready": True,
            "handoff_ready": True,
            "request": request,
            "proof_state": proof_state,
            "can_trust_execution": can_trust_execution,
            "approval_required": approval_required,
            "risk_signals": risk_signals,
            "risk_signal_count": len(risk_signals),
            "planned_actions": planned_actions,
            "planned_action_count": len(planned_actions),
            "approval_queue_forecast": approval_queue_forecast,
            "forecast_new_approvals": forecast_new_approvals,
            "forecast_reused_approval_ids": forecast_reused_approval_ids,
            "forecast_queue_before": forecast_queue_before,
            "forecast_queue_after_if_sent": forecast_queue_after_if_sent,
            "forecast_queue_delta_if_sent": forecast_new_approvals,
            "supplied_approval": bool(supplied_approval),
            "supplied_approval_id": supplied_approval_id,
            "approval_chain_status": approval_chain_status,
            "approval_chain_valid": approval_chain_valid,
            "approval_linked_run_ids": approval_linked_run_ids,
            "supplied_verification": bool(supplied_verification),
            "supplied_tests": bool(supplied_tests),
            "supplied_evidence": bool(supplied_evidence),
            "supplied_recovery": bool(supplied_recovery),
            "missing_receipts": missing_receipts,
            "missing_receipt_count": len(missing_receipts),
            "blockers": blockers,
            "blocker_count": len(blockers),
            "pending_approvals": len(pending),
            "open_tasks": len(open_tasks),
            "recent_runs": len(recent_runs),
            "readable_recent_tool_runs": len(readable_recent_runs),
            "unreadable_recent_tool_run_rows": unreadable_recent_run_rows,
            "recent_failed_runs": len(failed_runs),
            "verification_runs": len(verification_runs),
            "proof_lane_rows": proof_lane_rows,
            "proof_lanes": len(proof_lanes),
            "strong_lanes": strong_lanes,
            "partial_lanes": partial_lanes,
            "missing_lanes": missing_lanes,
            "proof_requirement_rows": proof_requirement_rows,
            "proof_requirements": len(proof_requirements),
            "missing_requirement_names": missing_requirement_names,
            "missing_requirement_count": len(missing_requirement_names),
            "next_proof_commands": next_proof_commands,
            "next_proof_command": next_proof_commands[0] if next_proof_commands else "",
            "first_pending_approval_id": first_pending_approval_id,
            "first_failed_run_id": first_failed_run_id,
            "execution_health_recovery_closure_state": recovery_closure["state"],
            "execution_health_recovery_closure_ready_to_retry": recovery_closure["ready_to_retry"],
            "execution_health_recovery_closure_missing": recovery_closure["missing"],
            "execution_health_recovery_closure_missing_count": recovery_closure["missing_count"],
            "execution_health_recovery_closure_required_commands": recovery_closure["required_commands"],
            "execution_health_recovery_closure_next_required_command": recovery_closure["next_required_command"],
            "execution_health_recovery_closure_checklist_command": _recovery_closure_checklist_command(recovery_closure),
            "execution_health_recovery_closure_should_open_checklist": bool(_recovery_closure_checklist_command(recovery_closure)),
            "execution_health_recovery_closure_proof_queue": recovery_closure["proof_queue"],
            "execution_health_recovery_closure_proof_queue_count": recovery_closure["proof_queue_count"],
            "execution_health_recovery_closure_next_proof_command": recovery_closure["next_proof_command"],
            "execution_health_recovery_closure_blocks_completion_claim": recovery_closure["blocks_completion_claim"],
            "execution_health_recovery_closure_target_run_id": recovery_closure["target_run_id"],
            "execution_health_recovery_closure_target_tool_name": recovery_closure["target_tool_name"],
            "execution_health_recovery_closure_target_verification_receipts": recovery_closure["target_verification_receipts"],
            "execution_health_recovery_closure_target_recovery_packets": recovery_closure["target_recovery_packets"],
            "execution_health_recovery_closure_target_after_action_learning_packets": recovery_closure["target_after_action_learning_packets"],
            "execution_learning_state": learning_debt["state"],
            "execution_learning_blocks_completion_claim": learning_debt["blocks_completion_claim"],
            "execution_learning_recent_action_runs": learning_debt["recent_action_runs"],
            "execution_learning_failed_or_blocked_action_runs": learning_debt["failed_or_blocked_action_runs"],
            "execution_learning_recent_verification_runs": learning_debt["recent_verification_runs"],
            "execution_learning_recent_recovery_runs": learning_debt["recent_recovery_runs"],
            "execution_learning_recent_after_action_learning_runs": learning_debt["recent_after_action_learning_runs"],
            "execution_learning_target_run_id": learning_debt["target_run_id"],
            "execution_learning_target_tool_name": learning_debt["target_tool_name"],
            "execution_learning_target_after_action_learning_packets": learning_debt["target_after_action_learning_packets"],
            "execution_learning_missing": learning_debt["missing"],
            "execution_learning_missing_count": learning_debt["missing_count"],
            "execution_learning_closure_command": learning_closure_command,
            "execution_learning_evidence_command": learning_debt["learning_evidence_command"],
            "execution_learning_after_action_learning_command": learning_debt["after_action_learning_command"],
            "execution_learning_required_commands": learning_debt["required_commands"],
            "execution_learning_next_required_command": learning_debt["next_required_command"],
            "execution_learning_proof_queue": learning_debt["proof_queue"],
            "execution_learning_proof_queue_count": learning_debt["proof_queue_count"],
            "execution_learning_next_proof_command": learning_debt["next_proof_command"],
            "execution_learning_actionable_required_commands": learning_debt["actionable_required_commands"],
            "execution_learning_actionable_required_command_count": len(learning_debt["actionable_required_commands"]),
            "execution_learning_actionable_proof_queue": learning_debt["actionable_required_commands"],
            "execution_learning_actionable_proof_queue_count": len(learning_debt["actionable_required_commands"]),
            "execution_learning_actionable_next_required_command": learning_debt["actionable_next_required_command"],
            "execution_learning_actionable_next_proof_command": learning_debt["actionable_next_proof_command"],
            "execution_learning_next_evidence_command": learning_debt["next_evidence_command"],
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "calls_external_service": False,
            "executes_tools": False,
            "reads_personal_data": False,
            "reads_private_data": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "queues_approval": False,
            "requires_approval": False,
            "controls_computer": False,
            "speaks": False,
            "completes_tasks": False,
        }

        return ToolResult(
            "execution_proof_bundle",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                proof_state=proof_state,
                risk_signals=risk_signals,
                approval_required=approval_required,
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                approval_queue_forecast=approval_queue_forecast,
                forecast_new_approvals=forecast_new_approvals,
                forecast_reused_approval_ids=forecast_reused_approval_ids,
                forecast_queue_before=forecast_queue_before,
                forecast_queue_after_if_sent=forecast_queue_after_if_sent,
                forecast_queue_delta_if_sent=forecast_new_approvals,
                supplied_approval=bool(supplied_approval),
                supplied_approval_id=supplied_approval_id,
                approval_chain_status=approval_chain_status,
                approval_chain_valid=approval_chain_valid,
                approval_linked_run_ids=approval_linked_run_ids,
                supplied_verification_run_ids=supplied_verification_run_ids,
                supplied_runtime_trace_message_ids=supplied_runtime_trace_message_ids,
                missing_supplied_verification_run_ids=missing_supplied_verification_run_ids,
                missing_supplied_runtime_trace_message_ids=missing_supplied_runtime_trace_message_ids,
                has_missing_supplied_verification_runs=bool(missing_supplied_verification_run_ids),
                has_missing_supplied_runtime_trace_messages=bool(missing_supplied_runtime_trace_message_ids),
                runtime_trace_verified_approval_run_ids=runtime_trace_verified_approval_run_ids,
                verified_approval_run_ids=verified_approval_run_ids,
                has_verified_approval_run=bool(verified_approval_run_ids),
                supplied_verification=bool(supplied_verification),
                supplied_tests=bool(supplied_tests),
                supplied_evidence=bool(supplied_evidence),
                supplied_recovery=bool(supplied_recovery),
                missing_receipts=missing_receipts,
                missing_receipt_count=len(missing_receipts),
                blockers=len(blockers),
                blocker_details=blockers,
                blocker_count=len(blockers),
                pending_approvals=len(pending),
                open_tasks=len(open_tasks),
                recent_runs=len(recent_runs),
                readable_recent_tool_runs=len(readable_recent_runs),
                unreadable_recent_tool_run_rows=unreadable_recent_run_rows,
                recent_failed_runs=len(failed_runs),
                verification_runs=len(verification_runs),
                execution_health_recovery_closure_state=recovery_closure["state"],
                execution_health_recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                execution_health_recovery_closure_missing=recovery_closure["missing"],
                execution_health_recovery_closure_missing_count=recovery_closure["missing_count"],
                execution_health_recovery_closure_required_commands=recovery_closure["required_commands"],
                execution_health_recovery_closure_next_required_command=recovery_closure["next_required_command"],
                execution_health_recovery_closure_checklist_command=_recovery_closure_checklist_command(recovery_closure),
                execution_health_recovery_closure_should_open_checklist=bool(_recovery_closure_checklist_command(recovery_closure)),
                execution_health_recovery_closure_proof_queue=recovery_closure["proof_queue"],
                execution_health_recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
                execution_health_recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
                execution_health_recovery_closure_blocks_completion_claim=recovery_closure["blocks_completion_claim"],
                execution_health_recovery_closure_target_run_id=recovery_closure["target_run_id"],
                execution_health_recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                execution_health_recovery_closure_target_verification_receipts=recovery_closure["target_verification_receipts"],
                execution_health_recovery_closure_target_recovery_packets=recovery_closure["target_recovery_packets"],
                execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure["target_after_action_learning_packets"],
                execution_learning_state=learning_debt["state"],
                execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
                execution_learning_recent_action_runs=learning_debt["recent_action_runs"],
                execution_learning_failed_or_blocked_action_runs=learning_debt["failed_or_blocked_action_runs"],
                execution_learning_recent_verification_runs=learning_debt["recent_verification_runs"],
                execution_learning_recent_recovery_runs=learning_debt["recent_recovery_runs"],
                execution_learning_recent_after_action_learning_runs=learning_debt["recent_after_action_learning_runs"],
                execution_learning_target_run_id=learning_debt["target_run_id"],
                execution_learning_target_tool_name=learning_debt["target_tool_name"],
                execution_learning_target_after_action_learning_packets=learning_debt["target_after_action_learning_packets"],
                execution_learning_missing=learning_debt["missing"],
                execution_learning_missing_count=learning_debt["missing_count"],
                execution_learning_closure_command=learning_closure_command,
                execution_learning_evidence_command=learning_debt["learning_evidence_command"],
                execution_learning_after_action_learning_command=learning_debt["after_action_learning_command"],
                execution_learning_required_commands=learning_debt["required_commands"],
                execution_learning_next_required_command=learning_debt["next_required_command"],
                execution_learning_proof_queue=learning_debt["proof_queue"],
                execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
                execution_learning_next_proof_command=learning_debt["next_proof_command"],
                execution_learning_actionable_required_commands=learning_debt["actionable_required_commands"],
                execution_learning_actionable_required_command_count=len(learning_debt["actionable_required_commands"]),
                execution_learning_actionable_proof_queue=learning_debt["actionable_required_commands"],
                execution_learning_actionable_proof_queue_count=len(learning_debt["actionable_required_commands"]),
                execution_learning_actionable_next_required_command=learning_debt["actionable_next_required_command"],
                execution_learning_actionable_next_proof_command=learning_debt["actionable_next_proof_command"],
                execution_learning_next_evidence_command=learning_debt["next_evidence_command"],
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                proof_lanes=len(proof_lanes),
                proof_lane_rows=proof_lane_rows,
                strong_lanes=strong_lanes,
                partial_lanes=partial_lanes,
                missing_lanes=missing_lanes,
                proof_requirements=len(proof_requirements),
                proof_requirement_rows=proof_requirement_rows,
                missing_requirement_names=missing_requirement_names,
                missing_requirement_count=len(missing_requirement_names),
                next_proof_commands=next_proof_commands,
                next_proof_command=next_proof_commands[0] if next_proof_commands else "",
                first_pending_approval_id=first_pending_approval_id,
                first_failed_run_id=first_failed_run_id,
                can_trust_execution=can_trust_execution,
                execution_proof_handoff_ready=True,
                execution_proof_handoff=execution_proof_handoff,
            ),
        )

    def execution_mission_control(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("objective"), limit=600)
        if not request:
            output = (
                "Request is required. "
                "Try `execution mission control: organize downloads and summarize what changed`. "
                f"{HARNESS_REQUEST_RECOVERY_ACTION}"
            )
            return ToolResult(
                "execution_mission_control",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(reason="missing_request"),
                    output=output,
                    action=HARNESS_REQUEST_RECOVERY_ACTION,
                ),
            )

        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=10)
        open_tasks = store.list_tasks(status="open", limit=10)
        recent_runs = store.recent_tool_runs(limit=25)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        risks = _risk_signals(request)
        approval_required = bool(risks)
        failed_runs, approval_held_runs = _recent_tool_run_attention_buckets(recent_runs)
        approval_held_review_command = _approval_review_command(approval_held_runs) if approval_held_runs else ""
        verification_runs = [
            row
            for row in recent_runs
            if str(row["tool_name"])
            in {
                "verification_receipt",
                "runtime_trace_receipt",
                "verification_packet",
                "execution_acceptance_gate",
                "execution_proof_bundle",
                "execution_runbook",
                "execution_mission_control",
            }
        ]

        route_packets = [
            name
            for name in [
                "execution_governor_packet",
                "dispatch_decision_packet",
                "planner_gap_packet",
                "execution_contract",
                "argument_contract_packet",
                "execution_readiness_matrix",
            ]
            if name in tool_names
        ]
        proof_packets = [
            name
            for name in [
                "verification_packet",
                "execution_acceptance_gate",
                "execution_proof_bundle",
                "execution_audit_gate",
                "execution_health_report",
                "approval_chain_proof",
            ]
            if name in tool_names
        ]
        recovery_packets = [
            name
            for name in [
                "execution_recovery_packet",
                "execution_health_report",
                "after_action_learning_packet",
                "checkpoint_recovery_preview",
                "checkpoint_recovery_receipt",
                "checkpoint_recovery_followthrough_packet",
                "autonomy_resume_gate",
                "autonomy_continuation_execution_packet",
                "autonomy_step_closure_packet",
                "failure_to_test_preview",
                "failure_promotion_packet",
                "learning_review",
            ]
            if name in tool_names
        ]
        learning_packets = [
            name
            for name in [
                "learning_review",
                "after_action_learning_packet",
                "session_learning_preview",
                "feedback_actions",
                "failure_to_test_preview",
                "failure_promotion_packet",
                "draft_skill_from_session",
            ]
            if name in tool_names
        ]

        blockers = []
        if pending:
            blockers.append(f"{len(pending)} pending approval(s) require readiness and last-look review")
        if failed_runs:
            blockers.append(f"{len(failed_runs)} recent failed/blocked run(s) require recovery review")
        if approval_held_runs:
            blockers.append(f"{len(approval_held_runs)} recent approval-held run(s) require approval review")
        if approval_required:
            blockers.append("risky wording requires approval before any real execution")
        if not route_packets:
            blockers.append("route packet tools are missing")
        if not proof_packets:
            blockers.append("proof packet tools are missing")

        if pending:
            mission_state = "HOLD_FOR_PENDING_APPROVAL_REVIEW"
            go_no_go = "NO_GO"
            next_command = f"approval readiness {pending[0]['id']}"
        elif failed_runs:
            mission_state = "HOLD_FOR_RECOVERY_REVIEW"
            go_no_go = "NO_GO"
            next_command = f"execution recovery packet {failed_runs[0]['id']}"
        elif approval_held_runs:
            mission_state = "HOLD_FOR_APPROVAL_REVIEW"
            go_no_go = "NO_GO"
            next_command = approval_held_review_command
        elif approval_required:
            mission_state = "READY_FOR_LAST_LOOK_PREFLIGHT"
            go_no_go = "PREFLIGHT_ONLY"
            next_command = f"execution governor: {request}"
        else:
            mission_state = "READY_FOR_AUTO_SAFE_PREFLIGHT"
            go_no_go = "PREFLIGHT_OK"
            next_command = f"execution governor: {request}"

        lifecycle = [
            ("perceive", "preserve the exact order text"),
            ("ground", "check memory, tasks, approvals, recent failures, and active goals"),
            ("route", "classify chat/read-only/local-safe/approval-gated work"),
            ("plan", "produce exact tool names and arguments before action"),
            ("gate", "auto-run only read-only or local-safe paths; stop for risky work"),
            ("act", "execute only through ToolRegistry after the gate allows it"),
            ("verify", "attach receipts, acceptance gate, proof bundle, and audit gate"),
            ("recover", "use recovery or failure-to-test packets before retrying"),
            ("learn", "promote repeated misses into tests, preferences, memory, or skills"),
        ]
        governor_command = f"execution governor: {request}"
        cockpit_command = f"command cockpit: {request}"
        dispatch_command = f"dispatch decision: {request}"
        runbook_command = f"execution runbook: {request}"
        proof_bundle_command = f"execution proof bundle: {request}; verification <target>; tests <check>; evidence <receipt>; recovery <stop>; approval <id if risky>"
        acceptance_command = f"acceptance gate: {request}; evidence <receipt>; tests <verification>; recovery <rollback or stop condition>"
        audit_command = "execution audit gate"
        recovery_command = f"execution recovery packet {failed_runs[0]['id']}" if failed_runs else "execution recovery packet"
        learning_target_run_id = learning_debt.get("target_run_id") or (failed_runs[0]["id"] if failed_runs else None)
        learning_command = (
            f"after-action learning packet {learning_target_run_id}"
            if learning_target_run_id
            else "after-action learning packet <run id>"
        )
        learning_closure_command = (
            f"execution learning closure {learning_target_run_id}"
            if learning_debt["blocks_completion_claim"] and learning_target_run_id is not None
            else "execution learning closure"
            if learning_debt["blocks_completion_claim"]
            else ""
        )
        mission_command_queue = [
            next_command,
            cockpit_command,
            governor_command,
            dispatch_command,
            runbook_command,
            proof_bundle_command,
            acceptance_command,
            "execution health report",
            audit_command,
            recovery_command,
        ]
        if approval_held_review_command:
            _append_unique(mission_command_queue, [approval_held_review_command])
        mission_command_queue.append(learning_command)
        if learning_closure_command:
            mission_command_queue.append(learning_closure_command)
        proof_bundle = execution_proof_bundle({"request": request})
        proof_bundle_metadata = dict(proof_bundle.metadata)
        execution_proof_handoff = dict(proof_bundle_metadata.get("execution_proof_handoff") or {})
        execution_proof_aliases = _execution_proof_handoff_aliases(execution_proof_handoff)
        lifecycle_rows = [
            {"stage": name, "detail": detail}
            for name, detail in lifecycle
        ]
        mission_packet_groups = {
            "route": route_packets,
            "proof": proof_packets,
            "recovery": recovery_packets,
            "learning": learning_packets,
        }
        execution_mission_handoff = {
            "source": "execution_mission_control",
            "execution_mission_handoff_ready": True,
            "handoff_ready": True,
            "request": request,
            "mission_state": mission_state,
            "go_no_go": go_no_go,
            "next_command": next_command,
            "risk_signals": risks,
            "risk_signal_count": len(risks),
            "approval_required": approval_required,
            "pending_approvals": len(pending),
            "open_tasks": len(open_tasks),
            "recent_failed_runs": len(failed_runs),
            "recent_approval_held_runs": len(approval_held_runs),
            "approval_held_review_command": approval_held_review_command,
            "recent_verification_runs": len(verification_runs),
            "lifecycle_rows": lifecycle_rows,
            "lifecycle_stages": len(lifecycle_rows),
            "route_packets": route_packets,
            "route_packet_count": len(route_packets),
            "proof_packets": proof_packets,
            "proof_packet_count": len(proof_packets),
            "recovery_packets": recovery_packets,
            "recovery_packet_count": len(recovery_packets),
            "learning_packets": learning_packets,
            "learning_packet_count": len(learning_packets),
            "mission_packet_groups": mission_packet_groups,
            "mission_command_queue": mission_command_queue,
            "mission_command_count": len(mission_command_queue),
            "governor_command": governor_command,
            "cockpit_command": cockpit_command,
            "dispatch_command": dispatch_command,
            "runbook_command": runbook_command,
            "proof_bundle_command": proof_bundle_command,
            "execution_proof_handoff": execution_proof_handoff,
            "execution_proof_handoff_present": bool(execution_proof_handoff),
            "execution_proof_state": proof_bundle_metadata.get("proof_state"),
            "execution_proof_can_trust_execution": proof_bundle_metadata.get("can_trust_execution"),
            "execution_proof_next_command": proof_bundle_metadata.get("next_proof_command"),
            "execution_proof_missing_receipt_count": proof_bundle_metadata.get("missing_receipt_count"),
            "execution_proof_blocker_count": proof_bundle_metadata.get("blocker_count"),
            **execution_proof_aliases,
            "acceptance_command": acceptance_command,
            "audit_command": audit_command,
            "recovery_command": recovery_command,
            "learning_closure_command": learning_closure_command,
            "learning_command": learning_command,
            "execution_learning_state": learning_debt["state"],
            "execution_learning_blocks_completion_claim": learning_debt["blocks_completion_claim"],
            "execution_learning_missing": learning_debt["missing"],
            "execution_learning_missing_count": learning_debt["missing_count"],
            "execution_learning_proof_queue": learning_debt["proof_queue"],
            "execution_learning_proof_queue_count": learning_debt["proof_queue_count"],
            "execution_learning_next_proof_command": learning_debt["next_proof_command"],
            "blocker_details": blockers,
            "blockers": len(blockers),
            "blocker_count": len(blockers),
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "calls_external_service": False,
            "executes_tools": False,
            "reads_personal_data": False,
            "reads_private_data": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "queues_approval": False,
            "requires_approval": False,
            "controls_computer": False,
            "speaks": False,
            "completes_tasks": False,
        }

        lines = [
            "Jarvis execution mission control:",
            "This is a single read-only rehearsal of the whole operational loop. It does not execute, approve, dismiss, write, control the computer, read private data, call external services, speak, complete tasks, or queue approvals.",
            "",
            f"Order: {request}",
            f"Mission state: {mission_state}",
            f"Go/no-go: {go_no_go}",
            f"Single next command: `{next_command}`",
            "",
            "Route board:",
            f"- risk signals: {', '.join(risks) if risks else 'none obvious from wording'}",
            f"- approval required before real execution: {'yes' if approval_required else 'no, unless the selected tool is risk-gated'}",
            f"- pending approvals: {len(pending)}",
            f"- open tasks visible: {len(open_tasks)}",
            f"- recent failed/blocked runs: {len(failed_runs)}",
            f"- recent approval-held runs: {len(approval_held_runs)}",
            f"- approval-held review: `{approval_held_review_command}`" if approval_held_review_command else "- approval-held review: none",
            f"- recent verification packets: {len(verification_runs)}",
            "",
            "Lifecycle rehearsal:",
        ]
        lines.extend(f"- {name}: {detail}" for name, detail in lifecycle)
        lines.extend(
            [
                "",
                "Mission packet sequence:",
                f"- route: {', '.join(route_packets) if route_packets else 'none'}",
                f"- proof: {', '.join(proof_packets) if proof_packets else 'none'}",
                f"- recovery: {', '.join(recovery_packets) if recovery_packets else 'none'}",
                f"- learning: {', '.join(learning_packets) if learning_packets else 'none'}",
                "",
                "Mission command queue:",
            ]
        )
        lines.extend(f"- `{command}`" for command in mission_command_queue)
        lines.extend(
            [
                "",
                "Operator checks:",
                f"- start with `{next_command}`",
                "- respect the operator's explicit stop times, work windows, pause commands, and newer instructions before continuing the mission queue.",
                f"- after blocker review, route the real order through `{governor_command}`",
                f"- use `{cockpit_command}` when the operator needs one compact dashboard before saving or executing the case",
                f"- then `{dispatch_command}` and `{runbook_command}`",
                f"- then `{proof_bundle_command}`",
                "- after any real run, inspect `runtime trace receipt` or `verification receipt <run id>`, then `execution audit gate`; if blocked, run `execution recovery packet` before retrying, capture `after-action learning packet <run id>` evidence, then use `execution learning closure <run id>` as the stop-check before promoting a memory, task, skill, or regression.",
                "",
                "Blockers:",
            ]
        )
        if blockers:
            lines.extend(f"- {blocker}" for blocker in blockers)
        else:
            lines.append("- none found by this read-only mission control packet")
        lines.extend(
            [
                "",
                "Boundary:",
                "- This packet is the steering wheel and dashboard, not the engine. It does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "execution_mission_control",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                mission_state=mission_state,
                go_no_go=go_no_go,
                next_command=next_command,
                risk_signals=risks,
                approval_required=approval_required,
                pending_approvals=len(pending),
                open_tasks=len(open_tasks),
                recent_failed_runs=len(failed_runs),
                recent_approval_held_runs=len(approval_held_runs),
                approval_held_review_command=approval_held_review_command,
                recent_verification_runs=len(verification_runs),
                lifecycle_rows=lifecycle_rows,
                lifecycle_stages=len(lifecycle),
                route_packets=route_packets,
                route_packet_count=len(route_packets),
                proof_packets=proof_packets,
                proof_packet_count=len(proof_packets),
                recovery_packets=recovery_packets,
                recovery_packet_count=len(recovery_packets),
                learning_packets=learning_packets,
                learning_packet_count=len(learning_packets),
                mission_packet_groups=mission_packet_groups,
                mission_command_queue=mission_command_queue,
                mission_command_count=len(mission_command_queue),
                governor_command=governor_command,
                cockpit_command=cockpit_command,
                dispatch_command=dispatch_command,
                runbook_command=runbook_command,
                proof_bundle_command=proof_bundle_command,
                execution_proof_handoff=execution_proof_handoff,
                execution_proof_handoff_present=bool(execution_proof_handoff),
                execution_proof_state=proof_bundle_metadata.get("proof_state"),
                execution_proof_can_trust_execution=proof_bundle_metadata.get("can_trust_execution"),
                execution_proof_next_command=proof_bundle_metadata.get("next_proof_command"),
                execution_proof_missing_receipt_count=proof_bundle_metadata.get("missing_receipt_count"),
                execution_proof_blocker_count=proof_bundle_metadata.get("blocker_count"),
                **execution_proof_aliases,
                acceptance_command=acceptance_command,
                audit_command=audit_command,
                recovery_command=recovery_command,
                learning_closure_command=learning_closure_command,
                learning_command=learning_command,
                execution_learning_state=learning_debt["state"],
                execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
                execution_learning_missing=learning_debt["missing"],
                execution_learning_missing_count=learning_debt["missing_count"],
                execution_learning_required_commands=learning_debt["required_commands"],
                execution_learning_next_required_command=learning_debt["next_required_command"],
                execution_learning_proof_queue=learning_debt["proof_queue"],
                execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
                execution_learning_next_proof_command=learning_debt["next_proof_command"],
                execution_learning_closure_command=learning_closure_command,
                blockers=len(blockers),
                blocker_details=blockers,
                blocker_count=len(blockers),
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                execution_mission_handoff_ready=True,
                execution_mission_handoff=execution_mission_handoff,
            ),
        )

    def execution_case_handoff_packet(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("objective"), limit=600)
        if not request:
            output = (
                "Request is required. "
                "Try `execution case handoff: organize downloads and summarize what changed`. "
                f"{HARNESS_REQUEST_RECOVERY_ACTION}"
            )
            return ToolResult(
                "execution_case_handoff_packet",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(reason="missing_request"),
                    output=output,
                    action=HARNESS_REQUEST_RECOVERY_ACTION,
                ),
            )

        mission = execution_mission_control({"request": request})
        mission_metadata = dict(mission.metadata)
        cockpit_command = f"command cockpit: {request}"
        save_case_command = f"save execution case: {request}"
        inspect_command = "execution case latest"
        gate_command = "execution case gate latest"
        review_command = "execution case review latest"
        timeline_command = "execution case timeline latest"
        mission_queue = list(mission_metadata.get("mission_command_queue") or [])
        execution_proof_handoff = dict(mission_metadata.get("execution_proof_handoff") or {})
        execution_proof_aliases = _execution_proof_handoff_aliases(execution_proof_handoff)
        handoff_queue: list[str] = []
        for command in [
            cockpit_command,
            mission_metadata.get("next_command", ""),
            *mission_queue,
            save_case_command,
            inspect_command,
            gate_command,
            review_command,
            timeline_command,
        ]:
            if command and command not in handoff_queue:
                handoff_queue.append(str(command))

        mission_state = str(mission_metadata.get("mission_state") or "UNKNOWN")
        go_no_go = str(mission_metadata.get("go_no_go") or "UNKNOWN")
        if mission_state.startswith("HOLD_FOR"):
            handoff_verdict = "CASE_HANDOFF_BLOCKED_UNTIL_REVIEW"
            ready_to_save = True
            next_command = str(mission_metadata.get("next_command") or cockpit_command)
        elif go_no_go == "PREFLIGHT_ONLY":
            handoff_verdict = "CASE_HANDOFF_READY_FOR_PREFLIGHT_CASE"
            ready_to_save = True
            next_command = cockpit_command
        else:
            handoff_verdict = "CASE_HANDOFF_READY_FOR_SAFE_CASE"
            ready_to_save = True
            next_command = cockpit_command
        execution_case_handoff = {
            "source": "execution_case_handoff_packet",
            "execution_case_handoff_ready": True,
            "handoff_ready": True,
            "request": request,
            "handoff_verdict": handoff_verdict,
            "mission_state": mission_state,
            "go_no_go": go_no_go,
            "ready_to_save_case": ready_to_save,
            "next_command": next_command,
            "cockpit_command": cockpit_command,
            "save_case_command": save_case_command,
            "inspect_command": inspect_command,
            "gate_command": gate_command,
            "review_command": review_command,
            "timeline_command": timeline_command,
            "mission_command_queue": mission_queue,
            "mission_command_count": len(mission_queue),
            "handoff_proof_queue": handoff_queue,
            "handoff_proof_queue_count": len(handoff_queue),
            "proof_queue": handoff_queue,
            "proof_queue_count": len(handoff_queue),
            "approval_required": mission_metadata.get("approval_required"),
            "risk_signals": mission_metadata.get("risk_signals") or [],
            "risk_signal_count": len(mission_metadata.get("risk_signals") or []),
            "pending_approvals": mission_metadata.get("pending_approvals", 0),
            "blockers": mission_metadata.get("blockers", 0),
            "blocker_details": mission_metadata.get("blocker_details", []),
            "blocker_count": mission_metadata.get("blocker_count", mission_metadata.get("blockers", 0)),
            "mission_handoff": mission_metadata.get("execution_mission_handoff"),
            "mission_handoff_present": bool(mission_metadata.get("execution_mission_handoff")),
            "execution_proof_handoff": execution_proof_handoff,
            "execution_proof_handoff_present": bool(execution_proof_handoff),
            "execution_proof_state": mission_metadata.get("execution_proof_state"),
            "execution_proof_next_command": mission_metadata.get("execution_proof_next_command"),
            "execution_proof_can_trust_execution": mission_metadata.get("execution_proof_can_trust_execution"),
            **execution_proof_aliases,
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "writes_case": False,
            "creates_case": False,
            "calls_model": False,
            "calls_external_service": False,
            "executes_tools": False,
            "reads_personal_data": False,
            "reads_private_data": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "queues_approval": False,
            "requires_approval": False,
            "controls_computer": False,
            "speaks": False,
            "completes_tasks": False,
        }

        lines = [
            "Jarvis execution case handoff packet:",
            "This is the bridge from the command cockpit to a durable execution case. It is read-only and does not create the case, run the order, approve requests, or write files.",
            "",
            f"Order: {request}",
            f"Handoff verdict: {handoff_verdict}",
            f"Mission state: {mission_state}",
            f"Go/no-go: {go_no_go}",
            f"Ready to save case: {'yes' if ready_to_save else 'no'}",
            f"Next safe command: `{next_command}`",
            "",
            "Cockpit-to-case path:",
            f"- cockpit: `{cockpit_command}`",
            f"- mission control: `execution mission control: {request}`",
            f"- save durable case: `{save_case_command}`",
            f"- inspect saved case: `{inspect_command}`",
            f"- gate saved case: `{gate_command}`",
            f"- human review packet: `{review_command}`",
            f"- timeline: `{timeline_command}`",
            "",
            "Mission command queue inherited for the case:",
        ]
        lines.extend(f"- `{command}`" for command in mission_queue)
        lines.extend(
            [
                "",
                "Handoff proof queue:",
                *[f"- `{command}`" for command in handoff_queue],
                "",
                "Case handoff rules:",
                "- Save a case before real execution when the order is risky, multi-step, approval-held, or needs resumable proof.",
                "- A saved case is not completion. Completion still requires evidence events, approval-chain proof when risky, verification receipts, recovery closure, and learning proof.",
                "- the operator's explicit stop times, work windows, pause commands, and newer instructions override the handoff queue.",
                "",
                "Boundary:",
                "- This packet does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, write notes, write the database, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        return ToolResult(
            "execution_case_handoff_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                handoff_verdict=handoff_verdict,
                mission_state=mission_state,
                go_no_go=go_no_go,
                ready_to_save_case=ready_to_save,
                next_command=next_command,
                cockpit_command=cockpit_command,
                save_case_command=save_case_command,
                inspect_command=inspect_command,
                gate_command=gate_command,
                review_command=review_command,
                timeline_command=timeline_command,
                mission_command_queue=mission_queue,
                mission_command_count=len(mission_queue),
                handoff_proof_queue=handoff_queue,
                handoff_proof_queue_count=len(handoff_queue),
                proof_queue=handoff_queue,
                proof_queue_count=len(handoff_queue),
                approval_required=mission_metadata.get("approval_required"),
                risk_signals=mission_metadata.get("risk_signals") or [],
                risk_signal_count=len(mission_metadata.get("risk_signals") or []),
                pending_approvals=mission_metadata.get("pending_approvals", 0),
                blockers=mission_metadata.get("blockers", 0),
                blocker_details=mission_metadata.get("blocker_details", []),
                blocker_count=mission_metadata.get("blocker_count", mission_metadata.get("blockers", 0)),
                mission_handoff=mission_metadata.get("execution_mission_handoff"),
                mission_handoff_present=bool(mission_metadata.get("execution_mission_handoff")),
                execution_proof_handoff=execution_proof_handoff,
                execution_proof_handoff_present=bool(execution_proof_handoff),
                execution_proof_state=mission_metadata.get("execution_proof_state"),
                execution_proof_next_command=mission_metadata.get("execution_proof_next_command"),
                execution_proof_can_trust_execution=mission_metadata.get("execution_proof_can_trust_execution"),
                **execution_proof_aliases,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                execution_case_handoff_ready=True,
                execution_case_handoff=execution_case_handoff,
                writes_case=False,
                creates_case=False,
            ),
        )

    def save_execution_case(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("objective"), limit=600)
        if not request:
            output = (
                "Request is required. "
                "Try `save execution case: organize downloads and summarize what changed`. "
                f"{HARNESS_REQUEST_RECOVERY_ACTION}"
            )
            return ToolResult(
                "save_execution_case",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(
                        reason="missing_request",
                        writes_files=False,
                        writes_database=False,
                        writes_notes=False,
                    ),
                    output=output,
                    action=HARNESS_REQUEST_RECOVERY_ACTION,
                ),
            )

        initial_evidence = _short(
            args.get("initial_evidence") or args.get("evidence") or args.get("verification") or args.get("tests") or "",
            limit=800,
        )
        initial_event_type = _short(args.get("initial_event_type") or args.get("event_type") or "initial_evidence", limit=80)
        initial_receipt_kind = _short(args.get("receipt_kind") or args.get("receipt") or "", limit=80)
        initial_receipt_id = _short(args.get("receipt_id") or args.get("run_id") or args.get("approval_id") or "", limit=80)
        inferred_initial_receipt_kind, inferred_initial_receipt_id = _receipt_reference_from_text(initial_evidence)
        explicit_initial_receipt_kind = _normalized_receipt_kind(initial_receipt_kind)
        inferred_initial_receipt_kind_normalized = _normalized_receipt_kind(inferred_initial_receipt_kind)
        explicit_initial_receipt_id = str(initial_receipt_id or "").strip()
        if (
            inferred_initial_receipt_kind_normalized
            and explicit_initial_receipt_kind
            and explicit_initial_receipt_kind != inferred_initial_receipt_kind_normalized
        ):
            initial_evidence_conflict_reason = "receipt_kind_conflict"
        elif inferred_initial_receipt_id and explicit_initial_receipt_id and explicit_initial_receipt_id != inferred_initial_receipt_id:
            initial_evidence_conflict_reason = "receipt_id_conflict"
        else:
            initial_evidence_conflict_reason = ""

        final_initial_receipt_kind = initial_receipt_kind
        final_initial_receipt_id = initial_receipt_id
        if not final_initial_receipt_kind and inferred_initial_receipt_kind:
            final_initial_receipt_kind = inferred_initial_receipt_kind
        if not final_initial_receipt_id and inferred_initial_receipt_id:
            final_initial_receipt_id = inferred_initial_receipt_id
        normalized_initial_receipt_kind = _normalized_receipt_kind(final_initial_receipt_kind)
        initial_receipt_target_status = "not_referenced"
        initial_receipt_target_exists = None
        initial_receipt_target_issue = ""
        numeric_initial_receipt_id: int | None = None
        if str(final_initial_receipt_id or "").strip().isdigit():
            numeric_initial_receipt_id = int(str(final_initial_receipt_id).strip())

        if normalized_initial_receipt_kind in {"verification_receipt", "tool_run", "execution_recovery_packet"} and numeric_initial_receipt_id is not None:
            initial_receipt_target_exists = store.get_tool_run(numeric_initial_receipt_id) is not None
            initial_receipt_target_status = "found" if initial_receipt_target_exists else "missing"
            if not initial_receipt_target_exists:
                initial_receipt_target_issue = f"referenced tool-run receipt #{numeric_initial_receipt_id} was not found"
        elif normalized_initial_receipt_kind == "runtime_trace_receipt" and numeric_initial_receipt_id is not None:
            initial_receipt_target_exists = not _missing_runtime_trace_message_ids(store, [numeric_initial_receipt_id])
            initial_receipt_target_status = "found" if initial_receipt_target_exists else "missing"
            if not initial_receipt_target_exists:
                initial_receipt_target_issue = f"referenced runtime-trace receipt message #{numeric_initial_receipt_id} was not found"
        elif normalized_initial_receipt_kind == "approval_packet" and numeric_initial_receipt_id is not None:
            initial_receipt_target_exists = _approval_any_status(store, numeric_initial_receipt_id) is not None
            initial_receipt_target_status = "found" if initial_receipt_target_exists else "missing"
            if not initial_receipt_target_exists:
                initial_receipt_target_issue = f"referenced approval #{numeric_initial_receipt_id} was not found"
        elif normalized_initial_receipt_kind and final_initial_receipt_id:
            initial_receipt_target_status = "unverified_kind"
            initial_receipt_target_issue = f"receipt kind `{normalized_initial_receipt_kind}` is not directly target-verified by this save packet"

        initial_evidence_ready_to_append = (
            bool(initial_evidence)
            and not initial_evidence_conflict_reason
            and initial_receipt_target_status != "missing"
        )
        initial_evidence_skip_reason = ""
        if initial_evidence and initial_evidence_conflict_reason:
            initial_evidence_skip_reason = initial_evidence_conflict_reason
        elif initial_evidence and initial_receipt_target_status == "missing":
            initial_evidence_skip_reason = "receipt_target_missing"

        mission = execution_mission_control({"request": request})
        metadata = dict(mission.metadata)
        governor_command = f"execution governor: {request}"
        mission_command_queue = list(metadata.get("mission_command_queue") or [])
        execution_mission_handoff = dict(metadata.get("execution_mission_handoff") or {})
        execution_proof_handoff = dict(metadata.get("execution_proof_handoff") or {})
        execution_proof_aliases = _execution_proof_handoff_aliases(execution_proof_handoff)
        case_id = store.add_execution_case(
            request=request,
            mission_state=str(metadata.get("mission_state") or "UNKNOWN"),
            go_no_go=str(metadata.get("go_no_go") or "UNKNOWN"),
            next_command=str(metadata.get("next_command") or ""),
            risk_signals=list(metadata.get("risk_signals") or []),
            metadata={
                "initial_governor_command": governor_command,
                "initial_cockpit_command": metadata.get("cockpit_command", f"command cockpit: {request}"),
                "mission_command_queue": mission_command_queue,
                "mission_command_count": len(mission_command_queue),
                "governor_command": metadata.get("governor_command", ""),
                "cockpit_command": metadata.get("cockpit_command", ""),
                "dispatch_command": metadata.get("dispatch_command", ""),
                "runbook_command": metadata.get("runbook_command", ""),
                "proof_bundle_command": metadata.get("proof_bundle_command", ""),
                "acceptance_command": metadata.get("acceptance_command", ""),
                "audit_command": metadata.get("audit_command", ""),
                "recovery_command": metadata.get("recovery_command", ""),
                "learning_closure_command": metadata.get("learning_closure_command", ""),
                "learning_command": metadata.get("learning_command", ""),
                "route_packets": metadata.get("route_packets", []),
                "proof_packets": metadata.get("proof_packets", []),
                "recovery_packets": metadata.get("recovery_packets", []),
                "learning_packets": metadata.get("learning_packets", []),
                "execution_mission_handoff": execution_mission_handoff,
                "execution_mission_handoff_present": bool(execution_mission_handoff),
                "execution_proof_handoff": execution_proof_handoff,
                "execution_proof_handoff_present": bool(execution_proof_handoff),
                "execution_proof_state": metadata.get("execution_proof_state"),
                "execution_proof_next_command": metadata.get("execution_proof_next_command"),
                "execution_proof_can_trust_execution": metadata.get("execution_proof_can_trust_execution"),
                **execution_proof_aliases,
            },
        )
        body = "\n".join(
            [
                f"# Execution Case {case_id}",
                "",
                f"Request: {request}",
                f"Mission state: {metadata.get('mission_state')}",
                f"Go/no-go: {metadata.get('go_no_go')}",
                f"Next command: `{metadata.get('next_command')}`",
                f"Initial governor command: `{governor_command}`",
                f"Initial cockpit command: `{metadata.get('cockpit_command', f'command cockpit: {request}')}`",
                "",
                "## Mission Command Queue",
                "",
                *[f"- `{command}`" for command in mission_command_queue],
                "",
                "## Mission Control Packet",
                "",
                mission.output,
            ]
        )
        path = vault.write_execution_case(case_id, request, body)
        store.update_execution_case_note_path(case_id, str(path))

        initial_evidence_event_id = None
        if initial_evidence_ready_to_append:
            initial_evidence_event_id = store.add_execution_case_event(
                case_id=case_id,
                event_type=initial_event_type,
                summary=initial_evidence,
                receipt_kind=final_initial_receipt_kind,
                receipt_id=final_initial_receipt_id,
                metadata={"source": "save_execution_case", "initial_evidence": True},
            )
            stamp = datetime.now().isoformat(timespec="seconds")
            note_lines = [
                "## Evidence Event",
                "",
                f"- event id: {initial_evidence_event_id}",
                f"- created: {stamp}",
                f"- type: {initial_event_type}",
                f"- summary: {initial_evidence}",
            ]
            if final_initial_receipt_kind or final_initial_receipt_id:
                note_lines.append(f"- receipt: {final_initial_receipt_kind or 'receipt'} {final_initial_receipt_id}".rstrip())
            path = vault.append_execution_case_event(str(path), "\n".join(note_lines))
        saved_execution_case_handoff = {
            "source": "save_execution_case",
            "case_id": case_id,
            "request": request,
            "mission_state": metadata.get("mission_state"),
            "go_no_go": metadata.get("go_no_go"),
            "next_command": metadata.get("next_command"),
            "initial_governor_command": governor_command,
            "initial_cockpit_command": metadata.get("cockpit_command", f"command cockpit: {request}"),
            "mission_command_queue": mission_command_queue,
            "mission_command_count": len(mission_command_queue),
            "execution_mission_handoff": execution_mission_handoff,
            "execution_mission_handoff_present": bool(execution_mission_handoff),
            "execution_proof_handoff": execution_proof_handoff,
            "execution_proof_handoff_present": bool(execution_proof_handoff),
            "execution_proof_state": metadata.get("execution_proof_state"),
            "execution_proof_next_command": metadata.get("execution_proof_next_command"),
            "execution_proof_can_trust_execution": metadata.get("execution_proof_can_trust_execution"),
            **execution_proof_aliases,
            "proof_bundle_command": metadata.get("proof_bundle_command", ""),
            "acceptance_command": metadata.get("acceptance_command", ""),
            "audit_command": metadata.get("audit_command", ""),
            "recovery_command": metadata.get("recovery_command", ""),
            "learning_closure_command": metadata.get("learning_closure_command", ""),
            "learning_command": metadata.get("learning_command", ""),
            "initial_evidence_supplied": bool(initial_evidence),
            "initial_evidence_attached": bool(initial_evidence_event_id),
            "initial_evidence_ready_to_append": initial_evidence_ready_to_append,
            "initial_evidence_event_id": initial_evidence_event_id,
            "initial_evidence_event_type": initial_event_type,
            "initial_evidence_receipt_kind": final_initial_receipt_kind,
            "initial_evidence_receipt_kind_normalized": normalized_initial_receipt_kind,
            "initial_evidence_receipt_id": final_initial_receipt_id,
            "initial_evidence_receipt_target_status": initial_receipt_target_status,
            "initial_evidence_receipt_target_exists": initial_receipt_target_exists,
            "initial_evidence_receipt_target_issue": initial_receipt_target_issue,
            "initial_evidence_conflict_reason": initial_evidence_conflict_reason,
            "initial_evidence_skip_reason": initial_evidence_skip_reason,
            "evidence_events_created": 1 if initial_evidence_event_id else 0,
            "approval_required": metadata.get("approval_required"),
            "risk_signals": metadata.get("risk_signals") or [],
            "risk_signal_count": len(metadata.get("risk_signals") or []),
            "pending_approvals": metadata.get("pending_approvals", 0),
            "blockers": metadata.get("blockers", 0),
            "note_path": str(path),
            "local_case_creation_only": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "writes_case": True,
            "creates_case": True,
            "writes_files": True,
            "writes_database": True,
            "writes_notes": True,
            "calls_model": False,
            "calls_external_service": False,
            "executes_tools": False,
            "reads_personal_data": False,
            "reads_private_data": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "queues_approval": False,
            "requires_approval": False,
            "controls_computer": False,
            "speaks": False,
            "completes_tasks": False,
        }

        lines = [
            f"Execution case #{case_id} saved.",
            f"Path: {path}",
            "",
            f"Mission state: {metadata.get('mission_state')}",
            f"Go/no-go: {metadata.get('go_no_go')}",
            f"Next command: `{metadata.get('next_command')}`",
            f"Initial governor command: `{governor_command}`",
            f"Initial cockpit command: `{metadata.get('cockpit_command', f'command cockpit: {request}')}`",
            "",
            "Mission command queue:",
            *[f"- `{command}`" for command in mission_command_queue],
            "",
            "Initial evidence:",
            f"- supplied: {'yes' if initial_evidence else 'no'}",
            f"- attached: {'yes' if initial_evidence_event_id else 'no'}",
            f"- event id: {initial_evidence_event_id}" if initial_evidence_event_id else f"- skipped reason: {initial_evidence_skip_reason or 'none'}",
            f"- receipt target status: {initial_receipt_target_status}",
            "",
            "Boundary:",
            "- This creates a Jarvis-owned case file and database row only. It does not execute tools, approve requests, dismiss approvals, read private data, control the computer, call external services, speak, complete tasks, or queue approvals.",
        ]
        return ToolResult(
            "save_execution_case",
            True,
            "\n".join(lines),
            _safe_metadata(
                case_id=case_id,
                request=request,
                mission_state=metadata.get("mission_state"),
                go_no_go=metadata.get("go_no_go"),
                next_command=metadata.get("next_command"),
                initial_governor_command=governor_command,
                initial_cockpit_command=metadata.get("cockpit_command", f"command cockpit: {request}"),
                mission_command_queue=mission_command_queue,
                mission_command_count=len(mission_command_queue),
                execution_mission_handoff=execution_mission_handoff,
                execution_mission_handoff_present=bool(execution_mission_handoff),
                execution_proof_handoff=execution_proof_handoff,
                execution_proof_handoff_present=bool(execution_proof_handoff),
                execution_proof_state=metadata.get("execution_proof_state"),
                execution_proof_next_command=metadata.get("execution_proof_next_command"),
                execution_proof_can_trust_execution=metadata.get("execution_proof_can_trust_execution"),
                **execution_proof_aliases,
                proof_bundle_command=metadata.get("proof_bundle_command", ""),
                acceptance_command=metadata.get("acceptance_command", ""),
                audit_command=metadata.get("audit_command", ""),
                recovery_command=metadata.get("recovery_command", ""),
                learning_closure_command=metadata.get("learning_closure_command", ""),
                learning_command=metadata.get("learning_command", ""),
                initial_evidence_supplied=bool(initial_evidence),
                initial_evidence_attached=bool(initial_evidence_event_id),
                initial_evidence_ready_to_append=initial_evidence_ready_to_append,
                initial_evidence_event_id=initial_evidence_event_id,
                initial_evidence_event_type=initial_event_type,
                initial_evidence_summary=initial_evidence,
                initial_evidence_receipt_kind=final_initial_receipt_kind,
                initial_evidence_receipt_kind_normalized=normalized_initial_receipt_kind,
                initial_evidence_receipt_id=final_initial_receipt_id,
                initial_evidence_receipt_target_status=initial_receipt_target_status,
                initial_evidence_receipt_target_exists=initial_receipt_target_exists,
                initial_evidence_receipt_target_issue=initial_receipt_target_issue,
                initial_evidence_conflict_reason=initial_evidence_conflict_reason,
                initial_evidence_skip_reason=initial_evidence_skip_reason,
                evidence_events_created=1 if initial_evidence_event_id else 0,
                approval_required=metadata.get("approval_required"),
                risk_signals=metadata.get("risk_signals") or [],
                risk_signal_count=len(metadata.get("risk_signals") or []),
                pending_approvals=metadata.get("pending_approvals", 0),
                blockers=metadata.get("blockers", 0),
                note_path=str(path),
                local_case_creation_only=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                saved_execution_case_handoff=saved_execution_case_handoff,
                writes_case=True,
                creates_case=True,
                writes_files=True,
                writes_database=True,
                writes_notes=True,
            ),
        )

    def inspect_execution_case(args: dict[str, Any]) -> ToolResult:
        case, case_error = _resolve_execution_case_arg("inspect_execution_case", args)
        if case_error is not None:
            return case_error

        if case is None:
            inspect_execution_case_handoff = _safe_metadata(
                source="inspect_execution_case",
                found=False,
                case_id=None,
                request="",
                mission_state="NO_CASE",
                go_no_go="UNKNOWN",
                next_command="",
                risk_signals=[],
                risk_signal_count=0,
                note_path="",
                events=0,
                mission_command_queue=[],
                mission_command_count=0,
                execution_mission_handoff={},
                execution_mission_handoff_present=False,
                execution_proof_handoff={},
                execution_proof_handoff_present=False,
                execution_proof_state="unknown",
                execution_proof_next_command="",
                execution_proof_can_trust_execution=False,
                initial_governor_command="",
                proof_bundle_command="",
                acceptance_command="",
                audit_command="",
                recovery_command="",
                learning_command="",
                execution_health_recovery_closure_state="unknown",
                execution_health_recovery_closure_ready_to_retry=False,
                execution_health_recovery_closure_missing=[],
                execution_health_recovery_closure_missing_count=0,
                execution_health_recovery_closure_proof_queue=[],
                execution_health_recovery_closure_proof_queue_count=0,
                execution_health_recovery_closure_next_required_command="",
                execution_health_recovery_closure_next_proof_command="",
                execution_health_recovery_closure_blocks_completion_claim=False,
                execution_learning_state="unknown",
                execution_learning_blocks_completion_claim=False,
                execution_learning_missing=[],
                execution_learning_missing_count=0,
                execution_learning_proof_queue=[],
                execution_learning_proof_queue_count=0,
                execution_learning_next_required_command="",
                execution_learning_next_proof_command="",
                evidence_preview_gate_count=0,
                evidence_preview_ready_count=0,
                evidence_preview_blocked_count=0,
                evidence_preview_verdicts=[],
                evidence_preview_latest_verdict="",
                evidence_preview_latest_event_id=None,
                evidence_preview_latest_handoff={},
                evidence_preview_latest_handoff_present=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
            )
            return ToolResult(
                "inspect_execution_case",
                True,
                "No execution case file has been saved yet. Try `save execution case: organize downloads and summarize what changed`.",
                _safe_metadata(
                    found=False,
                    case_id=None,
                    cases=0,
                    request="",
                    mission_state="NO_CASE",
                    go_no_go="UNKNOWN",
                    next_command="",
                    events=0,
                    mission_command_queue=[],
                    mission_command_count=0,
                    evidence_preview_gate_count=0,
                    evidence_preview_ready_count=0,
                    evidence_preview_blocked_count=0,
                    evidence_preview_verdicts=[],
                    evidence_preview_latest_verdict="",
                    evidence_preview_latest_event_id=None,
                    evidence_preview_latest_handoff={},
                    evidence_preview_latest_handoff_present=False,
                    draft_only=True,
                    requires_manual_send=True,
                    loads_without_execution=True,
                    authorizes_execution=False,
                    authorizes_completion_claim=False,
                    approval_granted=False,
                    inspect_execution_case_handoff=inspect_execution_case_handoff,
                ),
            )

        try:
            risk_signals = list(json.loads(case["risk_signals"] or "[]"))
        except Exception:
            risk_signals = []
        case_metadata = _execution_case_metadata(case)
        mission_command_queue = list(case_metadata.get("mission_command_queue") or [])
        execution_mission_handoff = dict(case_metadata.get("execution_mission_handoff") or {})
        execution_proof_handoff = dict(case_metadata.get("execution_proof_handoff") or {})
        execution_proof_aliases = _execution_proof_handoff_aliases(execution_proof_handoff)
        events = store.list_execution_case_events(int(case["id"]), limit=8)
        recent_runs = store.recent_tool_runs(limit=50)
        recovery_closure = _execution_health_recovery_closure_snapshot(store, recent_runs)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        gate_result = execution_case_gate({"case_id": int(case["id"])})
        gate_metadata = dict(gate_result.metadata) if gate_result.ok else {}
        evidence_preview_latest_handoff = gate_metadata.get("evidence_preview_latest_handoff")
        evidence_preview_latest_handoff = (
            evidence_preview_latest_handoff if isinstance(evidence_preview_latest_handoff, dict) else {}
        )
        inspect_execution_case_handoff = {
            "source": "inspect_execution_case",
            "found": True,
            "case_id": int(case["id"]),
            "request": case["request"],
            "mission_state": case["mission_state"],
            "go_no_go": case["go_no_go"],
            "next_command": case["next_command"],
            "risk_signals": risk_signals,
            "risk_signal_count": len(risk_signals),
            "note_path": case["note_path"],
            "events": len(events),
            "mission_command_queue": mission_command_queue,
            "mission_command_count": len(mission_command_queue),
            "execution_mission_handoff": execution_mission_handoff,
            "execution_mission_handoff_present": bool(execution_mission_handoff),
            "execution_proof_handoff": execution_proof_handoff,
            "execution_proof_handoff_present": bool(execution_proof_handoff),
            "execution_proof_state": case_metadata.get("execution_proof_state"),
            "execution_proof_next_command": case_metadata.get("execution_proof_next_command"),
            "execution_proof_can_trust_execution": case_metadata.get("execution_proof_can_trust_execution"),
            **execution_proof_aliases,
            "initial_governor_command": case_metadata.get("initial_governor_command", ""),
            "proof_bundle_command": case_metadata.get("proof_bundle_command", ""),
            "acceptance_command": case_metadata.get("acceptance_command", ""),
            "audit_command": case_metadata.get("audit_command", ""),
            "recovery_command": case_metadata.get("recovery_command", ""),
            "learning_command": case_metadata.get("learning_command", ""),
            "execution_health_recovery_closure_state": recovery_closure["state"],
            "execution_health_recovery_closure_ready_to_retry": recovery_closure["ready_to_retry"],
            "execution_health_recovery_closure_missing": recovery_closure["missing"],
            "execution_health_recovery_closure_missing_count": recovery_closure["missing_count"],
            "execution_health_recovery_closure_proof_queue": recovery_closure["proof_queue"],
            "execution_health_recovery_closure_proof_queue_count": recovery_closure["proof_queue_count"],
            "execution_health_recovery_closure_next_proof_command": recovery_closure["next_proof_command"],
            "execution_health_recovery_closure_blocks_completion_claim": recovery_closure["blocks_completion_claim"],
            "execution_learning_state": learning_debt["state"],
            "execution_learning_blocks_completion_claim": learning_debt["blocks_completion_claim"],
            "execution_learning_missing": learning_debt["missing"],
            "execution_learning_missing_count": learning_debt["missing_count"],
            "execution_learning_proof_queue": learning_debt["proof_queue"],
            "execution_learning_proof_queue_count": learning_debt["proof_queue_count"],
            "execution_learning_next_proof_command": learning_debt["next_proof_command"],
            "evidence_preview_gate_count": _metadata_int(gate_metadata.get("evidence_preview_gate_count")),
            "evidence_preview_ready_count": _metadata_int(gate_metadata.get("evidence_preview_ready_count")),
            "evidence_preview_blocked_count": _metadata_int(gate_metadata.get("evidence_preview_blocked_count")),
            "evidence_preview_verdicts": list(gate_metadata.get("evidence_preview_verdicts") or []),
            "evidence_preview_latest_verdict": str(gate_metadata.get("evidence_preview_latest_verdict") or ""),
            "evidence_preview_latest_event_id": gate_metadata.get("evidence_preview_latest_event_id"),
            "evidence_preview_latest_handoff": evidence_preview_latest_handoff,
            "evidence_preview_latest_handoff_present": _metadata_bool(gate_metadata.get("evidence_preview_latest_handoff_present")),
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "calls_model": False,
            "calls_external_service": False,
            "executes_tools": False,
            "reads_personal_data": False,
            "reads_private_data": False,
            "executes_side_effect": False,
            "external_side_effect": False,
            "writes_files": False,
            "writes_database": False,
            "writes_memory": False,
            "writes_notes": False,
            "queues_approval": False,
            "requires_approval": False,
            "controls_computer": False,
            "speaks": False,
            "completes_tasks": False,
        }
        lines = [
            f"Execution case #{case['id']}:",
            f"- request: {case['request']}",
            f"- mission state: {case['mission_state']}",
            f"- go/no-go: {case['go_no_go']}",
            f"- next command: `{case['next_command']}`",
            f"- risk signals: {', '.join(risk_signals) if risk_signals else 'none'}",
            f"- note path: {case['note_path'] or '(not written)'}",
            f"- created: {case['created_at']}",
            f"- updated: {case['updated_at']}",
            "",
            "Evidence log:",
        ]
        if events:
            for event in events:
                receipt = ""
                if event["receipt_kind"] or event["receipt_id"]:
                    receipt = f" [{event['receipt_kind']} {event['receipt_id']}]".strip()
                lines.append(f"- #{event['id']} {event['event_type']}{receipt}: {event['summary']}")
        else:
            lines.append("- none attached yet")
        lines.extend(["", "Mission command queue:"])
        if mission_command_queue:
            lines.extend(f"- `{command}`" for command in mission_command_queue)
        else:
            lines.append("- not stored for this case")
        lines.extend(
            [
                "",
                "Recovery closure debt:",
                f"- state: {recovery_closure['state']}",
                f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
                f"- next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required: none",
                "",
                "Execution learning debt:",
                f"- state: {learning_debt['state']}",
                f"- missing: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
                f"- next required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- next required: none",
                "",
                "Evidence preflight trace:",
                f"- events: {_metadata_int(gate_metadata.get('evidence_preview_gate_count'))}",
                f"- ready: {_metadata_int(gate_metadata.get('evidence_preview_ready_count'))}",
                f"- blocked: {_metadata_int(gate_metadata.get('evidence_preview_blocked_count'))}",
                f"- latest verdict: {str(gate_metadata.get('evidence_preview_latest_verdict') or 'none')}",
            ]
        )
        lines.extend(
            [
                "",
                "Use this case as the stable order/proof anchor before real execution. Risky steps still require approval readiness, last-look approval packets, approval chain proof, and linked verification receipts.",
                "",
                "Boundary:",
                "- This inspection is read-only. It does not execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        return ToolResult(
            "inspect_execution_case",
            True,
            "\n".join(lines),
            _safe_metadata(
                found=True,
                case_id=int(case["id"]),
                request=case["request"],
                mission_state=case["mission_state"],
                go_no_go=case["go_no_go"],
                next_command=case["next_command"],
                risk_signals=risk_signals,
                risk_signal_count=len(risk_signals),
                note_path=case["note_path"],
                events=len(events),
                mission_command_queue=mission_command_queue,
                mission_command_count=len(mission_command_queue),
                execution_mission_handoff=execution_mission_handoff,
                execution_mission_handoff_present=bool(execution_mission_handoff),
                execution_proof_handoff=execution_proof_handoff,
                execution_proof_handoff_present=bool(execution_proof_handoff),
                execution_proof_state=case_metadata.get("execution_proof_state"),
                execution_proof_next_command=case_metadata.get("execution_proof_next_command"),
                execution_proof_can_trust_execution=case_metadata.get("execution_proof_can_trust_execution"),
                **execution_proof_aliases,
                initial_governor_command=case_metadata.get("initial_governor_command", ""),
                proof_bundle_command=case_metadata.get("proof_bundle_command", ""),
                acceptance_command=case_metadata.get("acceptance_command", ""),
                audit_command=case_metadata.get("audit_command", ""),
                recovery_command=case_metadata.get("recovery_command", ""),
                learning_command=case_metadata.get("learning_command", ""),
                execution_health_recovery_closure_state=recovery_closure["state"],
                execution_health_recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                execution_health_recovery_closure_missing=recovery_closure["missing"],
                execution_health_recovery_closure_missing_count=recovery_closure["missing_count"],
                execution_health_recovery_closure_required_commands=recovery_closure["required_commands"],
                execution_health_recovery_closure_next_required_command=recovery_closure["next_required_command"],
                execution_health_recovery_closure_checklist_command=_recovery_closure_checklist_command(recovery_closure),
                execution_health_recovery_closure_should_open_checklist=bool(_recovery_closure_checklist_command(recovery_closure)),
                execution_health_recovery_closure_proof_queue=recovery_closure["proof_queue"],
                execution_health_recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
                execution_health_recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
                execution_health_recovery_closure_blocks_completion_claim=recovery_closure["blocks_completion_claim"],
                execution_health_recovery_closure_target_run_id=recovery_closure["target_run_id"],
                execution_health_recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                execution_health_recovery_closure_target_verification_receipts=recovery_closure["target_verification_receipts"],
                execution_health_recovery_closure_target_recovery_packets=recovery_closure["target_recovery_packets"],
                execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure["target_after_action_learning_packets"],
                execution_learning_state=learning_debt["state"],
                execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
                execution_learning_missing=learning_debt["missing"],
                execution_learning_missing_count=learning_debt["missing_count"],
                execution_learning_required_commands=learning_debt["required_commands"],
                execution_learning_next_required_command=learning_debt["next_required_command"],
                execution_learning_proof_queue=learning_debt["proof_queue"],
                execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
                execution_learning_next_proof_command=learning_debt["next_proof_command"],
                execution_learning_recent_action_runs=learning_debt["recent_action_runs"],
                execution_learning_failed_or_blocked_action_runs=learning_debt["failed_or_blocked_action_runs"],
                execution_learning_recent_after_action_learning_runs=learning_debt["recent_after_action_learning_runs"],
                execution_learning_target_run_id=learning_debt["target_run_id"],
                execution_learning_target_tool_name=learning_debt["target_tool_name"],
                execution_learning_target_after_action_learning_packets=learning_debt["target_after_action_learning_packets"],
                evidence_preview_gate_count=_metadata_int(gate_metadata.get("evidence_preview_gate_count")),
                evidence_preview_ready_count=_metadata_int(gate_metadata.get("evidence_preview_ready_count")),
                evidence_preview_blocked_count=_metadata_int(gate_metadata.get("evidence_preview_blocked_count")),
                evidence_preview_verdicts=list(gate_metadata.get("evidence_preview_verdicts") or []),
                evidence_preview_latest_verdict=str(gate_metadata.get("evidence_preview_latest_verdict") or ""),
                evidence_preview_latest_event_id=gate_metadata.get("evidence_preview_latest_event_id"),
                evidence_preview_latest_handoff=evidence_preview_latest_handoff,
                evidence_preview_latest_handoff_present=_metadata_bool(gate_metadata.get("evidence_preview_latest_handoff_present")),
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                inspect_execution_case_handoff=inspect_execution_case_handoff,
            ),
        )

    def execution_case_evidence_packet(args: dict[str, Any]) -> ToolResult:
        case, case_error = _resolve_execution_case_arg("execution_case_evidence_packet", args)
        if case_error is not None:
            return case_error
        if case is None:
            evidence_defaults = _execution_case_evidence_packet_defaults()
            evidence_handoff = _safe_metadata(source="execution_case_evidence_packet", **evidence_defaults)
            return ToolResult(
                "execution_case_evidence_packet",
                True,
                "Jarvis execution case evidence packet:\nNo saved execution case found. Save one first with `save execution case: <order>`.",
                _safe_metadata(**evidence_defaults, execution_case_evidence_handoff=evidence_handoff),
            )

        event_type = _short(args.get("event_type") or args.get("type") or "evidence", limit=80)
        summary = _short(args.get("summary") or args.get("evidence") or args.get("body") or args.get("note"), limit=800)
        receipt_kind = _short(args.get("receipt_kind") or args.get("receipt") or "", limit=80)
        receipt_id = _short(args.get("receipt_id") or args.get("run_id") or args.get("approval_id") or "", limit=80)
        if not summary:
            evidence_defaults = _execution_case_evidence_packet_defaults(
                found=True,
                reason="missing_summary",
                case_id=int(case["id"]),
                request=case["request"],
                verdict="EVIDENCE_SUMMARY_REQUIRED",
                event_type=event_type,
                receipt_target_status="not_checked",
                next_command=f"case evidence packet {int(case['id'])}: <evidence summary>",
            )
            evidence_handoff = _safe_metadata(source="execution_case_evidence_packet", **evidence_defaults)
            return ToolResult(
                "execution_case_evidence_packet",
                True,
                "Jarvis execution case evidence packet:\nEvidence summary is required. Try `case evidence packet latest: verification receipt 12 confirmed the output`.",
                _safe_metadata(**evidence_defaults, execution_case_evidence_handoff=evidence_handoff),
            )
        inferred_receipt_kind, inferred_receipt_id = _receipt_reference_from_text(summary)
        explicit_receipt_kind = _normalized_receipt_kind(receipt_kind)
        inferred_receipt_kind_normalized = _normalized_receipt_kind(inferred_receipt_kind)
        explicit_receipt_id = str(receipt_id or "").strip()
        if (
            inferred_receipt_kind_normalized
            and explicit_receipt_kind
            and explicit_receipt_kind != inferred_receipt_kind_normalized
        ):
            conflict_reason = "receipt_kind_conflict"
        elif inferred_receipt_id and explicit_receipt_id and explicit_receipt_id != inferred_receipt_id:
            conflict_reason = "receipt_id_conflict"
        else:
            conflict_reason = ""

        final_receipt_kind = receipt_kind
        final_receipt_id = receipt_id
        if not final_receipt_kind and inferred_receipt_kind:
            final_receipt_kind = inferred_receipt_kind
        if not final_receipt_id and inferred_receipt_id:
            final_receipt_id = inferred_receipt_id
        normalized_final_kind = _normalized_receipt_kind(final_receipt_kind)
        target_status = "not_referenced"
        target_exists = None
        target_lookup_command = ""
        target_issue = ""
        numeric_receipt_id: int | None = None
        if str(final_receipt_id or "").strip().isdigit():
            numeric_receipt_id = int(str(final_receipt_id).strip())

        if normalized_final_kind in {"verification_receipt", "tool_run", "execution_recovery_packet"} and numeric_receipt_id is not None:
            target_exists = store.get_tool_run(numeric_receipt_id) is not None
            target_status = "found" if target_exists else "missing"
            target_lookup_command = f"verification receipt {numeric_receipt_id}" if normalized_final_kind != "execution_recovery_packet" else f"execution recovery packet {numeric_receipt_id}"
            if not target_exists:
                target_issue = f"referenced tool-run receipt #{numeric_receipt_id} was not found"
        elif normalized_final_kind == "runtime_trace_receipt" and numeric_receipt_id is not None:
            target_exists = not _missing_runtime_trace_message_ids(store, [numeric_receipt_id])
            target_status = "found" if target_exists else "missing"
            target_lookup_command = f"runtime trace receipt {numeric_receipt_id}"
            if not target_exists:
                target_issue = f"referenced runtime-trace receipt message #{numeric_receipt_id} was not found"
        elif normalized_final_kind == "approval_packet" and numeric_receipt_id is not None:
            target_exists = _approval_any_status(store, numeric_receipt_id) is not None
            target_status = "found" if target_exists else "missing"
            target_lookup_command = f"approval packet {numeric_receipt_id}"
            if not target_exists:
                target_issue = f"referenced approval #{numeric_receipt_id} was not found"
        elif normalized_final_kind and final_receipt_id:
            target_status = "unverified_kind"
            target_issue = f"receipt kind `{normalized_final_kind}` is not directly target-verified by this packet"

        ready_to_append = not conflict_reason and bool(summary) and target_status != "missing"
        if conflict_reason:
            verdict = "EVIDENCE_CONFLICT"
        elif target_status == "missing":
            verdict = "EVIDENCE_RECEIPT_TARGET_MISSING"
        elif normalized_final_kind and final_receipt_id:
            verdict = "EVIDENCE_READY_WITH_RECEIPT"
        else:
            verdict = "EVIDENCE_READY_WITHOUT_RECEIPT"

        append_command = f"case evidence {case['id']}: {summary}"
        if normalized_final_kind and final_receipt_id:
            append_command = f"case evidence {case['id']}: {summary}"
        execution_case_evidence_handoff = _safe_metadata(
            source="execution_case_evidence_packet",
            found=True,
            case_id=int(case["id"]),
            request=case["request"],
            verdict=verdict,
            ready_to_append=ready_to_append,
            event_type=event_type,
            summary=summary,
            explicit_receipt_kind=explicit_receipt_kind,
            explicit_receipt_id=explicit_receipt_id,
            inferred_receipt_kind=inferred_receipt_kind_normalized,
            inferred_receipt_id=inferred_receipt_id,
            receipt_kind=final_receipt_kind,
            receipt_kind_normalized=normalized_final_kind,
            receipt_id=final_receipt_id,
            receipt_target_status=target_status,
            receipt_target_exists=target_exists,
            receipt_target_issue=target_issue,
            target_lookup_command=target_lookup_command,
            conflict_reason=conflict_reason,
            append_command=append_command if ready_to_append else "",
            next_command=append_command if ready_to_append else target_lookup_command,
            draft_only=True,
            requires_manual_send=True,
            loads_without_execution=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
            writes_case=False,
            writes_case_event=False,
            writes_evidence=False,
        )
        lines = [
            "Jarvis execution case evidence packet:",
            "This is the read-only last-look packet before attaching evidence to a saved execution case. It does not write the case file or database.",
            "",
            f"Case: #{case['id']}",
            f"Request: {case['request']}",
            f"Verdict: {verdict}",
            f"Ready to append: {'yes' if ready_to_append else 'no'}",
            "",
            "Evidence draft:",
            f"- type: {event_type}",
            f"- summary: {summary}",
            f"- explicit receipt: {explicit_receipt_kind or 'none'} {explicit_receipt_id}".rstrip(),
            f"- inferred receipt: {inferred_receipt_kind_normalized or 'none'} {inferred_receipt_id}".rstrip(),
            f"- final receipt: {normalized_final_kind or 'none'} {final_receipt_id}".rstrip(),
            "",
            "Receipt target check:",
            f"- status: {target_status}",
            f"- target exists: {'yes' if target_exists else 'no' if target_exists is False else 'not checked'}",
            f"- lookup command: `{target_lookup_command}`" if target_lookup_command else "- lookup command: none",
        ]
        if conflict_reason:
            lines.extend(
                [
                    "",
                    "Conflict:",
                    f"- reason: {conflict_reason}",
                    f"- summary references `{inferred_receipt_kind_normalized or 'receipt'} {inferred_receipt_id or '(no id)'}`",
                    f"- explicit metadata says `{explicit_receipt_kind or 'receipt'} {explicit_receipt_id or '(no id)'}`",
                ]
            )
        if target_issue:
            lines.extend(["", "Target issue:", f"- {target_issue}"])
        lines.extend(
            [
                "",
                "Next command:",
                f"- `{append_command}`" if ready_to_append else "- fix the evidence conflict or missing receipt target before appending",
                "",
                "Rules:",
                "- Evidence should point to observed output, verification receipts, approval-chain proof, recovery proof, or learning proof, not only intent.",
                "- Risky cases still need approval-chain proof and linked verification before completion can be trusted.",
                "- A saved evidence event is not a completion claim.",
                "",
                "Boundary:",
                "- This packet does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, write notes, write the database, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        return ToolResult(
            "execution_case_evidence_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                found=True,
                case_id=int(case["id"]),
                request=case["request"],
                verdict=verdict,
                ready_to_append=ready_to_append,
                event_type=event_type,
                summary=summary,
                explicit_receipt_kind=explicit_receipt_kind,
                explicit_receipt_id=explicit_receipt_id,
                inferred_receipt_kind=inferred_receipt_kind_normalized,
                inferred_receipt_id=inferred_receipt_id,
                receipt_kind=final_receipt_kind,
                receipt_kind_normalized=normalized_final_kind,
                receipt_id=final_receipt_id,
                receipt_target_status=target_status,
                receipt_target_exists=target_exists,
                receipt_target_issue=target_issue,
                target_lookup_command=target_lookup_command,
                conflict_reason=conflict_reason,
                append_command=append_command if ready_to_append else "",
                next_command=append_command if ready_to_append else target_lookup_command,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                execution_case_evidence_handoff=execution_case_evidence_handoff,
            ),
        )

    def append_execution_case_evidence(args: dict[str, Any]) -> ToolResult:
        case, case_error = _resolve_execution_case_arg("append_execution_case_evidence", args)
        if case_error is not None:
            return case_error
        if case is None:
            output = (
                "No execution case file exists yet. "
                "Try `save execution case: organize downloads and summarize what changed` first. "
                f"{HARNESS_CASE_RECOVERY_ACTION}"
            )
            return ToolResult(
                "append_execution_case_evidence",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(
                        reason="missing_case",
                        case_id=None,
                        local_evidence_append_only=False,
                        authorizes_execution=False,
                        authorizes_completion_claim=False,
                        approval_granted=False,
                        writes_case=False,
                        writes_case_event=False,
                        writes_evidence=False,
                        writes_files=False,
                        writes_database=False,
                        writes_notes=False,
                    ),
                    output=output,
                    action=HARNESS_CASE_RECOVERY_ACTION,
                ),
            )

        event_type = _short(args.get("event_type") or args.get("type") or "evidence", limit=80)
        summary = _short(args.get("summary") or args.get("evidence") or args.get("body") or args.get("note"), limit=800)
        receipt_kind = _short(args.get("receipt_kind") or args.get("receipt") or "", limit=80)
        receipt_id = _short(args.get("receipt_id") or args.get("run_id") or args.get("approval_id") or "", limit=80)
        if not summary:
            output = (
                "Evidence summary is required. "
                "Try `case evidence latest: verification receipt 12 confirmed the output`. "
                f"{HARNESS_EVIDENCE_RECOVERY_ACTION}"
            )
            return ToolResult(
                "append_execution_case_evidence",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(
                        reason="missing_summary",
                        case_id=int(case["id"]),
                        local_evidence_append_only=False,
                        authorizes_execution=False,
                        authorizes_completion_claim=False,
                        approval_granted=False,
                        writes_case=False,
                        writes_case_event=False,
                        writes_evidence=False,
                        writes_files=False,
                        writes_database=False,
                        writes_notes=False,
                    ),
                    output=output,
                    action=HARNESS_EVIDENCE_RECOVERY_ACTION,
                ),
            )
        evidence_preview = execution_case_evidence_packet(
            {
                "case_id": int(case["id"]),
                "event_type": event_type,
                "summary": summary,
                "receipt_kind": receipt_kind,
                "receipt_id": receipt_id,
            }
        )
        preview_metadata = dict(evidence_preview.metadata)
        evidence_handoff = dict(preview_metadata.get("execution_case_evidence_handoff") or {})
        conflict_reason = str(preview_metadata.get("conflict_reason") or "")
        if conflict_reason == "receipt_kind_conflict":
            output = (
                "Execution case evidence receipt conflict. "
                f"Summary references `{preview_metadata.get('inferred_receipt_kind') or 'receipt'} {preview_metadata.get('inferred_receipt_id') or '(no id)'}`, "
                f"but explicit receipt metadata says `{preview_metadata.get('explicit_receipt_kind') or 'receipt'} {preview_metadata.get('explicit_receipt_id') or '(no id)'}`. "
                "Rewrite the evidence with one matching receipt before saving. "
                f"{HARNESS_EVIDENCE_RECOVERY_ACTION}"
            )
            return ToolResult(
                "append_execution_case_evidence",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(
                        reason="receipt_conflict",
                        case_id=int(case["id"]),
                        summary=summary,
                        explicit_receipt_kind=preview_metadata.get("explicit_receipt_kind"),
                        explicit_receipt_id=preview_metadata.get("explicit_receipt_id"),
                        inferred_receipt_kind=preview_metadata.get("inferred_receipt_kind"),
                        inferred_receipt_id=preview_metadata.get("inferred_receipt_id"),
                        receipt_target_status=preview_metadata.get("receipt_target_status"),
                        target_lookup_command=preview_metadata.get("target_lookup_command"),
                        execution_case_evidence_handoff=evidence_handoff,
                        local_evidence_append_only=False,
                        authorizes_execution=False,
                        authorizes_completion_claim=False,
                        approval_granted=False,
                        writes_case=False,
                        writes_case_event=False,
                        writes_evidence=False,
                        writes_files=False,
                        writes_database=False,
                        writes_notes=False,
                    ),
                    output=output,
                    action=HARNESS_EVIDENCE_RECOVERY_ACTION,
                ),
            )
        if conflict_reason == "receipt_id_conflict":
            output = (
                "Execution case evidence receipt id conflict. "
                f"Summary references `{preview_metadata.get('inferred_receipt_kind') or 'receipt'} {preview_metadata.get('inferred_receipt_id')}`, "
                f"but explicit receipt id is `{preview_metadata.get('explicit_receipt_id')}`. "
                "Rewrite the evidence with one matching receipt id before saving. "
                f"{HARNESS_EVIDENCE_RECOVERY_ACTION}"
            )
            return ToolResult(
                "append_execution_case_evidence",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(
                        reason="receipt_id_conflict",
                        case_id=int(case["id"]),
                        summary=summary,
                        explicit_receipt_kind=preview_metadata.get("explicit_receipt_kind"),
                        explicit_receipt_id=preview_metadata.get("explicit_receipt_id"),
                        inferred_receipt_kind=preview_metadata.get("inferred_receipt_kind"),
                        inferred_receipt_id=preview_metadata.get("inferred_receipt_id"),
                        receipt_target_status=preview_metadata.get("receipt_target_status"),
                        target_lookup_command=preview_metadata.get("target_lookup_command"),
                        execution_case_evidence_handoff=evidence_handoff,
                        local_evidence_append_only=False,
                        authorizes_execution=False,
                        authorizes_completion_claim=False,
                        approval_granted=False,
                        writes_case=False,
                        writes_case_event=False,
                        writes_evidence=False,
                        writes_files=False,
                        writes_database=False,
                        writes_notes=False,
                    ),
                    output=output,
                    action=HARNESS_EVIDENCE_RECOVERY_ACTION,
                ),
            )
        if not preview_metadata.get("ready_to_append"):
            output = (
                "Execution case evidence is not ready to append. "
                f"Preview verdict: {preview_metadata.get('verdict')}. "
                f"Receipt target status: {preview_metadata.get('receipt_target_status')}. "
                f"{preview_metadata.get('receipt_target_issue') or 'Run the read-only case evidence packet and fix the blocker first.'} "
                f"{HARNESS_EVIDENCE_RECOVERY_ACTION}"
            )
            return ToolResult(
                "append_execution_case_evidence",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(
                        reason=str(preview_metadata.get("verdict") or "evidence_not_ready").lower(),
                        case_id=int(case["id"]),
                        summary=summary,
                        verdict=preview_metadata.get("verdict"),
                        ready_to_append=False,
                        receipt_kind=preview_metadata.get("receipt_kind"),
                        receipt_kind_normalized=preview_metadata.get("receipt_kind_normalized"),
                        receipt_id=preview_metadata.get("receipt_id"),
                        receipt_target_status=preview_metadata.get("receipt_target_status"),
                        receipt_target_exists=preview_metadata.get("receipt_target_exists"),
                        receipt_target_issue=preview_metadata.get("receipt_target_issue"),
                        target_lookup_command=preview_metadata.get("target_lookup_command"),
                        execution_case_evidence_handoff=evidence_handoff,
                        local_evidence_append_only=False,
                        authorizes_execution=False,
                        authorizes_completion_claim=False,
                        approval_granted=False,
                        writes_case=False,
                        writes_case_event=False,
                        writes_evidence=False,
                        writes_files=False,
                        writes_database=False,
                        writes_notes=False,
                    ),
                    output=output,
                    action=HARNESS_EVIDENCE_RECOVERY_ACTION,
                ),
            )
        receipt_kind = str(preview_metadata.get("receipt_kind") or "")
        receipt_id = str(preview_metadata.get("receipt_id") or "")

        event_id = store.add_execution_case_event(
            case_id=int(case["id"]),
            event_type=event_type,
            summary=summary,
            receipt_kind=receipt_kind,
            receipt_id=receipt_id,
            metadata={
                "source": "append_execution_case_evidence",
                "evidence_preview_verdict": preview_metadata.get("verdict"),
                "evidence_preview_ready_to_append": preview_metadata.get("ready_to_append"),
                "receipt_target_status": preview_metadata.get("receipt_target_status"),
                "receipt_target_exists": preview_metadata.get("receipt_target_exists"),
                "target_lookup_command": preview_metadata.get("target_lookup_command"),
                "execution_case_evidence_handoff": evidence_handoff,
            },
        )
        stamp = datetime.now().isoformat(timespec="seconds")
        note_lines = [
            "## Evidence Event",
            "",
            f"- event id: {event_id}",
            f"- created: {stamp}",
            f"- type: {event_type}",
            f"- summary: {summary}",
        ]
        if receipt_kind or receipt_id:
            note_lines.append(f"- receipt: {receipt_kind or 'receipt'} {receipt_id}".rstrip())
        path = vault.append_execution_case_event(str(case["note_path"]), "\n".join(note_lines))
        lines = [
            f"Execution case #{case['id']} evidence event #{event_id} saved.",
            f"Path: {path}",
            f"Type: {event_type}",
            f"Summary: {summary}",
        ]
        if receipt_kind or receipt_id:
            lines.append(f"Receipt: {receipt_kind or 'receipt'} {receipt_id}".rstrip())
        lines.extend(
            [
                "",
                "Boundary:",
                "- This appends a local Jarvis case evidence event only. It does not execute tools, approve requests, dismiss approvals, read private data, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        append_execution_case_evidence_handoff = _safe_metadata(
            source="append_execution_case_evidence",
            case_id=int(case["id"]),
            event_id=event_id,
            event_type=event_type,
            summary=summary,
            receipt_kind=receipt_kind,
            receipt_id=receipt_id,
            inferred_receipt_kind=preview_metadata.get("inferred_receipt_kind"),
            inferred_receipt_id=preview_metadata.get("inferred_receipt_id"),
            evidence_preview_verdict=preview_metadata.get("verdict"),
            evidence_preview_ready_to_append=preview_metadata.get("ready_to_append"),
            note_path=str(path),
            local_evidence_append_only=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
            writes_case=True,
            writes_case_event=True,
            writes_evidence=True,
            writes_files=True,
            writes_database=True,
            writes_notes=True,
        )
        return ToolResult(
            "append_execution_case_evidence",
            True,
            "\n".join(lines),
            _safe_metadata(
                case_id=int(case["id"]),
                event_id=event_id,
                event_type=event_type,
                summary=summary,
                receipt_kind=receipt_kind,
                receipt_id=receipt_id,
                inferred_receipt_kind=preview_metadata.get("inferred_receipt_kind"),
                inferred_receipt_id=preview_metadata.get("inferred_receipt_id"),
                evidence_preview_verdict=preview_metadata.get("verdict"),
                evidence_preview_ready_to_append=preview_metadata.get("ready_to_append"),
                execution_case_evidence_handoff=evidence_handoff,
                append_execution_case_evidence_handoff=append_execution_case_evidence_handoff,
                note_path=str(path),
                local_evidence_append_only=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                writes_case=True,
                writes_case_event=True,
                writes_evidence=True,
                writes_files=True,
                writes_database=True,
                writes_notes=True,
            ),
        )

    def execution_case_gate(args: dict[str, Any]) -> ToolResult:
        case, case_error = _resolve_execution_case_arg("execution_case_gate", args)
        if case_error is not None:
            return case_error

        if case is None:
            gate_defaults = _execution_case_no_case_gate_defaults()
            execution_case_gate_handoff = _safe_metadata(
                source="execution_case_gate",
                execution_case_gate_handoff_ready=True,
                handoff_ready=True,
                **gate_defaults,
            )
            return ToolResult(
                "execution_case_gate",
                True,
                "Jarvis execution case gate:\nNo saved execution case found. Save one first with `save execution case: <order>`.",
                _safe_metadata(
                    **gate_defaults,
                    execution_case_gate_handoff_ready=True,
                    execution_case_gate_handoff=execution_case_gate_handoff,
                ),
            )

        try:
            risk_signals = list(json.loads(case["risk_signals"] or "[]"))
        except Exception:
            risk_signals = []
        case_metadata = _execution_case_metadata(case)
        mission_command_queue = list(case_metadata.get("mission_command_queue") or [])
        execution_proof_handoff = dict(case_metadata.get("execution_proof_handoff") or {})
        execution_proof_aliases = _execution_proof_handoff_aliases(execution_proof_handoff)
        events = store.list_execution_case_events(int(case["id"]), limit=20)
        pending = store.list_pending_approvals(limit=20)
        recent_runs = store.recent_tool_runs(limit=50)
        failed_runs, approval_held_runs = _recent_tool_run_attention_buckets(recent_runs)
        approval_held_review_command = _approval_review_command(approval_held_runs) if approval_held_runs else ""
        recovery_closure = _execution_health_recovery_closure_snapshot(store, recent_runs)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        next_command = str(case["next_command"] or "")
        mission_state = str(case["mission_state"] or "")
        go_no_go = str(case["go_no_go"] or "")
        approval_required = bool(risk_signals)
        tools = list_tools()
        tools_by_name = {tool.name: tool for tool in tools}
        from jarvis_v2.agent.planner import RuleBasedPlanner

        plan = RuleBasedPlanner().plan(str(case["request"]), allow_risky_natural_dispatch=False)
        planned_actions: list[dict[str, Any]] = []
        for action in plan.actions:
            tool = tools_by_name.get(action.tool_name)
            if tool is None:
                risk = "UNKNOWN"
                toolset = "unknown"
                requires_approval = True
            else:
                risk = tool.risk.name
                toolset = tool.toolset
                requires_approval = risk in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
            approval_required = approval_required or requires_approval
            planned_actions.append(
                {
                    "tool": action.tool_name,
                    "toolset": toolset,
                    "risk": risk,
                    "requires_approval": requires_approval,
                    "args": {str(key): _short(value, limit=180) for key, value in action.args.items()},
                    "reason": _short(action.reason, limit=180),
                }
            )
        approval_queue_forecast = []
        for action, item in zip(plan.actions, planned_actions):
            if not item["requires_approval"]:
                continue
            existing = store.find_matching_pending_approval(str(case["request"]), action.tool_name, action.args)
            existing_id = int(existing["id"]) if existing is not None else None
            approval_queue_forecast.append(
                {
                    "tool_name": action.tool_name,
                    "risk": item["risk"],
                    "existing_approval_id": existing_id,
                    "would_queue_new_approval": existing_id is None,
                    "would_reuse_pending_approval": existing_id is not None,
                    "planned_arg_keys": sorted(str(key) for key in action.args),
                }
            )
        forecast_new_approvals = sum(1 for item in approval_queue_forecast if item["would_queue_new_approval"])
        forecast_reused_approval_ids = [
            item["existing_approval_id"]
            for item in approval_queue_forecast
            if item["existing_approval_id"] is not None
        ]
        forecast_queue_before = len(pending)
        forecast_queue_after_if_sent = forecast_queue_before + forecast_new_approvals

        event_text = " ".join(
            f"{event['event_type']} {event['summary']} {event['receipt_kind']} {event['receipt_id']}"
            for event in events
        ).lower()
        evidence_preview_events: list[dict[str, Any]] = []
        for event in events:
            event_metadata = _row_metadata(event)
            evidence_handoff = event_metadata.get("execution_case_evidence_handoff")
            evidence_handoff = evidence_handoff if isinstance(evidence_handoff, dict) else {}
            preview_verdict = str(
                event_metadata.get("evidence_preview_verdict")
                or evidence_handoff.get("verdict")
                or ""
            ).strip()
            ready_value = event_metadata.get("evidence_preview_ready_to_append")
            preview_ready = ready_value if isinstance(ready_value, bool) else bool(evidence_handoff.get("ready_to_append"))
            if not preview_verdict and not evidence_handoff:
                continue
            evidence_preview_events.append(
                {
                    "event_id": int(event["id"]),
                    "event_type": str(event["event_type"] or ""),
                    "receipt_kind": str(event["receipt_kind"] or ""),
                    "receipt_id": str(event["receipt_id"] or ""),
                    "verdict": preview_verdict,
                    "ready_to_append": preview_ready,
                    "handoff": evidence_handoff,
                }
            )
        evidence_preview_verdicts = [
            item["verdict"] for item in evidence_preview_events if item["verdict"]
        ]
        evidence_preview_ready_count = sum(
            1 for item in evidence_preview_events if item["ready_to_append"] is True
        )
        evidence_preview_blocked_count = len(evidence_preview_events) - evidence_preview_ready_count
        evidence_preview_latest = evidence_preview_events[0] if evidence_preview_events else {}
        evidence_preview_latest_handoff = evidence_preview_latest.get("handoff") or {}
        evidence_preview_latest_handoff = (
            evidence_preview_latest_handoff if isinstance(evidence_preview_latest_handoff, dict) else {}
        )
        has_verification_evidence = any(
            token in event_text
            for token in ("verification", "receipt", "test", "smoke", "passed", "confirmed", "runtime trace")
        )
        has_approval_evidence = any(
            token in event_text
            for token in (
                "approval packet",
                "approval reviewed",
                "approved approval",
                "approved request",
                "last-look approval",
                "approval receipt",
            )
        )
        approval_chain_status, approval_evidence_ids, approval_linked_run_ids = _approval_chain_state_for_events(store, events)
        has_approval_chain_evidence = approval_chain_status == "proven"
        verification_receipt_refs = _verification_receipt_refs_from_case_events(events)
        verification_receipt_ids = [
            receipt_id for kind, receipt_id in verification_receipt_refs if kind == "tool_run"
        ]
        runtime_trace_receipt_message_ids = [
            receipt_id for kind, receipt_id in verification_receipt_refs if kind == "runtime_trace_receipt"
        ]
        missing_verification_receipt_ids = _missing_tool_run_ids(store, verification_receipt_ids)
        missing_runtime_trace_receipt_message_ids = _missing_runtime_trace_message_ids(
            store, runtime_trace_receipt_message_ids
        )
        runtime_trace_verified_approval_run_ids = _approved_run_ids_from_runtime_trace_messages(
            store,
            runtime_trace_receipt_message_ids,
            approval_evidence_ids[0] if approval_evidence_ids else None,
        )
        verified_approval_run_ids = sorted(
            (set(approval_linked_run_ids) & set(verification_receipt_ids))
            | (set(approval_linked_run_ids) & set(runtime_trace_verified_approval_run_ids))
        )
        has_verified_approval_run = bool(verified_approval_run_ids)
        has_recovery_evidence = any(token in event_text for token in ("recovery", "audit", "rollback", "stop", "failure"))

        blockers: list[str] = []
        mission_ready = go_no_go in {"GO", "PREFLIGHT_OK"} and "HOLD" not in mission_state
        if not mission_ready:
            blockers.append(f"case go/no-go is {go_no_go or 'UNKNOWN'}")
        if "HOLD" in mission_state:
            blockers.append(f"mission state is {mission_state}")
        if pending:
            blockers.append(f"{len(pending)} pending approval(s) must be reviewed before real execution")
        if not events:
            blockers.append("no execution case evidence events are attached")
        elif not has_verification_evidence:
            blockers.append("case evidence does not include verification/test/receipt proof")
        if approval_required and not has_approval_evidence:
            blockers.append("risk signals require linked approval evidence before completion can be trusted")
        elif approval_required and not has_approval_chain_evidence:
            blockers.append(f"risk signals require a proven approval chain before completion can be trusted ({approval_chain_status})")
        elif approval_required and not has_verified_approval_run:
            blockers.append("approved risky run needs a linked verification/runtime-trace receipt before completion can be trusted")
        if missing_verification_receipt_ids:
            blockers.append(
                "case evidence references missing verification/tool-run receipt id(s): "
                + ", ".join(str(item) for item in missing_verification_receipt_ids)
            )
        if missing_runtime_trace_receipt_message_ids:
            blockers.append(
                "case evidence references missing runtime-trace receipt message id(s): "
                + ", ".join(str(item) for item in missing_runtime_trace_receipt_message_ids)
            )
        if failed_runs and not has_recovery_evidence:
            blockers.append(f"{len(failed_runs)} recent failed/blocked run(s) need recovery or audit evidence")
        if approval_held_runs:
            blockers.append(f"{len(approval_held_runs)} recent approval-held run(s) need approval review")
        if recovery_closure["blocks_completion_claim"]:
            blockers.append(f"execution health recovery closure is {recovery_closure['state']}")
        if learning_debt["blocks_completion_claim"]:
            blockers.append(f"execution learning debt is {learning_debt['state']}")

        if case is None:
            verdict = "CASE_NOT_FOUND"
        elif pending or "HOLD" in mission_state or go_no_go == "NO_GO":
            verdict = "CASE_HELD_FOR_APPROVAL"
        elif not events or not has_verification_evidence:
            verdict = "CASE_NEEDS_EVIDENCE"
        elif approval_required and not has_approval_evidence:
            verdict = "CASE_NEEDS_APPROVAL_EVIDENCE"
        elif approval_required and not has_approval_chain_evidence:
            verdict = "CASE_NEEDS_APPROVAL_CHAIN"
        elif approval_required and not has_verified_approval_run:
            verdict = "CASE_NEEDS_APPROVED_RUN_VERIFICATION"
        elif missing_verification_receipt_ids or missing_runtime_trace_receipt_message_ids:
            verdict = "CASE_NEEDS_RECEIPT_TARGETS"
        elif approval_held_runs:
            verdict = "CASE_NEEDS_APPROVAL_REVIEW"
        elif recovery_closure["blocks_completion_claim"]:
            verdict = "CASE_NEEDS_RECOVERY_CLOSURE"
        elif learning_debt["blocks_completion_claim"]:
            verdict = "CASE_NEEDS_LEARNING_REVIEW"
        elif failed_runs and not has_recovery_evidence:
            verdict = "CASE_NEEDS_RECOVERY_EVIDENCE"
        else:
            verdict = "CASE_READY_FOR_HUMAN_REVIEW"

        case_proof_requirements = [
            ("mission", "case mission state allows review", mission_ready, next_command or "dispatch decision before execution"),
            ("evidence", "at least one execution case evidence event", bool(events), f"case evidence {case['id']}: verification receipt <id> confirmed expected outcome"),
            ("verification", "verification/test/receipt proof attached", has_verification_evidence, f"case evidence {case['id']}: verification receipt <id> confirmed expected outcome"),
            ("receipt targets", "referenced receipt ids exist", not (missing_verification_receipt_ids or missing_runtime_trace_receipt_message_ids), "replace or remove missing receipt references before review"),
            ("approval-held review", "recent approval-held runs have approval review", not approval_held_runs, approval_held_review_command or "approval history"),
            ("recovery", "recent failed runs have recovery/audit evidence", not failed_runs or has_recovery_evidence, f"case evidence {case['id']}: execution recovery packet <id> reviewed stop condition"),
            ("recovery closure", "execution health recovery closure is satisfied", not recovery_closure["blocks_completion_claim"], recovery_closure["next_required_command"] or "execution health report"),
            ("execution learning", "execution learning debt is satisfied", not learning_debt["blocks_completion_claim"], learning_debt["next_required_command"] or "execution learning closure <run id>"),
        ]
        if approval_required:
            case_proof_requirements.insert(2, ("approval evidence", "approval evidence is attached", has_approval_evidence, f"case evidence {case['id']}: approval readiness <id>, approval packet <id>, and approval chain proof <id> reviewed"))
            case_proof_requirements.insert(3, ("approval chain", "approval is approved and linked to a successful rerun", has_approval_chain_evidence, "approval chain proof <approval id>"))
            case_proof_requirements.insert(4, ("approved run verification", "approved rerun has linked verification/runtime-trace receipt", has_verified_approval_run, f"case evidence {case['id']}: verification receipt <approved run id> confirmed expected outcome"))
        missing_case_proofs = [name for name, _label, present, _command in case_proof_requirements if not present]
        next_case_proof_commands: list[str] = []
        for _name, _label, present, command in case_proof_requirements:
            if not present and command not in next_case_proof_commands:
                next_case_proof_commands.append(command)
        _append_unique(next_case_proof_commands, list(recovery_closure["required_commands"]))
        _append_unique(next_case_proof_commands, list(learning_debt["required_commands"]))

        lines = [
            "Jarvis execution case gate:",
            "This read-only gate decides whether a saved case has enough evidence to move toward real execution or completion review.",
            "",
            f"Case: #{case['id']}",
            f"Request: {case['request']}",
            f"Verdict: {verdict}",
            f"Mission state: {mission_state or 'UNKNOWN'}",
            f"Go/no-go: {go_no_go or 'UNKNOWN'}",
            f"Next command: `{next_command}`",
            "",
            "Evidence posture:",
            f"- evidence events: {len(events)}",
            f"- evidence preflight events: {len(evidence_preview_events)}",
            f"- evidence preflight ready events: {evidence_preview_ready_count}",
            f"- latest evidence preflight verdict: {evidence_preview_latest.get('verdict') or 'none'}",
            f"- verification evidence: {'present' if has_verification_evidence else 'missing'}",
            f"- approval evidence: {'present' if has_approval_evidence else 'missing'}",
            f"- approval chain: {approval_chain_status}",
            f"- linked approval ids: {', '.join(str(item) for item in approval_evidence_ids) if approval_evidence_ids else 'none'}",
            f"- linked approved runs: {', '.join(str(item) for item in approval_linked_run_ids) if approval_linked_run_ids else 'none'}",
            f"- verification/tool-run receipt ids: {', '.join(str(item) for item in verification_receipt_ids) if verification_receipt_ids else 'none'}",
            f"- runtime-trace receipt message ids: {', '.join(str(item) for item in runtime_trace_receipt_message_ids) if runtime_trace_receipt_message_ids else 'none'}",
            f"- missing verification/tool-run receipt targets: {', '.join(str(item) for item in missing_verification_receipt_ids) if missing_verification_receipt_ids else 'none'}",
            f"- missing runtime-trace receipt targets: {', '.join(str(item) for item in missing_runtime_trace_receipt_message_ids) if missing_runtime_trace_receipt_message_ids else 'none'}",
            f"- verified approved runs: {', '.join(str(item) for item in verified_approval_run_ids) if verified_approval_run_ids else 'none'}",
            f"- recovery evidence: {'present' if has_recovery_evidence else 'missing'}",
            f"- risk signals: {', '.join(risk_signals) if risk_signals else 'none'}",
            f"- pending approvals: {len(pending)}",
            f"- would queue new approvals if sent: {forecast_new_approvals}",
            f"- would reuse pending approval ids: {', '.join(str(item) for item in forecast_reused_approval_ids) if forecast_reused_approval_ids else 'none'}",
            f"- forecast queue after if sent: {forecast_queue_after_if_sent}",
            f"- recent failed/blocked runs: {len(failed_runs)}",
            f"- recent approval-held runs: {len(approval_held_runs)}",
            f"- approval-held review: `{approval_held_review_command}`" if approval_held_review_command else "- approval-held review: none",
            "",
            "Execution health recovery closure:",
            f"- state: {recovery_closure['state']}",
            f"- target run: #{recovery_closure['target_run_id']} {recovery_closure['target_tool_name']}" if recovery_closure["target_run_id"] else "- target run: none",
            f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
            f"- next required command: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required command: none",
            "",
            "Execution learning debt:",
            f"- state: {learning_debt['state']}",
            f"- target run: #{learning_debt['target_run_id']} {learning_debt['target_tool_name']}" if learning_debt["target_run_id"] else "- target run: none",
            f"- missing: {', '.join(learning_debt['missing']) if learning_debt['missing'] else 'none'}",
            f"- next required command: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- next required command: none",
            "",
            "Case proof gaps:",
        ]
        if missing_case_proofs:
            for name, label, present, command in case_proof_requirements:
                if present:
                    continue
                lines.append(f"- {name}: missing - {label}")
                lines.append(f"  next: `{command}`")
        else:
            lines.append("- none; case proof is ready for human review")
        lines.extend(["", "Mission command queue:"])
        if mission_command_queue:
            lines.extend(f"- `{command}`" for command in mission_command_queue)
        else:
            lines.append("- not stored for this case")
        lines.extend(
            [
                "",
            "Blocking reasons:",
            ]
        )
        if blockers:
            lines.extend(f"- {blocker}" for blocker in blockers)
        else:
            lines.append("- none found by this read-only case gate")

        if verdict == "CASE_READY_FOR_HUMAN_REVIEW":
            next_step = f"execution proof bundle: {case['request']}; verification <receipt>; tests <check>; evidence case #{case['id']}; recovery <stop>"
        elif verdict == "CASE_HELD_FOR_APPROVAL":
            next_step = next_command or "review pending approval packet before any real execution"
        else:
            next_step = f"case evidence {case['id']}: verification receipt <id> confirmed expected outcome"
        execution_case_gate_handoff = _safe_metadata(
            source="execution_case_gate",
            execution_case_gate_handoff_ready=True,
            handoff_ready=True,
            found=True,
            case_id=int(case["id"]),
            verdict=verdict,
            request=case["request"],
            mission_state=mission_state,
            go_no_go=go_no_go,
            next_command=next_command,
            next_safe_command=next_step,
            risk_signals=risk_signals,
            approval_required=approval_required,
            planned_action_count=len(planned_actions),
            forecast_new_approvals=forecast_new_approvals,
            forecast_reused_approval_ids=forecast_reused_approval_ids,
            forecast_queue_before=forecast_queue_before,
            forecast_queue_after_if_sent=forecast_queue_after_if_sent,
            forecast_queue_delta_if_sent=forecast_new_approvals,
            pending_approvals=len(pending),
            recent_failed_runs=len(failed_runs),
            recent_approval_held_runs=len(approval_held_runs),
            approval_held_review_command=approval_held_review_command,
            events=len(events),
            evidence_preview_gate_count=len(evidence_preview_events),
            evidence_preview_ready_count=evidence_preview_ready_count,
            evidence_preview_blocked_count=evidence_preview_blocked_count,
            evidence_preview_verdicts=evidence_preview_verdicts,
            evidence_preview_latest_verdict=evidence_preview_latest.get("verdict") or "",
            evidence_preview_latest_event_id=evidence_preview_latest.get("event_id"),
            evidence_preview_latest_handoff=evidence_preview_latest_handoff,
            evidence_preview_latest_handoff_present=bool(evidence_preview_latest_handoff),
            has_verification_evidence=has_verification_evidence,
            has_approval_evidence=has_approval_evidence,
            has_approval_chain_evidence=has_approval_chain_evidence,
            approval_chain_status=approval_chain_status,
            approval_evidence_ids=approval_evidence_ids,
            approval_linked_run_ids=approval_linked_run_ids,
            verification_receipt_ids=verification_receipt_ids,
            runtime_trace_receipt_message_ids=runtime_trace_receipt_message_ids,
            missing_verification_receipt_ids=missing_verification_receipt_ids,
            missing_runtime_trace_receipt_message_ids=missing_runtime_trace_receipt_message_ids,
            has_missing_verification_receipts=bool(missing_verification_receipt_ids),
            has_missing_runtime_trace_receipts=bool(missing_runtime_trace_receipt_message_ids),
            verified_approval_run_ids=verified_approval_run_ids,
            has_verified_approval_run=has_verified_approval_run,
            has_recovery_evidence=has_recovery_evidence,
            execution_health_recovery_closure_state=recovery_closure["state"],
            execution_health_recovery_closure_missing=recovery_closure["missing"],
            execution_health_recovery_closure_missing_count=recovery_closure["missing_count"],
            execution_health_recovery_closure_required_commands=recovery_closure["required_commands"],
            execution_health_recovery_closure_next_required_command=recovery_closure["next_required_command"],
            execution_health_recovery_closure_proof_queue=recovery_closure["proof_queue"],
            execution_health_recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
            execution_health_recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
            execution_health_recovery_closure_blocks_completion_claim=recovery_closure["blocks_completion_claim"],
            execution_learning_state=learning_debt["state"],
            execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
            execution_learning_missing=learning_debt["missing"],
            execution_learning_missing_count=learning_debt["missing_count"],
            execution_learning_required_commands=learning_debt["required_commands"],
            execution_learning_next_required_command=learning_debt["next_required_command"],
            execution_learning_proof_queue=learning_debt["proof_queue"],
            execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
            execution_learning_next_proof_command=learning_debt["next_proof_command"],
            case_proof_requirements=len(case_proof_requirements),
            missing_case_proofs=missing_case_proofs,
            next_case_proof_commands=next_case_proof_commands,
            next_case_required_command=next_case_proof_commands[0] if next_case_proof_commands else next_step,
            next_case_proof_command=next_case_proof_commands[0] if next_case_proof_commands else next_step,
            mission_command_queue=mission_command_queue,
            mission_command_count=len(mission_command_queue),
            initial_governor_command=case_metadata.get("initial_governor_command", ""),
            proof_bundle_command=case_metadata.get("proof_bundle_command", ""),
            acceptance_command=case_metadata.get("acceptance_command", ""),
            audit_command=case_metadata.get("audit_command", ""),
            recovery_command=case_metadata.get("recovery_command", ""),
            learning_command=case_metadata.get("learning_command", ""),
            blockers=len(blockers),
            execution_proof_handoff=execution_proof_handoff,
            execution_proof_handoff_present=bool(execution_proof_handoff),
            execution_proof_state=case_metadata.get("execution_proof_state"),
            execution_proof_next_command=case_metadata.get("execution_proof_next_command"),
            execution_proof_can_trust_execution=case_metadata.get("execution_proof_can_trust_execution"),
            draft_only=True,
            requires_manual_send=True,
            loads_without_execution=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
            **execution_proof_aliases,
        )
        lines.extend(
            [
                "",
                f"Next safe command: `{next_step}`",
                "",
                "Boundary:",
                "- This gate is read-only. It does not execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )

        return ToolResult(
            "execution_case_gate",
            True,
            "\n".join(lines),
            _safe_metadata(
                found=True,
                case_id=int(case["id"]),
                verdict=verdict,
                request=case["request"],
                mission_state=mission_state,
                go_no_go=go_no_go,
                next_command=next_command,
                next_safe_command=next_step,
                risk_signals=risk_signals,
                approval_required=approval_required,
                planned_actions=planned_actions,
                planned_action_count=len(planned_actions),
                approval_queue_forecast=approval_queue_forecast,
                forecast_new_approvals=forecast_new_approvals,
                forecast_reused_approval_ids=forecast_reused_approval_ids,
                forecast_queue_before=forecast_queue_before,
                forecast_queue_after_if_sent=forecast_queue_after_if_sent,
                forecast_queue_delta_if_sent=forecast_new_approvals,
                pending_approvals=len(pending),
                recent_failed_runs=len(failed_runs),
                recent_approval_held_runs=len(approval_held_runs),
                approval_held_review_command=approval_held_review_command,
                events=len(events),
                evidence_preview_gate_count=len(evidence_preview_events),
                evidence_preview_ready_count=evidence_preview_ready_count,
                evidence_preview_blocked_count=evidence_preview_blocked_count,
                evidence_preview_verdicts=evidence_preview_verdicts,
                evidence_preview_latest_verdict=evidence_preview_latest.get("verdict") or "",
                evidence_preview_latest_event_id=evidence_preview_latest.get("event_id"),
                evidence_preview_latest_handoff=evidence_preview_latest_handoff,
                evidence_preview_latest_handoff_present=bool(evidence_preview_latest_handoff),
                has_verification_evidence=has_verification_evidence,
                has_approval_evidence=has_approval_evidence,
                has_approval_chain_evidence=has_approval_chain_evidence,
                approval_chain_status=approval_chain_status,
                approval_evidence_ids=approval_evidence_ids,
                approval_linked_run_ids=approval_linked_run_ids,
                verification_receipt_ids=verification_receipt_ids,
                runtime_trace_receipt_message_ids=runtime_trace_receipt_message_ids,
                missing_verification_receipt_ids=missing_verification_receipt_ids,
                missing_runtime_trace_receipt_message_ids=missing_runtime_trace_receipt_message_ids,
                has_missing_verification_receipts=bool(missing_verification_receipt_ids),
                has_missing_runtime_trace_receipts=bool(missing_runtime_trace_receipt_message_ids),
                runtime_trace_verified_approval_run_ids=runtime_trace_verified_approval_run_ids,
                verified_approval_run_ids=verified_approval_run_ids,
                has_verified_approval_run=has_verified_approval_run,
                has_recovery_evidence=has_recovery_evidence,
                execution_health_recovery_closure_state=recovery_closure["state"],
                execution_health_recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                execution_health_recovery_closure_missing=recovery_closure["missing"],
                execution_health_recovery_closure_missing_count=recovery_closure["missing_count"],
                execution_health_recovery_closure_required_commands=recovery_closure["required_commands"],
                execution_health_recovery_closure_next_required_command=recovery_closure["next_required_command"],
                execution_health_recovery_closure_checklist_command=_recovery_closure_checklist_command(recovery_closure),
                execution_health_recovery_closure_should_open_checklist=bool(_recovery_closure_checklist_command(recovery_closure)),
                execution_health_recovery_closure_proof_queue=recovery_closure["proof_queue"],
                execution_health_recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
                execution_health_recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
                execution_health_recovery_closure_blocks_completion_claim=recovery_closure["blocks_completion_claim"],
                execution_health_recovery_closure_target_run_id=recovery_closure["target_run_id"],
                execution_health_recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                execution_health_recovery_closure_target_verification_receipts=recovery_closure["target_verification_receipts"],
                execution_health_recovery_closure_target_recovery_packets=recovery_closure["target_recovery_packets"],
                execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure["target_after_action_learning_packets"],
                execution_learning_state=learning_debt["state"],
                execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
                execution_learning_recent_action_runs=learning_debt["recent_action_runs"],
                execution_learning_failed_or_blocked_action_runs=learning_debt["failed_or_blocked_action_runs"],
                execution_learning_recent_verification_runs=learning_debt["recent_verification_runs"],
                execution_learning_recent_recovery_runs=learning_debt["recent_recovery_runs"],
                execution_learning_recent_after_action_learning_runs=learning_debt["recent_after_action_learning_runs"],
                execution_learning_target_run_id=learning_debt["target_run_id"],
                execution_learning_target_tool_name=learning_debt["target_tool_name"],
                execution_learning_target_after_action_learning_packets=learning_debt["target_after_action_learning_packets"],
                execution_learning_missing=learning_debt["missing"],
                execution_learning_missing_count=learning_debt["missing_count"],
                execution_learning_required_commands=learning_debt["required_commands"],
                execution_learning_next_required_command=learning_debt["next_required_command"],
                execution_learning_proof_queue=learning_debt["proof_queue"],
                execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
                execution_learning_next_proof_command=learning_debt["next_proof_command"],
                case_proof_requirements=len(case_proof_requirements),
                missing_case_proofs=missing_case_proofs,
                next_case_proof_commands=next_case_proof_commands,
                next_case_required_command=next_case_proof_commands[0] if next_case_proof_commands else next_step,
                next_case_proof_command=next_case_proof_commands[0] if next_case_proof_commands else next_step,
                mission_command_queue=mission_command_queue,
                mission_command_count=len(mission_command_queue),
                initial_governor_command=case_metadata.get("initial_governor_command", ""),
                proof_bundle_command=case_metadata.get("proof_bundle_command", ""),
                acceptance_command=case_metadata.get("acceptance_command", ""),
                audit_command=case_metadata.get("audit_command", ""),
                recovery_command=case_metadata.get("recovery_command", ""),
                learning_command=case_metadata.get("learning_command", ""),
                blockers=len(blockers),
                execution_proof_handoff=execution_proof_handoff,
                execution_proof_handoff_present=bool(execution_proof_handoff),
                execution_proof_state=case_metadata.get("execution_proof_state"),
                execution_proof_next_command=case_metadata.get("execution_proof_next_command"),
                execution_proof_can_trust_execution=case_metadata.get("execution_proof_can_trust_execution"),
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                **execution_proof_aliases,
                execution_case_gate_handoff_ready=True,
                execution_case_gate_handoff=execution_case_gate_handoff,
            ),
        )

    def execution_case_review_packet(args: dict[str, Any]) -> ToolResult:
        case, case_error = _resolve_execution_case_arg("execution_case_review_packet", args)
        if case_error is not None:
            return case_error

        if case is None:
            gate_defaults = _execution_case_no_case_gate_defaults()
            gate_handoff = _safe_metadata(
                source="execution_case_gate",
                execution_case_gate_handoff_ready=True,
                handoff_ready=True,
                **gate_defaults,
            )
            review_defaults = {
                **gate_defaults,
                "review_state": "NO_CASE",
                "checklist_items": 0,
                "gate_handoff": gate_handoff,
                "gate_handoff_present": False,
                "execution_case_gate_handoff_ready": True,
                "draft_only": True,
                "requires_manual_send": True,
                "loads_without_execution": True,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            }
            execution_case_review_handoff = _safe_metadata(
                source="execution_case_review_packet",
                execution_case_review_handoff_ready=True,
                handoff_ready=True,
                **review_defaults,
            )
            return ToolResult(
                "execution_case_review_packet",
                True,
                "Jarvis execution case review packet:\nNo saved execution case found. Save one first with `save execution case: <order>`.",
                _safe_metadata(
                    **review_defaults,
                    execution_case_review_handoff_ready=True,
                    execution_case_review_handoff=execution_case_review_handoff,
                ),
            )

        try:
            risk_signals = list(json.loads(case["risk_signals"] or "[]"))
        except Exception:
            risk_signals = []
        events = store.list_execution_case_events(int(case["id"]), limit=20)
        pending = store.list_pending_approvals(limit=20)
        recent_runs = store.recent_tool_runs(limit=20)
        failed_runs, approval_held_runs = _recent_tool_run_attention_buckets(recent_runs)
        gate = execution_case_gate({"case_id": int(case["id"])})
        gate_metadata = dict(gate.metadata)
        execution_proof_handoff = dict(gate_metadata.get("execution_proof_handoff") or {})
        execution_proof_aliases = _execution_proof_handoff_aliases(execution_proof_handoff)
        verdict = str(gate_metadata.get("verdict") or "UNKNOWN")
        blockers = int(gate_metadata.get("blockers") or 0)
        next_safe_command = str(gate_metadata.get("next_safe_command") or case["next_command"] or "")
        approval_required = _metadata_bool(gate_metadata.get("approval_required"))
        has_verification_evidence = _metadata_bool(gate_metadata.get("has_verification_evidence"))
        has_approval_evidence = _metadata_bool(gate_metadata.get("has_approval_evidence"))
        has_approval_chain_evidence = _metadata_bool(gate_metadata.get("has_approval_chain_evidence"))
        approval_chain_status = str(gate_metadata.get("approval_chain_status") or "missing")
        approval_evidence_ids = list(gate_metadata.get("approval_evidence_ids") or [])
        approval_linked_run_ids = list(gate_metadata.get("approval_linked_run_ids") or [])
        verification_receipt_ids = list(gate_metadata.get("verification_receipt_ids") or [])
        runtime_trace_receipt_message_ids = list(gate_metadata.get("runtime_trace_receipt_message_ids") or [])
        missing_verification_receipt_ids = list(gate_metadata.get("missing_verification_receipt_ids") or [])
        missing_runtime_trace_receipt_message_ids = list(
            gate_metadata.get("missing_runtime_trace_receipt_message_ids") or []
        )
        verified_approval_run_ids = list(gate_metadata.get("verified_approval_run_ids") or [])
        has_verified_approval_run = _metadata_bool(gate_metadata.get("has_verified_approval_run"))
        has_recovery_evidence = _metadata_bool(gate_metadata.get("has_recovery_evidence"))
        recovery_closure_state = str(gate_metadata.get("execution_health_recovery_closure_state") or "unknown")
        recovery_closure_missing = list(gate_metadata.get("execution_health_recovery_closure_missing") or [])
        recovery_closure_required_commands = list(gate_metadata.get("execution_health_recovery_closure_required_commands") or [])
        recovery_closure_next_required_command = str(gate_metadata.get("execution_health_recovery_closure_next_required_command") or "")
        recovery_closure_proof_queue = list(gate_metadata.get("execution_health_recovery_closure_proof_queue") or [])
        recovery_closure_next_proof_command = str(gate_metadata.get("execution_health_recovery_closure_next_proof_command") or "")
        recovery_closure_blocks_completion_claim = _metadata_bool(gate_metadata.get("execution_health_recovery_closure_blocks_completion_claim"))
        recovery_closure_target_run_id = gate_metadata.get("execution_health_recovery_closure_target_run_id")
        recovery_closure_target_tool_name = str(gate_metadata.get("execution_health_recovery_closure_target_tool_name") or "")
        recovery_closure_target_verification_receipts = _metadata_int(gate_metadata.get("execution_health_recovery_closure_target_verification_receipts"))
        recovery_closure_target_recovery_packets = _metadata_int(gate_metadata.get("execution_health_recovery_closure_target_recovery_packets"))
        recovery_closure_target_after_action_learning_packets = _metadata_int(gate_metadata.get("execution_health_recovery_closure_target_after_action_learning_packets"))
        learning_state = str(gate_metadata.get("execution_learning_state") or "unknown")
        learning_missing = list(gate_metadata.get("execution_learning_missing") or [])
        learning_required_commands = list(gate_metadata.get("execution_learning_required_commands") or [])
        learning_next_required_command = str(gate_metadata.get("execution_learning_next_required_command") or "")
        learning_proof_queue = list(gate_metadata.get("execution_learning_proof_queue") or [])
        learning_next_proof_command = str(gate_metadata.get("execution_learning_next_proof_command") or "")
        learning_blocks_completion_claim = _metadata_bool(gate_metadata.get("execution_learning_blocks_completion_claim"))
        learning_target_run_id = gate_metadata.get("execution_learning_target_run_id")
        learning_target_tool_name = str(gate_metadata.get("execution_learning_target_tool_name") or "")
        missing_case_proofs = list(gate_metadata.get("missing_case_proofs") or [])
        next_case_proof_commands = list(gate_metadata.get("next_case_proof_commands") or [])
        mission_command_queue = list(gate_metadata.get("mission_command_queue") or [])
        approval_queue_forecast = list(gate_metadata.get("approval_queue_forecast") or [])
        forecast_new_approvals = _metadata_int(gate_metadata.get("forecast_new_approvals"))
        forecast_reused_approval_ids = list(gate_metadata.get("forecast_reused_approval_ids") or [])
        forecast_queue_before = _metadata_int(gate_metadata.get("forecast_queue_before"), len(pending))
        forecast_queue_after_if_sent = _metadata_int(gate_metadata.get("forecast_queue_after_if_sent"), forecast_queue_before)
        forecast_queue_delta_if_sent = _metadata_int(gate_metadata.get("forecast_queue_delta_if_sent"), forecast_new_approvals)
        recent_failed_run_count = _metadata_int(gate_metadata.get("recent_failed_runs"), len(failed_runs))
        recent_approval_held_run_count = _metadata_int(
            gate_metadata.get("recent_approval_held_runs"),
            len(approval_held_runs),
        )
        approval_held_review_command = str(
            gate_metadata.get("approval_held_review_command")
            or (_approval_review_command(approval_held_runs) if approval_held_runs else "")
        )
        evidence_preview_gate_count = _metadata_int(gate_metadata.get("evidence_preview_gate_count"))
        evidence_preview_ready_count = _metadata_int(gate_metadata.get("evidence_preview_ready_count"))
        evidence_preview_blocked_count = _metadata_int(gate_metadata.get("evidence_preview_blocked_count"))
        evidence_preview_verdicts = list(gate_metadata.get("evidence_preview_verdicts") or [])
        evidence_preview_latest_verdict = str(gate_metadata.get("evidence_preview_latest_verdict") or "")
        evidence_preview_latest_event_id = gate_metadata.get("evidence_preview_latest_event_id")
        evidence_preview_latest_handoff = gate_metadata.get("evidence_preview_latest_handoff")
        evidence_preview_latest_handoff = (
            evidence_preview_latest_handoff if isinstance(evidence_preview_latest_handoff, dict) else {}
        )
        evidence_preview_latest_handoff_present = _metadata_bool(gate_metadata.get("evidence_preview_latest_handoff_present"))

        event_rows = []
        for event in events[:6]:
            receipt = ""
            if event["receipt_kind"] or event["receipt_id"]:
                receipt = f" [{event['receipt_kind'] or 'receipt'} {event['receipt_id']}]".rstrip()
            event_rows.append(f"- #{event['id']} {event['event_type']}{receipt}: {_short(event['summary'], limit=160)}")

        if verdict == "CASE_READY_FOR_HUMAN_REVIEW":
            review_state = "READY_FOR_HUMAN_REVIEW"
        elif verdict == "CASE_NOT_FOUND":
            review_state = "NO_CASE"
        elif "APPROVAL" in verdict:
            review_state = "APPROVAL_REVIEW_REQUIRED"
        elif "RECOVERY" in verdict:
            review_state = "RECOVERY_REVIEW_REQUIRED"
        else:
            review_state = "EVIDENCE_REVIEW_REQUIRED"

        checklist = [
            "Confirm the request text still matches what the operator wants now.",
            "Confirm mission state, go/no-go, and next command are not stale.",
            "Inspect every evidence event and make sure it points to real verification, not only intent.",
            "If risk signals exist, confirm linked approval-chain evidence before trusting any action outcome.",
            "If risk signals exist, confirm the linked approved run also has a verification or runtime-trace receipt.",
            "If recent failures exist, confirm recovery/audit evidence before retrying or claiming completion.",
        ]
        if approval_required:
            checklist.append("Review approval readiness before the last-look approval packet, then run approval chain proof; approval must be one-shot and exact.")
        if has_verification_evidence:
            checklist.append("Attach or cite the verification receipt in the final outcome summary.")
        else:
            checklist.append("Add verification evidence before claiming the case worked.")

        lines = [
            "Jarvis execution case review packet:",
            "This is the human-review packet for a saved execution case. It is read-only and does not execute the order.",
            "",
            f"Case: #{case['id']}",
            f"Request: {case['request']}",
            f"Review state: {review_state}",
            f"Gate verdict: {verdict}",
            f"Mission state: {case['mission_state']}",
            f"Go/no-go: {case['go_no_go']}",
            f"Next safe command: `{next_safe_command}`",
            "",
            "Evidence summary:",
            f"- evidence events: {len(events)}",
            f"- evidence preflight events: {evidence_preview_gate_count}",
            f"- evidence preflight ready events: {evidence_preview_ready_count}",
            f"- latest evidence preflight verdict: {evidence_preview_latest_verdict or 'none'}",
            f"- verification evidence: {'present' if has_verification_evidence else 'missing'}",
            f"- approval evidence: {'present' if has_approval_evidence else 'missing'}",
            f"- approval chain: {approval_chain_status}",
            f"- linked approval ids: {', '.join(str(item) for item in approval_evidence_ids) if approval_evidence_ids else 'none'}",
            f"- linked approved runs: {', '.join(str(item) for item in approval_linked_run_ids) if approval_linked_run_ids else 'none'}",
            f"- verification/tool-run receipt ids: {', '.join(str(item) for item in verification_receipt_ids) if verification_receipt_ids else 'none'}",
            f"- runtime-trace receipt message ids: {', '.join(str(item) for item in runtime_trace_receipt_message_ids) if runtime_trace_receipt_message_ids else 'none'}",
            f"- missing verification/tool-run receipt targets: {', '.join(str(item) for item in missing_verification_receipt_ids) if missing_verification_receipt_ids else 'none'}",
            f"- missing runtime-trace receipt targets: {', '.join(str(item) for item in missing_runtime_trace_receipt_message_ids) if missing_runtime_trace_receipt_message_ids else 'none'}",
            f"- verified approved runs: {', '.join(str(item) for item in verified_approval_run_ids) if verified_approval_run_ids else 'none'}",
            f"- recovery evidence: {'present' if has_recovery_evidence else 'missing'}",
            f"- risk signals: {', '.join(risk_signals) if risk_signals else 'none'}",
            f"- pending approvals: {len(pending)}",
            f"- would queue new approvals if sent: {forecast_new_approvals}",
            f"- would reuse pending approval ids: {', '.join(str(item) for item in forecast_reused_approval_ids) if forecast_reused_approval_ids else 'none'}",
            f"- forecast queue after if sent: {forecast_queue_after_if_sent}",
            f"- recent failed/blocked runs: {recent_failed_run_count}",
            f"- recent approval-held runs: {recent_approval_held_run_count}",
            f"- approval-held review: `{approval_held_review_command}`" if approval_held_review_command else "- approval-held review: none",
            "",
            "Execution health recovery closure:",
            f"- state: {recovery_closure_state}",
            f"- target run: #{recovery_closure_target_run_id} {recovery_closure_target_tool_name}" if recovery_closure_target_run_id else "- target run: none",
            f"- target verification receipts: {recovery_closure_target_verification_receipts}",
            f"- target recovery packets: {recovery_closure_target_recovery_packets}",
            f"- target after-action learning packets: {recovery_closure_target_after_action_learning_packets}",
            f"- missing: {', '.join(recovery_closure_missing) if recovery_closure_missing else 'none'}",
            f"- blocks completion claim: {'yes' if recovery_closure_blocks_completion_claim else 'no'}",
            f"- next required command: `{recovery_closure_next_required_command}`" if recovery_closure_next_required_command else "- next required command: none",
            "",
            "Execution learning debt:",
            f"- state: {learning_state}",
            f"- target run: #{learning_target_run_id} {learning_target_tool_name}" if learning_target_run_id else "- target run: none",
            f"- missing: {', '.join(learning_missing) if learning_missing else 'none'}",
            f"- blocks completion claim: {'yes' if learning_blocks_completion_claim else 'no'}",
            f"- next required command: `{learning_next_required_command}`" if learning_next_required_command else "- next required command: none",
            "",
            "Case proof gaps:",
        ]
        if missing_case_proofs:
            lines.extend(f"- {item}" for item in missing_case_proofs)
        else:
            lines.append("- none")
        if next_case_proof_commands:
            lines.extend(["", "Next required commands:"])
            lines.extend(f"- `{command}`" for command in next_case_proof_commands)
        lines.extend(["", "Mission command queue:"])
        if mission_command_queue:
            lines.extend(f"- `{command}`" for command in mission_command_queue)
        else:
            lines.append("- not stored for this case")
        lines.extend(
            [
                "",
            "Recent case events:",
            ]
        )
        lines.extend(event_rows if event_rows else ["- none attached yet"])
        lines.extend(
            [
                "",
                "Review checklist:",
                *[f"- {item}" for item in checklist],
                "",
                "Decision guidance:",
            ]
        )
        if review_state == "READY_FOR_HUMAN_REVIEW":
            lines.append("- The case can move to human outcome review; still do not claim broad Jarvis completion from this case alone.")
        elif review_state == "APPROVAL_REVIEW_REQUIRED":
            lines.append(f"- Hold execution. Start with `{next_safe_command or 'approval review'}` and do not approve if the packet is stale or broad.")
        elif review_state == "RECOVERY_REVIEW_REQUIRED":
            lines.append("- Hold retry/completion. Inspect the failed run and attach recovery evidence to the case.")
        else:
            lines.append("- Hold completion. Attach verification/test/receipt evidence to the case first.")
        lines.extend(
            [
                "",
                "Follow-up commands:",
                f"- `execution case gate {case['id']}`",
                f"- `execution runbook: {case['request']}`",
                f"- `execution proof bundle: {case['request']}; verification <receipt>; tests <check>; evidence case #{case['id']}; recovery <stop>; approval <id if risky>`",
                f"- `case evidence {case['id']}: verification receipt <id> confirmed expected outcome`",
                "",
                "Boundary:",
                "- This review packet does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        execution_case_review_handoff = _safe_metadata(
            source="execution_case_review_packet",
            execution_case_review_handoff_ready=True,
            handoff_ready=True,
            found=True,
            case_id=int(case["id"]),
            request=case["request"],
            review_state=review_state,
            verdict=verdict,
            mission_state=case["mission_state"],
            go_no_go=case["go_no_go"],
            next_safe_command=next_safe_command,
            risk_signals=risk_signals,
            approval_required=approval_required,
            forecast_new_approvals=forecast_new_approvals,
            forecast_reused_approval_ids=forecast_reused_approval_ids,
            forecast_queue_before=forecast_queue_before,
            forecast_queue_after_if_sent=forecast_queue_after_if_sent,
            forecast_queue_delta_if_sent=forecast_queue_delta_if_sent,
            pending_approvals=len(pending),
            recent_failed_runs=recent_failed_run_count,
            recent_approval_held_runs=recent_approval_held_run_count,
            approval_held_review_command=approval_held_review_command,
            events=len(events),
            evidence_preview_gate_count=evidence_preview_gate_count,
            evidence_preview_ready_count=evidence_preview_ready_count,
            evidence_preview_blocked_count=evidence_preview_blocked_count,
            evidence_preview_verdicts=evidence_preview_verdicts,
            evidence_preview_latest_verdict=evidence_preview_latest_verdict,
            evidence_preview_latest_event_id=evidence_preview_latest_event_id,
            evidence_preview_latest_handoff=evidence_preview_latest_handoff,
            evidence_preview_latest_handoff_present=evidence_preview_latest_handoff_present,
            has_verification_evidence=has_verification_evidence,
            has_approval_evidence=has_approval_evidence,
            has_approval_chain_evidence=has_approval_chain_evidence,
            approval_chain_status=approval_chain_status,
            approval_evidence_ids=approval_evidence_ids,
            approval_linked_run_ids=approval_linked_run_ids,
            verification_receipt_ids=verification_receipt_ids,
            missing_verification_receipt_ids=missing_verification_receipt_ids,
            has_missing_verification_receipts=bool(missing_verification_receipt_ids),
            verified_approval_run_ids=verified_approval_run_ids,
            has_verified_approval_run=has_verified_approval_run,
            has_recovery_evidence=has_recovery_evidence,
            execution_health_recovery_closure_state=recovery_closure_state,
            execution_health_recovery_closure_missing=recovery_closure_missing,
            execution_health_recovery_closure_missing_count=len(recovery_closure_missing),
            execution_health_recovery_closure_proof_queue=recovery_closure_proof_queue,
            execution_health_recovery_closure_proof_queue_count=len(recovery_closure_proof_queue),
            execution_health_recovery_closure_next_proof_command=recovery_closure_next_proof_command,
            execution_health_recovery_closure_blocks_completion_claim=recovery_closure_blocks_completion_claim,
            execution_learning_state=learning_state,
            execution_learning_blocks_completion_claim=learning_blocks_completion_claim,
            execution_learning_missing=learning_missing,
            execution_learning_missing_count=len(learning_missing),
            execution_learning_proof_queue=learning_proof_queue,
            execution_learning_proof_queue_count=len(learning_proof_queue),
            execution_learning_next_proof_command=learning_next_proof_command,
            missing_case_proofs=missing_case_proofs,
            next_case_proof_commands=next_case_proof_commands,
            next_case_required_command=next_case_proof_commands[0] if next_case_proof_commands else next_safe_command,
            next_case_proof_command=next_case_proof_commands[0] if next_case_proof_commands else next_safe_command,
            mission_command_queue=mission_command_queue,
            mission_command_count=len(mission_command_queue),
            initial_governor_command=gate_metadata.get("initial_governor_command", ""),
            proof_bundle_command=gate_metadata.get("proof_bundle_command", ""),
            acceptance_command=gate_metadata.get("acceptance_command", ""),
            audit_command=gate_metadata.get("audit_command", ""),
            recovery_command=gate_metadata.get("recovery_command", ""),
            learning_command=gate_metadata.get("learning_command", ""),
            blockers=blockers,
            checklist_items=len(checklist),
            gate_handoff=gate_metadata.get("execution_case_gate_handoff"),
            gate_handoff_present=bool(gate_metadata.get("execution_case_gate_handoff")),
            execution_proof_handoff=execution_proof_handoff,
            execution_proof_handoff_present=bool(execution_proof_handoff),
            execution_proof_state=gate_metadata.get("execution_proof_state"),
            execution_proof_next_command=gate_metadata.get("execution_proof_next_command"),
            execution_proof_can_trust_execution=gate_metadata.get("execution_proof_can_trust_execution"),
            draft_only=True,
            requires_manual_send=True,
            loads_without_execution=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
            **execution_proof_aliases,
        )

        return ToolResult(
            "execution_case_review_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                found=True,
                case_id=int(case["id"]),
                request=case["request"],
                review_state=review_state,
                verdict=verdict,
                mission_state=case["mission_state"],
                go_no_go=case["go_no_go"],
                next_safe_command=next_safe_command,
                risk_signals=risk_signals,
                approval_required=approval_required,
                approval_queue_forecast=approval_queue_forecast,
                forecast_new_approvals=forecast_new_approvals,
                forecast_reused_approval_ids=forecast_reused_approval_ids,
                forecast_queue_before=forecast_queue_before,
                forecast_queue_after_if_sent=forecast_queue_after_if_sent,
                forecast_queue_delta_if_sent=forecast_queue_delta_if_sent,
                pending_approvals=len(pending),
                recent_failed_runs=recent_failed_run_count,
                recent_approval_held_runs=recent_approval_held_run_count,
                approval_held_review_command=approval_held_review_command,
                events=len(events),
                evidence_preview_gate_count=evidence_preview_gate_count,
                evidence_preview_ready_count=evidence_preview_ready_count,
                evidence_preview_blocked_count=evidence_preview_blocked_count,
                evidence_preview_verdicts=evidence_preview_verdicts,
                evidence_preview_latest_verdict=evidence_preview_latest_verdict,
                evidence_preview_latest_event_id=evidence_preview_latest_event_id,
                evidence_preview_latest_handoff=evidence_preview_latest_handoff,
                evidence_preview_latest_handoff_present=evidence_preview_latest_handoff_present,
                has_verification_evidence=has_verification_evidence,
                has_approval_evidence=has_approval_evidence,
                has_approval_chain_evidence=has_approval_chain_evidence,
                approval_chain_status=approval_chain_status,
                approval_evidence_ids=approval_evidence_ids,
                approval_linked_run_ids=approval_linked_run_ids,
                verification_receipt_ids=verification_receipt_ids,
                missing_verification_receipt_ids=missing_verification_receipt_ids,
                has_missing_verification_receipts=bool(missing_verification_receipt_ids),
                verified_approval_run_ids=verified_approval_run_ids,
                has_verified_approval_run=has_verified_approval_run,
                has_recovery_evidence=has_recovery_evidence,
                execution_health_recovery_closure_state=recovery_closure_state,
                execution_health_recovery_closure_missing=recovery_closure_missing,
                execution_health_recovery_closure_missing_count=len(recovery_closure_missing),
                execution_health_recovery_closure_required_commands=recovery_closure_required_commands,
                execution_health_recovery_closure_next_required_command=recovery_closure_next_required_command,
                execution_health_recovery_closure_checklist_command="recovery closure checklist" if recovery_closure_blocks_completion_claim else "",
                execution_health_recovery_closure_should_open_checklist=bool(recovery_closure_blocks_completion_claim),
                execution_health_recovery_closure_proof_queue=recovery_closure_proof_queue,
                execution_health_recovery_closure_proof_queue_count=len(recovery_closure_proof_queue),
                execution_health_recovery_closure_next_proof_command=recovery_closure_next_proof_command,
                execution_health_recovery_closure_blocks_completion_claim=recovery_closure_blocks_completion_claim,
                execution_health_recovery_closure_target_run_id=recovery_closure_target_run_id,
                execution_health_recovery_closure_target_tool_name=recovery_closure_target_tool_name,
                execution_health_recovery_closure_target_verification_receipts=recovery_closure_target_verification_receipts,
                execution_health_recovery_closure_target_recovery_packets=recovery_closure_target_recovery_packets,
                execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure_target_after_action_learning_packets,
                execution_learning_state=learning_state,
                execution_learning_blocks_completion_claim=learning_blocks_completion_claim,
                execution_learning_target_run_id=learning_target_run_id,
                execution_learning_target_tool_name=learning_target_tool_name,
                execution_learning_missing=learning_missing,
                execution_learning_missing_count=len(learning_missing),
                execution_learning_required_commands=learning_required_commands,
                execution_learning_next_required_command=learning_next_required_command,
                execution_learning_proof_queue=learning_proof_queue,
                execution_learning_proof_queue_count=len(learning_proof_queue),
                execution_learning_next_proof_command=learning_next_proof_command,
                missing_case_proofs=missing_case_proofs,
                next_case_proof_commands=next_case_proof_commands,
                next_case_required_command=next_case_proof_commands[0] if next_case_proof_commands else next_safe_command,
                next_case_proof_command=next_case_proof_commands[0] if next_case_proof_commands else next_safe_command,
                mission_command_queue=mission_command_queue,
                mission_command_count=len(mission_command_queue),
                initial_governor_command=gate_metadata.get("initial_governor_command", ""),
                proof_bundle_command=gate_metadata.get("proof_bundle_command", ""),
                acceptance_command=gate_metadata.get("acceptance_command", ""),
                audit_command=gate_metadata.get("audit_command", ""),
                recovery_command=gate_metadata.get("recovery_command", ""),
                learning_command=gate_metadata.get("learning_command", ""),
                blockers=blockers,
                checklist_items=len(checklist),
                execution_proof_handoff=execution_proof_handoff,
                execution_proof_handoff_present=bool(execution_proof_handoff),
                execution_proof_state=gate_metadata.get("execution_proof_state"),
                execution_proof_next_command=gate_metadata.get("execution_proof_next_command"),
                execution_proof_can_trust_execution=gate_metadata.get("execution_proof_can_trust_execution"),
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                **execution_proof_aliases,
                execution_case_review_handoff_ready=True,
                execution_case_review_handoff=execution_case_review_handoff,
            ),
        )

    def execution_case_closure_packet(args: dict[str, Any]) -> ToolResult:
        case, case_error = _resolve_execution_case_arg("execution_case_closure_packet", args)
        if case_error is not None:
            return case_error

        if case is None:
            gate_defaults = _execution_case_no_case_gate_defaults()
            gate_handoff = _safe_metadata(
                source="execution_case_gate",
                execution_case_gate_handoff_ready=True,
                handoff_ready=True,
                **gate_defaults,
            )
            review_defaults = {
                **gate_defaults,
                "review_state": "NO_CASE",
                "checklist_items": 0,
                "gate_handoff": gate_handoff,
                "gate_handoff_present": False,
                "execution_case_gate_handoff_ready": True,
                "draft_only": True,
                "requires_manual_send": True,
                "loads_without_execution": True,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            }
            review_handoff = _safe_metadata(
                source="execution_case_review_packet",
                execution_case_review_handoff_ready=True,
                handoff_ready=True,
                **review_defaults,
            )
            closure_defaults = {
                **gate_defaults,
                "closure_verdict": "CASE_CLOSURE_NOT_FOUND",
                "gate_verdict": "CASE_NOT_FOUND",
                "review_state": "NO_CASE",
                "case_closure_ready": False,
                "case_closure_blocks_completion_claim": True,
                "missing_case_proof_count": 1,
                "closure_proof_queue": ["save execution case: <order>"],
                "closure_proof_queue_count": 1,
                "next_closure_proof_command": "save execution case: <order>",
                "reason": "missing_case",
                "gate_handoff": gate_handoff,
                "gate_handoff_present": False,
                "review_handoff": review_handoff,
                "review_handoff_present": False,
                "execution_case_gate_handoff_ready": True,
                "execution_case_review_handoff_ready": True,
                "draft_only": True,
                "requires_manual_send": True,
                "loads_without_execution": True,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            }
            execution_case_closure_handoff = _safe_metadata(
                source="execution_case_closure_packet",
                execution_case_closure_handoff_ready=True,
                handoff_ready=True,
                **closure_defaults,
            )
            return ToolResult(
                "execution_case_closure_packet",
                True,
                "Jarvis execution case closure packet:\nNo saved execution case found. Save one first with `save execution case: <order>`.",
                _safe_metadata(
                    **closure_defaults,
                    execution_case_closure_handoff_ready=True,
                    execution_case_closure_handoff=execution_case_closure_handoff,
                ),
            )

        gate = execution_case_gate({"case_id": int(case["id"])})
        review = execution_case_review_packet({"case_id": int(case["id"])})
        gate_metadata = dict(gate.metadata)
        review_metadata = dict(review.metadata)
        execution_proof_handoff = dict(gate_metadata.get("execution_proof_handoff") or {})
        execution_proof_aliases = _execution_proof_handoff_aliases(execution_proof_handoff)
        gate_verdict = str(gate_metadata.get("verdict") or "UNKNOWN")
        review_state = str(review_metadata.get("review_state") or "UNKNOWN")
        missing_case_proofs = list(gate_metadata.get("missing_case_proofs") or [])
        next_case_proof_commands = list(gate_metadata.get("next_case_proof_commands") or [])
        evidence_preview_gate_count = _metadata_int(gate_metadata.get("evidence_preview_gate_count"))
        evidence_preview_ready_count = _metadata_int(gate_metadata.get("evidence_preview_ready_count"))
        evidence_preview_blocked_count = _metadata_int(gate_metadata.get("evidence_preview_blocked_count"))
        evidence_preview_verdicts = list(gate_metadata.get("evidence_preview_verdicts") or [])
        evidence_preview_latest_verdict = str(gate_metadata.get("evidence_preview_latest_verdict") or "")
        evidence_preview_latest_event_id = gate_metadata.get("evidence_preview_latest_event_id")
        evidence_preview_latest_handoff = gate_metadata.get("evidence_preview_latest_handoff")
        evidence_preview_latest_handoff = (
            evidence_preview_latest_handoff if isinstance(evidence_preview_latest_handoff, dict) else {}
        )
        evidence_preview_latest_handoff_present = _metadata_bool(gate_metadata.get("evidence_preview_latest_handoff_present"))
        recovery_proof_queue = list(gate_metadata.get("execution_health_recovery_closure_proof_queue") or [])
        learning_proof_queue = list(gate_metadata.get("execution_learning_proof_queue") or [])
        recent_failed_run_count = _metadata_int(gate_metadata.get("recent_failed_runs"))
        recent_approval_held_run_count = _metadata_int(gate_metadata.get("recent_approval_held_runs"))
        approval_held_review_command = str(gate_metadata.get("approval_held_review_command") or "")
        closure_proof_queue: list[str] = []
        _append_unique(closure_proof_queue, next_case_proof_commands)
        _append_unique(closure_proof_queue, recovery_proof_queue)
        _append_unique(closure_proof_queue, learning_proof_queue)
        if approval_held_review_command:
            _append_unique(closure_proof_queue, [approval_held_review_command])
        if gate_verdict != "CASE_READY_FOR_HUMAN_REVIEW":
            _append_unique(closure_proof_queue, [f"execution case gate {case['id']}"])
        else:
            _append_unique(closure_proof_queue, [f"execution case review {case['id']}"])

        case_closure_ready = (
            gate_verdict == "CASE_READY_FOR_HUMAN_REVIEW"
            and review_state == "READY_FOR_HUMAN_REVIEW"
            and not missing_case_proofs
            and not _metadata_bool(gate_metadata.get("execution_health_recovery_closure_blocks_completion_claim"))
            and not _metadata_bool(gate_metadata.get("execution_learning_blocks_completion_claim"))
            and not _metadata_bool(gate_metadata.get("has_missing_verification_receipts"))
            and not _metadata_bool(gate_metadata.get("has_missing_runtime_trace_receipts"))
        )
        if case_closure_ready:
            closure_verdict = "CASE_CLOSURE_READY_FOR_HUMAN_REVIEW"
        elif gate_verdict == "CASE_NOT_FOUND":
            closure_verdict = "CASE_CLOSURE_NOT_FOUND"
        else:
            closure_verdict = "CASE_CLOSURE_BLOCKED"
        case_closure_blocks_completion_claim = not case_closure_ready

        checklist = [
            (
                "mission",
                str(gate_metadata.get("go_no_go") or "") in {"GO", "PREFLIGHT_OK"}
                and "HOLD" not in str(gate_metadata.get("mission_state") or ""),
                gate_metadata.get("next_safe_command") or "dispatch decision before execution",
            ),
            ("case evidence", int(gate_metadata.get("events") or 0) > 0, f"case evidence {case['id']}: verification receipt <id> confirmed expected outcome"),
            ("verification", _metadata_bool(gate_metadata.get("has_verification_evidence")), f"case evidence {case['id']}: verification receipt <id> confirmed expected outcome"),
            ("receipt targets", not (gate_metadata.get("missing_verification_receipt_ids") or gate_metadata.get("missing_runtime_trace_receipt_message_ids")), "replace or remove missing receipt references before review"),
            ("approval evidence", not _metadata_bool(gate_metadata.get("approval_required")) or _metadata_bool(gate_metadata.get("has_approval_evidence")), f"case evidence {case['id']}: approval readiness <id>, approval packet <id>, and approval chain proof <id> reviewed"),
            ("approval chain", not _metadata_bool(gate_metadata.get("approval_required")) or _metadata_bool(gate_metadata.get("has_approval_chain_evidence")), "approval chain proof <approval id>"),
            ("approved run verification", not _metadata_bool(gate_metadata.get("approval_required")) or _metadata_bool(gate_metadata.get("has_verified_approval_run")), f"case evidence {case['id']}: verification receipt <approved run id> confirmed expected outcome"),
            ("approval-held review", recent_approval_held_run_count == 0, approval_held_review_command or "approval history"),
            ("recovery evidence", recent_failed_run_count == 0 or _metadata_bool(gate_metadata.get("has_recovery_evidence")), f"case evidence {case['id']}: execution recovery packet <id> reviewed stop condition"),
            ("recovery closure", not _metadata_bool(gate_metadata.get("execution_health_recovery_closure_blocks_completion_claim")), gate_metadata.get("execution_health_recovery_closure_next_proof_command") or "execution health report"),
            ("execution learning", not _metadata_bool(gate_metadata.get("execution_learning_blocks_completion_claim")), gate_metadata.get("execution_learning_next_proof_command") or "execution learning closure <run id>"),
            ("human review packet", review_state == "READY_FOR_HUMAN_REVIEW", f"execution case review {case['id']}"),
        ]

        lines = [
            "Jarvis execution case closure packet:",
            "This read-only packet decides whether a saved execution case has closed its proof debt before any completion claim.",
            "",
            f"Case: #{case['id']}",
            f"Request: {case['request']}",
            f"Closure verdict: {closure_verdict}",
            f"Gate verdict: {gate_verdict}",
            f"Review state: {review_state}",
            f"Closure ready: {'yes' if case_closure_ready else 'no'}",
            f"Blocks completion claim: {'yes' if case_closure_blocks_completion_claim else 'no'}",
            "",
            "Closure checklist:",
        ]
        for name, ready, command in checklist:
            lines.append(f"- {name}: {'closed' if ready else 'open'}")
            if not ready:
                lines.append(f"  next: `{command}`")
        lines.extend(["", "Proof debt:"])
        if missing_case_proofs:
            lines.extend(f"- {item}" for item in missing_case_proofs)
        else:
            lines.append("- none from the case gate")
        lines.extend(
            [
                "",
                "Recovery and learning:",
                f"- recent failed/blocked runs: {recent_failed_run_count}",
                f"- recent approval-held runs: {recent_approval_held_run_count}",
                f"- approval-held review: `{approval_held_review_command}`" if approval_held_review_command else "- approval-held review: none",
                f"- recovery closure state: {gate_metadata.get('execution_health_recovery_closure_state') or 'unknown'}",
                f"- recovery missing: {', '.join(gate_metadata.get('execution_health_recovery_closure_missing') or []) if gate_metadata.get('execution_health_recovery_closure_missing') else 'none'}",
                f"- execution learning state: {gate_metadata.get('execution_learning_state') or 'unknown'}",
                f"- execution learning missing: {', '.join(gate_metadata.get('execution_learning_missing') or []) if gate_metadata.get('execution_learning_missing') else 'none'}",
                "",
                "Approval and receipt closure:",
                f"- approval required: {'yes' if gate_metadata.get('approval_required') else 'no'}",
                f"- evidence preflight events: {evidence_preview_gate_count}",
                f"- evidence preflight ready events: {evidence_preview_ready_count}",
                f"- latest evidence preflight verdict: {evidence_preview_latest_verdict or 'none'}",
                f"- approval chain: {gate_metadata.get('approval_chain_status') or 'missing'}",
                f"- verified approved runs: {', '.join(str(item) for item in gate_metadata.get('verified_approval_run_ids') or []) if gate_metadata.get('verified_approval_run_ids') else 'none'}",
                f"- missing verification/tool-run receipt targets: {', '.join(str(item) for item in gate_metadata.get('missing_verification_receipt_ids') or []) if gate_metadata.get('missing_verification_receipt_ids') else 'none'}",
                f"- missing runtime-trace receipt targets: {', '.join(str(item) for item in gate_metadata.get('missing_runtime_trace_receipt_message_ids') or []) if gate_metadata.get('missing_runtime_trace_receipt_message_ids') else 'none'}",
                "",
                "Mission command queue:",
            ]
        )
        mission_command_queue = list(gate_metadata.get("mission_command_queue") or [])
        if mission_command_queue:
            lines.extend(f"- `{command}`" for command in mission_command_queue)
        else:
            lines.append("- not stored for this case")
        lines.extend(["", "Closure proof queue:"])
        if closure_proof_queue:
            lines.extend(f"- `{command}`" for command in closure_proof_queue)
        else:
            lines.append("- none; case can move to human outcome review")
        next_closure_proof_command = closure_proof_queue[0] if closure_proof_queue else f"execution case review {case['id']}"
        lines.extend(
            [
                "",
                f"Next closure command: `{next_closure_proof_command}`",
                "",
                "Boundary:",
                "- This closure packet does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, write notes, write the database, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        execution_case_closure_handoff = _safe_metadata(
            source="execution_case_closure_packet",
            execution_case_closure_handoff_ready=True,
            handoff_ready=True,
            found=True,
            case_id=int(case["id"]),
            request=case["request"],
            closure_verdict=closure_verdict,
            gate_verdict=gate_verdict,
            review_state=review_state,
            case_closure_ready=case_closure_ready,
            case_closure_blocks_completion_claim=case_closure_blocks_completion_claim,
            missing_case_proofs=missing_case_proofs,
            missing_case_proof_count=len(missing_case_proofs),
            next_case_proof_commands=next_case_proof_commands,
            next_case_required_command=next_case_proof_commands[0] if next_case_proof_commands else "",
            next_case_proof_command=next_case_proof_commands[0] if next_case_proof_commands else "",
            closure_proof_queue=closure_proof_queue,
            closure_proof_queue_count=len(closure_proof_queue),
            next_closure_proof_command=next_closure_proof_command,
            mission_command_queue=mission_command_queue,
            mission_command_count=len(mission_command_queue),
            events=gate_metadata.get("events"),
            evidence_preview_gate_count=evidence_preview_gate_count,
            evidence_preview_ready_count=evidence_preview_ready_count,
            evidence_preview_blocked_count=evidence_preview_blocked_count,
            evidence_preview_verdicts=evidence_preview_verdicts,
            evidence_preview_latest_verdict=evidence_preview_latest_verdict,
            evidence_preview_latest_event_id=evidence_preview_latest_event_id,
            evidence_preview_latest_handoff=evidence_preview_latest_handoff,
            evidence_preview_latest_handoff_present=evidence_preview_latest_handoff_present,
            approval_required=gate_metadata.get("approval_required"),
            has_verification_evidence=gate_metadata.get("has_verification_evidence"),
            has_approval_evidence=gate_metadata.get("has_approval_evidence"),
            has_approval_chain_evidence=gate_metadata.get("has_approval_chain_evidence"),
            approval_chain_status=gate_metadata.get("approval_chain_status"),
            approval_evidence_ids=gate_metadata.get("approval_evidence_ids"),
            approval_linked_run_ids=gate_metadata.get("approval_linked_run_ids"),
            verification_receipt_ids=gate_metadata.get("verification_receipt_ids"),
            runtime_trace_receipt_message_ids=gate_metadata.get("runtime_trace_receipt_message_ids"),
            missing_verification_receipt_ids=gate_metadata.get("missing_verification_receipt_ids"),
            missing_runtime_trace_receipt_message_ids=gate_metadata.get("missing_runtime_trace_receipt_message_ids"),
            has_missing_verification_receipts=gate_metadata.get("has_missing_verification_receipts"),
            has_missing_runtime_trace_receipts=gate_metadata.get("has_missing_runtime_trace_receipts"),
            verified_approval_run_ids=gate_metadata.get("verified_approval_run_ids"),
            has_verified_approval_run=gate_metadata.get("has_verified_approval_run"),
            has_recovery_evidence=gate_metadata.get("has_recovery_evidence"),
            recent_failed_runs=recent_failed_run_count,
            recent_approval_held_runs=recent_approval_held_run_count,
            approval_held_review_command=approval_held_review_command,
            execution_health_recovery_closure_state=gate_metadata.get("execution_health_recovery_closure_state"),
            execution_health_recovery_closure_missing=gate_metadata.get("execution_health_recovery_closure_missing"),
            execution_health_recovery_closure_missing_count=gate_metadata.get("execution_health_recovery_closure_missing_count"),
            execution_health_recovery_closure_required_commands=gate_metadata.get("execution_health_recovery_closure_required_commands"),
            execution_health_recovery_closure_next_required_command=gate_metadata.get("execution_health_recovery_closure_next_required_command"),
            execution_health_recovery_closure_proof_queue=recovery_proof_queue,
            execution_health_recovery_closure_proof_queue_count=len(recovery_proof_queue),
            execution_health_recovery_closure_next_proof_command=gate_metadata.get("execution_health_recovery_closure_next_proof_command"),
            execution_health_recovery_closure_blocks_completion_claim=gate_metadata.get("execution_health_recovery_closure_blocks_completion_claim"),
            execution_learning_state=gate_metadata.get("execution_learning_state"),
            execution_learning_blocks_completion_claim=gate_metadata.get("execution_learning_blocks_completion_claim"),
            execution_learning_missing=gate_metadata.get("execution_learning_missing"),
            execution_learning_missing_count=gate_metadata.get("execution_learning_missing_count"),
            execution_learning_required_commands=gate_metadata.get("execution_learning_required_commands"),
            execution_learning_next_required_command=gate_metadata.get("execution_learning_next_required_command"),
            execution_learning_proof_queue=learning_proof_queue,
            execution_learning_proof_queue_count=len(learning_proof_queue),
            execution_learning_next_proof_command=gate_metadata.get("execution_learning_next_proof_command"),
            blockers=gate_metadata.get("blockers"),
            gate_handoff=gate_metadata.get("execution_case_gate_handoff"),
            gate_handoff_present=bool(gate_metadata.get("execution_case_gate_handoff")),
            review_handoff=review_metadata.get("execution_case_review_handoff"),
            review_handoff_present=bool(review_metadata.get("execution_case_review_handoff")),
            execution_proof_handoff=execution_proof_handoff,
            execution_proof_handoff_present=bool(execution_proof_handoff),
            execution_proof_state=gate_metadata.get("execution_proof_state"),
            execution_proof_next_command=gate_metadata.get("execution_proof_next_command"),
            execution_proof_can_trust_execution=gate_metadata.get("execution_proof_can_trust_execution"),
            draft_only=True,
            requires_manual_send=True,
            loads_without_execution=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
            **execution_proof_aliases,
        )

        return ToolResult(
            "execution_case_closure_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                found=True,
                case_id=int(case["id"]),
                request=case["request"],
                closure_verdict=closure_verdict,
                gate_verdict=gate_verdict,
                review_state=review_state,
                case_closure_ready=case_closure_ready,
                case_closure_blocks_completion_claim=case_closure_blocks_completion_claim,
                missing_case_proofs=missing_case_proofs,
                missing_case_proof_count=len(missing_case_proofs),
                next_case_proof_commands=next_case_proof_commands,
                next_case_required_command=next_case_proof_commands[0] if next_case_proof_commands else "",
                next_case_proof_command=next_case_proof_commands[0] if next_case_proof_commands else "",
                closure_proof_queue=closure_proof_queue,
                closure_proof_queue_count=len(closure_proof_queue),
                next_closure_proof_command=next_closure_proof_command,
                mission_command_queue=mission_command_queue,
                mission_command_count=len(mission_command_queue),
                events=gate_metadata.get("events"),
                evidence_preview_gate_count=evidence_preview_gate_count,
                evidence_preview_ready_count=evidence_preview_ready_count,
                evidence_preview_blocked_count=evidence_preview_blocked_count,
                evidence_preview_verdicts=evidence_preview_verdicts,
                evidence_preview_latest_verdict=evidence_preview_latest_verdict,
                evidence_preview_latest_event_id=evidence_preview_latest_event_id,
                evidence_preview_latest_handoff=evidence_preview_latest_handoff,
                evidence_preview_latest_handoff_present=evidence_preview_latest_handoff_present,
                approval_required=gate_metadata.get("approval_required"),
                has_verification_evidence=gate_metadata.get("has_verification_evidence"),
                has_approval_evidence=gate_metadata.get("has_approval_evidence"),
                has_approval_chain_evidence=gate_metadata.get("has_approval_chain_evidence"),
                approval_chain_status=gate_metadata.get("approval_chain_status"),
                approval_evidence_ids=gate_metadata.get("approval_evidence_ids"),
                approval_linked_run_ids=gate_metadata.get("approval_linked_run_ids"),
                verification_receipt_ids=gate_metadata.get("verification_receipt_ids"),
                runtime_trace_receipt_message_ids=gate_metadata.get("runtime_trace_receipt_message_ids"),
                missing_verification_receipt_ids=gate_metadata.get("missing_verification_receipt_ids"),
                missing_runtime_trace_receipt_message_ids=gate_metadata.get("missing_runtime_trace_receipt_message_ids"),
                has_missing_verification_receipts=gate_metadata.get("has_missing_verification_receipts"),
                has_missing_runtime_trace_receipts=gate_metadata.get("has_missing_runtime_trace_receipts"),
                verified_approval_run_ids=gate_metadata.get("verified_approval_run_ids"),
                has_verified_approval_run=gate_metadata.get("has_verified_approval_run"),
                has_recovery_evidence=gate_metadata.get("has_recovery_evidence"),
                recent_failed_runs=recent_failed_run_count,
                recent_approval_held_runs=recent_approval_held_run_count,
                approval_held_review_command=approval_held_review_command,
                execution_health_recovery_closure_state=gate_metadata.get("execution_health_recovery_closure_state"),
                execution_health_recovery_closure_missing=gate_metadata.get("execution_health_recovery_closure_missing"),
                execution_health_recovery_closure_missing_count=gate_metadata.get("execution_health_recovery_closure_missing_count"),
                execution_health_recovery_closure_required_commands=gate_metadata.get("execution_health_recovery_closure_required_commands"),
                execution_health_recovery_closure_next_required_command=gate_metadata.get("execution_health_recovery_closure_next_required_command"),
                execution_health_recovery_closure_checklist_command=gate_metadata.get("execution_health_recovery_closure_checklist_command"),
                execution_health_recovery_closure_should_open_checklist=gate_metadata.get("execution_health_recovery_closure_should_open_checklist"),
                execution_health_recovery_closure_proof_queue=recovery_proof_queue,
                execution_health_recovery_closure_proof_queue_count=len(recovery_proof_queue),
                execution_health_recovery_closure_next_proof_command=gate_metadata.get("execution_health_recovery_closure_next_proof_command"),
                execution_health_recovery_closure_blocks_completion_claim=gate_metadata.get("execution_health_recovery_closure_blocks_completion_claim"),
                execution_learning_state=gate_metadata.get("execution_learning_state"),
                execution_learning_blocks_completion_claim=gate_metadata.get("execution_learning_blocks_completion_claim"),
                execution_learning_missing=gate_metadata.get("execution_learning_missing"),
                execution_learning_missing_count=gate_metadata.get("execution_learning_missing_count"),
                execution_learning_required_commands=gate_metadata.get("execution_learning_required_commands"),
                execution_learning_next_required_command=gate_metadata.get("execution_learning_next_required_command"),
                execution_learning_proof_queue=learning_proof_queue,
                execution_learning_proof_queue_count=len(learning_proof_queue),
                execution_learning_next_proof_command=gate_metadata.get("execution_learning_next_proof_command"),
                blockers=gate_metadata.get("blockers"),
                execution_proof_handoff=execution_proof_handoff,
                execution_proof_handoff_present=bool(execution_proof_handoff),
                execution_proof_state=gate_metadata.get("execution_proof_state"),
                execution_proof_next_command=gate_metadata.get("execution_proof_next_command"),
                execution_proof_can_trust_execution=gate_metadata.get("execution_proof_can_trust_execution"),
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                **execution_proof_aliases,
                execution_case_closure_handoff_ready=True,
                execution_case_closure_handoff=execution_case_closure_handoff,
            ),
        )

    def execution_case_timeline(args: dict[str, Any]) -> ToolResult:
        case, case_error = _resolve_execution_case_arg("execution_case_timeline", args)
        if case_error is not None:
            return case_error

        if case is None:
            execution_case_timeline_handoff = _safe_metadata(
                source="execution_case_timeline",
                execution_case_timeline_handoff_ready=True,
                handoff_ready=True,
                found=False,
                case_id=None,
                request="",
                review_state="NO_CASE",
                verdict="CASE_NOT_FOUND",
                timeline_items=0,
                evidence_event_count=0,
                recent_run_count=0,
                events=0,
                recent_runs=0,
                missing_stages=[],
                missing_stage_count=0,
                approval_required=False,
                approval_chain_status="missing",
                forecast_new_approvals=0,
                forecast_reused_approval_ids=[],
                forecast_queue_after_if_sent=0,
                execution_health_recovery_closure_state="unknown",
                execution_health_recovery_closure_next_proof_command="",
                execution_learning_state="unknown",
                execution_learning_next_proof_command="",
                mission_command_queue=[],
                mission_command_count=0,
                evidence_preview_gate_count=0,
                evidence_preview_ready_count=0,
                evidence_preview_blocked_count=0,
                evidence_preview_verdicts=[],
                evidence_preview_latest_verdict="",
                evidence_preview_latest_event_id=None,
                evidence_preview_latest_handoff={},
                evidence_preview_latest_handoff_present=False,
                gate_handoff={},
                gate_handoff_present=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
            )
            return ToolResult(
                "execution_case_timeline",
                True,
                "Jarvis execution case timeline:\nNo saved execution case found. Save one first with `save execution case: <order>`.",
                _safe_metadata(
                    found=False,
                    case_id=None,
                    events=0,
                    recent_runs=0,
                    verdict="CASE_NOT_FOUND",
                    evidence_preview_gate_count=0,
                    evidence_preview_ready_count=0,
                    evidence_preview_blocked_count=0,
                    evidence_preview_verdicts=[],
                    evidence_preview_latest_verdict="",
                    evidence_preview_latest_event_id=None,
                    evidence_preview_latest_handoff={},
                    evidence_preview_latest_handoff_present=False,
                    draft_only=True,
                    requires_manual_send=True,
                    loads_without_execution=True,
                    authorizes_execution=False,
                    authorizes_completion_claim=False,
                    approval_granted=False,
                    execution_case_timeline_handoff_ready=True,
                    execution_case_timeline_handoff=execution_case_timeline_handoff,
                ),
            )

        try:
            risk_signals = list(json.loads(case["risk_signals"] or "[]"))
        except Exception:
            risk_signals = []
        events_desc = store.list_execution_case_events(int(case["id"]), limit=50)
        events = list(reversed(events_desc))
        gate = execution_case_gate({"case_id": int(case["id"])})
        gate_metadata = dict(gate.metadata)
        execution_proof_handoff = dict(gate_metadata.get("execution_proof_handoff") or {})
        execution_proof_aliases = _execution_proof_handoff_aliases(execution_proof_handoff)
        verdict = str(gate_metadata.get("verdict") or "UNKNOWN")
        approval_required = _metadata_bool(gate_metadata.get("approval_required"))
        pending_approvals = int(gate_metadata.get("pending_approvals") or 0)
        blockers = int(gate_metadata.get("blockers") or 0)
        has_verification_evidence = _metadata_bool(gate_metadata.get("has_verification_evidence"))
        has_approval_evidence = _metadata_bool(gate_metadata.get("has_approval_evidence"))
        has_approval_chain_evidence = _metadata_bool(gate_metadata.get("has_approval_chain_evidence"))
        approval_chain_status = str(gate_metadata.get("approval_chain_status") or "missing")
        approval_evidence_ids = list(gate_metadata.get("approval_evidence_ids") or [])
        approval_linked_run_ids = list(gate_metadata.get("approval_linked_run_ids") or [])
        verification_receipt_ids = list(gate_metadata.get("verification_receipt_ids") or [])
        runtime_trace_receipt_message_ids = list(gate_metadata.get("runtime_trace_receipt_message_ids") or [])
        missing_verification_receipt_ids = list(gate_metadata.get("missing_verification_receipt_ids") or [])
        missing_runtime_trace_receipt_message_ids = list(
            gate_metadata.get("missing_runtime_trace_receipt_message_ids") or []
        )
        verified_approval_run_ids = list(gate_metadata.get("verified_approval_run_ids") or [])
        has_verified_approval_run = _metadata_bool(gate_metadata.get("has_verified_approval_run"))
        has_recovery_evidence = _metadata_bool(gate_metadata.get("has_recovery_evidence"))
        recovery_closure_state = str(gate_metadata.get("execution_health_recovery_closure_state") or "unknown")
        recovery_closure_missing = list(gate_metadata.get("execution_health_recovery_closure_missing") or [])
        recovery_closure_required_commands = list(gate_metadata.get("execution_health_recovery_closure_required_commands") or [])
        recovery_closure_next_required_command = str(gate_metadata.get("execution_health_recovery_closure_next_required_command") or "")
        recovery_closure_proof_queue = list(gate_metadata.get("execution_health_recovery_closure_proof_queue") or [])
        recovery_closure_next_proof_command = str(gate_metadata.get("execution_health_recovery_closure_next_proof_command") or "")
        recovery_closure_blocks_completion_claim = _metadata_bool(gate_metadata.get("execution_health_recovery_closure_blocks_completion_claim"))
        recovery_closure_target_run_id = gate_metadata.get("execution_health_recovery_closure_target_run_id")
        recovery_closure_target_tool_name = str(gate_metadata.get("execution_health_recovery_closure_target_tool_name") or "")
        recovery_closure_target_verification_receipts = _metadata_int(gate_metadata.get("execution_health_recovery_closure_target_verification_receipts"))
        recovery_closure_target_recovery_packets = _metadata_int(gate_metadata.get("execution_health_recovery_closure_target_recovery_packets"))
        recovery_closure_target_after_action_learning_packets = _metadata_int(gate_metadata.get("execution_health_recovery_closure_target_after_action_learning_packets"))
        learning_state = str(gate_metadata.get("execution_learning_state") or "unknown")
        learning_missing = list(gate_metadata.get("execution_learning_missing") or [])
        learning_required_commands = list(gate_metadata.get("execution_learning_required_commands") or [])
        learning_next_required_command = str(gate_metadata.get("execution_learning_next_required_command") or "")
        learning_proof_queue = list(gate_metadata.get("execution_learning_proof_queue") or [])
        learning_next_proof_command = str(gate_metadata.get("execution_learning_next_proof_command") or "")
        learning_blocks_completion_claim = _metadata_bool(gate_metadata.get("execution_learning_blocks_completion_claim"))
        learning_target_run_id = gate_metadata.get("execution_learning_target_run_id")
        learning_target_tool_name = str(gate_metadata.get("execution_learning_target_tool_name") or "")
        missing_case_proofs = list(gate_metadata.get("missing_case_proofs") or [])
        next_case_proof_commands = list(gate_metadata.get("next_case_proof_commands") or [])
        mission_command_queue = list(gate_metadata.get("mission_command_queue") or [])
        approval_queue_forecast = list(gate_metadata.get("approval_queue_forecast") or [])
        forecast_new_approvals = _metadata_int(gate_metadata.get("forecast_new_approvals"))
        forecast_reused_approval_ids = list(gate_metadata.get("forecast_reused_approval_ids") or [])
        forecast_queue_before = _metadata_int(gate_metadata.get("forecast_queue_before"), pending_approvals)
        forecast_queue_after_if_sent = _metadata_int(gate_metadata.get("forecast_queue_after_if_sent"), forecast_queue_before)
        forecast_queue_delta_if_sent = _metadata_int(gate_metadata.get("forecast_queue_delta_if_sent"), forecast_new_approvals)
        evidence_preview_gate_count = _metadata_int(gate_metadata.get("evidence_preview_gate_count"))
        evidence_preview_ready_count = _metadata_int(gate_metadata.get("evidence_preview_ready_count"))
        evidence_preview_blocked_count = _metadata_int(gate_metadata.get("evidence_preview_blocked_count"))
        evidence_preview_verdicts = list(gate_metadata.get("evidence_preview_verdicts") or [])
        evidence_preview_latest_verdict = str(gate_metadata.get("evidence_preview_latest_verdict") or "")
        evidence_preview_latest_event_id = gate_metadata.get("evidence_preview_latest_event_id")
        evidence_preview_latest_handoff = gate_metadata.get("evidence_preview_latest_handoff")
        evidence_preview_latest_handoff = (
            evidence_preview_latest_handoff if isinstance(evidence_preview_latest_handoff, dict) else {}
        )
        evidence_preview_latest_handoff_present = _metadata_bool(gate_metadata.get("evidence_preview_latest_handoff_present"))
        review_state = "READY_FOR_HUMAN_REVIEW" if verdict == "CASE_READY_FOR_HUMAN_REVIEW" else "REVIEW_REQUIRED"
        if "APPROVAL" in verdict:
            review_state = "APPROVAL_REVIEW_REQUIRED"
        elif "RECOVERY" in verdict:
            review_state = "RECOVERY_REVIEW_REQUIRED"
        elif verdict in {"CASE_NOT_FOUND", "UNKNOWN"}:
            review_state = "NO_CASE"
        elif verdict != "CASE_READY_FOR_HUMAN_REVIEW":
            review_state = "EVIDENCE_REVIEW_REQUIRED"

        recent_runs = store.recent_tool_runs(limit=20)
        failed_runs, approval_held_runs = _recent_tool_run_attention_buckets(recent_runs)
        approval_held_review_command = _approval_review_command(approval_held_runs) if approval_held_runs else ""
        event_text = " ".join(
            f"{event['event_type']} {event['summary']} {event['receipt_kind']} {event['receipt_id']}"
            for event in events
        ).lower()
        has_execution_evidence = any(
            token in event_text
            for token in ("executed", "approved run", "tool run", "run result", "action completed", "changed", "created", "updated")
        )
        verification_stage_ok = has_verification_evidence and ((not approval_required) or has_verified_approval_run)
        if approval_required and has_verified_approval_run:
            verification_stage_detail = "verification/test/receipt evidence linked to approved run"
        elif approval_required and has_verification_evidence:
            verification_stage_detail = "verification evidence present but not linked to the approved risky run"
        elif has_verification_evidence:
            verification_stage_detail = "verification/test/receipt evidence present"
        else:
            verification_stage_detail = "verification/test/receipt evidence missing"
        lifecycle_stages = [
            (
                "created",
                True,
                "case row and note anchor exist",
            ),
            (
                "approval",
                (not approval_required) or has_approval_chain_evidence,
                "not required" if not approval_required else "approval chain proven" if has_approval_chain_evidence else f"approval chain {approval_chain_status}",
            ),
            (
                "execution",
                has_execution_evidence,
                "execution/result evidence present" if has_execution_evidence else "no execution/result event attached",
            ),
            (
                "verification",
                verification_stage_ok,
                verification_stage_detail,
            ),
            (
                "recovery",
                not failed_runs or has_recovery_evidence,
                "not required by recent runs" if not failed_runs else "recovery/audit evidence present" if has_recovery_evidence else "recent failures need recovery/audit evidence",
            ),
            (
                "final review",
                verdict == "CASE_READY_FOR_HUMAN_REVIEW",
                "ready for human review" if verdict == "CASE_READY_FOR_HUMAN_REVIEW" else f"blocked by {verdict}",
            ),
        ]
        complete_stages = [name for name, ok, _detail in lifecycle_stages if ok]
        missing_stages = [name for name, ok, _detail in lifecycle_stages if not ok]
        timeline_lines = [
            f"- {case['created_at']} | case saved | mission={case['mission_state']} | go_no_go={case['go_no_go']}",
        ]
        for event in events:
            receipt = ""
            if event["receipt_kind"] or event["receipt_id"]:
                receipt = f" | receipt={event['receipt_kind'] or 'receipt'} {event['receipt_id']}".rstrip()
            timeline_lines.append(
                f"- {event['created_at']} | event #{event['id']} | {event['event_type']}{receipt} | {_short(event['summary'], limit=180)}"
            )
        if len(timeline_lines) == 1:
            timeline_lines.append("- no evidence events attached yet")

        run_lines = []
        for run in recent_runs[:5]:
            state = "ok" if run["ok"] else "approval-held" if _is_approval_held_tool_run(run) else "blocked"
            approval = f" approval={run['approval_id']}" if run["approval_id"] else ""
            run_lines.append(f"- #{run['id']} {run['created_at']} | {run['tool_name']} | {state} | risk={run['risk']}{approval}")

        next_safe_command = str(gate_metadata.get("next_safe_command") or case["next_command"] or "")
        lines = [
            "Jarvis execution case timeline:",
            "This read-only timeline shows a saved case from creation through evidence events and current gate posture.",
            "",
            f"Case: #{case['id']}",
            f"Request: {case['request']}",
            f"Review state: {review_state}",
            f"Gate verdict: {verdict}",
            f"Risk signals: {', '.join(risk_signals) if risk_signals else 'none'}",
            f"Approval chain: {approval_chain_status}",
            f"Verification/tool-run receipt ids: {', '.join(str(item) for item in verification_receipt_ids) if verification_receipt_ids else 'none'}",
            f"Runtime-trace receipt message ids: {', '.join(str(item) for item in runtime_trace_receipt_message_ids) if runtime_trace_receipt_message_ids else 'none'}",
            f"Missing verification/tool-run receipt targets: {', '.join(str(item) for item in missing_verification_receipt_ids) if missing_verification_receipt_ids else 'none'}",
            f"Missing runtime-trace receipt targets: {', '.join(str(item) for item in missing_runtime_trace_receipt_message_ids) if missing_runtime_trace_receipt_message_ids else 'none'}",
            f"Verified approved runs: {', '.join(str(item) for item in verified_approval_run_ids) if verified_approval_run_ids else 'none'}",
            f"Approval forecast: {forecast_new_approvals} new, {len(forecast_reused_approval_ids)} reused, queue after {forecast_queue_after_if_sent}",
            f"Evidence preflight: {evidence_preview_ready_count}/{evidence_preview_gate_count} ready, {evidence_preview_blocked_count} blocked",
            f"Evidence preflight verdict: {evidence_preview_latest_verdict or 'none'}",
            f"Next safe command: `{next_safe_command}`",
            "",
                "Execution health recovery closure:",
            f"- recent failed/blocked runs: {len(failed_runs)}",
            f"- recent approval-held runs: {len(approval_held_runs)}",
            f"- approval-held review: `{approval_held_review_command}`" if approval_held_review_command else "- approval-held review: none",
            f"- state: {recovery_closure_state}",
            f"- target run: #{recovery_closure_target_run_id} {recovery_closure_target_tool_name}" if recovery_closure_target_run_id else "- target run: none",
            f"- target verification receipts: {recovery_closure_target_verification_receipts}",
            f"- target recovery packets: {recovery_closure_target_recovery_packets}",
            f"- target after-action learning packets: {recovery_closure_target_after_action_learning_packets}",
            f"- missing: {', '.join(recovery_closure_missing) if recovery_closure_missing else 'none'}",
            f"- blocks completion claim: {'yes' if recovery_closure_blocks_completion_claim else 'no'}",
            f"- next required command: `{recovery_closure_next_required_command}`" if recovery_closure_next_required_command else "- next required command: none",
            "",
            "Execution learning debt:",
            f"- state: {learning_state}",
            f"- target run: #{learning_target_run_id} {learning_target_tool_name}" if learning_target_run_id else "- target run: none",
            f"- missing: {', '.join(learning_missing) if learning_missing else 'none'}",
            f"- blocks completion claim: {'yes' if learning_blocks_completion_claim else 'no'}",
            f"- next required command: `{learning_next_required_command}`" if learning_next_required_command else "- next required command: none",
            "",
            "Timeline:",
            *timeline_lines,
            "",
            "Lifecycle checklist:",
            *[f"- {name}: {'present' if ok else 'missing'} - {detail}" for name, ok, detail in lifecycle_stages],
            "",
            "Case proof gaps:",
        ]
        if missing_case_proofs:
            lines.extend(f"- {item}" for item in missing_case_proofs)
        else:
            lines.append("- none")
        if next_case_proof_commands:
            lines.extend(["", "Next required commands:"])
            lines.extend(f"- `{command}`" for command in next_case_proof_commands)
        lines.extend(["", "Mission command queue:"])
        if mission_command_queue:
            lines.extend(f"- `{command}`" for command in mission_command_queue)
        else:
            lines.append("- not stored for this case")
        lines.extend(
            [
                "",
            "Recent runtime context:",
            ]
        )
        lines.extend(run_lines if run_lines else ["- no recent tool runs recorded"])
        lines.extend(
            [
                "",
                "Use this before claiming completion:",
                "- creation, approval, execution, verification, recovery, and final review should be visible as ordered evidence.",
                "- missing evidence means the case is not ready for completion claims.",
                "",
                "Boundary:",
                "- This timeline is read-only. It does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        execution_case_timeline_handoff = _safe_metadata(
            source="execution_case_timeline",
            execution_case_timeline_handoff_ready=True,
            handoff_ready=True,
            found=True,
            case_id=int(case["id"]),
            request=case["request"],
            review_state=review_state,
            verdict=verdict,
            mission_state=case["mission_state"],
            go_no_go=case["go_no_go"],
            next_safe_command=next_safe_command,
            risk_signals=risk_signals,
            risk_signal_count=len(risk_signals),
            timeline_items=len(events) + 1,
            evidence_event_count=len(events),
            recent_run_count=len(recent_runs),
            events=len(events),
            recent_runs=len(recent_runs),
            pending_approvals=pending_approvals,
            blockers=blockers,
            approval_required=approval_required,
            approval_queue_forecast=approval_queue_forecast,
            forecast_new_approvals=forecast_new_approvals,
            forecast_reused_approval_ids=forecast_reused_approval_ids,
            forecast_reused_approval_count=len(forecast_reused_approval_ids),
            forecast_queue_before=forecast_queue_before,
            forecast_queue_after_if_sent=forecast_queue_after_if_sent,
            forecast_queue_delta_if_sent=forecast_queue_delta_if_sent,
            has_approval_evidence=has_approval_evidence,
            has_approval_chain_evidence=has_approval_chain_evidence,
            approval_chain_status=approval_chain_status,
            approval_evidence_ids=approval_evidence_ids,
            approval_linked_run_ids=approval_linked_run_ids,
            verification_receipt_ids=verification_receipt_ids,
            runtime_trace_receipt_message_ids=runtime_trace_receipt_message_ids,
            missing_verification_receipt_ids=missing_verification_receipt_ids,
            missing_runtime_trace_receipt_message_ids=missing_runtime_trace_receipt_message_ids,
            has_missing_verification_receipts=bool(missing_verification_receipt_ids),
            has_missing_runtime_trace_receipts=bool(missing_runtime_trace_receipt_message_ids),
            verified_approval_run_ids=verified_approval_run_ids,
            has_verified_approval_run=has_verified_approval_run,
            has_execution_evidence=has_execution_evidence,
            has_verification_evidence=has_verification_evidence,
            has_recovery_evidence=has_recovery_evidence,
            execution_health_recovery_closure_state=recovery_closure_state,
            execution_health_recovery_closure_missing=recovery_closure_missing,
            execution_health_recovery_closure_missing_count=len(recovery_closure_missing),
            execution_health_recovery_closure_required_commands=recovery_closure_required_commands,
            execution_health_recovery_closure_next_required_command=recovery_closure_next_required_command,
            execution_health_recovery_closure_proof_queue=recovery_closure_proof_queue,
            execution_health_recovery_closure_proof_queue_count=len(recovery_closure_proof_queue),
            execution_health_recovery_closure_next_proof_command=recovery_closure_next_proof_command,
            execution_health_recovery_closure_blocks_completion_claim=recovery_closure_blocks_completion_claim,
            execution_health_recovery_closure_target_run_id=recovery_closure_target_run_id,
            execution_health_recovery_closure_target_tool_name=recovery_closure_target_tool_name,
            execution_health_recovery_closure_target_verification_receipts=recovery_closure_target_verification_receipts,
            execution_health_recovery_closure_target_recovery_packets=recovery_closure_target_recovery_packets,
            execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure_target_after_action_learning_packets,
            execution_learning_state=learning_state,
            execution_learning_blocks_completion_claim=learning_blocks_completion_claim,
            execution_learning_target_run_id=learning_target_run_id,
            execution_learning_target_tool_name=learning_target_tool_name,
            execution_learning_missing=learning_missing,
            execution_learning_missing_count=len(learning_missing),
            execution_learning_required_commands=learning_required_commands,
            execution_learning_next_required_command=learning_next_required_command,
            execution_learning_proof_queue=learning_proof_queue,
            execution_learning_proof_queue_count=len(learning_proof_queue),
            execution_learning_next_proof_command=learning_next_proof_command,
            missing_case_proofs=missing_case_proofs,
            missing_case_proof_count=len(missing_case_proofs),
            next_case_proof_commands=next_case_proof_commands,
            next_case_required_command=next_case_proof_commands[0] if next_case_proof_commands else next_safe_command,
            next_case_proof_command=next_case_proof_commands[0] if next_case_proof_commands else next_safe_command,
            mission_command_queue=mission_command_queue,
            mission_command_count=len(mission_command_queue),
            evidence_preview_gate_count=evidence_preview_gate_count,
            evidence_preview_ready_count=evidence_preview_ready_count,
            evidence_preview_blocked_count=evidence_preview_blocked_count,
            evidence_preview_verdicts=evidence_preview_verdicts,
            evidence_preview_latest_verdict=evidence_preview_latest_verdict,
            evidence_preview_latest_event_id=evidence_preview_latest_event_id,
            evidence_preview_latest_handoff=evidence_preview_latest_handoff,
            evidence_preview_latest_handoff_present=evidence_preview_latest_handoff_present,
            lifecycle_stages=len(lifecycle_stages),
            complete_stages=complete_stages,
            complete_stage_count=len(complete_stages),
            missing_stages=missing_stages,
            missing_stage_count=len(missing_stages),
            recent_failed_runs=len(failed_runs),
            recent_approval_held_runs=len(approval_held_runs),
            approval_held_review_command=approval_held_review_command,
            gate_handoff=gate_metadata.get("execution_case_gate_handoff"),
            gate_handoff_present=bool(gate_metadata.get("execution_case_gate_handoff")),
            execution_proof_handoff=execution_proof_handoff,
            execution_proof_handoff_present=bool(execution_proof_handoff),
            execution_proof_state=gate_metadata.get("execution_proof_state"),
            execution_proof_next_command=gate_metadata.get("execution_proof_next_command"),
            execution_proof_can_trust_execution=gate_metadata.get("execution_proof_can_trust_execution"),
            draft_only=True,
            requires_manual_send=True,
            loads_without_execution=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
            **execution_proof_aliases,
        )

        return ToolResult(
            "execution_case_timeline",
            True,
            "\n".join(lines),
            _safe_metadata(
                found=True,
                case_id=int(case["id"]),
                request=case["request"],
                review_state=review_state,
                verdict=verdict,
                mission_state=case["mission_state"],
                go_no_go=case["go_no_go"],
                next_safe_command=next_safe_command,
                risk_signals=risk_signals,
                events=len(events),
                recent_runs=len(recent_runs),
                pending_approvals=pending_approvals,
                blockers=blockers,
                approval_required=approval_required,
                approval_queue_forecast=approval_queue_forecast,
                forecast_new_approvals=forecast_new_approvals,
                forecast_reused_approval_ids=forecast_reused_approval_ids,
                forecast_queue_before=forecast_queue_before,
                forecast_queue_after_if_sent=forecast_queue_after_if_sent,
                forecast_queue_delta_if_sent=forecast_queue_delta_if_sent,
                has_approval_evidence=has_approval_evidence,
                has_approval_chain_evidence=has_approval_chain_evidence,
                approval_chain_status=approval_chain_status,
                approval_evidence_ids=approval_evidence_ids,
                approval_linked_run_ids=approval_linked_run_ids,
                verification_receipt_ids=verification_receipt_ids,
                runtime_trace_receipt_message_ids=runtime_trace_receipt_message_ids,
                missing_verification_receipt_ids=missing_verification_receipt_ids,
                missing_runtime_trace_receipt_message_ids=missing_runtime_trace_receipt_message_ids,
                has_missing_verification_receipts=bool(missing_verification_receipt_ids),
                has_missing_runtime_trace_receipts=bool(missing_runtime_trace_receipt_message_ids),
                verified_approval_run_ids=verified_approval_run_ids,
                has_verified_approval_run=has_verified_approval_run,
                has_execution_evidence=has_execution_evidence,
                has_verification_evidence=has_verification_evidence,
                has_recovery_evidence=has_recovery_evidence,
                execution_health_recovery_closure_state=recovery_closure_state,
                execution_health_recovery_closure_missing=recovery_closure_missing,
                execution_health_recovery_closure_missing_count=len(recovery_closure_missing),
                execution_health_recovery_closure_required_commands=recovery_closure_required_commands,
                execution_health_recovery_closure_next_required_command=recovery_closure_next_required_command,
                execution_health_recovery_closure_checklist_command="recovery closure checklist" if recovery_closure_blocks_completion_claim else "",
                execution_health_recovery_closure_should_open_checklist=bool(recovery_closure_blocks_completion_claim),
                execution_health_recovery_closure_proof_queue=recovery_closure_proof_queue,
                execution_health_recovery_closure_proof_queue_count=len(recovery_closure_proof_queue),
                execution_health_recovery_closure_next_proof_command=recovery_closure_next_proof_command,
                execution_health_recovery_closure_blocks_completion_claim=recovery_closure_blocks_completion_claim,
                execution_health_recovery_closure_target_run_id=recovery_closure_target_run_id,
                execution_health_recovery_closure_target_tool_name=recovery_closure_target_tool_name,
                execution_health_recovery_closure_target_verification_receipts=recovery_closure_target_verification_receipts,
                execution_health_recovery_closure_target_recovery_packets=recovery_closure_target_recovery_packets,
                execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure_target_after_action_learning_packets,
                execution_learning_state=learning_state,
                execution_learning_blocks_completion_claim=learning_blocks_completion_claim,
                execution_learning_target_run_id=learning_target_run_id,
                execution_learning_target_tool_name=learning_target_tool_name,
                execution_learning_missing=learning_missing,
                execution_learning_missing_count=len(learning_missing),
                execution_learning_required_commands=learning_required_commands,
                execution_learning_next_required_command=learning_next_required_command,
                execution_learning_proof_queue=learning_proof_queue,
                execution_learning_proof_queue_count=len(learning_proof_queue),
                execution_learning_next_proof_command=learning_next_proof_command,
                missing_case_proofs=missing_case_proofs,
                next_case_proof_commands=next_case_proof_commands,
                next_case_required_command=next_case_proof_commands[0] if next_case_proof_commands else next_safe_command,
                next_case_proof_command=next_case_proof_commands[0] if next_case_proof_commands else next_safe_command,
                mission_command_queue=mission_command_queue,
                mission_command_count=len(mission_command_queue),
                evidence_preview_gate_count=evidence_preview_gate_count,
                evidence_preview_ready_count=evidence_preview_ready_count,
                evidence_preview_blocked_count=evidence_preview_blocked_count,
                evidence_preview_verdicts=evidence_preview_verdicts,
                evidence_preview_latest_verdict=evidence_preview_latest_verdict,
                evidence_preview_latest_event_id=evidence_preview_latest_event_id,
                evidence_preview_latest_handoff=evidence_preview_latest_handoff,
                evidence_preview_latest_handoff_present=evidence_preview_latest_handoff_present,
                initial_governor_command=gate_metadata.get("initial_governor_command", ""),
                proof_bundle_command=gate_metadata.get("proof_bundle_command", ""),
                acceptance_command=gate_metadata.get("acceptance_command", ""),
                audit_command=gate_metadata.get("audit_command", ""),
                recovery_command=gate_metadata.get("recovery_command", ""),
                learning_command=gate_metadata.get("learning_command", ""),
                recent_failed_runs=len(failed_runs),
                recent_approval_held_runs=len(approval_held_runs),
                approval_held_review_command=approval_held_review_command,
                lifecycle_stages=len(lifecycle_stages),
                complete_stages=complete_stages,
                missing_stages=missing_stages,
                missing_stage_count=len(missing_stages),
                execution_proof_handoff=execution_proof_handoff,
                execution_proof_handoff_present=bool(execution_proof_handoff),
                execution_proof_state=gate_metadata.get("execution_proof_state"),
                execution_proof_next_command=gate_metadata.get("execution_proof_next_command"),
                execution_proof_can_trust_execution=gate_metadata.get("execution_proof_can_trust_execution"),
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                **execution_proof_aliases,
                execution_case_timeline_handoff_ready=True,
                execution_case_timeline_handoff=execution_case_timeline_handoff,
            ),
        )

    def execution_runbook(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("order") or args.get("objective"), limit=600)
        if not request:
            output = (
                "Request is required. "
                "Try `execution runbook: organize downloads and summarize what changed`. "
                f"{HARNESS_REQUEST_RECOVERY_ACTION}"
            )
            execution_runbook_handoff = _safe_metadata(
                source="execution_runbook",
                execution_runbook_handoff_ready=True,
                handoff_ready=True,
                reason="missing_request",
                found=False,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
            )
            return ToolResult(
                "execution_runbook",
                False,
                output,
                _known_no_change_failure_metadata(
                    _safe_metadata(
                        source="execution_runbook",
                        reason="missing_request",
                        found=False,
                        draft_only=True,
                        requires_manual_send=True,
                        loads_without_execution=True,
                        authorizes_execution=False,
                        authorizes_completion_claim=False,
                        approval_granted=False,
                        execution_runbook_handoff_ready=True,
                        execution_runbook_handoff=execution_runbook_handoff,
                    ),
                    output=output,
                    action=HARNESS_REQUEST_RECOVERY_ACTION,
                ),
            )

        tools = list_tools()
        tool_names = {tool.name for tool in tools}
        pending = store.list_pending_approvals(limit=10)
        recent_runs = store.recent_tool_runs(limit=20)
        risks = _risk_signals(request)
        approval_required = bool(risks)
        failed_runs, approval_held_runs = _recent_tool_run_attention_buckets(recent_runs)
        approval_held_review_command = _approval_review_command(approval_held_runs) if approval_held_runs else ""
        recovery_closure = _execution_health_recovery_closure_snapshot(store, recent_runs)
        learning_debt = _execution_learning_debt_snapshot(recent_runs)
        learning_actionable_commands = list(learning_debt["actionable_required_commands"])
        recent_verification = [
            row
            for row in recent_runs
            if str(row["tool_name"])
            in {
                "verification_receipt",
                "runtime_trace_receipt",
                "verification_packet",
                "execution_acceptance_gate",
                "execution_proof_bundle",
                "execution_runbook",
            }
        ]
        route_packets = [
            tool_name
            for tool_name in [
                "execution_governor_packet",
                "dispatch_decision_packet",
                "planner_gap_packet",
                "execution_contract",
                "argument_contract_packet",
                "execution_readiness_matrix",
            ]
            if tool_name in tool_names
        ]
        proof_packets = [
            tool_name
            for tool_name in [
                "verification_packet",
                "execution_acceptance_gate",
                "execution_proof_bundle",
                "execution_audit_gate",
                "execution_health_report",
                "approval_chain_proof",
            ]
            if tool_name in tool_names
        ]
        recovery_packets = [
            tool_name
            for tool_name in [
                "execution_recovery_packet",
                "execution_health_report",
                "after_action_learning_packet",
                "checkpoint_recovery_preview",
                "checkpoint_recovery_receipt",
                "checkpoint_recovery_followthrough_packet",
                "autonomy_resume_gate",
                "autonomy_continuation_execution_packet",
                "autonomy_step_closure_packet",
                "failure_to_test_preview",
                "failure_promotion_packet",
                "learning_review",
            ]
            if tool_name in tool_names
        ]

        if pending:
            runbook_state = "HOLD_REVIEW_PENDING_APPROVALS"
            next_step = f"approval readiness {pending[0]['id']}"
        elif failed_runs:
            runbook_state = "HOLD_RECOVERY_REVIEW"
            next_step = f"execution recovery packet {failed_runs[0]['id']}"
        elif approval_held_runs:
            runbook_state = "HOLD_APPROVAL_REVIEW"
            next_step = approval_held_review_command
        elif approval_required:
            runbook_state = "READY_FOR_APPROVAL_PREFLIGHT"
            next_step = f"execution governor: {request}"
        else:
            runbook_state = "READY_FOR_SAFE_PREFLIGHT"
            next_step = f"execution governor: {request}"

        before_steps = [
            f"`execution governor: {request}`",
            f"`dispatch decision: {request}`",
            f"`argument contract: {request}`",
            f"`execution readiness matrix: {request}`",
            f"`verification packet: {request}`",
        ]
        if approval_required:
            before_steps.extend(
                [
                    f"`risk preflight: {request}`",
                    f"`action rehearsal: {request}`",
                    "Review approval readiness, the approval packet, and approval chain proof that the real risky route creates before any rerun.",
                ]
            )
        if pending:
            before_steps.insert(0, f"`approval readiness {pending[0]['id']}`")
            before_steps.insert(1, f"`approval packet {pending[0]['id']}`")
            before_steps.insert(2, f"`approval chain proof {pending[0]['id']}`")
        if failed_runs:
            before_steps.insert(0, f"`execution recovery packet {failed_runs[0]['id']}`")
            before_steps.insert(1, f"`verification receipt {failed_runs[0]['id']}`")
        if approval_held_review_command:
            approval_held_step = f"`{approval_held_review_command}`"
            if failed_runs:
                before_steps.insert(2, approval_held_step)
            elif pending:
                before_steps.insert(3, approval_held_step)
            else:
                before_steps.insert(0, approval_held_step)
        learning_target_run_id = learning_debt.get("target_run_id") or (failed_runs[0]["id"] if failed_runs else None)
        learning_closure_step = (
            f"`execution learning closure {learning_target_run_id}`"
            if learning_target_run_id
            else "`execution learning closure <run id>`"
        )
        after_action_learning_command = (
            f"after-action learning packet {learning_target_run_id}"
            if learning_target_run_id
            else "after-action learning packet <run id>"
        )
        after_action_learning_step = f"`{after_action_learning_command}`"

        during_steps = [
            "Run only the selected read-only/local-safe route automatically.",
            "If the selected route is risk-gated, stop after the safety receipt and wait for explicit approval.",
            "Respect the operator's explicit stop times, work windows, pause commands, and newer instructions before continuing any stage.",
            "Audit every tool result with tool name, risk level, ok/failure state, and approval id when applicable.",
            "Stop immediately on ambiguous target, missing exact arguments, failed proof, stale approval, or unexpected output.",
        ]
        after_steps = [
            "Attach `verification receipt <run id>` or `runtime trace receipt` to the outcome.",
            f"`acceptance gate: {request}; evidence <receipt>; tests <smoke or check>; recovery <stop or rollback>`",
            f"`execution proof bundle: {request}; verification <target>; tests <check>; evidence <receipt>; recovery <stop>; approval <id if risky>`",
            "`execution audit gate` before claiming the behavior worked.",
            "If the audit gate finds a failed, blocked, or weakly evidenced run, inspect `execution recovery packet` before retrying.",
            after_action_learning_step,
            "Capture after-action learning evidence before running the learning closure stop-check.",
            learning_closure_step,
            "Use execution learning closure as the stop-check after after-action learning evidence before turning the outcome into memory, tasks, skills, or regression tests.",
            "If a failure repeats, run `failure to test: <failure>` or `failure promotion packet` before patching.",
        ]

        blockers = []
        if pending:
            blockers.append(f"{len(pending)} pending approval(s) must be reviewed first")
        if failed_runs:
            blockers.append(f"{len(failed_runs)} recent failed/blocked run(s) need recovery review")
        if approval_held_runs:
            blockers.append(f"{len(approval_held_runs)} recent approval-held run(s) need approval review")
        if recovery_closure["blocks_completion_claim"]:
            blockers.append(
                "execution health recovery closure is incomplete: "
                + ", ".join(recovery_closure["missing"] or [str(recovery_closure["state"])])
            )
        if learning_debt["blocks_completion_claim"]:
            blockers.append(
                "execution learning debt is incomplete: "
                + ", ".join(learning_debt["missing"] or [str(learning_debt["state"])])
            )
        if approval_required:
            blockers.append("real execution requires a last-look approval packet and approval chain proof before any risky tool rerun")
        if not route_packets:
            blockers.append("route packet tools are missing")
        if not proof_packets:
            blockers.append("proof packet tools are missing")
        runbook_proof_queue = []
        for step in [next_step, *before_steps, *after_steps, "execution audit gate"]:
            command = step.strip()
            if command.startswith("`") and command.endswith("`"):
                command = command[1:-1]
            if command and command not in runbook_proof_queue:
                runbook_proof_queue.append(command)
        for command in [*recovery_closure["required_commands"], *learning_actionable_commands, *learning_debt["required_commands"]]:
            command = str(command or "").strip()
            if command and command not in runbook_proof_queue:
                runbook_proof_queue.append(command)

        lines = [
            "Jarvis execution runbook:",
            "This is the operational runbook for a real order. It is read-only and does not execute, approve, dismiss, write, control the computer, read private data, call external services, speak, complete tasks, or queue approvals.",
            "",
            f"Order: {request}",
            f"Runbook state: {runbook_state}",
            f"Next safe step: `{next_step}`",
            "",
            "Risk posture:",
            f"- risk signals: {', '.join(risks) if risks else 'none obvious from wording'}",
            f"- approval required before real execution: {'yes' if approval_required else 'no, unless the selected tool is risk-gated'}",
            f"- pending approvals: {len(pending)}",
            f"- recent failed/blocked runs: {len(failed_runs)}",
            f"- recent approval-held runs: {len(approval_held_runs)}",
            f"- approval-held review: `{approval_held_review_command}`" if approval_held_review_command else "- approval-held review: none",
            f"- recent verification packets: {len(recent_verification)}",
            f"- recovery closure state: {recovery_closure['state']}",
            f"- recovery closure next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- recovery closure next required: none",
            f"- execution learning state: {learning_debt['state']}",
            f"- execution learning actionable next: `{learning_debt['next_evidence_command']}`" if learning_debt["next_evidence_command"] else "- execution learning actionable next: none",
            f"- execution learning next required: `{learning_debt['next_required_command']}`" if learning_debt["next_required_command"] else "- execution learning next required: none",
            "",
            "Before acting:",
        ]
        lines.extend(f"- {step}" for step in before_steps)
        lines.extend(["", "During execution:"])
        lines.extend(f"- {step}" for step in during_steps)
        lines.extend(["", "After execution:"])
        lines.extend(f"- {step}" for step in after_steps)
        lines.extend(
            [
                "",
                "Proof surfaces available:",
                f"- route packets: {', '.join(route_packets) if route_packets else 'none'}",
                f"- proof packets: {', '.join(proof_packets) if proof_packets else 'none'}",
                f"- recovery/learning packets: {', '.join(recovery_packets) if recovery_packets else 'none'}",
                "",
                "Blockers:",
            ]
        )
        if blockers:
            lines.extend(f"- {blocker}" for blocker in blockers)
        else:
            lines.append("- none found by this read-only runbook")
        lines.extend(
            [
                "",
                "Boundary:",
                "- This runbook is a steering artifact. It does not call models, execute tools, approve requests, dismiss approvals, read private data, write files, control the computer, call external services, speak, complete tasks, or queue approvals.",
            ]
        )
        execution_runbook_handoff = _safe_metadata(
            source="execution_runbook",
            execution_runbook_handoff_ready=True,
            handoff_ready=True,
            request=request,
            runbook_state=runbook_state,
            next_step=next_step,
            risk_signals=risks,
            risk_signal_count=len(risks),
            approval_required=approval_required,
            pending_approvals=len(pending),
            recent_failed_runs=len(failed_runs),
            recent_approval_held_runs=len(approval_held_runs),
            approval_held_review_command=approval_held_review_command,
            recent_verification_runs=len(recent_verification),
            before_step_names=before_steps,
            before_steps=len(before_steps),
            during_step_names=during_steps,
            during_steps=len(during_steps),
            after_step_names=after_steps,
            after_steps=len(after_steps),
            route_packets=route_packets,
            route_packet_count=len(route_packets),
            proof_packets=proof_packets,
            proof_packet_count=len(proof_packets),
            recovery_packets=recovery_packets,
            recovery_packet_count=len(recovery_packets),
            blocker_details=blockers,
            blocker_count=len(blockers),
            blockers=len(blockers),
            learning_closure_step=learning_closure_step,
            execution_learning_closure_command=learning_closure_step.strip("`"),
            execution_health_recovery_closure_state=recovery_closure["state"],
            execution_health_recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
            execution_health_recovery_closure_missing=recovery_closure["missing"],
            execution_health_recovery_closure_missing_count=recovery_closure["missing_count"],
            execution_health_recovery_closure_required_commands=recovery_closure["required_commands"],
            execution_health_recovery_closure_next_required_command=recovery_closure["next_required_command"],
            execution_health_recovery_closure_checklist_command=_recovery_closure_checklist_command(recovery_closure),
            execution_health_recovery_closure_should_open_checklist=bool(_recovery_closure_checklist_command(recovery_closure)),
            execution_health_recovery_closure_proof_queue=recovery_closure["proof_queue"],
            execution_health_recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
            execution_health_recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
            execution_health_recovery_closure_blocks_completion_claim=recovery_closure["blocks_completion_claim"],
            execution_health_recovery_closure_target_run_id=recovery_closure["target_run_id"],
            execution_health_recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
            execution_health_recovery_closure_target_verification_receipts=recovery_closure["target_verification_receipts"],
            execution_health_recovery_closure_target_recovery_packets=recovery_closure["target_recovery_packets"],
            execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure["target_after_action_learning_packets"],
            execution_learning_state=learning_debt["state"],
            execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
            execution_learning_recent_action_runs=learning_debt["recent_action_runs"],
            execution_learning_failed_or_blocked_action_runs=learning_debt["failed_or_blocked_action_runs"],
            execution_learning_recent_verification_runs=learning_debt["recent_verification_runs"],
            execution_learning_recent_recovery_runs=learning_debt["recent_recovery_runs"],
            execution_learning_recent_after_action_learning_runs=learning_debt["recent_after_action_learning_runs"],
            execution_learning_target_run_id=learning_debt["target_run_id"],
            execution_learning_target_tool_name=learning_debt["target_tool_name"],
            execution_learning_target_after_action_learning_packets=learning_debt["target_after_action_learning_packets"],
            execution_learning_missing=learning_debt["missing"],
            execution_learning_missing_count=learning_debt["missing_count"],
            execution_learning_evidence_command=learning_debt["learning_evidence_command"],
            execution_learning_after_action_learning_command=after_action_learning_command,
            execution_learning_required_commands=learning_debt["required_commands"],
            execution_learning_next_required_command=learning_debt["next_required_command"],
            execution_learning_proof_queue=learning_debt["proof_queue"],
            execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
            execution_learning_next_proof_command=learning_debt["next_proof_command"],
            execution_learning_actionable_required_commands=learning_actionable_commands,
            execution_learning_actionable_required_command_count=len(learning_actionable_commands),
            execution_learning_actionable_proof_queue=learning_actionable_commands,
            execution_learning_actionable_proof_queue_count=len(learning_actionable_commands),
            execution_learning_actionable_next_required_command=learning_debt["actionable_next_required_command"],
            execution_learning_actionable_next_proof_command=learning_debt["actionable_next_proof_command"],
            execution_learning_next_evidence_command=learning_debt["next_evidence_command"],
            runbook_proof_queue=runbook_proof_queue,
            runbook_proof_queue_count=len(runbook_proof_queue),
            runbook_next_proof_command=runbook_proof_queue[0] if runbook_proof_queue else next_step,
            draft_only=True,
            requires_manual_send=True,
            loads_without_execution=True,
            authorizes_execution=False,
            authorizes_completion_claim=False,
            approval_granted=False,
        )

        return ToolResult(
            "execution_runbook",
            True,
            "\n".join(lines),
            _safe_metadata(
                source="execution_runbook",
                request=request,
                runbook_state=runbook_state,
                next_step=next_step,
                risk_signals=risks,
                approval_required=approval_required,
                pending_approvals=len(pending),
                recent_failed_runs=len(failed_runs),
                recent_approval_held_runs=len(approval_held_runs),
                approval_held_review_command=approval_held_review_command,
                recent_verification_runs=len(recent_verification),
                before_step_names=before_steps,
                before_steps=len(before_steps),
                during_step_names=during_steps,
                during_steps=len(during_steps),
                after_step_names=after_steps,
                after_steps=len(after_steps),
                route_packets=route_packets,
                proof_packets=proof_packets,
                recovery_packets=recovery_packets,
                blocker_details=blockers,
                blocker_count=len(blockers),
                blockers=len(blockers),
                learning_closure_step=learning_closure_step,
                execution_learning_closure_command=learning_closure_step.strip("`"),
                execution_health_recovery_closure_state=recovery_closure["state"],
                execution_health_recovery_closure_ready_to_retry=recovery_closure["ready_to_retry"],
                execution_health_recovery_closure_missing=recovery_closure["missing"],
                execution_health_recovery_closure_missing_count=recovery_closure["missing_count"],
                execution_health_recovery_closure_required_commands=recovery_closure["required_commands"],
                execution_health_recovery_closure_next_required_command=recovery_closure["next_required_command"],
                execution_health_recovery_closure_checklist_command=_recovery_closure_checklist_command(recovery_closure),
                execution_health_recovery_closure_should_open_checklist=bool(_recovery_closure_checklist_command(recovery_closure)),
                execution_health_recovery_closure_proof_queue=recovery_closure["proof_queue"],
                execution_health_recovery_closure_proof_queue_count=recovery_closure["proof_queue_count"],
                execution_health_recovery_closure_next_proof_command=recovery_closure["next_proof_command"],
                execution_health_recovery_closure_blocks_completion_claim=recovery_closure["blocks_completion_claim"],
                execution_health_recovery_closure_target_run_id=recovery_closure["target_run_id"],
                execution_health_recovery_closure_target_tool_name=recovery_closure["target_tool_name"],
                execution_health_recovery_closure_target_verification_receipts=recovery_closure["target_verification_receipts"],
                execution_health_recovery_closure_target_recovery_packets=recovery_closure["target_recovery_packets"],
                execution_health_recovery_closure_target_after_action_learning_packets=recovery_closure["target_after_action_learning_packets"],
                execution_learning_state=learning_debt["state"],
                execution_learning_blocks_completion_claim=learning_debt["blocks_completion_claim"],
                execution_learning_recent_action_runs=learning_debt["recent_action_runs"],
                execution_learning_failed_or_blocked_action_runs=learning_debt["failed_or_blocked_action_runs"],
                execution_learning_recent_verification_runs=learning_debt["recent_verification_runs"],
                execution_learning_recent_recovery_runs=learning_debt["recent_recovery_runs"],
                execution_learning_recent_after_action_learning_runs=learning_debt["recent_after_action_learning_runs"],
                execution_learning_target_run_id=learning_debt["target_run_id"],
                execution_learning_target_tool_name=learning_debt["target_tool_name"],
                execution_learning_target_after_action_learning_packets=learning_debt["target_after_action_learning_packets"],
                execution_learning_missing=learning_debt["missing"],
                execution_learning_missing_count=learning_debt["missing_count"],
                execution_learning_evidence_command=learning_debt["learning_evidence_command"],
                execution_learning_after_action_learning_command=after_action_learning_command,
                execution_learning_required_commands=learning_debt["required_commands"],
                execution_learning_next_required_command=learning_debt["next_required_command"],
                execution_learning_proof_queue=learning_debt["proof_queue"],
                execution_learning_proof_queue_count=learning_debt["proof_queue_count"],
                execution_learning_next_proof_command=learning_debt["next_proof_command"],
                execution_learning_actionable_required_commands=learning_actionable_commands,
                execution_learning_actionable_required_command_count=len(learning_actionable_commands),
                execution_learning_actionable_proof_queue=learning_actionable_commands,
                execution_learning_actionable_proof_queue_count=len(learning_actionable_commands),
                execution_learning_actionable_next_required_command=learning_debt["actionable_next_required_command"],
                execution_learning_actionable_next_proof_command=learning_debt["actionable_next_proof_command"],
                execution_learning_next_evidence_command=learning_debt["next_evidence_command"],
                runbook_proof_queue=runbook_proof_queue,
                runbook_proof_queue_count=len(runbook_proof_queue),
                runbook_next_proof_command=runbook_proof_queue[0] if runbook_proof_queue else next_step,
                draft_only=True,
                requires_manual_send=True,
                loads_without_execution=True,
                authorizes_execution=False,
                authorizes_completion_claim=False,
                approval_granted=False,
                execution_runbook_handoff_ready=True,
                execution_runbook_handoff=execution_runbook_handoff,
            ),
        )

    def harness_doctrine(_: dict[str, Any]) -> ToolResult:
        build_filter = [
            "steering",
            "pedals",
            "brakes",
            "dashboard",
            "memory/state",
            "tool orchestration",
            "approval flow",
            "audit",
            "recovery",
            "verification",
            "learning",
        ]
        doctrine_rows = [
            {
                "name": item["name"],
                "principle": item["principle"],
                "jarvis_requirement": item["jarvis_requirement"],
                "non_authorizing": True,
            }
            for item in HARNESS_DOCTRINE
        ]
        try:
            tool_names = {str(getattr(tool, "name", "") or "") for tool in list_tools()}
        except Exception:
            tool_names = set()
        layer_contract_rows = _harness_layer_contract_rows(tool_names)
        layer_contract_metadata = _harness_layer_contract_metadata(layer_contract_rows)
        doctrine_metadata = {
            "principles": len(HARNESS_DOCTRINE),
            "doctrine_rows": doctrine_rows,
            "doctrine_row_count": len(doctrine_rows),
            "priority": "finish_jarvis_agent_harness",
            "build_filter": build_filter,
            "build_filter_count": len(build_filter),
            **layer_contract_metadata,
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "operator_timeboxes_override_priority": True,
            "stop_times_override_priority": True,
        }
        harness_doctrine_handoff = _safe_metadata(source="harness_doctrine", **doctrine_metadata)
        lines = [
            "Jarvis agent harness doctrine:",
            "",
            "Working thesis:",
            "- A stronger AI brain is not enough by itself. Jarvis must be the operating layer that lets the operator steer, stop, inspect, verify, recover, and teach the assistant.",
            "- The completion target is a dependable personal-agent harness first, then stronger brains can be hosted inside it safely.",
            "",
            "Typed AGI harness contract:",
        ]
        for row in layer_contract_rows:
            lines.extend(
                [
                    f"- {row['title']}: {row['status']}",
                    f"  Contract: {row['contract']}",
                    f"  Boundary: {row['boundary']}",
                ]
            )
        lines.extend(
            [
                "",
                "Doctrine:",
            ]
        )
        for index, item in enumerate(HARNESS_DOCTRINE, start=1):
            lines.extend(
                [
                    f"{index}. {item['name']}",
                    f"   Principle: {item['principle']}",
                    f"   Jarvis requirement: {item['jarvis_requirement']}",
                ]
            )

        lines.extend(
            [
                "",
                "Build filter:",
                "- Prefer features that improve steering, pedals, brakes, dashboard, memory/state, tool orchestration, approval flow, audit, recovery, verification, or learning.",
                "- Defer features that only make Jarvis look smarter while leaving execution, safety, or recovery vague.",
                "",
                "Safety floor:",
                "- This doctrine is read-only and does not override approval gates.",
                "- This priority does not override the operator's explicit stop times, work windows, pause commands, or newer instructions.",
            ]
        )

        return ToolResult(
            "harness_doctrine",
            True,
            "\n".join(lines),
            _safe_metadata(
                **doctrine_metadata,
                harness_doctrine_handoff=harness_doctrine_handoff,
            ),
        )

    def coding_discipline_packet(args: dict[str, Any]) -> ToolResult:
        objective = str(args.get("objective") or args.get("request") or "continue Jarvis harness work").strip()
        likely_files = _split_csvish(args.get("files") or args.get("changed_files") or "")
        verification = str(args.get("verification") or args.get("tests") or "").strip()
        ambiguity = "review" if any(token in objective.lower() for token in ("maybe", "somehow", "fix it", "make it better", "improve", "stuff")) else "low"
        rows = [
            {
                "principle": "Think before coding",
                "status": "ready" if ambiguity == "low" else "review",
                "check": "state assumptions, ambiguity, and tradeoffs before editing",
                "non_authorizing": True,
            },
            {
                "principle": "Simplicity first",
                "status": "ready",
                "check": "choose the smallest implementation that satisfies the current objective",
                "non_authorizing": True,
            },
            {
                "principle": "Surgical changes",
                "status": "ready" if likely_files else "review",
                "check": "touch only files and lines tied to the request; avoid unrelated cleanup",
                "non_authorizing": True,
            },
            {
                "principle": "Goal-driven execution",
                "status": "ready" if verification else "review",
                "check": "define focused success criteria and run the nearest verification loop",
                "non_authorizing": True,
            },
        ]
        proof_queue = [
            f"read owning files for: {', '.join(likely_files) if likely_files else '<target files>'}",
            "write or update the nearest focused smoke test",
            verification or "python3 -m py_compile <touched files>",
            "completion claim gate: <objective>",
        ]
        ready = all(row["status"] == "ready" for row in rows)
        discipline_metadata = {
            "objective": objective[:300],
            "objective_chars": len(objective[:300]),
            "likely_files": likely_files,
            "likely_file_count": len(likely_files),
            "verification": verification,
            "verification_supplied": bool(verification),
            "ambiguity": ambiguity,
            "discipline_rows": rows,
            "discipline_row_count": len(rows),
            "ready_rows": len([row for row in rows if row["status"] == "ready"]),
            "review_rows": len([row for row in rows if row["status"] == "review"]),
            "ready_for_implementation_review": ready,
            "proof_queue": proof_queue,
            "proof_queue_count": len(proof_queue),
            "next_proof_command": proof_queue[0],
            "authorizes_execution": False,
            "authorizes_edits": False,
            "authorizes_risky_work": False,
            "authorizes_test_execution": False,
            "authorizes_completion_claim": False,
            "draft_only": True,
            "requires_manual_send": True,
            "loads_without_execution": True,
            "approval_granted": False,
            "operator_timeboxes_override_priority": True,
            "stop_times_override_priority": True,
        }
        coding_discipline_handoff = _safe_metadata(source="coding_discipline_packet", **discipline_metadata)
        lines = [
            "Jarvis coding discipline packet:",
            "This is the pre-code harness filter inspired by the Karpathy coding-agent guidelines. It is read-only and does not edit files or run tests.",
            "",
            f"Objective: {objective}",
            "",
            "Discipline rows:",
            *[
                f"- {row['principle']}: {row['status']} - {row['check']}; authorizes edits no; authorizes risky work no"
                for row in rows
            ],
            "",
            "Surgical scope:",
            f"- likely files: {', '.join(likely_files) if likely_files else 'not supplied'}",
            "- unrelated refactors: held",
            "- speculative abstractions: held",
            "",
            "Verification contract:",
            f"- supplied verification: {verification or 'not supplied'}",
            f"- ready for implementation review: {'yes' if ready else 'no'}",
            "",
            "Proof queue:",
            *[f"- `{command}`" for command in proof_queue],
            "",
            "Boundary:",
            "- This packet guides coding behavior only. Shell/code execution, computer control, personal data, destructive work, and external side effects remain approval-gated.",
        ]
        return ToolResult(
            "coding_discipline_packet",
            True,
            "\n".join(lines),
            _safe_metadata(
                **discipline_metadata,
                coding_discipline_handoff=coding_discipline_handoff,
            ),
        )

    return priority_goal, harness_status, harness_cycle_preview, harness_lifecycle_state, harness_control_surface, harness_operations_brief, agi_gate_report, agi_next_build_move, harness_completion_assessment, harness_doctrine, coding_discipline_packet, completion_audit_packet, evidence_ledger, completion_claim_gate, completion_next_proof_packet, completion_proof_refresh_packet, operator_handoff_packet, harness_readiness_digest, execution_proof_bundle, execution_mission_control, execution_case_handoff_packet, save_execution_case, inspect_execution_case, execution_case_evidence_packet, append_execution_case_evidence, execution_case_gate, execution_case_review_packet, execution_case_closure_packet, execution_case_timeline, execution_runbook
