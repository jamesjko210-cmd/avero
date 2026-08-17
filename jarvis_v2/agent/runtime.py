from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

from jarvis_v2.agent.chat import ChatBrain
from jarvis_v2.agent.command_suggest import suggest_command
from jarvis_v2.agent.execution_outcome import (
    APPROVED_EXECUTION_OUTCOME_UNKNOWN,
    ApprovedExecutionOutcome,
    classify_approved_execution_outcome,
)
from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.model_planner import ModelBackedPlanner
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.subagent_fleet import SubagentFleet, spawn_subagent_fleet
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, RuntimeResult, ToolResult
from jarvis_v2.agent.verifier import Verifier
from jarvis_v2.config import JarvisConfig, load_config
from jarvis_v2.memory.decision_projection import reconcile_pending_decision_projections
from jarvis_v2.memory.goal_projection import reconcile_pending_goal_projections
from jarvis_v2.memory.memory_projection import reconcile_pending_memory_projections
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.person_projection import reconcile_pending_person_projections
from jarvis_v2.memory.preference_projection import reconcile_pending_preference_projections
from jarvis_v2.memory.profile_projection import reconcile_pending_profile_projections
from jarvis_v2.memory.skill_projection import reconcile_pending_skill_projections
from jarvis_v2.memory.store import (
    SQLITE_BUSY_TIMEOUT_SECONDS,
    STARTUP_RECOVERY_HEARTBEAT_SECONDS,
    ApprovalExecutionClaim,
    MemoryStore,
    approval_action_digest,
)
from jarvis_v2.scripts.startup import StartupRecoveryUnavailable
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import build_core_registry
from jarvis_v2.v3_commands import V3_BOOTSTRAP_CHECK_COMMAND


MAX_RUNTIME_OUTPUT_PREVIEW_CHARS = 240
STARTUP_RECOVERY_HEARTBEAT_JOIN_SECONDS = SQLITE_BUSY_TIMEOUT_SECONDS + 1.0
MAX_AUTO_MUTATION_REQUEST_TOKEN_CHARS = 512
AUTO_MUTATION_STALE_RUNNING_SECONDS = 60 * 60
LOCAL_PATH_RE = re.compile(
    r"/(?:Users|private|var/folders|tmp)/[^\n\r]*",
    re.IGNORECASE,
)
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_HISTORY_PROVENANCE_FIELDS = {
    "role",
    "lineage_state",
    "policy_epoch_id",
    "policy_fingerprint",
    "provider",
    "destination_class",
    "future_history_allowed",
    "source_count",
    "source_digest",
    "lineage_token_digest",
}
_HISTORY_PROVENANCE_LINEAGE = {
    "direct": "direct_current",
    "personal_derived": "personal_derived",
    "tool_derived": "tool_derived",
    "mixed_restricted": "mixed_restricted",
}


def _startup_recovery_exception_type(exc: BaseException) -> str:
    if isinstance(exc, sqlite3.Error):
        return "SQLiteError"
    for error_type, label in (
        (PermissionError, "PermissionError"),
        (TimeoutError, "TimeoutError"),
        (ValueError, "ValueError"),
        (TypeError, "TypeError"),
        (OSError, "OSError"),
        (RuntimeError, "RuntimeError"),
    ):
        if isinstance(exc, error_type):
            return label
    return "Exception"


_RUNTIME_EXACT_TOOL_ALIASES = {
    "cockpit": "capability_cockpit",
    "attention": "capability_cockpit",
    "attention status": "capability_cockpit",
    "ability count": "capability_map",
    "abilities count": "capability_map",
    "approval gated count": "risk_matrix",
    "approval gated tool count": "risk_matrix",
    "approval gated tools count": "risk_matrix",
    "capability cockpit": "capability_cockpit",
    "capability cockpit plan": "capability_cockpit",
    "capability count": "capability_map",
    "capabilities count": "capability_map",
    "cockpit attention": "capability_cockpit",
    "cockpit summary": "capability_cockpit",
    "count abilities": "capability_map",
    "count capabilities": "capability_map",
    "count integrations": "capability_map",
    "count tools": "capability_map",
    "eval pack plan": "capability_cockpit",
    "frozen redos finding": "frozen_routing_risk_report",
    "frozen regex finding": "frozen_routing_risk_report",
    "frozen routing risk report": "frozen_routing_risk_report",
    "command length safety report": "planner_input_guard_report",
    "jarvis cockpit summary": "capability_cockpit",
    "operator eval pack plan": "capability_cockpit",
    "trust checklist": "capability_cockpit",
    "earned trust": "capability_cockpit",
    "earned trust checklist": "capability_cockpit",
    "can i trust jarvis": "capability_cockpit",
    "can i trust you": "capability_cockpit",
    "why can i trust jarvis": "capability_cockpit",
    "why should i trust jarvis": "capability_cockpit",
    "what makes jarvis reliable": "capability_cockpit",
    "what makes jarvis trustworthy": "capability_cockpit",
    "how many abilities": "capability_map",
    "how many capabilities": "capability_map",
    "how many capabilities do you have": "capability_map",
    "how many integrations": "capability_map",
    "how many approval gated tools": "risk_matrix",
    "how many high risk tools": "risk_matrix",
    "how many local safe tools": "risk_matrix",
    "how many personal data tools": "risk_matrix",
    "how many read only tools": "risk_matrix",
    "how many risk gated tools": "risk_matrix",
    "how many tools": "capability_map",
    "how many tools do you have": "capability_map",
    "how do we unlock the moat": "work_queue",
    "high risk count": "risk_matrix",
    "high risk tool count": "risk_matrix",
    "integration count": "capability_map",
    "integrations count": "capability_map",
    "jarvis three month plan": "work_queue",
    "local safe count": "risk_matrix",
    "local safe tool count": "risk_matrix",
    "personal data count": "risk_matrix",
    "personal data tool count": "risk_matrix",
    "read only count": "risk_matrix",
    "read only tool count": "risk_matrix",
    "risk count": "risk_matrix",
    "risk counts": "risk_matrix",
    "risk gated count": "risk_matrix",
    "risk gated tool count": "risk_matrix",
    "regex dos status": "frozen_routing_risk_report",
    "regex risk status": "frozen_routing_risk_report",
    "regex safety report": "frozen_routing_risk_report",
    "regex safety status": "frozen_routing_risk_report",
    "redos proof matrix": "frozen_routing_risk_report",
    "redos status": "frozen_routing_risk_report",
    "input length guard status": "planner_input_guard_report",
    "long command redos risk": "planner_input_guard_report",
    "long command safety status": "planner_input_guard_report",
    "long input redos status": "planner_input_guard_report",
    "long input safety status": "planner_input_guard_report",
    "long message planner safety": "planner_input_guard_report",
    "long message redos status": "planner_input_guard_report",
    "planner input cap status": "planner_input_guard_report",
    "planner input guard decision": "planner_input_guard_report",
    "planner input guard report": "planner_input_guard_report",
    "planner input length guard": "planner_input_guard_report",
    "send call redos": "frozen_routing_risk_report",
    "send call routing risk report": "frozen_routing_risk_report",
    "send routing redos": "frozen_routing_risk_report",
    "tool risk count": "risk_matrix",
    "tool risk counts": "risk_matrix",
    "show cockpit": "capability_cockpit",
    "show me cockpit": "capability_cockpit",
    "show me the cockpit": "capability_cockpit",
    "tool count": "capability_map",
    "tools count": "capability_map",
    "three month plan": "work_queue",
    "three months of work": "work_queue",
    "what are the three months of work": "work_queue",
    "what is capability cockpit plan": "capability_cockpit",
    "what is eval pack": "capability_cockpit",
    "what is phone control center": "capability_cockpit",
    "what is the capability cockpit plan": "capability_cockpit",
    "what is the eval pack": "capability_cockpit",
    "what is the frozen redos risk": "frozen_routing_risk_report",
    "what is the input length guard decision": "planner_input_guard_report",
    "what is the operator eval pack": "capability_cockpit",
    "what is the phone control center": "capability_cockpit",
    "what is the three month plan": "work_queue",
    "what is the three month plan for jarvis": "work_queue",
    "check codex tasks": "work_queue",
    "check current instructions": "work_queue",
    "codex current instructions": "work_queue",
    "codex tasks": "work_queue",
    "codex work queue": "work_queue",
    "current instructions": "work_queue",
    "current work queue": "work_queue",
    "show current instructions": "work_queue",
    "what are current instructions": "work_queue",
    "what is the current work queue": "work_queue",
    "what command should i run": "safe_next_actions",
    "what command should i run now": "safe_next_actions",
    "what is jarvis next step": "safe_next_actions",
    "what is next for jarvis": "safe_next_actions",
    "what s next for jarvis": "safe_next_actions",
    "what should i check now": "safe_next_actions",
    "what should i run now": "safe_next_actions",
    "what should codex do next": "work_queue",
    "what should codex work on": "work_queue",
    "whats next for jarvis": "safe_next_actions",
    "handoff": "handoff_brief",
    "handoff brief status": "handoff_brief",
    "handoff notice": "handoff_brief",
    "handoff report": "handoff_brief",
    "handoff status": "handoff_brief",
    "what is the handoff": "handoff_brief",
    "what did claude leave": "handoff_brief",
    "what did claude leave for codex": "handoff_brief",
    "what did claude tell codex": "handoff_brief",
    "what did claude hand off": "handoff_brief",
    "what did claude say": "handoff_brief",
    "what did codex change": "build_progress_report",
    "what did codex just change": "build_progress_report",
    "what did codex just do": "build_progress_report",
    "what did codex modify": "build_progress_report",
    "what changed after codex": "build_progress_report",
    "what changed in code": "build_progress_report",
    "what did you change": "build_progress_report",
    "what did you edit": "build_progress_report",
    "what did you just change": "build_progress_report",
    "what did you just do": "build_progress_report",
    "what files changed": "build_progress_report",
    "what files did codex edit": "build_progress_report",
    "which files changed": "build_progress_report",
    "which files did you edit": "build_progress_report",
    "show changed files": "build_progress_report",
    "show modified files": "build_progress_report",
    "code changes status": "build_progress_report",
    "diff status": "build_progress_report",
    "patch status": "build_progress_report",
    "what should claude review": "build_progress_report",
    "what should i review in this change": "build_progress_report",
    "what real workflows should jarvis prove": "capability_cockpit",
    "what should jarvis build before integrations": "work_queue",
    "what should jarvis focus on for the next three months": "work_queue",
    "what should jarvis prove live": "capability_cockpit",
    "how do i report live test results": "capability_cockpit",
    "how should i report live test results": "capability_cockpit",
    "how should i report the live matrix": "capability_cockpit",
    "how to report live test results": "capability_cockpit",
    "live matrix result format": "capability_cockpit",
    "live test result format": "capability_cockpit",
    "live test results format": "capability_cockpit",
    "report live matrix results": "capability_cockpit",
    "report live test results": "capability_cockpit",
    "what format should i use for live test results": "capability_cockpit",
    "what live results do you need from me": "capability_cockpit",
    "what results do you need from me": "capability_cockpit",
    "what should i send after testing": "capability_cockpit",
    "what long input risk is open": "planner_input_guard_report",
    "what redos is open": "frozen_routing_risk_report",
    "what regex dos is open": "frozen_routing_risk_report",
    "what unlocks jarvis moat": "work_queue",
    "what unlocks jarvis s moat": "work_queue",
    "what unlocks the moat": "work_queue",
    "when should jarvis add integrations": "work_queue",
    "should jarvis add integrations now": "work_queue",
    "should we add input length guard": "planner_input_guard_report",
    "total input length guard": "planner_input_guard_report",
    "planner redos finding": "frozen_routing_risk_report",
    "planner regex risk": "frozen_routing_risk_report",
    "phone control center plan": "capability_cockpit",
    "show frozen routing risk": "frozen_routing_risk_report",
    "where is the cockpit": "capability_cockpit",
    "anything failing": "execution_health_report",
    "anything broken": "recent_tool_runs",
    "audit trail": "recent_tool_runs",
    "broken status": "recent_tool_runs",
    "error summary": "recent_tool_runs",
    "execution audit": "recent_tool_runs",
    "failure details": "execution_health_report",
    "is anything broken": "recent_tool_runs",
    "last execution receipt": "verification_receipt",
    "last error": "recent_tool_runs",
    "last failed tool": "recent_tool_runs",
    "last failure": "recent_tool_runs",
    "last tool run": "recent_tool_runs",
    "latest execution receipt": "verification_receipt",
    "latest error": "recent_tool_runs",
    "latest failure": "recent_tool_runs",
    "latest tool run": "recent_tool_runs",
    "recent errors": "recent_tool_runs",
    "recent run count": "recent_tool_runs",
    "recent runs count": "recent_tool_runs",
    "recent tool run": "recent_tool_runs",
    "runtime trace latest": "runtime_trace_receipt",
    "show audit trail": "recent_tool_runs",
    "show execution audit": "recent_tool_runs",
    "show last error": "recent_tool_runs",
    "show last failure": "recent_tool_runs",
    "show me what broke": "recent_tool_runs",
    "show me what is broken": "recent_tool_runs",
    "audit count": "recent_tool_runs",
    "count tool runs": "recent_tool_runs",
    "how many recent runs": "recent_tool_runs",
    "how many tool runs": "recent_tool_runs",
    "tool execution history": "recent_tool_runs",
    "tool run count": "recent_tool_runs",
    "tool run history": "recent_tool_runs",
    "tool runs count": "recent_tool_runs",
    "what broke": "recent_tool_runs",
    "what did jarvis run last": "recent_tool_runs",
    "what errors are there": "recent_tool_runs",
    "what errors happened": "recent_tool_runs",
    "what failed last": "recent_tool_runs",
    "what ran last": "recent_tool_runs",
    "what tool ran last": "recent_tool_runs",
    "what is broken": "recent_tool_runs",
    "what is wrong": "recent_tool_runs",
    "what s broken": "recent_tool_runs",
    "what s wrong": "recent_tool_runs",
    "what was the last error": "recent_tool_runs",
    "what was the last failure": "recent_tool_runs",
    "what went wrong": "recent_tool_runs",
    "whats broken": "recent_tool_runs",
    "whats wrong": "recent_tool_runs",
    "why did it fail": "recent_tool_runs",
    "failed status": "execution_health_report",
    "failures": "execution_health_report",
    "failure report": "execution_health_report",
    "failure status": "execution_health_report",
    "repeated failure status": "repeated_failure_clusters",
    "repeated failures status": "repeated_failure_clusters",
    "health report": "execution_health_report",
    "recent failures": "execution_health_report",
    "show failures": "execution_health_report",
    "show me failures": "execution_health_report",
    "show me what failed": "execution_health_report",
    "what failed": "execution_health_report",
    "what failed recently": "execution_health_report",
    "what is failing": "execution_health_report",
    "what is unhealthy": "execution_health_report",
    "what just failed": "execution_health_report",
    "what needs recovery": "recovery_closure_checklist",
    "what s failing": "execution_health_report",
    "whats failing": "execution_health_report",
    "how do i recover": "recovery_closure_checklist",
    "is recovery debt closed": "recovery_closure_checklist",
    "recovery closure status": "recovery_closure_checklist",
    "recovery debt status": "recovery_closure_checklist",
    "recovery learning status": "recovery_closure_checklist",
    "recovery plan": "recovery_closure_checklist",
    "recovery status": "recovery_closure_checklist",
    "chat latency": "model_routing_status",
    "chat latency acceptance status": "model_routing_status",
    "chat latency status": "model_routing_status",
    "chat p95": "model_routing_status",
    "chat p95 status": "model_routing_status",
    "chat p95 target": "model_routing_status",
    "chat acceptance latency": "model_routing_status",
    "chat history window": "model_routing_status",
    "chat history messages": "model_routing_status",
    "chat max history messages": "model_routing_status",
    "chat max reply tokens": "model_routing_status",
    "chat reply token cap": "model_routing_status",
    "chat response length status": "model_routing_status",
    "chat slow": "model_routing_status",
    "chat speed status": "model_routing_status",
    "chat speed acceptance status": "model_routing_status",
    "chat speed settings": "model_routing_status",
    "chat token cap": "model_routing_status",
    "chat tuning": "model_routing_status",
    "chat tuning knobs": "model_routing_status",
    "conversation acceptance latency": "model_routing_status",
    "conversation speed status": "model_routing_status",
    "conversation p95 status": "model_routing_status",
    "8 second chat target": "model_routing_status",
    "8 second conversation target": "model_routing_status",
    "8s chat target": "model_routing_status",
    "8s conversation target": "model_routing_status",
    "did chat meet the 8 second target": "model_routing_status",
    "did chat pass latency": "model_routing_status",
    "did mixed conversation pass latency": "model_routing_status",
    "how do i make chat faster": "model_routing_status",
    "conversation latency status": "model_routing_status",
    "how fast is chat": "model_routing_status",
    "is chat under 8 seconds": "model_routing_status",
    "is jarvis under 8 seconds": "model_routing_status",
    "is chat fast enough": "model_routing_status",
    "is mixed conversation under 8 seconds": "model_routing_status",
    "jarvis latency status": "model_routing_status",
    "jarvis speed status": "model_routing_status",
    "latency status": "model_routing_status",
    "lower chat history window": "model_routing_status",
    "lower chat token cap": "model_routing_status",
    "lower reply token cap": "model_routing_status",
    "make chat faster": "model_routing_status",
    "make jarvis faster": "model_routing_status",
    "make jarvis replies shorter": "model_routing_status",
    "mixed conversation latency": "model_routing_status",
    "mixed conversation latency status": "model_routing_status",
    "mixed conversation acceptance latency": "model_routing_status",
    "mixed conversation acceptance status": "model_routing_status",
    "mixed conversation p95": "model_routing_status",
    "mixed conversation p95 status": "model_routing_status",
    "mixed conversation status": "model_routing_status",
    "p95 chat target": "model_routing_status",
    "p95 latency status": "model_routing_status",
    "reduce chat history": "model_routing_status",
    "reduce reply tokens": "model_routing_status",
    "reply length status": "model_routing_status",
    "response length status": "model_routing_status",
    "should we lower chat history": "model_routing_status",
    "should we lower reply tokens": "model_routing_status",
    "speed up chat": "model_routing_status",
    "speed up jarvis": "model_routing_status",
    "tune chat speed": "model_routing_status",
    "tune jarvis speed": "model_routing_status",
    "what are the chat speed knobs": "model_routing_status",
    "what are the chat tuning knobs": "model_routing_status",
    "under 8 seconds status": "model_routing_status",
    "why are replies slow": "model_routing_status",
    "why are chat replies slow": "model_routing_status",
    "why is chat slow": "model_routing_status",
    "why is jarvis slow": "model_routing_status",
    "anything need approval": "list_pending_approvals",
    "anything needs my approval": "list_pending_approvals",
    "anything pending approval": "list_pending_approvals",
    "anything waiting on me": "list_pending_approvals",
    "any approvals pending": "list_pending_approvals",
    "approval count": "list_pending_approvals",
    "approval needed": "list_pending_approvals",
    "approval queue": "list_pending_approvals",
    "are approvals pending": "list_pending_approvals",
    "count approvals": "list_pending_approvals",
    "do you need anything from me": "list_pending_approvals",
    "do you need me for anything": "list_pending_approvals",
    "do you need my approval": "list_pending_approvals",
    "how many approvals": "list_pending_approvals",
    "how many approvals are pending": "list_pending_approvals",
    "needs approval": "list_pending_approvals",
    "pending approval": "list_pending_approvals",
    "pending approval count": "list_pending_approvals",
    "pending approval status": "list_pending_approvals",
    "pending approvals": "list_pending_approvals",
    "pending approvals count": "list_pending_approvals",
    "pending review": "list_pending_approvals",
    "review queue": "list_pending_approvals",
    "show approval queue": "list_pending_approvals",
    "show approvals": "list_pending_approvals",
    "show me approvals": "list_pending_approvals",
    "show me pending approvals": "list_pending_approvals",
    "show pending approval status": "list_pending_approvals",
    "show pending approvals": "list_pending_approvals",
    "waiting for me": "list_pending_approvals",
    "what approvals are pending": "list_pending_approvals",
    "what approvals are waiting": "list_pending_approvals",
    "what needs my approval": "list_pending_approvals",
    "what do you need from me": "list_pending_approvals",
    "what is blocked by me": "list_pending_approvals",
    "what is pending": "list_pending_approvals",
    "what is pending approval": "list_pending_approvals",
    "what is waiting on me": "list_pending_approvals",
    "what should i review": "list_pending_approvals",
    "which approvals are pending": "list_pending_approvals",
    "approval broken": "approval_queue_summary",
    "approval button broken": "approval_queue_summary",
    "approval button error": "approval_queue_summary",
    "approval button failed": "approval_queue_summary",
    "approval button failure": "approval_queue_summary",
    "approval button issue": "approval_queue_summary",
    "approval button not working": "approval_queue_summary",
    "approval button problem": "approval_queue_summary",
    "approval button stuck": "approval_queue_summary",
    "approval dashboard": "approval_queue_summary",
    "approval error": "approval_queue_summary",
    "approval failed": "approval_queue_summary",
    "approval failure": "approval_queue_summary",
    "approval gate": "approval_queue_summary",
    "approval gate status": "approval_queue_summary",
    "approval gates": "approval_queue_summary",
    "approval health": "approval_queue_summary",
    "approval issue": "approval_queue_summary",
    "approval not working": "approval_queue_summary",
    "approval problem": "approval_queue_summary",
    "approval queue broken": "approval_queue_summary",
    "approval queue error": "approval_queue_summary",
    "approval queue failed": "approval_queue_summary",
    "approval queue failure": "approval_queue_summary",
    "approval queue issue": "approval_queue_summary",
    "approval queue not working": "approval_queue_summary",
    "approval queue problem": "approval_queue_summary",
    "approval queue stuck": "approval_queue_summary",
    "approval report": "approval_queue_summary",
    "approval status": "approval_queue_summary",
    "approval stuck": "approval_queue_summary",
    "approval summary": "approval_queue_summary",
    "approvals status": "approval_queue_summary",
    "approve button broken": "approval_queue_summary",
    "approve button error": "approval_queue_summary",
    "approve button failed": "approval_queue_summary",
    "approve button failure": "approval_queue_summary",
    "approve button issue": "approval_queue_summary",
    "approve button not working": "approval_queue_summary",
    "approve button problem": "approval_queue_summary",
    "approve button stuck": "approval_queue_summary",
    "approve failed": "approval_queue_summary",
    "approve failure": "approval_queue_summary",
    "approve not working": "approval_queue_summary",
    "approve problem": "approval_queue_summary",
    "approve stuck": "approval_queue_summary",
    "blocked by approval": "approval_queue_summary",
    "what is blocked by approval": "approval_queue_summary",
    "why did approval button fail": "approval_queue_summary",
    "why did approval fail": "approval_queue_summary",
    "why did approve fail": "approval_queue_summary",
    "diagnostic report": "jarvis_doctor",
    "diagnostic status": "jarvis_doctor",
    "diagnostics": "jarvis_doctor",
    "diagnostics report": "jarvis_doctor",
    "diagnostics status": "jarvis_doctor",
    "error report": "jarvis_doctor",
    "error status": "jarvis_doctor",
    "jarvis health": "jarvis_doctor",
    "system health": "jarvis_doctor",
    "is jarvis healthy": "jarvis_doctor",
    "health status": "capability_cockpit",
    "is everything healthy": "capability_cockpit",
    "overall health": "capability_cockpit",
    "overall status": "capability_cockpit",
    "status report": "capability_cockpit",
    "what is healthy": "capability_cockpit",
    "bootstrap status": "setup_check",
    "config status": "setup_check",
    "configuration status": "setup_check",
    "environment status": "setup_check",
    "env status": "setup_check",
    "is setup ready": "setup_check",
    "launcher status": "setup_check",
    "setup readiness": "setup_check",
    "setup ready": "setup_check",
    "startup status": "setup_check",
    "status config": "setup_check",
    "automation status": "list_scheduled_jobs",
    "automations status": "list_scheduled_jobs",
    "background job status": "list_scheduled_jobs",
    "brief status": "list_scheduled_jobs",
    "daily brief status": "list_scheduled_jobs",
    "is morning brief scheduled": "list_scheduled_jobs",
    "is the brief scheduled": "list_scheduled_jobs",
    "job status": "list_scheduled_jobs",
    "jobs status": "list_scheduled_jobs",
    "count scheduled jobs": "list_scheduled_jobs",
    "how many scheduled jobs": "list_scheduled_jobs",
    "morning brief health": "list_scheduled_jobs",
    "morning brief next run": "list_scheduled_jobs",
    "morning brief schedule": "list_scheduled_jobs",
    "morning brief status": "list_scheduled_jobs",
    "next morning brief": "list_scheduled_jobs",
    "recurring task status": "list_scheduled_jobs",
    "recurring tasks": "list_scheduled_jobs",
    "schedule status": "list_scheduled_jobs",
    "scheduled brief status": "list_scheduled_jobs",
    "scheduled job count": "list_scheduled_jobs",
    "scheduled job status": "list_scheduled_jobs",
    "scheduled jobs": "list_scheduled_jobs",
    "scheduled jobs count": "list_scheduled_jobs",
    "scheduled jobs status": "list_scheduled_jobs",
    "scheduler health": "list_scheduled_jobs",
    "scheduler report": "list_scheduled_jobs",
    "scheduler status": "list_scheduled_jobs",
    "what is scheduled": "list_scheduled_jobs",
    "what is the morning schedule": "list_scheduled_jobs",
    "what is the schedule": "list_scheduled_jobs",
    "what jobs are running": "list_scheduled_jobs",
    "when is morning brief": "list_scheduled_jobs",
    "when is next morning brief": "list_scheduled_jobs",
    "brain memory status": "memory_stats",
    "knowledge status": "memory_stats",
    "count memories": "memory_stats",
    "count memory": "memory_stats",
    "how many memories": "memory_stats",
    "how many memories do i have": "memory_stats",
    "how many memories does jarvis have": "memory_stats",
    "memory count": "memory_stats",
    "memories count": "memory_stats",
    "memory check": "memory_stats",
    "memory health": "memory_stats",
    "memory report": "memory_stats",
    "memory status": "memory_stats",
    "personal context status": "personal_context_status",
    "my personal context status": "personal_context_status",
    "my personal knowledge status": "personal_context_status",
    "personal knowledge status": "personal_context_status",
    "count jarvis notes": "list_jarvis_notes",
    "count notes": "list_jarvis_notes",
    "how many jarvis notes": "list_jarvis_notes",
    "how many jarvis notes do i have": "list_jarvis_notes",
    "how many notes": "list_jarvis_notes",
    "how many notes do i have": "list_jarvis_notes",
    "jarvis note count": "list_jarvis_notes",
    "jarvis notes count": "list_jarvis_notes",
    "note count": "list_jarvis_notes",
    "notes count": "list_jarvis_notes",
    "count goals": "list_goals",
    "goal count": "list_goals",
    "goals count": "list_goals",
    "how many goals": "list_goals",
    "how many goals do i have": "list_goals",
    "count tasks": "task_overview",
    "count todos": "task_overview",
    "how many tasks": "task_overview",
    "how many tasks do i have": "task_overview",
    "how many todos": "task_overview",
    "how many todos do i have": "task_overview",
    "task count": "task_overview",
    "tasks count": "task_overview",
    "todo count": "task_overview",
    "todos count": "task_overview",
    "count preferences": "list_preferences",
    "count preference": "list_preferences",
    "how many preferences": "list_preferences",
    "how many preferences do i have": "list_preferences",
    "preference count": "list_preferences",
    "preferences count": "list_preferences",
    "count decisions": "list_decisions",
    "count decision": "list_decisions",
    "decision count": "list_decisions",
    "decisions count": "list_decisions",
    "how many decisions": "list_decisions",
    "how many decisions do i have": "list_decisions",
    "count people": "list_people",
    "count person": "list_people",
    "how many people": "list_people",
    "how many people do i have": "list_people",
    "people count": "list_people",
    "person count": "list_people",
    "count skills": "list_skills",
    "count skill": "list_skills",
    "how many skills": "list_skills",
    "how many skills do i have": "list_skills",
    "skill count": "list_skills",
    "skills count": "list_skills",
    "memory review": "learning_review",
    "learning health": "learning_review",
    "learning loop status": "learning_review",
    "learning report": "learning_review",
    "learning status": "learning_review",
    "after action learning status": "after_action_learning_packet",
    "after-action learning status": "after_action_learning_packet",
    "failure learning status": "failure_learning_cockpit",
    "is learning debt closed": "execution_learning_closure_packet",
    "learning debt status": "execution_learning_closure_packet",
    "learning loop proof": "execution_learning_closure_packet",
    "learning proof matrix": "execution_learning_closure_packet",
    "review memories": "learning_review",
    "review memory": "learning_review",
    "what did jarvis learn from last run": "after_action_learning_packet",
    "what did jarvis learn from the last failure": "after_action_learning_packet",
    "what did jarvis learn from the last run": "after_action_learning_packet",
    "what did you learn": "learning_review",
    "what does jarvis remember": "recent_memories",
    "what have you learned": "learning_review",
    "what learning debt is open": "execution_learning_closure_packet",
    "what should jarvis learn": "learning_review",
    "count feedback": "feedback_report",
    "feedback count": "feedback_report",
    "feedback counts": "feedback_report",
    "how many feedback items": "feedback_report",
    "how much feedback": "feedback_report",
    "count sessions": "list_sessions",
    "conversation count": "list_sessions",
    "conversations count": "list_sessions",
    "how many conversations": "list_sessions",
    "how many sessions": "list_sessions",
    "session count": "list_sessions",
    "sessions count": "list_sessions",
    "asr status": "voice_setup_check",
    "is voice ready": "voice_setup_check",
    "mic check": "voice_setup_check",
    "mic status": "voice_setup_check",
    "microphone check": "voice_setup_check",
    "microphone status": "voice_setup_check",
    "push to talk status": "voice_setup_check",
    "voice check": "voice_setup_check",
    "voice input status": "voice_setup_check",
    "voice readiness": "voice_setup_check",
    "voice setup status": "voice_setup_check",
    "voice status": "voice_setup_check",
    "voice warmup status": "voice_setup_check",
    "whisper status": "voice_setup_check",
    "whisper warmup status": "voice_setup_check",
    "daemon proof": "capability_cockpit",
    "daemon proof matrix": "capability_cockpit",
    "daemon startup proof matrix": "capability_cockpit",
    "daemon startup status": "capability_cockpit",
    "daemon status": "capability_cockpit",
    "launchagent status": "capability_cockpit",
    "launchd status": "capability_cockpit",
    "network loss recovery status": "capability_cockpit",
    "network recovery status": "capability_cockpit",
    "reboot proof": "capability_cockpit",
    "reboot proof matrix": "capability_cockpit",
    "reboot readiness": "capability_cockpit",
    "reboot status": "capability_cockpit",
    "reboot survival status": "capability_cockpit",
    "scheduler daemon status": "capability_cockpit",
    "status server daemon status": "capability_cockpit",
    "telegram daemon status": "capability_cockpit",
    "restart telegram control daemon": "telegram_control_restart_guidance",
    "restart the telegram control daemon": "telegram_control_restart_guidance",
    "restart telegram daemon": "telegram_control_restart_guidance",
    "restart the telegram daemon": "telegram_control_restart_guidance",
    "restart telegram bot daemon": "telegram_control_restart_guidance",
    "restart it": "restart_target_clarification",
    "restart this": "restart_target_clarification",
    "restart that": "restart_target_clarification",
    "restart the telegram bot daemon": "telegram_control_restart_guidance",
    "will jarvis recover from network loss": "capability_cockpit",
    "will jarvis survive reboot": "capability_cockpit",
    "acceptance gaps": "capability_cockpit",
    "acceptance coverage": "capability_cockpit",
    "acceptance coverage drift": "capability_cockpit",
    "acceptance coverage status": "capability_cockpit",
    "acceptance harness status": "capability_cockpit",
    "acceptance harness rows": "capability_cockpit",
    "acceptance next proof": "completion_next_proof_packet",
    "aggregate smoke status": "capability_cockpit",
    "are smokes green": "capability_cockpit",
    "are tests green": "capability_cockpit",
    "does live check cover acceptance": "capability_cockpit",
    "does live check cover every checklist section": "capability_cockpit",
    "does live check cover the august checklist": "capability_cockpit",
    "is korean voice covered": "capability_cockpit",
    "is korean voice tested": "capability_cockpit",
    "korean voice proof status": "capability_cockpit",
    "is live check missing rows": "capability_cockpit",
    "live check": "capability_cockpit",
    "live check coverage": "capability_cockpit",
    "live check missing rows": "capability_cockpit",
    "live check one screen table": "capability_cockpit",
    "live check rows": "capability_cockpit",
    "live check status": "capability_cockpit",
    "live proof status": "capability_cockpit",
    "latest aggregate smoke proof": "capability_cockpit",
    "one screen acceptance": "capability_cockpit",
    "one screen acceptance table": "capability_cockpit",
    "one screen live check table": "capability_cockpit",
    "full smoke status": "capability_cockpit",
    "smoke lock status": "capability_cockpit",
    "smoke status": "capability_cockpit",
    "smoke suite lock status": "capability_cockpit",
    "smoke suite status": "capability_cockpit",
    "smoke test status": "capability_cockpit",
    "suite lock status": "capability_cockpit",
    "telegram voice proof status": "capability_cockpit",
    "test coverage": "capability_cockpit",
    "test status": "capability_cockpit",
    "voice proof status": "capability_cockpit",
    "august checklist": "capability_cockpit",
    "definition of done": "capability_cockpit",
    "evidence status": "evidence_ledger",
    "finish plan": "capability_cockpit",
    "next acceptance proof": "completion_next_proof_packet",
    "next acceptance test": "completion_next_proof_packet",
    "next live proof": "completion_next_proof_packet",
    "next live test": "completion_next_proof_packet",
    "next proof": "completion_next_proof_packet",
    "next proof to run": "completion_next_proof_packet",
    "proofs": "evidence_ledger",
    "show proofs": "evidence_ledger",
    "show me proofs": "evidence_ledger",
    "what broke status": "recent_tool_runs",
    "call check": "channel_health",
    "call channel status": "channel_health",
    "call health": "channel_health",
    "call status": "channel_health",
    "channel check": "channel_health",
    "check call": "channel_health",
    "check call status": "channel_health",
    "check channel": "channel_health",
    "check channel health": "channel_health",
    "check channel status": "channel_health",
    "check facetime": "channel_health",
    "check facetime status": "channel_health",
    "check imessage": "channel_health",
    "check imessage status": "channel_health",
    "check instagram": "channel_health",
    "check instagram status": "channel_health",
    "check kakao": "channel_health",
    "check kakao status": "channel_health",
    "check message": "channel_health",
    "check message status": "channel_health",
    "check messaging": "channel_health",
    "check phone": "channel_health",
    "check phone status": "channel_health",
    "check telegram": "channel_health",
    "check telegram health": "channel_health",
    "check telegram status": "channel_health",
    "facetime check": "channel_health",
    "facetime channel status": "channel_health",
    "facetime health": "channel_health",
    "facetime status": "channel_health",
    "imessage check": "channel_health",
    "imessage channel status": "channel_health",
    "imessage health": "channel_health",
    "imessage status": "channel_health",
    "instagram check": "channel_health",
    "instagram channel status": "channel_health",
    "instagram health": "channel_health",
    "instagram status": "channel_health",
    "kakao check": "channel_health",
    "kakao channel status": "channel_health",
    "kakao health": "channel_health",
    "kakao status": "channel_health",
    "message check": "channel_health",
    "message channel status": "channel_health",
    "message health": "channel_health",
    "message status": "channel_health",
    "messaging check": "channel_health",
    "messages status": "channel_health",
    "messaging status": "channel_health",
    "mobile status": "channel_health",
    "owner phone status": "channel_health",
    "phone check": "channel_health",
    "phone call status": "channel_health",
    "phone channel status": "channel_health",
    "phone health": "channel_health",
    "phone status": "channel_health",
    "telegram check": "channel_health",
    "telegram channel status": "channel_health",
    "telegram health": "channel_health",
    "telegram status": "channel_health",
    "open acceptance gaps": "capability_cockpit",
    "open checklist items": "capability_cockpit",
    "remaining acceptance gaps": "capability_cockpit",
    "remaining checklist items": "capability_cockpit",
    "cockpit next action": "safe_next_actions",
    "next cockpit action": "safe_next_actions",
    "show august checklist": "capability_cockpit",
    "show finish plan": "capability_cockpit",
    "show one screen acceptance table": "capability_cockpit",
    "unfinished jarvis work": "capability_cockpit",
    "unverified checklist items": "capability_cockpit",
    "what acceptance items remain": "capability_cockpit",
    "what acceptance proof is next": "completion_next_proof_packet",
    "what command should i run next": "safe_next_actions",
    "what definition of done items are open": "capability_cockpit",
    "what dod items are open": "capability_cockpit",
    "what acceptance proofs are pending": "capability_cockpit",
    "what acceptance tests are pending": "capability_cockpit",
    "what do i need to prove": "capability_cockpit",
    "what does the operator need to test": "capability_cockpit",
    "what rows are in live check": "capability_cockpit",
    "what rows does live check show": "capability_cockpit",
    "what is jarvis finish line": "capability_cockpit",
    "what is jarvis s finish line": "capability_cockpit",
    "what is left for august": "capability_cockpit",
    "what is left before august": "capability_cockpit",
    "what is left before jarvis is done": "capability_cockpit",
    "what is left to finish": "capability_cockpit",
    "what is left unfinished": "capability_cockpit",
    "what s not finished in jarvis": "capability_cockpit",
    "what is missing": "capability_cockpit",
    "what is not done": "capability_cockpit",
    "what is still missing": "capability_cockpit",
    "what is still not done": "capability_cockpit",
    "what is still open": "capability_cockpit",
    "what is unfinished": "capability_cockpit",
    "what is the august finish line": "capability_cockpit",
    "what is the finish line": "capability_cockpit",
    "what live proof is missing": "capability_cockpit",
    "what live proofs are missing": "capability_cockpit",
    "what live proofs should i run": "capability_cockpit",
    "what live tests should i run": "capability_cockpit",
    "what open acceptance items remain": "capability_cockpit",
    "what has been tested": "capability_cockpit",
    "what have we verified": "capability_cockpit",
    "what is verified": "capability_cockpit",
    "what is the next acceptance proof": "completion_next_proof_packet",
    "what proof is next": "completion_next_proof_packet",
    "what proof should i run next": "completion_next_proof_packet",
    "what remains before jarvis is done": "capability_cockpit",
    "what remains to finish jarvis": "capability_cockpit",
    "what remains unfinished": "capability_cockpit",
    "what remains unproven": "capability_cockpit",
    "what needs attention": "capability_cockpit",
    "what needs my attention": "capability_cockpit",
    "what needs review": "capability_cockpit",
    "what should i check next": "safe_next_actions",
    "what should i test next": "completion_next_proof_packet",
    "what should operator test next": "completion_next_proof_packet",
    "which proof is next": "completion_next_proof_packet",
    "what s left to finish": "capability_cockpit",
    "whats left to finish": "capability_cockpit",
    "what should i prove next": "completion_next_proof_packet",
    "what should i run next": "safe_next_actions",
    "what should i verify live": "capability_cockpit",
    "what should the operator prove": "capability_cockpit",
    "what should operator test live": "capability_cockpit",
    "which acceptance items are blocked": "capability_cockpit",
    "which checklist items are blocked": "capability_cockpit",
    "which lanes need attention": "capability_cockpit",
    "jarvis remaining work": "capability_cockpit",
    "jarvis unfinished work": "capability_cockpit",
    "what blocks august completion": "completion_audit_packet",
    "what blocks completion": "completion_audit_packet",
    "completion status": "completion_audit_packet",
    "jarvis completion status": "completion_audit_packet",
    "did acceptance pass": "completion_claim_gate",
    "done status": "completion_claim_gate",
    "has jarvis passed acceptance": "completion_claim_gate",
    "jarvis done status": "completion_claim_gate",
    "what is blocking august completion": "completion_audit_packet",
    "what is blocking completion": "completion_audit_packet",
    "what is blocking jarvis from being done": "completion_audit_packet",
    "whats blocking completion": "completion_audit_packet",
    "approval button proof matrix": "capability_cockpit",
    "approval button result format": "capability_cockpit",
    "approval buttons proof matrix": "capability_cockpit",
    "approval callback proof matrix": "capability_cockpit",
    "approval callback result format": "capability_cockpit",
    "approval from phone proof matrix": "capability_cockpit",
    "approval queue button proof matrix": "capability_cockpit",
    "approval proof matrix": "capability_cockpit",
    "approval proof result format": "capability_cockpit",
    "approve deny proof matrix": "capability_cockpit",
    "approve dismiss proof matrix": "capability_cockpit",
    "approve from phone proof matrix": "capability_cockpit",
    "brief delivery proof matrix": "capability_cockpit",
    "chat latency proof matrix": "capability_cockpit",
    "chat latency proof result format": "capability_cockpit",
    "conversation proof matrix": "capability_cockpit",
    "conversation proof result format": "capability_cockpit",
    "conversation research proof matrix": "capability_cockpit",
    "conversation research proof result format": "capability_cockpit",
    "daily value proof matrix": "capability_cockpit",
    "daily value proof result format": "capability_cockpit",
    "daemon proof result format": "capability_cockpit",
    "did calendar write pass": "capability_cockpit",
    "did email send pass": "capability_cockpit",
    "did imessage pass": "capability_cockpit",
    "did kakao pass": "capability_cockpit",
    "did morning brief pass": "capability_cockpit",
    "did reboot proof pass": "capability_cockpit",
    "did reminders pass": "capability_cockpit",
    "did telegram pass": "capability_cockpit",
    "did voice pass": "capability_cockpit",
    "are error messages actionable": "capability_cockpit",
    "connector error proof matrix": "capability_cockpit",
    "connector recovery proof matrix": "capability_cockpit",
    "do error messages name the fix": "capability_cockpit",
    "do user errors name the fix": "capability_cockpit",
    "error guidance proof matrix": "capability_cockpit",
    "error guidance proof result format": "capability_cockpit",
    "error message proof matrix": "capability_cockpit",
    "error message proof result format": "capability_cockpit",
    "error messages status": "capability_cockpit",
    "error recovery proof": "capability_cockpit",
    "error recovery status": "capability_cockpit",
    "how do i prove approval flow": "capability_cockpit",
    "how do i prove calendar write": "capability_cockpit",
    "how do i prove contact lookup": "capability_cockpit",
    "how do i prove daemon recovery": "capability_cockpit",
    "how do i prove email read": "capability_cockpit",
    "how do i prove email search": "capability_cockpit",
    "how do i prove email send": "capability_cockpit",
    "how do i prove korean voice": "capability_cockpit",
    "how do i prove morning brief": "capability_cockpit",
    "how do i prove phone approval": "capability_cockpit",
    "how do i prove reboot survival": "capability_cockpit",
    "how do i prove reminders": "capability_cockpit",
    "how do i prove telegram voice": "capability_cockpit",
    "how do i test approval flow": "capability_cockpit",
    "how do i test calendar write": "capability_cockpit",
    "how do i test contact lookup": "capability_cockpit",
    "how do i test daemon recovery": "capability_cockpit",
    "how do i test email read": "capability_cockpit",
    "how do i test email search": "capability_cockpit",
    "how do i test email send": "capability_cockpit",
    "how do i test korean voice": "capability_cockpit",
    "how do i test morning brief": "capability_cockpit",
    "how do i test phone approval": "capability_cockpit",
    "how do i test reboot survival": "capability_cockpit",
    "how do i test reminders": "capability_cockpit",
    "how do i test telegram voice": "capability_cockpit",
    "calendar create update delete proof": "capability_cockpit",
    "calendar create update delete proof matrix": "capability_cockpit",
    "calendar live proof status": "capability_cockpit",
    "calendar proof status": "capability_cockpit",
    "calendar status": "capability_cockpit",
    "calendar write proof": "capability_cockpit",
    "calendar write proof matrix": "capability_cockpit",
    "calendar write proof status": "capability_cockpit",
    "calendar write result format": "capability_cockpit",
    "calendar write status": "capability_cockpit",
    "contact lookup proof matrix": "capability_cockpit",
    "contact lookup status": "capability_cockpit",
    "contact lookup proof": "capability_cockpit",
    "contact lookup proof status": "capability_cockpit",
    "contacts proof": "capability_cockpit",
    "contacts proof status": "capability_cockpit",
    "contacts status": "capability_cockpit",
    "email proof": "capability_cockpit",
    "email proof status": "capability_cockpit",
    "email read proof": "capability_cockpit",
    "email read proof matrix": "capability_cockpit",
    "email read status": "capability_cockpit",
    "email search proof": "capability_cockpit",
    "email search proof matrix": "capability_cockpit",
    "email search status": "capability_cockpit",
    "email send proof": "capability_cockpit",
    "email send proof matrix": "capability_cockpit",
    "email send status": "capability_cockpit",
    "email status": "capability_cockpit",
    "approval flow status": "capability_cockpit",
    "cockpit status": "capability_cockpit",
    "error guidance status": "capability_cockpit",
    "how should i report daily value proof": "capability_cockpit",
    "how should i report calendar write proof": "capability_cockpit",
    "how should i report contact lookup proof": "capability_cockpit",
    "how should i report email read proof": "capability_cockpit",
    "how should i report email search proof": "capability_cockpit",
    "how should i report email send proof": "capability_cockpit",
    "how should i report error messages": "capability_cockpit",
    "how should i report error guidance proof": "capability_cockpit",
    "how should i report error recovery proof": "capability_cockpit",
    "how should i report reminder proof": "capability_cockpit",
    "how should i report set reminder proof": "capability_cockpit",
    "how should i report chat latency proof": "capability_cockpit",
    "how should i report conversation proof": "capability_cockpit",
    "how should i report mixed conversation proof": "capability_cockpit",
    "how should i report personal proof results": "capability_cockpit",
    "how should i report phone control proof": "capability_cockpit",
    "how should i report scheduler proof": "capability_cockpit",
    "how should i report scheduled job streak proof": "capability_cockpit",
    "how should i report korean voice proof": "capability_cockpit",
    "how should i report local voice proof": "capability_cockpit",
    "how should i report network recovery proof": "capability_cockpit",
    "how should i report telegram voice proof": "capability_cockpit",
    "how should i report voice proof": "capability_cockpit",
    "are scheduled jobs running daily": "capability_cockpit",
    "daily job streak proof": "capability_cockpit",
    "daily streak proof": "capability_cockpit",
    "korean voice status": "capability_cockpit",
    "korean voice proof matrix": "capability_cockpit",
    "korean voice proof result format": "capability_cockpit",
    "local voice proof matrix": "capability_cockpit",
    "local voice proof result format": "capability_cockpit",
    "mixed conversation proof matrix": "capability_cockpit",
    "mixed conversation proof result format": "capability_cockpit",
    "morning brief proof matrix": "capability_cockpit",
    "morning brief proof result format": "capability_cockpit",
    "morning brief delivery proof matrix": "capability_cockpit",
    "network loss proof matrix": "capability_cockpit",
    "network loss proof result format": "capability_cockpit",
    "network recovery proof matrix": "capability_cockpit",
    "network recovery proof result format": "capability_cockpit",
    "personal integration proof gaps": "capability_cockpit",
    "personal integration proof matrix": "capability_cockpit",
    "personal integration proof status": "capability_cockpit",
    "personal integration result format": "capability_cockpit",
    "personal integrations proof gaps": "capability_cockpit",
    "personal integrations proof matrix": "capability_cockpit",
    "personal integrations proof status": "capability_cockpit",
    "personal integrations result format": "capability_cockpit",
    "personal proof matrix": "capability_cockpit",
    "personal proof result format": "capability_cockpit",
    "personal proof status": "capability_cockpit",
    "personal proofs status": "capability_cockpit",
    "phone approval proof": "capability_cockpit",
    "phone approval buttons proof matrix": "capability_cockpit",
    "phone approval result format": "capability_cockpit",
    "phone approval status": "capability_cockpit",
    "phone approval-flow proof report format": "capability_cockpit",
    "phone approve deny proof matrix": "capability_cockpit",
    "phone control proof matrix": "capability_cockpit",
    "phone control proof result format": "capability_cockpit",
    "reboot proof result format": "capability_cockpit",
    "research proof matrix": "capability_cockpit",
    "reminder proof": "capability_cockpit",
    "reminder proof matrix": "capability_cockpit",
    "reminder proof status": "capability_cockpit",
    "reminder status": "capability_cockpit",
    "reminders status": "capability_cockpit",
    "scheduled jobs proof matrix": "capability_cockpit",
    "scheduled jobs 7 day proof": "capability_cockpit",
    "scheduled jobs streak": "capability_cockpit",
    "scheduled job streak status": "capability_cockpit",
    "scheduled delivery proof matrix": "capability_cockpit",
    "scheduler daemon proof matrix": "capability_cockpit",
    "scheduler proof matrix": "capability_cockpit",
    "scheduler proof result format": "capability_cockpit",
    "scheduler streak status": "capability_cockpit",
    "section a proof matrix": "capability_cockpit",
    "set reminder proof": "capability_cockpit",
    "set reminder proof matrix": "capability_cockpit",
    "set reminder proof status": "capability_cockpit",
    "telegram voice status": "capability_cockpit",
    "telegram approval buttons proof matrix": "capability_cockpit",
    "telegram approval result format": "capability_cockpit",
    "telegram voice proof matrix": "capability_cockpit",
    "telegram voice proof result format": "capability_cockpit",
    "telegram proof passed": "capability_cockpit",
    "user error proof matrix": "capability_cockpit",
    "user visible error proof matrix": "capability_cockpit",
    "voice live proof format": "capability_cockpit",
    "voice proof matrix": "capability_cockpit",
    "voice proof result format": "capability_cockpit",
    "push to talk proof matrix": "capability_cockpit",
    "push-to-talk proof matrix": "capability_cockpit",
    "spoken reply proof matrix": "capability_cockpit",
    "talk.py proof matrix": "capability_cockpit",
    "talk py proof matrix": "capability_cockpit",
    "voice speak proof matrix": "capability_cockpit",
    "web research proof matrix": "capability_cockpit",
    "what approval flow should i test": "capability_cockpit",
    "what approval buttons should i test": "capability_cockpit",
    "what approval callback should i test": "capability_cockpit",
    "what approve deny buttons should i test": "capability_cockpit",
    "what calendar write should i test": "capability_cockpit",
    "what contact lookup should i test": "capability_cockpit",
    "what daily value should i test": "capability_cockpit",
    "what email read should i test": "capability_cockpit",
    "what email search should i test": "capability_cockpit",
    "what email send should i test": "capability_cockpit",
    "what error messages should i test": "capability_cockpit",
    "what recovery guidance should i test": "capability_cockpit",
    "what reminder proof should i test": "capability_cockpit",
    "what set reminder proof should i test": "capability_cockpit",
    "what user facing errors need proof": "capability_cockpit",
    "what personal integrations should i test": "capability_cockpit",
    "what personal integration proofs are missing": "capability_cockpit",
    "what personal integrations are unproven": "capability_cockpit",
    "what personal integrations proofs are missing": "capability_cockpit",
    "what personal proofs should i test": "capability_cockpit",
    "what phone control should i test": "capability_cockpit",
    "what phone approval should i test": "capability_cockpit",
    "what phone approvals should i test": "capability_cockpit",
    "what research proof should i test": "capability_cockpit",
    "what chat proof should i run": "capability_cockpit",
    "what conversation should i test": "capability_cockpit",
    "what mixed conversation should i test": "capability_cockpit",
    "what korean voice should i test": "capability_cockpit",
    "what local voice should i test": "capability_cockpit",
    "what telegram voice should i test": "capability_cockpit",
    "what voice proofs should i test": "capability_cockpit",
    "what scheduled jobs should i test": "capability_cockpit",
    "what scheduler streak should i test": "capability_cockpit",
    "what network recovery should i test": "capability_cockpit",
    "what should i test after reboot": "capability_cockpit",
    "what is the 7 day scheduler proof": "capability_cockpit",
    "did scheduled jobs run for 7 days": "capability_cockpit",
    "job streak proof matrix": "capability_cockpit",
    "jobs 7 day proof": "capability_cockpit",
    "jobs seven day proof": "capability_cockpit",
    "post reboot proof matrix": "capability_cockpit",
    "post-reboot proof matrix": "capability_cockpit",
    "launchd proof matrix": "capability_cockpit",
    "telegram control daemon status": "capability_cockpit",
    "telegram control proof matrix": "capability_cockpit",
    "status server proof matrix": "capability_cockpit",
    "what voice should i test": "capability_cockpit",
    "what broke proof matrix": "capability_cockpit",
    "deny from phone proof matrix": "capability_cockpit",
    "how should i report approval button proof": "capability_cockpit",
    "how should i report approval callback proof": "capability_cockpit",
    "how should i report approve deny proof": "capability_cockpit",
    "how should i report phone approval proof": "capability_cockpit",
    "risky action approval proof matrix": "capability_cockpit",
    "cockpit approval proof matrix": "capability_cockpit",
    "cockpit approvals proof matrix": "capability_cockpit",
    "cockpit proof matrix": "capability_cockpit",
    "is calendar write proven": "capability_cockpit",
    "is contact lookup proven": "capability_cockpit",
    "is email send proven": "capability_cockpit",
    "is imessage proven": "capability_cockpit",
    "is kakao proven": "capability_cockpit",
    "is korean voice proven": "capability_cockpit",
    "is local talk proven": "capability_cockpit",
    "is morning brief proven": "capability_cockpit",
    "is reboot proven": "capability_cockpit",
    "is error guidance proven": "capability_cockpit",
    "is telegram proven": "capability_cockpit",
    "recovery guidance proof matrix": "capability_cockpit",
    "recovery guidance status": "capability_cockpit",
    "show error guidance": "capability_cockpit",
    "show recovery guidance": "capability_cockpit",
    "agent fleet status": "subagent_fleet_status",
    "agent fleet health": "subagent_fleet_status",
    "agent readiness": "subagent_fleet_status",
    "agent status": "subagent_fleet_status",
    "agent health": "subagent_fleet_status",
    "agents ready": "subagent_fleet_status",
    "agents health": "subagent_fleet_status",
    "are agents ready": "subagent_fleet_status",
    "are my agents ready": "subagent_fleet_status",
    "how many agents are ready": "subagent_fleet_status",
    "internal agents status": "subagent_fleet_status",
    "internal orchestration": "subagent_fleet_status",
    "internal orchestration status": "subagent_fleet_status",
    "internal worker status": "subagent_fleet_status",
    "orchestration status": "subagent_fleet_status",
    "parallel agent status": "subagent_fleet_status",
    "parallel agents status": "subagent_fleet_status",
    "ready agent count": "subagent_fleet_status",
    "ready agents": "subagent_fleet_status",
    "ready agents status": "subagent_fleet_status",
    "subagent fleet": "subagent_fleet_status",
    "subagent fleet health": "subagent_fleet_status",
    "subagent fleet readiness": "subagent_fleet_status",
    "subagent fleet status": "subagent_fleet_status",
    "subagent health": "subagent_fleet_status",
    "subagent readiness": "subagent_fleet_status",
    "subagent status": "subagent_fleet_status",
    "show agent status": "subagent_fleet_status",
    "show subagent status": "subagent_fleet_status",
    "show worker status": "subagent_fleet_status",
    "what agents are ready": "subagent_fleet_status",
    "what agents are running": "subagent_fleet_status",
    "which agents are ready": "subagent_fleet_status",
    "which agents are running": "subagent_fleet_status",
    "tool orchestration status": "subagent_fleet_status",
    "worker fleet status": "subagent_fleet_status",
    "worker fleet health": "subagent_fleet_status",
    "worker readiness": "subagent_fleet_status",
    "worker health": "subagent_fleet_status",
    "worker status": "subagent_fleet_status",
    "show storage stats": "storage_status",
    "storage stats": "storage_status",
    "storage statistics": "storage_status",
    "what is storage status": "storage_status",
}
_RUNTIME_LOCAL_SAFE_PROJECTION_REPAIR_ALIASES = {
    "repair memory projections": "repair_memory_projections",
    "repair pending memory projections": "repair_memory_projections",
    "repair decision projections": "repair_decision_projections",
    "repair pending decision projections": "repair_decision_projections",
    "repair person projections": "repair_person_projections",
    "repair pending person projections": "repair_person_projections",
    "repair preference projections": "repair_preference_projections",
    "repair pending preference projections": "repair_preference_projections",
}
_RUNTIME_EXACT_TOOL_ALIAS_RAW = {
    "approval_queue_summary": "approval_queue_summary",
    "capability_map": "capability_map",
    "channel_health": "channel_health",
    "completion_claim_gate": "completion_claim_gate",
    "evidence_ledger": "evidence_ledger",
    "execution_health_report": "execution_health_report",
    "harness_status": "harness_status",
    "harness_readiness_digest": "harness_readiness_digest",
    "jarvis_doctor": "jarvis_doctor",
    "jarvis_status": "jarvis_status",
    "learning_review": "learning_review",
    "list_scheduled_jobs": "list_scheduled_jobs",
    "list_tools": "list_tools",
    "memory_stats": "memory_stats",
    "model_routing_status": "model_routing_status",
    "privacy_report": "privacy_report",
    "readiness_report": "readiness_report",
    "recent_tool_runs": "recent_tool_runs",
    "risk_matrix": "risk_matrix",
    "safety_status": "safety_status",
    "storage_recovery_check": "storage_recovery_check",
    "storage_recovery_plan": "storage_recovery_plan",
    "storage_status": "storage_status",
    "subagent_fleet_status": "subagent_fleet_status",
}
_RUNTIME_EXACT_TOOL_ALIAS_RAW_EXCLUDES = {
    # This read-only tool is not useful as a no-arg raw command; it needs a
    # transcript and proof fields to produce a verified cockpit packet.
    "voice_command_cockpit",
}
_RUNTIME_EXACT_TOOL_ALIAS_DYNAMIC_NO_ARG_NAMES = {
    "after_action_learning_packet",
    "computer_control_readiness",
    "computer_control_status",
    "current_time",
    "agi_next_build_move",
    "autonomy_continuation_execution_packet",
    "autonomy_resume_gate",
    "autonomy_step_closure_packet",
    "build_target_packet",
    "chat_continuity_brief",
    "chat_response_health",
    "checkpoint_recovery_cockpit",
    "checkpoint_recovery_apply_packet",
    "checkpoint_recovery_followthrough_packet",
    "checkpoint_recovery_preview",
    "coding_discipline_packet",
    "completion_audit_packet",
    "completion_next_proof_packet",
    "completion_proof_refresh_packet",
    "continuation_packet",
    "execution_case_closure_packet",
    "execution_case_evidence_packet",
    "execution_case_gate",
    "execution_case_review_packet",
    "execution_case_timeline",
    "execution_audit_gate",
    "execution_learning_closure_packet",
    "execution_recovery_packet",
    "failure_learning_cockpit",
    "feedback_actions",
    "flip_coin",
    "focus_brief",
    "generate_password",
    "generate_uuid",
    "handoff_brief",
    "harness_build_slice",
    "harness_completion_assessment",
    "harness_operations_brief",
    "inspect_execution_case",
    "integration_execution_matrix",
    "jarvis_help",
    "list_decisions",
    "list_duplicate_memories",
    "list_goals",
    "list_people",
    "list_preferences",
    "list_reminders",
    "list_sessions",
    "list_skills",
    "list_tasks",
    "list_voices",
    "list_weak_memories",
    "morning_startup",
    "next_actions",
    "next_action_packet",
    "next_session_plan",
    "next_task",
    "overdue_tasks",
    "observe_act_verify_cockpit",
    "operator_instruction_supersession_packet",
    "operator_handoff_packet",
    "operator_timebox_contract",
    "priority_goal",
    "priority_stack",
    "prototype_readiness_checklist",
    "recent_conversation",
    "recent_memories",
    "recent_saved_notes",
    "recovery_closure_checklist",
    "repeated_failure_clusters",
    "return_brief",
    "safe_next_actions",
    "scheduler_context_refresh_packet",
    "session_closeout",
    "setup_check",
    "system_info",
    "task_board",
    "task_overview",
    "voice_setup_check",
    "work_block_checkpoint",
    "work_queue",
    "work_session_packet",
}
_RUNTIME_EXACT_TOOL_ALIAS_DYNAMIC_NO_ARG_SUFFIXES = (
    "_dashboard",
    "_digest",
    "_doctrine",
    "_history",
    "_ledger",
    "_map",
    "_report",
    "_review",
    "_stats",
    "_status",
)
_RUNTIME_EXACT_TOOL_ALIAS_DYNAMIC_ARG_SUFFIXES = (
    "_acceptance",
    "_audit",
    "_bridge",
    "_checklist",
    "_closure",
    "_contract",
    "_gate",
    "_handoff",
    "_packet",
    "_plan",
    "_preview",
    "_receipt",
    "_rehearsal",
    "_runbook",
)
_RUNTIME_EXACT_TOOL_ALIAS_DYNAMIC_ARG_PREFIXES = (
    "calculate",
    "choose_",
    "convert_",
    "count_",
    "days_",
    "extract_",
    "fetch_",
    "get_",
    "inspect_",
    "random_",
    "read_",
    "roll_",
    "screen_",
    "search_",
    "spell_",
    "summarize_",
    "tool_",
    "transform_",
    "verification_",
    "voice_",
    "web_",
)
_RUNTIME_EXACT_TOOL_REQUIRED_ARG_EXAMPLES = {
    "action_readiness_packet": "action readiness: run a script and email me the result",
    "action_rehearsal": "rehearse: run command python3 --version",
    "argument_contract_packet": "argument contract: run command python3 --version",
    "approval_chain_proof": "approval chain proof <approval id>",
    "approval_execution_packet": "approval packet <approval id>",
    "approval_readiness_packet": "approval readiness <approval id>",
    "choose_option": "choose option: <option A> | <option B>",
    "command_cockpit_packet": "command cockpit: organize my downloads and summarize what changed",
    "command_diagnosis": "command diagnosis: run command python3 --version",
    "command_intake_packet": "command intake: organize my downloads and summarize what changed",
    "computer_task_plan": "computer task plan: <task>",
    "days_until": "days until 2026-12-31",
    "dispatch_decision_packet": "dispatch decision: summarize my recent Jarvis work",
    "execution_acceptance_gate": (
        "acceptance gate: organize downloads; evidence recent tool run ok; tests smoke passed"
    ),
    "execution_contract": "execution contract: organize my downloads and summarize what changed",
    "execution_governor_packet": "execution governor: organize my downloads and summarize what changed",
    "execution_readiness_matrix": (
        "execution readiness matrix: organize my downloads and summarize what changed"
    ),
    "failure_promotion_packet": "failure promotion packet: dashboard layout",
    "failure_to_test_preview": "failure to test preview: Jarvis overlapped dashboard text",
    "extract_links": "extract links https://example.com",
    "fetch_page": "fetch page https://example.com",
    "get_memory": "get memory <query>",
    "planner_gap_packet": "planner gap: organize my downloads",
    "random_number": "random number 1 to 10",
    "roll_dice": "roll 2d6",
    "screen_verification_contract": "screen verification: <expected screen state>",
    "tool_detail": "tool detail: <tool name>",
    "verification_packet": "verification packet: organize my downloads and summarize what changed",
    "verification_receipt": "verification receipt <run id>",
    "voice_command_cockpit": "voice command cockpit: <transcript>",
    "voice_confirmation_packet": "voice confirmation: <transcript>",
    "voice_confirmation_receipt": "voice confirmation receipt: <transcript> confirmed=true",
    "voice_confirmation_audit_ledger": (
        "voice confirmation audit ledger: <transcript>; confirmed=true; "
        "privacy_receipt_id=<receipt>; receipt_id=<receipt>; receipt_nonce=<nonce>"
    ),
    "voice_route_gate_packet": "voice route gate: <transcript> confirmed=true",
    "voice_route_proof_bundle": "voice route proof bundle: <transcript> confirmed=true",
    "voice_runtime_bridge_packet": "voice runtime bridge: <transcript> confirmed=true",
    "voice_transcript_review": "voice transcript review: <transcript>",
    "voice_action_audit_packet": "voice action audit: <transcript> confirmed=true",
    "voice_execution_handoff_packet": "voice execution handoff: <transcript> confirmed=true",
    "voice_post_run_closure_packet": "voice post-run closure: <transcript> confirmed=true",
    "voice_cycle_ledger": "voice cycle ledger: <transcript> confirmed=true",
    "voice_audio_file_gate_packet": "voice audio file gate: /path/to/audio.m4a consent=true",
    "voice_reply_preview": "voice reply preview: <text>",
    "web_lookup": "web lookup <query>",
    "web_search": "web search <query>",
}
_RUNTIME_REQUIRED_ARG_COMMAND_STUBS = {
    "approval chain proof": "approval_chain_proof",
    "approval packet": "approval_execution_packet",
    "approval readiness": "approval_readiness_packet",
    "approval readiness packet": "approval_readiness_packet",
    "choose option": "choose_option",
    "computer task plan": "computer_task_plan",
    "days until": "days_until",
    "extract links": "extract_links",
    "fetch page": "fetch_page",
    "get memory": "get_memory",
    "screen verification": "screen_verification_contract",
    "screen verification contract": "screen_verification_contract",
    "tool detail": "tool_detail",
    "verification receipt for": "verification_receipt",
    "voice command cockpit": "voice_command_cockpit",
    "voice confirmation": "voice_confirmation_packet",
    "voice confirmation audit ledger": "voice_confirmation_audit_ledger",
    "voice confirmation ledger": "voice_confirmation_audit_ledger",
    "voice confirmation packet": "voice_confirmation_packet",
    "voice confirmation receipt": "voice_confirmation_receipt",
    "voice route gate": "voice_route_gate_packet",
    "voice route gate packet": "voice_route_gate_packet",
    "voice route proof bundle": "voice_route_proof_bundle",
    "voice runtime bridge": "voice_runtime_bridge_packet",
    "voice runtime bridge packet": "voice_runtime_bridge_packet",
    "web search": "web_search",
    "voice transcript review": "voice_transcript_review",
    "voice action audit": "voice_action_audit_packet",
    "voice action audit packet": "voice_action_audit_packet",
    "voice execution handoff": "voice_execution_handoff_packet",
    "voice execution handoff packet": "voice_execution_handoff_packet",
    "voice post run closure": "voice_post_run_closure_packet",
    "voice post run closure packet": "voice_post_run_closure_packet",
    "voice postrun closure": "voice_post_run_closure_packet",
    "voice cycle ledger": "voice_cycle_ledger",
    "voice audio file gate": "voice_audio_file_gate_packet",
    "voice audio file gate packet": "voice_audio_file_gate_packet",
    "voice reply preview": "voice_reply_preview",
    "web lookup": "web_lookup",
}
_RUNTIME_REQUIRED_ARG_RISKY_COMMAND_STUBS = {
    "execute a command": "run_shell_command",
    "execute command": "run_shell_command",
    "execute shell command": "run_shell_command",
    "execute a shell command": "run_shell_command",
    "execute terminal command": "run_shell_command",
    "execute a terminal command": "run_shell_command",
    "run a command": "run_shell_command",
    "run command": "run_shell_command",
    "run command please": "run_shell_command",
    "run shell command": "run_shell_command",
    "run a shell command": "run_shell_command",
    "run terminal command": "run_shell_command",
    "run a terminal command": "run_shell_command",
    "shell command": "run_shell_command",
    "terminal command": "run_shell_command",
}
_RUNTIME_RISKY_TOOL_REQUIRED_ARG_EXAMPLES = {
    "run_shell_command": "run command <command>",
}
_RUNTIME_EXACT_TOOL_ALIAS_COMPACT = {
    "콕핏": "capability_cockpit",
    "콕핏요약": "capability_cockpit",
    "검수체크리스트": "capability_cockpit",
    "남은기준": "capability_cockpit",
    "다음라이브증명": "completion_next_proof_packet",
    "다음라이브테스트": "completion_next_proof_packet",
    "다음에뭐증명해": "completion_next_proof_packet",
    "다음에뭐테스트해": "completion_next_proof_packet",
    "다음증명": "completion_next_proof_packet",
    "다음증명뭐야": "completion_next_proof_packet",
    "뭐남았어": "capability_cockpit",
    "뭐증명해야해": "capability_cockpit",
    "완료뭐남았어": "capability_cockpit",
    "완료기준": "capability_cockpit",
    "완료뭐막혀": "completion_audit_packet",
    "검토필요": "capability_cockpit",
    "내가봐야할것": "capability_cockpit",
    "봐야할것": "capability_cockpit",
    "주의상태": "capability_cockpit",
    "주의필요": "capability_cockpit",
    "콕핏주의": "capability_cockpit",
    "콕핏주의상태": "capability_cockpit",
    "신뢰체크리스트": "capability_cockpit",
    "자비스신뢰체크리스트": "capability_cockpit",
    "자비스믿어도돼": "capability_cockpit",
    "믿어도되는이유": "capability_cockpit",
    "자비스믿어도되는이유": "capability_cockpit",
    "능력콕핏계획": "capability_cockpit",
    "폰컨트롤센터계획": "capability_cockpit",
    "평가팩계획": "capability_cockpit",
    "제임스평가팩계획": "capability_cockpit",
    "실제워크플로우뭐증명해": "capability_cockpit",
    "자비스뭐증명해야해": "capability_cockpit",
    "자비스3개월계획": "work_queue",
    "자비스세달계획": "work_queue",
    "자비스다음3개월뭐해": "work_queue",
    "자비스해자어떻게열어": "work_queue",
    "해자어떻게열어": "work_queue",
    "통합지금추가해도돼": "work_queue",
    "통합언제추가해": "work_queue",
    "자비스통합언제추가해": "work_queue",
    "claude가뭐남겼어": "handoff_brief",
    "claude가뭐라고했어": "handoff_brief",
    "claude핸드오프": "handoff_brief",
    "claude인수인계": "handoff_brief",
    "클로드가뭐남겼어": "handoff_brief",
    "클로드가뭐라고했어": "handoff_brief",
    "클로드핸드오프": "handoff_brief",
    "클로드인수인계": "handoff_brief",
    "인수인계": "handoff_brief",
    "인수인계상태": "handoff_brief",
    "핸드오프": "handoff_brief",
    "핸드오프상태": "handoff_brief",
    "코덱스작업": "work_queue",
    "코덱스작업뭐야": "work_queue",
    "코덱스작업큐": "work_queue",
    "코덱스작업목록": "work_queue",
    "코덱스태스크": "work_queue",
    "코덱스태스크확인": "work_queue",
    "코덱스할일": "work_queue",
    "코덱스현재지시": "work_queue",
    "코덱스현재지시사항": "work_queue",
    "현재지시": "work_queue",
    "현재지시사항": "work_queue",
    "현재지시사항보여줘": "work_queue",
    "작업큐": "work_queue",
    "작업큐보여줘": "work_queue",
    "작업목록보여줘": "work_queue",
    "작업대기열": "work_queue",
    "작업대기열보여줘": "work_queue",
    "현재작업큐": "work_queue",
    "현재작업목록": "work_queue",
    "뭐수정했어": "build_progress_report",
    "어떤파일바꿨어": "build_progress_report",
    "변경파일보여줘": "build_progress_report",
    "수정파일보여줘": "build_progress_report",
    "코덱스가뭐바꿨어": "build_progress_report",
    "코덱스변경사항": "build_progress_report",
    "최근변경사항": "build_progress_report",
    "패치상태": "build_progress_report",
    "변경사항확인": "build_progress_report",
    "대화지연상태": "model_routing_status",
    "대화속도상태": "model_routing_status",
    "대화기록상태": "model_routing_status",
    "대화p95상태": "model_routing_status",
    "대화8초목표": "model_routing_status",
    "8초대화상태": "model_routing_status",
    "8초채팅상태": "model_routing_status",
    "답변길이상태": "model_routing_status",
    "응답길이상태": "model_routing_status",
    "자비스느려": "model_routing_status",
    "자비스속도상태": "model_routing_status",
    "자비스왜느려": "model_routing_status",
    "채팅기록창": "model_routing_status",
    "채팅히스토리상태": "model_routing_status",
    "채팅p95상태": "model_routing_status",
    "채팅8초목표": "model_routing_status",
    "채팅느려": "model_routing_status",
    "채팅느린이유": "model_routing_status",
    "채팅지연상태": "model_routing_status",
    "채팅속도상태": "model_routing_status",
    "채팅속도설정": "model_routing_status",
    "채팅토큰상태": "model_routing_status",
    "채팅튜닝": "model_routing_status",
    "혼합대화p95상태": "model_routing_status",
    "혼합대화지연상태": "model_routing_status",
    "혼합대화속도상태": "model_routing_status",
    "기억몇개": "memory_stats",
    "기억개수": "memory_stats",
    "메모리몇개": "memory_stats",
    "메모리개수": "memory_stats",
    "노트몇개": "list_jarvis_notes",
    "노트개수": "list_jarvis_notes",
    "자비스노트몇개": "list_jarvis_notes",
    "자비스노트개수": "list_jarvis_notes",
    "목표몇개": "list_goals",
    "목표개수": "list_goals",
    "골몇개": "list_goals",
    "골개수": "list_goals",
    "작업몇개": "task_overview",
    "작업개수": "task_overview",
    "태스크몇개": "task_overview",
    "태스크개수": "task_overview",
    "할일몇개": "task_overview",
    "할일개수": "task_overview",
    "선호몇개": "list_preferences",
    "선호개수": "list_preferences",
    "선호사항몇개": "list_preferences",
    "선호사항개수": "list_preferences",
    "설정몇개": "list_preferences",
    "설정개수": "list_preferences",
    "결정몇개": "list_decisions",
    "결정개수": "list_decisions",
    "의사결정몇개": "list_decisions",
    "의사결정개수": "list_decisions",
    "사람몇명": "list_people",
    "사람몇개": "list_people",
    "인물몇명": "list_people",
    "인물몇개": "list_people",
    "스킬몇개": "list_skills",
    "스킬개수": "list_skills",
    "도구몇개": "capability_map",
    "도구개수": "capability_map",
    "자비스도구몇개": "capability_map",
    "자비스도구개수": "capability_map",
    "기능몇개": "capability_map",
    "기능개수": "capability_map",
    "능력몇개": "capability_map",
    "능력개수": "capability_map",
    "통합몇개": "capability_map",
    "통합개수": "capability_map",
    "리스크몇개": "risk_matrix",
    "리스크개수": "risk_matrix",
    "위험몇개": "risk_matrix",
    "위험개수": "risk_matrix",
    "위험도구몇개": "risk_matrix",
    "위험도구개수": "risk_matrix",
    "고위험몇개": "risk_matrix",
    "고위험개수": "risk_matrix",
    "고위험도구몇개": "risk_matrix",
    "고위험도구개수": "risk_matrix",
    "승인필요몇개": "risk_matrix",
    "승인필요개수": "risk_matrix",
    "승인필요도구몇개": "risk_matrix",
    "승인필요도구개수": "risk_matrix",
    "승인필요한도구몇개": "risk_matrix",
    "승인필요한도구개수": "risk_matrix",
    "읽기전용도구몇개": "risk_matrix",
    "읽기전용도구개수": "risk_matrix",
    "로컬안전도구몇개": "risk_matrix",
    "로컬안전도구개수": "risk_matrix",
    "개인정보도구몇개": "risk_matrix",
    "개인정보도구개수": "risk_matrix",
    "개인데이터도구몇개": "risk_matrix",
    "개인데이터도구개수": "risk_matrix",
    "건강상태": "capability_cockpit",
    "시스템헬스": "jarvis_doctor",
    "오류보고서": "jarvis_doctor",
    "오류상태": "jarvis_doctor",
    "자비스헬스": "jarvis_doctor",
    "전체상태": "capability_cockpit",
    "전체헬스": "capability_cockpit",
    "진단보고서": "jarvis_doctor",
    "진단상태": "jarvis_doctor",
    "고장났어": "recent_tool_runs",
    "고장상태": "recent_tool_runs",
    "마지막에러": "recent_tool_runs",
    "마지막오류": "recent_tool_runs",
    "마지막실패": "execution_health_report",
    "무슨오류있어": "recent_tool_runs",
    "뭐가고장났어": "recent_tool_runs",
    "뭐가문제야": "recent_tool_runs",
    "뭐가안돼": "recent_tool_runs",
    "문제뭐야": "recent_tool_runs",
    "문제상태": "recent_tool_runs",
    "문제있어": "recent_tool_runs",
    "무슨문제야": "recent_tool_runs",
    "무슨문제있어": "recent_tool_runs",
    "감사로그": "recent_tool_runs",
    "감사로그몇개": "recent_tool_runs",
    "감사로그개수": "recent_tool_runs",
    "감사상태": "recent_tool_runs",
    "도구실행기록": "recent_tool_runs",
    "도구실행몇개": "recent_tool_runs",
    "도구실행개수": "recent_tool_runs",
    "런타임추적": "runtime_trace_receipt",
    "마지막도구실행": "recent_tool_runs",
    "마지막실행": "recent_tool_runs",
    "실행기록": "recent_tool_runs",
    "실행기록몇개": "recent_tool_runs",
    "실행기록개수": "recent_tool_runs",
    "실행영수증": "verification_receipt",
    "최근도구실행": "recent_tool_runs",
    "최근도구실행몇개": "recent_tool_runs",
    "최근도구실행개수": "recent_tool_runs",
    "최근실행기록": "recent_tool_runs",
    "최근실행몇개": "recent_tool_runs",
    "최근실행개수": "recent_tool_runs",
    "오류뭐야": "recent_tool_runs",
    "왜실패했어": "recent_tool_runs",
    "최근에러": "recent_tool_runs",
    "최근오류": "recent_tool_runs",
    "뭐가실패했어": "execution_health_report",
    "실패목록": "execution_health_report",
    "실패보여줘": "execution_health_report",
    "실패상태": "execution_health_report",
    "실패한거있어": "execution_health_report",
    "최근실패": "execution_health_report",
    "반복실패상태": "repeated_failure_clusters",
    "복구상태": "recovery_closure_checklist",
    "복구마감상태": "recovery_closure_checklist",
    "복구부채상태": "recovery_closure_checklist",
    "복구종료상태": "recovery_closure_checklist",
    "복구클로저상태": "recovery_closure_checklist",
    "복구계획": "recovery_closure_checklist",
    "복구학습상태": "recovery_closure_checklist",
    "뭐복구해야해": "recovery_closure_checklist",
    "무엇을복구해야해": "recovery_closure_checklist",
    "검토할거있어": "list_pending_approvals",
    "나기다리는거있어": "list_pending_approvals",
    "내가승인해야할거있어": "list_pending_approvals",
    "뭐기다려": "list_pending_approvals",
    "리뷰큐": "list_pending_approvals",
    "대기중인승인": "list_pending_approvals",
    "대기승인몇개": "list_pending_approvals",
    "대기승인개수": "list_pending_approvals",
    "승인대기": "list_pending_approvals",
    "승인대기뭐있어": "list_pending_approvals",
    "승인뭐남았어": "list_pending_approvals",
    "승인대기몇개": "list_pending_approvals",
    "승인대기개수": "list_pending_approvals",
    "승인목록": "list_pending_approvals",
    "승인몇개": "list_pending_approvals",
    "승인개수": "list_pending_approvals",
    "승인큐": "list_pending_approvals",
    "승인큐몇개": "list_pending_approvals",
    "승인큐개수": "list_pending_approvals",
    "승인필요한거몇개": "list_pending_approvals",
    "승인필요한거개수": "list_pending_approvals",
    "승인필요한거있어": "list_pending_approvals",
    "승인필요한작업": "list_pending_approvals",
    "승인확인": "list_pending_approvals",
    "승인할거있어": "list_pending_approvals",
    "승인고장": "approval_queue_summary",
    "승인대기문제": "approval_queue_summary",
    "승인대기안됨": "approval_queue_summary",
    "승인대시보드": "approval_queue_summary",
    "승인문제": "approval_queue_summary",
    "승인버튼고장": "approval_queue_summary",
    "승인버튼문제": "approval_queue_summary",
    "승인버튼실패": "approval_queue_summary",
    "승인버튼안돼": "approval_queue_summary",
    "승인버튼안되": "approval_queue_summary",
    "승인버튼안됨": "approval_queue_summary",
    "승인버튼오류": "approval_queue_summary",
    "승인상태": "approval_queue_summary",
    "승인실패": "approval_queue_summary",
    "승인안돼": "approval_queue_summary",
    "승인안되": "approval_queue_summary",
    "승인안됨": "approval_queue_summary",
    "승인이슈": "approval_queue_summary",
    "승인오류": "approval_queue_summary",
    "승인큐멈춤": "approval_queue_summary",
    "승인큐문제": "approval_queue_summary",
    "왜승인실패": "approval_queue_summary",
    "왜승인안됨": "approval_queue_summary",
    "왜승인오류": "approval_queue_summary",
    "허가문제": "approval_queue_summary",
    "허가실패": "approval_queue_summary",
    "허가안돼": "approval_queue_summary",
    "허가안되": "approval_queue_summary",
    "허가안됨": "approval_queue_summary",
    "허가오류": "approval_queue_summary",
    "구성상태": "setup_check",
    "부트스트랩상태": "setup_check",
    "부트상태": "setup_check",
    "설정상태": "setup_check",
    "설정확인": "setup_check",
    "설치상태": "setup_check",
    "셋업상태": "setup_check",
    "시작상태": "setup_check",
    "환경확인": "setup_check",
    "환경상태": "setup_check",
    "db상태": "storage_status",
    "데이터베이스상태": "storage_status",
    "디비상태": "storage_status",
    "스토리지상태": "storage_status",
    "스토리지통계": "storage_status",
    "저장소상태": "storage_status",
    "저장소통계": "storage_status",
    "저장소확인": "storage_status",
    "대화몇개": "list_sessions",
    "대화개수": "list_sessions",
    "대화세션몇개": "list_sessions",
    "대화세션개수": "list_sessions",
    "세션몇개": "list_sessions",
    "세션개수": "list_sessions",
    "다음브리핑언제": "list_scheduled_jobs",
    "모닝브리프상태": "list_scheduled_jobs",
    "모닝브리프예약확인": "list_scheduled_jobs",
    "모닝브리프일정": "list_scheduled_jobs",
    "반복작업": "list_scheduled_jobs",
    "백그라운드작업": "list_scheduled_jobs",
    "브리핑상태": "list_scheduled_jobs",
    "브리핑언제야": "list_scheduled_jobs",
    "브리핑예약": "list_scheduled_jobs",
    "스케줄몇개": "list_scheduled_jobs",
    "스케줄개수": "list_scheduled_jobs",
    "스케줄작업몇개": "list_scheduled_jobs",
    "스케줄작업개수": "list_scheduled_jobs",
    "스케줄상태": "list_scheduled_jobs",
    "예약된작업몇개": "list_scheduled_jobs",
    "예약된작업개수": "list_scheduled_jobs",
    "예약작업몇개": "list_scheduled_jobs",
    "예약작업개수": "list_scheduled_jobs",
    "예약작업상태": "list_scheduled_jobs",
    "자동화상태": "list_scheduled_jobs",
    "작업예약몇개": "list_scheduled_jobs",
    "작업예약개수": "list_scheduled_jobs",
    "기억리뷰": "learning_review",
    "기억상태": "memory_stats",
    "기억정리": "list_weak_memories",
    "기억확인": "memory_stats",
    "메모리상태": "memory_stats",
    "메모리확인": "memory_stats",
    "개인컨텍스트상태": "personal_context_status",
    "내개인컨텍스트상태": "personal_context_status",
    "내개인지식상태": "personal_context_status",
    "개인지식상태": "personal_context_status",
    "무엇을기억해": "recent_memories",
    "무엇을배웠어": "learning_review",
    "뭘기억해": "recent_memories",
    "뭘배웠어": "learning_review",
    "배운거": "learning_review",
    "약한기억": "list_weak_memories",
    "자비스가뭘기억해": "recent_memories",
    "중복기억": "list_duplicate_memories",
    "학습리뷰": "learning_review",
    "학습루프상태": "learning_review",
    "학습부채상태": "execution_learning_closure_packet",
    "학습부채닫혔어": "execution_learning_closure_packet",
    "학습증명매트릭스": "execution_learning_closure_packet",
    "실패학습상태": "failure_learning_cockpit",
    "피드백몇개": "feedback_report",
    "피드백개수": "feedback_report",
    "학습상태": "learning_review",
    "학습확인": "learning_review",
    "녹음상태": "voice_setup_check",
    "마이크상태": "voice_setup_check",
    "마이크확인": "voice_setup_check",
    "위스퍼상태": "voice_setup_check",
    "위스퍼워밍업상태": "voice_setup_check",
    "음성상태": "voice_setup_check",
    "음성워밍업상태": "voice_setup_check",
    "음성입력상태": "voice_setup_check",
    "음성준비됐어": "voice_setup_check",
    "전사상태": "voice_setup_check",
    "푸시투톡상태": "voice_setup_check",
    "대시보드상태": "capability_cockpit",
    "상태판보여줘": "capability_cockpit",
    "자비스상태판": "capability_cockpit",
    "콕핏보여줘": "capability_cockpit",
    "콕핏상태": "capability_cockpit",
    "데몬상태": "capability_cockpit",
    "데몬증명": "capability_cockpit",
    "데몬증명매트릭스": "capability_cockpit",
    "런치디상태": "capability_cockpit",
    "런치에이전트상태": "capability_cockpit",
    "네트워크복구상태": "capability_cockpit",
    "네트워크복구증명": "capability_cockpit",
    "스케줄러데몬상태": "capability_cockpit",
    "재부팅상태": "capability_cockpit",
    "재부팅증명": "capability_cockpit",
    "재부팅증명매트릭스": "capability_cockpit",
    "텔레그램데몬상태": "capability_cockpit",
    "텔레그램제어데몬재시작": "telegram_control_restart_guidance",
    "텔레그램컨트롤데몬재시작": "telegram_control_restart_guidance",
    "텔레그램데몬재시작": "telegram_control_restart_guidance",
    "재시작해": "restart_target_clarification",
    "다시시작해": "restart_target_clarification",
    "그거재시작해": "restart_target_clarification",
    "이거재시작해": "restart_target_clarification",
    "라이브증명상태": "capability_cockpit",
    "라이브체크": "capability_cockpit",
    "라이브체크누락": "capability_cockpit",
    "라이브체크빠진항목": "capability_cockpit",
    "라이브체크커버리지": "capability_cockpit",
    "라이브체크표": "capability_cockpit",
    "라이브체크항목": "capability_cockpit",
    "스모크상태": "capability_cockpit",
    "스모크테스트상태": "capability_cockpit",
    "원스크린인수": "capability_cockpit",
    "원스크린인수표": "capability_cockpit",
    "인수커버리지": "capability_cockpit",
    "인수상태": "capability_cockpit",
    "인수표": "capability_cockpit",
    "음성증명상태": "capability_cockpit",
    "테스트상태": "capability_cockpit",
    "테스트초록": "capability_cockpit",
    "한국어음성검증": "capability_cockpit",
    "한국어음성증명": "capability_cockpit",
    "데몬복구어떻게테스트해": "capability_cockpit",
    "리마인더어떻게증명해": "capability_cockpit",
    "리마인더어떻게테스트해": "capability_cockpit",
    "아침브리핑어떻게증명해": "capability_cockpit",
    "아침브리핑어떻게테스트해": "capability_cockpit",
    "승인흐름어떻게테스트해": "capability_cockpit",
    "연락처조회어떻게증명해": "capability_cockpit",
    "연락처조회어떻게테스트해": "capability_cockpit",
    "이메일보내기어떻게증명해": "capability_cockpit",
    "이메일읽기어떻게테스트해": "capability_cockpit",
    "재부팅생존어떻게증명해": "capability_cockpit",
    "캘린더쓰기어떻게증명해": "capability_cockpit",
    "캘린더쓰기어떻게테스트해": "capability_cockpit",
    "텔레그램음성어떻게테스트해": "capability_cockpit",
    "폰승인어떻게증명해": "capability_cockpit",
    "한국어음성어떻게증명해": "capability_cockpit",
    "개인연동뭐테스트해": "capability_cockpit",
    "개인증명결과형식": "capability_cockpit",
    "개인증명매트릭스": "capability_cockpit",
    "개인증명뭐해야해": "capability_cockpit",
    "검색증명매트릭스": "capability_cockpit",
    "네트워크복구증명매트릭스": "capability_cockpit",
    "대화연구증명매트릭스": "capability_cockpit",
    "대화증명뭐해야해": "capability_cockpit",
    "대화증명어떻게보고해": "capability_cockpit",
    "대화증명결과형식": "capability_cockpit",
    "대화증명매트릭스": "capability_cockpit",
    "대화지연증명형식": "capability_cockpit",
    "혼합대화증명": "capability_cockpit",
    "혼합대화뭐테스트해": "capability_cockpit",
    "챗지연증명형식": "capability_cockpit",
    "뭐가고장났어증명형식": "capability_cockpit",
    "브리핑전송증명": "capability_cockpit",
    "스케줄증명결과형식": "capability_cockpit",
    "스케줄증명매트릭스": "capability_cockpit",
    "승인결과형식": "capability_cockpit",
    "승인거절증명": "capability_cockpit",
    "승인거절증명매트릭스": "capability_cockpit",
    "승인버튼결과형식": "capability_cockpit",
    "승인버튼증명": "capability_cockpit",
    "승인버튼증명매트릭스": "capability_cockpit",
    "승인콜백결과형식": "capability_cockpit",
    "승인콜백증명": "capability_cockpit",
    "승인콜백증명매트릭스": "capability_cockpit",
    "승인증명매트릭스": "capability_cockpit",
    "승인증명뭐해야해": "capability_cockpit",
    "승인흐름뭐테스트해": "capability_cockpit",
    "텔레그램승인결과형식": "capability_cockpit",
    "텔레그램승인증명": "capability_cockpit",
    "폰승인버튼증명": "capability_cockpit",
    "폰승인버튼증명매트릭스": "capability_cockpit",
    "아침브리핑증명형식": "capability_cockpit",
    "연구증명매트릭스": "capability_cockpit",
    "예약작업뭐테스트해": "capability_cockpit",
    "예약작업증명형식": "capability_cockpit",
    "연락처조회증명매트릭스": "capability_cockpit",
    "이메일검색증명": "capability_cockpit",
    "이메일보내기증명": "capability_cockpit",
    "이메일읽기증명": "capability_cockpit",
    "복구안내증명": "capability_cockpit",
    "오류안내증명매트릭스": "capability_cockpit",
    "오류안내증명": "capability_cockpit",
    "오류안내형식": "capability_cockpit",
    "오류메시지상태": "capability_cockpit",
    "오류복구상태": "capability_cockpit",
    "동결라우팅위험": "frozen_routing_risk_report",
    "긴명령안전상태": "planner_input_guard_report",
    "긴메시지레도스위험": "planner_input_guard_report",
    "긴입력안전상태": "planner_input_guard_report",
    "레도스상태": "frozen_routing_risk_report",
    "입력길이제한상태": "planner_input_guard_report",
    "전송라우팅레도스": "frozen_routing_risk_report",
    "정규식위험상태": "frozen_routing_risk_report",
    "플래너입력제한": "planner_input_guard_report",
    "음성증명결과형식": "capability_cockpit",
    "음성증명매트릭스": "capability_cockpit",
    "음성증명뭐해야해": "capability_cockpit",
    "음성증명어떻게보고해": "capability_cockpit",
    "음성뭐테스트해": "capability_cockpit",
    "일일가치증명매트릭스": "capability_cockpit",
    "재부팅증명형식": "capability_cockpit",
    "리마인더증명매트릭스": "capability_cockpit",
    "캘린더쓰기증명매트릭스": "capability_cockpit",
    "전화제어증명매트릭스": "capability_cockpit",
    "챗지연증명매트릭스": "capability_cockpit",
    "통화증명매트릭스": "capability_cockpit",
    "폰제어증명형식": "capability_cockpit",
    "폰컨트롤증명매트릭스": "capability_cockpit",
    "로컬음성증명": "capability_cockpit",
    "텔레그램음성증명": "capability_cockpit",
    "한국어음성증명형식": "capability_cockpit",
    "회복안내증명매트릭스": "capability_cockpit",
    "증거상태": "evidence_ledger",
    "증거보고서": "evidence_ledger",
    "증명상태": "evidence_ledger",
    "근거상태": "evidence_ledger",
    "다음행동": "safe_next_actions",
    "다음명령": "safe_next_actions",
    "다음뭐해야해": "safe_next_actions",
    "다음뭐해": "safe_next_actions",
    "다음뭐할까": "safe_next_actions",
    "다음에뭘실행해": "safe_next_actions",
    "다음에뭘실행해야해": "safe_next_actions",
    "다음에뭐해": "safe_next_actions",
    "무슨명령실행해": "safe_next_actions",
    "무슨명령실행해야해": "safe_next_actions",
    "뭘실행해야해": "safe_next_actions",
    "뭐실행해야해": "safe_next_actions",
    "자비스다음단계": "safe_next_actions",
    "자비스다음뭐해": "safe_next_actions",
    "자비스뭐부터해": "safe_next_actions",
    "채널상태": "channel_health",
    "채널확인": "channel_health",
    "채널헬스": "channel_health",
    "메세지상태": "channel_health",
    "메세지채널상태": "channel_health",
    "메시지상태": "channel_health",
    "메시지채널상태": "channel_health",
    "메시징상태": "channel_health",
    "문자상태": "channel_health",
    "문자채널상태": "channel_health",
    "전송상태": "channel_health",
    "전송채널상태": "channel_health",
    "발송채널상태": "channel_health",
    "텔레그램상태": "channel_health",
    "텔레그램채널상태": "channel_health",
    "카카오상태": "channel_health",
    "카카오톡상태": "channel_health",
    "카카오채널상태": "channel_health",
    "카톡상태": "channel_health",
    "인스타상태": "channel_health",
    "인스타그램상태": "channel_health",
    "아이메시지상태": "channel_health",
    "imessage상태": "channel_health",
    "통화상태": "channel_health",
    "통화채널상태": "channel_health",
    "전화상태": "channel_health",
    "전화채널상태": "channel_health",
    "콜상태": "channel_health",
    "페이스타임상태": "channel_health",
    "facetime상태": "channel_health",
    "폰상태": "channel_health",
    "모바일상태": "channel_health",
    "내부오케스트레이션": "subagent_fleet_status",
    "내부오케스트레이션확인": "subagent_fleet_status",
    "내부오케스트레이션상태": "subagent_fleet_status",
    "내부워커상태": "subagent_fleet_status",
    "서브에이전트개수": "subagent_fleet_status",
    "서브에이전트몇개": "subagent_fleet_status",
    "서브에이전트상태": "subagent_fleet_status",
    "서브에이전트상태확인": "subagent_fleet_status",
    "서브에이전트준비상태": "subagent_fleet_status",
    "서브에이전트헬스": "subagent_fleet_status",
    "에이전트상태": "subagent_fleet_status",
    "에이전트상태확인": "subagent_fleet_status",
    "에이전트준비상태": "subagent_fleet_status",
    "에이전트헬스": "subagent_fleet_status",
    "병렬에이전트상태": "subagent_fleet_status",
    "준비된에이전트": "subagent_fleet_status",
    "준비된에이전트개수": "subagent_fleet_status",
    "준비된에이전트몇개": "subagent_fleet_status",
    "워커개수": "subagent_fleet_status",
    "워커몇개": "subagent_fleet_status",
    "워커상태": "subagent_fleet_status",
    "워커준비상태": "subagent_fleet_status",
    "워커헬스": "subagent_fleet_status",
    "작업자개수": "subagent_fleet_status",
    "작업자몇개": "subagent_fleet_status",
    "작업자상태": "subagent_fleet_status",
    "준비상태": "readiness_report",
}
_RUNTIME_COCKPIT_LANE_EXACT_ALIASES = (
    "approval lane",
    "approvals lane",
    "brief lane",
    "briefing lane",
    "calendar email lane",
    "calendar lane",
    "call lane",
    "calls lane",
    "contact lane",
    "contacts lane",
    "diagnostic lane",
    "diagnostics lane",
    "email lane",
    "failure lane",
    "failures lane",
    "feedback lane",
    "guardrail lane",
    "guardrails lane",
    "operator evals lane",
    "operator workflow evals lane",
    "learning lane",
    "market lane",
    "markets lane",
    "memory lane",
    "message lane",
    "messages lane",
    "messaging lane",
    "mic lane",
    "morning brief lane",
    "notes lane",
    "orchestration lane",
    "people lane",
    "personal integrations lane",
    "personal lane",
    "personal proof lane",
    "personal proofs lane",
    "research lane",
    "schedule lane",
    "scheduled jobs lane",
    "scheduler lane",
    "search lane",
    "subagent lane",
    "subagents lane",
    "voice lane",
    "weather lane",
    "web lane",
    "workflow evals lane",
    "worker lane",
    "workers lane",
    "writing lane",
)
for _phrase in _RUNTIME_COCKPIT_LANE_EXACT_ALIASES:
    _RUNTIME_EXACT_TOOL_ALIASES[_phrase] = "capability_cockpit"

_RUNTIME_COCKPIT_LANE_COMPACT_ALIASES = (
    "개인증명레인",
    "개인통합레인",
    "검색레인",
    "고장레인",
    "글쓰기레인",
    "기억레인",
    "날씨레인",
    "노트레인",
    "러닝레인",
    "리서치레인",
    "마이크레인",
    "마켓레인",
    "메모리레인",
    "메세지레인",
    "메시지레인",
    "메시징레인",
    "모닝브리핑레인",
    "문자레인",
    "브리핑레인",
    "사람레인",
    "서브에이전트레인",
    "스케줄레인",
    "시장레인",
    "쓰기레인",
    "아침브리핑레인",
    "에이전트레인",
    "연구레인",
    "연락처레인",
    "예약작업레인",
    "오케스트레이션레인",
    "음성레인",
    "이메일레인",
    "일정레인",
    "작성레인",
    "작업자레인",
    "전화레인",
    "제임스평가레인",
    "조사레인",
    "진단레인",
    "카렌더레인",
    "캘린더레인",
    "캘린더이메일레인",
    "타이핑레인",
    "통합증명레인",
    "통화레인",
    "피드백레인",
    "학습레인",
    "승인레인",
    "실패레인",
    "워커레인",
    "워크플로우평가레인",
    "웹레인",
)
for _phrase in _RUNTIME_COCKPIT_LANE_COMPACT_ALIASES:
    _RUNTIME_EXACT_TOOL_ALIAS_COMPACT[_phrase] = "capability_cockpit"

_PRE_PLANNER_SUGGESTION_NORMALIZED = {
    "agent readiness",
    "agent status",
    "agents ready",
    # "audit log" / "show audit log" / "audit trail" / "show audit trail" /
    # "system status" are deliberately NOT in
    # this set: they already resolve deterministically and safely to the
    # READ_ONLY `recent_tool_runs` / `system_info` tools via the planner, so
    # routing them through a suggestion first was a pure UX regression (same
    # class of bug as the "active goals" / "show my goals" find earlier this
    # session -- see [[jarvis-pre-planner-suggestion-shadow]]). "system health"
    # stays in this set: it does not resolve cleanly through the planner, so
    # the suggestion step is protective there, not redundant.
    "automation health",
    "automation report",
    "automation status",
    "automations health",
    "automations status",
    "address book capabilities",
    "address book commands",
    "address book examples",
    "anything broken",
    "anything need approval",
    "anything needs my approval",
    "anything pending approval",
    "anything waiting on me",
    "background job status",
    "background jobs",
    "beginner commands",
    "brief last run",
    "brief ran today",
    "brief sent today",
    "brief error",
    "brief failed",
    "brief failure",
    "brief issue",
    "brief job status",
    "brief missing",
    "brief not working",
    "brief problem",
    "brief did not arrive",
    "brief did not send",
    "brief didn t arrive",
    "brief didnt arrive",
    "brief not delivered",
    "call capabilities",
    "call commands",
    "call error",
    "call examples",
    "call failed",
    "call failure",
    "call issue",
    "call not working",
    "call problem",
    "calling capabilities",
    "calling commands",
    "calling error",
    "calling examples",
    "calling failed",
    "calling failure",
    "calling issue",
    "calling problem",
    "calendar capabilities",
    "command examples",
    "common commands",
    "contact capabilities",
    "contact commands",
    "contact examples",
    "contact lookup capabilities",
    "contact lookup commands",
    "contact lookup examples",
    "contacts capabilities",
    "contacts commands",
    "contacts examples",
    "daily commands",
    "daily brief job status",
    "dashboard capabilities",
    "delivery status",
    "did brief run today",
    "did morning brief send",
    "did my morning brief run",
    "did scheduler run",
    "did the morning brief send",
    "diagnostic report",
    "diagnostic status",
    "diagnostics",
    "diagnostics report",
    "diagnostics status",
    "error report",
    "error status",
    "example commands",
    "examples",
    "failed status",
    "failure report",
    "failure status",
    "failures",
    "health report",
    "health status",
    "last error",
    "last failed tool",
    "latest error",
    "latest failed tool",
    "email capabilities",
    "gmail capabilities",
    "how do i use jarvis",
    "how to use jarvis",
    "help with address book",
    "help with contact lookup",
    "help with contacts",
    "help with messages",
    "help with messaging",
    "help with telegram",
    "help with voice",
    "imessage capabilities",
    "imessage commands",
    "imessage broken",
    "imessage did not deliver",
    "imessage did not send",
    "imessage didn t deliver",
    "imessage didn t send",
    "imessage didnt deliver",
    "imessage didnt send",
    "imessage error",
    "imessage examples",
    "imessage failed",
    "imessage failure",
    "imessage health",
    "imessage issue",
    "imessage not delivered",
    "imessage not sent",
    "imessage not working",
    "imessage problem",
    "imessage status",
    "is anything broken",
    "instagram capabilities",
    "instagram commands",
    "instagram broken",
    "instagram did not deliver",
    "instagram did not send",
    "instagram didn t deliver",
    "instagram didn t send",
    "instagram didnt deliver",
    "instagram didnt send",
    "instagram error",
    "instagram examples",
    "instagram failed",
    "instagram failure",
    "instagram health",
    "instagram issue",
    "instagram not delivered",
    "instagram not sent",
    "instagram not working",
    "instagram problem",
    "instagram status",
    "internal orchestration",
    "internal orchestration status",
    "is morning brief scheduled",
    "is the brief scheduled",
    "job status",
    "jobs not running",
    "jobs status",
    "kakao capabilities",
    "kakao commands",
    "kakao broken",
    "kakao did not deliver",
    "kakao did not send",
    "kakao didn t deliver",
    "kakao didn t send",
    "kakao didnt deliver",
    "kakao didnt send",
    "kakao error",
    "kakao examples",
    "kakao failed",
    "kakao failure",
    "kakao health",
    "kakao issue",
    "kakao not delivered",
    "kakao not sent",
    "kakao not working",
    "kakao problem",
    "kakao status",
    "operator eval pack",
    "operator evals",
    "operator real tasks",
    "operator task evals",
    "operator workflow evals",
    "operator workflow tests",
    "operator workflows",
    "jarvis health",
    "jarvis report",
    "jarvis examples",
    "jarvis usage",
    "channel proof matrix",
    "channel proof status",
    "live proof matrix",
    "live proof status",
    "live test matrix",
    "live test status",
    "live matrix status",
    "market capabilities",
    "markets capabilities",
    "memory capabilities",
    "message capabilities",
    "message commands",
    "message delivery status",
    "message did not deliver",
    "message did not send",
    "message didn t deliver",
    "message didn t send",
    "message didnt deliver",
    "message didnt send",
    "message error",
    "message examples",
    "message failed",
    "message failure",
    "message issue",
    "message not delivered",
    "message not sent",
    "message not working",
    "message problem",
    "message status",
    "messages capabilities",
    "messages commands",
    "messages examples",
    "messages status",
    "messaging capabilities",
    "messaging commands",
    "messaging examples",
    "messaging issue",
    "messaging problem",
    "mobile control",
    "mobile status",
    "last actions",
    "last morning brief",
    "last thing you did",
    "latest actions",
    "morning brief job status",
    "morning brief error",
    "morning brief failed",
    "morning brief failure",
    "morning brief issue",
    "morning brief not working",
    "morning brief missing",
    "morning brief problem",
    "morning brief did not arrive",
    "morning brief did not send",
    "morning brief didn t arrive",
    "morning brief didnt arrive",
    "morning brief last run",
    "morning brief next run",
    "morning brief not delivered",
    "morning job status",
    "morning brief schedule",
    "next morning brief",
    "news capabilities",
    "note capabilities",
    "notes capabilities",
    "owner phone status",
    "owner commands",
    "orchestration status",
    "overall health",
    "overall status",
    "phone commands",
    "phone control center",
    "phone control status",
    "phone call capabilities",
    "phone call commands",
    "phone call error",
    "phone call examples",
    "phone call failed",
    "phone call failure",
    "phone call issue",
    "phone call problem",
    "phone examples",
    "phone health",
    "phone help",
    "phone shortcuts",
    "phone status",
    "recurring job status",
    "recurring jobs",
    "recurring task status",
    "recurring tasks",
    "real workflow evals",
    "real workflow tests",
    "research capabilities",
    "schedule health",
    "schedule capabilities",
    "schedule report",
    "schedule status",
    "send capabilities",
    "send commands",
    "send did not work",
    "send error",
    "send examples",
    "send failed",
    "send failure",
    "send not working",
    "send health",
    "send issue",
    "send problem",
    "send status",
    "sending capabilities",
    "sending commands",
    "sending error",
    "sending examples",
    "sending failed",
    "sending failure",
    "sending issue",
    "sending problem",
    "sending status",
    "schedule ran today",
    "scheduled brief status",
    "scheduled job status",
    "scheduled jobs",
    "scheduled jobs status",
    "scheduler capabilities",
    "scheduler health",
    "scheduler not running",
    "scheduler ran today",
    "scheduler report",
    "scheduler status",
    "sample commands",
    "recent actions",
    # "recent activity" is deliberately NOT in this set: it already resolves
    # deterministically and safely to the READ_ONLY `activity_digest` tool via
    # the planner, so routing it through a suggestion first was a pure UX
    # regression (same class as [[jarvis-pre-planner-suggestion-shadow]]).
    "recent logs",
    "recent runs",
    "show failures",
    "show commands",
    "show last error",
    "show latest error",
    "show examples",
    "show messaging examples",
    "show shortcuts",
    "show voice commands",
    "shortcuts",
    "starter commands",
    "status report",
    "test matrix",
    "test matrix status",
    "ready agents",
    "stock capabilities",
    "stocks capabilities",
    "crypto capabilities",
    "currency capabilities",
    "conversion capabilities",
    "execution logs",
    "translation capabilities",
    "translate capabilities",
    "utility capabilities",
    "utilities capabilities",
    "task capabilities",
    "tasks capabilities",
    "todo capabilities",
    "todos capabilities",
    "weather capabilities",
    "writing capabilities",
    "write capabilities",
    "system health",
    "subagent fleet",
    "subagent fleet status",
    "subagent readiness",
    "subagent status",
    "telegram capabilities",
    "telegram commands",
    "telegram channel status",
    "telegram control center",
    "telegram control status",
    "telegram broken",
    "telegram error",
    "telegram examples",
    "telegram failed",
    "telegram failure",
    "telegram health",
    "telegram help",
    "telegram issue",
    "telegram message examples",
    "telegram message did not send",
    "telegram message didn t send",
    "telegram message didnt send",
    "telegram message not delivered",
    "telegram message not sent",
    "telegram not working",
    "telegram problem",
    "telegram did not deliver",
    "telegram did not send",
    "telegram didn t deliver",
    "telegram didn t send",
    "telegram didnt deliver",
    "telegram didnt send",
    "telegram report",
    "telegram send failed",
    "telegram send did not work",
    "telegram send didn t work",
    "telegram send didnt work",
    "telegram shortcuts",
    "telegram status",
    "available tools",
    "available tool status",
    "available capabilities",
    "approval dashboard",
    "capabilities list",
    "capabilities cockpit",
    "capabilities dashboard",
    "capability cockpit status",
    "capability dashboard",
    "capability list",
    "capability report",
    "list capabilities",
    "registered tools",
    "show available tools",
    "show capabilities",
    "show cockpit",
    "show control plane",
    "show dashboard status",
    "show risk levels",
    "tool health",
    "tool list",
    "tool registry",
    "tool registry status",
    "tool report",
    "tool status",
    "tools health",
    "tools list",
    "tools report",
    "tools status",
    "what automations are running",
    "what tools are available",
    "what tools do you have",
    "what can i ask",
    "what can i call",
    "what can i send",
    "what can i do from my phone",
    "what can i do in telegram",
    "what can i do on telegram",
    "what can i send jarvis on telegram",
    "what can i text jarvis",
    "what can i try",
    # "what changed" / "what changed recently" are deliberately NOT in this
    # set for the same reason as "recent activity" above: both already
    # resolve deterministically and safely to `activity_digest`.
    "what commands work on telegram",
    "what can jarvis do",
    "what can jarvis actually do",
    "what can jarvis do with address book",
    "what can jarvis do with contact lookup",
    "what can jarvis do with contacts",
    "what can you do",
    "what can you actually do",
    "what can you do with address book",
    "what can you do with contact lookup",
    "what can you do with contacts",
    "what did jarvis do",
    "what did you do",
    "what ran recently",
    "what is healthy",
    "what is unhealthy",
    "what is automated",
    "what is running automatically",
    "what is scheduled",
    "what is jarvis capable of",
    "what is the morning schedule",
    "what is the risk level",
    "what is the schedule",
    "what agents are ready",
    "what agents are running",
    "what do you need from me",
    "what is blocked by me",
    "what is blocked by approval",
    "what is pending",
    "what is waiting on me",
    "what jobs are running",
    "what failed last",
    "what just failed",
    "what needs attention",
    "what needs my attention",
    "what was the last error",
    "what was the last failure",
    "what should i review",
    "what should i try",
    "when is morning brief",
    "when is next morning brief",
    "where is my brief",
    "where is my morning brief",
    "workflow evals",
    "workflow status",
    "workflow tests",
    "what happened to imessage",
    "what happened to instagram",
    "what happened to kakao",
    "what happened to telegram",
    "what happened with imessage",
    "what happened with instagram",
    "what happened with kakao",
    "what happened with telegram",
    "was morning brief delivered",
    "was morning brief sent",
    "why did brain fail",
    "why did call fail",
    "why did calling fail",
    "why did chat fail",
    "why did chat model fail",
    "why did approval button fail",
    "why did approval fail",
    "why did approve fail",
    "why did it fail",
    "why did imessage fail",
    "why did imessage not deliver",
    "why did imessage not send",
    "why did instagram fail",
    "why did instagram not deliver",
    "why did instagram not send",
    "why did kakao fail",
    "why did kakao not deliver",
    "why did kakao not send",
    "why did llm fail",
    "why did local model fail",
    "why did message fail",
    "why did message not deliver",
    "why did message not send",
    "why did mic fail",
    "why did microphone fail",
    "why did model fail",
    "why did morning brief fail",
    "why did morning brief not arrive",
    "why did morning brief not send",
    "why didn t morning brief arrive",
    "why didn t morning brief send",
    "why didnt morning brief arrive",
    "why didnt morning brief send",
    "why did scheduler not run",
    "why did ollama fail",
    "why did phone call fail",
    "why did planner fail",
    "why did planner model fail",
    "why did recording fail",
    "why did send fail",
    "why did send not work",
    "why did sending fail",
    "why did speech fail",
    "why did telegram fail",
    "why did telegram not deliver",
    "why did telegram not send",
    "why did telegram send fail",
    "why did telegram send not work",
    "why did transcription fail",
    "why did voice fail",
    "why did voice input fail",
    "why did whisper fail",
    "worker fleet status",
    "worker readiness",
    "worker status",
    "which agents are ready",
    "which agents are running",
    "control panel",
    "control plane",
    "dashboard status",
    "jarvis control panel",
    "jarvis control plane",
    "jarvis cockpit",
    "can i trust jarvis",
    "can i trust you",
    "can i use jarvis safely",
    "can jarvis refuse",
    "can jarvis be trusted",
    "commands that need approval",
    "commands that require approval",
    "high risk commands",
    "high risk tools",
    "is jarvis reliable",
    "is jarvis safe to use",
    "is jarvis trustworthy",
    "reliability health",
    "reliability report",
    "reliability status",
    "risk dashboard",
    "risk levels",
    "safe to use jarvis",
    "safety dashboard",
    "show reliability status",
    "show trust status",
    "trust dashboard",
    "trust health",
    "trust report",
    "trust status",
    "what are jarvis boundaries",
    "what are jarvis limits",
    "what are the approval boundaries",
    "what are the safety boundaries",
    "what are your boundaries",
    "what are your limits",
    "what are you allowed to do",
    "what are you not allowed to do",
    "what actions need approval",
    "what commands need approval",
    "what can jarvis not do",
    "what can you not do",
    "what cant jarvis do",
    "what cant you do",
    "what is approval gated",
    "what is approval-gated",
    "what is high risk",
    "what needs approval",
    "what needs my permission",
    "what permissions do you need",
    "what requires approval",
    "what requires my permission",
    "what tools require approval",
    "which actions are risky",
    "which commands are risky",
    "which tools are high risk",
    "which tools require approval",
    "is everything healthy",
    "is jarvis healthy",
    "teach me jarvis",
    "usage",
    "approval broken",
    "approval button broken",
    "approval button error",
    "approval button failed",
    "approval button failure",
    "approval button issue",
    "approval button not working",
    "approval button problem",
    "approval button stuck",
    "approval error",
    "approval failed",
    "approval failure",
    "approval issue",
    "approval not working",
    "approval problem",
    "approval queue broken",
    "approval queue error",
    "approval queue failed",
    "approval queue failure",
    "approval queue issue",
    "approval queue not working",
    "approval queue problem",
    "approval queue stuck",
    "approval stuck",
    "approval capabilities",
    "approvals capabilities",
    "approval needed",
    "approve button broken",
    "approve button error",
    "approve button failed",
    "approve button failure",
    "approve button issue",
    "approve button not working",
    "approve button problem",
    "approve button stuck",
    "approve failed",
    "approve failure",
    "approve not working",
    "approve problem",
    "approve stuck",
    "blocked by approval",
    "do you need anything from me",
    "do you need me for anything",
    "do you need my approval",
    "needs approval",
    "pending approval",
    "pending review",
    "ready for me",
    "review queue",
    "waiting for me",
    "safety capabilities",
    "brain broken",
    "brain error",
    "brain failed",
    "brain failure",
    "brain issue",
    "brain not working",
    "brain problem",
    "chat broken",
    "chat error",
    "chat failed",
    "chat failure",
    "chat issue",
    "chat model broken",
    "chat model error",
    "chat model failed",
    "chat model failure",
    "chat model issue",
    "chat model not working",
    "chat model problem",
    "chat not working",
    "chat problem",
    "llm broken",
    "llm error",
    "llm failed",
    "llm failure",
    "llm issue",
    "llm not working",
    "llm problem",
    "local model broken",
    "local model error",
    "local model failed",
    "local model failure",
    "local model issue",
    "local model not working",
    "local model problem",
    "mic broken",
    "mic error",
    "mic failed",
    "mic failure",
    "mic issue",
    "mic not working",
    "mic problem",
    "microphone broken",
    "microphone error",
    "microphone failed",
    "microphone failure",
    "microphone issue",
    "microphone not working",
    "microphone problem",
    "model broken",
    "model error",
    "model failed",
    "model failure",
    "model issue",
    "model not working",
    "model problem",
    "my mic is broken",
    "my mic is not working",
    "my microphone is broken",
    "my microphone is not working",
    "ollama broken",
    "ollama error",
    "ollama failed",
    "ollama failure",
    "ollama issue",
    "ollama not working",
    "ollama problem",
    "planner broken",
    "planner error",
    "planner failed",
    "planner failure",
    "planner issue",
    "planner model broken",
    "planner model error",
    "planner model failed",
    "planner model failure",
    "planner model issue",
    "planner model not working",
    "planner model problem",
    "planner not working",
    "planner problem",
    "recording broken",
    "recording error",
    "recording failed",
    "recording failure",
    "recording issue",
    "recording not working",
    "recording problem",
    "speech broken",
    "speech error",
    "speech failed",
    "speech failure",
    "speech issue",
    "speech not working",
    "speech problem",
    "transcription broken",
    "transcription error",
    "transcription failed",
    "transcription failure",
    "transcription issue",
    "transcription not working",
    "transcription problem",
    "voice broken",
    "voice capabilities",
    "voice command capabilities",
    "voice commands",
    "voice error",
    "voice examples",
    "voice failed",
    "voice failure",
    "voice input broken",
    "voice input error",
    "voice input failed",
    "voice input failure",
    "voice input issue",
    "voice input not working",
    "voice input problem",
    "voice issue",
    "voice not working",
    "voice problem",
    "whisper broken",
    "whisper error",
    "whisper failed",
    "whisper failure",
    "whisper issue",
    "whisper not working",
    "whisper problem",
}

_VOICE_STOP_PRE_PLANNER = {
    "cancel speech",
    "cancel voice",
    "cancel voice input",
    "interrupt speech",
    "interrupt voice",
    "mute jarvis",
    "mute speech",
    "mute voice",
    "silence jarvis",
    "silence speech",
    "silence voice",
    "stop listening",
    "stop recording",
    "stop speaking",
    "stop talking",
    "stop voice",
    "stop voice input",
}
_VOICE_SETUP_PRE_PLANNER = {
    "asr status",
    "is speech input ready",
    "is voice ready",
    "mic status",
    "microphone status",
    "push to talk status",
    "speech input status",
    "voice input status",
    "voice ready",
    "voice readiness",
    "voice warmup status",
    "whisper status",
    "whisper warmup status",
}
_VOICE_PRIVACY_PRE_PLANNER = {
    "browser microphone privacy",
    "mic privacy",
    "microphone privacy",
    "push to talk privacy",
    "voice privacy",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_VOICE_STOP_PRE_PLANNER)
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_VOICE_SETUP_PRE_PLANNER)
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_VOICE_PRIVACY_PRE_PLANNER)

_FREEZE_GUARDRAIL_PRE_PLANNER = {
    "can codex edit",
    "can codex edit planner",
    "can codex edit send code",
    "can codex edit the planner",
    "can codex edit the send code",
    "can codex touch",
    "can you edit send code",
    "can you edit the planner",
    "can you edit the send code",
    "freeze list",
    "freeze status",
    "frozen files",
    "guardrail status",
    "guardrails",
    "live freeze list",
    "live channel proof status",
    "live proof freeze",
    "live proof matrix",
    "live proof status",
    "live test matrix",
    "proofs pending",
    "show freeze list",
    "show guardrails",
    "what can codex edit",
    "what can codex touch",
    "what can you edit",
    "what files are frozen",
    "what is frozen",
    "what is the freeze list",
    "what is the live test matrix",
    "what should codex avoid",
    "what should codex leave alone",
    "what should codex not edit",
    "what should codex not touch",
    "what should you not edit",
    "what should you not touch",
    "why frozen",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_FREEZE_GUARDRAIL_PRE_PLANNER)

_CONTROL_PLANE_PRE_PLANNER = {
    "control plane health",
    "control plane status",
    "control status",
    "open cockpit",
    "open control plane",
    "show me cockpit",
    "show me control plane",
    "show me dashboard status",
    "show me the cockpit",
    "show me the control panel",
    "show me the control plane",
    "what can u do",
    "where is control plane",
    "where is the cockpit",
    "where is the control plane",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_CONTROL_PLANE_PRE_PLANNER)

_SHOW_ME_DISCOVERY_PRE_PLANNER = {
    "show me what broke",
    "show me what is broken",
    "show me what you can do",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_SHOW_ME_DISCOVERY_PRE_PLANNER)

_CAPABILITY_DISCOVERY_PRE_PLANNER = {
    "can i talk to you",
    "can jarvis check weather",
    "can jarvis check email",
    "can jarvis send messages",
    "can you browse the web",
    "can you brief me every morning",
    "can you calculate things",
    "can you call my contacts",
    "can you call people",
    "can you check email",
    "can you check stock prices",
    "can you check weather",
    "can you convert currency",
    "can you define words",
    "can you give morning brief",
    "can you get weather",
    "can you hear me",
    "can you look up wikipedia",
    "can you message people",
    "can you read email",
    "can you read files",
    "can you read my calendar",
    "can you read news",
    "can you remember things",
    "can you research the web",
    "can you run commands",
    "can you search memory",
    "can you search the internet",
    "can you send imessage",
    "can you send kakao",
    "can you send kakao messages",
    "can you send kakaotalk",
    "can you send messages",
    "can you send telegram",
    "can you send telegram messages",
    "can you set reminders",
    "can you set timers",
    "can you summarize news",
    "can you take notes",
    "can you tell me bitcoin price",
    "can you tell me weather",
    "can you text people",
    "can you translate",
    "can you use voice",
    "can you write emails",
    "can you write text",
    "do you support kakao",
    "do you support telegram",
    "how do i use voice",
    "what is broken",
    "what s broken",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_CAPABILITY_DISCOVERY_PRE_PLANNER)

_LIVE_PROOF_TEST_PRE_PLANNER = {
    "live proof result",
    "live proof results",
    "live proof test status",
    "live test result",
    "live test results",
    "pending live proofs",
    "pending live tests",
    "proof pending status",
    "test matrix status",
    "tests pending",
    "what live proof do you need",
    "what live proofs are pending",
    "what live tests are pending",
    "what channels should i test",
    "which channels should i test",
    "what live channels are pending",
    "which channels need live proof",
    "what channels need proof",
    "what proof is pending",
    "what proofs are still pending",
    "what live results do you need from me",
    "what results do you need from me",
    "what should i test",
    "what should operator test",
    "what tests should i run",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_LIVE_PROOF_TEST_PRE_PLANNER)

_OPERATOR_WORKFLOW_EVAL_PRE_PLANNER = {
    "can jarvis safely send korean messages",
    "hangul message proof",
    "hangul send proof",
    "is hangul messaging covered",
    "is korean message covered",
    "is korean messaging covered",
    "is korean messaging tested",
    "is korean telegram covered",
    "korean message eval",
    "korean message proof",
    "korean message test",
    "korean send proof",
    "korean telegram eval",
    "korean telegram proof",
    "korean telegram test",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_OPERATOR_WORKFLOW_EVAL_PRE_PLANNER)

_EVIDENCE_PROOF_PRE_PLANNER = {
    "evidence",
    "evidence report",
    "evidence status",
    "proof",
    "proofs",
    "show evidence",
    "show proof",
    "show proofs",
    "show me evidence",
    "show me proof",
    "show me proofs",
    "show the evidence",
    "show the proof",
    "show the proofs",
    "what evidence do we have",
    "what evidence do you have",
    "what evidence exists",
    "what proof do we have",
    "what proof do you have",
    "what proof is there",
    "what proofs do we have",
    "what proofs exist",
    "what have we verified",
    "what is verified",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_EVIDENCE_PROOF_PRE_PLANNER)

_CHANGE_REVIEW_PRE_PLANNER = {
    "code changes status",
    "diff status",
    "patch status",
    "show changed files",
    "show modified files",
    "what changed in code",
    "what did codex change",
    "what did codex just change",
    "what did codex just do",
    "what did you edit",
    "what files changed",
    "what files did codex edit",
    "what should claude review",
    "what should i review in this change",
    "which files changed",
    "which files did you edit",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_CHANGE_REVIEW_PRE_PLANNER)

_AGENT_LANDSCAPE_RESEARCH_PRE_PLANNER = {
    "agent landscape research",
    "agent research note",
    "ai agent landscape",
    "ai agent landscape research",
    "all ai agent research",
    "all ai agents research",
    "devin research",
    "every ai agent research",
    "every ai agents research",
    "hermes research",
    "jarvis direction",
    "jarvis moat",
    "jarvis models research",
    "jarvis product strategy",
    "jarvis strategy",
    "lindy research",
    "manus research",
    "openclaw research",
    "other ai agent research",
    "other ai agents research",
    "other jarvis models",
    "other jarvis models research",
    "what did claude recommend about companions",
    "what did claude say about zoey",
    "what did the agent research say",
    "what did research say to avoid",
    "what did you find about ai agents",
    "what did you find about other ai agents",
    "what did you find about other jarvis models",
    "what did you find about zoey",
    "what did you learn from zoey",
    "what about other jarvis models",
    "what is jarvis moat",
    "what is the jarvis strategy",
    "what is the jarvis moat",
    "what is the right move for jarvis",
    "what is the strategy after the research",
    "what should jarvis avoid from ai agents",
    "what should jarvis avoid from zoey",
    "what should jarvis borrow from lindy",
    "what should jarvis borrow from openclaw",
    "what should jarvis borrow from zoey",
    "what should jarvis build after zoey research",
    "what should jarvis copy from zoey",
    "what should jarvis copy from zoey os",
    "what should jarvis copy from lindy",
    "what should jarvis copy from openclaw",
    "what should jarvis do after the research",
    "what should we do after zoey research",
    "what was the ai agent research recommendation",
    "why no companion personas",
    "why no companions",
    "why not companion personas",
    "should jarvis add companion personas",
    "should jarvis add companions",
    "should jarvis build companion personas",
    "should jarvis build companions",
    "should jarvis copy lindy",
    "should jarvis copy openclaw",
    "should jarvis copy zoey",
    "should jarvis copy zoey os",
    "should jarvis have companion personas",
    "should jarvis have companions",
    "should jarvis use companion personas",
    "should jarvis use companions",
    "zoey os research",
    "zoey research",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_AGENT_LANDSCAPE_RESEARCH_PRE_PLANNER)

_HANDOFF_WORK_QUEUE_PRE_PLANNER = {
    "check codex tasks",
    "check current instructions",
    "codex current instructions",
    "codex tasks",
    "codex work queue",
    "current instructions",
    "current work queue",
    "handoff",
    "handoff brief status",
    "handoff notice",
    "handoff report",
    "handoff status",
    "show current instructions",
    "what are current instructions",
    "what did claude hand off",
    "what did claude leave",
    "what did claude say",
    "what is the current work queue",
    "what is the handoff",
    "what should codex do next",
    "what should codex work on",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_HANDOFF_WORK_QUEUE_PRE_PLANNER)

_BUILD_PROGRESS_STATUS_PRE_PLANNER = {
    "agi progress",
    "build status",
    "how is jarvis going",
    "jarvis build progress",
    "jarvis progress",
    "progress",
    "progress report",
    "project status",
    "show build progress",
    "what have you built",
    "what is the build progress",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_BUILD_PROGRESS_STATUS_PRE_PLANNER)

_ROADMAP_STATUS_PRE_PLANNER = {
    "agi roadmap",
    "jarvis roadmap",
    "roadmap status",
    "show roadmap",
    "show the roadmap",
    "what is next on the roadmap",
    "what is the roadmap",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_ROADMAP_STATUS_PRE_PLANNER)

_GOAL_STATUS_PRE_PLANNER = {
    # "active goals" / "goals" / "show goals" / "show my goals" / "what are my
    # goals" / "what goals do i have" are deliberately NOT in this set: they
    # already resolve deterministically and safely to the READ_ONLY
    # `list_goals` tool via the planner's `goal_list_alias_match`, so routing
    # them through a suggestion first ("Did you mean 'list goals'? Send that
    # and I'll run it.") was a pure UX regression that made a working,
    # unambiguous command take two turns instead of one. Real gap found live
    # 2026-07-09. "goal status" and "goals status" stay here: they do not
    # resolve as cleanly (the former actually false-matches `create_goal`),
    # so the extra confirmation step is protective, not redundant.
    "goal status",
    "goals status",
}
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_GOAL_STATUS_PRE_PLANNER)

_CHANNEL_HEALTH_PRE_PLANNER_SERVICES = ("telegram", "kakao", "imessage", "instagram")
_CHANNEL_HEALTH_PRE_PLANNER_GENERIC = {
    "can i resend",
    "can i send again",
    "channel failures",
    "channel last failure",
    "channel last success",
    "channel success count",
    "channel 7 day success count",
    "channel seven day success count",
    "did it go through",
    "did it send",
    "did my message go through",
    "did my message send",
    "did the message go through",
    "did the message send",
    "message delivery health",
    "ready to resend",
    "ready to send again",
    "resend readiness",
    "resend status",
    "safe to resend",
    "safe to send again",
    "should i resend",
    "should i send again",
    "show channel failures",
    "was it sent",
    "was message delivered",
    "was message sent",
    "was my message delivered",
    "was my message sent",
    "was the message delivered",
    "was the message sent",
}
_RECOVERY_READINESS_PRE_PLANNER = {
    "can i retry",
    "can i retry now",
    "is retry ready",
    "ready to retry",
    "retry readiness",
    "retry status",
    "safe to retry",
    "should i retry",
}
_CHANNEL_RETRY_PRE_PLANNER = {
    "send again",
    "try send again",
    "try sending again",
}
_CHANNEL_HEALTH_PRE_PLANNER_SUFFIXES = (
    "delivery health",
    "last failure",
    "last success",
    "success count",
    "7 day success count",
    "seven day success count",
)
_CHANNEL_HEALTH_PRE_PLANNER_PATTERNS = (
    "did {service} go through",
    "did {service} send",
    "did {service} work recently",
    "did my {service} go through",
    "did the {service} go through",
    "last {service} failure",
    "last {service} success",
    "resend {service}",
    "retry {service}",
    "send {service} again",
    "try {service} again",
    "was {service} delivered",
    "was {service} sent",
    "what broke on {service}",
    "why did {service} break",
    "why did {service} not send",
)
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_CHANNEL_HEALTH_PRE_PLANNER_GENERIC)
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_RECOVERY_READINESS_PRE_PLANNER)
_PRE_PLANNER_SUGGESTION_NORMALIZED.update(_CHANNEL_RETRY_PRE_PLANNER)
for _service in _CHANNEL_HEALTH_PRE_PLANNER_SERVICES:
    for _suffix in _CHANNEL_HEALTH_PRE_PLANNER_SUFFIXES:
        _PRE_PLANNER_SUGGESTION_NORMALIZED.add(f"{_service} {_suffix}")
    for _pattern in _CHANNEL_HEALTH_PRE_PLANNER_PATTERNS:
        _PRE_PLANNER_SUGGESTION_NORMALIZED.add(_pattern.format(service=_service))

_PRE_PLANNER_SUGGESTION_COMPACT = {
    "감사기록",
    "감사기록보여줘",
    "감사로그",
    "감사로그보여줘",
    "가드레일",
    "가드레일상태",
    "고장상태",
    "최근오류",
    "최근오류보여줘",
    "최근에러",
    "최근에러보여줘",
    "마지막오류",
    "마지막오류보여줘",
    "마지막에러",
    "마지막에러보여줘",
    "동결상태",
    "동결파일",
    "뭘수정하면안돼",
    "라이브채널대기",
    "라이브채널뭐테스트해",
    "라이브채널어떤거테스트해",
    "어떤채널검증해야해",
    "어떤채널테스트해",
    "채널검증뭐남았어",
    "채널뭐테스트해",
    "반복작업",
    "반복작업상태",
    "백그라운드작업",
    "백그라운드작업상태",
    "브리핑예약",
    "브리핑작업상태",
    "수정금지파일",
    "프리즈리스트",
    "프리즈상태",
    "브리핑문제",
    "브리핑실패",
    "브리핑이슈",
    "브리핑오류",
    "브리핑안됨",
    "브리핑안왔어",
    "브리핑안옴",
    "브리핑어디",
    "메세지상태",
    "메세지문제",
    "메세지실패",
    "메세지이슈",
    "메세지오류",
    "메세지안됨",
    "메시지상태",
    "메시지문제",
    "메시지실패",
    "메시지이슈",
    "메시지오류",
    "메시지안됨",
    "메시징상태",
    "메시징실패",
    "메시징오류",
    "메시징안됨",
    "문자상태",
    "문자문제",
    "문자실패",
    "문자이슈",
    "문자오류",
    "문자안됨",
    "무슨문제있어",
    "무슨문제야",
    "문제상태",
    "실행기록",
    "실행기록보여줘",
    "실행로그",
    "스케줄상태",
    "스케줄확인",
    "스케줄러상태",
    "아침브리프예약",
    "아침브리프문제",
    "아침브리프실패",
    "아침브리프이슈",
    "아침브리프오류",
    "아침브리프안왔어",
    "아침브리프안옴",
    "아침브리프어디",
    "아침브리핑문제",
    "아침브리핑실패",
    "아침브리핑이슈",
    "아침브리핑오류",
    "아이메시지상태",
    "아이메시지문제",
    "아이메시지실패",
    "아이메시지이슈",
    "아이메시지오류",
    "아이메시지안됨",
    "아이메시지확인",
    "예약작업",
    "예약작업상태",
    "자동화상태",
    "자동화확인",
    "전송상태",
    "전송문제",
    "전송실패",
    "전송이슈",
    "전송오류",
    "전송안됨",
    "카카오상태",
    "카카오문제",
    "카카오실패",
    "카카오이슈",
    "카카오오류",
    "카카오안됨",
    "카카오고장",
    "카카오확인",
    "카톡상태",
    "카톡문제",
    "카톡실패",
    "카톡이슈",
    "카톡오류",
    "카톡안됨",
    "카톡고장",
    "카톡확인",
    "모바일상태",
    "모바일컨트롤",
    "마지막실행",
    "마지막작업",
    "최근기록",
    "최근로그",
    "최근실행",
    "최근에뭐했어",
    "최근작업",
    "최근활동",
    "모델고장",
    "모델문제",
    "모델실패",
    "모델안돼",
    "모델안되",
    "모델안됨",
    "모델오류",
    "모델이슈",
    "모닝브리프문제",
    "모닝브리프실패",
    "모닝브리프이슈",
    "모닝브리프오류",
    "모닝브리프안됨",
    "모닝브리프안왔어",
    "모닝브리프안옴",
    "모닝브리프어디",
    "모닝브리프왜안왔어",
    "모닝브리핑문제",
    "모닝브리핑실패",
    "모닝브리핑이슈",
    "폰단축키",
    "폰도움말",
    "폰명령어",
    "폰에서뭐할수있어",
    "폰상태",
    "폰컨트롤센터",
    "폰컨트롤상태",
    "왜메세지실패",
    "왜메시지실패",
    "왜메시지안보내짐",
    "왜메시지안보내졌어",
    "왜문자실패",
    "왜아이메시지실패",
    "왜전송실패",
    "왜전송안됨",
    "왜전송안보내짐",
    "왜전송안보내졌어",
    "왜카카오실패",
    "왜카톡실패",
    "왜텔레그램실패",
    "왜텔레그램안보내짐",
    "왜텔레그램안보내졌어",
    "왜승인실패",
    "왜승인안됨",
    "왜승인오류",
    "왜실패했어",
    "왜인스타그램실패",
    "왜인스타실패",
    "왜전화실패",
    "왜통화실패",
    "허가문제",
    "허가실패",
    "허가안돼",
    "허가안되",
    "허가안됨",
    "허가오류",
    "텔레그램단축키",
    "텔레그램도움말",
    "텔레그램헬스",
    "텔레그램명령어",
    "텔레그램에서뭐할수있어",
    "텔레그램상태",
    "텔레그램문제",
    "텔레그램실패",
    "텔레그램이슈",
    "텔레그램오류",
    "텔레그램안됨",
    "텔레그램고장",
    "텔레그램컨트롤센터",
    "텔레그램컨트롤상태",
    "텔레그램확인",
    "텔레그램메시지예시",
    "인스타그램상태",
    "인스타그램기능",
    "인스타그램문제",
    "인스타그램실패",
    "인스타그램이슈",
    "인스타그램오류",
    "인스타그램안됨",
    "인스타그램예시",
    "인스타그램명령어",
    "인스타상태",
    "인스타기능",
    "인스타문제",
    "인스타실패",
    "인스타이슈",
    "인스타오류",
    "인스타안됨",
    "인스타예시",
    "인스타명령어",
    "가능한기능",
    "가능한도구",
    "기능리스트",
    "사용가능한기능",
    "사용가능한도구",
    "도구목록",
    "도구리스트",
    "도구보고서",
    "도구상태",
    "도구헬스",
    "도구레지스트리",
    "도움말",
    "실패목록",
    "실패보여줘",
    "실패상태",
    "건강상태",
    "오류상태",
    "명령예시",
    "명령어예시",
    "캘린더기능",
    "일정기능",
    "이메일기능",
    "메일기능",
    "리마인더기능",
    "타이머기능",
    "할일기능",
    "투두기능",
    "날씨기능",
    "뉴스기능",
    "정보기능",
    "시장기능",
    "마켓기능",
    "주식기능",
    "코인기능",
    "암호화폐기능",
    "번역기능",
    "환율기능",
    "유틸기능",
    "유틸리티기능",
    "검색기능",
    "조사기능",
    "리서치기능",
    "글쓰기기능",
    "작성기능",
    "메모기능",
    "노트기능",
    "기억기능",
    "메모리기능",
    "모닝브리프보냈어",
    "모닝브리프예약확인",
    "브리핑실행됐어",
    "브리핑언제야",
    "아침브리핑보냈어",
    "예약작업문제",
    "예약작업안돌아",
    "스케줄러실행됐어",
    "다음브리핑언제",
    "오늘브리핑보냈어",
    "검토할거있어",
    "나기다리는거있어",
    "내가봐야할거있어",
    "내가승인해야할거있어",
    "대기상태",
    "뭐기다려",
    "리뷰큐",
    "승인기능",
    "승인고장",
    "승인대기문제",
    "승인대기안됨",
    "승인문제",
    "승인버튼고장",
    "승인버튼문제",
    "승인버튼실패",
    "승인버튼안돼",
    "승인버튼안되",
    "승인버튼안됨",
    "승인버튼오류",
    "승인실패",
    "승인안돼",
    "승인안되",
    "승인안됨",
    "승인이슈",
    "승인오류",
    "승인큐멈춤",
    "승인큐문제",
    "안전기능",
    "연락처기능",
    "연락처명령어",
    "연락처예시",
    "연락처검색기능",
    "연락처검색명령어",
    "연락처검색예시",
    "연락처조회기능",
    "연락처조회명령어",
    "연락처조회예시",
    "연락처찾기기능",
    "주소록기능",
    "주소록명령어",
    "주소록예시",
    "주소록검색기능",
    "주소록검색명령어",
    "주소록검색예시",
    "주소록조회기능",
    "주소록조회명령어",
    "주소록조회예시",
    "대시보드기능",
    "콕핏기능",
    "내부오케스트레이션",
    "내부오케스트레이션상태",
    "서브에이전트상태",
    "서브에이전트상태확인",
    "에이전트상태",
    "에이전트상태확인",
    "준비된에이전트",
    "워커상태",
    "작업자상태",
    "메세지기능",
    "메세지실패",
    "메세지오류",
    "메세지안됨",
    "메세지예시",
    "메세지명령어",
    "메시지기능",
    "메시지실패",
    "메시지오류",
    "메시지안됨",
    "메시지예시",
    "메시지명령어",
    "메시징기능",
    "메시징실패",
    "메시징오류",
    "메시징안됨",
    "메시징예시",
    "메시징명령어",
    "뭐부터해",
    "뭘보낼수있어",
    "뭘물어볼수있어",
    "무엇부터해",
    "무엇을보낼수있어",
    "무엇을물어볼수있어",
    "브레인고장",
    "브레인문제",
    "브레인실패",
    "브레인안돼",
    "브레인안되",
    "브레인안됨",
    "브레인오류",
    "브레인이슈",
    "두뇌고장",
    "두뇌문제",
    "두뇌실패",
    "두뇌안됨",
    "두뇌오류",
    "사용법",
    "샘플명령어",
    "시스템상태",
    "시스템헬스",
    "상태보고서",
    "자비스보고서",
    "자비스도움말",
    "자비스사용법",
    "자비스헬스",
    "예시",
    "메세지안보내짐",
    "메세지못보냄",
    "메시지안보내짐",
    "메시지못보냄",
    "아이메시지기능",
    "아이메시지예시",
    "아이메시지명령어",
    "전체상태",
    "전체헬스",
    "전반상태",
    "전반헬스",
    "진단상태",
    "진단보고서",
    "텔레그램예시",
    "텔레그램기능",
    "폰예시",
    "시작명령어",
    "자주쓰는명령어",
    "툴목록",
    "툴리스트",
    "문자기능",
    "문자실패",
    "문자오류",
    "문자안됨",
    "문자예시",
    "문자명령어",
    "전송기능",
    "전송실패",
    "전송오류",
    "전송안됨",
    "전송안돼",
    "전송안되",
    "전송안보내짐",
    "전송못보냄",
    "전송예시",
    "전송명령어",
    "카카오기능",
    "카카오실패",
    "카카오오류",
    "카카오안됨",
    "카카오고장",
    "카카오예시",
    "카카오명령어",
    "카톡기능",
    "카톡실패",
    "카톡오류",
    "카톡안됨",
    "카톡고장",
    "카톡예시",
    "카톡명령어",
    "전화기능",
    "전화문제",
    "전화실패",
    "전화이슈",
    "전화오류",
    "전화안됨",
    "전화예시",
    "전화명령어",
    "통화기능",
    "통화문제",
    "통화실패",
    "통화이슈",
    "통화오류",
    "통화안됨",
    "통화예시",
    "통화명령어",
    "음성기능",
    "음성고장",
    "음성예시",
    "음성문제",
    "음성실패",
    "음성안돼",
    "음성안되",
    "음성안됨",
    "음성오류",
    "음성이슈",
    "음성명령어",
    "음성명령어예시",
    "올라마고장",
    "올라마문제",
    "올라마실패",
    "올라마안돼",
    "올라마안되",
    "올라마안됨",
    "올라마오류",
    "올라마이슈",
    "오라마고장",
    "오라마문제",
    "오라마실패",
    "오라마안됨",
    "오라마오류",
    "로컬모델고장",
    "로컬모델문제",
    "로컬모델실패",
    "로컬모델안됨",
    "로컬모델오류",
    "플래너고장",
    "플래너문제",
    "플래너실패",
    "플래너안됨",
    "플래너오류",
    "플래너모델고장",
    "플래너모델문제",
    "플래너모델실패",
    "플래너모델안됨",
    "플래너모델오류",
    "llm고장",
    "llm문제",
    "llm실패",
    "llm안됨",
    "llm오류",
    "채팅모델고장",
    "채팅모델문제",
    "채팅모델실패",
    "채팅모델안됨",
    "채팅모델오류",
    "마이크고장",
    "마이크문제",
    "마이크실패",
    "마이크안돼",
    "마이크안되",
    "마이크안됨",
    "마이크오류",
    "마이크이슈",
    "목소리고장",
    "목소리기능",
    "목소리문제",
    "목소리예시",
    "목소리실패",
    "목소리안돼",
    "목소리안되",
    "목소리안됨",
    "목소리오류",
    "목소리이슈",
    "녹음문제",
    "녹음실패",
    "녹음안돼",
    "녹음안되",
    "녹음안됨",
    "녹음오류",
    "녹음이슈",
    "보이스고장",
    "보이스기능",
    "보이스문제",
    "보이스예시",
    "보이스명령어",
    "보이스실패",
    "보이스안돼",
    "보이스안되",
    "보이스안됨",
    "보이스오류",
    "보이스이슈",
    "위스퍼문제",
    "위스퍼실패",
    "위스퍼안돼",
    "위스퍼안되",
    "위스퍼안됨",
    "위스퍼오류",
    "전사문제",
    "전사실패",
    "전사안됨",
    "전사오류",
    "대시보드보여줘",
    "뭐할수있는지보여줘",
    "콕핏보여줘",
    "컨트롤플레인보여줘",
    "툴보고서",
    "툴상태",
    "툴헬스",
    "툴레지스트리",
}

_PRE_PLANNER_SUGGESTION_COMPACT.update(
    {
        "계산가능해",
        "기억할수있어",
        "날씨확인가능해",
        "뉴스읽을수있어",
        "뉴스요약가능해",
        "리마인더가능해",
        "명령실행가능해",
        "말로할수있어",
        "메모가능해",
        "메시지보낼수있어",
        "메세지보낼수있어",
        "문자보낼수있어",
        "번역가능해",
        "비트코인가격볼수있어",
        "웹검색가능해",
        "위키찾아볼수있어",
        "아침브리핑가능해",
        "음성가능해",
        "이메일확인가능해",
        "전화걸수있어",
        "주식확인가능해",
        "카톡보낼수있어",
        "캘린더볼수있어",
        "타이머설정가능해",
        "텔레그램보낼수있어",
        "파일읽을수있어",
        "환율계산가능해",
        "글쓰기가능해",
        "메세지갔어",
        "메세지갔나요",
        "메세지보내졌어",
        "메세지보냈어",
        "메시지갔어",
        "메시지갔나요",
        "메시지보내졌어",
        "메시지보냈어",
        "다시보내도돼",
        "다시보내도되나요",
        "다시보낼까",
        "전송갔어",
        "전송됐나요",
        "전송됐어",
        "전송됬나요",
        "전송됬어",
        "재시도상태",
        "재시도준비",
        "재시도해도돼",
        "재시도해도되나요",
        "재전송상태",
        "채널마지막실패",
        "채널최근실패",
        "채널최근성공",
        "채널성공횟수",
        "채널실패보여줘",
        "채널실패기록",
        "텔레그램갔어",
        "텔레그램마지막실패",
        "텔레그램보내졌어",
        "텔레그램보냈어",
        "텔레그램다시보낼까",
        "텔레그램재시도",
        "텔레그램재전송",
        "텔레그램최근실패",
        "텔레그램최근성공",
        "텔레그램성공횟수",
        "텔레그램최근에됐어",
        "카카오갔어",
        "카카오마지막실패",
        "카카오보내졌어",
        "카카오보냈어",
        "카카오다시보낼까",
        "카카오재시도",
        "카카오재전송",
        "카카오최근실패",
        "카카오최근성공",
        "카카오성공횟수",
        "카톡갔어",
        "카톡마지막실패",
        "카톡보내졌어",
        "카톡보냈어",
        "카톡다시보낼까",
        "카톡재시도",
        "카톡재전송",
        "카톡최근실패",
        "카톡최근성공",
        "카톡성공횟수",
        "아이메시지갔어",
        "아이메시지마지막실패",
        "아이메시지보내졌어",
        "아이메시지보냈어",
        "아이메시지다시보낼까",
        "아이메시지재시도",
        "아이메시지재전송",
        "아이메시지최근실패",
        "아이메시지최근성공",
        "아이메시지성공횟수",
        "인스타그램마지막실패",
        "인스타그램최근실패",
        "인스타그램최근성공",
        "인스타그램성공횟수",
        "제임스평가",
        "제임스평가팩",
        "제임스워크플로우",
        "제임스워크플로우평가",
        "제임스워크플로우테스트",
        "실제워크플로우평가",
        "실제워크플로우테스트",
        "워크플로우평가",
        "워크플로우테스트",
        "검증대기",
        "검증대기뭐야",
        "라이브검증대기",
        "라이브검증상태",
        "라이브테스트결과",
        "라이브테스트대기",
        "라이브테스트뭐해야해",
        "라이브테스트상태",
        "뭐테스트해야해",
        "테스트매트릭스",
        "테스트매트릭스보여줘",
        "테스트매트릭스상태",
        "테스트대기",
        "테스트해야할거",
        "라이브테스트매트릭스",
        "라이브매트릭스상태",
        "라이브증명상태",
        "채널증명상태",
        "채널증명매트릭스",
        "컨트롤플레인",
        "컨트롤패널",
        "자비스컨트롤플레인",
        "자비스컨트롤패널",
        "대시보드상태",
        "상태대시보드",
        "능력대시보드",
        "기능대시보드",
        "신뢰대시보드",
        "신뢰보고서",
        "신뢰상태",
        "신뢰도상태",
        "신뢰도보고서",
        "믿어도돼",
        "믿어도되나요",
        "믿을만해",
        "신뢰가능",
        "신뢰할수있어",
        "신뢰할수있나요",
        "경계",
        "못하는것",
        "무엇을못해",
        "무엇을할수없어",
        "뭐못해",
        "뭘못해",
        "뭘할수없어",
        "한계",
        "고위험도구",
        "고위험명령",
        "권한필요",
        "권한필요한것",
        "뭐승인필요",
        "승인필요",
        "승인필요한것",
        "승인필요한명령",
        "승인이필요한것",
        "승인이필요한명령",
        "어떤도구가승인필요",
        "어떤명령이승인필요",
        "위험한도구",
        "위험한명령",
        "안전하게써도돼",
        "안전하게써도되나요",
        "자비스믿어도돼",
        "자비스믿어도되나요",
        "자비스믿을만해",
        "자비스신뢰가능",
        "자비스신뢰상태",
        "자비스경계",
        "자비스한계",
        "자비스뭐못해",
        "자비스뭘못해",
        "자비스못하는것",
        "자비스안전하게써도돼",
        "자비스안전하게써도되나요",
        "기능상태",
        "능력상태",
        "안전대시보드",
        "승인대시보드",
        "리스크매트릭스",
        "위험수준",
        "위험수준보여줘",
        "위험상태",
        "그만말해",
        "말그만",
        "말멈춰",
        "목소리멈춰",
        "보이스멈춰",
        "자비스조용히",
        "음성음소거",
        "마이크상태",
        "마이크개인정보",
        "마이크프라이버시",
        "음성워밍업상태",
        "음성준비됐어",
        "위스퍼상태",
        "위스퍼워밍업상태",
        "푸시투톡상태",
        "푸시투톡개인정보",
        "증거",
        "증거상태",
        "증거보고서",
        "증거보여줘",
        "증거있어",
        "증거뭐있어",
        "증명상태",
        "증명보여줘",
        "근거상태",
        "근거보여줘",
        "근거있어",
        "근거뭐있어",
        "완료증거",
        "신뢰증거",
        "검증증거",
        "검증증명",
        "뭐가검증됐어",
        "뭐가검증됐나요",
        "무엇이검증됐어",
        "무엇이검증됐나요",
        "무엇을검증했어",
        "무엇을검증했나요",
        "최근검증",
        "최신검증",
        "최근증거",
        "최근증명",
        "agi상태",
        "agi준비",
        "agi준비상태",
        "agi게이트",
        "agi게이트상태",
        "에이지아이상태",
        "에이지아이준비",
        "에이지아이준비상태",
        "하네스상태",
        "하네스준비",
        "하네스준비상태",
        "자비스끝났어",
        "자비스끝났니",
        "자비스다끝났어",
        "자비스완료됐어",
        "자비스완료됬어",
        "자비스완료상태",
        "자비스완성됐어",
        "완료상태",
        "완료주장",
        "완료게이트",
        "완료클레임",
        "완료주장게이트",
        "완료증명",
        "다음완료증명",
    }
)

_PRE_PLANNER_SUGGESTION_COMPACT.update(
    {
        "agi진행상황",
        "agi진행",
        "빌드상태",
        "빌드진행상황",
        "빌드진행",
        "자비스진행상황",
        "자비스진행",
        "진행상황",
        "진행보고",
        "뭐만들었어",
        "무엇을만들었어",
        "agi로드맵",
        "로드맵보여줘",
        "자비스로드맵",
        "다음로드맵",
        "다음에뭐만들어",
        "다음에무엇을만들어",
        "목표",
        "목표상태",
        # "목표보여줘" / "목표목록" removed 2026-07-10 (Fable checkpoint plan
        # item B2): the planner now routes these deterministically to the
        # READ_ONLY `list_goals` tool (spaced and unspaced forms), so the
        # pre-planner suggestion made a working, unambiguous command take two
        # turns -- the same pure-UX-regression class documented at
        # _GOAL_STATUS_PRE_PLANNER above. Bare "목표"/"목표상태"/"자비스목표"/
        # "활성목표" stay: those are genuinely ambiguous between the user's
        # personal goal list and the Jarvis build roadmap, so the extra
        # confirmation step is protective there, not redundant.
        "자비스목표",
        "활성목표",
    }
)

_PRE_PLANNER_SUGGESTION_COMPACT.update(
    {
        "claude가뭐남겼어",
        "claude가뭐라고했어",
        "claude핸드오프",
        "claude인수인계",
        "클로드가뭐남겼어",
        "클로드가뭐라고했어",
        "클로드핸드오프",
        "클로드인수인계",
        "인수인계",
        "인수인계상태",
        "핸드오프",
        "핸드오프상태",
        "코덱스작업",
        "코덱스작업뭐야",
        "코덱스작업큐",
        "코덱스작업목록",
        "코덱스태스크",
        "코덱스태스크확인",
        "코덱스할일",
        "코덱스현재지시",
        "코덱스현재지시사항",
        "현재지시",
        "현재지시사항",
        "현재지시사항보여줘",
        "작업큐보여줘",
        "작업목록보여줘",
        "작업대기열",
        "작업대기열보여줘",
        "현재작업큐",
        "현재작업목록",
        "뭐수정했어",
        "어떤파일바꿨어",
        "변경파일보여줘",
        "수정파일보여줘",
        "코덱스가뭐바꿨어",
        "코덱스변경사항",
        "패치상태",
        "변경사항확인",
        "다음뭐해야해",
        "다음뭐해",
        "다음뭐할까",
        "다음에뭘실행해",
        "다음에뭘실행해야해",
        "무슨명령실행해",
        "무슨명령실행해야해",
        "뭘실행해야해",
        "자비스다음단계",
        "자비스다음뭐해",
        "자비스뭐부터해",
        "조이조사결과",
        "zoey조사결과",
        "에이전트조사결과",
        "ai에이전트랜드스케이프",
        "모든ai에이전트조사",
        "전체ai에이전트조사",
        "다른ai에이전트조사",
        "다른자비스모델",
        "다른jarvis모델",
        "자비스모델조사",
        "다른자비스모델조사",
        "자비스차별점",
        "자비스해자",
        "조이에서피해야할것",
        "조이따라가야해",
        "조이따라야해",
        "조이따라할까",
        "조이따라해야해",
        "자비스동반자해야해",
        "자비스방향",
        "자비스전략",
        "자비스조이이후뭐만들까",
        "자비스조이이후뭐만들어",
        "자비스조이따라가야해",
        "자비스조이따라야해",
        "자비스조이따라할까",
        "자비스조이따라해야해",
        "자비스컴패니언해야해",
        "조이이후뭐만들까",
        "조이이후뭐만들어",
        "린디연구",
        "마누스연구",
        "오픈클로연구",
        "헤르메스연구",
        "왜동반자안해",
        "왜컴패니언안해",
        "동반자페르소나피해야해",
        "동반자레이어해야해",
        "컴패니언페르소나피해야해",
        "컴패니언레이어해야해",
        "한국어메시지검증",
        "한국어메시지증명",
        "한국어메시지테스트됐어",
        "한국어텔레그램검증",
        "한국어텔레그램안전해",
        "한국어텔레그램증명",
        "한글메시지증명",
        "한글전송증명",
        "가상연락처이메시지테스트",
        "가상연락처이전송증명",
        "가상연락처이전송테스트",
    }
)


def _runtime_scrub_text(value: object) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", str(value or ""))


_RUNTIME_KNOWLEDGE_PROMOTION_TOKEN_RE = re.compile(
    r"(\bpromote\s+memory\s+#?\d+\s+revision\s+#?\d+\s+token\s+)"
    r"[0-9a-f]{64}(?=\s+to\s+(?:a\s+)?(?:decision|preference)\s*:)",
    re.IGNORECASE,
)
_RUNTIME_TASK_COMPLETION_EVIDENCE_RE = re.compile(
    r"(\b(?:complete|finish|done)\s+task\s+#?\d+\s+with\s+).+$",
    re.IGNORECASE | re.DOTALL,
)
_RUNTIME_TASK_COMPLETION_PACKET_EVIDENCE_RE = re.compile(
    r"(\b(?:task\s+completion\s+packet|completion\s+packet\s+for\s+task|"
    r"can\s+complete\s+task|can\s+finish\s+task)\s+#?\d+)"
    r"(?::\s*|\s+).+$",
    re.IGNORECASE | re.DOTALL,
)
_RUNTIME_APPLE_REMINDERS_LIST_RE = re.compile(
    r"((?:show\s+)?(?:my\s+)?(?:apple|mac|macos|iphone)\s+reminders?\s+"
    r"(?:from|in)\s+(?:the\s+)?list\s*:\s*).+?"
    r"(\s*;\s*limit\s*:\s*[^;\r\n]+\s*)?$",
    re.IGNORECASE | re.DOTALL,
)
_RUNTIME_APPLE_REMINDERS_LIST_KO_RE = re.compile(
    r"((?:내\s+)?(?:애플|맥|아이폰)\s*리마인더\s*(?:목록|리스트)\s*:\s*).+?"
    r"(\s*;\s*(?:limit|제한)\s*:\s*[^;\r\n]+\s*)?$",
    re.IGNORECASE | re.DOTALL,
)


def _runtime_sensitive_command_display(value: object) -> str:
    display = _RUNTIME_KNOWLEDGE_PROMOTION_TOKEN_RE.sub(
        r"\1<review-token>", str(value or "")
    )
    display = _RUNTIME_TASK_COMPLETION_EVIDENCE_RE.sub(
        r"\1<completion-evidence>", display
    )
    display = _RUNTIME_TASK_COMPLETION_PACKET_EVIDENCE_RE.sub(
        r"\1: <completion-evidence>", display
    )
    display = _RUNTIME_APPLE_REMINDERS_LIST_RE.sub(
        r"\1<private-list>\2", display
    )
    return _RUNTIME_APPLE_REMINDERS_LIST_KO_RE.sub(
        r"\1<private-list>\2", display
    )


def _runtime_private_arg_key(value: object) -> bool:
    key = str(value or "")
    return key.endswith("_binding") or key in {
        "evidence",
        "review_token",
        "target_binding",
        "verification_run_id",
    }


def _runtime_output_preview(value: object, *, limit: int = MAX_RUNTIME_OUTPUT_PREVIEW_CHARS) -> str:
    text = " ".join(_runtime_scrub_text(value).split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _runtime_persisted_tool_output(result: ToolResult) -> str:
    """Return the durable audit form of a tool result.

    Some personal-data connectors need to show a result to the operator once
    without copying that private content into tool audit rows, assistant
    messages, runtime traces, or later model context. The connector opts in,
    but the runtime owns the durable wording so connector-controlled prose can
    never smuggle private content into the receipt.
    """
    if result.metadata.get("suppress_output_persistence") is not True:
        return str(result.output)
    tool_name = str(result.tool_name or "")
    if re.fullmatch(r"[a-z][a-z0-9_]{0,79}", tool_name) is None:
        tool_name = "tool"
    status = "verified" if result.ok is True else "unavailable"
    return (
        f"{tool_name}: sensitive result {status}; private content was displayed "
        "transiently and not retained."
    )


def _runtime_persisted_response(results: list[ToolResult], response: str) -> str:
    if not any(
        result.metadata.get("suppress_output_persistence") is True
        for result in results
    ):
        return response
    summaries = [_runtime_persisted_tool_output(result) for result in results]
    return "\n".join(summaries) + "\nPrivate result content was displayed transiently and not retained."


def _runtime_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _runtime_scrub_text(value)
    if isinstance(value, dict):
        return {str(key): _runtime_safe_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_runtime_safe_value(item) for item in value]
    return _runtime_scrub_text(value)


def _runtime_public_approval_args(args: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in args.items():
        safe_key = _runtime_output_preview(key, limit=80)
        public[safe_key] = (
            "bound to reviewed memory version"
            if _runtime_private_arg_key(safe_key)
            else _runtime_safe_value(value)
        )
    return public


def _runtime_public_plan(plan: Plan, results: list[ToolResult]) -> Plan:
    changed = False
    actions: list[PlannedAction] = []
    for index, action in enumerate(plan.actions):
        private_argument_failure = (
            index < len(results)
            and results[index].metadata.get("failure_kind")
            in {
                "tool_arguments_invalid",
                "tool_argument_plan_preflight_blocked",
                "auto_mutation_duplicate_operation",
                "auto_mutation_semantic_preflight_rejected",
            }
        )
        if private_argument_failure:
            actions.append(
                replace(
                    action,
                    args={"<redacted>": "<redacted>"} if action.args else {},
                )
            )
            changed = True
        elif type(action.args) is not dict:
            actions.append(action)
        elif any(_runtime_private_arg_key(key) for key in action.args):
            actions.append(replace(action, args=_runtime_public_approval_args(action.args)))
            changed = True
        else:
            actions.append(action)
    return replace(plan, actions=actions) if changed else plan


def _runtime_normalize_command_text(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _runtime_compact_command_text(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _runtime_raw_command_text(value: object) -> str:
    return str(value or "").strip().lower()


def _runtime_should_pre_planner_suggest(user_input: str) -> bool:
    normalized = _runtime_normalize_command_text(user_input)
    if normalized in _PRE_PLANNER_SUGGESTION_NORMALIZED:
        return True
    return _runtime_compact_command_text(user_input) in _PRE_PLANNER_SUGGESTION_COMPACT


def _runtime_registered_read_only_tool_name(raw_input: str, registry: Any | None) -> str | None:
    if raw_input in _RUNTIME_EXACT_TOOL_ALIAS_RAW_EXCLUDES:
        return None
    tool_name = _runtime_registered_read_only_tool(raw_input, registry)
    if tool_name is None:
        return None
    if tool_name in _RUNTIME_EXACT_TOOL_ALIAS_RAW_EXCLUDES:
        return None
    if not _runtime_registered_read_only_tool_accepts_empty_args(tool_name):
        return None
    return tool_name


def _runtime_registered_read_only_tool(raw_input: str, registry: Any | None) -> str | None:
    if registry is None:
        return None
    if not re.fullmatch(r"[a-z][a-z0-9_]*", raw_input):
        return None
    try:
        tool = registry.get(raw_input)
    except Exception:
        return None
    if getattr(getattr(tool, "risk", None), "name", "") != "READ_ONLY":
        return None
    return str(getattr(tool, "name", "") or raw_input)


def _runtime_registered_required_arg_hint_tool(raw_input: str, registry: Any | None) -> str | None:
    if registry is None or not re.fullmatch(r"[a-z][a-z0-9_]*", raw_input):
        return None
    try:
        tool = registry.get(raw_input)
    except Exception:
        return None
    if getattr(getattr(tool, "risk", None), "name", "") not in {"READ_ONLY", "LOCAL_SAFE"}:
        return None
    return str(getattr(tool, "name", "") or raw_input)


def _runtime_tool_allows_required_arg_hint(tool_name: str, registry: Any | None) -> bool:
    if registry is None:
        return True
    try:
        tool = registry.get(tool_name)
    except Exception:
        return False
    return getattr(getattr(tool, "risk", None), "name", "") in {"READ_ONLY", "LOCAL_SAFE"}


def _runtime_read_only_tool_required_arg_suggestion(user_input: str, registry: Any | None) -> str | None:
    raw_input = _runtime_raw_command_text(user_input)
    tool_name = _runtime_registered_required_arg_hint_tool(raw_input, registry)
    if tool_name is None:
        tool_name = _RUNTIME_REQUIRED_ARG_COMMAND_STUBS.get(_runtime_normalize_command_text(user_input))
    if (
        tool_name is None
        or _runtime_registered_read_only_tool_accepts_empty_args(tool_name)
        or not _runtime_tool_allows_required_arg_hint(tool_name, registry)
    ):
        return None
    example = _RUNTIME_EXACT_TOOL_REQUIRED_ARG_EXAMPLES.get(tool_name, f"{tool_name} <arguments>")
    risk_name = "READ_ONLY"
    if registry is not None:
        try:
            risk_name = getattr(getattr(registry.get(tool_name), "risk", None), "name", "") or risk_name
        except Exception:
            pass
    risk_label = "read-only" if risk_name == "READ_ONLY" else "local-safe"
    return (
        f"`{tool_name}` is a {risk_label} command, but it needs arguments before Jarvis can route it.\n"
        f"Did you mean: `{example}`"
    )


def _runtime_risky_tool_required_arg_suggestion(user_input: str, registry: Any | None) -> str | None:
    tool_name = _RUNTIME_REQUIRED_ARG_RISKY_COMMAND_STUBS.get(_runtime_normalize_command_text(user_input))
    if tool_name is None or registry is None:
        return None
    try:
        tool = registry.get(tool_name)
    except Exception:
        return None
    risk_name = getattr(getattr(tool, "risk", None), "name", "")
    if risk_name in {"", "READ_ONLY"}:
        return None
    example = _RUNTIME_RISKY_TOOL_REQUIRED_ARG_EXAMPLES.get(tool_name, f"{tool_name} <arguments>")
    return (
        f"`{tool_name}` is risk level {risk_name} and needs a concrete command before Jarvis can queue approval.\n"
        f"Did you mean: `{example}`"
    )


def _runtime_registered_read_only_tool_accepts_empty_args(tool_name: str) -> bool:
    if tool_name in _RUNTIME_EXACT_TOOL_ALIAS_DYNAMIC_NO_ARG_NAMES:
        return True
    if tool_name.endswith(_RUNTIME_EXACT_TOOL_ALIAS_DYNAMIC_ARG_SUFFIXES):
        return False
    if tool_name.startswith(_RUNTIME_EXACT_TOOL_ALIAS_DYNAMIC_ARG_PREFIXES):
        return False
    return tool_name.endswith(_RUNTIME_EXACT_TOOL_ALIAS_DYNAMIC_NO_ARG_SUFFIXES)


def _runtime_exact_alias_tool_is_read_only(tool_name: str, registry: Any | None) -> bool:
    if registry is None:
        return True
    try:
        tool = registry.get(tool_name)
    except Exception:
        return False
    return getattr(getattr(tool, "risk", None), "name", "") == "READ_ONLY"


def _runtime_projection_repair_alias_plan(
    user_input: str,
    registry: Any | None = None,
) -> Plan | None:
    tool_name = _RUNTIME_LOCAL_SAFE_PROJECTION_REPAIR_ALIASES.get(
        _runtime_normalize_command_text(user_input)
    )
    if tool_name is None or registry is None:
        return None
    try:
        tool = registry.get(tool_name)
    except Exception:
        return None
    if getattr(getattr(tool, "risk", None), "name", "") != "LOCAL_SAFE":
        return None
    return Plan(
        goal="Repair pending owned note projections without repeating source mutations.",
        actions=[
            PlannedAction(
                tool_name,
                {},
                reason="Explicit local-safe owner command for projection reconciliation.",
            )
        ],
        needs_model=False,
        notes="runtime_projection_repair_alias",
        metadata={
            "runtime_projection_repair_alias": True,
            "alias_tool_name": tool_name,
            "approval_granted": False,
            "authorizes_completion_claim": False,
        },
    )


def _runtime_goal_step_status_alias_plan(user_input: str, registry: Any | None = None) -> Plan | None:
    if not _runtime_exact_alias_tool_is_read_only("goal_status", registry):
        return None
    text = _runtime_raw_command_text(user_input)
    patterns = (
        r"^(?:list|show|view|display)\s+(?:the\s+)?(?:steps|step\s+list)\s+(?:for|in|of)\s+(?:goal|project)\s+#?(?P<goal_id>\d+)$",
        r"^(?:steps|step\s+list)\s+(?:for|in|of)\s+(?:goal|project)\s+#?(?P<goal_id>\d+)$",
        r"^(?:goal|project)\s+#?(?P<goal_id>\d+)\s+(?:steps|step\s+list)$",
        r"^(?:show|list|view|display)\s+(?:goal|project)\s+#?(?P<goal_id>\d+)\s+(?:steps|step\s+list)$",
        r"^(?:목표|프로젝트)\s*#?(?P<goal_id>\d+)\s*(?:단계|스텝)(?:\s*(?:목록|보여줘|보여\s+줘|확인))?$",
        r"^(?:단계|스텝)\s*(?:목록\s*)?(?:보여줘\s*)?(?:목표|프로젝트)\s*#?(?P<goal_id>\d+)$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if not match:
            continue
        goal_id = int(match.group("goal_id"))
        if goal_id <= 0:
            return None
        return Plan(
            goal="Show goal steps.",
            actions=[
                PlannedAction(
                    "goal_status",
                    {"goal_id": goal_id},
                    reason="Runtime alias for a read-only goal-step listing command.",
                )
            ],
            needs_model=False,
            notes="runtime_goal_step_status_alias",
            metadata={
                "runtime_goal_step_status_alias": True,
                "alias_input": _runtime_scrub_text(user_input),
                "alias_tool_name": "goal_status",
                "goal_id": goal_id,
            },
        )
    return None


def _runtime_auto_mutation_reconciliation_alias_plan(
    user_input: str,
    registry: Any | None = None,
) -> Plan | None:
    text = _runtime_raw_command_text(user_input)
    queue_phrases = {
        "uncertain mutations",
        "uncertain mutation receipts",
        "auto mutation review",
        "mutation reconciliation queue",
        "불확실 실행",
        "불확실 실행 목록",
    }
    if text in queue_phrases:
        if not _runtime_exact_alias_tool_is_read_only("auto_mutation_reconciliation_queue", registry):
            return None
        return Plan(
            goal="Review uncertain local mutations.",
            actions=[
                PlannedAction(
                    "auto_mutation_reconciliation_queue",
                    {},
                    reason="Explicit runtime alias for the content-free mutation reconciliation queue.",
                )
            ],
            needs_model=False,
            notes="runtime_auto_mutation_reconciliation_alias",
            metadata={
                "runtime_auto_mutation_reconciliation_alias": True,
                "reconciliation_operation": "list",
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
        )

    inspect_patterns = (
        r"^(?:uncertain mutation|auto mutation receipt|mutation receipt)\s+#?(?P<receipt_id>\d+)$",
        r"^불확실\s*실행\s*#?(?P<receipt_id>\d+)(?:\s*(?:확인|상태))?$",
    )
    for pattern in inspect_patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if not match:
            continue
        if not _runtime_exact_alias_tool_is_read_only("auto_mutation_reconciliation_status", registry):
            return None
        receipt_id = int(match.group("receipt_id"))
        if receipt_id <= 0:
            return None
        return Plan(
            goal="Inspect one uncertain local mutation.",
            actions=[
                PlannedAction(
                    "auto_mutation_reconciliation_status",
                    {"receipt_id": receipt_id},
                    reason="Explicit runtime alias for content-free mutation receipt inspection.",
                )
            ],
            needs_model=False,
            notes="runtime_auto_mutation_reconciliation_alias",
            metadata={
                "runtime_auto_mutation_reconciliation_alias": True,
                "reconciliation_operation": "inspect",
                "reconciliation_receipt_id": receipt_id,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
        )

    resolve_patterns = (
        r"^resolve\s+(?:uncertain\s+mutation|auto\s+mutation(?:\s+receipt)?|mutation\s+receipt)\s+#?(?P<receipt_id>\d+)\s+(?:as\s+)?(?P<disposition>applied|not[ -]applied)$",
        r"^불확실\s*실행\s*#?(?P<receipt_id>\d+)\s*(?P<disposition>적용됨|미적용)\s*(?:으로\s*)?(?:해결|확정)?$",
    )
    for pattern in resolve_patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            tool = registry.get("resolve_auto_mutation_receipt") if registry is not None else None
        except Exception:
            return None
        if getattr(getattr(tool, "risk", None), "name", "") != "LOCAL_SAFE":
            return None
        receipt_id = int(match.group("receipt_id"))
        if receipt_id <= 0:
            return None
        raw_disposition = match.group("disposition").lower().replace("-", " ")
        disposition = (
            "confirmed_not_applied"
            if raw_disposition in {"not applied", "미적용"}
            else "confirmed_applied"
        )
        return Plan(
            goal="Resolve one reviewed uncertain local mutation.",
            actions=[
                PlannedAction(
                    "resolve_auto_mutation_receipt",
                    {"receipt_id": receipt_id, "disposition": disposition},
                    reason="Explicit operator reconciliation command after content-free receipt inspection.",
                )
            ],
            needs_model=False,
            notes="runtime_auto_mutation_reconciliation_alias",
            metadata={
                "runtime_auto_mutation_reconciliation_alias": True,
                "reconciliation_operation": "resolve",
                "reconciliation_receipt_id": receipt_id,
                "reconciliation_disposition": disposition,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
        )
    return None


def _runtime_exact_tool_alias_plan(user_input: str, registry: Any | None = None) -> Plan | None:
    projection_repair_plan = _runtime_projection_repair_alias_plan(user_input, registry)
    if projection_repair_plan is not None:
        return projection_repair_plan
    reconciliation_plan = _runtime_auto_mutation_reconciliation_alias_plan(user_input, registry)
    if reconciliation_plan is not None:
        return reconciliation_plan
    goal_step_plan = _runtime_goal_step_status_alias_plan(user_input, registry)
    if goal_step_plan is not None:
        return goal_step_plan
    raw_input = _runtime_raw_command_text(user_input)
    tool_name = _runtime_registered_read_only_tool_name(raw_input, registry)
    if tool_name is None:
        mapped_raw_tool = _RUNTIME_EXACT_TOOL_ALIAS_RAW.get(raw_input)
        if mapped_raw_tool is not None and _runtime_exact_alias_tool_is_read_only(mapped_raw_tool, registry):
            tool_name = mapped_raw_tool
    if tool_name is None:
        tool_name = _RUNTIME_EXACT_TOOL_ALIASES.get(_runtime_normalize_command_text(user_input))
    if tool_name is None:
        tool_name = _RUNTIME_EXACT_TOOL_ALIAS_COMPACT.get(_runtime_compact_command_text(user_input))
    if tool_name is not None and not _runtime_exact_alias_tool_is_read_only(tool_name, registry):
        return None
    if tool_name is None:
        return None
    return Plan(
        goal="Run an exact read-only runtime command alias.",
        actions=[
            PlannedAction(
                tool_name,
                {},
                reason="Exact runtime alias for a read-only owner command.",
            )
        ],
        needs_model=False,
        notes="runtime_exact_tool_alias",
        metadata={
            "runtime_exact_tool_alias": True,
            "alias_input": _runtime_scrub_text(user_input),
            "alias_tool_name": tool_name,
        },
    )


def _runtime_metadata_bool(value: Any, default: bool = False) -> bool:
    if value is True:
        return True
    if value is False:
        return False
    return default


def _runtime_args_preview(args: dict[str, Any]) -> dict[str, str]:
    preview: dict[str, str] = {}
    for key, value in sorted((args or {}).items(), key=lambda item: str(item[0])):
        safe_key = _runtime_output_preview(key, limit=80)
        preview[safe_key] = (
            "bound to reviewed memory version"
            if _runtime_private_arg_key(safe_key)
            else _runtime_output_preview(value)
        )
    return preview


def _runtime_empty_planner_metadata() -> dict[str, Any]:
    return {
        "planner_type": None,
        "model_planner_attempted": False,
        "model_planner_state": None,
        "model_planner_used": False,
        "model_planner_fell_back": False,
        "model_planner_fallback_reason": None,
        "model_planner_fallback_detail": None,
        "model_planner_recovery_hint": None,
        "model_planner_exception_type": None,
        "model_planner_model": None,
        "model_planner_timeout_seconds": None,
        "model_planner_action_count": None,
        "model_planner_ignored_unknown_tools": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


class JarvisRuntime:
    def __init__(self, config: JarvisConfig | None = None):
        self.config = config or load_config()
        self.session_id = str(uuid.uuid4())[:8]
        self.history_session_generation = secrets.randbelow((1 << 63) - 1) + 1
        self._history_policy_epoch_id: str | None = None
        self._history_policy_fingerprint: str | None = None
        self._history_chat_epoch_id: str | None = None
        self._history_policy_store: MemoryStore | None = None
        self._history_policy_binding_token: str | None = None
        self._history_policy_status = "unbound"
        self._storage_read_only_startup = False
        self.subagent_fleet = spawn_subagent_fleet(count=10)
        self.vault = ObsidianVault(self.config.obsidian_vault, self.config.obsidian_root)
        self.storage_fallback: dict[str, Any] | None = None
        try:
            self.store = MemoryStore(self.config.db_path)
            self.store.init()
        except Exception as exc:
            if not self._is_storage_write_error(exc) or not self._activate_default_storage_fallback(exc):
                raise
        self._storage_read_only_startup = bool(self.store.read_only_startup)
        if not self._storage_read_only_startup:
            self._recover_stale_auto_mutations()
            self.subagent_fleet = spawn_subagent_fleet(
                count=10,
                store=self.store,
                runtime_id=self.session_id,
            )
            self._run_startup_projection_recovery_audit()
        else:
            read_only_status = {"status": "skipped_read_only_startup"}
            self.person_projection_recovery_status = dict(read_only_status)
            self.decision_projection_recovery_status = dict(read_only_status)
            self.preference_projection_recovery_status = dict(read_only_status)
            self.memory_projection_recovery_status = dict(read_only_status)
            self.skill_projection_recovery_status = dict(read_only_status)
            self.startup_recovery_audit_status = dict(read_only_status)

        self.registry = build_core_registry(
            self.store,
            self.vault,
            self.config,
            self.session_id,
            lambda: self.storage_fallback,
            lambda: self.subagent_fleet,
            lambda: getattr(self, "chat", None),
        )
        self.planner = (
            ModelBackedPlanner(
                self.config.planner_model,
                self.registry,
                timeout_seconds=self.config.model_timeout_seconds,
                provider=self.config.model_provider,
                reasoning_effort=self.config.planner_reasoning_effort,
                openai_max_output_tokens=self.config.openai_max_output_tokens,
            )
            if self.config.use_model_planner
            else RuleBasedPlanner()
        )
        self.executor = Executor(self.registry, PermissionPolicy())
        self.verifier = Verifier()
        self.chat = ChatBrain(
            self.config.chat_model,
            self.store,
            self.vault,
            model_timeout_seconds=self.config.chat_timeout_seconds,
            max_reply_tokens=self.config.chat_max_reply_tokens,
            max_history_messages=self.config.chat_max_history_messages,
            provider=self.config.model_provider,
            reasoning_effort=self.config.chat_reasoning_effort,
            openai_max_output_tokens=self.config.openai_max_output_tokens,
            allow_remote_personal_context=self.config.allow_remote_personal_context,
        )
        if getattr(self, "_storage_read_only_startup", False):
            self._history_policy_status = "read_only"
        else:
            self._activate_fresh_history_policy_binding()

    def _history_policy_values(self, decision: Any) -> dict[str, Any]:
        return {
            "policy_fingerprint": decision.fingerprint,
            "provider": decision.provider,
            "model_identifier": decision.model,
            "destination_class": decision.destination_class,
            "session_generation": self.history_session_generation,
            "explicit_consent_satisfied": bool(decision.stored_context_consent),
        }

    def _activate_fresh_history_policy_binding(self) -> bool:
        decision = self.chat._history_policy_decision(commit=False)
        try:
            epoch_id = self.store.start_history_policy_epoch(
                **self._history_policy_values(decision)
            )
        except Exception:
            self._history_policy_store = self.store
            self._history_policy_epoch_id = None
            self._history_policy_fingerprint = decision.fingerprint
            self._history_chat_epoch_id = decision.epoch_id
            self._history_policy_binding_token = None
            self._history_policy_status = "unavailable"
            self.chat.clear_history_disclosure_callbacks()
            return False
        self._history_policy_store = self.store
        self._history_policy_epoch_id = epoch_id
        self._history_policy_fingerprint = decision.fingerprint
        self._history_chat_epoch_id = decision.epoch_id
        self._history_policy_binding_token = secrets.token_hex(32)
        self._history_policy_status = "active"
        self.chat.install_history_disclosure_callbacks(
            prepare=self._prepare_history_disclosure,
            validate=self._validate_history_disclosure,
            finalize=self._finalize_history_disclosure,
            fence=self._history_disclosure_fence,
        )
        return True

    def _invalidate_history_policy_binding(self) -> None:
        old_chat = getattr(self, "chat", None)
        if old_chat is not None:
            old_chat.clear_history_disclosure_callbacks()
        old_store = self._history_policy_store
        old_epoch = self._history_policy_epoch_id
        if old_store is not None and old_epoch is not None:
            try:
                with old_store.history_epoch_mutation_fence():
                    with old_store.connect() as conn:
                        conn.execute("BEGIN IMMEDIATE")
                        active = conn.execute(
                        "SELECT active_epoch_id FROM active_history_policy_epoch "
                        "WHERE singleton_id = 1"
                    ).fetchone()
                        if active is not None and str(active["active_epoch_id"]) == old_epoch:
                            MemoryStore._activate_history_epoch_on_connection(
                                conn,
                                (
                                    secrets.token_hex(32),
                                    "runtime",
                                    "invalidated",
                                    "local-only",
                                    self.history_session_generation,
                                    False,
                                ),
                            )
            except Exception:
                pass
        self._history_policy_store = None
        self._history_policy_epoch_id = None
        self._history_policy_fingerprint = None
        self._history_chat_epoch_id = None
        self._history_policy_binding_token = None
        self._history_policy_status = "invalidated"

    def _refresh_history_policy_binding(self) -> bool:
        decision = self.chat._history_policy_decision(commit=False)
        if (
            decision.fingerprint == self._history_policy_fingerprint
            and decision.epoch_id == self._history_chat_epoch_id
        ):
            return self._active_history_policy() is not None
        store = self._history_policy_store
        owned_epoch = self._history_policy_epoch_id
        if store is None or store is not self.store or owned_epoch is None:
            self._history_policy_status = "stale"
            return False
        values = self._history_policy_values(decision)
        try:
            with store.history_epoch_mutation_fence():
                with store.connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    active = conn.execute(
                    "SELECT active_epoch_id FROM active_history_policy_epoch "
                    "WHERE singleton_id = 1"
                ).fetchone()
                    if active is None or str(active["active_epoch_id"]) != owned_epoch:
                        self._history_policy_status = "stale"
                        return False
                    epoch_id = MemoryStore._activate_history_epoch_on_connection(
                        conn,
                        (
                            values["policy_fingerprint"],
                            values["provider"],
                            values["model_identifier"],
                            values["destination_class"],
                            values["session_generation"],
                            values["explicit_consent_satisfied"],
                        ),
                    )
        except Exception:
            self._history_policy_status = "unavailable"
            return False
        self._history_policy_epoch_id = epoch_id
        self._history_policy_fingerprint = decision.fingerprint
        self._history_chat_epoch_id = decision.epoch_id
        self._history_policy_binding_token = secrets.token_hex(32)
        self._history_policy_status = "active"
        return True

    def _active_history_policy(self) -> dict[str, Any] | None:
        store = getattr(self, "_history_policy_store", None)
        epoch_id = getattr(self, "_history_policy_epoch_id", None)
        if store is None or store is not getattr(self, "store", None) or epoch_id is None:
            return None
        try:
            active = store.get_active_history_policy_epoch()
        except Exception:
            return None
        if (
            active.get("epoch_id") != epoch_id
            or active.get("policy_fingerprint") != self._history_policy_fingerprint
            or active.get("session_generation") != self.history_session_generation
        ):
            self._history_policy_status = "stale"
            return None
        return active

    def _prepare_history_disclosure(self, payload: dict[str, object]) -> object:
        if type(payload) is not dict or set(payload) != {
            "session_generation",
            "policy_epoch_id",
            "policy_fingerprint",
            "provider",
            "destination_class",
            "source_count",
            "source_digest",
        }:
            raise ValueError("history disclosure payload is malformed")
        decision = self.chat._history_policy_decision(commit=False)
        active = self._active_history_policy()
        if active is None:
            raise ValueError("history disclosure authority is unavailable")
        source_count = payload["source_count"]
        source_digest = payload["source_digest"]
        if any(
            (
                type(payload["session_generation"]) is not str,
                payload["session_generation"] != self.chat._session_generation,
                payload["policy_epoch_id"] != self._history_chat_epoch_id,
                payload["policy_epoch_id"] != decision.epoch_id,
                payload["policy_fingerprint"] != decision.fingerprint,
                payload["policy_fingerprint"] != active["policy_fingerprint"],
                payload["provider"] != decision.provider,
                payload["provider"] != active["provider"],
                payload["destination_class"] != decision.destination_class,
                payload["destination_class"] != active["destination_class"],
                active["model_identifier"] != decision.model,
                bool(active["explicit_consent_satisfied"])
                != bool(decision.stored_context_consent),
                type(source_count) is not int,
                type(source_digest) is not str,
            )
        ):
            raise ValueError("history disclosure authority is unavailable")
        if source_count < 1 or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
            raise ValueError("history disclosure authority is unavailable")
        receipt_id = self.store.prepare_history_disclosure_receipt(
            session_id=self.session_id,
            provider=str(payload["provider"]),
            destination_class=str(payload["destination_class"]),
            source_count=source_count,
            source_digest=source_digest,
            policy_epoch_id=str(active["epoch_id"]),
        )
        return (
            self._history_policy_binding_token,
            self.store,
            str(active["epoch_id"]),
            receipt_id,
        )

    def _validate_history_disclosure(self, handle: object) -> bool:
        if type(handle) is not tuple or len(handle) != 4:
            return False
        binding_token, store, epoch_id, receipt_id = handle
        if (
            type(binding_token) is not str
            or binding_token != self._history_policy_binding_token
            or store is not self._history_policy_store
            or store is not self.store
            or type(epoch_id) is not str
            or epoch_id != self._history_policy_epoch_id
            or type(receipt_id) is not str
        ):
            return False
        active = self._active_history_policy()
        return bool(active is not None and active.get("epoch_id") == epoch_id)

    @contextmanager
    def _history_disclosure_fence(self, handle: object) -> Generator[None, None, None]:
        if type(handle) is not tuple or len(handle) != 4:
            raise ValueError("history disclosure fence handle is malformed")
        binding_token, store, epoch_id, receipt_id = handle
        if (
            type(binding_token) is not str
            or binding_token != self._history_policy_binding_token
            or store is not self._history_policy_store
            or store is not self.store
            or type(epoch_id) is not str
            or epoch_id != self._history_policy_epoch_id
            or type(receipt_id) is not str
        ):
            raise PermissionError("history disclosure fence authority is unavailable")
        receipt = store.get_history_disclosure_receipt(receipt_id)
        decision = self.chat._history_policy_decision(commit=False)
        if receipt is None:
            raise PermissionError("history disclosure receipt is unavailable")
        with store.history_disclosure_egress_fence(
            receipt_id=receipt_id,
            policy_epoch_id=epoch_id,
            provider=decision.provider,
            model_identifier=decision.model,
            destination_class=decision.destination_class,
            source_count=int(receipt["source_count"]),
            source_digest=str(receipt["source_digest"]),
        ):
            self.chat._revalidate_history_policy(decision)
            if not self._validate_history_disclosure(handle):
                raise PermissionError("history disclosure fence authority changed")
            yield

    def _finalize_history_disclosure(self, handle: object, outcome: str) -> str:
        if (
            type(handle) is not tuple
            or len(handle) != 4
            or outcome not in {"confirmed", "blocked", "uncertain"}
        ):
            raise ValueError("history disclosure finalization is malformed")
        binding_token, store, epoch_id, receipt_id = handle
        if (
            type(receipt_id) is not str
            or re.fullmatch(r"[0-9a-f]{64}", receipt_id) is None
            or type(epoch_id) is not str
            or re.fullmatch(r"[0-9a-f]{64}", epoch_id) is None
        ):
            raise ValueError("history disclosure finalization is malformed")
        local_binding_matches = bool(
            type(binding_token) is str
            and binding_token == self._history_policy_binding_token
            and store is self._history_policy_store
            and store is self.store
            and epoch_id == self._history_policy_epoch_id
        )
        final_outcome = (
            "uncertain" if outcome == "confirmed" and not local_binding_matches else outcome
        )
        receipt = store.get_history_disclosure_receipt(receipt_id)
        if receipt is None or receipt["policy_epoch_id"] != epoch_id:
            raise ValueError("history disclosure receipt does not match its handle")
        finalized = store.finalize_history_disclosure_receipt(receipt_id, final_outcome)
        return str(finalized["state"])

    def _strict_message_provenance(
        self,
        value: object,
        *,
        role: str,
        active: dict[str, Any],
    ) -> dict[str, Any] | None:
        if type(value) is not dict or set(value) != _HISTORY_PROVENANCE_FIELDS:
            return None
        if value.get("role") != role:
            return None
        lineage_value = value.get("lineage_state")
        lineage = (
            _HISTORY_PROVENANCE_LINEAGE.get(lineage_value)
            if type(lineage_value) is str
            else None
        )
        count = value.get("source_count")
        digest = value.get("source_digest")
        if (
            lineage is None
            or type(count) is not int
            or count < 0
            or type(value.get("future_history_allowed")) is not bool
            or type(value.get("lineage_token_digest")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value["lineage_token_digest"]) is None
            or value.get("policy_epoch_id") != self._history_chat_epoch_id
            or value.get("policy_fingerprint") != active["policy_fingerprint"]
            or value.get("provider") != active["provider"]
            or value.get("destination_class") != active["destination_class"]
        ):
            return None
        if lineage == "direct_current":
            if count != 0 or digest is not None:
                return None
        elif (
            count < 1
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            return None
        remote_eligible = bool(
            value["future_history_allowed"]
            and active["epoch_id"] == self._history_policy_epoch_id
            and active["explicit_consent_satisfied"]
            and lineage not in {"tool_derived", "mixed_restricted"}
        )
        return {
            "role": role,
            "policy_epoch_id": active["epoch_id"],
            "lineage_state": lineage,
            "source_count": count,
            "source_digest": digest,
            "destination_class": active["destination_class"],
            "remote_eligible": remote_eligible,
        }

    def _runtime_message_provenance(
        self, metadata: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        chat_response = metadata.get("chat_response")
        if type(chat_response) is not dict or chat_response.get("runtime_route") == "command_suggestion":
            return None
        provenance = chat_response.get("history_message_provenance")
        if type(provenance) is not dict or set(provenance) != {"user", "assistant"}:
            return None
        active = self._active_history_policy()
        if active is None:
            return None
        user = self._strict_message_provenance(provenance["user"], role="user", active=active)
        assistant = self._strict_message_provenance(
            provenance["assistant"], role="assistant", active=active
        )
        if user is None or assistant is None:
            return None
        return user, assistant

    @staticmethod
    def _mark_history_provenance_warning(metadata: dict[str, Any], phase: str) -> None:
        warning = {
            "status": "degraded",
            "phase": phase[:32],
            "local_only": True,
            "exception_detail_exposed": False,
        }
        metadata["history_provenance_warning"] = warning
        trace = metadata.get("runtime_trace")
        if isinstance(trace, dict):
            trace["history_provenance_warning"] = dict(warning)

    def _reconcile_pending_skill_projections_at_startup(self) -> None:
        try:
            summary = reconcile_pending_skill_projections(self.store, self.vault, limit=20)
        except Exception as exc:
            self.skill_projection_recovery_status = {
                "status": "failed",
                "exception_type": _startup_recovery_exception_type(exc),
            }
            return
        self.skill_projection_recovery_status = {
            "status": "completed" if summary.pending == 0 else "pending",
            "attempted": summary.attempted,
            "completed": summary.completed,
            "pending": summary.pending,
        }

    def _reconcile_pending_person_projections_at_startup(self) -> None:
        try:
            summary = reconcile_pending_person_projections(self.store, self.vault, limit=20)
        except Exception as exc:
            self.person_projection_recovery_status = {
                "status": "failed",
                "exception_type": _startup_recovery_exception_type(exc),
            }
            return
        self.person_projection_recovery_status = {
            "status": "completed" if summary.pending == 0 else "pending",
            "attempted": summary.attempted,
            "completed": summary.completed,
            "pending": summary.pending,
        }

    def _reconcile_pending_decision_projections_at_startup(self) -> None:
        try:
            summary = reconcile_pending_decision_projections(
                self.store,
                self.vault,
                limit=20,
            )
        except Exception as exc:
            self.decision_projection_recovery_status = {
                "status": "failed",
                "exception_type": _startup_recovery_exception_type(exc),
            }
            return
        self.decision_projection_recovery_status = {
            "status": "completed" if summary.pending == 0 else "pending",
            "attempted": summary.attempted,
            "completed": summary.completed,
            "pending": summary.pending,
            "custody_backfilled": summary.custody_backfilled,
        }

    def _reconcile_pending_preference_projections_at_startup(self) -> None:
        try:
            summary = reconcile_pending_preference_projections(self.store, self.vault)
        except Exception as exc:
            self.preference_projection_recovery_status = {
                "status": "failed",
                "exception_type": _startup_recovery_exception_type(exc),
            }
            return
        self.preference_projection_recovery_status = {
            "status": (
                "completed"
                if summary.pending == 0 and summary.custody_conflicts == 0
                else "pending"
            ),
            "attempted": summary.attempted,
            "completed": summary.completed,
            "pending": summary.pending,
            "custody_backfilled": summary.custody_backfilled,
            "custody_conflicts": summary.custody_conflicts,
        }

    def _reconcile_pending_memory_projections_at_startup(self) -> None:
        try:
            profile_summary = reconcile_pending_profile_projections(
                self.store,
                self.vault,
                limit=20,
            )
            goal_summary = reconcile_pending_goal_projections(
                self.store,
                self.vault,
                limit=20,
            )
            summary = reconcile_pending_memory_projections(self.store, self.vault, limit=20)
        except Exception as exc:
            self.memory_projection_recovery_status = {
                "status": "failed",
                "exception_type": _startup_recovery_exception_type(exc),
            }
            return
        attempted = profile_summary.attempted + goal_summary.attempted + summary.attempted
        completed = profile_summary.completed + goal_summary.completed + summary.completed
        pending = profile_summary.pending + goal_summary.pending + summary.pending
        self.memory_projection_recovery_status = {
            "status": "completed" if pending == 0 else "pending",
            "attempted": attempted,
            "completed": completed,
            "pending": pending,
        }

    def _run_startup_projection_recovery_audit(self) -> None:
        try:
            run_id = self.store.begin_startup_recovery_run(self.session_id)
        except Exception as exc:
            self.startup_recovery_audit_status = {
                "status": "failed",
                "exception_type": _startup_recovery_exception_type(exc),
            }
            raise StartupRecoveryUnavailable() from None

        self.startup_recovery_audit_status = {
            "status": "running",
            "run_id": run_id,
        }
        components = (
            ("person", self._reconcile_pending_person_projections_at_startup, "person_projection_recovery_status"),
            ("decision", self._reconcile_pending_decision_projections_at_startup, "decision_projection_recovery_status"),
            ("preference", self._reconcile_pending_preference_projections_at_startup, "preference_projection_recovery_status"),
            ("memory", self._reconcile_pending_memory_projections_at_startup, "memory_projection_recovery_status"),
            ("skill", self._reconcile_pending_skill_projections_at_startup, "skill_projection_recovery_status"),
        )
        heartbeat_stop = threading.Event()
        heartbeat_failed = threading.Event()

        def renew_audit_lease() -> None:
            while not heartbeat_stop.is_set():
                try:
                    if not self.store.heartbeat_startup_recovery_run(run_id):
                        heartbeat_failed.set()
                        return
                except Exception:
                    heartbeat_failed.set()
                    return
                if heartbeat_stop.wait(STARTUP_RECOVERY_HEARTBEAT_SECONDS):
                    return

        heartbeat_thread = threading.Thread(
            target=renew_audit_lease,
            name="jarvis-startup-recovery-heartbeat",
            daemon=True,
        )
        heartbeat_started = False
        try:
            heartbeat_thread.start()
            heartbeat_started = True
            self.vault.init()
            for component, recover, status_attribute in components:
                if heartbeat_failed.is_set():
                    raise RuntimeError("Startup projection recovery audit lease was lost.")
                recover()
                if heartbeat_failed.is_set():
                    raise RuntimeError("Startup projection recovery audit lease was lost.")
                summary = getattr(self, status_attribute)
                durable_summary = dict(summary)
                if durable_summary.get("status") == "pending":
                    durable_summary["status"] = "partial"
                if not self.store.record_startup_recovery_component(
                    run_id,
                    component,
                    durable_summary,
                ):
                    raise RuntimeError(
                        f"Startup projection recovery audit lost {component} custody."
                    )
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=STARTUP_RECOVERY_HEARTBEAT_JOIN_SECONDS)
            if heartbeat_thread.is_alive() or heartbeat_failed.is_set():
                raise RuntimeError("Startup projection recovery audit lease was lost.")
            final_state = self.store.finalize_startup_recovery_run(run_id)
        except Exception as exc:
            try:
                self.store.fail_startup_recovery_run(
                    run_id,
                    _startup_recovery_exception_type(exc),
                )
            except Exception:
                pass
            self.startup_recovery_audit_status = {
                "status": "failed",
                "run_id": run_id,
                "exception_type": _startup_recovery_exception_type(exc),
            }
            raise StartupRecoveryUnavailable() from None
        finally:
            heartbeat_stop.set()
            if heartbeat_started:
                heartbeat_thread.join(timeout=STARTUP_RECOVERY_HEARTBEAT_JOIN_SECONDS)
        self.startup_recovery_audit_status = {
            "status": final_state,
            "run_id": run_id,
        }

    def handle(
        self,
        user_input: str,
        approved: bool = False,
        approved_approval_id: int | None = None,
        request_token: str | None = None,
    ) -> RuntimeResult:
        public_user_input = _runtime_sensitive_command_display(user_input)
        if getattr(self, "_storage_read_only_startup", False):
            return self._storage_degraded_result(
                public_user_input,
                PermissionError("read-only storage startup"),
            )
        storage_fallback_note = self._storage_fallback_note() if self.storage_fallback else ""
        try:
            user_message_id = self.store.log_message(
                self.session_id, "user", public_user_input
            )
        except Exception as exc:
            if self._is_storage_write_error(exc):
                if self._activate_default_storage_fallback(exc):
                    user_message_id = self.store.log_message(
                        self.session_id, "user", public_user_input
                    )
                    storage_fallback_note = self._storage_fallback_note()
                    exc = None
                else:
                    return self._storage_degraded_result(public_user_input, exc)
            if exc is not None:
                raise
        self._recover_stale_auto_mutations()
        approval_queue_before = len(self.store.list_pending_approvals(limit=100))

        approval_binding_error = ""
        approval_execution_claim: ApprovalExecutionClaim | None = None
        if approved:
            exact_alias_plan, approval_binding_error, approval_execution_claim = self._approved_rerun_plan(
                public_user_input,
                approved_approval_id,
            )
        else:
            exact_alias_plan = _runtime_exact_tool_alias_plan(user_input, self.registry)

        pre_planner_suggestion = None
        if not approval_binding_error and exact_alias_plan is None:
            pre_planner_suggestion = _runtime_read_only_tool_required_arg_suggestion(user_input, self.registry)
        if pre_planner_suggestion is None and exact_alias_plan is None:
            pre_planner_suggestion = _runtime_risky_tool_required_arg_suggestion(user_input, self.registry)
        if pre_planner_suggestion is None and exact_alias_plan is None and _runtime_should_pre_planner_suggest(user_input):
            try:
                pre_planner_suggestion = suggest_command(user_input)
            except Exception:
                pre_planner_suggestion = None

        if approval_binding_error:
            plan = exact_alias_plan
            results = []
            verified = False
            verification = approval_binding_error
            response = f"Approved action did not run. {approval_binding_error}"
            approval_queue_after = len(self.store.list_pending_approvals(limit=100))
            metadata = {
                "runtime_route": "tools",
                "chat_response": {},
                "runtime_trace": self._build_runtime_trace(
                    user_input=public_user_input,
                    route="tools",
                    plan=plan,
                    results=results,
                    verified=verified,
                    verification=verification,
                    chat_response={},
                    approved=approved,
                    approved_approval_id=approved_approval_id,
                    approval_queue_before=approval_queue_before,
                    approval_queue_after=approval_queue_after,
                ),
            }
        elif pre_planner_suggestion is not None:
            plan = Plan(
                goal="Offer a read-only command suggestion before planner routing.",
                actions=[],
                needs_model=True,
                notes="pre_planner_command_suggestion",
                metadata={
                    "pre_planner_command_suggestion": True,
                    "authorizes_execution": False,
                    "authorizes_completion_claim": False,
                    "approval_granted": False,
                },
            )
            results = []
            verified = True
            response = pre_planner_suggestion
            chat_response = {"runtime_route": "command_suggestion", "pre_planner_command_suggestion": True}
            metadata = {
                "runtime_route": "chat",
                "chat_response": chat_response,
                "runtime_trace": self._build_runtime_trace(
                    user_input=public_user_input,
                    route="chat",
                    plan=plan,
                    results=results,
                    verified=verified,
                    verification="pre-planner command suggestion generated",
                    chat_response=chat_response,
                    approved=approved,
                    approved_approval_id=approved_approval_id,
                    approval_queue_before=approval_queue_before,
                    approval_queue_after=len(self.store.list_pending_approvals(limit=100)),
                ),
            }
        else:
            plan = exact_alias_plan if exact_alias_plan is not None else self.planner.plan(user_input)
            if plan.needs_model and not plan.actions:
                results = []
                verified = True
                # Before the generic chat model (which can wrongly deny capabilities
                # Jarvis actually has), check whether this is a typo / near-miss for a
                # real command and offer a "did you mean?" redirect instead. Never let
                # a suggester hiccup break a normal conversational turn.
                try:
                    suggestion = suggest_command(user_input)
                except Exception:
                    suggestion = None
                if suggestion is not None:
                    response = suggestion
                    chat_response = {"runtime_route": "command_suggestion"}
                else:
                    chat_started = time.perf_counter()
                    self._refresh_history_policy_binding()
                    response = self.chat.respond(user_input)
                    chat_response = dict(self.chat.last_turn_metadata)
                    chat_latency_ms = round((time.perf_counter() - chat_started) * 1000, 1)
                    chat_response["latency_ms"] = chat_latency_ms
                    chat_response["duration_ms"] = chat_latency_ms
                    chat_response["latency_recorded"] = True
                metadata = {
                    "runtime_route": "chat",
                    "chat_response": chat_response,
                    "runtime_trace": self._build_runtime_trace(
                        user_input=public_user_input,
                        route="chat",
                        plan=plan,
                        results=results,
                        verified=verified,
                        verification="chat response generated",
                        chat_response=chat_response,
                        approved=approved,
                        approved_approval_id=approved_approval_id,
                        approval_queue_before=approval_queue_before,
                        approval_queue_after=len(self.store.list_pending_approvals(limit=100)),
                    ),
                }
            else:
                if approved:
                    results = [self.executor.execute(action, approved=True) for action in plan.actions]
                    auto_mutation_finalized_indexes: set[int] = set()
                else:
                    results, auto_mutation_finalized_indexes = self._execute_plan_with_auto_mutation_receipts(
                        plan,
                        request_token=request_token,
                        user_message_id=user_message_id,
                    )
                if approved and approved_approval_id is not None:
                    for result in results:
                        result.metadata["approved_approval_id"] = approved_approval_id
                        result.metadata["approval_rerun_exact"] = True
                        result.metadata["rerun_tool_name"] = result.tool_name
                        result.metadata["rerun_arg_keys"] = sorted(plan.actions[0].args)
                storage_warnings = []
                approved_execution_outcomes: list[ApprovedExecutionOutcome] = []
                if approval_execution_claim is not None:
                    if len(results) != 1 or len(plan.actions) != 1 or approved_approval_id is None:
                        raise RuntimeError("approved rerun did not resolve to exactly one stored action")
                    classified, approval_warnings = self._finalize_approved_execution(
                        approval_execution_claim,
                        plan.actions[0],
                        results[0],
                    )
                    approved_execution_outcomes.append(classified)
                    storage_warnings.extend(approval_warnings)
                else:
                    try:
                        self._log_tool_runs(
                            [
                                result
                                for index, result in enumerate(results)
                                if index not in auto_mutation_finalized_indexes
                            ],
                            approved,
                            approved_approval_id,
                        )
                    except Exception as exc:
                        if not self._is_storage_write_error(exc):
                            raise
                        storage_warnings.append(self._storage_warning("tool-run audit logging", exc))
                        for result in results:
                            result.metadata["audit_logging_failed"] = True
                            result.metadata["storage_degraded"] = True
                            result.metadata["storage_error"] = type(exc).__name__
                try:
                    self._log_pending_approvals(public_user_input, results)
                except Exception as exc:
                    if not self._is_storage_write_error(exc):
                        raise
                    storage_warnings.append(self._storage_warning("approval queue logging", exc))
                    for result in results:
                        if result.metadata.get("requires_confirmation"):
                            result.metadata["approval_queue_failed"] = True
                            result.metadata["storage_degraded"] = True
                            result.metadata["storage_error"] = type(exc).__name__
                verified, verification = self.verifier.verify(plan, results)
                response = self._compose_response(results, verified, verification)
                if any(outcome.outcome_unknown for outcome in approved_execution_outcomes):
                    verified = False
                    verification = "Approved action outcome is unknown; verify the target state before retrying."
                    response = (
                        "The approved action was attempted, but its outcome is unknown. Do not retry it "
                        "automatically; verify the target state first.\n\n"
                        + self._compose_response(results, False, verification)
                    )
                if storage_warnings:
                    verified = False
                    response = response + "\n\n" + "\n".join(storage_warnings)
                rerun_response, approved_rerun_results = self._rerun_approved_request(plan, results)
                if rerun_response:
                    unknown_reruns = [
                        result
                        for result in approved_rerun_results
                        if result.metadata.get("approved_execution_outcome")
                        == APPROVED_EXECUTION_OUTCOME_UNKNOWN
                    ]
                    failed_reruns = [result for result in approved_rerun_results if not result.ok]
                    audit_failed_reruns = [
                        result
                        for result in approved_rerun_results
                        if result.metadata.get("audit_logging_failed")
                        or result.metadata.get("approval_execution_finalization_failed")
                    ]
                    if unknown_reruns:
                        verified = False
                        verification = "Approved action outcome is unknown; verify the target state before retrying."
                        response = (
                            "Approval was recorded and the action was attempted, but its outcome is unknown. "
                            "Do not retry it automatically; verify the target state first.\n\n"
                            + rerun_response
                        )
                    elif failed_reruns:
                        verified = False
                        failure_detail = "; ".join(
                            f"{result.tool_name}: "
                            f"{_runtime_output_preview(_runtime_persisted_tool_output(result))}"
                            for result in failed_reruns
                        )
                        verification = f"Approved action did not complete: {failure_detail}"
                        response = "Approval was recorded, but the approved action did not complete.\n\n" + rerun_response
                    elif audit_failed_reruns:
                        verified = False
                        verification = "Approved action ran, but durable audit logging did not complete."
                        response = (
                            "Approval was recorded and the action ran, but its audit record did not complete.\n\n"
                            + rerun_response
                        )
                    else:
                        response = response + "\n\n" + rerun_response
                approval_queue_after = len(self.store.list_pending_approvals(limit=100))
                metadata = {
                    "runtime_route": "tools",
                    "chat_response": {},
                    "runtime_trace": self._build_runtime_trace(
                        user_input=public_user_input,
                        route="tools",
                        plan=plan,
                        results=results,
                        verified=verified,
                        verification=verification,
                        chat_response={},
                        approved=approved,
                        approved_approval_id=approved_approval_id,
                        approval_queue_before=approval_queue_before,
                        approval_queue_after=approval_queue_after,
                    ),
                }

        message_provenance = self._runtime_message_provenance(metadata)
        chat_brain = getattr(self, "chat", None)
        if (
            chat_brain is not None
            and metadata.get("runtime_route") == "tools"
            and verified
            and not approved
            and results
            and all(result.ok and not result.metadata.get("requires_confirmation") for result in results)
            and all(
                self.registry.get(action.tool_name).risk <= RiskLevel.PERSONAL_DATA
                for action in plan.actions
            )
            and not any(
                result.metadata.get(key)
                for result in results
                for key in (
                    "external_side_effect",
                    "writes_files",
                    "writes_memory",
                    "writes_notes",
                    "controls_computer",
                    "sends_message",
                    "places_call",
                    "changes_state",
                )
            )
            and not any(
                result.metadata.get("suppress_output_persistence") is True
                for result in results
            )
        ):
            chat_brain.record_tool_turn(
                user_input,
                response,
                tuple(action.tool_name for action in plan.actions),
            )
        assistant_provenance: dict[str, Any] | None = None
        if message_provenance is not None:
            user_provenance, assistant_provenance = message_provenance
            try:
                self.store.attach_message_provenance(
                    user_message_id,
                    role="user",
                    policy_epoch_id=user_provenance["policy_epoch_id"],
                    lineage_state=user_provenance["lineage_state"],
                    source_count=user_provenance["source_count"],
                    source_digest=user_provenance["source_digest"],
                    destination_class=user_provenance["destination_class"],
                    remote_eligible=user_provenance["remote_eligible"],
                )
            except Exception:
                assistant_provenance["remote_eligible"] = False
                self._mark_history_provenance_warning(metadata, "user_attachment")

        try:
            if storage_fallback_note:
                response = response + "\n\n" + storage_fallback_note
                runtime_trace = metadata.get("runtime_trace")
                if isinstance(runtime_trace, dict):
                    runtime_trace["storage_fallback_active"] = True
                    runtime_trace["storage_fallback_db_path"] = self.storage_fallback.get("db_path") if self.storage_fallback else ""
                    runtime_trace["storage_fallback_db_path_display"] = self.storage_fallback.get("db_path_display") if self.storage_fallback else ""
                    runtime_trace["storage_fallback_reason"] = self.storage_fallback.get("reason") if self.storage_fallback else ""
                    runtime_trace["storage_fallback_exception_type"] = self.storage_fallback.get("exception_type") if self.storage_fallback else ""
                    runtime_trace["storage_fallback_vault_path"] = self.storage_fallback.get("vault_path") if self.storage_fallback else ""
                    runtime_trace["storage_fallback_vault_path_display"] = self.storage_fallback.get("vault_path_display") if self.storage_fallback else ""
                metadata = {
                    **metadata,
                    "storage_fallback_active": True,
                    "storage_fallback_db_path": self.storage_fallback.get("db_path") if self.storage_fallback else "",
                    "storage_fallback_db_path_display": self.storage_fallback.get("db_path_display") if self.storage_fallback else "",
                    "storage_fallback_reason": self.storage_fallback.get("reason") if self.storage_fallback else "",
                    "storage_fallback_exception_type": self.storage_fallback.get("exception_type") if self.storage_fallback else "",
                    "storage_fallback_vault_path": self.storage_fallback.get("vault_path") if self.storage_fallback else "",
                    "storage_fallback_vault_path_display": self.storage_fallback.get("vault_path_display") if self.storage_fallback else "",
                }
            persisted_response = _runtime_persisted_response(results, response)
            self.store.log_message(
                self.session_id,
                "assistant",
                persisted_response,
                metadata,
                provenance=assistant_provenance,
            )
        except Exception as exc:
            if assistant_provenance is not None and isinstance(
                exc, (TypeError, ValueError, RuntimeError, sqlite3.IntegrityError)
            ):
                self._mark_history_provenance_warning(metadata, "assistant_atomic_log")
                try:
                    self.store.log_message(
                        self.session_id,
                        "assistant",
                        _runtime_persisted_response(results, response),
                        metadata,
                    )
                except Exception as retry_exc:
                    exc = retry_exc
                else:
                    exc = None
            if exc is None:
                return RuntimeResult(
                    public_user_input,
                    _runtime_public_plan(plan, results),
                    results,
                    verified,
                    response,
                    metadata,
                )
            if not self._is_storage_write_error(exc):
                raise
            warning = self._storage_warning("assistant message logging", exc)
            response = response + "\n\n" + warning
            metadata = {
                **metadata,
                "storage_degraded": True,
                "storage_error": type(exc).__name__,
                "storage_warning": warning,
            }
            runtime_trace = metadata.get("runtime_trace")
            if isinstance(runtime_trace, dict):
                runtime_trace["storage_degraded"] = True
                runtime_trace["storage_error"] = type(exc).__name__
                runtime_trace["storage_warning"] = warning
        return RuntimeResult(
            public_user_input,
            _runtime_public_plan(plan, results),
            results,
            verified,
            response,
            metadata,
        )

    def _recover_stale_auto_mutations(self) -> None:
        store = getattr(self, "store", None)
        if store is None:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=AUTO_MUTATION_STALE_RUNNING_SECONDS)
        running_before = cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        try:
            store.cleanup_stale_prepared_auto_mutation_receipts(
                running_before,
                limit=100,
            )
        except Exception:
            pass
        try:
            store.recover_stale_auto_mutation_receipts(running_before, limit=100)
        except Exception:
            pass

    def _private_auto_mutation_request_token(
        self,
        request_token: str | None,
        user_message_id: int,
    ) -> str:
        if request_token is None:
            return f"runtime-turn:{self.session_id}:{user_message_id}"
        if not isinstance(request_token, str) or not request_token or len(request_token) > MAX_AUTO_MUTATION_REQUEST_TOKEN_CHARS:
            raise ValueError("invalid auto-mutation request token")
        return request_token

    def _auto_mutation_action_is_eligible(self, action: PlannedAction) -> bool:
        try:
            return self.registry.get(action.tool_name).auto_mutation_contract is not None
        except Exception:
            return False

    @staticmethod
    def _auto_mutation_blocked_result(
        action: PlannedAction,
        *,
        failure_kind: str,
        output: str,
        handler_invoked: bool = False,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        metadata: dict[str, Any] = {
            "failure_kind": failure_kind,
            "requires_confirmation": False,
            "executed_handler": handler_invoked,
            "handler_invoked": handler_invoked,
            "planned_arg_keys": sorted(str(key)[:80] for key in action.args),
            "authorizes_retry": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return ToolResult(action.tool_name, False, output, metadata)

    def _auto_mutation_plan_blocked_results(
        self,
        plan: Plan,
        preparation_status: str,
    ) -> list[ToolResult]:
        if preparation_status == "REQUEST_TOKEN_COLLISION":
            failure_kind = "auto_mutation_request_collision"
            output = (
                "This request key is already bound to a different mutation plan. Nothing ran. "
                "Use a new request key only for a genuinely new request."
            )
        elif preparation_status == "DUPLICATE_OPERATION_DIGEST":
            failure_kind = "auto_mutation_duplicate_operation"
            output = (
                "This mutation plan contains the same logical operation more than once. "
                "Nothing ran and no mutation receipts were created."
            )
        elif preparation_status == "UNRESOLVED_ACTION_BLOCKED":
            failure_kind = "auto_mutation_unresolved_action"
            output = (
                "A matching mutation from another request is still unresolved. Nothing in this plan ran. "
                "Review the target state and recent tool runs before issuing a new request."
            )
        elif preparation_status == "INVALID_REQUEST_TOKEN":
            failure_kind = "auto_mutation_request_key_invalid"
            output = "The request key was invalid or too long. Nothing in this plan ran."
        elif preparation_status == "POLICY_BLOCKED":
            failure_kind = "auto_mutation_policy_blocked"
            output = "Current policy does not allow this local mutation. Nothing in this plan ran."
        elif preparation_status == "EXECUTION_BINDING_FAILED":
            failure_kind = "auto_mutation_execution_binding_failed"
            output = (
                "Jarvis could not bind this mutation to its reviewed source state. "
                "Nothing in this plan ran and no mutation receipt was created."
            )
        else:
            failure_kind = "auto_mutation_prepare_failed"
            output = (
                "Jarvis could not durably prepare this mutation plan, so nothing ran. "
                "Check storage health before issuing a new request."
            )
        return [
            self._auto_mutation_blocked_result(
                action,
                failure_kind=failure_kind,
                output=output,
            )
            for action in plan.actions
        ]

    def _auto_mutation_nonclaim_result(self, action: PlannedAction, status: str) -> ToolResult:
        if status == "RECONCILED_APPLIED":
            return self._auto_mutation_blocked_result(
                action,
                failure_kind="auto_mutation_reconciled_applied",
                output=(
                    "This uncertain mutation was explicitly reconciled as already applied. "
                    "The original request was not run again. Use a new request only for a genuinely new mutation."
                ),
                extra_metadata={"reconciliation_disposition": "confirmed_applied"},
            )
        if status == "RECONCILED_NOT_APPLIED":
            return self._auto_mutation_blocked_result(
                action,
                failure_kind="auto_mutation_reconciled_not_applied",
                output=(
                    "This uncertain mutation was explicitly reconciled as not applied. "
                    "The original request was not rerun; issue a new request if the mutation is still wanted."
                ),
                extra_metadata={"reconciliation_disposition": "confirmed_not_applied"},
            )
        if status == "COMPLETED_COALESCED":
            return self._auto_mutation_blocked_result(
                action,
                failure_kind="auto_mutation_completed_replay",
                output=(
                    "This mutation already completed for this request and was not run again. "
                    "Use a new request key only when you intentionally want a new mutation."
                ),
            )
        if status == "RUNNING_COALESCED":
            return self._auto_mutation_blocked_result(
                action,
                failure_kind="auto_mutation_running_replay",
                output=(
                    "This mutation is already running or awaiting durable completion for this request. "
                    "It was not run again; review the target state before taking further action."
                ),
            )
        if status == "UNCERTAIN_BLOCKED":
            return self._auto_mutation_blocked_result(
                action,
                failure_kind="auto_mutation_outcome_uncertain",
                output=(
                    "The outcome of this mutation is uncertain, so it was not run again. "
                    "Review the target state and recent tool runs before issuing a new request."
                ),
            )
        return self._auto_mutation_blocked_result(
            action,
            failure_kind="auto_mutation_claim_failed",
            output=(
                "Jarvis could not durably claim this mutation, so it did not run. "
                "Check storage health and the target state before issuing a new request."
            ),
        )

    def _mark_auto_mutation_uncertain(self, receipt_id: int, run_token: str) -> None:
        try:
            self.store.mark_auto_mutation_uncertain(receipt_id, run_token)
        except Exception:
            pass

    @staticmethod
    def _auto_mutation_failure_is_definite_no_effect(
        contract: Any,
        result: ToolResult,
    ) -> bool:
        reason = result.metadata.get("reason")
        allowed = getattr(contract, "definite_no_effect_failure_reasons", frozenset())
        if type(reason) is not str or reason not in allowed:
            return False
        if result.metadata.get("auto_mutation_effects_started") is not False:
            return False
        return all(
            result.metadata.get(key) is not True
            for key in (
                "state_changed",
                "writes_files",
                "writes_database",
                "writes_memory",
                "writes_notes",
                "external_side_effect",
                "controls_computer",
            )
        )

    def _auto_mutation_uncertain_result(
        self,
        action: PlannedAction,
        result: ToolResult,
    ) -> ToolResult:
        failure_kind = str(result.metadata.get("failure_kind") or "auto_mutation_handler_failed")[:80]
        metadata: dict[str, Any] = {
            "failure_kind": failure_kind,
            "requires_confirmation": False,
            "executed_handler": True,
            "handler_invoked": True,
            "planned_arg_keys": sorted(str(key)[:80] for key in action.args),
            "authorizes_retry": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "auto_mutation_outcome_uncertain": True,
        }
        reason = result.metadata.get("reason")
        if isinstance(reason, str) and re.fullmatch(r"[A-Za-z0-9_:-]{1,80}", reason):
            metadata["reason"] = reason
        for key in ("error_type", "exception_type", "risk_level", "risk_value", "toolset"):
            value = result.metadata.get(key)
            if value is not None:
                metadata[key] = value
        return ToolResult(
            action.tool_name,
            False,
            (
                "The local mutation handler did not produce a verified success after it started. "
                "Its outcome is uncertain and it will not be retried automatically; review the target state."
            ),
            metadata,
        )

    def _execute_plan_with_auto_mutation_receipts(
        self,
        plan: Plan,
        *,
        request_token: str | None,
        user_message_id: int,
    ) -> tuple[list[ToolResult], set[int]]:
        argument_failures = [
            self.executor.validate_action_arguments(action)
            for action in plan.actions
        ]
        if any(result is not None for result in argument_failures):
            return [
                result
                if result is not None
                else self.executor.argument_contract_plan_blocked_result(action)
                for action, result in zip(plan.actions, argument_failures)
            ], set()

        if any(action.tool_name == "resolve_auto_mutation_receipt" for action in plan.actions) and not (
            plan.metadata.get("runtime_auto_mutation_reconciliation_alias") is True
            and plan.metadata.get("reconciliation_operation") == "resolve"
        ):
            return [
                self._auto_mutation_blocked_result(
                    action,
                    failure_kind=(
                        "reconciliation_explicit_route_required"
                        if action.tool_name == "resolve_auto_mutation_receipt"
                        else "reconciliation_plan_preflight_blocked"
                    ),
                    output=(
                        "Mutation reconciliation runs only from the explicit reviewed resolution command. "
                        "Nothing in this plan ran."
                    ),
                )
                for action in plan.actions
            ], set()

        eligible = [
            (plan_index, action)
            for plan_index, action in enumerate(plan.actions)
            if self._auto_mutation_action_is_eligible(action)
        ]
        if not eligible:
            return [self.executor.execute(action, approved=False) for action in plan.actions], set()

        try:
            for _plan_index, action in eligible:
                tool = self.registry.get(action.tool_name)
                if not self.executor.policy.check(tool, approved=False).allowed:
                    return self._auto_mutation_plan_blocked_results(plan, "POLICY_BLOCKED"), set()
        except Exception:
            return self._auto_mutation_plan_blocked_results(plan, "PREPARE_FAILED"), set()

        try:
            private_request_token = self._private_auto_mutation_request_token(
                request_token,
                user_message_id,
            )
        except (TypeError, ValueError):
            return self._auto_mutation_plan_blocked_results(plan, "INVALID_REQUEST_TOKEN"), set()

        prepared_actions: list[
            tuple[str, dict[str, Any], str, dict[str, Any], tuple[str, ...]]
        ] = []
        operation_prepare_failed = False
        try:
            for _plan_index, action in eligible:
                contract = self.registry.get(action.tool_name).auto_mutation_contract
                if contract is None:
                    raise RuntimeError("eligible mutation lost its registry contract")
                operation_args = (
                    contract.operation_key_builder(dict(action.args))
                    if contract.operation_key_builder is not None
                    else dict(action.args)
                )
                if not isinstance(operation_args, dict):
                    raise TypeError("auto mutation operation key builder must return a dictionary")
                prepared_actions.append(
                    (
                        action.tool_name,
                        action.args,
                        contract.operation_scope or action.tool_name,
                        operation_args,
                        tuple(sorted(contract.legacy_operation_aliases)),
                    )
                )
        except Exception:
            operation_prepare_failed = True
            prepared_actions = []

        preparation = None
        if not operation_prepare_failed:
            try:
                inspection = self.store.inspect_auto_mutation_receipts(
                    private_request_token,
                    prepared_actions,
                )
            except Exception:
                return self._auto_mutation_plan_blocked_results(plan, "PREPARE_FAILED"), set()
            if inspection.status == "RESUMED":
                preparation = inspection
            elif inspection.status != "ABSENT":
                return self._auto_mutation_plan_blocked_results(plan, inspection.status), set()

        semantic_failure = ""
        semantic_failure_plan_index: int | None = None
        try:
            if preparation is None:
                for plan_index, action in eligible:
                    contract = self.registry.get(action.tool_name).auto_mutation_contract
                    if contract is None:
                        raise RuntimeError("eligible mutation lost its registry contract")
                    if contract.semantic_preflight is not None:
                        reason = contract.semantic_preflight(dict(action.args))
                        if reason is not None:
                            if (
                                not isinstance(reason, str)
                                or re.fullmatch(r"[A-Za-z0-9_:-]{1,80}", reason) is None
                            ):
                                raise ValueError("invalid semantic preflight reason")
                            semantic_failure = reason
                            semantic_failure_plan_index = plan_index
                            break
        except Exception:
            return self._auto_mutation_plan_blocked_results(plan, "PREPARE_FAILED"), set()
        if semantic_failure:
            blocked_results: list[ToolResult] = []
            for plan_index, action in enumerate(plan.actions):
                custom_result: ToolResult | None = None
                if plan_index == semantic_failure_plan_index:
                    try:
                        contract = self.registry.get(action.tool_name).auto_mutation_contract
                        builder = (
                            contract.semantic_preflight_result_builder
                            if contract is not None
                            else None
                        )
                        if builder is not None:
                            candidate = builder(dict(action.args), semantic_failure)
                            if (
                                isinstance(candidate, ToolResult)
                                and candidate.tool_name == action.tool_name
                                and type(candidate.ok) is bool
                                and candidate.ok is False
                                and type(candidate.output) is str
                                and candidate.metadata.get("handler_invoked") is False
                                and candidate.metadata.get("failure_kind")
                                == "auto_mutation_semantic_preflight_rejected"
                                and all(
                                    candidate.metadata.get(key) is False
                                    for key in (
                                        "requires_confirmation",
                                        "executed_handler",
                                        "handler_invoked",
                                        "authorizes_retry",
                                        "authorizes_execution",
                                        "authorizes_completion_claim",
                                        "approval_granted",
                                        "queues_approval",
                                        "requires_approval",
                                        "state_changed",
                                        "writes_files",
                                        "writes_database",
                                        "writes_memory",
                                        "writes_notes",
                                        "external_side_effect",
                                        "controls_computer",
                                    )
                                )
                            ):
                                custom_result = candidate
                    except Exception:
                        custom_result = None
                blocked_results.append(
                    custom_result
                    if custom_result is not None
                    else self._auto_mutation_blocked_result(
                        action,
                        failure_kind="auto_mutation_semantic_preflight_rejected",
                        output=(
                            "This local mutation failed its deterministic preflight, so nothing ran and no "
                            "mutation receipt was created. Correct the request and try again."
                        ),
                        extra_metadata={"reason": semantic_failure},
                    )
                )
            return blocked_results, set()

        if operation_prepare_failed:
            return self._auto_mutation_plan_blocked_results(plan, "PREPARE_FAILED"), set()

        execution_envelopes_by_plan_index: dict[int, Any] = {}
        try:
            for (plan_index, action), (
                _tool_name,
                _args,
                _scope,
                operation_args,
                _aliases,
            ) in zip(eligible, prepared_actions):
                contract = self.registry.get(action.tool_name).auto_mutation_contract
                if contract is None:
                    raise RuntimeError("eligible mutation lost its registry contract")
                if contract.execution_args_builder is None:
                    continue
                candidate = contract.execution_args_builder(
                    dict(action.args),
                    dict(operation_args),
                )
                envelope = self.executor.prepare_auto_mutation_execution(action, candidate)
                if isinstance(envelope, ToolResult):
                    raise ValueError("invalid internal mutation execution binding")
                execution_envelopes_by_plan_index[plan_index] = envelope
        except Exception:
            return self._auto_mutation_plan_blocked_results(
                plan,
                "EXECUTION_BINDING_FAILED",
            ), set()

        if preparation is None:
            try:
                preparation = self.store.prepare_auto_mutation_receipts(
                    private_request_token,
                    prepared_actions,
                )
            except Exception:
                return self._auto_mutation_plan_blocked_results(plan, "PREPARE_FAILED"), set()
        if preparation.status not in {"PREPARED", "RESUMED"} or len(preparation.receipts) != len(eligible):
            return self._auto_mutation_plan_blocked_results(plan, preparation.status), set()

        receipts_by_plan_index = {
            plan_index: receipt
            for (plan_index, _action), receipt in zip(eligible, preparation.receipts)
        }
        results: list[ToolResult] = []
        finalized_indexes: set[int] = set()
        for plan_index, action in enumerate(plan.actions):
            receipt = receipts_by_plan_index.get(plan_index)
            if receipt is None:
                results.append(self.executor.execute(action, approved=False))
                continue
            if receipt.receipt_id is None:
                results.append(self._auto_mutation_nonclaim_result(action, "MALFORMED_RECEIPT"))
                continue
            try:
                claim = self.store.claim_auto_mutation_receipt(receipt.receipt_id, private_request_token)
            except Exception:
                results.append(self._auto_mutation_nonclaim_result(action, "CLAIM_FAILED"))
                continue
            if claim.status != "CLAIMED" or not claim.run_token or claim.receipt_id is None:
                results.append(self._auto_mutation_nonclaim_result(action, claim.status))
                continue

            result = self.executor.execute(
                action,
                approved=False,
                execution_envelope=execution_envelopes_by_plan_index.get(plan_index),
            )
            handler_invoked = result.metadata.get("handler_invoked") is True
            exact_success = type(result.ok) is bool and result.ok is True and handler_invoked
            if not exact_success:
                contract = self.registry.get(action.tool_name).auto_mutation_contract
                definite_no_effect = (
                    handler_invoked
                    and len(eligible) == 1
                    and contract is not None
                    and self._auto_mutation_failure_is_definite_no_effect(contract, result)
                )
                if definite_no_effect:
                    try:
                        abandoned = self.store.abandon_auto_mutation_receipt_before_effects(
                            claim.receipt_id,
                            claim.run_token,
                        )
                    except Exception:
                        abandoned = False
                    if abandoned:
                        result.metadata["auto_mutation_definite_no_effect"] = True
                        result.metadata["auto_mutation_outcome_uncertain"] = False
                        results.append(result)
                        continue
                self._mark_auto_mutation_uncertain(claim.receipt_id, claim.run_token)
                if not handler_invoked:
                    result = self._auto_mutation_blocked_result(
                        action,
                        failure_kind="auto_mutation_claimed_without_handler",
                        output=(
                            "The mutation was claimed but no handler execution could be verified. "
                            "Its outcome is blocked for review and it will not be retried automatically."
                        ),
                    )
                else:
                    result = self._auto_mutation_uncertain_result(action, result)
                results.append(result)
                continue

            try:
                tool = self.registry.get(action.tool_name)
                run_id = self.store.complete_auto_mutation_receipt(
                    receipt_id=claim.receipt_id,
                    run_token=claim.run_token,
                    session_id=self.session_id,
                    risk=tool.risk.name,
                    ok=True,
                    output=result.output,
                    metadata=self._tool_run_audit_metadata(result),
                )
            except Exception as exc:
                self._mark_auto_mutation_uncertain(claim.receipt_id, claim.run_token)
                results.append(
                    self._auto_mutation_blocked_result(
                        action,
                        failure_kind="auto_mutation_completion_failed",
                        output=(
                            "The mutation handler ran, but durable audit completion failed. "
                            "The outcome is uncertain and will not be retried automatically; review the target state."
                        ),
                        handler_invoked=True,
                        extra_metadata={
                            "audit_logging_failed": True,
                            "storage_degraded": True,
                            "storage_error": type(exc).__name__,
                        },
                    )
                )
                continue
            result.metadata["logged_tool_run_id"] = run_id
            results.append(result)
            finalized_indexes.add(plan_index)
        return results, finalized_indexes

    def _approved_rerun_plan(
        self,
        user_input: str,
        approved_approval_id: int | None,
    ) -> tuple[Plan, str, ApprovalExecutionClaim | None]:
        metadata: dict[str, Any] = {
            "approval_rerun_bound": False,
            "approval_rerun_exact": False,
            "approved_approval_id": approved_approval_id,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        }

        def refused(reason: str, detail: str) -> tuple[Plan, str, ApprovalExecutionClaim | None]:
            return (
                Plan(
                    goal="Refuse an unbound approved rerun.",
                    actions=[],
                    needs_model=False,
                    notes="approval_rerun_binding_refused",
                    metadata={**metadata, "approval_rerun_binding_failure": reason},
                ),
                detail,
                None,
            )

        if isinstance(approved_approval_id, bool) or not isinstance(approved_approval_id, int) or approved_approval_id < 1:
            return refused("missing_approval_id", "A valid stored approval ID is required; nothing was executed.")

        try:
            claim = self.store.claim_approval_execution(
                approved_approval_id,
                expected_user_input=user_input,
            )
        except Exception:
            return refused(
                "approval_claim_failed",
                f"Approval #{approved_approval_id} could not be durably claimed; nothing was executed.",
            )
        if claim.status == "approval_already_used":
            return refused(
                "approval_already_used",
                f"Approval #{approved_approval_id} has already been used; one-shot approval cannot be replayed.",
            )
        if claim.status == "approval_request_mismatch":
            return refused(
                "approval_request_mismatch",
                f"Approval #{approved_approval_id} does not match this request; nothing was executed.",
            )
        if claim.status != "claimed":
            return refused(
                "approval_not_approved_or_missing",
                f"Approval #{approved_approval_id} is missing or no longer approved; nothing was executed.",
            )

        action, binding_reason, binding_error = self._action_from_approval_execution_claim(claim)
        if action is None:
            self._finalize_approval_execution_claim(claim, "binding_failed")
            return refused(
                binding_reason,
                binding_error,
            )
        return (
            Plan(
                goal=f"Execute exact stored approval #{approved_approval_id}.",
                actions=[action],
                needs_model=False,
                notes="approval_rerun_exact_binding",
                metadata={
                    **metadata,
                    "approval_rerun_bound": True,
                    "approval_rerun_exact": True,
                    "approval_rerun_tool_name": action.tool_name,
                    "approval_rerun_arg_keys": sorted(action.args),
                },
            ),
            "",
            claim,
        )

    def _action_from_approval_execution_claim(
        self,
        claim: ApprovalExecutionClaim,
    ) -> tuple[PlannedAction | None, str, str]:
        row = claim.approval
        if row is None:
            return None, "malformed_stored_action", "The stored approval action is unavailable; nothing was executed."
        approval_id = int(row["id"])
        tool_name = str(row["tool_name"] or "").strip()
        try:
            planned_args = json.loads(row["planned_args"])
        except (TypeError, ValueError, json.JSONDecodeError):
            planned_args = None
        if not tool_name or not isinstance(planned_args, dict):
            return (
                None,
                "malformed_stored_action",
                f"Approval #{approval_id} has no valid stored action; nothing was executed.",
            )
        try:
            self.registry.get(tool_name)
        except KeyError:
            return (
                None,
                "stored_tool_unavailable",
                f"Approval #{approval_id} references an unavailable tool; nothing was executed.",
            )
        return (
            PlannedAction(
                tool_name,
                dict(planned_args),
                f"Execute the exact stored action for approval #{approval_id}.",
            ),
            "",
            "",
        )

    def _finalize_approval_execution_claim(
        self,
        claim: ApprovalExecutionClaim,
        outcome: str,
    ) -> bool:
        row = claim.approval
        if row is None or not claim.token:
            return False
        try:
            return self.store.finalize_approval_execution(int(row["id"]), claim.token, outcome)
        except Exception:
            return False

    def _finalize_approved_execution(
        self,
        claim: ApprovalExecutionClaim,
        action: PlannedAction,
        result: ToolResult,
    ) -> tuple[ApprovedExecutionOutcome, list[str]]:
        try:
            tool = self.registry.get(action.tool_name)
            risk = tool.risk
            risk_name = tool.risk.name
        except Exception:
            risk = None
            risk_name = "UNKNOWN"
        classified = classify_approved_execution_outcome(
            ok=result.ok,
            metadata=result.metadata,
            risk=risk,
        )
        result.metadata["approved_execution_outcome"] = classified.outcome
        result.metadata["approved_execution_outcome_reason"] = classified.reason

        warnings: list[str] = []
        try:
            row = claim.approval
            if row is None:
                raise RuntimeError("approved execution claim has no approval row")
            run_id = self.store.record_approval_execution_result(
                approval_id=int(row["id"]),
                claim_token=claim.token,
                action_digest=approval_action_digest(action.tool_name, action.args),
                session_id=self.session_id,
                tool_name=result.tool_name,
                risk=risk_name,
                ok=result.ok,
                output=_runtime_persisted_tool_output(result),
                metadata=self._tool_run_audit_metadata(result),
                claim_outcome=classified.outcome,
            )
            result.metadata["logged_tool_run_id"] = run_id
        except Exception as exc:
            result.metadata["audit_logging_failed"] = True
            result.metadata["storage_degraded"] = True
            result.metadata["storage_error"] = type(exc).__name__
            warnings.append(self._approval_execution_storage_warning("tool-run audit logging"))
            if not self._finalize_approval_execution_claim(claim, "audit_failed"):
                result.metadata["approval_execution_finalization_failed"] = True
                warnings.append(self._approval_execution_storage_warning("execution outcome logging"))
        return classified, warnings

    def _approval_execution_storage_warning(self, phase: str) -> str:
        return (
            f"Storage warning: {phase} did not complete after the approved action was durably claimed. "
            "This approval is consumed and cannot be replayed. Verify the target state before issuing a fresh request."
        )

    def _activate_default_storage_fallback(self, exc: Exception) -> bool:
        if self.storage_fallback is not None:
            return True
        if os.getenv("JARVIS_DB_PATH", "").strip() or os.getenv("JARVIS_DATA_DIR", "").strip():
            return False
        if os.getenv("JARVIS_DISABLE_STORAGE_FALLBACK", "").strip().lower() in TRUE_ENV_VALUES:
            return False

        fallback_dir_env = os.getenv("JARVIS_STORAGE_FALLBACK_DIR", "").strip()
        fallback_dir = Path(fallback_dir_env or Path.cwd() / ".jarvis_v3_runtime").expanduser()
        fallback_db_path = fallback_dir / "jarvis.sqlite"
        fallback_vault_path = fallback_dir / "Vault"
        had_chat = getattr(self, "chat", None) is not None
        if had_chat:
            self._invalidate_history_policy_binding()
        try:
            from jarvis_v2.tools.storage import storage_diagnostics

            primary_storage_diagnostics = storage_diagnostics(self.config)
            fallback_store = MemoryStore(fallback_db_path)
            fallback_store.init()
            fallback_vault = ObsidianVault(fallback_vault_path, self.config.obsidian_root)
            fallback_vault.init()
            fallback_config = replace(
                self.config,
                data_dir=fallback_dir,
                db_path=fallback_db_path,
                obsidian_vault=fallback_vault_path,
            )
            self.config = fallback_config
            self.store = fallback_store
            self.vault = fallback_vault
            self.subagent_fleet = spawn_subagent_fleet(
                count=10,
                store=self.store,
                runtime_id=self.session_id,
            )
            self.storage_fallback = {
                "db_path": str(fallback_db_path),
                "db_path_display": "workspace-local fallback database",
                "data_dir": str(fallback_dir),
                "vault_path": str(fallback_vault_path),
                "vault_path_display": "workspace-local fallback notes",
                "reason": "primary_storage_not_writable",
                "exception_type": type(exc).__name__,
                "primary_storage_diagnostics": primary_storage_diagnostics,
            }
            self.registry = build_core_registry(
                self.store,
                self.vault,
                self.config,
                self.session_id,
                lambda: self.storage_fallback,
                lambda: self.subagent_fleet,
                lambda: getattr(self, "chat", None),
            )
            self.planner = (
                ModelBackedPlanner(
                    self.config.planner_model,
                    self.registry,
                    timeout_seconds=self.config.model_timeout_seconds,
                    provider=self.config.model_provider,
                    reasoning_effort=self.config.planner_reasoning_effort,
                    openai_max_output_tokens=self.config.openai_max_output_tokens,
                )
                if self.config.use_model_planner
                else RuleBasedPlanner()
            )
            self.executor = Executor(self.registry, PermissionPolicy())
            self.chat = ChatBrain(
                self.config.chat_model,
                self.store,
                self.vault,
                model_timeout_seconds=self.config.chat_timeout_seconds,
                max_reply_tokens=self.config.chat_max_reply_tokens,
                max_history_messages=self.config.chat_max_history_messages,
                provider=self.config.model_provider,
                reasoning_effort=self.config.chat_reasoning_effort,
                openai_max_output_tokens=self.config.openai_max_output_tokens,
                allow_remote_personal_context=self.config.allow_remote_personal_context,
            )
            if had_chat:
                self._activate_fresh_history_policy_binding()
            return True
        except Exception:
            return False

    def _storage_fallback_note(self) -> str:
        if not self.storage_fallback:
            return ""
        return (
            "Storage fallback: the default Jarvis database was not writable, so this turn was audited in the "
            f"{self.storage_fallback.get('db_path_display', 'workspace-local fallback database')} and Jarvis-owned notes "
            f"were routed to {self.storage_fallback.get('vault_path_display', 'workspace-local fallback notes')}. "
            "Set `JARVIS_DATA_DIR`, `JARVIS_DB_PATH`, or `JARVIS_OBSIDIAN_VAULT` to writable durable locations "
            "to restore the primary memory store."
        )

    def _storage_degraded_result(self, user_input: str, exc: Exception) -> RuntimeResult:
        plan = RuleBasedPlanner().plan(user_input)
        planned_actions = [self._action_trace(action) for action in plan.actions]
        warning = self._storage_warning("session logging", exc)
        response = (
            "Jarvis storage is currently not writable, so I stopped before routing or executing this command.\n\n"
            f"{warning}\n\n"
            "Safe recovery:\n"
            "- Check that `JARVIS_DB_PATH` points to a writable SQLite file.\n"
            "- Check the database file and parent directory permissions.\n"
            "- Set `JARVIS_DATA_DIR` to a writable location if this is a sandbox or moved workspace.\n"
            "- Retry the command after storage is writable so Jarvis can audit the turn."
        )
        metadata = {
            "runtime_route": "storage_degraded",
            "chat_response": {},
            "runtime_trace": {
                "request": user_input,
                "session_id": self.session_id,
                "route": "storage_degraded",
                "goal": plan.goal,
                "needs_model": plan.needs_model,
                "approved": False,
                "approved_approval_id": None,
                "verified": False,
                "verification": "storage unavailable before command routing",
                "planned_actions": planned_actions,
                "tool_results": [],
                "risk_levels": sorted({action["risk"] for action in planned_actions if action.get("risk")}),
                "approval_required": False,
                "queued_approval_ids": [],
                "new_approval_ids": [],
                "reused_approval_ids": [],
                "approval_queue_before": 0,
                "approval_queue_after": 0,
                "approval_queue_delta": 0,
                "referenced_approval_ids": [],
                "approved_reruns": 0,
                "approved_rerun_run_ids": [],
                "approved_rerun_approval_ids": [],
                "ran_tool_handlers": False,
                "chat_response": {},
                "storage_degraded": True,
                "storage_error": type(exc).__name__,
                "storage_warning": warning,
                "stages": [
                    {
                        "stage": "perception",
                        "status": "ok",
                        "detail": "accepted text command",
                    },
                    {
                        "stage": "storage",
                        "status": "blocked",
                        "detail": "memory/audit database is not writable",
                    },
                    {
                        "stage": "execution",
                        "status": "held",
                        "detail": "stopped before planning side effects or tool execution",
                    },
                ],
                "safety_boundary": {
                    "auto_allows": ["READ_ONLY", "LOCAL_SAFE"],
                    "approval_required_for": ["PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"],
                    "storage_required_for_execution": True,
                    "queued_before_risky_execution": False,
                },
            },
        }
        return RuntimeResult(user_input, plan, [], False, response, metadata)

    def _is_storage_write_error(self, exc: Exception) -> bool:
        if isinstance(exc, (sqlite3.OperationalError, sqlite3.DatabaseError, PermissionError, OSError)):
            message = str(exc).lower()
            return any(
                marker in message
                for marker in (
                    "readonly",
                    "read-only",
                    "attempt to write",
                    "permission denied",
                    "unable to open database file",
                    "not a directory",
                    "disk i/o error",
                )
            )
        return False

    def _storage_warning(self, phase: str, exc: Exception) -> str:
        return (
            f"Storage warning: {phase} could not write to the memory/audit store. "
            "Set JARVIS_DATA_DIR and JARVIS_DB_PATH to writable local paths, run "
            f"`{V3_BOOTSTRAP_CHECK_COMMAND}`, then retry. Jarvis will not treat "
            "this turn as fully verified until the memory/audit store is writable."
        )

    def diagnose_command(self, user_input: str) -> dict[str, Any]:
        from jarvis_v2.agent.planner import RuleBasedPlanner

        message = user_input.strip()
        display_message = _runtime_output_preview(message, limit=MAX_RUNTIME_OUTPUT_PREVIEW_CHARS)
        plan = RuleBasedPlanner().plan(message)
        planner_metadata = _runtime_empty_planner_metadata()
        chat_preview = self.chat.preview_loop(message)
        planned_actions = [self._action_trace(action) for action in plan.actions]
        from jarvis_v2.tools import autonomy

        recovery_closure = autonomy._execution_health_recovery_closure_snapshot(self.store.recent_tool_runs(limit=12))
        approval_required = any(
            action.get("risk") in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK", "UNKNOWN"}
            for action in planned_actions
        )
        read_only_actions = bool(planned_actions) and all(action.get("risk") == "READ_ONLY" for action in planned_actions)
        recovery_closure_blocks_current_command = bool(recovery_closure["blocks_auto_execution"] and not read_only_actions)
        recovery_closure_allows_read_only_diagnostics = bool(recovery_closure["blocks_auto_execution"] and read_only_actions)
        pending_approvals = len(self.store.list_pending_approvals(limit=100))
        approval_queue_forecast = []
        for action, action_trace in zip(plan.actions, planned_actions):
            if action_trace.get("risk") not in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK", "UNKNOWN"}:
                continue
            existing = self.store.find_matching_pending_approval(message, action.tool_name, action.args)
            existing_id = int(existing["id"]) if existing is not None else None
            approval_queue_forecast.append(
                {
                    "tool_name": action.tool_name,
                    "risk": action_trace.get("risk"),
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
        if plan.needs_model and not plan.actions:
            route = "chat"
            recommendation = "ANSWER_IN_CHAT"
        elif approval_required and pending_approvals:
            route = "hold"
            recommendation = "REVIEW_EXISTING_APPROVALS"
        elif recovery_closure_blocks_current_command:
            route = "recovery_closure"
            recommendation = "RECOVERY_CLOSURE_REQUIRED"
        elif approval_required:
            route = "approval"
            recommendation = "ENTER_EXECUTION_GOVERNOR"
        elif plan.actions:
            route = "auto_tool"
            recommendation = "AUTO_RUN_LOCAL_SAFE"
        else:
            route = "none"
            recommendation = "ASK_FOR_MORE_DETAIL"
        if route == "approval":
            recommended_next_commands = [
                f"execution governor: {display_message}",
                "send this command normally to queue an approval receipt",
                "approval readiness latest",
                "approval packet latest",
                "approval chain proof latest",
                "pending approvals",
                "approve approval latest",
            ]
        elif route == "hold":
            recommended_next_commands = [
                "pending approvals",
                "approval review",
                "approval readiness latest",
                "approval packet latest",
                "approval chain proof latest",
                "approve approval latest",
            ]
        elif route == "recovery_closure":
            recommended_next_commands = list(recovery_closure["required_commands"]) or ["execution health report"]
        elif route == "auto_tool":
            recommended_next_commands = [f"execution governor: {display_message}", "send this command normally"]
        elif route == "chat":
            recommended_next_commands = ["send this message normally"]
        else:
            recommended_next_commands = ["add more detail, or ask for `jarvis help`"]

        diagnosis = {
            "request": display_message,
            "display_request": display_message,
            "route": route,
            "recommendation": recommendation,
            "next_command": recommended_next_commands[0],
            "goal": _runtime_output_preview(plan.goal, limit=MAX_RUNTIME_OUTPUT_PREVIEW_CHARS),
            "needs_model": plan.needs_model,
            "planner_notes": plan.notes,
            "planner_metadata": planner_metadata,
            "planner_model_planner_attempted": planner_metadata["model_planner_attempted"],
            "planner_model_planner_state": planner_metadata["model_planner_state"],
            "planner_model_planner_used": planner_metadata["model_planner_used"],
            "planner_model_planner_fell_back": planner_metadata["model_planner_fell_back"],
            "planner_model_planner_fallback_reason": planner_metadata["model_planner_fallback_reason"],
            "planner_model_planner_fallback_detail": planner_metadata["model_planner_fallback_detail"],
            "planner_model_planner_recovery_hint": planner_metadata.get("model_planner_recovery_hint"),
            "planner_model_planner_exception_type": planner_metadata["model_planner_exception_type"],
            "planner_model_planner_model": planner_metadata["model_planner_model"],
            "planner_model_planner_timeout_seconds": planner_metadata["model_planner_timeout_seconds"],
            "planner_model_planner_action_count": planner_metadata["model_planner_action_count"],
            "planner_model_planner_ignored_unknown_tools": planner_metadata["model_planner_ignored_unknown_tools"],
            "planned_actions": planned_actions,
            "approval_required": approval_required,
            "pending_approvals": pending_approvals,
            "approval_queue_forecast": approval_queue_forecast,
            "forecast_new_approvals": forecast_new_approvals,
            "forecast_reused_approval_ids": forecast_reused_approval_ids,
            "forecast_queue_before": pending_approvals,
            "forecast_queue_after_if_sent": pending_approvals + forecast_new_approvals,
            "forecast_queue_delta_if_sent": forecast_new_approvals,
            "recommended_next_commands": recommended_next_commands,
            "recovery_closure_state": recovery_closure["state"],
            "recovery_closure_ready_to_retry": recovery_closure["ready_to_retry"],
            "recovery_closure_missing": recovery_closure["missing"],
            "recovery_closure_missing_count": recovery_closure["missing_count"],
            "recovery_closure_required_commands": recovery_closure["required_commands"],
            "recovery_closure_next_required_command": recovery_closure["next_required_command"],
            "recovery_closure_blocks_auto_execution": recovery_closure["blocks_auto_execution"],
            "recovery_closure_blocks_current_command": recovery_closure_blocks_current_command,
            "recovery_closure_allows_read_only_diagnostics": recovery_closure_allows_read_only_diagnostics,
            "recovery_closure_target_run_id": recovery_closure["target_run_id"],
            "recovery_closure_target_tool_name": recovery_closure["target_tool_name"],
            "recovery_closure_proof_queue": recovery_closure["required_commands"],
            "recovery_closure_proof_queue_count": len(recovery_closure["required_commands"]),
            "recovery_closure_next_proof_command": recovery_closure["next_required_command"],
            "chat_loop_preview": {
                **chat_preview,
                "used_for_response": False,
                "runtime_route": "diagnosis_only",
            },
            "safe_to_execute_now": route in {"chat", "auto_tool"},
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "approves_request": False,
            "dismisses_request": False,
            "controls_computer": False,
            "reads_private_data": False,
            "writes_files": False,
            "safety_boundary": {
                "auto_allows": ["READ_ONLY", "LOCAL_SAFE"],
                "approval_required_for": ["PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"],
                "diagnosis_only": True,
            },
        }
        return _runtime_safe_value(diagnosis)

    def _build_runtime_trace(
        self,
        user_input: str,
        route: str,
        plan,
        results,
        verified: bool,
        verification: str,
        chat_response: dict[str, Any],
        approved: bool,
        approved_approval_id: int | None,
        approval_queue_before: int,
        approval_queue_after: int,
    ) -> dict[str, Any]:
        result_summaries = [self._result_trace(result) for result in results]
        argument_failure_kinds = {
            "tool_arguments_invalid",
            "tool_argument_plan_preflight_blocked",
            "auto_mutation_duplicate_operation",
            "auto_mutation_semantic_preflight_rejected",
        }
        planned_actions = [
            self._action_trace(
                action,
                redact_arguments=(
                    index < len(results)
                    and (
                        results[index].metadata.get("failure_kind") in argument_failure_kinds
                        or results[index].metadata.get("suppress_output_persistence") is True
                    )
                ),
            )
            for index, action in enumerate(plan.actions)
        ]
        queued_approval_ids = [
            result.metadata.get("approval_id")
            for result in results
            if result.metadata.get("requires_confirmation") and result.metadata.get("approval_id") is not None
        ]
        new_approval_ids = [
            result.metadata.get("approval_id")
            for result in results
            if result.metadata.get("requires_confirmation")
            and result.metadata.get("approval_id") is not None
            and not result.metadata.get("reused_pending_approval")
        ]
        reused_approval_ids = [
            result.metadata.get("approval_id")
            for result in results
            if result.metadata.get("requires_confirmation")
            and result.metadata.get("approval_id") is not None
            and result.metadata.get("reused_pending_approval")
        ]
        referenced_approval_ids = sorted(
            {
                result.metadata.get("approval_id")
                for result in results
                if not result.metadata.get("requires_confirmation") and isinstance(result.metadata.get("approval_id"), int)
            }
        )
        approved_rerun_results = [
            summary.get("approved_rerun_result")
            for summary in result_summaries
            if isinstance(summary.get("approved_rerun_result"), dict)
        ]
        approved_rerun_run_ids = [
            rerun.get("logged_tool_run_id")
            for rerun in approved_rerun_results
            if isinstance(rerun.get("logged_tool_run_id"), int)
        ]
        approved_rerun_approval_ids = sorted(
            {
                rerun.get("approved_approval_id")
                for rerun in approved_rerun_results
                if isinstance(rerun.get("approved_approval_id"), int)
            }
        )
        approval_required = any(
            _runtime_metadata_bool(result.metadata.get("requires_confirmation"))
            for result in results
        )
        handler_execution_states = [
            (
                _runtime_metadata_bool(result.metadata.get("handler_invoked"))
                if "handler_invoked" in result.metadata
                else _runtime_metadata_bool(result.metadata.get("executed_handler"))
                if "executed_handler" in result.metadata
                else result.metadata.get("requires_confirmation") is False
                if "requires_confirmation" in result.metadata
                else False
            )
            for result in results
        ]
        executed_handler_count = sum(1 for executed in handler_execution_states if executed)
        ran_tool_handlers = executed_handler_count > 0
        if results and executed_handler_count == len(results):
            execution_status = "completed"
        elif executed_handler_count:
            execution_status = "partial"
        elif approval_required:
            execution_status = "held"
        else:
            execution_status = "not_run"
        risk_levels = sorted({action["risk"] for action in planned_actions if action.get("risk")})
        planner_metadata = _runtime_safe_value(getattr(plan, "metadata", {}) or {})
        stages = [
            {
                "stage": "perception",
                "status": "ok",
                "detail": "accepted text command",
            },
            {
                "stage": "planning",
                "status": "ok",
                "detail": plan.goal,
                "actions": len(plan.actions),
                "needs_model": plan.needs_model,
                "notes": plan.notes,
                "planner_metadata": planner_metadata,
            },
        ]
        if route == "chat":
            stages.append(
                {
                    "stage": "response",
                    "status": "ok",
                    "detail": chat_response.get("reply_path") or chat_response.get("source") or "chat",
                    "source": chat_response.get("source", "unknown"),
                }
            )
        else:
            stages.extend(
                [
                    {
                        "stage": "permission",
                        "status": "approval_required" if approval_required else "allowed",
                        "detail": "explicit approval required" if approval_required else "within automatic risk boundary",
                    },
                    {
                        "stage": "execution",
                        "status": execution_status,
                        "detail": (
                            f"{executed_handler_count} of {len(result_summaries)} tool handler(s) ran"
                        ),
                    },
                    {
                        "stage": "verification",
                        "status": "ok" if verified else "failed",
                        "detail": verification,
                    },
                ]
            )

        return {
            "request": user_input,
            "session_id": self.session_id,
            "route": route,
            "goal": plan.goal,
            "needs_model": plan.needs_model,
            "planner_notes": plan.notes,
            "planner_metadata": planner_metadata,
            "approved": approved,
            "approved_approval_id": approved_approval_id,
            "verified": verified,
            "verification": verification,
            "planned_actions": planned_actions,
            "tool_results": result_summaries,
            "risk_levels": risk_levels,
            "approval_required": approval_required,
            "queued_approval_ids": queued_approval_ids,
            "new_approval_ids": new_approval_ids,
            "reused_approval_ids": reused_approval_ids,
            "approval_queue_before": approval_queue_before,
            "approval_queue_after": approval_queue_after,
            "approval_queue_delta": approval_queue_after - approval_queue_before,
            "referenced_approval_ids": referenced_approval_ids,
            "approved_reruns": len(approved_rerun_results),
            "approved_rerun_run_ids": approved_rerun_run_ids,
            "approved_rerun_approval_ids": approved_rerun_approval_ids,
            "ran_tool_handlers": ran_tool_handlers,
            "executed_handler_count": executed_handler_count,
            "tool_result_count": len(result_summaries),
            "chat_response": chat_response,
            "stages": stages,
            "safety_boundary": {
                "auto_allows": ["READ_ONLY", "LOCAL_SAFE"],
                "approval_required_for": ["PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"],
                "queued_before_risky_execution": approval_required,
            },
        }

    def _action_trace(self, action, *, redact_arguments: bool = False) -> dict[str, Any]:
        try:
            tool = self.registry.get(action.tool_name)
            risk = tool.risk.name
            toolset = tool.toolset
            description = tool.description
        except Exception:
            risk = "UNKNOWN"
            toolset = "unknown"
            description = ""
        if redact_arguments:
            args_preview = {"<redacted>": "<redacted>"} if action.args else {}
            arg_keys = ["<redacted>"] if action.args else []
        else:
            args_preview = _runtime_args_preview(action.args)
            arg_keys = sorted(str(key) for key in action.args)
        return {
            "tool_name": action.tool_name,
            "toolset": toolset,
            "risk": risk,
            "reason": action.reason,
            "args": args_preview,
            "arg_keys": arg_keys,
            "description": description,
        }

    def _result_trace(self, result) -> dict[str, Any]:
        trace = {
            "tool_name": result.tool_name,
            "ok": result.ok,
            "requires_confirmation": _runtime_metadata_bool(result.metadata.get("requires_confirmation")),
            "approval_id": result.metadata.get("approval_id"),
            "approved_approval_id": result.metadata.get("approved_approval_id"),
            "failure_kind": result.metadata.get("failure_kind"),
            "output_preview": _runtime_output_preview(_runtime_persisted_tool_output(result)),
        }
        if result.metadata.get("executed_handler") is not None:
            trace["executed_handler"] = _runtime_metadata_bool(result.metadata.get("executed_handler"))
        if result.metadata.get("risk_level"):
            trace["risk_level"] = result.metadata.get("risk_level")
        if result.metadata.get("risk_value") is not None:
            trace["risk_value"] = result.metadata.get("risk_value")
        if result.metadata.get("toolset"):
            trace["toolset"] = result.metadata.get("toolset")
        if result.metadata.get("approval_rerun_exact") is not None:
            trace["approval_rerun_exact"] = _runtime_metadata_bool(result.metadata.get("approval_rerun_exact"))
        if result.metadata.get("rerun_tool_name"):
            trace["rerun_tool_name"] = result.metadata.get("rerun_tool_name")
        if result.metadata.get("rerun_arg_keys") is not None:
            trace["rerun_arg_keys"] = result.metadata.get("rerun_arg_keys")
        if result.metadata.get("approved_rerun_result"):
            trace["approved_rerun_result"] = result.metadata.get("approved_rerun_result")
        if result.metadata.get("logged_tool_run_id") is not None:
            trace["logged_tool_run_id"] = result.metadata.get("logged_tool_run_id")
        if result.metadata.get("approved_execution_outcome") is not None:
            trace["approved_execution_outcome"] = result.metadata.get("approved_execution_outcome")
        if result.metadata.get("suppress_output_persistence") is True:
            trace["suppress_output_persistence"] = True
            trace["private_content_retained"] = False
            trace["content_retention"] = "transient_only"
        for key in (
            "recipient_window_title_verified",
            "one_to_one_chat_verified",
            "send_key_requested",
            "delivery_verified",
            "confirmation_required",
            "operator_confirmation_required",
        ):
            if type(result.metadata.get(key)) is bool:
                trace[key] = result.metadata.get(key)
        if result.tool_name == "apple_reminders":
            for key in (
                "returned_content_bounded",
                "exact_list_filter_applied",
                "exact_list_identity_verified",
                "truncated",
                "apple_event_materialization_bounded",
                "query_verified",
            ):
                if type(result.metadata.get(key)) is bool:
                    trace[key] = result.metadata.get(key)
            for key in ("requested_limit", "returned_reminder_count"):
                value = result.metadata.get(key)
                if type(value) is int and value >= 0:
                    trace[key] = value
        for key in (
            "execution_started",
            "process_spawned",
            "action_attempted",
            "side_effect_possible",
            "retry_safe",
            "automatic_retry_allowed",
            "authorizes_retry",
            "output_capture_streaming",
            "capture_limit_bytes_per_stream",
            "stdout_bytes_seen",
            "stderr_bytes_seen",
            "timed_out",
            "post_start_failed",
            "output_capture_failed",
            "process_reaped",
            "termination_escalated",
            "output_truncated",
            "stdout_truncated",
            "stderr_truncated",
            "returncode",
            "process_group_cleanup_attempted",
            "descendant_cleanup_scope",
            "all_descendants_terminated_verified",
            "detached_descendants_may_survive",
        ):
            if key in result.metadata:
                trace[key] = result.metadata.get(key)
        return trace

    def _log_tool_runs(self, results, approved: bool, approved_approval_id: int | None = None) -> None:
        for result in results:
            atomic_run_id = result.metadata.get("logged_tool_run_id")
            if (
                result.tool_name == "resolve_auto_mutation_receipt"
                and result.metadata.get("audit_finalized_atomically") is True
                and type(atomic_run_id) is int
            ):
                existing = self.store.get_tool_run(atomic_run_id)
                if (
                    existing is not None
                    and str(existing["tool_name"]) == result.tool_name
                    and type(result.ok) is bool
                    and int(existing["ok"]) == (1 if result.ok else 0)
                    and int(existing["approved"]) == 0
                    and existing["approval_id"] is None
                    and existing["approval_action_digest"] is None
                ):
                    continue
            try:
                tool = self.registry.get(result.tool_name)
                risk = tool.risk.name
            except Exception:
                risk = "UNKNOWN"
            run_id = self.store.log_tool_run(
                self.session_id,
                result.tool_name,
                risk,
                result.ok,
                approved,
                _runtime_persisted_tool_output(result),
                approved_approval_id if approved else None,
                self._tool_run_audit_metadata(result),
            )
            result.metadata["logged_tool_run_id"] = run_id

    def _tool_run_audit_metadata(self, result) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in (
            "failure_kind",
            "requires_confirmation",
            "executed_handler",
            "handler_invoked",
            "handler_exception_after_invocation",
            "auto_mutation_definite_no_effect",
            "execution_outcome_unknown",
            "outcome_known",
            "durability_uncertain",
            "retry_safe",
            "automatic_retry_allowed",
            "authorizes_retry",
            "approved_execution_outcome",
            "approved_execution_outcome_reason",
            "error_type",
            "exception_type",
            "risk_level",
            "risk_value",
            "toolset",
        ):
            if key in result.metadata:
                metadata[key] = result.metadata.get(key)
        if result.metadata.get("suppress_output_persistence") is True:
            metadata["suppress_output_persistence"] = True
            metadata["private_content_retained"] = False
            metadata["content_retention"] = "transient_only"
        for key in (
            "recipient_window_title_verified",
            "one_to_one_chat_verified",
            "send_key_requested",
            "delivery_verified",
            "confirmation_required",
            "operator_confirmation_required",
        ):
            if type(result.metadata.get(key)) is bool:
                metadata[key] = result.metadata.get(key)
        if result.tool_name == "apple_reminders":
            for key in (
                "returned_content_bounded",
                "exact_list_filter_applied",
                "exact_list_identity_verified",
                "truncated",
                "apple_event_materialization_bounded",
                "query_verified",
            ):
                if type(result.metadata.get(key)) is bool:
                    metadata[key] = result.metadata.get(key)
            for key in ("requested_limit", "returned_reminder_count"):
                value = result.metadata.get(key)
                if type(value) is int and value >= 0:
                    metadata[key] = value
        if result.tool_name == "export_state_snapshot":
            path_display = result.metadata.get("path_display")
            if (
                type(path_display) is str
                and 0 < len(path_display) <= 4096
                and not path_display.startswith(("/", "\\"))
                and ".." not in Path(path_display).parts
                and not any(char in path_display for char in ("\n", "\r", "\x00"))
            ):
                metadata["path_display"] = path_display
            for key in ("content_sha256", "source_revision"):
                value = result.metadata.get(key)
                if type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value):
                    metadata[key] = value
        for key in (
            "execution_started",
            "process_spawned",
            "action_attempted",
            "side_effect_possible",
            "output_capture_streaming",
            "capture_limit_bytes_per_stream",
            "stdout_bytes_seen",
            "stderr_bytes_seen",
            "stdout_truncated",
            "stderr_truncated",
            "output_truncated",
            "timed_out",
            "post_start_failed",
            "output_capture_failed",
            "termination_escalated",
            "process_reaped",
            "returncode",
            "process_group_cleanup_attempted",
            "descendant_cleanup_scope",
            "all_descendants_terminated_verified",
            "detached_descendants_may_survive",
        ):
            if key in result.metadata:
                metadata[key] = result.metadata.get(key)
        for key in ("telegram_send_stage", "telegram_send_detail"):
            if key in result.metadata:
                metadata[key] = result.metadata.get(key)
        for key in (
            "telegram_call_stage",
            "instagram_send_stage",
            "instagram_call_stage",
            "kakao_send_stage",
            "kakao_call_stage",
            "imessage_send_stage",
            "call_stage",
            "failure_stage",
        ):
            value = result.metadata.get(key)
            if type(value) is str:
                value = " ".join(value.strip().split())
                if value:
                    metadata[key] = value[:160]
        for key in ("run_id", "verdict", "tool_name", "recovery_kind", "problem_found"):
            if key in result.metadata:
                metadata[key] = result.metadata.get(key)
        approval_id = result.metadata.get("approval_id")
        if type(approval_id) is int and 0 < approval_id <= 9223372036854775807:
            metadata["approval_id"] = approval_id
        if result.tool_name == "approval_chain_proof":
            valid_execution_proof = result.metadata.get("valid_execution_proof")
            if type(valid_execution_proof) is bool:
                metadata["valid_execution_proof"] = valid_execution_proof
        for key in (
            "checkpoint_recovery_execute_handoff_ready",
            "checkpoint_recovery_execute_handoff_state",
            "checkpoint_recovery_execute_handoff_reviewed",
            "checkpoint_recovery_execute_handoff_missing_fields",
            "checkpoint_recovery_execute_handoff_missing_field_count",
            "checkpoint_recovery_execute_handoff_normal_followthrough_allowed",
            "checkpoint_recovery_execute_handoff_recovery_followthrough_gate_state",
            "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence",
            "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence_count",
            "checkpoint_recovery_execute_handoff_recovery_closure_approval_boundary",
            "checkpoint_recovery_execute_handoff_next_safe_command",
            "checkpoint_recovery_execute_handoff_risky_recovery_signal_count",
            "checkpoint_recovery_execute_handoff_approval_reference_provided",
            "checkpoint_recovery_execute_handoff_recovery_step_approval_required_before_recovery",
            "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue",
            "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue_count",
            "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present",
            "checkpoint_recovery_execute_handoff_contains_raw_step",
            "checkpoint_recovery_execute_handoff_contains_raw_verification",
        ):
            if key in result.metadata:
                metadata[key] = result.metadata.get(key)
        planned_args = result.metadata.get("planned_args")
        if isinstance(planned_args, dict):
            metadata["planned_arg_keys"] = sorted(str(key) for key in planned_args)
        planner_reason = str(result.metadata.get("planner_reason") or "").strip()
        if planner_reason:
            metadata["planner_reason"] = _runtime_output_preview(planner_reason, limit=260)
        return metadata

    def _log_pending_approvals(self, user_input: str, results) -> None:
        for result in results:
            if result.metadata.get("requires_confirmation"):
                planned_args = result.metadata.get("planned_args", {})
                approval_id, reused = self.store.get_or_add_pending_approval(
                    self.session_id,
                    user_input,
                    result.tool_name,
                    result.output,
                    planned_args,
                )
                result.metadata["planned_args"] = _runtime_public_approval_args(planned_args)
                result.metadata["reused_pending_approval"] = reused
                result.metadata["approval_id"] = approval_id
                logged_tool_run_id = result.metadata.get("logged_tool_run_id")
                if isinstance(logged_tool_run_id, int):
                    result.metadata["approval_linked_tool_run"] = self.store.link_tool_run_approval(
                        logged_tool_run_id,
                        approval_id,
                    )
                try:
                    self.vault.write_pending_approvals(self.store.list_pending_approvals(limit=100))
                except Exception:
                    pass

    def _compose_response(self, results, verified: bool, verification: str) -> str:
        if not results:
            return verification
        if len(results) == 1:
            return self._format_result_output(results[0])
        lines = [self._format_result_output(result) for result in results]
        if not verified:
            lines.append(f"Verification failed: {verification}")
        return "\n".join(lines)

    def _format_result_output(self, result) -> str:
        if not result.metadata.get("requires_confirmation"):
            return result.output
        approval_id = result.metadata.get("approval_id")
        if approval_id is None:
            return result.output
        receipt_state = "already queued as" if result.metadata.get("reused_pending_approval") else "queued as"
        receipt = (
            f"Safety receipt: {receipt_state} approval #{approval_id}.\n"
            f"Review: pending approvals\n"
            f"Readiness: approval readiness {approval_id}\n"
            f"Required last look: approval packet {approval_id}\n"
            f"Proof check: approval chain proof {approval_id}\n"
            f"Then, if trusted: approve approval {approval_id}\n"
            f"Skip: dismiss approval {approval_id}"
        )
        return result.output + "\n\n" + receipt

    def _rerun_approved_request(self, plan, results) -> tuple[str, list]:
        responses: list[str] = []
        rerun_results = []
        for planned_action, result in zip(plan.actions, results):
            if (
                planned_action.tool_name != "approve_pending_approval"
                or result.tool_name != "approve_pending_approval"
                or not result.ok
            ):
                continue
            approval_id = result.metadata.get("approved_approval_id")
            if isinstance(approval_id, bool) or not isinstance(approval_id, int) or approval_id < 1:
                continue
            requested_approval_id = planned_action.args.get("approval_id")
            if isinstance(requested_approval_id, bool):
                continue
            if isinstance(requested_approval_id, int) and requested_approval_id != approval_id:
                continue
            if isinstance(requested_approval_id, str) and requested_approval_id.strip().isdigit():
                if int(requested_approval_id.strip()) != approval_id:
                    continue

            try:
                claim = self.store.claim_approval_execution(approval_id)
            except Exception:
                claim = ApprovalExecutionClaim("approval_claim_failed")
            if claim.status != "claimed":
                if claim.status == "approval_already_used":
                    detail = f"Approval #{approval_id} has already been used; one-shot approval cannot be replayed."
                elif claim.status == "approval_claim_failed":
                    detail = f"Approval #{approval_id} could not be durably claimed; nothing was executed."
                else:
                    detail = f"Approval #{approval_id} is missing or no longer approved; nothing was executed."
                blocked_result = ToolResult(
                    "approval_execution",
                    False,
                    detail,
                    {
                        "approved_approval_id": approval_id,
                        "approval_rerun_exact": False,
                        "approval_rerun_refused": claim.status,
                    },
                )
                rerun_results.append(blocked_result)
                result.metadata["approved_rerun_result"] = self._result_trace(blocked_result)
                result.metadata["approved_rerun_ok"] = False
                responses.append(detail)
                continue

            action, binding_reason, binding_error = self._action_from_approval_execution_claim(claim)
            if action is None:
                self._finalize_approval_execution_claim(claim, "binding_failed")
                blocked_result = ToolResult(
                    "approval_execution",
                    False,
                    binding_error,
                    {
                        "approved_approval_id": approval_id,
                        "approval_rerun_exact": False,
                        "approval_rerun_refused": binding_reason,
                    },
                )
                rerun_results.append(blocked_result)
                result.metadata["approved_rerun_result"] = self._result_trace(blocked_result)
                result.metadata["approved_rerun_ok"] = False
                responses.append(binding_error)
                continue

            rerun_result = self.executor.execute(action, approved=True)
            rerun_results.append(rerun_result)
            rerun_result.metadata["approved_approval_id"] = approval_id
            rerun_result.metadata["approval_rerun_exact"] = True
            rerun_result.metadata["rerun_tool_name"] = action.tool_name
            rerun_result.metadata["rerun_arg_keys"] = sorted(action.args)
            _classified, execution_warnings = self._finalize_approved_execution(
                claim,
                action,
                rerun_result,
            )
            result.metadata["approved_rerun_result"] = self._result_trace(rerun_result)
            result.metadata["approved_rerun_tool_name"] = action.tool_name
            result.metadata["approved_rerun_arg_keys"] = sorted(action.args)
            result.metadata["approved_rerun_ok"] = rerun_result.ok
            if rerun_result.metadata.get("suppress_output_persistence") is True:
                result.metadata["suppress_output_persistence"] = True
            rerun_response = "Approved run result:\n" + self._format_result_output(rerun_result)
            if execution_warnings:
                rerun_response += "\n\n" + "\n".join(execution_warnings)
            responses.append(rerun_response)
        return "\n\n".join(responses), rerun_results
