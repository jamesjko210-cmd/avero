"""Capability cockpit: one read-only surface answering "what can Jarvis do,
is each capability healthy, and what do I type to use it?".

Direction (2026-07-04 research note in CODEX_TASKS.md): the moat is a visible
control plane — capability, risk, approval, last success, last failure stage,
smoke coverage, example command — not a companion roster. This tool derives
every row from the live registry and audit trail; nothing here is a hardcoded
claim about health.

Read-only: no model calls, no tool execution, no approval, no side effects.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult

_LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\s\"'`<>]*")

_AUDIT_SCAN_LIMIT = 400
_MAX_DETAIL_CHARS = 90

# Curated lanes. Tool names are verified against the live registry at call
# time — a renamed/removed tool shows up as "missing", never as a crash.
CAPABILITY_LANES: list[dict[str, Any]] = [
    {
        "key": "messaging",
        "title": "Messaging (KR/EN)",
        "tools": ["send_telegram", "send_kakao", "send_imessage"],
        "example": "send 가상연락처이 a telegram saying hello",
        "smoke": [
            "smoke_test_telegram_control",
            "smoke_test_social_connectors",
            "smoke_test_imessage_connector",
            "smoke_test_channel_health",
        ],
    },
    {
        "key": "calls",
        "title": "Calls",
        "tools": ["call_contact", "call_telegram", "call_kakao", "call_instagram"],
        "example": "call 가상연락처일 on telegram",
        "smoke": ["smoke_test_call_connector", "smoke_test_channel_health"],
    },
    {
        "key": "morning_brief",
        "title": "Morning Brief",
        "tools": ["daily_brief", "daily_briefing", "schedule_morning_brief", "run_job_now"],
        "example": "run job morning brief now",
        "smoke": ["smoke_test_brief", "smoke_test_morning_brief_scheduler"],
    },
    {
        "key": "contacts",
        "title": "Contacts & People",
        "tools": ["find_contact", "get_person", "list_people"],
        "example": "who is 가상연락처이",
        "smoke": ["smoke_test_people", "smoke_test_contacts_connector", "smoke_test_contacts_fuzzy"],
    },
    {
        "key": "calendar_email",
        "title": "Calendar & Email",
        "tools": ["list_calendars", "read_emails", "search_emails", "send_email"],
        "example": "read my recent emails",
        "smoke": ["smoke_test_calendar_connector", "smoke_test_email_connector"],
    },
    {
        "key": "writing",
        "title": "Writing",
        "tools": ["compose_and_write", "human_write", "paste_text"],
        "example": "write a paragraph about Jarvis and type it",
        "smoke": ["smoke_test_compose_connector", "smoke_test_writer_connector"],
        "proof_points": [
            "compose_and_write stops before typing or pasting when model generation fails",
            "human_write and paste_text remain HIGH_RISK and approval-gated before computer control",
            "writer failures expose Accessibility recovery guidance without leaking raw exceptions or local paths",
        ],
        "notes": [
            "draft-to-write flows are inspectable, approval-gated, and covered by mocked writer/compose smokes",
        ],
    },
    {
        "key": "markets",
        "title": "Markets & Weather",
        "tools": ["get_markets_overview", "convert_currency", "get_weather"],
        "example": "markets overview",
        "smoke": ["smoke_test_markets_connector", "smoke_test_currency_connector", "smoke_test_weather_connector"],
    },
    {
        "key": "memory",
        "title": "Memory & Notes",
        "tools": ["remember", "memory_stats"],
        "example": "remember that the operator prefers direct answers",
        "smoke": ["smoke_test_core", "smoke_test_memory_stats"],
    },
    {
        "key": "voice",
        "title": "Voice",
        "tools": ["voice_command_cockpit", "list_voices"],
        "example": "voice command cockpit",
        "smoke": ["smoke_test_voice", "smoke_test_talk"],
    },
    {
        "key": "diagnostics",
        "title": "Diagnostics",
        "tools": ["readiness_report", "execution_health_report", "channel_health", "jarvis_doctor"],
        "example": "what broke",
        "smoke": ["smoke_test_doctor", "smoke_test_audit", "smoke_test_channel_health"],
    },
    {
        "key": "approvals",
        "title": "Approvals",
        "tools": [
            "list_pending_approvals",
            "inspect_pending_approval",
            "approval_readiness_packet",
            "approval_execution_packet",
            "approval_chain_proof",
            "approval_queue_summary",
            "review_pending_approvals",
            "approval_history",
        ],
        "example": "pending approvals",
        "smoke": ["smoke_test_approvals", "smoke_test_audit"],
        "proof_points": [
            "approval review tools expose pending requests, readiness, last-look packets, and chain proof without approving",
            "approve and dismiss tools are intentionally excluded from this read-only cockpit lane",
        ],
        "notes": [
            "approval lane is for review only; approving or dismissing remains an explicit owner action outside this lane",
        ],
    },
    {
        "key": "acceptance_harness",
        "title": "Acceptance Harness",
        "tools": ["capability_cockpit", "readiness_report", "channel_health", "jarvis_doctor"],
        "example": "live proof status",
        "smoke": ["smoke_test_live_check", "smoke_test_capability_cockpit", "smoke_test_chat_timeout"],
        "proof_points": [
            "live_check publishes one-screen acceptance rows for config, contacts, send resolution, voice, research, daily brief, chat, channels, and phone control",
            "acceptance coverage, acceptance gaps, acceptance next, aggregate smoke, operator evals, jobs 7-day proof, and phone approvals rows keep open-work and next-proof status visible",
            "aggregate smoke runs serialize through a bounded local suite lock so overlapping Claude/Codex aggregates cannot race shared smoke state",
            "personal proofs row audits bounded metadata only and reports missing live evidence without exposing targets or content",
            "scheduled-job freshness row checks last_run_at drift, overdue next_run_at, invalid metadata, and first-run not-yet-proven state without running jobs",
            "daemon startup row checks launcher contracts, LaunchAgent templates, scheduler ticker ownership, and restart documentation without process control",
            "error guidance row checks recovery wording without account access, web fetches, model calls, transcription, approvals, or sends",
            "daily brief degraded-section row requires recovery-hint proof before reporting unavailable sections",
            "acceptance gaps points to acceptance next for prioritized proof guidance without exposing checklist text",
            "acceptance next safe-report fields list lane, result, evidence, and stage while omitting private content and handles",
            "chat path and mixed conversation rows separate model-backed readiness from fallback or latency drift",
        ],
        "notes": [
            "diagnostic only; run opt-in live_check with operator present for real acceptance proof",
            "readiness rows do not replace live delivery, approval, phone, reboot, or daily-streak proof",
        ],
    },
    {
        "key": "build_guardrails",
        "title": "Build Guardrails",
        "tools": ["capability_cockpit", "readiness_report", "channel_health", "jarvis_doctor"],
        "example": "cockpit",
        "smoke": ["smoke_test_capability_cockpit", "smoke_test_status_server_core"],
        "notes": [
            "live-proof freeze active for planner/send/call/HUD files until the operator posts channel test results",
            "pending live-proof matrix covers Kakao, Instagram, Telegram, iMessage, phone, and FaceTime channel checks",
            "report live matrix results as channel, pass/fail, and last visible stage/error; do not include secrets or message content",
            "autonomous passes stay local-safe: diagnostics, handoff accuracy, docs, readiness reports, adjacent smokes",
        ],
    },
    {
        "key": "operator_workflow_evals",
        "title": "Operator Workflow Evals",
        "tools": ["send_telegram", "run_job_now", "find_contact", "get_markets_overview", "channel_health"],
        "example": "push today's brief to my phone",
        "smoke": ["smoke_test_operator_evals"],
        "proof_points": [
            "Korean Telegram send stops at approval with Hangul recipient/body intact",
            "Korean contact lookup resolves 가상연락처이 with relationship context",
            "Morning Brief can be pushed on demand to the owner phone channel",
            "Phone control shortcuts expose status, failures, channels, approvals, cockpit, lanes, and guardrails",
            "Voice-note bilingual send commands still stop at the same approval gate",
            "Markets summaries include Korean output; weather, currency, calendar, memory, and scheduler paths are mocked and covered",
            "Clean-state cockpit has zero false attention lanes",
        ],
        "notes": [
            "pins operator-real workflows: Korean sends, phone brief, channel diagnostics, contacts, markets",
            "send-side effects still stop at approval; the eval pack proves the gate and receipt path",
        ],
    },
    {
        "key": "personal_proofs",
        "title": "Personal Proofs",
        "tools": [
            "list_events",
            "create_event",
            "update_event",
            "delete_event",
            "read_emails",
            "search_emails",
            "send_email",
            "set_reminder",
            "find_contact",
        ],
        "example": "personal proofs status",
        "smoke": [
            "smoke_test_live_check",
            "smoke_test_calendar_connector",
            "smoke_test_email_connector",
            "smoke_test_reminders",
            "smoke_test_contacts_connector",
        ],
        "proof_points": [
            "personal proofs are still live-acceptance evidence, not mocked completion claims",
            "calendar create/update/delete and email send remain HIGH_RISK and approval-gated",
            "set_reminder proof uses Jarvis's local Telegram reminder path, not macOS Reminders creation",
            "contact proof requires distinct successful lookups while suppressing names, handles, and message content",
        ],
        "notes": [
            "read-only proof lane for WS2: calendar writes, email read/search/send, set_reminder delivery, and contact lookup",
            "run opt-in live_check with operator present for actual proof; this lane does not access accounts or send anything",
        ],
    },
    {
        "key": "research_web",
        "title": "Research & Web",
        "tools": ["web_lookup", "research", "fetch_page", "web_search", "recent_browser_pages", "summarize_page"],
        "example": "research Zoey OS",
        "smoke": ["smoke_test_research_connector", "smoke_test_next_layer", "smoke_test_live_check"],
        "proof_points": [
            "research and web_lookup emit no-authority handoff packets with content excluded from metadata",
            "live_check verifies research/web registration, risk gates, and route shape without fetching the web or calling models",
            "research failures expose DuckDuckGo Lite recovery guidance without leaking raw exceptions or local paths",
        ],
        "notes": [
            "research may call external search/page services and the local model only when explicitly run",
            "the cockpit lane itself is read-only and does not fetch pages, synthesize answers, write notes, or queue approvals",
        ],
    },
    {
        "key": "learning_loop",
        "title": "Learning Loop",
        "tools": [
            "learning_review",
            "session_learning_preview",
            "after_action_learning_packet",
            "execution_learning_closure_packet",
            "feedback_actions",
            "failure_learning_cockpit",
            "queue_learning_tasks",
        ],
        "example": "learning review",
        "smoke": ["smoke_test_learning_review"],
        "proof_points": [
            "learning_review surfaces feedback, memory health, preferences, skills, and execution learning debt without saving",
            "after_action_learning_packet and execution_learning_closure_packet bind approved runs to verification, audit, recovery, and learning evidence",
            "feedback_actions and failure_learning_cockpit turn repeated failures into reviewable fixes instead of silent behavior drift",
            "queue_learning_tasks is LOCAL_SAFE and creates only local review tasks from bounded learning signals",
        ],
        "notes": [
            "learning is a visible review loop, not autonomous self-modification",
            "saving reviews or queueing learning tasks remains local-safe and non-authorizing; risky execution still needs approvals",
        ],
    },
    {
        "key": "scheduler",
        "title": "Scheduled Jobs",
        "tools": ["list_scheduled_jobs", "run_job_now"],
        "example": "list scheduled jobs",
        "smoke": ["smoke_test_scheduler_basics"],
    },
    {
        "key": "orchestration",
        "title": "Internal Orchestration",
        "tools": ["subagent_fleet_status"],
        "example": "jarvis status",
        "smoke": ["smoke_test_subagent_fleet"],
        "proof_points": [
            "subagent_fleet_status exposes worker readiness and capacity without spawning tasks",
            "subagents stay internal orchestration capacity, not user-facing companion personas",
            "subagent smoke coverage pins readiness without granting autonomy, approvals, or tool execution",
        ],
        "notes": [
            "orchestration lane is read-only; inspect worker capacity before parallel work",
            "worker orchestration does not approve, dispatch, or run risky actions",
        ],
    },
    {
        "key": "agent_landscape",
        "title": "Agent Landscape Guardrails",
        "tools": ["work_queue", "capability_cockpit", "harness_doctrine"],
        "example": "work queue",
        "smoke": ["smoke_test_next_step", "smoke_test_capability_cockpit", "smoke_test_continuity"],
        "proof_points": [
            "Zoey/OpenClaw/Hermes research keeps Jarvis pointed at a visible control plane, not companion personas",
            "OpenAI Agents SDK/LangGraph/CrewAI/n8n lessons are traces, evals, guardrails, and human-in-the-loop checks",
            "OpenClaw/Hermes-style always-on agents warn against broad privileges, skill supply-chain risk, and persistent prompt-injection",
            "Build focus stays Korean messaging reliability, phone control, morning brief, contact lookup, and operator-real evals before broad integrations",
        ],
        "notes": [
            "copy capability discovery and health visibility; avoid autonomy theater and companion identity sprawl",
            "read-only lane; does not fetch the web, install frameworks, spawn workers, or authorize integration bridges",
        ],
    },
]

_AUTO_RUN_RISK_NAMES = {"READ_ONLY", "LOCAL_SAFE"}
_RISK_SEVERITY_ORDER = (
    "HIGH_RISK",
    "EXTERNAL_SIDE_EFFECT",
    "PERSONAL_DATA",
    "LOCAL_SAFE",
    "READ_ONLY",
)

_LANE_DIAGNOSTIC_COMMANDS = {
    "messaging": "channel health",
    "calls": "channel health",
    "morning_brief": "list scheduled jobs",
    "contacts": "capability cockpit",
    "calendar_email": "setup check",
    "personal_proofs": "personal proofs status",
    "writing": "capability cockpit",
    "markets": "capability cockpit",
    "research_web": "capability cockpit",
    "memory": "memory stats",
    "voice": "voice setup check",
    "diagnostics": "jarvis doctor",
    "approvals": "pending approvals",
    "build_guardrails": "readiness report",
    "operator_workflow_evals": "channel health",
    "scheduler": "list scheduled jobs",
    "orchestration": "jarvis status",
    "agent_landscape": "work queue",
}

_DIRECTION_SUMMARY = "One Jarvis coordinator, visible capability cockpit, internal workers only."
_DIRECTION_PRINCIPLES = [
    "Show capabilities, health, risk, approvals, smoke coverage, and next commands before adding autonomy.",
    "Keep subagents/internal workers invisible as personas; expose them only as inspectable orchestration capacity.",
    "Do not add Zoey-style companion identities unless a real workflow proves the need.",
    "Delay broad integration bridges until Korean messaging, phone control, morning brief, and contact lookup are live-proven.",
]
_EARNED_TRUST_CONTRACT = [
    "Every capability lane names its registered tools and max risk before use.",
    "Side-effect or personal-data lanes must expose an approval boundary before execution.",
    "Each lane points to aggregate smoke coverage plus the next review, diagnostic, or example command.",
    "operator-real workflow claims carry explicit proof points instead of broad completion claims.",
]

_NEXT_COMMAND_KIND_PRIORITY = {
    "approval_review": 0,
    "diagnostic": 1,
    "example": 2,
}

_NEXT_COMMAND_STATUS_PRIORITY = {
    "awaiting approval": 0,
    "attention": 1,
    "missing": 2,
    "ready": 3,
    "ok": 4,
}


def _max_risk_name(risk_names: set[str]) -> str:
    if not risk_names:
        return "NONE"
    if "UNKNOWN" in risk_names:
        return "UNKNOWN"
    for candidate in _RISK_SEVERITY_ORDER:
        if candidate in risk_names:
            return candidate
    return "UNKNOWN"


def _safe_ok(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "ok", "success"}:
            return True
        if normalized in {"0", "false", "no", "failed", "failure"}:
            return False
    return None


def _registered_smoke_modules() -> set[str] | None:
    try:
        from jarvis_v2.scripts.smoke_test_all import TEST_MODULES
    except Exception:
        return None
    modules: set[str] = set()
    for module in TEST_MODULES:
        try:
            modules.add(str(module).rsplit(".", 1)[-1])
        except Exception:
            continue
    return modules


def _smoke_coverage(smoke_modules: list[Any], registered_smokes: set[str] | None) -> dict[str, Any]:
    listed: list[str] = []
    hidden = 0
    for module in smoke_modules:
        module_name = _smoke_module_name_or_empty(module)
        if module_name:
            listed.append(module_name)
        else:
            hidden += 1
    if not listed:
        return {
            "smoke_coverage": "uncovered",
            "smoke_coverage_ready": False,
            "registered_smoke_modules": [],
            "missing_smoke_modules": [],
            "hidden_smoke_modules": hidden,
        }
    if registered_smokes is None:
        return {
            "smoke_coverage": "unknown",
            "smoke_coverage_ready": False,
            "registered_smoke_modules": [],
            "missing_smoke_modules": listed,
            "hidden_smoke_modules": hidden,
        }
    registered = [module for module in listed if module in registered_smokes]
    missing = [module for module in listed if module not in registered_smokes]
    if not registered:
        coverage = "unregistered"
    elif missing:
        coverage = "partial"
    else:
        coverage = "registered"
    return {
        "smoke_coverage": coverage,
        "smoke_coverage_ready": coverage == "registered",
        "registered_smoke_modules": registered,
        "missing_smoke_modules": missing,
        "hidden_smoke_modules": hidden,
    }


def _raw_registry_values_from_bound_list(list_tools: Callable[[], list[Any]]) -> tuple[list[Any], bool]:
    registry = getattr(list_tools, "__self__", None)
    raw_tools = getattr(registry, "_tools", None)
    if not isinstance(raw_tools, dict):
        return [], False
    try:
        return list(raw_tools.values()), True
    except Exception:
        return [], False


def _tool_registry_snapshot(list_tools: Callable[[], list[Any]]) -> tuple[dict[str, Any], int]:
    """Best-effort registry view; one malformed tool must not blind the cockpit."""
    registered: dict[str, Any] = {}
    unreadable = 0
    tools, recovered = _raw_registry_values_from_bound_list(list_tools)
    if not recovered:
        try:
            tools = list_tools()
        except Exception:
            return {}, 1
    try:
        iterable = list(tools or [])
    except Exception:
        return {}, 1
    for tool in iterable:
        try:
            name = str(tool.name)
        except Exception:
            unreadable += 1
            continue
        if not name or _LOCAL_PATH_RE.search(name):
            unreadable += 1
            continue
        registered[name] = tool
    return registered, unreadable


def _tool_risk_name(tool: Any) -> str:
    try:
        return str(tool.risk.name)
    except Exception:
        return "UNKNOWN"


def _scrub(value: Any, limit: int = _MAX_DETAIL_CHARS) -> str:
    try:
        text = _LOCAL_PATH_RE.sub("<local-path>", "" if value is None else str(value))
    except Exception:
        return "<unreadable>"
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _scrub_or(value: Any, default: str = "", limit: int = _MAX_DETAIL_CHARS) -> str:
    text = _scrub(value, limit)
    return text or default


def _bounded_text_list(value: Any, limit: int) -> list[str]:
    if value is None:
        return []
    bounded: list[str] = []
    for item in _metadata_sequence(value):
        text = _scrub(item, limit)
        if text:
            bounded.append(text)
    return bounded


def _metadata_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        try:
            return list(value)
        except Exception:
            return [value]
    return [value]


def _metadata_name_list(value: Any, limit: int = 80) -> tuple[list[str], int]:
    names: list[str] = []
    hidden = 0
    for item in _metadata_sequence(value):
        text = _scrub(item, limit)
        if not text or text in {"<local-path>", "<unreadable>"}:
            hidden += 1
            continue
        names.append(text)
    return names, hidden


def _smoke_module_name_or_empty(value: Any) -> str:
    text = _scrub(value, 80)
    if text in {"<local-path>", "<unreadable>"}:
        return ""
    return text


def _truthy_metadata(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _exact_bool(value: Any) -> bool:
    return value is True


def _safe_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _safe_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed < 0:
        return None
    return parsed


def _row_optional_value(row: Any, key: str) -> Any:
    try:
        return row[key]
    except Exception:
        return None


def _approval_review_commands(approval_id: int | None) -> list[str]:
    if approval_id is None:
        return ["approval readiness latest", "approval packet latest", "approval chain proof latest"]
    return [
        f"approval readiness {approval_id}",
        f"approval packet {approval_id}",
        f"approval chain proof {approval_id}",
    ]


def _approval_is_currently_pending(
    store: Any,
    approval_id: int | None,
    cache: dict[int, bool],
) -> bool:
    """Bind an audit hold to current queue state before advertising review."""
    if approval_id is None:
        return False
    if approval_id in cache:
        return cache[approval_id]
    try:
        cache[approval_id] = store.get_pending_approval(approval_id, status="pending") is not None
    except Exception:
        cache[approval_id] = False
    return cache[approval_id]


def _command_text_or_empty(value: Any, limit: int = 120) -> str:
    text = _scrub(value, limit)
    if text in {"<local-path>", "<unreadable>"}:
        return ""
    return text


def _lane_next_command(spec: dict[str, Any], status: str, approval_commands: list[str]) -> tuple[str, str]:
    if status == "awaiting approval" and approval_commands:
        return approval_commands[0], "approval_review"
    if status in {"attention", "missing"}:
        key = _scrub(spec.get("key"), 80)
        command = _command_text_or_empty(spec.get("diagnostic"))
        if not command:
            command = _command_text_or_empty(_LANE_DIAGNOSTIC_COMMANDS.get(key))
        if not command:
            command = "capability cockpit"
        return command, "diagnostic"
    return _command_text_or_empty(spec.get("example")), "example"


def _lane_status_counts(lanes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lane in lanes:
        status = _scrub_or(lane.get("status"), "unknown", 40)
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _coverage_summary(lanes: list[dict[str, Any]]) -> dict[str, Any]:
    lane_count = len(lanes)
    tool_ready_count = sum(1 for lane in lanes if _exact_bool(lane.get("tool_coverage_ready")))
    smoke_ready_count = sum(1 for lane in lanes if _exact_bool(lane.get("smoke_coverage_ready")))
    hidden_tool_metadata_count = sum(_safe_non_negative_int(lane.get("hidden_tool_metadata")) or 0 for lane in lanes)
    hidden_smoke_metadata_count = sum(_safe_non_negative_int(lane.get("hidden_smoke_modules")) or 0 for lane in lanes)
    approval_required_count = sum(1 for lane in lanes if _exact_bool(lane.get("approval_required")))
    auto_run_safe_count = lane_count - approval_required_count
    ready = (
        lane_count > 0
        and tool_ready_count == lane_count
        and smoke_ready_count == lane_count
        and hidden_tool_metadata_count == 0
        and hidden_smoke_metadata_count == 0
    )
    summary_parts = [
        f"tools {tool_ready_count}/{lane_count}",
        f"smokes {smoke_ready_count}/{lane_count}",
    ]
    if hidden_tool_metadata_count:
        summary_parts.append(f"hidden tool metadata {hidden_tool_metadata_count}")
    if hidden_smoke_metadata_count:
        summary_parts.append(f"hidden smoke metadata {hidden_smoke_metadata_count}")
    summary_parts.extend(
        [
            f"approval-gated {approval_required_count}",
            f"auto-run safe {auto_run_safe_count}",
        ]
    )
    return {
        "lane_count": lane_count,
        "tool_ready_count": tool_ready_count,
        "smoke_ready_count": smoke_ready_count,
        "hidden_tool_metadata_count": hidden_tool_metadata_count,
        "hidden_smoke_metadata_count": hidden_smoke_metadata_count,
        "approval_required_count": approval_required_count,
        "auto_run_safe_count": auto_run_safe_count,
        "coverage_ready": ready,
        "summary": ", ".join(summary_parts),
    }


def _next_command_queue(lanes: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    queued: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for index, lane in enumerate(lanes):
        command = _command_text_or_empty(lane.get("next_command"), 120)
        if not command:
            continue
        kind = _scrub_or(lane.get("next_command_kind"), "example", 40)
        status = _scrub_or(lane.get("status"), "unknown", 40)
        dedupe_key = (kind, command)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        sort_key = (
            _NEXT_COMMAND_KIND_PRIORITY.get(kind, 99),
            _NEXT_COMMAND_STATUS_PRIORITY.get(status, 99),
            index,
        )
        queued.append(
            (
                sort_key,
                {
                    "command": command,
                    "kind": kind,
                    "lane_key": _scrub(lane.get("key"), 60),
                    "lane_title": _scrub_or(lane.get("title"), _scrub(lane.get("key"), 80), 80),
                    "status": status,
                    "approval_required": _exact_bool(lane.get("approval_required")),
                    "last_failure": _scrub(lane.get("last_failure")),
                    "last_failure_kind": _scrub(lane.get("last_failure_kind"), 40),
                    "last_approval_hold": _scrub(lane.get("last_approval_hold")),
                    "last_approval_id": _safe_positive_int(lane.get("last_approval_id")),
                    "approval_next_commands": [
                        _scrub(command, 120)
                        for command in (lane.get("approval_next_commands") if isinstance(lane.get("approval_next_commands"), list) else [])
                        if _scrub(command, 120)
                    ][:3],
                },
            )
        )
    queued.sort(key=lambda item: item[0])
    return [entry for _, entry in queued[:limit]]


def _proof_entries(lanes: list[dict[str, Any]], *, point_limit: int = 2) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for lane in lanes:
        raw_points = lane.get("proof_points")
        if not isinstance(raw_points, list):
            continue
        points = [_scrub(point, 160) for point in raw_points if _scrub(point, 160)]
        if not points:
            continue
        entries.append(
            {
                "lane_key": _scrub(lane.get("key"), 60),
                "lane_title": _scrub_or(lane.get("title"), _scrub(lane.get("key"), 80), 80),
                "proof_count": len(points),
                "sample_proofs": points[:point_limit],
            }
        )
    return entries


def _proof_summary(lanes: list[dict[str, Any]], *, lane_limit: int = 4, point_limit: int = 2) -> list[dict[str, Any]]:
    return _proof_entries(lanes, point_limit=point_limit)[:lane_limit]


def _lane_trust_checklist(
    *,
    tool_coverage_ready: bool,
    tool_coverage: str,
    smoke_coverage_ready: bool,
    smoke_coverage: str,
    risk_names: set[str],
    max_risk: str,
    approval_required: bool,
    next_command: str,
    next_command_kind: str,
    proof_points: list[str],
) -> dict[str, Any]:
    """Explain why a lane is trustable without authorizing execution."""
    risk_ready = bool(risk_names) and max_risk != "UNKNOWN"
    if approval_required:
        boundary_label = "approval boundary"
        boundary_detail = f"{max_risk} lane stops for owner approval"
    else:
        boundary_label = "auto-run boundary"
        boundary_detail = f"{max_risk} lane is read-only/local-safe"

    checks = [
        {
            "key": "registered_tools",
            "label": "registered tools",
            "ready": bool(tool_coverage_ready),
            "detail": _scrub_or(tool_coverage, "unknown", 80),
        },
        {
            "key": "aggregate_smoke",
            "label": "aggregate smoke coverage",
            "ready": bool(smoke_coverage_ready),
            "detail": _scrub_or(smoke_coverage, "unknown", 80),
        },
        {
            "key": "risk_boundary",
            "label": boundary_label,
            "ready": risk_ready,
            "detail": _scrub_or(boundary_detail, "risk unknown", 120),
        },
        {
            "key": "next_command",
            "label": "next command",
            "ready": bool(_command_text_or_empty(next_command, 120)),
            "detail": _scrub_or(next_command_kind, "example", 80),
        },
    ]
    if proof_points:
        checks.append(
            {
                "key": "explicit_proofs",
                "label": "explicit proof points",
                "ready": True,
                "detail": f"{len(proof_points)} proof point(s)",
            }
        )
    ready_count = sum(1 for check in checks if check.get("ready") is True)
    return {
        "trust_checklist": checks,
        "trust_check_count": len(checks),
        "trust_ready_count": ready_count,
        "trust_ready": ready_count == len(checks),
        "trust_summary": f"{ready_count}/{len(checks)} checks ready",
        "trust_non_authorizing": True,
    }


def _trust_summary(lanes: list[dict[str, Any]]) -> dict[str, Any]:
    lane_count = len(lanes)
    ready_count = sum(1 for lane in lanes if _exact_bool(lane.get("trust_ready")))
    total_checks = sum(_safe_non_negative_int(lane.get("trust_check_count")) or 0 for lane in lanes)
    ready_checks = sum(_safe_non_negative_int(lane.get("trust_ready_count")) or 0 for lane in lanes)
    partial_lanes = [
        _scrub_or(lane.get("key"), "<unreadable>", 60)
        for lane in lanes
        if not _exact_bool(lane.get("trust_ready"))
    ]
    return {
        "lane_count": lane_count,
        "trust_ready_lane_count": ready_count,
        "trust_partial_lane_count": len(partial_lanes),
        "trust_partial_lanes": partial_lanes,
        "trust_check_count": total_checks,
        "trust_ready_check_count": ready_checks,
        "trust_ready": lane_count > 0 and ready_count == lane_count,
        "trust_non_authorizing": True,
        "summary": f"trust checklist {ready_count}/{lane_count} lanes, {ready_checks}/{total_checks} checks ready",
    }


def _normalize_failure_kind(value: Any, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata if isinstance(metadata, dict) else {}
    raw = _scrub(value, 40)
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    approval_gate_aliases = {
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
    if normalized in approval_gate_aliases:
        return "approval_required"
    if "approval" in normalized and any(token in normalized for token in ("required", "requires", "gate", "gated")):
        return "approval_required"
    if any(_truthy_metadata(metadata.get(key)) for key in ("requires_confirmation", "requires_approval", "approval_required")):
        return "approval_required"
    return raw


def _row_payload(row: Any) -> dict[str, Any] | None:
    """Guarded audit-row conversion; malformed rows become None, never leak."""
    try:
        ok = _safe_ok(_row_optional_value(row, "ok"))
        if ok is None:
            return None
        tool_name = _scrub(_row_optional_value(row, "tool_name"), 120)
        if not tool_name or tool_name in {"<local-path>", "<unreadable>"}:
            return None
        payload = {
            "id": _safe_positive_int(_row_optional_value(row, "id")),
            "tool_name": tool_name,
            "ok": ok,
            "created_at": _scrub(_row_optional_value(row, "created_at"), 80),
            "approval_id": _safe_positive_int(_row_optional_value(row, "approval_id")),
        }
    except Exception:
        return None
    try:
        raw_metadata = _row_optional_value(row, "metadata")
        metadata = json.loads("{}" if raw_metadata is None else str(raw_metadata))
        if isinstance(metadata, dict):
            kind = metadata.get("failure_kind") or metadata.get("failure_stage") or ""
            payload["failure_kind"] = _normalize_failure_kind(kind, metadata) if kind else _normalize_failure_kind("", metadata)
        else:
            payload["failure_kind"] = ""
    except Exception:
        payload["failure_kind"] = ""
    return payload


def make_capability_cockpit_tool(store: Any, list_tools: Callable[[], list[Any]]):
    def capability_cockpit(args: dict[str, Any]) -> ToolResult:
        registered, unreadable_registry_tools = _tool_registry_snapshot(list_tools)
        try:
            raw_rows = store.recent_tool_runs(limit=_AUDIT_SCAN_LIMIT)
        except Exception:
            raw_rows = []
        runs: list[dict[str, Any]] = []
        unreadable_rows = 0
        for row in raw_rows:
            payload = _row_payload(row)
            if payload is None:
                unreadable_rows += 1
            else:
                runs.append(payload)
        pending_approval_cache: dict[int, bool] = {}

        lanes: list[dict[str, Any]] = []
        attention_lanes: list[str] = []
        registered_smokes = _registered_smoke_modules()
        for spec in CAPABILITY_LANES:
            tool_names, hidden_tool_metadata = _metadata_name_list(spec.get("tools"), 80)
            present = [name for name in tool_names if name in registered]
            missing = [name for name in tool_names if name not in registered]
            present_labels = [_scrub(name, 40) for name in present]
            missing_labels = [_scrub(name, 40) for name in missing]
            if missing and present:
                tool_coverage = "partial"
            elif missing:
                tool_coverage = "missing"
            elif present:
                tool_coverage = "registered"
            else:
                tool_coverage = "empty"
            tool_coverage_ready = tool_coverage == "registered"
            risk_names: set[str] = set()
            unreadable_risk_labels: list[str] = []
            for name in present:
                risk_name = _tool_risk_name(registered[name])
                risk_names.add(risk_name)
                if risk_name == "UNKNOWN":
                    unreadable_risk_labels.append(_scrub(name, 40))
            approval_required = bool(risk_names - _AUTO_RUN_RISK_NAMES)
            max_risk = _max_risk_name(risk_names)
            smoke = _smoke_coverage(_metadata_sequence(spec.get("smoke")), registered_smokes)
            guardrail_notes = _bounded_text_list(spec.get("notes"), 140)
            proof_points = _bounded_text_list(spec.get("proof_points"), 160)
            attention_reasons: list[str] = []
            if tool_coverage == "partial":
                attention_reasons.append(f"missing tools: {', '.join(missing_labels)}")
            elif tool_coverage == "missing":
                attention_reasons.append("all tools missing")
            elif tool_coverage == "empty":
                attention_reasons.append("no tools configured")
            if unreadable_risk_labels:
                attention_reasons.append(f"unreadable tool risk: {', '.join(unreadable_risk_labels)}")
            if hidden_tool_metadata:
                attention_reasons.append(f"unreadable tool metadata: {hidden_tool_metadata}")
            if not smoke["smoke_coverage_ready"]:
                missing_smokes = list(smoke["missing_smoke_modules"])
                if smoke["smoke_coverage"] == "uncovered":
                    attention_reasons.append("no aggregate smoke coverage")
                elif missing_smokes:
                    attention_reasons.append(f"unregistered smoke: {', '.join(missing_smokes)}")
                else:
                    attention_reasons.append(f"smoke coverage {smoke['smoke_coverage']}")
            hidden_smokes = _safe_non_negative_int(smoke.get("hidden_smoke_modules")) or 0
            if hidden_smokes:
                attention_reasons.append(f"unreadable smoke metadata: {hidden_smokes}")

            lane_tools = set(present)
            last_success = ""
            last_failure = ""
            last_failure_kind = ""
            last_approval_hold = ""
            last_approval_id: int | None = None
            approval_next_commands: list[str] = []
            latest_event_ok: bool | None = None
            latest_failure_kind = ""
            for run in runs:  # newest first
                if run["tool_name"] not in lane_tools:
                    continue
                approval_gate = not run["ok"] and run["failure_kind"] == "approval_required"
                if approval_gate and not _approval_is_currently_pending(
                    store,
                    run.get("approval_id"),
                    pending_approval_cache,
                ):
                    # A held audit row is historical once its durable approval
                    # is approved, dismissed, missing, or unreadable.
                    continue
                if latest_event_ok is None:
                    latest_event_ok = run["ok"]
                    latest_failure_kind = run["failure_kind"]
                if approval_gate and not last_approval_hold:
                    last_approval_hold = f"{run['tool_name']} @ {run['created_at']}"
                    last_approval_id = run.get("approval_id")
                    approval_next_commands = _approval_review_commands(last_approval_id)
                if run["ok"] and not last_success:
                    last_success = f"{run['tool_name']} @ {run['created_at']}"
                elif not run["ok"] and run["failure_kind"] != "approval_required" and not last_failure:
                    last_failure = f"{run['tool_name']} @ {run['created_at']}"
                    last_failure_kind = run["failure_kind"]
                if last_success and last_failure:
                    break

            if missing and not present:
                status = "missing"
            elif latest_event_ok is None:
                status = "ready"
            elif latest_event_ok:
                status = "ok"
            elif latest_failure_kind == "approval_required":
                # Stopping at the approval gate is the safety design working,
                # not a malfunction — surface it as its own state.
                status = "awaiting approval"
            else:
                status = "attention"
            if latest_event_ok is False and latest_failure_kind != "approval_required":
                detail = latest_failure_kind or "tool run failed"
                attention_reasons.append(f"latest failure: {detail}")
            if unreadable_risk_labels and status != "missing":
                status = "attention"
            if status != "missing" and not tool_coverage_ready:
                status = "attention"
            if status in {"ready", "ok"} and hidden_tool_metadata:
                status = "attention"
            if status in {"ready", "ok"} and not smoke["smoke_coverage_ready"]:
                status = "attention"
            if status in {"ready", "ok"} and hidden_smokes:
                status = "attention"
            if status not in {"attention", "missing"}:
                attention_reasons = []
            if status in {"attention", "missing"}:
                attention_lanes.append(_scrub_or(spec.get("key"), "<unreadable>", 60))

            next_command, next_command_kind = _lane_next_command(spec, status, approval_next_commands)
            trust_metadata = _lane_trust_checklist(
                tool_coverage_ready=tool_coverage_ready,
                tool_coverage=tool_coverage,
                smoke_coverage_ready=bool(smoke["smoke_coverage_ready"]),
                smoke_coverage=str(smoke["smoke_coverage"]),
                risk_names=risk_names,
                max_risk=max_risk,
                approval_required=approval_required,
                next_command=next_command,
                next_command_kind=next_command_kind,
                proof_points=proof_points,
            )
            lanes.append(
                {
                    "key": _scrub_or(spec.get("key"), "<unreadable>", 60),
                    "title": _scrub_or(spec.get("title"), _scrub_or(spec.get("key"), "<unreadable>", 80), 80),
                    "status": status,
                    "risk": max_risk,
                    "approval_required": approval_required,
                    "tool_count": len(present),
                    "tool_coverage": tool_coverage,
                    "tool_coverage_ready": tool_coverage_ready,
                    "registered_tools": present_labels,
                    "missing_tools": missing_labels,
                    "hidden_tool_metadata": hidden_tool_metadata,
                    "attention_reasons": attention_reasons,
                    "guardrail_notes": guardrail_notes,
                    "proof_points": proof_points,
                    "last_success": _scrub(last_success),
                    "last_failure": _scrub(last_failure),
                    "last_failure_kind": last_failure_kind,
                    "last_approval_hold": _scrub(last_approval_hold),
                    "last_approval_id": last_approval_id,
                    "approval_next_commands": approval_next_commands,
                    "next_command": next_command,
                    "next_command_kind": next_command_kind,
                    "example_command": _scrub(spec.get("example"), 120),
                    "smoke_modules": list(smoke["registered_smoke_modules"]) + list(smoke["missing_smoke_modules"]),
                    "smoke_coverage": smoke["smoke_coverage"],
                    "smoke_coverage_ready": smoke["smoke_coverage_ready"],
                    "registered_smoke_modules": smoke["registered_smoke_modules"],
                    "missing_smoke_modules": smoke["missing_smoke_modules"],
                    "hidden_smoke_modules": hidden_smokes,
                    **trust_metadata,
                }
            )

        status_counts = _lane_status_counts(lanes)
        coverage_summary = _coverage_summary(lanes)
        trust_summary = _trust_summary(lanes)
        next_command_queue = _next_command_queue(lanes)
        next_commands = [entry["command"] for entry in next_command_queue]
        proof_entries = _proof_entries(lanes)
        proof_summary = proof_entries[:4]
        hidden_proof_entries = proof_entries[len(proof_summary) :]
        hidden_proof_lanes = [
            {
                "lane_key": entry["lane_key"],
                "lane_title": entry["lane_title"],
                "proof_count": entry["proof_count"],
            }
            for entry in hidden_proof_entries[:6]
        ]
        proof_lane_count = len(proof_entries)
        proof_point_count = sum(entry["proof_count"] for entry in proof_entries)

        direction_summary = _scrub(_DIRECTION_SUMMARY, 180)
        direction_principles = [_scrub(principle, 180) for principle in _DIRECTION_PRINCIPLES]
        lines = ["Capability cockpit (derived from live registry + audit trail):"]
        lines.append(f"Direction: {direction_summary}")
        for principle in direction_principles:
            lines.append(f"- Direction guardrail: {principle}")
        if status_counts:
            counts_text = ", ".join(f"{status}: {count}" for status, count in status_counts.items())
            lines.append(f"Lane status counts: {counts_text}")
        if coverage_summary.get("summary"):
            lines.append(f"Coverage summary: {coverage_summary['summary']}")
        if trust_summary.get("summary"):
            lines.append(f"Trust summary: {trust_summary['summary']}")
        if proof_point_count:
            lines.append(f"Proof coverage: {proof_lane_count} lane(s), {proof_point_count} proof point(s)")
            for entry in proof_summary:
                sample_text = "; ".join(entry["sample_proofs"])
                if sample_text:
                    lines.append(f"- {entry['lane_title']}: {sample_text}")
            if hidden_proof_entries:
                hidden_titles = ", ".join(entry["lane_title"] for entry in hidden_proof_entries[:4] if entry["lane_title"])
                if hidden_titles:
                    lines.append(
                        f"Proof summary has {len(hidden_proof_entries)} more lane(s): {hidden_titles}"
                    )
                else:
                    lines.append(f"Proof summary has {len(hidden_proof_entries)} more lane(s).")
        if next_command_queue:
            lines.append("Next command queue:")
            for entry in next_command_queue[:5]:
                lane_label = entry["lane_title"] or entry["lane_key"] or "capability"
                lines.append(f"- `{entry['command']}` ({entry['kind']}, {lane_label})")
        for lane in lanes:
            gate = "approval-gated" if lane["approval_required"] else "auto-run safe"
            lines.append(
                f"- {lane['title']}: {lane['status']} | {gate} | "
                f"tools: {lane['tool_coverage']} ({lane['tool_count']}) | "
                f"smoke: {lane['smoke_coverage']} | "
                f"trust: {lane['trust_summary']} | "
                f"try: `{lane['example_command']}` | "
                f"next: `{lane['next_command']}` ({lane['next_command_kind']})"
            )
            if lane["last_failure"]:
                kind = f" ({lane['last_failure_kind']})" if lane["last_failure_kind"] else ""
                lines.append(f"    last failure: {lane['last_failure']}{kind}")
            if lane["last_success"]:
                lines.append(f"    last success: {lane['last_success']}")
            if lane["last_approval_hold"]:
                commands = ", ".join(f"`{command}`" for command in lane["approval_next_commands"][:3])
                lines.append(f"    approval hold: {lane['last_approval_hold']} | next: {commands}")
            if lane["attention_reasons"]:
                lines.append(f"    attention: {'; '.join(lane['attention_reasons'])}")
            if lane["proof_points"]:
                lines.append(f"    proofs: {'; '.join(lane['proof_points'])}")
            if lane["guardrail_notes"]:
                lines.append(f"    guardrail: {'; '.join(lane['guardrail_notes'])}")
        if unreadable_rows:
            lines.append(f"Note: {unreadable_rows} unreadable audit row(s) hidden for safety.")

        return ToolResult(
            "capability_cockpit",
            True,
            "\n".join(lines),
            {
                "capability_lanes": lanes,
                "direction_summary": direction_summary,
                "direction_principles": direction_principles,
                "direction_rejects_companion_roster": True,
                "direction_exposes_internal_workers_as_capacity": True,
                "lane_count": len(lanes),
                "status_counts": status_counts,
                "coverage_summary": coverage_summary,
                "coverage_ready": coverage_summary["coverage_ready"],
                "earned_trust_contract": [_scrub(item, 180) for item in _EARNED_TRUST_CONTRACT],
                "trust_summary": trust_summary,
                "trust_ready": trust_summary["trust_ready"],
                "trust_ready_lane_count": trust_summary["trust_ready_lane_count"],
                "trust_partial_lane_count": trust_summary["trust_partial_lane_count"],
                "trust_ready_check_count": trust_summary["trust_ready_check_count"],
                "trust_check_count": trust_summary["trust_check_count"],
                "trust_non_authorizing": True,
                "tool_coverage_ready_lane_count": coverage_summary["tool_ready_count"],
                "smoke_coverage_ready_lane_count": coverage_summary["smoke_ready_count"],
                "approval_required_lane_count": coverage_summary["approval_required_count"],
                "auto_run_safe_lane_count": coverage_summary["auto_run_safe_count"],
                "attention_lanes": attention_lanes,
                "attention_lane_count": len(attention_lanes),
                "proof_lane_count": proof_lane_count,
                "proof_point_count": proof_point_count,
                "proof_summary": proof_summary,
                "proof_summary_limit": len(proof_summary),
                "proof_summary_hidden_lane_count": len(hidden_proof_entries),
                "proof_summary_hidden_lanes": hidden_proof_lanes,
                "next_command": next_commands[0] if next_commands else "",
                "next_commands": next_commands,
                "next_command_count": len(next_commands),
                "next_command_queue": next_command_queue,
                "unreadable_registry_tools": unreadable_registry_tools,
                "readable_tool_run_rows": len(runs),
                "unreadable_tool_run_rows": unreadable_rows,
                "calls_model": False,
                "executes_tools": False,
                "queues_approval": False,
                "requires_approval": False,
                "external_side_effect": False,
                "reads_personal_data": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
        )

    return capability_cockpit
