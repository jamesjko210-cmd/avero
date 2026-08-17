from __future__ import annotations

from typing import Any

from jarvis_v2.agent.types import ToolResult


ARCHITECTURE_LAYERS = [
    (
        "1. Perception and input handling",
        "Text chat is implemented through the CLI/runtime; voice output exists through macOS speech. Wake-word and ASR are future migrations from the old Jarvis code.",
        "chat loop, planner routing, voice output",
    ),
    (
        "2. Natural language understanding",
        "Deterministic routing handles known commands, optional model planning handles uncaught tool requests, and chat mode handles natural conversation.",
        "RuleBasedPlanner, ModelBackedPlanner, ChatBrain",
    ),
    (
        "3. Reasoning and planning",
        "Jarvis separates planning from execution, can draft safe autonomy plans, and keeps risky actions behind approval readiness, last-look approval packets, and approval chain proof.",
        "autonomy_plan, focus_brief, safe_next_actions",
    ),
    (
        "4. Memory",
        "Short-term conversation history, SQLite long-term memory, and human-readable Obsidian notes are all implemented. Preferences, people, goals, tasks, decisions, skills, sessions, and Current Context are inspectable.",
        "SQLite memory store, Obsidian Memory Tree, profile/preferences/skills",
    ),
    (
        "5. Action and tool execution",
        "Tools are registered with risk levels. Read-only and local-safe tools can run; shell, file writes, clipboard reads, computer control, reminders, deletions, and external effects require approval.",
        "ToolRegistry, PermissionPolicy, approval queue, audit log",
    ),
    (
        "6. Learning and continuous improvement",
        "Jarvis can draft reusable skills from sessions, summarize activity, produce reviews, and maintain Mission Control. Self-modifying code stays out of the autonomous loop.",
        "draft_skill_from_session, weekly_review, memory_tree_summary",
    ),
]


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
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
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def architecture_map(_: dict[str, Any]) -> ToolResult:
    lines = [
        "Jarvis V2 architecture map:",
        "",
        "Core idea: conversation first, memory second, planning third, tools fourth, verification and safety around everything.",
        "Completion idea: Jarvis should become an agent harness, not just an agent. The model is the reasoning core; the harness is the runtime that supplies durable state, tool boundaries, approvals, audit, diagnostics, recovery, and verification.",
        "",
    ]
    for title, status, examples in ARCHITECTURE_LAYERS:
        lines.extend(
            [
                title,
                f"- Status: {status}",
                f"- V2 pieces: {examples}",
                "",
            ]
        )

    lines.extend(
        [
            "Harness completion gates:",
            "- Stable chat loop: text and speech commands route through the same planner and safety spine.",
            "- State and memory: every long-running task has resumable context, notes, and a current-status view.",
            "- Tool boundary: all actions pass through ToolRegistry, PermissionPolicy, approvals, and audit.",
            "- Recovery: failures degrade to preview, plan, or safe chat instead of hidden retries or silent action.",
            "- Verification: Jarvis reports observed result, confidence, and next safe step after acting.",
            "",
            "Safety spine:",
            "- Every tool has a risk level.",
            "- Personal data, computer control, shell/code, destructive actions, reminders, and external effects require explicit approval.",
            "- Blocked actions create safety receipts and stay visible in `pending approvals` until approval readiness, a last-look approval packet, and approval chain proof are reviewed.",
            "- Jarvis should verify and report what happened instead of pretending.",
            "",
            "Useful commands:",
            "- `capability map` for the live tool inventory",
            "- `safety status` for risk gates and pending approvals",
            "- `focus brief` to start a work session",
            "- `autonomy plan: ...` before risky multi-step work",
            "- `return brief` when coming back after time away",
            "",
            "Next architecture gaps:",
            "- migrate ASR/wake-word into V2 with explicit microphone/privacy boundaries",
            "- migrate calendar/email/message tools as approval-gated personal integrations",
            "- add stronger vision understanding to observe-act-verify loops",
            "- improve model routing so planning, chat, compression, and vision can use different models",
        ]
    )
    return ToolResult(
        "architecture_map",
        True,
        "\n".join(lines),
        _safe_metadata(layers=len(ARCHITECTURE_LAYERS), harness_gates=5),
    )
