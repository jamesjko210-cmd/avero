from __future__ import annotations

import re
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.store import MemoryStore


PERSONAL_TOOLSETS = {"computer", "personal", "system", "browser", "files", "code", "ingest", "notes"}
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)


def _safe_preview(value: Any, *, limit: int = 240) -> str:
    try:
        text = " ".join(str(value or "").strip().split())
    except Exception:
        return "<unreadable>"
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _safe_approval_rows(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        try:
            payloads.append(
                {
                    "id": int(row["id"]),
                    "tool_name": _safe_preview(row["tool_name"], limit=80),
                    "user_input": _safe_preview(row["user_input"]),
                }
            )
        except Exception:
            unreadable += 1
    return payloads, unreadable


def _safe_configured_path_display(label: str, index: int | None = None) -> str:
    suffix = f" #{index}" if index is not None else ""
    return f"{label}{suffix} configured"


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "approves_request": False,
        "dismisses_request": False,
        "controls_computer": False,
        "reads_private_data": False,
        "writes_files": False,
        "writes_notes": False,
        "writes_memory": False,
        "external_side_effect": False,
        "requires_approval": False,
        "speaks": False,
    }
    metadata.update(extra)
    return metadata


def make_privacy_tools(store: MemoryStore, config: JarvisConfig | None, list_tools: Callable[[], list[Any]]):
    def privacy_report(_: dict[str, Any]) -> ToolResult:
        tools = list_tools()
        raw_pending = store.list_pending_approvals(limit=8)
        pending, unreadable_pending_rows = _safe_approval_rows(raw_pending)
        watched_dirs = config.watched_dirs if config is not None else ()
        obsidian_root = config.obsidian_root if config is not None else None
        gated_v2_connectors = ["calendar", "email", "messages", "contacts"]
        still_blocked_legacy_scope = [
            "broad calendar/email/messages account access beyond the current gated V2 commands",
            "logged-in browser state, cookies, history, and authenticated page actions",
            "full contact-card export or contact create/edit/delete",
            "wake-word/ASR microphone input",
        ]

        lines = [
            "Jarvis privacy boundary report:",
            "",
            "Default rule:",
            "- Jarvis may use read-only and local-safe context to help you.",
            "- Jarvis must ask before reading personal data, controlling the computer, running shell/code, deleting/editing important state, creating reminders, or causing outside-world effects.",
            "",
            "Local knowledge Jarvis may use by default:",
            "- Jarvis-owned SQLite memory, tasks, goals, decisions, preferences, people, skills, sessions, and audit metadata.",
            "- Jarvis-owned Obsidian folder: "
            f"{_safe_configured_path_display('Obsidian root') if obsidian_root else 'configured vault unavailable'}",
            "- Metadata-only watched-file digests, if configured.",
        ]
        if watched_dirs:
            for index, _path in enumerate(watched_dirs, start=1):
                lines.append(f"  - watched metadata path: {_safe_configured_path_display('watched directory', index)}")
        else:
            lines.append("  - no watched directories configured.")

        lines.extend(
            [
                "",
                "Approval-gated privacy surfaces:",
                "- Clipboard reads: PERSONAL_DATA.",
                "- Focused and visible running application activity: PERSONAL_DATA.",
                "- Screenshots, observe-screen, clicks, mouse moves, and typing: PERSONAL_DATA or HIGH_RISK.",
                "- Shell/code execution and file writes/deletes: HIGH_RISK.",
                "- Reminders and future calendar/email/messages: HIGH_RISK or EXTERNAL_SIDE_EFFECT.",
                "- Browser opening is local-safe; fetched pages and search are read-only, but saved page notes write only into Jarvis-owned Obsidian.",
                "",
                "Risk-gated tools touching private or important surfaces:",
            ]
        )
        risk_order = {"PERSONAL_DATA": 0, "EXTERNAL_SIDE_EFFECT": 1, "HIGH_RISK": 2}
        gated = sorted(
            [
                tool
                for tool in tools
                if tool.risk.name in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
            ],
            key=lambda tool: (risk_order.get(tool.risk.name, 99), tool.toolset, tool.name),
        )
        for tool in gated[:20]:
            lines.append(f"- {tool.name} [{tool.toolset}, {tool.risk.name}]: {tool.description}")
        if len(gated) > 20:
            lines.append(f"- ...and {len(gated) - 20} more approval-gated tool(s)")

        lines.extend(["", "Pending privacy approvals:"])
        if pending:
            for approval in pending:
                lines.append(f"- #{approval['id']} {approval['tool_name']}: {approval['user_input']}")
        if unreadable_pending_rows:
            lines.append(
                f"- {unreadable_pending_rows} pending approval row(s) hidden for safety; inspect with `pending approvals`."
            )
        if not pending and not unreadable_pending_rows:
            lines.append("- No pending approvals.")

        lines.extend(["", "Current connector privacy posture:"])
        for connector in gated_v2_connectors:
            lines.append(f"- {connector}: active through narrow V2 commands, with personal reads and side effects still approval-gated.")
        lines.extend(["", "Still blocked legacy scope:"])
        for item in still_blocked_legacy_scope:
            lines.append(f"- {item}.")
        lines.append("- Future migrations must keep explicit per-source privacy boundaries, route locks, focused smokes, and aggregate smoke proof.")

        return ToolResult(
            "privacy_report",
            True,
            "\n".join(lines),
            _safe_metadata(
                pending_approvals=len(raw_pending),
                readable_pending_approvals=len(pending),
                unreadable_pending_approval_rows=unreadable_pending_rows,
                risk_gated_tools=len(gated),
                watched_dirs=len(watched_dirs),
                displayed_risk_gated_tools=min(len(gated), 20),
                gated_v2_connector_names=gated_v2_connectors,
                gated_v2_connector_count=len(gated_v2_connectors),
                still_blocked_legacy_scope=still_blocked_legacy_scope,
                still_blocked_legacy_scope_count=len(still_blocked_legacy_scope),
                microphone_input_active=False,
            ),
        )

    return privacy_report
