from __future__ import annotations

from collections import Counter
import re
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.store import MemoryStore


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


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "calls_model": False,
        "executes_tools": False,
        "calls_external_service": False,
        "queues_approval": False,
        "requires_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "controls_computer": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "speaks": False,
        "reads_clipboard": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def make_safety_tools(store: MemoryStore, list_tools: Callable[[], list[Any]]):
    def safety_status(_: dict[str, Any]) -> ToolResult:
        tools = list_tools()
        counts = Counter(tool.risk.name for tool in tools)
        raw_pending = store.list_pending_approvals(limit=10)
        pending, unreadable_pending_rows = _safe_approval_rows(raw_pending)
        high_risk = [tool for tool in tools if tool.risk.name in {"HIGH_RISK", "PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT"}]

        lines = [
            "Jarvis safety status:",
            "- auto-run ceiling: READ_ONLY and LOCAL_SAFE only",
            "- explicit approval required for: PERSONAL_DATA, EXTERNAL_SIDE_EFFECT, HIGH_RISK",
            "- computer control: disabled by default; enabling it is HIGH_RISK",
            "- shell commands and file writes: HIGH_RISK",
            "- brain-dump organizer: LOCAL_SAFE; writes only Jarvis SQLite/Obsidian records",
            "",
            "Tool risk inventory:",
        ]
        for name in ["READ_ONLY", "LOCAL_SAFE", "PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"]:
            lines.append(f"- {name}: {counts.get(name, 0)}")

        lines.extend(["", "Approval queue:"])
        if pending:
            for row in pending:
                lines.append(f"- #{row['id']} {row['tool_name']}: {row['user_input']}")
                lines.append(f"  Readiness before last look: `approval readiness {row['id']}`")
                lines.append(f"  Last look before approval: `approval packet {row['id']}`")
                lines.append(f"  Chain proof before trust: `approval chain proof {row['id']}`")
        if unreadable_pending_rows:
            lines.append(
                f"- {unreadable_pending_rows} pending approval row(s) hidden for safety; inspect with `pending approvals`."
            )
        if not pending and not unreadable_pending_rows:
            lines.append("- No pending approvals.")

        lines.extend(["", "Risk-gated tools:"])
        for tool in high_risk[:20]:
            lines.append(f"- {tool.name} [{tool.risk.name}]: {tool.description}")
        if len(high_risk) > 20:
            lines.append(f"- ...and {len(high_risk) - 20} more")

        return ToolResult(
            "safety_status",
            True,
            "\n".join(lines),
            _safe_metadata(
                pending_approvals=len(raw_pending),
                readable_pending_approvals=len(pending),
                unreadable_pending_approval_rows=unreadable_pending_rows,
                read_only_tools=counts.get("READ_ONLY", 0),
                local_safe_tools=counts.get("LOCAL_SAFE", 0),
                risk_gated_tools=len(high_risk),
            ),
        )

    return safety_status


def make_frozen_routing_risk_tool():
    def frozen_routing_risk_report(_: dict[str, Any]) -> ToolResult:
        lines = [
            "Jarvis frozen routing risk report:",
            "- status: OPEN / deliberately not fixed while the live-proof freeze is active",
            "- finding: unbounded lazy regex captures in the frozen send/call routing block can make the planner slow on long inputs that never satisfy the required service phrase.",
            "- affected surface: frozen send/call routing block in jarvis_v2/agent/planner.py, including the KakaoTalk, Telegram, and iMessage trailing-service send forms plus same-shape frozen send/call patterns noted in CODEX_TASKS.",
            "- observed shape: harmless-looking long text can spend seconds in routing before any tool executes; no send/call is executed by this report.",
            "- known comparison: the sibling Instagram pattern was reported as bounded and not affected by the same probe.",
            "- current boundary: do not edit the frozen send/call routing block until the operator posts live channel results or explicitly overrides the freeze.",
            "- next safe decision: either keep waiting for live channel proof results, or the operator can explicitly authorize a narrow regex-safety fix in the frozen block.",
            "- next safe command: `check codex tasks` or `frozen routing risk report`.",
            "",
            "This report is read-only: it does not inspect source files at runtime, run timing probes, edit planner routing, restart daemons, send messages/calls, approve anything, or claim the frozen risk fixed.",
        ]
        return ToolResult(
            "frozen_routing_risk_report",
            True,
            "\n".join(lines),
            _safe_metadata(
                finding_status="open",
                freeze_active=True,
                frozen_files_unchanged=True,
                affected_area="planner send/call routing block",
                affected_connectors=["kakao", "telegram", "imessage"],
                affected_scope="frozen send/call routing block same-shape patterns",
                known_unaffected_comparison=["instagram"],
                next_safe_command="frozen routing risk report",
                requires_operator_decision=True,
                inspected_source_at_runtime=False,
                ran_timing_probe=False,
                edited_frozen_code=False,
                sends_messages=False,
                makes_calls=False,
            ),
        )

    return frozen_routing_risk_report


def make_planner_input_guard_report_tool():
    def planner_input_guard_report(_: dict[str, Any]) -> ToolResult:
        lines = [
            "Jarvis planner input guard report:",
            "- status: DESIGN DECISION OPEN / no entry-point input-length guard has been implemented",
            "- why this exists: the frozen send/call routing ReDoS and later regex-hardening notes show that very long non-matching inputs can spend too long in planner matching before any tool executes.",
            "- possible mitigation: add a total command-length guard at the start of RuleBasedPlanner.plan() so oversized commands stop before any regex matcher, including the frozen send/call block.",
            "- why it is pending: a global input cap is a broad behavior change; it affects long research prompts, long notes, writing requests, pasted text, and future tool payloads, not only hostile inputs.",
            "- current boundary: do not apply a planner-wide cap autonomously while the live-proof freeze is active; the operator should choose or approve the limit and fallback behavior.",
            "- relationship to frozen risk: this would reduce worst-case planner exposure without editing the frozen send/call block, but it still does not replace a later targeted regex fix.",
            "- next safe decision: pick an input-length policy and fallback message, or keep waiting until the frozen send/call proof work is complete.",
            "- next safe command: `planner input guard report` or `frozen routing risk report`.",
            "",
            "This report is read-only: it does not inspect source files at runtime, run timing probes, edit planner routing, add an input cap, restart daemons, send messages/calls, approve anything, or claim the ReDoS fixed.",
        ]
        return ToolResult(
            "planner_input_guard_report",
            True,
            "\n".join(lines),
            _safe_metadata(
                design_status="open",
                input_length_guard_enabled=False,
                planner_wide_behavior_changed=False,
                freeze_active=True,
                requires_operator_decision=True,
                proposed_mitigation="entry_point_total_input_length_guard",
                affects_long_legitimate_inputs=True,
                related_report_command="frozen routing risk report",
                next_safe_command="planner input guard report",
                inspected_source_at_runtime=False,
                ran_timing_probe=False,
                edited_planner=False,
                added_input_cap=False,
                sends_messages=False,
                makes_calls=False,
            ),
        )

    return planner_input_guard_report
