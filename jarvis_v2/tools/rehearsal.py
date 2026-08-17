from __future__ import annotations

from collections import Counter
import re
from typing import Any, Callable

from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore


APPROVAL_RISKS = {
    RiskLevel.PERSONAL_DATA,
    RiskLevel.EXTERNAL_SIDE_EFFECT,
    RiskLevel.HIGH_RISK,
}

MAX_REHEARSAL_TEXT_CHARS = 500
MAX_ARG_DISPLAY_CHARS = 260
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")


def _metadata_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _learning_debt_blocks_rehearsal(execution_learning_debt: dict[str, Any] | None) -> bool:
    return (
        execution_learning_debt
        and _metadata_bool(execution_learning_debt.get("blocks_completion_claim"))
        and _metadata_int(execution_learning_debt.get("failed_or_blocked_action_runs")) > 0
    )


def _is_approval_held_review(recovery_closure: dict[str, Any] | None) -> bool:
    return bool(recovery_closure and recovery_closure.get("state") == "approval_held_review_required")


def _approval_held_review_commands(recovery_closure: dict[str, Any] | None) -> list[str]:
    if not _is_approval_held_review(recovery_closure):
        return []
    return list(
        recovery_closure.get("approval_review_commands")
        or recovery_closure.get("required_commands")
        or []
    )


def _short(value: Any, *, limit: int = MAX_REHEARSAL_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_short(value: Any, *, limit: int = MAX_REHEARSAL_TEXT_CHARS) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit=limit))


def _short_args(args: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _safe_short(value, limit=MAX_ARG_DISPLAY_CHARS) for key, value in args.items()}


def _turn_route_metadata(
    *,
    request: str,
    mode: str,
    approval_required: bool,
    planned_actions: list[dict[str, Any]],
    recovery_closure: dict[str, Any] | None = None,
    execution_learning_debt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    read_only_actions = bool(planned_actions) and all(action.get("risk") == "READ_ONLY" for action in planned_actions)
    if _is_approval_held_review(recovery_closure) and mode != "chat" and not read_only_actions:
        route = "approval_held_review"
        recommendation = "APPROVAL_HELD_REVIEW_REQUIRED"
        recommended_next_commands = _approval_held_review_commands(recovery_closure) or ["pending approvals"]
    elif (
        recovery_closure
        and _metadata_bool(recovery_closure.get("blocks_auto_execution"))
        and mode != "chat"
        and not read_only_actions
    ):
        route = "recovery_closure"
        recommendation = "RECOVERY_CLOSURE_REQUIRED"
        recommended_next_commands = list(recovery_closure.get("required_commands") or []) or ["execution health report"]
    elif _learning_debt_blocks_rehearsal(execution_learning_debt) and mode != "chat" and not read_only_actions:
        route = "execution_learning_debt"
        recommendation = "EXECUTION_LEARNING_REQUIRED"
        recommended_next_commands = list(execution_learning_debt.get("required_commands") or []) or ["after-action learning packet"]
    elif mode == "chat":
        route = "chat"
        recommendation = "ANSWER_IN_CHAT"
        recommended_next_commands = ["send this message normally"]
    elif approval_required:
        route = "approval"
        recommendation = "ENTER_EXECUTION_GOVERNOR"
        recommended_next_commands = [
            f"execution governor: {request}",
            "send this command normally to queue an approval receipt",
            "approval readiness latest",
            "approval packet latest",
            "approval chain proof latest",
            "pending approvals",
        ]
    elif planned_actions:
        route = "auto_tool"
        recommendation = "AUTO_RUN_LOCAL_SAFE"
        recommended_next_commands = [f"execution governor: {request}", "send this command normally"]
    else:
        route = "none"
        recommendation = "ASK_FOR_MORE_DETAIL"
        recommended_next_commands = ["add more detail, or ask for `jarvis help`"]
    return {
        "route": route,
        "recommendation": recommendation,
        "recommended_next_commands": recommended_next_commands,
        "next_command": recommended_next_commands[0],
        "safe_to_execute_now": route in {"chat", "auto_tool"},
    }


def _rehearsal_handoff(
    *,
    source: str,
    text: str,
    mode: str,
    route_metadata: dict[str, Any],
    planned_actions: list[dict[str, Any]],
    risk_counts: Counter[str],
    approval_required: bool,
    recovery_closure: dict[str, Any] | None,
    execution_learning_debt: dict[str, Any] | None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recovery_closure = recovery_closure or {}
    execution_learning_debt = execution_learning_debt or {}
    recommended_next_commands = list(route_metadata.get("recommended_next_commands") or [])
    recovery_proof_queue = list(recovery_closure.get("required_commands") or [])
    approval_held_commands = _approval_held_review_commands(recovery_closure)
    learning_proof_queue = list(execution_learning_debt.get("required_commands") or [])
    handoff = {
        "source": source,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "text_length": len(text),
        "mode": mode,
        "route": route_metadata.get("route") or "none",
        "recommendation": route_metadata.get("recommendation") or "",
        "next_command": route_metadata.get("next_command") or "",
        "recommended_next_commands": recommended_next_commands,
        "recommended_next_command_count": len(recommended_next_commands),
        "next_safe_command": route_metadata.get("next_command") or "",
        "next_safe_commands": recommended_next_commands,
        "next_safe_command_count": len(recommended_next_commands),
        "safe_to_execute_now": _metadata_bool(route_metadata.get("safe_to_execute_now")),
        "planned_action_count": len(planned_actions),
        "planned_tools": [str(action.get("tool") or "") for action in planned_actions],
        "risk_counts": dict(risk_counts),
        "approval_required": bool(approval_required),
        "recovery_closure_state": recovery_closure.get("state") or "not_available",
        "recovery_closure_blocks_auto_execution": _metadata_bool(recovery_closure.get("blocks_auto_execution")),
        "recovery_closure_proof_queue": recovery_proof_queue,
        "recovery_closure_proof_queue_count": len(recovery_proof_queue),
        "recovery_closure_next_proof_command": recovery_closure.get("next_required_command") or "",
        "approval_held_review_required": _is_approval_held_review(recovery_closure),
        "approval_held_review_commands": approval_held_commands,
        "approval_held_review_command_count": len(approval_held_commands),
        "approval_held_review_next_command": approval_held_commands[0] if approval_held_commands else "",
        "approval_held_review_target_run_id": recovery_closure.get("approval_held_target_run_id") or recovery_closure.get("target_run_id"),
        "approval_held_review_target_tool_name": recovery_closure.get("approval_held_target_tool_name") or recovery_closure.get("target_tool_name") or "",
        "execution_learning_state": execution_learning_debt.get("state") or "NO_RECENT_ACTION_RUNS",
        "execution_learning_blocks_completion_claim": _metadata_bool(execution_learning_debt.get("blocks_completion_claim")),
        "execution_learning_failed_or_blocked_action_runs": _metadata_int(execution_learning_debt.get("failed_or_blocked_action_runs")),
        "execution_learning_approval_held_action_runs": _metadata_int(execution_learning_debt.get("approval_held_action_runs")),
        "execution_learning_proof_queue": learning_proof_queue,
        "execution_learning_proof_queue_count": len(learning_proof_queue),
        "execution_learning_next_proof_command": execution_learning_debt.get("next_required_command") or "",
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
    if context is not None:
        handoff["context"] = dict(context)
    return handoff


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
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
    metadata.update(extra)
    return metadata


def _recovery_closure_metadata(recovery_closure: dict[str, Any] | None) -> dict[str, Any]:
    recovery_closure = recovery_closure or {
        "state": "not_available",
        "ready_to_retry": True,
        "missing": [],
        "missing_count": 0,
        "required_commands": [],
        "next_required_command": "",
        "blocks_auto_execution": False,
        "target_run_id": None,
        "target_tool_name": "",
        "approval_held_action_runs": 0,
        "approval_held_target_run_id": None,
        "approval_held_target_tool_name": "",
        "approval_review_commands": [],
    }
    approval_held_commands = _approval_held_review_commands(recovery_closure)
    return {
        "recovery_closure_state": recovery_closure["state"],
        "recovery_closure_ready_to_retry": recovery_closure["ready_to_retry"],
        "recovery_closure_missing": recovery_closure["missing"],
        "recovery_closure_missing_count": recovery_closure["missing_count"],
        "recovery_closure_required_commands": recovery_closure["required_commands"],
        "recovery_closure_next_required_command": recovery_closure["next_required_command"],
        "recovery_closure_blocks_auto_execution": _metadata_bool(recovery_closure["blocks_auto_execution"]),
        "recovery_closure_target_run_id": recovery_closure["target_run_id"],
        "recovery_closure_target_tool_name": recovery_closure["target_tool_name"],
        "recovery_closure_proof_queue": recovery_closure["required_commands"],
        "recovery_closure_proof_queue_count": len(recovery_closure["required_commands"]),
        "recovery_closure_next_proof_command": recovery_closure["next_required_command"],
        "recovery_closure_approval_held_action_runs": _metadata_int(recovery_closure.get("approval_held_action_runs")),
        "approval_held_review_required": _is_approval_held_review(recovery_closure),
        "approval_held_review_commands": approval_held_commands,
        "approval_held_review_command_count": len(approval_held_commands),
        "approval_held_review_next_command": approval_held_commands[0] if approval_held_commands else "",
        "approval_held_review_target_run_id": recovery_closure.get("approval_held_target_run_id") or recovery_closure.get("target_run_id"),
        "approval_held_review_target_tool_name": recovery_closure.get("approval_held_target_tool_name") or recovery_closure.get("target_tool_name") or "",
    }


def _recovery_closure_lines(recovery_closure: dict[str, Any] | None) -> list[str]:
    if not recovery_closure or not _metadata_bool(recovery_closure["blocks_auto_execution"]):
        return []
    commands = recovery_closure["required_commands"]
    if _is_approval_held_review(recovery_closure):
        return [
            "",
            "Approval-held execution review:",
            f"- state: {recovery_closure['state']}",
            f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
            "- reason: safety gate held this before execution; review the approval chain before treating it as a failed tool run.",
            f"- next review: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next review: none",
            f"- review queue: {', '.join(f'`{command}`' for command in commands) if commands else 'none'}",
        ]
    return [
        "",
        "Execution health recovery closure:",
        f"- state: {recovery_closure['state']}",
        f"- ready to retry: {'yes' if recovery_closure['ready_to_retry'] else 'no'}",
        f"- target run: #{recovery_closure['target_run_id']} `{recovery_closure['target_tool_name']}`" if recovery_closure["target_run_id"] is not None else "- target run: none",
        f"- missing: {', '.join(recovery_closure['missing']) if recovery_closure['missing'] else 'none'}",
        f"- next required: `{recovery_closure['next_required_command']}`" if recovery_closure["next_required_command"] else "- next required: none",
        f"- proof queue: {', '.join(f'`{command}`' for command in commands) if commands else 'none'}",
    ]


def _execution_learning_debt_metadata(execution_learning_debt: dict[str, Any] | None) -> dict[str, Any]:
    execution_learning_debt = execution_learning_debt or {
        "state": "NO_RECENT_ACTION_RUNS",
        "blocks_completion_claim": False,
        "recent_action_runs": 0,
        "failed_or_blocked_action_runs": 0,
        "approval_held_action_runs": 0,
        "recent_verification_runs": 0,
        "recent_recovery_runs": 0,
        "recent_after_action_learning_runs": 0,
        "target_run_id": None,
        "target_tool_name": "",
        "target_after_action_learning_packets": 0,
        "missing": [],
        "missing_count": 0,
        "required_commands": [],
        "next_required_command": "",
    }
    return {
        "execution_learning_state": execution_learning_debt["state"],
        "execution_learning_blocks_completion_claim": _metadata_bool(execution_learning_debt["blocks_completion_claim"]),
        "execution_learning_recent_action_runs": execution_learning_debt["recent_action_runs"],
        "execution_learning_failed_or_blocked_action_runs": execution_learning_debt["failed_or_blocked_action_runs"],
        "execution_learning_approval_held_action_runs": _metadata_int(execution_learning_debt.get("approval_held_action_runs")),
        "execution_learning_recent_verification_runs": execution_learning_debt["recent_verification_runs"],
        "execution_learning_recent_recovery_runs": execution_learning_debt["recent_recovery_runs"],
        "execution_learning_recent_after_action_learning_runs": execution_learning_debt["recent_after_action_learning_runs"],
        "execution_learning_target_run_id": execution_learning_debt["target_run_id"],
        "execution_learning_target_tool_name": execution_learning_debt["target_tool_name"],
        "execution_learning_target_after_action_learning_packets": execution_learning_debt["target_after_action_learning_packets"],
        "execution_learning_missing": execution_learning_debt["missing"],
        "execution_learning_missing_count": execution_learning_debt["missing_count"],
        "execution_learning_required_commands": execution_learning_debt["required_commands"],
        "execution_learning_next_required_command": execution_learning_debt["next_required_command"],
        "execution_learning_proof_queue": execution_learning_debt["required_commands"],
        "execution_learning_proof_queue_count": len(execution_learning_debt["required_commands"]),
        "execution_learning_next_proof_command": execution_learning_debt["next_required_command"],
    }


def _execution_learning_debt_lines(execution_learning_debt: dict[str, Any] | None) -> list[str]:
    if not execution_learning_debt or not _metadata_bool(execution_learning_debt["blocks_completion_claim"]):
        return []
    commands = execution_learning_debt["required_commands"]
    return [
        "",
        "Execution learning debt:",
        f"- state: {execution_learning_debt['state']}",
        f"- blocks completion claim: {'yes' if _metadata_bool(execution_learning_debt['blocks_completion_claim']) else 'no'}",
        f"- target run: #{execution_learning_debt['target_run_id']} `{execution_learning_debt['target_tool_name']}`" if execution_learning_debt["target_run_id"] is not None else "- target run: none",
        f"- missing: {', '.join(execution_learning_debt['missing']) if execution_learning_debt['missing'] else 'none'}",
        f"- next learning required: `{execution_learning_debt['next_required_command']}`" if execution_learning_debt["next_required_command"] else "- next learning required: none",
        f"- learning proof queue: {', '.join(f'`{command}`' for command in commands) if commands else 'none'}",
    ]


def _execution_health_recovery_closure(store: MemoryStore | None, *, limit: int = 12) -> dict[str, Any] | None:
    if store is None:
        return None
    from jarvis_v2.tools import autonomy

    return autonomy._execution_health_recovery_closure_snapshot(store.recent_tool_runs(limit=limit))


def _execution_learning_debt(store: MemoryStore | None, *, limit: int = 12) -> dict[str, Any] | None:
    if store is None:
        return None
    from jarvis_v2.tools import harness

    return harness._execution_learning_debt_snapshot(store.recent_tool_runs(limit=limit))


def make_rehearsal_tool(get_tool: Callable[[str], Any], store: MemoryStore | None = None):
    def action_rehearsal(args: dict[str, Any]) -> ToolResult:
        raw_request = _short(args.get("request"))
        request = _safe_short(raw_request)
        if not request:
            return ToolResult(
                "action_rehearsal",
                False,
                "Give Jarvis a request to rehearse, for example: `rehearse: run command python3 --version`.",
                _safe_metadata(),
            )

        from jarvis_v2.agent.planner import RuleBasedPlanner

        plan = RuleBasedPlanner().plan(raw_request)
        recovery_closure = _execution_health_recovery_closure(store)
        execution_learning_debt = _execution_learning_debt(store)
        risk_counts: Counter[str] = Counter()
        action_summaries: list[dict[str, Any]] = []
        mode = "chat" if plan.needs_model and not plan.actions else "tool"
        lines = [
            "Jarvis action rehearsal",
            f"Request: {request}",
            f"Planned goal: {_safe_short(plan.goal)}",
            "",
            "Execution preview:",
        ]

        if plan.needs_model and not plan.actions:
            lines.append("- Route to the chat brain for a conversational answer; no tool would run.")
        elif not plan.actions:
            lines.append("- No tool action was planned.")
        else:
            for index, action in enumerate(plan.actions, start=1):
                try:
                    tool = get_tool(action.tool_name)
                    risk_name = tool.risk.name
                    toolset = tool.toolset
                    description = tool.description
                    needs_approval = tool.risk in APPROVAL_RISKS
                    risk_counts[risk_name] += 1
                    action_summaries.append(
                        {
                            "tool": action.tool_name,
                            "risk": risk_name,
                            "toolset": toolset,
                            "requires_approval": needs_approval,
                            "args": _short_args(action.args),
                        }
                    )
                    approval_text = "approval required" if needs_approval else "would be allowed by default"
                    lines.append(
                        f"- {index}. {action.tool_name} [{toolset}, {risk_name}]: {approval_text}. {description}"
                    )
                    if action.reason:
                        lines.append(f"  Reason: {_safe_short(action.reason)}")
                    if action.args:
                        lines.append(f"  Planned args: {_short_args(action.args)}")
                except KeyError:
                    risk_counts["UNKNOWN"] += 1
                    action_summaries.append(
                        {
                            "tool": action.tool_name,
                            "risk": "UNKNOWN",
                            "toolset": "unknown",
                            "requires_approval": True,
                            "args": _short_args(action.args),
                        }
                    )
                    lines.append(f"- {index}. {action.tool_name} [unknown]: approval required because the tool is not registered.")

        approval_required = any(item["requires_approval"] for item in action_summaries)
        if plan.needs_model and plan.actions:
            lines.append("- Planner note: a model may be needed to refine this request before execution.")
        if plan.notes:
            lines.append(f"- Planner notes: {_safe_short(plan.notes)}")

        lines.extend(
            [
                "",
                "Safety result:",
                "- This rehearsal is read-only and does not execute tools, approve requests, read personal data, control the computer, write files, or call external services.",
            ]
        )
        if approval_required:
            lines.append("- If run for real, at least one planned action would require explicit approval first.")
            lines.append("- After the real request creates a safety receipt, run `approval readiness #ID`, `approval packet #ID`, and `approval chain proof #ID` before approving it.")
        else:
            lines.append("- If run for real, the planned actions are within the default auto-run risk boundary.")
        if recovery_closure and _is_approval_held_review(recovery_closure):
            lines.append("- Current blocker: approval-held execution review must be completed before new tool execution is presented as safe.")
        elif recovery_closure and _metadata_bool(recovery_closure["blocks_auto_execution"]):
            lines.append("- Current blocker: execution health recovery closure must be completed before new tool execution is presented as safe.")
        if _learning_debt_blocks_rehearsal(execution_learning_debt):
            lines.append("- Current blocker: execution learning debt must be closed before new tool execution is presented as complete.")

        route_metadata = _turn_route_metadata(
            request=request,
            mode=mode,
            approval_required=approval_required,
            planned_actions=action_summaries,
            recovery_closure=recovery_closure,
            execution_learning_debt=execution_learning_debt,
        )
        lines.extend(_recovery_closure_lines(recovery_closure))
        if mode != "chat":
            lines.extend(_execution_learning_debt_lines(execution_learning_debt))
        lines.extend(
            [
                "",
                "Next command:",
                f"- {route_metadata['next_command']}",
            ]
        )
        handoff = _rehearsal_handoff(
            source="action_rehearsal",
            text=request,
            mode=mode,
            route_metadata=route_metadata,
            planned_actions=action_summaries,
            risk_counts=risk_counts,
            approval_required=approval_required,
            recovery_closure=recovery_closure,
            execution_learning_debt=execution_learning_debt,
        )

        return ToolResult(
            "action_rehearsal",
            True,
            "\n".join(lines),
            _safe_metadata(
                action_rehearsal_handoff_ready=handoff["handoff_ready"],
                action_rehearsal_ready_for_operator=handoff["ready_for_operator"],
                action_rehearsal_state_changed=handoff["state_changed"],
                action_rehearsal_changed=handoff["changed"],
                action_rehearsal_content_in_handoff=handoff["content_in_handoff"],
                action_rehearsal_next_safe_command=handoff["next_safe_command"],
                action_rehearsal_next_safe_commands=handoff["next_safe_commands"],
                action_rehearsal_next_safe_command_count=handoff["next_safe_command_count"],
                action_rehearsal_authorizes_execution=handoff["authorizes_execution"],
                action_rehearsal_authorizes_completion_claim=handoff["authorizes_completion_claim"],
                action_rehearsal_approval_granted=handoff["approval_granted"],
                request=request,
                mode=mode,
                planned_goal=_safe_short(plan.goal),
                planned_actions=action_summaries,
                needs_model=plan.needs_model,
                approval_required=approval_required,
                risk_counts=dict(risk_counts),
                **route_metadata,
                **_recovery_closure_metadata(recovery_closure),
                **_execution_learning_debt_metadata(execution_learning_debt),
                action_rehearsal_handoff=handoff,
            ),
        )

    return action_rehearsal


def make_assistant_turn_rehearsal_tool(
    get_tool: Callable[[str], Any],
    store: MemoryStore,
    vault: ObsidianVault,
    session_id: str,
):
    def assistant_turn_rehearsal(args: dict[str, Any]) -> ToolResult:
        raw_message = _short(args.get("message") or args.get("request"))
        message = _safe_short(raw_message)
        if not message:
            return ToolResult(
                "assistant_turn_rehearsal",
                False,
                "Give Jarvis a message to rehearse, for example: `assistant turn rehearsal: can we talk about Jarvis memory?`.",
                _safe_metadata(),
            )

        from jarvis_v2.agent.planner import RuleBasedPlanner

        plan = RuleBasedPlanner().plan(raw_message)
        recovery_closure = _execution_health_recovery_closure(store)
        execution_learning_debt = _execution_learning_debt(store)
        preferences = store.list_preferences(status="active", limit=8)
        memories = store.search_memories(" ".join(_terms(raw_message)[:6]), limit=5) if _terms(raw_message) else []
        if not memories:
            memories = store.recent_memories(limit=5)
        skills = store.search_active_skills(" ".join(_terms(raw_message)[:6]), limit=3) if _terms(raw_message) else []
        recent = store.recent_messages(limit=6, session_id=session_id)
        try:
            profile_text = vault.read_profile(max_chars=900).strip()
        except Exception:
            profile_text = ""
        if profile_text == "# Profile":
            profile_text = ""

        action_summaries: list[dict[str, Any]] = []
        risk_counts: Counter[str] = Counter()
        mode = "chat" if plan.needs_model and not plan.actions else "tool"
        lines = [
            "Jarvis assistant turn rehearsal:",
            "",
            "Message:",
            f"- {message}",
            "",
            "Routing preview:",
            f"- mode: {mode}",
            f"- planned goal: {_safe_short(plan.goal)}",
        ]
        if mode == "chat":
            lines.extend(
                [
                    "- Jarvis would answer conversationally rather than execute tools.",
                    "- Chat may use visible profile notes, active preferences, relevant memories, saved skills, and recent conversation context.",
                ]
            )
        elif not plan.actions:
            lines.append("- No tool action was planned.")
        else:
            lines.append("- Jarvis would route one or more tool actions through ToolRegistry and PermissionPolicy.")
            for index, action in enumerate(plan.actions, start=1):
                try:
                    tool = get_tool(action.tool_name)
                    needs_approval = tool.risk in APPROVAL_RISKS
                    risk_counts[tool.risk.name] += 1
                    action_summaries.append(
                        {
                            "tool": action.tool_name,
                            "risk": tool.risk.name,
                            "toolset": tool.toolset,
                            "requires_approval": needs_approval,
                            "args": _short_args(action.args),
                        }
                    )
                    approval_text = "approval required" if needs_approval else "would be allowed by default"
                    lines.append(f"- {index}. {action.tool_name} [{tool.toolset}, {tool.risk.name}]: {approval_text}.")
                    if action.args:
                        lines.append(f"  Planned args: {_short_args(action.args)}")
                except KeyError:
                    risk_counts["UNKNOWN"] += 1
                    action_summaries.append(
                        {
                            "tool": action.tool_name,
                            "risk": "UNKNOWN",
                            "toolset": "unknown",
                            "requires_approval": True,
                            "args": _short_args(action.args),
                        }
                    )
                    lines.append(f"- {index}. {action.tool_name} [unknown]: approval required because the tool is not registered.")

        approval_required = any(item["requires_approval"] for item in action_summaries)
        lines.extend(
            [
                "",
                "Context preview:",
                f"- profile context: {'present' if profile_text else 'empty'}",
                f"- active preferences: {len(preferences)}",
                f"- relevant/recent memories visible: {len(memories)}",
                f"- matching skills visible: {len(skills)}",
                f"- recent conversation messages visible: {len(recent)}",
                "",
                "Safety boundary:",
                "- This rehearsal is read-only and does not call the chat model, execute tools, approve requests, read private connectors, control the computer, write files, save memory, or queue approvals.",
                "- If the message is conversational, Jarvis should answer from visible context and say when evidence is thin.",
                "- If the message is actionable, real execution still goes through planner, ToolRegistry, PermissionPolicy, approval queue, and audit log.",
            ]
        )
        if approval_required:
            lines.append("- If run for real, at least one planned action would require explicit approval first.")
            lines.append("- After the real request creates a safety receipt, run `approval readiness #ID`, `approval packet #ID`, and `approval chain proof #ID` before approving it.")
        elif mode == "tool":
            lines.append("- If run for real, the planned tool actions are within the default auto-run risk boundary.")
        else:
            lines.append("- If answered for real, no tool would run from this preview.")
        if recovery_closure and _is_approval_held_review(recovery_closure) and mode != "chat":
            lines.append("- Current blocker: approval-held execution review must be completed before new tool execution is presented as safe.")
        elif recovery_closure and _metadata_bool(recovery_closure["blocks_auto_execution"]) and mode != "chat":
            lines.append("- Current blocker: execution health recovery closure must be completed before new tool execution is presented as safe.")
        if _learning_debt_blocks_rehearsal(execution_learning_debt) and mode != "chat":
            lines.append("- Current blocker: execution learning debt must be closed before new tool execution is presented as complete.")

        route_metadata = _turn_route_metadata(
            request=message,
            mode=mode,
            approval_required=approval_required,
            planned_actions=action_summaries,
            recovery_closure=recovery_closure,
            execution_learning_debt=execution_learning_debt,
        )
        lines.extend(_recovery_closure_lines(recovery_closure))
        if mode != "chat":
            lines.extend(_execution_learning_debt_lines(execution_learning_debt))
        lines.extend(
            [
                "",
                "Next command:",
                f"- {route_metadata['next_command']}",
            ]
        )

        context = {
            "has_profile": bool(profile_text),
            "preferences": len(preferences),
            "memories": len(memories),
            "skills": len(skills),
            "recent_messages": len(recent),
        }
        handoff = _rehearsal_handoff(
            source="assistant_turn_rehearsal",
            text=message,
            mode=mode,
            route_metadata=route_metadata,
            planned_actions=action_summaries,
            risk_counts=risk_counts,
            approval_required=approval_required,
            recovery_closure=recovery_closure,
            execution_learning_debt=execution_learning_debt,
            context=context,
        )

        return ToolResult(
            "assistant_turn_rehearsal",
            True,
            "\n".join(lines),
            _safe_metadata(
                assistant_turn_rehearsal_handoff_ready=handoff["handoff_ready"],
                assistant_turn_rehearsal_ready_for_operator=handoff["ready_for_operator"],
                assistant_turn_rehearsal_state_changed=handoff["state_changed"],
                assistant_turn_rehearsal_changed=handoff["changed"],
                assistant_turn_rehearsal_content_in_handoff=handoff["content_in_handoff"],
                assistant_turn_rehearsal_next_safe_command=handoff["next_safe_command"],
                assistant_turn_rehearsal_next_safe_commands=handoff["next_safe_commands"],
                assistant_turn_rehearsal_next_safe_command_count=handoff["next_safe_command_count"],
                assistant_turn_rehearsal_authorizes_execution=handoff["authorizes_execution"],
                assistant_turn_rehearsal_authorizes_completion_claim=handoff["authorizes_completion_claim"],
                assistant_turn_rehearsal_approval_granted=handoff["approval_granted"],
                message=message,
                mode=mode,
                planned_goal=_safe_short(plan.goal),
                planned_actions=action_summaries,
                approval_required=approval_required,
                risk_counts=dict(risk_counts),
                **route_metadata,
                **_recovery_closure_metadata(recovery_closure),
                **_execution_learning_debt_metadata(execution_learning_debt),
                context=context,
                assistant_turn_rehearsal_handoff=handoff,
            ),
        )

    return assistant_turn_rehearsal


def _terms(text: str) -> list[str]:
    terms = [word.strip(".,?!:;()[]{}'\"`<>-_/\\|").lower() for word in text.split()]
    return [word for word in terms if len(word) >= 4 and word.replace("_", "").isalnum()]
