from __future__ import annotations

from dataclasses import replace
import json
import re
from typing import Any

from jarvis_v2.agent.planner import NEGATED_REQUEST_PLAN_NOTE, RuleBasedPlanner
from jarvis_v2.agent.model_provider import (
    ModelProviderError,
    SUPPORTED_MODEL_PROVIDERS,
    generate_model_text,
    normalized_model_provider,
    ollama_local_only_policy,
    provider_output_token_limit,
    resolve_ollama_destination,
    safe_model_usage_receipt,
)
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel
from jarvis_v2.tools.registry import ToolRegistry, tool_argument_contract_summary


LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
MAX_PLANNER_DETAIL_CHARS = 240
MAX_PLANNER_OUTPUT_TOKENS = 600
MAX_MODEL_PLANNER_USER_INPUT_CHARS = 4_000
MODEL_PLANNER_PLAN_GOAL = "model_planned_tool_actions"
MODEL_PLANNER_ACTION_REASON = "model_selected_registered_tool"
MODEL_PLANNER_UNKNOWN_TOOL_LABEL = "<unrecognized-model-tool>"
MODEL_EXCEPTION_RECOVERY_DIAGNOSTIC = "planner_model_unavailable"
MODEL_PLANNER_DENIED_ACTION_TOOLS = frozenset(
    {
        "approve_pending_approval",
        "dismiss_pending_approval",
        "resolve_auto_mutation_receipt",
    }
)


def _safe_detail(value: Any, *, limit: int = MAX_PLANNER_DETAIL_CHARS) -> str:
    text = " ".join(str(value or "").strip().split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _model_exception_recovery_hint(
    model: str,
    provider: str = "ollama",
    exc: Exception | None = None,
) -> str:
    safe_model = _safe_detail(model, limit=120) or "the configured planner model"
    if normalized_model_provider(provider) == "openai":
        if isinstance(exc, ModelProviderError):
            return f"{exc.recovery_hint} Diagnostic: {MODEL_EXCEPTION_RECOVERY_DIAGNOSTIC}."
        return (
            "Run `model routing status` to check OpenAI configuration before retrying. "
            f"Diagnostic: {MODEL_EXCEPTION_RECOVERY_DIAGNOSTIC}."
        )
    if isinstance(exc, ModelProviderError):
        return f"{exc.recovery_hint} Diagnostic: {MODEL_EXCEPTION_RECOVERY_DIAGNOSTIC}."
    return (
        "Run `model routing status`, start Ollama if it is stopped, and run "
        f"`ollama pull {safe_model}` outside Jarvis if the planner model is missing. "
        f"Diagnostic: {MODEL_EXCEPTION_RECOVERY_DIAGNOSTIC}."
    )


PLANNER_PROMPT = """You are Jarvis's action planner.

Decide whether the user is asking for a tool action or normal conversation.
Use tools only for clear, concrete requests. If the user is chatting, brainstorming,
asking for advice, or being emotionally conversational, return mode "chat".

Return JSON only:
{
  "mode": "tool" | "chat",
  "goal": "short goal",
  "actions": [
    {"tool_name": "name", "args": {}, "reason": "why this tool"}
  ]
}

Rules:
- Only use tool names from the available tools list.
- Never invent tools.
- Never propose approval or dismissal decision tools. Approval decisions require an explicit deterministic route.
- Prefer one tool unless the request truly needs multiple steps.
- Do not use high-risk tools unless the user explicitly asked for that side effect.
- For ambiguous requests, choose chat.
"""


class ModelBackedPlanner:
    def __init__(
        self,
        model: str,
        registry: ToolRegistry,
        base: RuleBasedPlanner | None = None,
        timeout_seconds: float = 2.5,
        provider: str = "ollama",
        reasoning_effort: str = "low",
        openai_max_output_tokens: int = 25_000,
    ):
        self.model = model
        self.registry = registry
        self.base = base or RuleBasedPlanner()
        self.timeout_seconds = timeout_seconds
        self.provider = normalized_model_provider(provider)
        self.reasoning_effort = reasoning_effort
        self.openai_max_output_tokens = openai_max_output_tokens

    def plan(self, user_input: str) -> Plan:
        base_plan = self.base.plan(user_input)
        if base_plan.actions or not base_plan.needs_model:
            return base_plan
        if base_plan.notes == NEGATED_REQUEST_PLAN_NOTE:
            return base_plan

        if self.provider not in SUPPORTED_MODEL_PROVIDERS:
            exc = ModelProviderError(
                "model_provider_invalid",
                "Set JARVIS_MODEL_PROVIDER to ollama or openai, then retry `model routing status`.",
            )
            return self._fallback_plan(
                base_plan,
                reason="invalid_provider",
                detail=type(exc).__name__,
                exception_type=type(exc).__name__,
                recovery_hint=_model_exception_recovery_hint(
                    self.model, self.provider, exc
                ),
                provider_error=exc.diagnostic,
                model_call_attempted=False,
                model_response_received=False,
            )

        if len(user_input) > MAX_MODEL_PLANNER_USER_INPUT_CHARS:
            return self._fallback_plan(
                base_plan,
                reason="user_input_too_large",
                detail="request_exceeded_model_input_limit",
                recovery_hint=(
                    f"Shorten the request to {MAX_MODEL_PLANNER_USER_INPUT_CHARS} "
                    "characters or fewer before retrying."
                ),
                model_call_attempted=False,
                model_response_received=False,
            )

        tool_descriptions = self._tool_descriptions()
        model_usage: dict[str, object] = {}
        model_call_attempted = False
        try:
            model_output_tokens = provider_output_token_limit(
                self.provider,
                local_output_tokens=MAX_PLANNER_OUTPUT_TOKENS,
                openai_output_tokens=self.openai_max_output_tokens,
            )
            model_call_attempted = True
            content = generate_model_text(
                provider=self.provider,
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "system", "content": f"Available tools:\n{tool_descriptions}"},
                    {"role": "user", "content": user_input},
                ],
                timeout_seconds=self.timeout_seconds,
                max_output_tokens=model_output_tokens,
                temperature=0.1,
                reasoning_effort=self.reasoning_effort,
                keep_alive="30m",
                usage_metadata=model_usage,
            )
            data = self._extract_json(content)
            if not data or data.get("mode") != "tool":
                mode = data.get("mode") if isinstance(data, dict) else ""
                return self._fallback_plan(
                    base_plan,
                    reason="model_chose_chat" if mode == "chat" else "invalid_model_json",
                    detail="response_not_recorded",
                    model_usage=model_usage,
                    model_call_attempted=model_call_attempted,
                    model_response_received=True,
                )

            action_items = data.get("actions")
            if not isinstance(action_items, list) or not all(
                isinstance(item, dict) for item in action_items
            ):
                return self._fallback_plan(
                    base_plan,
                    reason="invalid_model_json",
                    detail="response_not_recorded",
                    model_usage=model_usage,
                    model_call_attempted=model_call_attempted,
                    model_response_received=True,
                )

            actions = []
            ignored_unknown_tools = []
            ignored_disallowed_tools = []
            available_tool_names = {tool.name for tool in self.registry.list()}
            for item in action_items:
                name = str(item.get("tool_name") or "")
                if name in MODEL_PLANNER_DENIED_ACTION_TOOLS:
                    ignored_disallowed_tools.append(_safe_detail(name, limit=80))
                    continue
                if name not in available_tool_names:
                    if name:
                        ignored_unknown_tools.append(MODEL_PLANNER_UNKNOWN_TOOL_LABEL)
                    continue
                # Preserve an explicitly malformed root so the executor's typed
                # boundary can reject it. Only a missing args field means `{}`.
                args = item.get("args") if "args" in item else {}
                actions.append(
                    PlannedAction(name, args, MODEL_PLANNER_ACTION_REASON)  # type: ignore[arg-type]
                )
            if not actions:
                return self._fallback_plan(
                    base_plan,
                    reason="no_valid_model_actions",
                    detail=", ".join(ignored_unknown_tools + ignored_disallowed_tools),
                    ignored_unknown_tools=ignored_unknown_tools,
                    ignored_disallowed_tools=ignored_disallowed_tools,
                    model_usage=model_usage,
                    model_call_attempted=model_call_attempted,
                    model_response_received=True,
                )
            return Plan(
                MODEL_PLANNER_PLAN_GOAL,
                actions,
                needs_model=False,
                notes="model_planner",
                metadata=self._metadata(
                    state="used",
                    fallback_reason="",
                    action_count=len(actions),
                    ignored_unknown_tools=ignored_unknown_tools,
                    ignored_disallowed_tools=ignored_disallowed_tools,
                    model_usage=model_usage,
                    model_call_attempted=model_call_attempted,
                    model_response_received=True,
                ),
            )
        except Exception as exc:
            return self._fallback_plan(
                base_plan,
                reason="model_exception",
                detail=type(exc).__name__,
                exception_type=type(exc).__name__,
                recovery_hint=_model_exception_recovery_hint(self.model, self.provider, exc),
                model_usage=model_usage,
                provider_error=(
                    exc.diagnostic if isinstance(exc, ModelProviderError) else ""
                ),
                model_call_attempted=model_call_attempted,
                model_response_received=False,
            )

    def _metadata(
        self,
        *,
        state: str,
        fallback_reason: str,
        action_count: int = 0,
        detail: str = "",
        ignored_unknown_tools: list[str] | None = None,
        ignored_disallowed_tools: list[str] | None = None,
        exception_type: str = "",
        recovery_hint: str = "",
        model_usage: dict[str, object] | None = None,
        provider_error: str = "",
        model_call_attempted: bool = True,
        model_response_received: bool = False,
    ) -> dict[str, Any]:
        usage_receipt = safe_model_usage_receipt(model_usage, self.provider)
        ollama_destination = (
            resolve_ollama_destination() if self.provider == "ollama" else None
        )
        ollama_local_only = (
            ollama_local_only_policy(self.model) if self.provider == "ollama" else None
        )
        destination_allowed = bool(
            self.provider == "openai"
            or (
                self.provider == "ollama"
                and ollama_destination is not None
                and ollama_destination.allowed
            )
        )
        provider_valid = self.provider in SUPPORTED_MODEL_PROVIDERS
        destination_blocked = bool(
            model_call_attempted
            and self.provider == "ollama"
            and not destination_allowed
        )
        cloud_model_blocked = bool(
            model_call_attempted
            and ollama_local_only is not None
            and ollama_local_only.cloud_model_alias
        )
        request_blocked = bool(
            not provider_valid or destination_blocked or cloud_model_blocked
        )
        if model_response_received:
            model_execution_status = "response_received"
            model_execution_occurred: bool | None = True
        elif model_call_attempted and not request_blocked:
            model_execution_status = "outcome_unknown"
            model_execution_occurred = None
        else:
            model_execution_status = "not_executed"
            model_execution_occurred = False
        if model_response_received and self.provider == "openai":
            external_processing_status = "confirmed"
            external_processing_occurred: bool | None = True
        elif model_call_attempted and not request_blocked:
            external_processing_status = "unknown"
            external_processing_occurred = None
        else:
            external_processing_status = "not_executed"
            external_processing_occurred = False
        return {
            "planner_type": "model_backed",
            "model_planner_attempted": True,
            "model_planner_state": state,
            "model_planner_used": state == "used",
            "model_planner_fell_back": state == "fallback",
            "model_planner_fallback_reason": fallback_reason,
            "model_planner_fallback_detail": _safe_detail(detail),
            "model_planner_recovery_hint": _safe_detail(recovery_hint),
            "model_planner_exception_type": _safe_detail(exception_type, limit=80),
            "model_planner_model": _safe_detail(self.model, limit=120),
            "model_planner_provider": self.provider,
            "model_planner_provider_valid": provider_valid,
            "model_planner_error": _safe_detail(provider_error, limit=120),
            "model_planner_calls_model": model_call_attempted,
            "model_planner_calls_external_service": bool(
                model_call_attempted and self.provider == "openai"
            ),
            "model_planner_shares_request_with_external_model": bool(
                model_call_attempted and self.provider == "openai"
            ),
            "model_planner_model_request_attempted": model_call_attempted,
            "model_planner_model_response_received": model_response_received,
            "model_planner_model_execution_status": model_execution_status,
            "model_planner_model_execution_occurred": model_execution_occurred,
            "model_planner_external_processing_status": external_processing_status,
            "model_planner_external_processing_occurred": external_processing_occurred,
            "model_planner_request_processed_externally": external_processing_occurred,
            "model_planner_destination_allowed": destination_allowed,
            "model_planner_request_blocked_by_invalid_provider": bool(
                not provider_valid
            ),
            "model_planner_request_blocked_by_destination_policy": bool(
                destination_blocked
            ),
            "model_planner_request_blocked_by_cloud_policy": bool(
                cloud_model_blocked
            ),
            "model_planner_destination_policy": (
                "openai_external"
                if self.provider == "openai"
                else (
                    "ollama_loopback"
                    if self.provider == "ollama" and destination_allowed
                    else (
                        "ollama_blocked_nonlocal"
                        if self.provider == "ollama"
                        else "invalid_provider"
                    )
                )
            ),
            "model_planner_execution_location_policy": (
                "external_provider"
                if self.provider == "openai"
                else (
                    "loopback_daemon_execution_unverified"
                    if self.provider == "ollama"
                    else "not_applicable_invalid_provider"
                )
            ),
            "model_planner_ollama_destination_value_exposed": False,
            "model_planner_ollama_no_cloud_requested": bool(
                ollama_local_only is not None and ollama_local_only.requested
            ),
            "model_planner_ollama_no_cloud_valid": bool(
                ollama_local_only is None or ollama_local_only.valid
            ),
            "model_planner_ollama_cloud_model_alias": bool(
                ollama_local_only is not None and ollama_local_only.cloud_model_alias
            ),
            "model_planner_ollama_execution_locality_verified": False,
            "model_planner_ollama_cloud_policy_value_exposed": False,
            "model_planner_external_side_effect": False,
            "model_planner_request_content_in_metadata": False,
            "model_planner_response_content_in_metadata": False,
            "model_planner_reasoning_effort": self.reasoning_effort,
            "model_planner_timeout_seconds": self.timeout_seconds,
            "model_planner_user_input_limit_chars": MAX_MODEL_PLANNER_USER_INPUT_CHARS,
            "model_planner_max_output_tokens": provider_output_token_limit(
                self.provider,
                local_output_tokens=MAX_PLANNER_OUTPUT_TOKENS,
                openai_output_tokens=self.openai_max_output_tokens,
            ),
            "model_planner_action_count": action_count,
            "model_planner_ignored_unknown_tools": ignored_unknown_tools or [],
            "model_planner_ignored_disallowed_tools": ignored_disallowed_tools or [],
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "model_planner_usage_available": usage_receipt["model_usage_available"],
            "model_planner_usage_consistent": usage_receipt["model_usage_consistent"],
            "model_planner_input_tokens": usage_receipt["model_input_tokens"],
            "model_planner_cached_input_tokens": usage_receipt["model_cached_input_tokens"],
            "model_planner_cached_input_tokens_available": usage_receipt[
                "model_cached_input_tokens_available"
            ],
            "model_planner_output_tokens": usage_receipt["model_output_tokens"],
            "model_planner_reasoning_tokens": usage_receipt["model_reasoning_tokens"],
            "model_planner_reasoning_tokens_available": usage_receipt[
                "model_reasoning_tokens_available"
            ],
            "model_planner_total_tokens": usage_receipt["model_total_tokens"],
            "model_planner_usage_content_recorded": False,
            "model_planner_ollama_num_ctx_requested": usage_receipt.get(
                "ollama_num_ctx_requested"
            ),
            "model_planner_ollama_num_ctx_requested_available": usage_receipt.get(
                "ollama_num_ctx_requested_available", False
            ),
            "model_planner_ollama_num_ctx_effective": usage_receipt.get(
                "ollama_num_ctx_effective"
            ),
            "model_planner_ollama_num_ctx_effective_available": usage_receipt.get(
                "ollama_num_ctx_effective_available", False
            ),
            "model_planner_ollama_prompt_truncated": usage_receipt.get(
                "ollama_prompt_truncated"
            ),
            "model_planner_ollama_prompt_truncation_verified": usage_receipt.get(
                "ollama_prompt_truncation_verified", False
            ),
            "model_planner_ollama_context_compatibility_verified": usage_receipt.get(
                "ollama_context_compatibility_verified", False
            ),
        }

    def _fallback_plan(
        self,
        base_plan: Plan,
        *,
        reason: str,
        detail: str = "",
        ignored_unknown_tools: list[str] | None = None,
        ignored_disallowed_tools: list[str] | None = None,
        exception_type: str = "",
        recovery_hint: str = "",
        model_usage: dict[str, object] | None = None,
        provider_error: str = "",
        model_call_attempted: bool = True,
        model_response_received: bool = False,
    ) -> Plan:
        return replace(
            base_plan,
            metadata={
                **base_plan.metadata,
                **self._metadata(
                    state="fallback",
                    fallback_reason=reason,
                    detail=detail,
                    ignored_unknown_tools=ignored_unknown_tools,
                    ignored_disallowed_tools=ignored_disallowed_tools,
                    exception_type=exception_type,
                    recovery_hint=recovery_hint,
                    model_usage=model_usage,
                    provider_error=provider_error,
                    model_call_attempted=model_call_attempted,
                    model_response_received=model_response_received,
                ),
            },
        )

    def _tool_descriptions(self) -> str:
        lines = []
        for tool in self.registry.list():
            if tool.name in MODEL_PLANNER_DENIED_ACTION_TOOLS:
                continue
            if tool.risk in {RiskLevel.EXTERNAL_SIDE_EFFECT, RiskLevel.HIGH_RISK}:
                risk_note = "requires explicit approval"
            elif tool.risk == RiskLevel.PERSONAL_DATA:
                risk_note = "personal data; requires explicit approval"
            else:
                risk_note = "safe"
            argument_summary = tool_argument_contract_summary(tool)
            suffix = f" [{argument_summary}]" if argument_summary else ""
            lines.append(
                f"- {tool.name}({tool.toolset}, {tool.risk.name}, {risk_note}): "
                f"{tool.description}{suffix}"
            )
        return "\n".join(lines)

    def _extract_json(self, content: str) -> dict[str, Any] | None:
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                return None
            try:
                decoded = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return decoded if isinstance(decoded, dict) else None
