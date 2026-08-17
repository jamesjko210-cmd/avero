from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_v2.agent.model_planner import (
    MAX_MODEL_PLANNER_USER_INPUT_CHARS,
    MODEL_PLANNER_ACTION_REASON,
    MODEL_PLANNER_PLAN_GOAL,
    MODEL_PLANNER_UNKNOWN_TOOL_LABEL,
    ModelBackedPlanner,
)
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import Plan, RiskLevel, ToolResult
from jarvis_v2.tools.registry import Tool, ToolRegistry


class ModelOnlyPlanner:
    def plan(self, user_input: str) -> Plan:
        return Plan(user_input, [], needs_model=True, notes="model_only_fixture")


def _planner(*, provider: str = "ollama", model: str = "fixture-model") -> ModelBackedPlanner:
    registry = ToolRegistry()
    registry.register(
        Tool(
            "fixture_status",
            "Return fixture status.",
            RiskLevel.READ_ONLY,
            lambda _args: ToolResult("fixture_status", True, "ok"),
            "fixture",
        )
    )
    return ModelBackedPlanner(model, registry, base=ModelOnlyPlanner(), provider=provider)


def test_model_goal_and_reason_content_never_enters_plan_surfaces() -> None:
    marker = "PRIVATE_MODEL_PROSE_MARKER"
    oversized_prose = marker + ("x" * 20_000)
    expected_args = {"query": "preserve this exactly", "limit": 7}
    response = json.dumps(
        {
            "mode": "tool",
            "goal": oversized_prose,
            "actions": [
                {
                    "tool_name": "fixture_status",
                    "args": expected_args,
                    "reason": oversized_prose,
                }
            ],
        }
    )

    with patch("jarvis_v2.agent.model_planner.generate_model_text", return_value=response):
        plan = _planner().plan("fixture request")

    if plan.goal != MODEL_PLANNER_PLAN_GOAL:
        raise SystemExit(f"model goal was not replaced by the deterministic label: {plan.goal!r}")
    if len(plan.actions) != 1 or plan.actions[0].tool_name != "fixture_status":
        raise SystemExit(f"model tool selection drifted: {plan}")
    if plan.actions[0].args != expected_args:
        raise SystemExit(f"model tool arguments drifted: {plan.actions[0].args}")
    if plan.actions[0].reason != MODEL_PLANNER_ACTION_REASON:
        raise SystemExit(f"model reason was not replaced by the deterministic label: {plan.actions[0]}")

    metadata_surfaces = json.dumps(
        {
            "goal": plan.goal,
            "reasons": [action.reason for action in plan.actions],
            "metadata": plan.metadata,
        },
        ensure_ascii=False,
        default=str,
    )
    if marker in metadata_surfaces or oversized_prose in metadata_surfaces:
        raise SystemExit("model goal/reason prose entered Plan metadata surfaces")
    if len(plan.goal) > 80 or any(len(action.reason) > 80 for action in plan.actions):
        raise SystemExit("deterministic model planner labels exceeded their smoke bound")
    if plan.metadata.get("model_planner_response_content_in_metadata") is not False:
        raise SystemExit(f"model planner privacy flag drifted: {plan.metadata}")


def test_json_arrays_and_scalars_are_invalid_responses_not_provider_failures() -> None:
    cases = [
        [],
        [{"mode": "tool", "actions": []}],
        None,
        True,
        42,
        "scalar model response",
        {"mode": "tool", "actions": 42},
        {"mode": "tool", "actions": [42]},
    ]
    planner = _planner()

    for payload in cases:
        with patch(
            "jarvis_v2.agent.model_planner.generate_model_text",
            return_value=json.dumps(payload),
        ):
            plan = planner.plan("fixture request")

        metadata = plan.metadata
        if metadata.get("model_planner_fallback_reason") != "invalid_model_json":
            raise SystemExit(
                f"malformed JSON was not classified as an invalid response: {payload!r}, {metadata}"
            )
        if metadata.get("model_planner_fallback_detail") != "response_not_recorded":
            raise SystemExit(f"invalid response retained content detail: {payload!r}, {metadata}")
        if metadata.get("model_planner_exception_type"):
            raise SystemExit(f"provider response was misclassified as an exception: {payload!r}, {metadata}")
        if metadata.get("model_planner_recovery_hint"):
            raise SystemExit(f"provider response incorrectly emitted outage recovery: {payload!r}, {metadata}")
        expected_execution = {
            "model_planner_model_execution_status": "response_received",
            "model_planner_model_execution_occurred": True,
            "model_planner_external_processing_status": "unknown",
            "model_planner_external_processing_occurred": None,
        }
        wrong_execution = {
            key: metadata.get(key)
            for key, expected in expected_execution.items()
            if metadata.get(key) != expected
        }
        if wrong_execution:
            raise SystemExit(
                f"local planner response execution receipt drifted: {wrong_execution} / {metadata}"
            )
        recovery_hint = str(metadata.get("model_planner_recovery_hint") or "").lower()
        if "start ollama" in recovery_hint or "ollama pull" in recovery_hint:
            raise SystemExit(
                f"invalid provider response told the operator to recover Ollama: {payload!r}, {metadata}"
            )


def test_unknown_model_tool_names_are_content_free() -> None:
    marker = "PRIVATE_UNKNOWN_TOOL_NAME_MUST_NOT_PERSIST"
    response = json.dumps(
        {
            "mode": "tool",
            "actions": [{"tool_name": marker, "args": {}, "reason": marker}],
        }
    )
    with patch("jarvis_v2.agent.model_planner.generate_model_text", return_value=response):
        plan = _planner().plan("private fixture request")
    metadata_text = json.dumps(plan.metadata, ensure_ascii=False, default=str)
    if marker in metadata_text:
        raise SystemExit("unknown model tool name entered planner metadata")
    if plan.metadata.get("model_planner_ignored_unknown_tools") != [
        MODEL_PLANNER_UNKNOWN_TOOL_LABEL
    ]:
        raise SystemExit(f"unknown model tool did not retain content-free count evidence: {plan.metadata}")
    if marker in str(plan.metadata.get("model_planner_fallback_detail") or ""):
        raise SystemExit("unknown model tool name entered fallback detail")


def test_blocked_remote_ollama_destination_falls_back_before_client_construction() -> None:
    raw_host = "private-model-planner.example.invalid:22999"
    client_constructions: list[dict[str, object]] = []
    client_calls: list[dict[str, object]] = []

    class ForbiddenClient:
        def __init__(self, **kwargs: object) -> None:
            client_constructions.append(dict(kwargs))
            raise AssertionError("blocked model planner destination constructed an Ollama client")

        def chat(self, **kwargs: object) -> object:
            client_calls.append(dict(kwargs))
            raise AssertionError("blocked model planner destination called an Ollama client")

    with (
        patch.dict(os.environ, {"OLLAMA_HOST": raw_host}, clear=False),
        patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=ForbiddenClient)}),
    ):
        plan = _planner().plan("private fixture request")

    if client_constructions or client_calls:
        raise SystemExit(
            "blocked remote OLLAMA_HOST crossed the provider client boundary: "
            f"constructions={client_constructions}, calls={client_calls}"
        )
    if plan.actions or plan.needs_model is not True or plan.notes != "model_only_fixture":
        raise SystemExit(f"blocked model planner request did not preserve the base fallback: {plan}")

    metadata = plan.metadata
    expected = {
        "model_planner_attempted": True,
        "model_planner_state": "fallback",
        "model_planner_used": False,
        "model_planner_fell_back": True,
        "model_planner_fallback_reason": "model_exception",
        "model_planner_fallback_detail": "ModelProviderError",
        "model_planner_exception_type": "ModelProviderError",
        "model_planner_provider": "ollama",
        "model_planner_destination_allowed": False,
        "model_planner_request_blocked_by_destination_policy": True,
        "model_planner_destination_policy": "ollama_blocked_nonlocal",
        "model_planner_ollama_destination_value_exposed": False,
        "model_planner_model_execution_status": "not_executed",
        "model_planner_model_execution_occurred": False,
        "model_planner_external_processing_status": "not_executed",
        "model_planner_external_processing_occurred": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SystemExit(
                f"blocked model planner destination metadata drifted at {key}: {metadata}"
            )
    if metadata.get("model_planner_state") == "used":
        raise SystemExit(f"blocked model planner destination claimed it was used: {metadata}")
    recovery_hint = str(metadata.get("model_planner_recovery_hint") or "")
    if "numeric loopback address" not in recovery_hint or "planner_model_unavailable" not in recovery_hint:
        raise SystemExit(f"blocked model planner fallback missed destination recovery: {metadata}")
    if raw_host in json.dumps(metadata, ensure_ascii=False, default=str):
        raise SystemExit("blocked model planner metadata exposed the configured remote destination")


def test_invalid_provider_falls_back_without_provider_or_client_call() -> None:
    client_constructions: list[dict[str, object]] = []

    class ForbiddenClient:
        def __init__(self, **kwargs: object) -> None:
            client_constructions.append(dict(kwargs))
            raise AssertionError("invalid provider constructed a client")

    planner = _planner(provider="not-a-provider")
    with (
        patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=ForbiddenClient)}),
        patch(
            "jarvis_v2.agent.model_planner.generate_model_text",
            side_effect=AssertionError("invalid provider called provider function"),
        ) as provider_call,
    ):
        plan = planner.plan("private invalid-provider request")

    if provider_call.call_count or client_constructions:
        raise SystemExit(
            "invalid model planner provider crossed the provider boundary: "
            f"provider_calls={provider_call.call_count}, clients={client_constructions}"
        )
    if plan.actions or plan.needs_model is not True or plan.notes != "model_only_fixture":
        raise SystemExit(f"invalid provider did not preserve the base fallback: {plan}")
    expected = {
        "model_planner_state": "fallback",
        "model_planner_fallback_reason": "invalid_provider",
        "model_planner_provider": "invalid",
        "model_planner_provider_valid": False,
        "model_planner_error": "model_provider_invalid",
        "model_planner_calls_model": False,
        "model_planner_calls_external_service": False,
        "model_planner_shares_request_with_external_model": False,
        "model_planner_model_request_attempted": False,
        "model_planner_model_response_received": False,
        "model_planner_request_blocked_by_invalid_provider": True,
        "model_planner_request_blocked_by_destination_policy": False,
        "model_planner_request_blocked_by_cloud_policy": False,
        "model_planner_destination_policy": "invalid_provider",
        "model_planner_execution_location_policy": "not_applicable_invalid_provider",
        "model_planner_model_execution_status": "not_executed",
        "model_planner_model_execution_occurred": False,
        "model_planner_external_processing_status": "not_executed",
        "model_planner_external_processing_occurred": False,
        "model_planner_request_processed_externally": False,
    }
    wrong = {
        key: plan.metadata.get(key)
        for key, value in expected.items()
        if plan.metadata.get(key) != value
    }
    if wrong:
        raise SystemExit(f"invalid model planner metadata drifted: {wrong} / {plan.metadata}")
    recovery = str(plan.metadata.get("model_planner_recovery_hint") or "")
    if "JARVIS_MODEL_PROVIDER" not in recovery or "start Ollama" in recovery:
        raise SystemExit(f"invalid provider fallback recovery was not truthful: {plan.metadata}")


def test_oversized_user_input_falls_back_before_model_dispatch() -> None:
    marker = "PRIVATE_OVERSIZED_PLANNER_INPUT_MUST_NOT_REACH_MODEL"
    oversized = marker + ("x" * MAX_MODEL_PLANNER_USER_INPUT_CHARS)
    with patch(
        "jarvis_v2.agent.model_planner.generate_model_text",
        side_effect=AssertionError("oversized planner input reached the model provider"),
    ) as model_call:
        plan = _planner().plan(oversized)

    if model_call.call_count:
        raise SystemExit("oversized planner input crossed the model-provider boundary")
    metadata = plan.metadata
    expected = {
        "model_planner_state": "fallback",
        "model_planner_fallback_reason": "user_input_too_large",
        "model_planner_fallback_detail": "request_exceeded_model_input_limit",
        "model_planner_calls_model": False,
        "model_planner_model_request_attempted": False,
        "model_planner_model_execution_status": "not_executed",
        "model_planner_user_input_limit_chars": MAX_MODEL_PLANNER_USER_INPUT_CHARS,
        "model_planner_request_content_in_metadata": False,
    }
    wrong = {
        key: metadata.get(key)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if wrong:
        raise SystemExit(f"oversized planner fallback metadata drifted: {wrong} / {metadata}")
    if marker in json.dumps(metadata, ensure_ascii=False, default=str):
        raise SystemExit("oversized planner input entered fallback metadata")
    recovery = str(metadata.get("model_planner_recovery_hint") or "")
    if str(MAX_MODEL_PLANNER_USER_INPUT_CHARS) not in recovery or marker in recovery:
        raise SystemExit(f"oversized planner fallback recovery was not bounded: {metadata}")


def test_user_input_at_limit_dispatches_unchanged_with_truthful_context_receipt() -> None:
    bounded_input = "b" * MAX_MODEL_PLANNER_USER_INPUT_CHARS
    captured_messages: list[dict[str, str]] = []

    def fake_generate_model_text(**kwargs):
        captured_messages.extend(kwargs.get("messages") or [])
        usage = kwargs.get("usage_metadata")
        if not isinstance(usage, dict):
            raise AssertionError("planner omitted the model usage receipt")
        usage.update(
            {
                "ollama_num_ctx_requested": 24_000,
                "ollama_num_ctx_requested_available": True,
                "ollama_num_ctx_effective": None,
                "ollama_num_ctx_effective_available": False,
                "ollama_prompt_truncated": None,
                "ollama_prompt_truncation_verified": False,
                "ollama_context_compatibility_verified": False,
            }
        )
        return json.dumps({"mode": "chat", "actions": []})

    with patch(
        "jarvis_v2.agent.model_planner.generate_model_text",
        side_effect=fake_generate_model_text,
    ) as model_call:
        plan = _planner().plan(bounded_input)

    if model_call.call_count != 1 or not captured_messages:
        raise SystemExit("planner input at the exact bound did not dispatch once")
    if captured_messages[-1] != {"role": "user", "content": bounded_input}:
        raise SystemExit("planner changed input at the exact dispatch bound")
    expected_receipt = {
        "model_planner_ollama_num_ctx_requested": 24_000,
        "model_planner_ollama_num_ctx_requested_available": True,
        "model_planner_ollama_num_ctx_effective": None,
        "model_planner_ollama_num_ctx_effective_available": False,
        "model_planner_ollama_prompt_truncated": None,
        "model_planner_ollama_prompt_truncation_verified": False,
        "model_planner_ollama_context_compatibility_verified": False,
    }
    wrong = {
        key: plan.metadata.get(key)
        for key, expected in expected_receipt.items()
        if plan.metadata.get(key) != expected
    }
    if wrong:
        raise SystemExit(f"planner lost truthful Ollama context receipt fields: {wrong}")
    if bounded_input in json.dumps(plan.metadata, ensure_ascii=False, default=str):
        raise SystemExit("bounded planner request content entered metadata")


def test_negated_and_hypothetical_reminder_cancellation_never_reaches_model_tools() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            "cancel_reminders",
            "Cancel reminders.",
            RiskLevel.LOCAL_SAFE,
            lambda _args: ToolResult("cancel_reminders", True, "cancelled"),
            "personal",
        )
    )
    planner = ModelBackedPlanner("fixture-model", registry, base=RuleBasedPlanner())
    commands = (
        "don't cancel my reminders",
        "tell me how to cancel reminders",
        "can I cancel all reminders?",
        "what happens if I cancel all reminders?",
        "suppose I cancel all reminders",
        "please don't cancel my reminders",
        "I don't want to cancel my reminders",
        "I don't want my reminders cancelled",
        "cancel all reminders and then don't actually do it",
    )
    with patch(
        "jarvis_v2.agent.model_planner.generate_model_text",
        side_effect=AssertionError("withdrawn reminder request reached the model planner"),
    ) as model_call:
        for command in commands:
            plan = planner.plan(command)
            if any(action.tool_name == "cancel_reminders" for action in plan.actions):
                raise SystemExit(f"withdrawn reminder request planned cancellation: {command!r} -> {plan}")
            if command == "don't cancel my reminders":
                if plan.actions or plan.needs_model is not True:
                    raise SystemExit(f"negated reminder request did not preserve chat fallback: {command!r} -> {plan}")
            elif [action.tool_name for action in plan.actions] != ["respond"] or plan.needs_model:
                raise SystemExit(f"withdrawn reminder request did not stop at deterministic response: {command!r} -> {plan}")
    if model_call.call_count:
        raise SystemExit("withdrawn reminder request crossed the model planner boundary")


def main() -> None:
    test_model_goal_and_reason_content_never_enters_plan_surfaces()
    test_json_arrays_and_scalars_are_invalid_responses_not_provider_failures()
    test_unknown_model_tool_names_are_content_free()
    test_blocked_remote_ollama_destination_falls_back_before_client_construction()
    test_invalid_provider_falls_back_without_provider_or_client_call()
    test_oversized_user_input_falls_back_before_model_dispatch()
    test_user_input_at_limit_dispatches_unchanged_with_truthful_context_receipt()
    test_negated_and_hypothetical_reminder_cancellation_never_reaches_model_tools()
    print("Model planner smoke passed")


if __name__ == "__main__":
    main()
