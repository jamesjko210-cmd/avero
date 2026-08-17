from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

import jarvis_v2.agent.model_provider as model_provider_module
from jarvis_v2.agent.model_provider import ModelProviderError
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.v3_commands import V3_DASHBOARD_COMMAND, V3_DASHBOARD_INFO_COMMAND


EXPECTED_EMPTY_PLANNER_METADATA = {
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


def assert_empty_planner_diagnostics(metadata: dict, label: str) -> None:
    planner_metadata = metadata.get("planner_metadata")
    if planner_metadata != EXPECTED_EMPTY_PLANNER_METADATA:
        raise AssertionError(f"{label} missed stable empty planner metadata: {metadata}")
    for key, value in EXPECTED_EMPTY_PLANNER_METADATA.items():
        flat_key = f"planner_{key}"
        if flat_key in metadata and metadata.get(flat_key) != value:
            raise AssertionError(f"{label} flat planner field {flat_key} diverged from nested metadata: {metadata}")
    if metadata.get("planner_model_planner_attempted") is not False:
        raise AssertionError(f"{label} should report no model planner attempt: {metadata}")
    if metadata.get("planner_model_planner_ignored_unknown_tools") != []:
        raise AssertionError(f"{label} should report an empty ignored-tool list: {metadata}")
    for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
        if planner_metadata.get(key) is not False:
            raise AssertionError(f"{label} planner metadata should not grant {key}: {metadata}")


def test_system_prompt_pins_non_hostility() -> None:
    """the operator's standing rule: Jarvis must NEVER be hostile to anyone. Pin the
    system-prompt guardrail so a rewrite can't silently drop it."""
    from jarvis_v2.agent.chat import SYSTEM_PROMPT

    for required in ["NEVER be hostile", "stay respectful", "cruelty never is"]:
        if required not in SYSTEM_PROMPT:
            raise AssertionError(f"system prompt lost the non-hostility guardrail phrase {required!r}")


def test_ollama_recovery_hint_differentiates_causes() -> None:
    """Adapter diagnostics retain exact, content-free recovery without SDK types."""
    from jarvis_v2.agent.chat import _ollama_recovery_hint

    cases = (
        (
            "ollama_timeout",
            "Ollama took too long to respond. Try again without replaying any action.",
        ),
        (
            "ollama_unreachable",
            "Ollama does not appear to be running. Start it, then run `model status` in Jarvis.",
        ),
        (
            "ollama_model_unavailable",
            "The configured Ollama model is not pulled. Pull it outside Jarvis, then retry.",
        ),
        (
            "ollama_service_error",
            "The local Ollama daemon could not complete the request. Retry later.",
        ),
    )
    with patch.dict(sys.modules, {"httpx": None, "ollama": None, "pydantic": None}):
        hints = [
            _ollama_recovery_hint(ModelProviderError(diagnostic, recovery), "llama3.1")
            for diagnostic, recovery in cases
        ]
        generic_hint = _ollama_recovery_hint(RuntimeError("something unexpected"), "llama3.1")

    if hints != [recovery for _diagnostic, recovery in cases]:
        raise AssertionError(f"ModelProviderError recovery guidance drifted: {hints!r}")

    if "something unexpected" in generic_hint:
        raise AssertionError(f"unrecognized-exception case should not leak raw exception text: {generic_hint!r}")
    if "ollama pull llama3.1" not in generic_hint or "start ollama" not in generic_hint.lower():
        raise AssertionError(f"unrecognized-exception case should keep the full generic guidance: {generic_hint!r}")


def test_chat_risk_matching_uses_unicode_token_phrases(runtime: JarvisRuntime) -> None:
    benign_messages = (
        "Brunch plans should stay relaxed.",
        "The runway discussion is conceptual.",
        "Shellfish allergies are worth discussing.",
        "Sunscreen matters on bright days.",
        "Let's discuss credit scores.",
        "How can I improve my posture?",
        "Why does the trail descend here?",
        "What makes a strong professional profile?",
        "Based only on the official source, what exact command lists the models?",
        "What codename and finish target did I give you in my previous message?",
        "Remember this only within this conversation without saving it.",
        "What is Blue Lantern's finish target?",
        "Summarize this conversation, including the research answer.",
    )
    risky_messages = {
        "Could we discuss how to RUN, A SCRIPT?": {"shell/code"},
        "Would you OPEN - THE TERMINAL?": {"shell/code"},
        "Could you EMAIL ME, THE RESULT?": {"personal-data", "external-side-effect"},
        "Could you \uff52\uff55\uff4e \uff41 \uff53\uff43\uff52\uff49\uff50\uff54?": {"shell/code"},
    }
    approvals_before = [dict(row) for row in runtime.store.list_pending_approvals(status="pending")]

    with patch("jarvis_v2.agent.chat.generate_model_text", side_effect=RuntimeError("mocked offline model")):
        for message in benign_messages:
            result = runtime.handle(message)
            chat_response = result.metadata.get("chat_response", {})
            if result.metadata.get("runtime_route") != "chat" or result.tool_results:
                raise AssertionError(f"benign substring escaped normal chat: {message!r}: {result.metadata}")
            if chat_response.get("source") != "fallback":
                raise AssertionError(f"benign substring bypassed fallback chat: {message!r}: {chat_response}")
            if chat_response.get("turn_type") != "conversation" or chat_response.get("risk_signals") != []:
                raise AssertionError(f"benign substring produced risky classification: {message!r}: {chat_response}")

        for message, expected_risks in risky_messages.items():
            result = runtime.handle(message)
            chat_response = result.metadata.get("chat_response", {})
            if result.metadata.get("runtime_route") != "chat" or result.tool_results:
                raise AssertionError(f"risky phrase escaped chat preflight: {message!r}: {result.metadata}")
            if chat_response.get("source") != "safety_preflight_guidance":
                raise AssertionError(f"risky phrase did not use deterministic preflight: {message!r}: {chat_response}")
            if not expected_risks.issubset(set(chat_response.get("risk_signals", []))):
                raise AssertionError(f"risky phrase missed risk signals: {message!r}: {chat_response}")
            if "execution governor:" not in result.response.lower():
                raise AssertionError(f"risky phrase preflight missed execution governor: {message!r}")

    approvals_after = [dict(row) for row in runtime.store.list_pending_approvals(status="pending")]
    if approvals_after != approvals_before:
        raise AssertionError("chat risk matching changed the approval queue")


def main() -> None:
    test_system_prompt_pins_non_hostility()
    test_ollama_recovery_hint_differentiates_causes()
    with TemporaryDirectory(prefix="jarvis-chat-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-missing-smoke-model",
            use_model_planner=False,
        )
        runtime = JarvisRuntime(config)
        test_chat_risk_matching_uses_unicode_token_phrases(runtime)
        cases = [
            "set preference response style to direct and warm category communication",
            "Hey Jarvis, can we just talk normally now?",
            "What do you think the brain of this project should feel like?",
            "Can you control my computer safely?",
            "How do you keep Jarvis from causing harm?",
            "How should Jarvis use Obsidian memory?",
            "What should we build next for Jarvis?",
            "remember that conversational Jarvis mode was added",
            "Do you remember anything about conversational mode?",
        ]
        class MissingModelOpener:
            def open(self, request, *, timeout):
                raise HTTPError(
                    request.full_url,
                    404,
                    "private missing-model server text SHOULD NOT APPEAR",
                    {},
                    None,
                )

        for case in cases:
            with (
                patch.object(
                    model_provider_module,
                    "build_opener",
                    return_value=MissingModelOpener(),
                ),
                patch.dict(
                    sys.modules,
                    {"httpx": None, "ollama": None, "pydantic": None},
                ),
            ):
                result = runtime.handle(case)
            kind = "chat" if result.plan.needs_model and not result.plan.actions else "tool"
            print(f"[{kind}] {case}")
            print(result.response)
            print()
            if kind == "chat":
                chat_response = result.metadata.get("chat_response", {})
                runtime_trace = result.metadata.get("runtime_trace", {})
                if result.metadata.get("runtime_route") != "chat" or not chat_response:
                    raise AssertionError("chat result did not include chat response metadata")
                if runtime_trace.get("route") != "chat" or runtime_trace.get("approval_required"):
                    raise AssertionError("chat result did not include safe chat runtime trace")
                if runtime_trace.get("chat_response") != chat_response:
                    raise AssertionError("chat runtime trace did not mirror chat response metadata")
                if not any(stage.get("stage") == "response" for stage in runtime_trace.get("stages", [])):
                    raise AssertionError("chat runtime trace missed response stage")
                if chat_response.get("source") not in {"model", "fallback", "grounded_memory", "safety_preflight_guidance"}:
                    raise AssertionError("chat response metadata had unknown source")
                if chat_response.get("model_timeout_seconds") != config.chat_timeout_seconds:
                    raise AssertionError("chat response metadata missed chat timeout")
                if chat_response.get("latency_recorded") is not True:
                    raise AssertionError(f"chat response metadata missed latency_recorded: {chat_response}")
                latency_ms = chat_response.get("latency_ms")
                if not isinstance(latency_ms, (int, float)) or latency_ms < 0:
                    raise AssertionError(f"chat response metadata missed bounded latency_ms: {chat_response}")
                if chat_response.get("duration_ms") != latency_ms:
                    raise AssertionError(f"chat response metadata should mirror latency_ms as duration_ms: {chat_response}")
                grounded_memory = any(term in case.lower() for term in ("memory", "remember"))
                safety_preflight = any(term in case.lower() for term in ("control my computer", "click", "screen", "run a script", "send"))
                expected_source = (
                    "grounded_memory"
                    if grounded_memory
                    else "safety_preflight_guidance"
                    if safety_preflight
                    else "fallback"
                )
                if chat_response.get("source") != expected_source:
                    raise AssertionError(f"chat source was {chat_response.get('source')}, expected {expected_source}")
                if "local chat model is not responding yet" in result.response:
                    raise AssertionError("chat fallback returned the old unavailable-model response")
                if chat_response.get("source") == "fallback":
                    if chat_response.get("model_error") != "ollama_model_unavailable":
                        raise AssertionError(f"chat fallback leaked raw model error metadata: {chat_response}")
                    if not chat_response.get("model_exception_type"):
                        raise AssertionError(f"chat fallback missed bounded model exception type: {chat_response}")
                    for expected in (
                        "not pulled",
                        "Pull it outside Jarvis",
                        "Diagnostic: ollama_model_unavailable",
                    ):
                        if expected not in result.response:
                            raise AssertionError(f"chat fallback missed model recovery guidance {expected!r}: {result.response}")
                    for forbidden in ["status code", "not found", "ResponseError", "model '"]:
                        if forbidden in result.response or forbidden in str(chat_response.get("model_error")):
                            raise AssertionError(f"chat fallback leaked raw model failure text: {result.response}")
                if not grounded_memory and not safety_preflight and "fallback chat" not in result.response:
                    raise AssertionError("chat fallback did not explain the degraded local-model path")
                low_response = result.response.lower()
                if "control my computer" in case.lower() and "observe" not in low_response:
                    raise AssertionError("computer-control chat fallback did not mention observe/approval flow")
                if safety_preflight and ("command-first harness loop" not in low_response or "execution governor:" not in low_response):
                    raise AssertionError("action-like chat did not route through the command-first execution governor loop")
                if "causing harm" in case.lower() and "approval" not in low_response:
                    raise AssertionError("safety chat fallback did not mention approvals")
                if grounded_memory and "safety boundary" not in low_response:
                    raise AssertionError("grounded memory chat did not include safety boundary")
                if "obsidian memory" in case.lower() and "memory" not in low_response:
                    raise AssertionError("memory chat did not mention memory")
                if "build next" in case.lower() and "mission control" not in low_response:
                    raise AssertionError("next-work chat fallback did not mention Mission Control")

        recall = runtime.handle("Do you remember anything about conversational mode?")
        if "conversational Jarvis mode was added" not in recall.response:
            raise AssertionError("chat fallback did not include relevant memory context")
        if "response style" not in recall.response:
            raise AssertionError("chat fallback did not include active preference context")

        path_preview = runtime.handle("chat loop preview: run command cat /\x55sers/example/private-chat-loop.txt /private/tmp/chat-token.txt /var/folders/zc/chat-cache.txt /tmp/chat-raw.txt")
        if path_preview.tool_results[0].tool_name != "chat_loop_preview":
            raise AssertionError("path-shaped chat loop preview did not route to chat_loop_preview")
        path_preview_metadata = path_preview.tool_results[0].metadata
        path_preview_text = f"{path_preview.response}\n{path_preview_metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in path_preview_text:
                raise AssertionError(f"chat loop preview leaked local path fragment {forbidden!r}: {path_preview_text}")
        if "<local-path>" not in path_preview_text:
            raise AssertionError("chat loop preview should retain a redacted local-path marker")
        if path_preview_metadata.get("next_command") != "execution governor: run command cat <local-path>":
            raise AssertionError(f"chat loop preview should redact next_command metadata: {path_preview_metadata}")
        commands = path_preview_metadata.get("recommended_next_commands", [])
        if not commands or commands[0] != "execution governor: run command cat <local-path>":
            raise AssertionError(f"chat loop preview should redact recommended route commands: {path_preview_metadata}")
        if path_preview_metadata.get("executes_tools") or path_preview_metadata.get("queues_approval") or path_preview_metadata.get("writes_memory"):
            raise AssertionError(f"chat loop preview should remain read-only: {path_preview_metadata}")

        safe_tool = runtime.handle("jarvis status")
        safe_trace = safe_tool.metadata.get("runtime_trace", {})
        if safe_tool.metadata.get("runtime_route") != "tools":
            raise AssertionError("safe tool result did not use tools route")
        if safe_trace.get("route") != "tools" or safe_trace.get("approval_required"):
            raise AssertionError("safe tool runtime trace missed automatic tools route")
        if safe_trace.get("ran_tool_handlers") is not True:
            raise AssertionError("safe tool runtime trace did not record tool execution")
        if not safe_trace.get("planned_actions") or safe_trace["planned_actions"][0].get("risk") != "READ_ONLY":
            raise AssertionError("safe tool runtime trace missed READ_ONLY action")
        for expected in [
            "active project: Jarvis V3",
            V3_DASHBOARD_COMMAND,
            V3_DASHBOARD_INFO_COMMAND,
        ]:
            if expected not in safe_tool.response:
                raise AssertionError(f"safe tool status missed discovery hint: {expected}")
        safe_tool_metadata = safe_tool.tool_results[0].metadata
        if safe_tool_metadata.get("project_name") != "Jarvis V3" or safe_tool_metadata.get("dashboard_launcher_exists") is not True:
            raise AssertionError(f"safe tool status missed project metadata: {safe_tool_metadata}")

        stop_result = runtime.handle("stop Jarvis")
        stop_trace = stop_result.metadata.get("runtime_trace", {})
        if not stop_result.verified or stop_result.tool_results[0].tool_name != "stop_jarvis":
            raise AssertionError("stop Jarvis did not route to the non-destructive stop tool")
        if "did not delete the message" not in stop_result.response or "approval queue visible" not in stop_result.response:
            raise AssertionError("stop Jarvis response missed persistent-message and approval visibility wording")
        if stop_trace.get("approval_required") or stop_trace.get("queued_approval_ids"):
            raise AssertionError("stop Jarvis should not require or queue approval")
        stop_metadata = stop_result.tool_results[0].metadata
        for key in ["executes_tools", "queues_approval", "approves_request", "dismisses_request", "controls_computer", "reads_private_data", "writes_files"]:
            if stop_metadata.get(key) is not False:
                raise AssertionError(f"stop Jarvis should report {key}=False")

        diagnosis = runtime.handle("command diagnosis: run command python3 --version")
        if diagnosis.tool_results[0].tool_name != "command_diagnosis":
            raise AssertionError("command diagnosis did not route to command_diagnosis tool")
        diagnosis_metadata = diagnosis.tool_results[0].metadata
        if diagnosis_metadata.get("route") not in {"approval", "hold"}:
            raise AssertionError("command diagnosis missed risky route")
        if diagnosis_metadata.get("approval_required") is not True:
            raise AssertionError("command diagnosis missed approval requirement")
        if not diagnosis_metadata.get("next_command"):
            raise AssertionError("command diagnosis missed next_command metadata")
        if diagnosis_metadata.get("route") == "approval" and not str(diagnosis_metadata.get("next_command", "")).startswith("execution governor: "):
            raise AssertionError("command diagnosis missed approval next_command metadata")
        if diagnosis_metadata.get("route") == "hold" and diagnosis_metadata.get("next_command") != "pending approvals":
            raise AssertionError("command diagnosis missed hold next_command metadata")
        if diagnosis_metadata.get("executes_tools") or diagnosis_metadata.get("queues_approval"):
            raise AssertionError("command diagnosis should stay read-only")
        assert_empty_planner_diagnostics(diagnosis_metadata, "command diagnosis")
        if (
            diagnosis_metadata.get("forecast_queue_before") != 0
            or diagnosis_metadata.get("forecast_queue_after_if_sent") != 1
            or diagnosis_metadata.get("forecast_queue_delta_if_sent") != 1
            or diagnosis_metadata.get("forecast_new_approvals") != 1
            or diagnosis_metadata.get("forecast_reused_approval_ids") != []
        ):
            raise AssertionError(f"command diagnosis missed new-approval forecast: {diagnosis_metadata}")
        forecast = diagnosis_metadata.get("approval_queue_forecast") or []
        if not forecast or forecast[0].get("tool_name") != "run_shell_command" or forecast[0].get("would_queue_new_approval") is not True:
            raise AssertionError(f"command diagnosis missed per-action approval forecast: {diagnosis_metadata}")
        commands = diagnosis_metadata.get("recommended_next_commands", [])
        if diagnosis_metadata.get("route") == "approval" and (not commands or not str(commands[0]).startswith("execution governor: ")):
            raise AssertionError("command diagnosis missed governor-first next command")
        for command in ["approval readiness latest", "approval packet latest", "approval chain proof latest"]:
            if command not in commands:
                raise AssertionError(f"command diagnosis missed approval-chain next command: {command}")
        for expected in [
            "Next safe commands",
            "execution governor",
            "approval readiness latest",
            "approval packet latest",
            "approval chain proof latest",
            "Risk category: shell/code execution",
            "Why gated",
            "Safer check",
            "Safe to execute now: no",
            "Approval queue forecast",
            "would queue new approvals if sent: 1",
            "forecast queue after if sent: 1",
        ]:
            if expected not in diagnosis.response:
                raise AssertionError(f"command diagnosis missed review text: {expected}")

        file_diagnosis = runtime.handle("command diagnosis: read file /\x55sers/example/private-note.txt")
        file_metadata = file_diagnosis.tool_results[0].metadata
        if file_metadata.get("route") not in {"approval", "hold"}:
            raise AssertionError("file command diagnosis missed personal-data approval route")
        if file_metadata.get("approval_required") is not True or file_metadata.get("safe_to_execute_now") is not False:
            raise AssertionError("file command diagnosis missed approval safety metadata")
        if not file_metadata.get("next_command"):
            raise AssertionError("file command diagnosis missed next_command metadata")
        planned = file_metadata.get("planned_actions") or []
        if not planned or planned[0].get("risk") != "PERSONAL_DATA" or planned[0].get("tool") != "read_text_file":
            raise AssertionError("file command diagnosis missed PERSONAL_DATA read_text_file plan")
        file_diagnosis_text = f"{file_diagnosis.response}\n{file_metadata}"
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in file_diagnosis_text:
                raise AssertionError(f"file command diagnosis leaked local path fragment {forbidden!r}: {file_diagnosis_text}")
        if "<local-path>" not in file_diagnosis_text:
            raise AssertionError("file command diagnosis should retain a redacted local-path marker")
        if planned[0].get("args", {}).get("path") != "<local-path>":
            raise AssertionError(f"file command diagnosis missed redacted planned path: {file_metadata}")
        if file_metadata.get("next_command") != "execution governor: read file <local-path>":
            raise AssertionError(f"file command diagnosis missed redacted next_command metadata: {file_metadata}")
        if "execution governor: read file <local-path>" not in file_metadata.get("recommended_next_commands", []):
            raise AssertionError(f"file command diagnosis missed redacted route command: {file_metadata}")
        for command in ["approval readiness latest", "approval packet latest", "approval chain proof latest"]:
            if command not in file_metadata.get("recommended_next_commands", []):
                raise AssertionError(f"file command diagnosis missed approval-chain next command: {command}")
        for expected in [
            "Risk category: personal file content read",
            "File contents may contain private notes",
            "Planned arguments: path='<local-path>'",
            "Safe to execute now: no",
        ]:
            if expected not in file_diagnosis.response:
                raise AssertionError(f"file command diagnosis missed review text: {expected}")

        natural_risky_order = runtime.handle("use my computer to inspect the screen run a python script write a summary file and email me the result")
        if natural_risky_order.metadata.get("runtime_route") != "tools":
            raise AssertionError("natural risky order should route to read-only dispatch tooling")
        if not natural_risky_order.tool_results or natural_risky_order.tool_results[0].tool_name != "dispatch_decision_packet":
            raise AssertionError("natural risky order did not route to dispatch_decision_packet")
        dispatch_metadata = natural_risky_order.tool_results[0].metadata
        if dispatch_metadata.get("decision") != "ASK_FOR_EXACT_ACTION_OR_PREFLIGHT":
            raise AssertionError(f"natural risky order should ask for exact preflight before execution: {dispatch_metadata}")
        if dispatch_metadata.get("can_auto_run") is not False or dispatch_metadata.get("approval_required") is not True:
            raise AssertionError(f"natural risky order missed safe dispatch metadata: {dispatch_metadata}")
        if "computer" not in dispatch_metadata.get("matched_risks", []) or "shell/code" not in dispatch_metadata.get("matched_risks", []):
            raise AssertionError(f"natural risky order missed risk signals: {dispatch_metadata}")
        for key in ["calls_model", "executes_tools", "queues_approval", "approves_request", "dismisses_request", "reads_private_data", "writes_files", "controls_computer", "calls_external_services"]:
            if dispatch_metadata.get(key):
                raise AssertionError(f"natural risky dispatch should report {key}=False")
        if "Dispatch decision: ASK_FOR_EXACT_ACTION_OR_PREFLIGHT" not in natural_risky_order.response:
            raise AssertionError("natural risky order response missed dispatch decision")
        if "execution readiness matrix" not in natural_risky_order.response:
            raise AssertionError("natural risky order response missed preflight route")

        risky_tool = runtime.handle("run command python3 --version")
        risky_trace = risky_tool.metadata.get("runtime_trace", {})
        if risky_trace.get("route") != "tools" or risky_trace.get("approval_required") is not True:
            raise AssertionError("risky tool runtime trace missed approval requirement")
        if not risky_trace.get("queued_approval_ids"):
            raise AssertionError("risky tool runtime trace missed queued approval id")
        if risky_trace.get("ran_tool_handlers"):
            raise AssertionError("risky tool runtime trace should not record handler execution before approval")
        if "HIGH_RISK" not in risky_trace.get("risk_levels", []):
            raise AssertionError("risky tool runtime trace missed HIGH_RISK level")
        duplicate_diagnosis = runtime.handle("command diagnosis: run command python3 --version")
        duplicate_metadata = duplicate_diagnosis.tool_results[0].metadata
        if (
            duplicate_metadata.get("forecast_queue_before") != 1
            or duplicate_metadata.get("forecast_queue_after_if_sent") != 1
            or duplicate_metadata.get("forecast_queue_delta_if_sent") != 0
            or duplicate_metadata.get("forecast_new_approvals") != 0
            or duplicate_metadata.get("forecast_reused_approval_ids") != [1]
        ):
            raise AssertionError(f"duplicate command diagnosis missed approval reuse forecast: {duplicate_metadata}")
        duplicate_forecast = duplicate_metadata.get("approval_queue_forecast") or []
        if not duplicate_forecast or duplicate_forecast[0].get("existing_approval_id") != 1 or duplicate_forecast[0].get("would_reuse_pending_approval") is not True:
            raise AssertionError(f"duplicate command diagnosis missed per-action approval reuse: {duplicate_metadata}")
        for expected in [
            "would queue new approvals if sent: 0",
            "would reuse pending approval ids: 1",
            "forecast queue after if sent: 1",
        ]:
            if expected not in duplicate_diagnosis.response:
                raise AssertionError(f"duplicate command diagnosis missed reuse text: {expected}")

    with TemporaryDirectory(prefix="jarvis-chat-diagnosis-closure-") as temp:
        root = Path(temp)
        runtime = JarvisRuntime(
            JarvisConfig(
                data_dir=root,
                db_path=root / "jarvis.sqlite",
                obsidian_vault=root / "Vault",
                obsidian_root="Jarvis",
                chat_model="jarvis-v2-missing-smoke-model",
                use_model_planner=False,
            )
        )
        run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "checkpoint_recovery_execute",
            "LOCAL_SAFE",
            False,
            False,
            "reviewed local-safe step failed verification",
            metadata={"failure_kind": "verification_failed"},
        )
        diagnosis = runtime.diagnose_command("remember that diagnosis should close recovery first")
        if diagnosis.get("route") != "recovery_closure" or diagnosis.get("recommendation") != "RECOVERY_CLOSURE_REQUIRED":
            raise AssertionError(f"runtime diagnosis should hold local-safe work behind recovery closure: {diagnosis}")
        assert_empty_planner_diagnostics(diagnosis, "runtime recovery diagnosis")
        if diagnosis.get("safe_to_execute_now") is not False:
            raise AssertionError(f"runtime diagnosis missed recovery safe-to-execute blocker: {diagnosis}")
        for expected in [f"verification receipt {run_id}", f"execution recovery packet {run_id}", f"execution learning closure {run_id}", f"after-action learning packet {run_id}"]:
            if expected not in diagnosis.get("recovery_closure_required_commands", []):
                raise AssertionError(f"runtime diagnosis missed required closure command {expected!r}: {diagnosis}")
        closure_commands = diagnosis.get("recovery_closure_required_commands", [])
        learning_closure = f"execution learning closure {run_id}"
        after_action = f"after-action learning packet {run_id}"
        if closure_commands.index(after_action) > closure_commands.index(learning_closure):
            raise AssertionError(f"runtime diagnosis should require after-action learning before learning closure: {diagnosis}")
        if diagnosis.get("recovery_closure_next_required_command") != closure_commands[0]:
            raise AssertionError(f"runtime diagnosis missed next required recovery proof: {diagnosis}")
        if diagnosis.get("recovery_closure_proof_queue") != closure_commands:
            raise AssertionError(f"runtime diagnosis recovery proof queue diverged from commands: {diagnosis}")
        if diagnosis.get("recovery_closure_proof_queue_count") != len(closure_commands):
            raise AssertionError(f"runtime diagnosis recovery proof queue count diverged: {diagnosis}")
        if diagnosis.get("recovery_closure_next_proof_command") != closure_commands[0]:
            raise AssertionError(f"runtime diagnosis missed next recovery proof alias: {diagnosis}")
        chat_diagnosis = runtime.diagnose_command("can we talk about memory?")
        if chat_diagnosis.get("route") != "chat" or chat_diagnosis.get("safe_to_execute_now") is not True:
            raise AssertionError(f"runtime diagnosis should not recovery-block pure chat: {chat_diagnosis}")
        assert_empty_planner_diagnostics(chat_diagnosis, "runtime chat diagnosis")

    with TemporaryDirectory(prefix="jarvis-runtime-diagnosis-redaction-") as temp:
        root = Path(temp)
        runtime = JarvisRuntime(
            JarvisConfig(
                data_dir=root,
                db_path=root / "jarvis.sqlite",
                obsidian_vault=root / "Vault",
                obsidian_root="Jarvis",
                chat_model="jarvis-v2-missing-smoke-model",
                use_model_planner=False,
            )
        )
        path_diagnosis = runtime.diagnose_command(
            "run command cat /\x55sers/example/private-runtime-diagnosis.txt /private/tmp/runtime-token.txt /var/folders/zc/runtime-cache.txt /tmp/runtime-raw.txt"
        )
        diagnosis_text = str(path_diagnosis)
        for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
            if forbidden in diagnosis_text:
                raise AssertionError(f"runtime diagnosis leaked local path fragment {forbidden!r}: {path_diagnosis}")
        if "<local-path>" not in diagnosis_text:
            raise AssertionError(f"runtime diagnosis should retain a redacted local-path marker: {path_diagnosis}")
        if path_diagnosis.get("request") != "run command cat <local-path>":
            raise AssertionError(f"runtime diagnosis missed redacted request: {path_diagnosis}")
        if path_diagnosis.get("next_command") != "execution governor: run command cat <local-path>":
            raise AssertionError(f"runtime diagnosis missed redacted next command: {path_diagnosis}")
        assert_empty_planner_diagnostics(path_diagnosis, "runtime path diagnosis")
        if "execution governor: run command cat <local-path>" not in path_diagnosis.get("recommended_next_commands", []):
            raise AssertionError(f"runtime diagnosis missed redacted recommended command: {path_diagnosis}")
        planned_actions = path_diagnosis.get("planned_actions", [])
        if not planned_actions or planned_actions[0].get("args", {}).get("command") != "cat <local-path>":
            raise AssertionError(f"runtime diagnosis missed redacted planned command arg: {path_diagnosis}")


if __name__ == "__main__":
    main()
