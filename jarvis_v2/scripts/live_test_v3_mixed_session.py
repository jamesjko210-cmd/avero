from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.scripts import live_check


TEST_PHRASE = "Silver Compass"
TASK_MARKER = "review the Silver Compass safety checklist"
CHAT_PROMPTS = (
    "For this synthetic conversation, use Silver Compass as our test phrase and reply with it.",
    "What test phrase did I give you earlier? Reply with only that phrase.",
    "In one short sentence, connect our test phrase to safety-first assistance.",
)


def _stub_tool(tool_name: str, output: str):
    def handler(_args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool_name,
            True,
            output,
            {
                "calls_external_service": False,
                "reads_personal_data": False,
                "external_side_effect": False,
                "synthetic_acceptance_fixture": True,
            },
        )

    return handler


def _replace_handler(runtime: JarvisRuntime, tool_name: str, output: str) -> None:
    tool = runtime.registry.get(tool_name)
    runtime.registry._tools[tool_name] = replace(tool, handler=_stub_tool(tool_name, output))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(ordered[lower], 1)
    weight = index - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 1)


def run_proof(*, model: str, timeout_seconds: float) -> tuple[dict[str, object], int]:
    with TemporaryDirectory(prefix="jarvis-v3-mixed-session-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root / "data",
            db_path=root / "data" / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model=model,
            planner_model=model,
            chat_timeout_seconds=timeout_seconds,
            use_model_planner=False,
            model_provider="ollama",
            allow_remote_conversation_compaction=False,
            allow_remote_personal_context=False,
            watched_dirs=(),
        )
        safe_environment = {
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_NO_CLOUD": "1",
            # The only retained context is this temporary synthetic session. This consent does not
            # authorize personal profile, memory, connector, or production-history disclosure.
            "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
            "JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT": "0",
            "JARVIS_ALLOW_REMOTE_COMPACTION": "0",
            "JARVIS_DISABLE_STORAGE_FALLBACK": "1",
        }
        with patch.dict(os.environ, safe_environment, clear=False):
            runtime = JarvisRuntime(config)
            for name, output in (
                ("research", "Synthetic research fixture: Silver Compass emphasizes bounded verification."),
                ("web_search", "Synthetic search fixture: independent safety evidence is available."),
                ("list_events", "Synthetic calendar fixture: no events tomorrow."),
                ("check_availability", "Synthetic calendar fixture: tomorrow afternoon is available."),
            ):
                _replace_handler(runtime, name, output)

            cases = (
                (CHAT_PROMPTS[0], None),
                (f"add task {TASK_MARKER}", "add_task"),
                ("list tasks", "list_tasks"),
                ("research safe AI agent verification", "research"),
                ("web search for safe AI agent testing", "web_search"),
                ("what is on my calendar tomorrow", "list_events"),
                ("am I free tomorrow afternoon", "check_availability"),
                (CHAT_PROMPTS[1], None),
                ("jarvis status", "jarvis_status"),
                (CHAT_PROMPTS[2], None),
            )
            results = []
            expected_tool_routes = True
            approval_free = True
            for message, expected_tool in cases:
                result = runtime.handle(message)
                results.append(result)
                tool_names = [item.tool_name for item in result.tool_results]
                if expected_tool is None:
                    expected_tool_routes = expected_tool_routes and not tool_names
                else:
                    expected_tool_routes = expected_tool_routes and tool_names == [expected_tool]
                approval_free = approval_free and not runtime.store.list_pending_approvals(status="pending")

        chat_results = [result for result in results if result.metadata.get("runtime_route") == "chat"]
        chat_metadata = [result.metadata.get("chat_response") for result in chat_results]
        chat_metadata = [item for item in chat_metadata if isinstance(item, dict)]
        latencies = [
            float(item["latency_ms"])
            for item in chat_metadata
            if isinstance(item.get("latency_ms"), (int, float))
            and not isinstance(item.get("latency_ms"), bool)
            and math.isfinite(float(item["latency_ms"]))
        ]
        p95_ms = _percentile(latencies, 0.95)
        task_results = [result for result in results if any(item.tool_name == "list_tasks" for item in result.tool_results)]
        phrase_coherent = len(chat_results) == 3 and all(
            TEST_PHRASE.casefold() in result.response.casefold() for result in chat_results
        )
        task_coherent = len(task_results) == 1 and TASK_MARKER in task_results[0].response
        mixed_ok, mixed_detail = live_check._check_mixed_conversation_health(config)
        checks = {
            "ten_assistant_turns": len(results) >= 10,
            "three_model_chat_turns": len(chat_metadata) == 3
            and all(item.get("used_model") is True for item in chat_metadata),
            "no_chat_fallback": all(item.get("used_fallback") is False for item in chat_metadata),
            "three_latency_samples": len(latencies) >= 3,
            "p95_within_target": p95_ms is not None and p95_ms <= 8000.0,
            "required_tool_routes": expected_tool_routes,
            "phrase_coherent": phrase_coherent,
            "task_state_coherent": task_coherent,
            "live_check_mixed_gate": mixed_ok is True,
            "no_approval_queued": approval_free,
            "primary_storage": runtime.storage_fallback is None,
            "synthetic_connectors_only": all(
                item.metadata.get("synthetic_acceptance_fixture") is True
                for result in results
                for item in result.tool_results
                if item.tool_name in {"research", "web_search", "list_events", "check_availability"}
            ),
        }
        passed = all(checks.values())
        receipt: dict[str, object] = {
            "proof": "jarvis_v3_mixed_session",
            "passed": passed,
            "assistant_turns": len(results),
            "model_chat_turns": len(chat_metadata),
            "latency_samples": len(latencies),
            "chat_p95_ms": p95_ms,
            "required_lanes": ["chat", "research", "calendar", "tasks"],
            "checks": checks,
            "content_recorded": False,
            "connector_mode": "synthetic_no_network_no_account",
            "temporary_storage_removed_on_exit": True,
            "model_execution_locality_verified": all(
                item.get("model_execution_locality_verified") is True for item in chat_metadata
            ),
            "locality_limitation": (
                "Loopback routing and OLLAMA_NO_CLOUD express the requested policy but do not attest "
                "the daemon's model-execution location."
            ),
            "mixed_health_detail_recorded": False,
        }
        if mixed_ok is not True and isinstance(mixed_detail, str):
            receipt["mixed_health_state"] = "not_ready"
        return receipt, 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an isolated 10-turn Jarvis V3 mixed-session acceptance proof."
    )
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    receipt, exit_code = run_proof(model=args.model, timeout_seconds=max(1.0, min(120.0, args.timeout)))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
