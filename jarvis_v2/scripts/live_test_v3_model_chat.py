from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig


PROOF_PROMPT = (
    "In one short sentence, introduce yourself as Jarvis V3 and mention that safety comes first."
)
EXPECTED_PROVIDER = "ollama"
EXPECTED_DESTINATION_POLICY = "ollama_loopback"


def _safe_latency(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    latency = float(value)
    if not math.isfinite(latency) or latency < 0:
        return None
    return round(latency, 1)


def run_proof(*, model: str, timeout_seconds: float) -> tuple[dict[str, object], int]:
    with TemporaryDirectory(prefix="jarvis-v3-model-chat-") as temp:
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
            model_provider=EXPECTED_PROVIDER,
            allow_remote_conversation_compaction=False,
            allow_remote_personal_context=False,
            watched_dirs=(),
        )
        safe_model_environment = {
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_NO_CLOUD": "1",
            "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "0",
            "JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT": "0",
            "JARVIS_ALLOW_REMOTE_COMPACTION": "0",
            "JARVIS_DISABLE_STORAGE_FALLBACK": "1",
        }
        with patch.dict(os.environ, safe_model_environment, clear=False):
            runtime = JarvisRuntime(config)
            result = runtime.handle(PROOF_PROMPT)

        chat = result.metadata.get("chat_response")
        if not isinstance(chat, dict):
            chat = {}
        latency_ms = _safe_latency(chat.get("latency_ms"))
        pending_approvals = runtime.store.list_pending_approvals(status="pending")
        checks = {
            "runtime_route_chat": result.metadata.get("runtime_route") == "chat",
            "response_nonempty": bool(result.response.strip()),
            "source_model": chat.get("source") == "model",
            "used_model": chat.get("used_model") is True,
            "no_fallback": chat.get("used_fallback") is False,
            "response_received": chat.get("model_execution_status") == "response_received",
            "provider_ollama": chat.get("model_provider") == EXPECTED_PROVIDER,
            "loopback_destination": chat.get("model_destination_policy") == EXPECTED_DESTINATION_POLICY,
            "model_matches": chat.get("model") == model,
            "latency_recorded": latency_ms is not None and chat.get("latency_recorded") is True,
            "no_tool_handlers": not result.plan.actions and not result.tool_results,
            "no_approval_queued": not pending_approvals,
            "primary_storage": runtime.storage_fallback is None,
            "stored_context_withheld": chat.get("stored_personal_context_in_model_request") is False,
            "history_withheld": chat.get("history_in_model_request") is False,
            "no_external_request": chat.get("external_model_request_attempted") is False,
        }
        passed = all(checks.values())
        receipt: dict[str, object] = {
            "proof": "jarvis_v3_configured_model_chat",
            "passed": passed,
            "provider": EXPECTED_PROVIDER,
            "model": model,
            "destination_policy": EXPECTED_DESTINATION_POLICY,
            "latency_ms": latency_ms,
            "checks": checks,
            "content_recorded": False,
            "temporary_storage_removed_on_exit": True,
            "model_execution_locality_verified": chat.get("model_execution_locality_verified") is True,
            "locality_limitation": (
                "Loopback routing and OLLAMA_NO_CLOUD express the requested policy but do not attest "
                "the daemon's model-execution location."
            ),
        }
        return receipt, 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one content-free, isolated Jarvis V3 configured-model chat proof."
    )
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    receipt, exit_code = run_proof(model=args.model, timeout_seconds=max(1.0, min(120.0, args.timeout)))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
