from __future__ import annotations

import contextlib
import io
from unittest.mock import patch

from jarvis_v2.scripts.live_test_v3_mixed_session import TEST_PHRASE, run_proof


def main() -> None:
    model_calls = 0

    def fake_model_text(**_kwargs: object) -> str:
        nonlocal model_calls
        model_calls += 1
        return f"{TEST_PHRASE}: safety-first assistance stays bounded and verified."

    stdout = io.StringIO()
    with (
        contextlib.redirect_stdout(stdout),
        patch("jarvis_v2.agent.chat.generate_model_text", side_effect=fake_model_text),
    ):
        receipt, exit_code = run_proof(model="offline-mixed-model", timeout_seconds=2.0)

    if stdout.getvalue():
        raise SystemExit("mixed-session helper printed conversation content")
    if exit_code != 0 or receipt.get("passed") is not True:
        raise SystemExit(f"mixed-session harness did not accept the deterministic path: {receipt}")
    if model_calls != 3 or receipt.get("model_chat_turns") != 3:
        raise SystemExit(f"mixed-session harness should make three chat model calls: {model_calls} {receipt}")
    if receipt.get("assistant_turns") != 10 or receipt.get("latency_samples") != 3:
        raise SystemExit(f"mixed-session counts drifted: {receipt}")
    if receipt.get("content_recorded") is not False or "response" in receipt:
        raise SystemExit(f"mixed-session receipt retained content: {receipt}")
    checks = receipt.get("checks")
    if not isinstance(checks, dict) or not all(value is True for value in checks.values()):
        raise SystemExit(f"mixed-session checks were not all true: {checks}")
    print("V3 mixed-session proof harness smoke passed")


if __name__ == "__main__":
    main()
