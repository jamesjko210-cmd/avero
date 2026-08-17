from __future__ import annotations

import contextlib
import io
from unittest.mock import patch

from jarvis_v2.scripts.live_test_v3_model_chat import PROOF_PROMPT, run_proof


def main() -> None:
    captured_messages: list[list[dict[str, str]]] = []

    def fake_model_text(**kwargs: object) -> str:
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            raise AssertionError("model proof did not provide a message list")
        captured_messages.append(messages)
        return "I am Jarvis V3, and safety comes first."

    stdout = io.StringIO()
    with (
        contextlib.redirect_stdout(stdout),
        patch("jarvis_v2.agent.chat.generate_model_text", side_effect=fake_model_text),
    ):
        receipt, exit_code = run_proof(model="offline-proof-model", timeout_seconds=2.0)

    if stdout.getvalue():
        raise SystemExit("model proof helper printed model content instead of returning a content-free receipt")
    if exit_code != 0 or receipt.get("passed") is not True:
        raise SystemExit(f"model proof harness did not accept the deterministic model path: {receipt}")
    if receipt.get("content_recorded") is not False or "response" in receipt:
        raise SystemExit(f"model proof receipt retained response content: {receipt}")
    if len(captured_messages) != 1:
        raise SystemExit(f"model proof should make exactly one model call: {len(captured_messages)}")
    flattened = [message for message in captured_messages[0] if isinstance(message, dict)]
    user_messages = [message.get("content") for message in flattened if message.get("role") == "user"]
    if user_messages != [PROOF_PROMPT]:
        raise SystemExit(f"model proof sent unexpected user content: {user_messages!r}")
    checks = receipt.get("checks")
    if not isinstance(checks, dict) or not all(value is True for value in checks.values()):
        raise SystemExit(f"model proof checks were not all true: {checks}")
    print("V3 configured-model chat proof harness smoke passed")


if __name__ == "__main__":
    main()
