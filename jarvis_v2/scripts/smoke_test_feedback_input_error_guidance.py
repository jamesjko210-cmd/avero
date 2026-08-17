"""Offline canonical-guidance coverage for feedback input refusals."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.feedback import MAX_FEEDBACK_BODY_CHARS, make_feedback_tools


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        runtime = make_temp_runtime(Path(temp_dir))
        record_feedback = make_feedback_tools(runtime.store, runtime.vault)[0]
        cases = [
            record_feedback({"body": ""}),
            record_feedback({"body": "x" * (MAX_FEEDBACK_BODY_CHARS + 1)}),
            record_feedback({"body": "review /\x55sers/example/private/feedback.txt"}),
            record_feedback({"body": "clear wording", "theme": "/tmp/private-theme"}),
            record_feedback({"body": "clear wording", "title": "/private/feedback-title"}),
        ]

    expected_reasons = {
        "missing_body",
        "body_too_large",
        "invalid_body",
        "invalid_theme",
        "invalid_title",
    }
    seen: set[str] = set()
    for result in cases:
        if result.ok:
            raise SystemExit(f"feedback refusal unexpectedly succeeded: {result}")
        reason = result.metadata.get("reason")
        if isinstance(reason, str):
            seen.add(reason)
        guidance = result.metadata.get("recovery_guidance")
        if not isinstance(guidance, dict) or guidance.get("version") != 1:
            raise SystemExit(f"feedback refusal missed canonical guidance: {result.metadata}")
        action = guidance.get("action")
        if not isinstance(action, str) or action not in result.output:
            raise SystemExit(f"feedback recovery action is not public: {result}")
        for key, expected in {
            "outcome_known": True,
            "outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }.items():
            if result.metadata.get(key) is not expected:
                raise SystemExit(f"feedback refusal has unsafe {key}: {result.metadata}")
        if "/\x55sers/" in result.output or "/private/" in result.output or "/tmp/" in result.output:
            raise SystemExit(f"feedback refusal leaked a local path: {result.output}")
    if seen != expected_reasons:
        raise SystemExit(f"feedback refusal coverage drifted: {seen}")

    print("Feedback input error-guidance smoke passed: 5 refusal branches")


if __name__ == "__main__":
    main()
