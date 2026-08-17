"""Offline canonical-guidance coverage for decision refusals."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.decisions import MAX_DECISION_OUTCOME_SUMMARY_CHARS, make_decision_tools


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        runtime = make_temp_runtime(Path(temp_dir))
        record, outcome, list_decisions, get_decision, set_status = make_decision_tools(
            runtime.store,
            runtime.vault,
        )
        cases = [
            record({"title": ""}),
            record({"title": "/\x55sers/example/private/choice"}),
            outcome({"decision_id": 0, "summary": "worked"}),
            outcome({"decision_id": 1, "summary": 123}),
            outcome({"decision_id": 1, "summary": ""}),
            outcome({"decision_id": 1, "summary": "x" * (MAX_DECISION_OUTCOME_SUMMARY_CHARS + 1)}),
            outcome({"decision_id": 1, "summary": "/private/result.txt"}),
            list_decisions({"status": "invalid"}),
            get_decision({"decision_id": "bad"}),
            get_decision({"decision_id": 999}),
            set_status({"decision_id": "bad", "status": "active"}),
            set_status({"decision_id": 1, "status": "invalid"}),
            set_status({"decision_id": 999, "status": "active"}),
        ]

    for result in cases:
        if result.ok:
            raise SystemExit(f"decision refusal unexpectedly succeeded: {result}")
        guidance = result.metadata.get("recovery_guidance")
        if not isinstance(guidance, dict) or guidance.get("version") != 1:
            raise SystemExit(f"decision refusal missed canonical guidance: {result.metadata}")
        action = guidance.get("action")
        if not isinstance(action, str) or action not in result.output:
            raise SystemExit(f"decision guidance action is not public: {result}")
        for key, expected in {
            "outcome_known": True,
            "outcome_unknown": False,
            "side_effect_possible": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }.items():
            if result.metadata.get(key) is not expected:
                raise SystemExit(f"decision refusal has unsafe {key}: {result.metadata}")
        if "/\x55sers/" in result.output or "/private/" in result.output or "/tmp/" in result.output:
            raise SystemExit(f"decision refusal leaked a local path: {result.output}")

    print("Decision error-guidance smoke passed: 13 refusal branches")


if __name__ == "__main__":
    main()
