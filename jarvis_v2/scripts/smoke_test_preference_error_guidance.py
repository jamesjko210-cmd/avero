"""Offline canonical-guidance coverage for preference refusals."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.preferences import make_preference_tools


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        runtime = make_temp_runtime(Path(temp_dir))
        set_preference, list_preferences, get_preference, set_status = make_preference_tools(
            runtime.store,
            runtime.vault,
        )
        cases = [
            set_preference({"key": "", "value": "concise"}),
            set_preference({"key": "/\x55sers/example/private/key", "value": "concise"}),
            set_preference({"key": "response style", "value": ""}),
            list_preferences({"status": "invalid"}),
            get_preference({"key": ""}),
            get_preference({"key": "/private/preference-key"}),
            set_status({"preference_id": "bad", "status": "active"}),
            set_status({"preference_id": 1, "status": "invalid"}),
            set_status({"preference_id": 999, "status": "active"}),
        ]

    for result in cases:
        if result.ok:
            raise SystemExit(f"preference refusal unexpectedly succeeded: {result}")
        guidance = result.metadata.get("recovery_guidance")
        if not isinstance(guidance, dict) or guidance.get("version") != 1:
            raise SystemExit(f"preference refusal missed canonical guidance: {result.metadata}")
        action = guidance.get("action")
        if not isinstance(action, str) or action not in result.output:
            raise SystemExit(f"preference guidance action is not public: {result}")
        for key, expected in {
            "outcome_known": True,
            "outcome_unknown": False,
            "side_effect_possible": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }.items():
            if result.metadata.get(key) is not expected:
                raise SystemExit(f"preference refusal has unsafe {key}: {result.metadata}")
        if "/\x55sers/" in result.output or "/private/" in result.output or "/tmp/" in result.output:
            raise SystemExit(f"preference refusal leaked a local path: {result.output}")

    print("Preference error-guidance smoke passed: 9 refusal branches")


if __name__ == "__main__":
    main()
