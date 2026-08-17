"""Focused offline proof for utility input failure guidance."""

from __future__ import annotations

from jarvis_v2.agent.failure_guidance import LOCAL_READ_INPUT_RECOVERY_ACTION
from jarvis_v2.config import load_config
from jarvis_v2.tools import utilities
from jarvis_v2.tools.countdown_connector import make_countdown_tools


PRIVATE_MARKERS = (
    "/\x55sers/example/private/input.txt",
    "/private/input.txt",
    "/var/folders/input.txt",
    "/tmp/input.txt",
    "sk_" + "live_SUPERSECRET123",
    "traceback",
)


def _assert_failure(result, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(
                f"{label} utility outcome field {key} drifted: {result.metadata}"
            )
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": LOCAL_READ_INPUT_RECOVERY_ACTION,
        "commands": [],
    }:
        raise SystemExit(f"{label} utility guidance drifted: {result.metadata}")
    if LOCAL_READ_INPUT_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} hid the recovery action: {result.output}")
    public_proof = f"{result.output}\n{result.metadata}".lower()
    for marker in PRIVATE_MARKERS:
        if marker.lower() in public_proof:
            raise SystemExit(f"{label} leaked private failure detail: {public_proof}")


def test_mocked_offline_utility_failures() -> None:
    cases = (
        (utilities.calculate({}), "calculate missing expression"),
        (
            utilities.calculate({"expression": "1" * 501}),
            "calculate oversized expression",
        ),
        (
            utilities.calculate({"expression": "/private/input.txt"}),
            "calculate local path",
        ),
        (
            utilities.calculate({"expression": "__import__('os')"}),
            "calculate private name",
        ),
        (utilities.spell_word({}), "spell missing text"),
        (
            utilities.spell_word({"word": "/tmp/input.txt"}),
            "spell local path",
        ),
        (utilities.spell_word({"word": "@#$"}), "spell invalid text"),
        (utilities.count_text({}), "count missing text"),
        (
            utilities.count_text({"text": "/\x55sers/example/private/input.txt"}),
            "count local path",
        ),
        (
            make_countdown_tools(load_config())[0].handler({}),
            "countdown missing target",
        ),
    )
    if len(cases) != 10:
        raise SystemExit(f"offline utility mocked scope drifted: {len(cases)}/10")
    for result, label in cases:
        _assert_failure(result, label)


def main() -> None:
    test_mocked_offline_utility_failures()
    print("Offline utility error-guidance smoke passed: 10 branches across 4 tools.")


if __name__ == "__main__":
    main()
