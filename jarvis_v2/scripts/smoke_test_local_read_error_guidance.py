"""Focused offline proof for local read/search failure guidance."""

from __future__ import annotations

from jarvis_v2.agent.failure_guidance import LOCAL_READ_INPUT_RECOVERY_ACTION
from jarvis_v2.tools.goals import make_goal_tools
from jarvis_v2.tools.people import make_people_tools
from jarvis_v2.tools.preferences import make_preference_tools
from jarvis_v2.tools.tasks import make_task_tools


PRIVATE_MARKERS = (
    "/\x55sers/example/private/read.txt",
    "/private/read.txt",
    "/var/folders/read.txt",
    "/tmp/read.txt",
    "sk_" + "live_SUPERSECRET123",
    "traceback",
)


def _assert_local_read_failure(result, label: str) -> None:
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
                f"{label} local-read outcome field {key} drifted: "
                f"{result.metadata}"
            )
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": LOCAL_READ_INPUT_RECOVERY_ACTION,
        "commands": [],
    }:
        raise SystemExit(f"{label} local-read guidance drifted: {result.metadata}")
    if LOCAL_READ_INPUT_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} hid the public recovery action: {result.output}")
    if result.metadata.get("next_command") and result.metadata.get(
        "next_command"
    ) not in result.output:
        # Existing local read metadata may retain a non-authorizing navigation
        # hint. It must never be confused with the canonical retry declaration.
        if result.metadata.get("recovery_commands"):
            raise SystemExit(f"{label} hid a recovery command: {result.metadata}")
    public_proof = f"{result.output}\n{result.metadata.get('recovery_guidance')}".lower()
    for marker in PRIVATE_MARKERS:
        if marker.lower() in public_proof:
            raise SystemExit(f"{label} leaked private guidance detail: {public_proof}")


def test_mocked_local_read_failures() -> None:
    # Every selected branch rejects before touching either dependency. Passing
    # sentinels makes that side-effect-free contract explicit and deterministic.
    sentinel_store = object()
    sentinel_vault = object()

    task_tools = make_task_tools(sentinel_store, sentinel_vault)  # type: ignore[arg-type]
    task_cases = (
        (task_tools[1]({"status": "invalid"}), "list tasks bad status"),
        (task_tools[2]({"task_id": "invalid"}), "inspect task bad id"),
        (task_tools[3]({}), "search tasks missing query"),
        (
            task_tools[3]({"query": "local", "status": "invalid"}),
            "search tasks bad status",
        ),
    )

    goal_tools = make_goal_tools(sentinel_store, sentinel_vault)  # type: ignore[arg-type]
    goal_cases = (
        (goal_tools[1]({"status": "invalid"}), "list goals bad status"),
        (goal_tools[2]({"goal_id": "invalid"}), "goal status bad id"),
    )

    people_tools = make_people_tools(sentinel_store, sentinel_vault)  # type: ignore[arg-type]
    people_cases = (
        (people_tools[2]({}), "get person missing selector"),
        (people_tools[2]({"person_id": "invalid"}), "get person bad id"),
    )

    preference_tools = make_preference_tools(sentinel_store, sentinel_vault)  # type: ignore[arg-type]
    preference_cases = (
        (
            preference_tools[1]({"status": "invalid"}),
            "list preferences bad status",
        ),
        (preference_tools[2]({}), "get preference missing key"),
    )

    cases = (*task_cases, *goal_cases, *people_cases, *preference_cases)
    if len(cases) != 10:
        raise SystemExit(f"local-read mocked scope drifted: {len(cases)}/10")
    for result, label in cases:
        _assert_local_read_failure(result, label)


def main() -> None:
    test_mocked_local_read_failures()
    print("Local read error-guidance smoke passed: 10 branches across 8 tools.")


if __name__ == "__main__":
    main()
