"""Focused offline proof for brain-tool input failure guidance."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.failure_guidance import LOCAL_READ_INPUT_RECOVERY_ACTION
from jarvis_v2.scripts.test_runtime import make_temp_runtime


PRIVATE_MARKERS = (
    "/\x55sers/example/private/brain.txt",
    "/private/brain.txt",
    "/var/folders/brain.txt",
    "/tmp/brain.txt",
    "sk_" + "live_BRAINSECRET123456",
    "traceback",
)
KNOWN_READ_FAILURE_FIELDS = {
    "outcome_known": True,
    "outcome_unknown": False,
    "execution_outcome_unknown": False,
    "side_effect_possible": False,
    "retry_safe": True,
    "automatic_retry_allowed": False,
    "authorizes_retry": False,
}


def _assert_failure(result, *, tool_name: str, reason: str, label: str) -> None:
    if result.ok or result.tool_name != tool_name:
        raise SystemExit(f"{label} should be a {tool_name} failure: {result}")
    if LOCAL_READ_INPUT_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} hid its recovery action: {result.output}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": LOCAL_READ_INPUT_RECOVERY_ACTION,
        "commands": [],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    if result.metadata.get("refusal_reason") != reason:
        raise SystemExit(f"{label} refusal reason drifted: {result.metadata}")
    for key, expected in KNOWN_READ_FAILURE_FIELDS.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} outcome field {key} drifted: {result.metadata}")
    for key in (
        "calls_model",
        "writes_memory",
        "writes_files",
        "writes_notes",
        "executes_tools",
        "executes_side_effect",
        "external_calls",
        "controls_computer",
        "queues_approval",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "speaks",
        "completes_tasks",
    ):
        if result.metadata.get(key) is not False:
            raise SystemExit(f"{label} changed the read-only boundary {key}: {result.metadata}")
    public = f"{result.output}\n{result.metadata}".lower()
    for marker in PRIVATE_MARKERS:
        if marker.lower() in public:
            raise SystemExit(f"{label} leaked private failure detail: {public}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-brain-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        brain_search = runtime.registry.get("brain_search").handler
        brain_think = runtime.registry.get("brain_think").handler
        brain_neighbors = runtime.registry.get("brain_neighbors").handler
        cases = (
            (brain_search({}), "brain_search", "missing_query", "missing search query"),
            (
                brain_search({"query": "/\x55sers/example/private/brain.txt"}),
                "brain_search",
                "local_path_query",
                "path-shaped search query",
            ),
            (brain_think({}), "brain_think", "missing_question", "missing think question"),
            (
                brain_think({"question": "/tmp/brain.txt"}),
                "brain_think",
                "local_path_question",
                "path-shaped think question",
            ),
            (
                brain_neighbors({"memory_id": True}),
                "brain_neighbors",
                "bad_memory_id",
                "boolean memory id",
            ),
            (
                brain_neighbors({"memory_id": "memory-abc"}),
                "brain_neighbors",
                "bad_memory_id",
                "non-numeric memory id",
            ),
            (
                brain_neighbors({"memory_id": 0}),
                "brain_neighbors",
                "bad_memory_id",
                "non-positive memory id",
            ),
            (
                brain_neighbors({"memory_id": "sk_" + "live_BRAINSECRET123456"}),
                "brain_neighbors",
                "bad_memory_id",
                "private-looking memory id",
            ),
        )
        if len(cases) != 8:
            raise SystemExit(f"brain guidance scope drifted: {len(cases)}/8")
        for result, tool_name, reason, label in cases:
            _assert_failure(result, tool_name=tool_name, reason=reason, label=label)
        if cases[-1][0].metadata.get("raw_memory_id") != "<private-value>":
            raise SystemExit(
                "private-looking memory id was not replaced with the bounded placeholder: "
                f"{cases[-1][0].metadata}"
            )

    print("Brain error-guidance smoke passed: 8 offline input failures.")


if __name__ == "__main__":
    main()
