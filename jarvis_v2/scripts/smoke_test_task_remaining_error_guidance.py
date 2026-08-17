from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import tasks as task_tools


PRIVATE_MARKERS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")


def _assert_guided(result: Any, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("version") != 1:
        raise SystemExit(f"{label} omitted canonical recovery guidance: {result}")
    action = guidance.get("action")
    commands = guidance.get("commands")
    if action not in {
        task_tools.TASK_INPUT_RECOVERY_ACTION,
        task_tools.TASK_STATE_RECOVERY_ACTION,
    }:
        raise SystemExit(f"{label} used the wrong recovery action: {result}")
    if action not in result.output or not isinstance(commands, list):
        raise SystemExit(f"{label} hid its recovery declaration: {result}")
    if any(command not in result.output for command in commands):
        raise SystemExit(f"{label} hid a declared recovery command: {result}")
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
            raise SystemExit(f"{label} field {key} drifted: {result.metadata}")
    for key in (
        "queues_approval",
        "approves_request",
        "dismisses_request",
        "controls_computer",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "state_changed",
        "side_effect_possible",
    ):
        if result.metadata.get(key):
            raise SystemExit(f"{label} unexpectedly set {key}: {result.metadata}")
    if any(marker in result.output or marker in str(result.metadata) for marker in PRIVATE_MARKERS):
        raise SystemExit(f"{label} leaked a private local path: {result}")


def _closure(*, blocking: bool) -> dict[str, Any]:
    return {
        "state": "blocked" if blocking else "clear",
        "ready_to_retry": not blocking,
        "missing": ["status"] if blocking else [],
        "missing_count": 1 if blocking else 0,
        "required_commands": ["recovery closure status"] if blocking else [],
        "next_required_command": "recovery closure status" if blocking else "",
        "blocks_task_completion": blocking,
        "target_run_id": 7 if blocking else None,
        "target_tool_name": "example_tool" if blocking else "",
    }


def _runtime_cases(runtime: Any, task_id: int) -> list[tuple[str, Any]]:
    handler = lambda name, args: runtime.registry.get(name).handler(args)
    cases = [
        ("add missing body", handler("add_task", {"body": ""})),
        ("add bad priority", handler("add_task", {"body": "x", "priority": "urgent"})),
        ("complete bad id", handler("complete_task", {"task_id": "abc"})),
        ("completion packet bad id", handler("task_completion_packet", {"task_id": "abc"})),
        ("evidence completion bad id", handler("complete_task_with_evidence", {"task_id": "abc"})),
        (
            "invalid completion evidence",
            handler("complete_task_with_evidence", {"task_id": task_id, "evidence": ""}),
        ),
        ("status bad id", handler("update_task_status", {"task_id": "abc", "status": "done"})),
        ("status bad value", handler("update_task_status", {"task_id": task_id, "status": "maybe"})),
        ("details bad id", handler("update_task_details", {"task_id": "abc", "body": "x"})),
        ("details empty body", handler("update_task_details", {"task_id": task_id, "body": ""})),
        ("details bad priority", handler("update_task_details", {"task_id": task_id, "priority": "urgent"})),
        ("details invalid unicode", handler("update_task_details", {"task_id": task_id, "body": "\ud800"})),
        ("details missing update", handler("update_task_details", {"task_id": task_id})),
    ]

    original_snapshot = task_tools._execution_health_recovery_closure_snapshot
    original_complete = runtime.store.complete_task_with_evidence
    try:
        task_tools._execution_health_recovery_closure_snapshot = lambda _runs: _closure(blocking=True)
        cases.append(
            (
                "completion recovery closure",
                handler(
                    "complete_task_with_evidence",
                    {"task_id": task_id, "evidence": "proof"},
                ),
            )
        )

        task_tools._execution_health_recovery_closure_snapshot = lambda _runs: _closure(blocking=False)

        def state_changed(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("task completion recovery state changed")

        runtime.store.complete_task_with_evidence = state_changed  # type: ignore[method-assign]
        cases.append(
            (
                "completion state changed",
                handler(
                    "complete_task_with_evidence",
                    {"task_id": task_id, "evidence": "proof"},
                ),
            )
        )
    finally:
        task_tools._execution_health_recovery_closure_snapshot = original_snapshot
        runtime.store.complete_task_with_evidence = original_complete  # type: ignore[method-assign]
    return cases


def _preflight_cases() -> list[tuple[str, Any]]:
    return [
        (
            "preflight status bad value",
            task_tools.update_task_status_auto_mutation_preflight_result(
                {"task_id": 1, "status": "maybe"}, "bad_status"
            ),
        ),
        (
            "preflight status bad id",
            task_tools.update_task_status_auto_mutation_preflight_result(
                {"task_id": "abc", "status": "done"}, "bad_task_id"
            ),
        ),
        (
            "preflight evidence invalid",
            task_tools.complete_task_with_evidence_auto_mutation_preflight_result(
                {"task_id": 1}, "missing_evidence"
            ),
        ),
        (
            "preflight evidence recovery closure",
            task_tools.complete_task_with_evidence_auto_mutation_preflight_result(
                {"task_id": 1}, "recovery_closure_required"
            ),
        ),
        (
            "preflight evidence bad id",
            task_tools.complete_task_with_evidence_auto_mutation_preflight_result(
                {"task_id": "abc"}, "bad_task_id"
            ),
        ),
        (
            "preflight details empty body",
            task_tools.update_task_details_auto_mutation_preflight_result(
                {"task_id": 1, "body": ""}, "missing_body"
            ),
        ),
        (
            "preflight details bad priority",
            task_tools.update_task_details_auto_mutation_preflight_result(
                {"task_id": 1, "priority": "urgent"}, "bad_priority"
            ),
        ),
        (
            "preflight details missing update",
            task_tools.update_task_details_auto_mutation_preflight_result(
                {"task_id": 1}, "missing_update"
            ),
        ),
        (
            "preflight details invalid unicode",
            task_tools.update_task_details_auto_mutation_preflight_result(
                {"task_id": 1, "body": "\ud800"}, "invalid_unicode"
            ),
        ),
        (
            "preflight details bad id",
            task_tools.update_task_details_auto_mutation_preflight_result(
                {"task_id": "abc", "body": "x"}, "bad_task_id"
            ),
        ),
    ]


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-task-remaining-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        created = runtime.registry.get("add_task").handler({"body": "guided fixture"})
        if not created.ok:
            raise SystemExit(f"could not create task fixture: {created}")
        task_id = int(created.metadata["task_id"])
        cases = _runtime_cases(runtime, task_id) + _preflight_cases()
        if len(cases) != 25:
            raise SystemExit(f"task remaining-guidance branch inventory drifted: {len(cases)}")
        for label, result in cases:
            _assert_guided(result, label)

    print("Jarvis remaining task error-guidance smoke test passed (25 branches).")


if __name__ == "__main__":
    main()
