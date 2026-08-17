from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from jarvis_v2.automations.scheduler import Scheduler
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.scheduler import (
    SCHEDULER_INPUT_RECOVERY_ACTION,
    SCHEDULER_MUTATION_UNKNOWN_ACTION,
    SCHEDULER_NOT_FOUND_RECOVERY_ACTION,
    SCHEDULER_READ_RECOVERY_ACTION,
    SCHEDULER_RUN_INSPECTION_ACTION,
    SCHEDULER_RUN_NOT_STARTED_ACTION,
    make_scheduler_tools,
)


PRIVATE_MARKERS = (
    "/\x55sers/example/private/scheduler.sqlite",
    "/private/tmp/scheduler-secret",
    "sk_" + "live_schedulerFailureGuidanceSecret123",
)


def _tool_map(runtime) -> dict[str, object]:
    return {
        handler.__name__: handler
        for handler in make_scheduler_tools(runtime.store, runtime.vault, runtime.config)
    }


def _assert_private_safe(result, label: str) -> None:
    public = f"{result.output}\n{result.metadata}"
    if any(marker in public for marker in PRIVATE_MARKERS):
        raise SystemExit(f"{label} leaked private detail: {public}")
    if any(path in public for path in ("/\x55sers/", "/private/", "/tmp/")):
        raise SystemExit(f"{label} leaked a local path: {public}")


def _assert_guided(
    result,
    *,
    action: str,
    outcome_known: bool,
    side_effect_possible: bool,
    retry_safe: bool,
    state_changed,
    label: str,
) -> None:
    if result.ok:
        raise SystemExit(f"{label} should fail closed: {result}")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("action") != action:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result.metadata}")
    if action not in result.output:
        raise SystemExit(f"{label} hid its recovery action: {result.output}")
    expected = {
        "outcome_known": outcome_known,
        "outcome_unknown": not outcome_known,
        "execution_outcome_unknown": not outcome_known,
        "side_effect_possible": side_effect_possible,
        "retry_safe": retry_safe,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "state_changed": state_changed,
        "queues_approval": False,
        "controls_computer": False,
        "calls_model": False,
        "executes_tools": False,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} truth field {key} drifted: {result.metadata}")
    commands = guidance.get("commands")
    if not isinstance(commands, list) or any(command not in result.output for command in commands):
        raise SystemExit(f"{label} did not expose its declared commands: {result}")
    _assert_private_safe(result, label)


def _test_pre_mutation_and_read_failures(runtime, tools: dict[str, object]) -> int:
    count = 0
    with mock.patch.object(
        Scheduler,
        "schedule_morning_brief",
        side_effect=AssertionError("invalid input reached mutation"),
    ):
        invalid = tools["schedule_morning_brief"]({"time": PRIVATE_MARKERS[2]})
    _assert_guided(
        invalid,
        action=SCHEDULER_INPUT_RECOVERY_ACTION,
        outcome_known=True,
        side_effect_possible=False,
        retry_safe=True,
        state_changed=False,
        label="invalid morning brief time",
    )
    count += 1

    for tool_name in ("pause_job", "resume_job", "delete_job", "run_job_now"):
        blank = tools[tool_name]({"name": "   "})
        _assert_guided(
            blank,
            action=SCHEDULER_INPUT_RECOVERY_ACTION,
            outcome_known=True,
            side_effect_possible=False,
            retry_safe=True,
            state_changed=False,
            label=f"{tool_name} blank name",
        )
        count += 1

    with mock.patch.object(runtime.store, "list_jobs", side_effect=RuntimeError(PRIVATE_MARKERS[0])):
        for tool_name in ("list_scheduled_jobs", "scheduler_context_refresh_packet", "run_job_now"):
            failed_read = tools[tool_name]({"name": "Read Test"} if tool_name == "run_job_now" else {})
            _assert_guided(
                failed_read,
                action=SCHEDULER_READ_RECOVERY_ACTION,
                outcome_known=True,
                side_effect_possible=False,
                retry_safe=True,
                state_changed=False,
                label=f"{tool_name} storage read",
            )
            count += 1
    return count


def _test_mutation_failures(runtime, tools: dict[str, object]) -> int:
    cases = {
        "schedule_daily_brief": ("schedule_job", {}),
        "schedule_morning_brief": ("schedule_morning_brief", {"time": "07:30"}),
        "schedule_goal_nudge": ("schedule_job", {}),
        "schedule_weekly_review": ("schedule_job", {}),
        "schedule_inbox_ingest": ("schedule_job", {}),
        "schedule_file_digest": ("schedule_job", {}),
        "schedule_assistant_basics": ("schedule_job_if_missing", {}),
        "pause_job": ("pause_job", {"name": "Daily Brief"}),
        "resume_job": ("resume_job", {"name": "Daily Brief"}),
        "delete_job": ("delete_job", {"name": "Daily Brief"}),
    }
    for tool_name, (method_name, args) in cases.items():
        with mock.patch.object(
            Scheduler,
            method_name,
            side_effect=RuntimeError(PRIVATE_MARKERS[1]),
        ):
            result = tools[tool_name](args)
        _assert_guided(
            result,
            action=SCHEDULER_MUTATION_UNKNOWN_ACTION,
            outcome_known=False,
            side_effect_possible=True,
            retry_safe=False,
            state_changed=None,
            label=f"{tool_name} mutation exception",
        )
        if result.metadata.get("mutation_outcome") != "unknown" or result.metadata.get("mutates_scheduled_job") is not True:
            raise SystemExit(f"{tool_name} lost unknown mutation truth: {result.metadata}")

    for tool_name in ("pause_job", "resume_job", "delete_job"):
        with mock.patch.object(
            Scheduler,
            tool_name,
            return_value=f"No job found named {PRIVATE_MARKERS[2]}.",
        ):
            missing = tools[tool_name]({"name": PRIVATE_MARKERS[2]})
        _assert_guided(
            missing,
            action=SCHEDULER_NOT_FOUND_RECOVERY_ACTION,
            outcome_known=True,
            side_effect_possible=False,
            retry_safe=True,
            state_changed=False,
            label=f"{tool_name} missing job",
        )
        if missing.metadata.get("writes_database") or missing.metadata.get("mutates_scheduled_job"):
            raise SystemExit(f"{tool_name} missing resource claimed a mutation: {missing.metadata}")
    return len(cases) + 3


def _test_run_failures(runtime, tools: dict[str, object]) -> int:
    run_now = tools["run_job_now"]
    cases = (
        (
            "No job found named Missing Job.",
            SCHEDULER_NOT_FOUND_RECOVERY_ACTION,
            True,
            False,
            True,
            False,
        ),
        (
            "Job Busy Job is already running; run skipped.",
            SCHEDULER_RUN_NOT_STARTED_ACTION,
            True,
            False,
            True,
            False,
        ),
        (
            "Skipped stale claim for Stale Job; job was not started.",
            SCHEDULER_RUN_NOT_STARTED_ACTION,
            True,
            False,
            True,
            True,
        ),
        (
            f"Ran Mystery Job:\nUnknown job type: {PRIVATE_MARKERS[2]}",
            SCHEDULER_RUN_INSPECTION_ACTION,
            True,
            True,
            False,
            True,
        ),
        (
            "Ran Morning Brief:\ndelivery failed: outcome is unknown; local receipt is uncertain",
            SCHEDULER_RUN_INSPECTION_ACTION,
            False,
            True,
            False,
            None,
        ),
    )
    for output, action, known, possible, retry_safe, changed in cases:
        with mock.patch.object(Scheduler, "run_job_now", return_value=output):
            result = run_now({"name": "Morning Brief"})
        _assert_guided(
            result,
            action=action,
            outcome_known=known,
            side_effect_possible=possible,
            retry_safe=retry_safe,
            state_changed=changed,
            label=f"run_job_now {output.splitlines()[0]}",
        )

    with mock.patch.object(Scheduler, "run_job_now", side_effect=RuntimeError(PRIVATE_MARKERS[0])):
        unknown = run_now({"name": "Morning Brief"})
    _assert_guided(
        unknown,
        action=SCHEDULER_RUN_INSPECTION_ACTION,
        outcome_known=False,
        side_effect_possible=True,
        retry_safe=False,
        state_changed=None,
        label="run_job_now exception",
    )

    with mock.patch.object(Scheduler, "run_due_jobs", side_effect=RuntimeError(PRIVATE_MARKERS[1])):
        due_unknown = tools["run_due_jobs"]({})
    _assert_guided(
        due_unknown,
        action=SCHEDULER_RUN_INSPECTION_ACTION,
        outcome_known=False,
        side_effect_possible=True,
        retry_safe=False,
        state_changed=None,
        label="run_due_jobs exception",
    )
    return len(cases) + 2


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        tools = _tool_map(runtime)
        count = 0
        count += _test_pre_mutation_and_read_failures(runtime, tools)
        count += _test_mutation_failures(runtime, tools)
        count += _test_run_failures(runtime, tools)
    print(f"Scheduler failure-guidance smoke passed: {count} isolated failure paths")


if __name__ == "__main__":
    main()
