from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import jarvis_v2.agent.runtime as runtime_module
from jarvis_v2.agent.types import Plan, PlannedAction, RuntimeResult
from jarvis_v2.memory.store import (
    DecisionRecord,
    GoalRecord,
    PreferenceRecord,
    TaskRecord,
    auto_mutation_action_digest,
    auto_mutation_request_digest,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import AutoMutationEffect


CONTRACTED_TOOLS = {
    "complete_task",
    "update_task_status",
    "complete_goal_step",
    "set_goal_status",
    "set_preference_status",
    "set_decision_status",
}
AUTHORIZATION_FLAGS = (
    "authorizes_retry",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "requires_confirmation",
)


class StaticPlanner:
    def __init__(self, action: PlannedAction):
        self.action = action

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise one contracted local mutation through the real runtime.",
            [self.action],
            needs_model=False,
        )


def _set_action(runtime: Any, tool_name: str, args: dict[str, Any]) -> None:
    runtime.planner = StaticPlanner(PlannedAction(tool_name, args, "rollout smoke"))


def _rows(runtime: Any, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute(query, params)]


def _receipt_rows(runtime: Any) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM auto_mutation_receipts ORDER BY id")


def _tool_run_rows(runtime: Any) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM tool_runs ORDER BY id")


def _assert_contract(runtime: Any, tool_name: str) -> None:
    tool = runtime.registry.get(tool_name)
    contract = tool.auto_mutation_contract
    expected_effects = {
        AutoMutationEffect.LOCAL_DATABASE,
        AutoMutationEffect.OBSIDIAN_VAULT,
    }
    if contract is None or contract.effects != expected_effects:
        raise SystemExit(f"{tool_name} current registry contract drifted: {contract}")


def _assert_no_authorization(result: RuntimeResult, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} returned an unexpected result count: {result.tool_results}")
    metadata = result.tool_results[0].metadata
    for key in AUTHORIZATION_FLAGS:
        if metadata.get(key):
            raise SystemExit(f"{label} unexpectedly authorized via {key}: {metadata}")


def _assert_private_surfaces(
    runtime: Any,
    results: list[RuntimeResult],
    tokens: list[str],
    tool_name: str,
    args: dict[str, Any],
) -> None:
    receipts = _receipt_rows(runtime)
    secrets = set(tokens)
    secrets.add(auto_mutation_action_digest(tool_name, args))
    secrets.update(auto_mutation_request_digest(token) for token in tokens)
    for receipt in receipts:
        for key in ("request_digest", "action_digest", "operation_digest", "uncertainty_digest", "run_token"):
            value = receipt.get(key)
            if isinstance(value, str) and value:
                secrets.add(value)
    audit_surfaces = [
        {"output": row.get("output"), "metadata": row.get("metadata")}
        for row in _tool_run_rows(runtime)
    ]
    protected = json.dumps(
        {"runtime_results": [repr(result) for result in results], "audit": audit_surfaces},
        sort_keys=True,
    )
    leaked = [secret for secret in secrets if secret in protected]
    if leaked:
        raise SystemExit(f"{tool_name} leaked receipt-private token/digest data: {leaked}")


def _assert_completed_linkage(runtime: Any, tool_name: str, expected: int) -> None:
    receipts = _receipt_rows(runtime)
    runs = _tool_run_rows(runtime)
    successful_runs = [row for row in runs if row["ok"] == 1]
    if len(receipts) != expected or len(successful_runs) != expected:
        raise SystemExit(
            f"{tool_name} expected {expected} receipts and successful ordinary runs: receipts={receipts}, runs={runs}"
        )
    run_by_id = {row["id"]: row for row in successful_runs}
    linked_ids: set[int] = set()
    for receipt in receipts:
        run_id = receipt.get("tool_run_id")
        run = run_by_id.get(run_id)
        if (
            receipt.get("tool_name") != tool_name
            or receipt.get("state") != "completed"
            or receipt.get("result") != "succeeded"
            or run is None
            or run.get("tool_name") != tool_name
            or run.get("ok") != 1
            or run.get("approved") != 0
            or run.get("approval_id") is not None
            or run.get("approval_action_digest") is not None
        ):
            raise SystemExit(f"{tool_name} receipt/run linkage is not ordinary approved=0: {receipt}, {run}")
        linked_ids.add(int(run_id))
    if len(linked_ids) != expected:
        raise SystemExit(f"{tool_name} receipts did not link one-to-one to tool runs: {receipts}")


def _exercise_success(
    runtime: Any,
    *,
    tool_name: str,
    args: dict[str, Any],
    table: str,
    verify_projection: Any,
) -> None:
    _assert_contract(runtime, tool_name)
    _set_action(runtime, tool_name, args)
    before_count = _rows(runtime, f"SELECT COUNT(*) AS count FROM {table}")[0]["count"]
    first_token = f"private-{tool_name}-request-one"
    second_token = f"private-{tool_name}-request-two"

    first = runtime.handle(f"rollout first {tool_name}", request_token=first_token)
    if len(first.tool_results) != 1 or not first.tool_results[0].ok:
        raise SystemExit(f"{tool_name} initial runtime execution failed: {first.tool_results}")
    _assert_completed_linkage(runtime, tool_name, 1)
    verify_projection()

    replay = runtime.handle(f"rollout replay {tool_name}", request_token=first_token)
    if replay.tool_results[0].metadata.get("failure_kind") != "auto_mutation_completed_replay":
        raise SystemExit(f"{tool_name} same-token retry was not a completed replay: {replay.tool_results}")
    _assert_no_authorization(replay, f"{tool_name} completed replay")
    _assert_completed_linkage(runtime, tool_name, 1)

    repeated = runtime.handle(f"rollout intentional repeat {tool_name}", request_token=second_token)
    if len(repeated.tool_results) != 1 or not repeated.tool_results[0].ok:
        raise SystemExit(f"{tool_name} new-token intentional repeat failed: {repeated.tool_results}")
    after_count = _rows(runtime, f"SELECT COUNT(*) AS count FROM {table}")[0]["count"]
    if after_count != before_count:
        raise SystemExit(f"{tool_name} status repeat created a new {table} row: {before_count} -> {after_count}")
    _assert_completed_linkage(runtime, tool_name, 2)
    verify_projection()
    _assert_private_surfaces(runtime, [first, replay, repeated], [first_token, second_token], tool_name, args)


def test_successful_task_goal_and_setter_rollout() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-rollout-task-") as temp:
        runtime = make_temp_runtime(Path(temp))
        task_id = runtime.store.add_task(TaskRecord("rollout task projection"))
        task_note = runtime.vault.sync_open_tasks(runtime.store)

        def verify_task() -> None:
            row = runtime.store.get_task(task_id)
            text = task_note.read_text(encoding="utf-8")
            if row is None or row["status"] != "done" or "rollout task projection" in text:
                raise SystemExit(f"complete_task SQLite/Obsidian projection diverged: {dict(row) if row else None}, {text}")

        _exercise_success(
            runtime,
            tool_name="complete_task",
            args={"task_id": task_id},
            table="tasks",
            verify_projection=verify_task,
        )

    with TemporaryDirectory(prefix="jarvis-auto-rollout-task-status-") as temp:
        runtime = make_temp_runtime(Path(temp))
        task_id = runtime.store.add_task(TaskRecord("rollout paused task projection"))
        task_note = runtime.vault.sync_open_tasks(runtime.store)

        def verify_task_status() -> None:
            row = runtime.store.get_task(task_id)
            text = task_note.read_text(encoding="utf-8")
            if row is None or row["status"] != "paused" or "rollout paused task projection" in text:
                raise SystemExit(
                    f"update_task_status SQLite/Obsidian projection diverged: "
                    f"{dict(row) if row else None}, {text}"
                )

        _exercise_success(
            runtime,
            tool_name="update_task_status",
            args={"task_id": task_id, "status": "paused"},
            table="tasks",
            verify_projection=verify_task_status,
        )

    with TemporaryDirectory(prefix="jarvis-auto-rollout-goal-step-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id = runtime.store.create_goal(GoalRecord("Rollout Step Goal"))
        step_id = runtime.store.add_goal_step(goal_id, "publish the rollout step")
        seeded_goal = runtime.registry.get("export_goal").handler({"goal_id": goal_id})
        goal_notes = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if not seeded_goal.ok or len(goal_notes) != 1:
            raise SystemExit("complete_goal_step fixture did not establish durable projection custody")
        goal_note = goal_notes[0]

        def verify_step() -> None:
            step = _rows(runtime, "SELECT * FROM goal_steps WHERE id = ?", (step_id,))[0]
            text = goal_note.read_text(encoding="utf-8")
            if step["status"] != "done" or f"- [x] #{step_id} publish the rollout step" not in text:
                raise SystemExit(f"complete_goal_step SQLite/Obsidian projection diverged: {step}, {text}")

        _exercise_success(
            runtime,
            tool_name="complete_goal_step",
            args={"step_id": step_id},
            table="goal_steps",
            verify_projection=verify_step,
        )

    with TemporaryDirectory(prefix="jarvis-auto-rollout-goal-status-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id = runtime.store.create_goal(GoalRecord("Rollout Status Goal"))
        seeded_goal = runtime.registry.get("export_goal").handler({"goal_id": goal_id})
        goal_notes = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if not seeded_goal.ok or len(goal_notes) != 1:
            raise SystemExit("set_goal_status fixture did not establish durable projection custody")
        goal_note = goal_notes[0]

        def verify_goal_status() -> None:
            row = runtime.store.get_goal(goal_id)
            text = goal_note.read_text(encoding="utf-8")
            if row is None or row["status"] != "paused" or 'status: "paused"' not in text:
                raise SystemExit(f"set_goal_status SQLite/Obsidian projection diverged: {dict(row) if row else None}, {text}")

        _exercise_success(
            runtime,
            tool_name="set_goal_status",
            args={"goal_id": goal_id, "status": "paused"},
            table="goals",
            verify_projection=verify_goal_status,
        )

    with TemporaryDirectory(prefix="jarvis-auto-rollout-preference-") as temp:
        runtime = make_temp_runtime(Path(temp))
        seeded = runtime.registry.get("set_preference").handler(
            {"key": "rollout tone", "value": "direct", "category": "communication"}
        )
        if not seeded.ok:
            raise SystemExit(f"preference rollout fixture did not seed: {seeded}")
        preference_id = int(seeded.metadata["preference_id"])
        preference_note = runtime.vault.root_path / "Memory Tree" / "Preferences.md"

        def verify_preference() -> None:
            row = _rows(runtime, "SELECT * FROM preferences WHERE id = ?", (preference_id,))[0]
            text = preference_note.read_text(encoding="utf-8")
            expected = f"- #{preference_id} rollout tone: direct [retired]"
            if row["status"] != "retired" or expected not in text:
                raise SystemExit(f"set_preference_status SQLite/Obsidian projection diverged: {row}, {text}")

        _exercise_success(
            runtime,
            tool_name="set_preference_status",
            args={"preference_id": preference_id, "status": "retired"},
            table="preferences",
            verify_projection=verify_preference,
        )

    with TemporaryDirectory(prefix="jarvis-auto-rollout-decision-") as temp:
        runtime = make_temp_runtime(Path(temp))
        decision_id = runtime.store.add_decision(DecisionRecord("Use rollout receipts", "exactly once", "safe retry"))
        decision_note = runtime.vault.write_decision(runtime.store.get_decision(decision_id))

        def verify_decision() -> None:
            row = runtime.store.get_decision(decision_id)
            text = decision_note.read_text(encoding="utf-8")
            if row is None or row["status"] != "superseded" or "status: superseded" not in text:
                raise SystemExit(f"set_decision_status SQLite/Obsidian projection diverged: {dict(row) if row else None}, {text}")

        _exercise_success(
            runtime,
            tool_name="set_decision_status",
            args={"decision_id": decision_id, "status": "superseded"},
            table="decisions",
            verify_projection=verify_decision,
        )


def _assert_uncertain_failure(
    runtime: Any,
    *,
    tool_name: str,
    args: dict[str, Any],
    token: str,
    publication_calls: list[str],
) -> list[RuntimeResult]:
    _assert_contract(runtime, tool_name)
    _set_action(runtime, tool_name, args)
    failed = runtime.handle(f"rollout publication failure {tool_name}", request_token=token)
    if failed.tool_results[0].metadata.get("failure_kind") not in {
        "tool_error",
        "auto_mutation_handler_failed",
    }:
        raise SystemExit(f"{tool_name} publication failure did not preserve handler failure: {failed.tool_results}")
    _assert_no_authorization(failed, f"{tool_name} publication failure")
    receipts = _receipt_rows(runtime)
    if len(receipts) != 1 or receipts[0]["state"] != "uncertain" or receipts[0]["result"] != "unknown":
        raise SystemExit(f"{tool_name} publication failure did not become uncertain: {receipts}")

    same = runtime.handle(f"rollout same retry {tool_name}", request_token=token)
    cross_token = token + "-cross-request"
    cross = runtime.handle(f"rollout cross retry {tool_name}", request_token=cross_token)
    if same.tool_results[0].metadata.get("failure_kind") != "auto_mutation_outcome_uncertain":
        raise SystemExit(f"{tool_name} same-request uncertain retry was not blocked: {same.tool_results}")
    if cross.tool_results[0].metadata.get("failure_kind") != "auto_mutation_unresolved_action":
        raise SystemExit(f"{tool_name} cross-request uncertain retry was not blocked: {cross.tool_results}")
    if len(publication_calls) != 1:
        raise SystemExit(f"{tool_name} handler/publication was invoked more than once: {publication_calls}")
    if any(row["ok"] == 1 for row in _tool_run_rows(runtime)):
        raise SystemExit(f"{tool_name} uncertain outcome left a successful audit row: {_tool_run_rows(runtime)}")
    if len(_receipt_rows(runtime)) != 1:
        raise SystemExit(f"{tool_name} blocked cross-request retry created a receipt")
    for result, label in ((same, "same retry"), (cross, "cross retry")):
        _assert_no_authorization(result, f"{tool_name} {label}")
    _assert_private_surfaces(runtime, [failed, same, cross], [token, cross_token], tool_name, args)
    return [failed, same, cross]


def test_db_commit_before_vault_publication_is_uncertain() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-rollout-task-failure-") as temp:
        runtime = make_temp_runtime(Path(temp))
        task_id = runtime.store.add_task(TaskRecord("task remains in stale projection"))
        task_note = runtime.vault.sync_open_tasks(runtime.store)
        publication_calls: list[str] = []

        def fail_task_publication(_store: Any) -> Path:
            publication_calls.append("sync_open_tasks")
            raise OSError("injected after task DB commit")

        runtime.vault.sync_open_tasks = fail_task_publication
        _assert_uncertain_failure(
            runtime,
            tool_name="complete_task",
            args={"task_id": task_id},
            token="private-task-publication-failure",
            publication_calls=publication_calls,
        )
        row = runtime.store.get_task(task_id)
        if row is None or row["status"] != "done":
            raise SystemExit(f"task failure injection did not occur after DB commit: {dict(row) if row else None}")
        if "task remains in stale projection" not in task_note.read_text(encoding="utf-8"):
            raise SystemExit("task failure injection unexpectedly published the post-commit projection")

    with TemporaryDirectory(prefix="jarvis-auto-rollout-task-status-failure-") as temp:
        runtime = make_temp_runtime(Path(temp))
        task_id = runtime.store.add_task(TaskRecord("status remains in stale projection"))
        task_note = runtime.vault.sync_open_tasks(runtime.store)
        publication_calls = []

        def fail_task_status_publication(_store: Any) -> Path:
            publication_calls.append("sync_open_tasks")
            raise OSError("injected after task status DB commit")

        runtime.vault.sync_open_tasks = fail_task_status_publication
        _assert_uncertain_failure(
            runtime,
            tool_name="update_task_status",
            args={"task_id": task_id, "status": "paused"},
            token="private-task-status-publication-failure",
            publication_calls=publication_calls,
        )
        row = runtime.store.get_task(task_id)
        if row is None or row["status"] != "paused":
            raise SystemExit(
                f"task-status failure injection did not occur after DB commit: "
                f"{dict(row) if row else None}"
            )
        if "status remains in stale projection" not in task_note.read_text(encoding="utf-8"):
            raise SystemExit("task-status failure injection unexpectedly published the post-commit projection")

    with TemporaryDirectory(prefix="jarvis-auto-rollout-preference-failure-") as temp:
        runtime = make_temp_runtime(Path(temp))
        seeded = runtime.registry.get("set_preference").handler(
            {"key": "failure tone", "value": "quiet", "category": "communication"}
        )
        if not seeded.ok:
            raise SystemExit(f"preference failure fixture did not seed: {seeded}")
        preference_id = int(seeded.metadata["preference_id"])
        preference_note = runtime.vault.root_path / "Memory Tree" / "Preferences.md"
        publication_calls = []

        def fail_preference_publication(
            _preferences: Any, *, store_identity: str, generation: int
        ) -> tuple[Path, str]:
            publication_calls.append("write_preferences")
            raise OSError("injected after preference DB commit")

        runtime.vault.write_preferences_with_evidence = fail_preference_publication
        _assert_uncertain_failure(
            runtime,
            tool_name="set_preference_status",
            args={"preference_id": preference_id, "status": "retired"},
            token="private-preference-publication-failure",
            publication_calls=publication_calls,
        )
        row = _rows(runtime, "SELECT * FROM preferences WHERE id = ?", (preference_id,))[0]
        text = preference_note.read_text(encoding="utf-8")
        if row["status"] != "retired":
            raise SystemExit(f"preference failure injection did not occur after DB commit: {row}")
        if f"- #{preference_id} failure tone: quiet [active]" not in text or "[retired]" in text:
            raise SystemExit("preference failure injection unexpectedly published the post-commit projection")


def _assert_refusal_becomes_uncertain(
    runtime: Any,
    *,
    tool_name: str,
    args: dict[str, Any],
    token: str,
    expected_reason: str,
) -> None:
    _assert_contract(runtime, tool_name)
    _set_action(runtime, tool_name, args)
    result = runtime.handle(f"rollout refusal {tool_name}", request_token=token)
    if result.tool_results[0].ok or result.tool_results[0].metadata.get("reason") != expected_reason:
        raise SystemExit(f"{tool_name} refusal shape drifted: {result.tool_results}")
    _assert_no_authorization(result, f"{tool_name} refusal")
    receipts = _receipt_rows(runtime)
    if len(receipts) != 1 or receipts[0]["state"] != "uncertain" or receipts[0]["result"] != "unknown":
        raise SystemExit(f"{tool_name} refusal was not conservatively uncertain: {receipts}")
    if any(row["ok"] == 1 or row["approved"] == 1 for row in _tool_run_rows(runtime)):
        raise SystemExit(f"{tool_name} refusal produced authorization/success evidence: {_tool_run_rows(runtime)}")
    if _rows(runtime, "SELECT COUNT(*) AS count FROM pending_approvals")[0]["count"] != 0:
        raise SystemExit(f"{tool_name} refusal queued an approval")
    replay = runtime.handle(f"rollout refusal replay {tool_name}", request_token=token)
    if replay.tool_results[0].metadata.get("failure_kind") != "auto_mutation_outcome_uncertain":
        raise SystemExit(f"{tool_name} refusal retry did not stay blocked: {replay.tool_results}")
    _assert_no_authorization(replay, f"{tool_name} refusal replay")
    _assert_private_surfaces(runtime, [result, replay], [token], tool_name, args)


def _assert_shape_refusal_stops_before_receipt(
    runtime: Any,
    *,
    tool_name: str,
    args: dict[str, Any],
    token: str,
) -> None:
    _assert_contract(runtime, tool_name)
    _set_action(runtime, tool_name, args)
    result = runtime.handle(f"rollout shape refusal {tool_name}", request_token=token)
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != "tool_arguments_invalid":
        raise SystemExit(f"{tool_name} typed refusal shape drifted: {result.tool_results}")
    if item.metadata.get("handler_invoked") is not False or item.metadata.get("executed_handler") is not False:
        raise SystemExit(f"{tool_name} typed refusal claimed handler execution: {item.metadata}")
    if _receipt_rows(runtime):
        raise SystemExit(f"{tool_name} typed refusal created a mutation receipt")
    _assert_no_authorization(result, f"{tool_name} typed refusal")
    if any(row["ok"] == 1 or row["approved"] == 1 for row in _tool_run_rows(runtime)):
        raise SystemExit(f"{tool_name} typed refusal produced authorization/success evidence: {_tool_run_rows(runtime)}")
    if _rows(runtime, "SELECT COUNT(*) AS count FROM pending_approvals")[0]["count"] != 0:
        raise SystemExit(f"{tool_name} typed refusal queued an approval")
    replay = runtime.handle(f"rollout shape refusal replay {tool_name}", request_token=token)
    if replay.tool_results[0].metadata.get("failure_kind") != "tool_arguments_invalid":
        raise SystemExit(f"{tool_name} typed refusal replay did not revalidate safely: {replay.tool_results}")
    if _receipt_rows(runtime):
        raise SystemExit(f"{tool_name} typed refusal replay created a mutation receipt")
    _assert_private_surfaces(runtime, [result, replay], [token], tool_name, args)


def test_malformed_and_missing_targets_fail_uncertain_without_authority() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-rollout-malformed-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _assert_shape_refusal_stops_before_receipt(
            runtime,
            tool_name="complete_task",
            args={"task_id": "not-a-task-id"},
            token="private-malformed-target",
        )

    with TemporaryDirectory(prefix="jarvis-auto-rollout-status-malformed-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _assert_shape_refusal_stops_before_receipt(
            runtime,
            tool_name="update_task_status",
            args={"task_id": "not-a-task-id", "status": "paused"},
            token="private-malformed-status-target",
        )

    with TemporaryDirectory(prefix="jarvis-auto-rollout-missing-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _assert_refusal_becomes_uncertain(
            runtime,
            tool_name="set_decision_status",
            args={"decision_id": 999999, "status": "retired"},
            token="private-missing-target",
            expected_reason="not_found",
        )


def main() -> None:
    original_suggest_command = runtime_module.suggest_command
    runtime_module.suggest_command = lambda _text: None
    try:
        test_successful_task_goal_and_setter_rollout()
        test_db_commit_before_vault_publication_is_uncertain()
        test_malformed_and_missing_targets_fail_uncertain_without_authority()
    finally:
        runtime_module.suggest_command = original_suggest_command
    print("Auto mutation task/goal rollout smoke passed")


if __name__ == "__main__":
    main()
