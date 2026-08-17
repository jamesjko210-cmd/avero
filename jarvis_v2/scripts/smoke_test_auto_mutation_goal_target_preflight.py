from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any, Callable
from unittest.mock import patch

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.memory.store import GoalRecord, goal_projection_source_digest
from jarvis_v2.scripts.test_runtime import make_temp_runtime


@dataclass(frozen=True)
class GoalMutationCase:
    tool_name: str
    target_kind: str
    missing_reason: str
    mutation: str
    corrected_args: Callable[[int, int | None], dict[str, Any]]
    missing_args: Callable[[int], dict[str, Any]]
    expected_recovery: tuple[str, ...]


CASES = (
    GoalMutationCase(
        "add_goal_step",
        "goal",
        "missing_goal",
        "step_create",
        lambda goal_id, _step_id: {"goal_id": goal_id, "body": "preflight step"},
        lambda missing_id: {"goal_id": missing_id, "body": "preflight step"},
        (
            "list goals",
            "goal <correct goal id> status",
            "add step to goal <correct goal id>: <next step>",
        ),
    ),
    GoalMutationCase(
        "complete_goal_step",
        "step",
        "missing_step",
        "step_completion",
        lambda _goal_id, step_id: {"step_id": step_id},
        lambda missing_id: {"step_id": missing_id},
        ("next actions", "complete goal step <correct step id>"),
    ),
    GoalMutationCase(
        "set_goal_status",
        "goal",
        "missing_goal",
        "status_update",
        lambda goal_id, _step_id: {"goal_id": goal_id, "status": "paused"},
        lambda missing_id: {"goal_id": missing_id, "status": "paused"},
        (
            "list goals",
            "goal <correct goal id> status",
            "goal <correct goal id> <active|paused|done|dropped>",
        ),
    ),
    GoalMutationCase(
        "export_goal",
        "goal",
        "missing_goal",
        "goal_export",
        lambda goal_id, _step_id: {"goal_id": goal_id},
        lambda missing_id: {"goal_id": missing_id},
        (
            "list goals",
            "goal <correct goal id> status",
            "export goal <correct goal id> to obsidian",
        ),
    ),
)


class StaticPlanner:
    def __init__(self, tool_name: str, args: dict[str, Any]):
        self.tool_name = tool_name
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise one goal target preflight.",
            [PlannedAction(self.tool_name, dict(self.args), "goal target preflight smoke")],
            needs_model=False,
        )


def _set_action(runtime: JarvisRuntime, case: GoalMutationCase, args: dict[str, Any]) -> None:
    runtime.planner = StaticPlanner(case.tool_name, args)


def _rows(runtime: JarvisRuntime, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute(sql, params)]


def _receipt_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM auto_mutation_receipts ORDER BY id")


def _goal_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM goals ORDER BY id")


def _step_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM goal_steps ORDER BY id")


def _fixture(runtime: JarvisRuntime, case: GoalMutationCase) -> tuple[int, int]:
    goal_id = runtime.store.create_goal(
        GoalRecord("Goal target preflight", "deterministic fixture", "this week")
    )
    step_id = (
        runtime.store.add_goal_step(goal_id, "complete target preflight")
        if case.target_kind == "step"
        else 0
    )
    return goal_id, step_id


def _install_counted_handler(runtime: JarvisRuntime, tool_name: str) -> list[int]:
    tool = runtime.registry.get(tool_name)
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools[tool_name] = replace(tool, handler=counted)
    return calls


def _install_mock_publication(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    publications: list[dict[str, Any]] = []

    def publish(
        goal: Any,
        steps: Any,
        *,
        store_identity: str,
        expected_prior_content_digest: str | None = None,
    ) -> tuple[Path, str, str]:
        publications.append(
            {
                "goal": dict(goal),
                "steps": [dict(step) for step in steps],
                "store_identity": store_identity,
            }
        )
        source_digest = goal_projection_source_digest(
            goal["id"],
            goal["revision"],
            goal["title"],
            goal["purpose"],
            goal["horizon"],
            goal["status"],
            goal["created_at"],
            goal["updated_at"],
            steps,
        )
        return (
            runtime.vault.root_path / "Projects" / "mock-goal.md",
            "a" * 64,
            source_digest,
        )

    def candidate(
        goal: Any,
        steps: Any,
        *,
        store_identity: str,
    ) -> tuple[str, str]:
        source_digest = goal_projection_source_digest(
            goal["id"],
            goal["revision"],
            goal["title"],
            goal["purpose"],
            goal["horizon"],
            goal["status"],
            goal["created_at"],
            goal["updated_at"],
            steps,
        )
        return "a" * 64, source_digest

    runtime.vault.goal_projection_candidate_evidence = candidate  # type: ignore[method-assign]
    runtime.vault.write_goal_with_evidence = publish  # type: ignore[method-assign]
    return publications


def _assert_recovery_metadata(
    item: ToolResult,
    case: GoalMutationCase,
    target_id: int,
    *,
    semantic_preflight: bool,
) -> None:
    metadata = item.metadata
    target_key = "step_id" if case.target_kind == "step" else "goal_id"
    refresh_key = (
        "retry_requires_goal_step_refresh"
        if case.target_kind == "step"
        else "retry_requires_goal_refresh"
    )
    expected_next = "next actions" if case.target_kind == "step" else "list goals"
    handoff = metadata.get("goal_refusal_handoff")
    if (
        item.ok
        or metadata.get("reason") != case.missing_reason
        or metadata.get(target_key) != target_id
        or metadata.get("next_command") != expected_next
        or tuple(metadata.get("recovery_commands") or ()) != case.expected_recovery
        or metadata.get(refresh_key) is not True
        or metadata.get("retry_requires_corrected_id") is not True
        or metadata.get("recovery_commands_require_normal_policy") is not True
        or metadata.get("authorizes_retry") is not False
        or metadata.get("authorizes_goal_mutation") is not False
        or metadata.get("goal_refusal_handoff_ready") is not True
        or metadata.get("goal_mutation_handoff_ready") is not False
        or not isinstance(handoff, dict)
        or handoff.get("source") != case.tool_name
        or handoff.get("reason") != case.missing_reason
        or handoff.get("mutation") != case.mutation
        or handoff.get(target_key) != target_id
        or handoff.get("refused") is not True
        or handoff.get("handoff_ready") is not True
    ):
        raise SystemExit(f"{case.tool_name} lost missing-target recovery metadata: {item}")
    boundaries = handoff.get("boundaries") or {}
    if not boundaries or any(value is not False for value in boundaries.values()):
        raise SystemExit(f"{case.tool_name} recovery handoff crossed a boundary: {handoff}")
    if semantic_preflight and (
        metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
        or metadata.get("handler_invoked") is not False
        or metadata.get("executed_handler") is not False
        or metadata.get("state_changed") is not False
        or metadata.get("auto_mutation_effects_started") is not False
    ):
        raise SystemExit(f"{case.tool_name} missing target crossed semantic preflight: {item}")


def _assert_completed_receipts(
    runtime: JarvisRuntime,
    case: GoalMutationCase,
    expected: int,
) -> None:
    receipts = _receipt_rows(runtime)
    successful_runs = [
        row
        for row in _rows(runtime, "SELECT * FROM tool_runs ORDER BY id")
        if row["ok"] == 1 and row["tool_name"] == case.tool_name
    ]
    if (
        len(receipts) != expected
        or len(successful_runs) != expected
        or any(receipt["tool_name"] != case.tool_name for receipt in receipts)
        or any(receipt["state"] != "completed" for receipt in receipts)
        or any(receipt["result"] != "succeeded" for receipt in receipts)
        or any(receipt["resolution"] != "recorded" for receipt in receipts)
        or {receipt["tool_run_id"] for receipt in receipts}
        != {run["id"] for run in successful_runs}
        or any(run["approved"] != 0 for run in successful_runs)
        or any(run["approval_id"] is not None for run in successful_runs)
        or any(run["approval_action_digest"] is not None for run in successful_runs)
    ):
        raise SystemExit(
            f"{case.tool_name} lost completed receipt/audit linkage: "
            f"{receipts} / {successful_runs}"
        )


def _assert_success_effect(
    runtime: JarvisRuntime,
    case: GoalMutationCase,
    goal_id: int,
    step_id: int,
    publications: list[dict[str, Any]],
    expected_publications: int,
) -> None:
    goal = runtime.store.get_goal(goal_id)
    step = runtime.store.get_goal_step(step_id)
    if len(publications) != expected_publications:
        raise SystemExit(f"{case.tool_name} publication count drifted: {publications}")
    if case.tool_name == "add_goal_step":
        matches = [row for row in runtime.store.list_goal_steps(goal_id) if row["body"] == "preflight step"]
        valid = len(matches) == expected_publications
    elif case.tool_name == "complete_goal_step":
        valid = step is not None and step["status"] == "done"
    elif case.tool_name == "set_goal_status":
        valid = goal is not None and goal["status"] == "paused"
    else:
        valid = publications[0]["goal"]["id"] == goal_id
    if not valid:
        raise SystemExit(f"{case.tool_name} corrected target did not receive its exact effect")


def _assert_completed_replay(result: Any, case: GoalMutationCase, calls: list[int]) -> None:
    item = result.tool_results[0]
    if (
        item.ok
        or item.metadata.get("failure_kind") != "auto_mutation_completed_replay"
        or calls[0] != 1
    ):
        raise SystemExit(f"{case.tool_name} same-token replay did not coalesce: {item}")


def _restore_row(runtime: JarvisRuntime, table: str, row: dict[str, Any]) -> None:
    columns = tuple(row)
    placeholders = ", ".join("?" for _ in columns)
    with runtime.store.connect() as conn:
        conn.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(row[column] for column in columns),
        )


def test_target_identity_contracts_and_invalid_unicode_preflight() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-target-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        goal_id = runtime.store.create_goal(GoalRecord("Contract target", "identity proof"))
        step_id = runtime.store.add_goal_step(goal_id, "contract step")
        if step_id is None:
            raise SystemExit("goal target contract fixture did not create a step")
        other_goal_id = runtime.store.create_goal(GoalRecord("Other target", "separation proof"))
        other_step_id = runtime.store.add_goal_step(other_goal_id, "other contract step")
        if other_step_id is None:
            raise SystemExit("goal target separation fixture did not create a step")
        expected_keys = {
            "add_goal_step": ({"goal_id": goal_id, "body": "one"}, {"goal_id": goal_id, "body": "two"}, {"goal_id": other_goal_id, "body": "one"}),
            "complete_goal_step": ({"step_id": step_id}, {"step_id": step_id}, {"step_id": other_step_id}),
            "set_goal_status": ({"goal_id": goal_id, "status": "paused"}, {"goal_id": goal_id, "status": "done"}, {"goal_id": other_goal_id, "status": "paused"}),
            "export_goal": ({"goal_id": goal_id}, {"goal_id": goal_id}, {"goal_id": other_goal_id}),
        }
        for tool_name, (first_args, changed_args, other_args) in expected_keys.items():
            contract = runtime.registry.get(tool_name).auto_mutation_contract
            if contract is None or contract.operation_key_builder is None:
                raise SystemExit(f"{tool_name} lost its target-only operation key")
            first_key = contract.operation_key_builder(first_args)
            changed_key = contract.operation_key_builder(changed_args)
            other_key = contract.operation_key_builder(other_args)
            expected_key = {"step_id": step_id} if tool_name == "complete_goal_step" else {"goal_id": goal_id}
            if first_key != expected_key or changed_key != expected_key or other_key == expected_key:
                raise SystemExit(f"{tool_name} operation identity included mutable arguments")

        unsafe_body = "unsafe-\ud800-step"
        case = CASES[0]
        _set_action(runtime, case, {"goal_id": goal_id, "body": unsafe_body})
        calls = _install_counted_handler(runtime, case.tool_name)
        before = _step_rows(runtime)
        refused = runtime.handle("invalid unicode goal step", request_token="goal-invalid-unicode")
        item = refused.tool_results[0]
        surfaces = json.dumps(
            {
                "output": item.output,
                "metadata": item.metadata,
                "runs": _rows(runtime, "SELECT output, metadata FROM tool_runs ORDER BY id"),
            },
            ensure_ascii=False,
            default=str,
        )
        if (
            item.ok
            or item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
            or item.metadata.get("reason") != "invalid_unicode"
            or item.metadata.get("handler_invoked") is not False
            or calls[0] != 0
            or _receipt_rows(runtime)
            or _step_rows(runtime) != before
            or unsafe_body in surfaces
        ):
            raise SystemExit(f"invalid Unicode crossed the goal mutation boundary: {item}")

        unsafe_cases = (
            (CASES[2], {"goal_id": goal_id, "status": "unsafe-\ud800-status"}),
            (CASES[0], {"goal_id": "unsafe-\ud800-id", "body": "safe body"}),
            (CASES[1], {"step_id": "unsafe-\ud800-id"}),
            (CASES[2], {"goal_id": "unsafe-\ud800-id", "status": "done"}),
            (CASES[3], {"goal_id": "unsafe-\ud800-id"}),
        )
        for unsafe_case, unsafe_args in unsafe_cases:
            _set_action(runtime, unsafe_case, unsafe_args)
            unsafe_calls = _install_counted_handler(runtime, unsafe_case.tool_name)
            refused = runtime.handle(
                "invalid Unicode goal argument",
                request_token=f"goal-invalid-unicode-{unsafe_case.tool_name}",
            )
            item = refused.tool_results[0]
            surfaces = json.dumps(
                {
                    "output": item.output,
                    "metadata": item.metadata,
                    "runs": _rows(runtime, "SELECT output, metadata FROM tool_runs ORDER BY id"),
                },
                ensure_ascii=False,
                default=str,
            )
            if (
                item.ok
                or item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
                or unsafe_calls[0] != 0
                or "\ud800" in surfaces
            ):
                raise SystemExit(f"{unsafe_case.tool_name} leaked invalid Unicode: {item}")

        huge_id = 10**100
        for huge_case in CASES:
            huge_args = huge_case.missing_args(huge_id)
            _set_action(runtime, huge_case, huge_args)
            huge_calls = _install_counted_handler(runtime, huge_case.tool_name)
            refused = runtime.handle(
                "oversized goal target",
                request_token=f"goal-oversized-{huge_case.tool_name}",
            )
            item = refused.tool_results[0]
            if (
                item.ok
                or item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
                or item.metadata.get("reason") not in {"bad_goal_id", "bad_step_id"}
                or huge_calls[0] != 0
                or _receipt_rows(runtime)
            ):
                raise SystemExit(f"{huge_case.tool_name} oversized ID crossed preflight: {item}")

        oversized_string_id = "9" * 100_000
        for huge_case in CASES:
            huge_args = huge_case.missing_args(oversized_string_id)  # type: ignore[arg-type]
            _set_action(runtime, huge_case, huge_args)
            huge_calls = _install_counted_handler(runtime, huge_case.tool_name)
            refused = runtime.handle(
                "oversized string goal target",
                request_token=f"goal-oversized-string-{huge_case.tool_name}",
            )
            item = refused.tool_results[0]
            serialized = json.dumps(item.metadata, ensure_ascii=False, default=str)
            if (
                item.ok
                or item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
                or item.metadata.get("reason") not in {"bad_goal_id", "bad_step_id"}
                or huge_calls[0] != 0
                or _receipt_rows(runtime)
                or oversized_string_id in serialized
            ):
                raise SystemExit(f"{huge_case.tool_name} oversized string ID crossed preflight: {item}")


def test_numeric_string_targets_reach_goal_specific_handlers() -> None:
    for case in CASES:
        with TemporaryDirectory(prefix=f"jarvis-goal-string-id-{case.tool_name}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            goal_id, step_id = _fixture(runtime, case)
            args = case.corrected_args(goal_id, step_id)
            target_key = "step_id" if case.target_kind == "step" else "goal_id"
            args[target_key] = str(args[target_key])
            _set_action(runtime, case, args)
            calls = _install_counted_handler(runtime, case.tool_name)
            publications = _install_mock_publication(runtime)
            result = runtime.handle(
                "numeric string goal target",
                request_token=f"goal-string-id-{case.tool_name}",
            )
            if not result.tool_results[0].ok or calls[0] != 1 or len(publications) != 1:
                raise SystemExit(f"{case.tool_name} numeric-string target bypassed its handler: {result.tool_results}")


def test_goal_mutation_timestamps_are_sampled_after_write_lock() -> None:
    for mutation in ("create", "add", "complete", "status"):
        with TemporaryDirectory(prefix=f"jarvis-goal-timestamp-lock-{mutation}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            goal_id = runtime.store.create_goal(GoalRecord("Timestamp target", "lock ordering"))
            step_id = runtime.store.add_goal_step(goal_id, "timestamp step")
            if step_id is None:
                raise SystemExit("timestamp ordering fixture did not create a step")
            original_connect = runtime.store.connect
            begin_entered = Event()
            release_begin = Event()
            timestamp_sampled = Event()

            @contextmanager
            def controlled_connect():
                with original_connect() as connection:
                    class ConnectionProxy:
                        def execute(self, sql: str, *args: Any, **kwargs: Any):
                            if sql.strip().upper() == "BEGIN IMMEDIATE":
                                begin_entered.set()
                                if not release_begin.wait(timeout=5):
                                    raise RuntimeError("timestamp lock-order smoke timed out")
                            return connection.execute(sql, *args, **kwargs)

                        def __getattr__(self, name: str) -> Any:
                            return getattr(connection, name)

                    yield ConnectionProxy()

            def sampled_now() -> str:
                timestamp_sampled.set()
                return "2099-01-01T00:00:00Z"

            runtime.store.connect = controlled_connect  # type: ignore[method-assign]
            with patch("jarvis_v2.memory.store.utc_now", side_effect=sampled_now):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    if mutation == "create":
                        future = pool.submit(
                            runtime.store.create_goal,
                            GoalRecord("Locked timestamp goal", "lock proof"),
                        )
                    elif mutation == "add":
                        future = pool.submit(
                            runtime.store.add_goal_step,
                            goal_id,
                            "locked timestamp step",
                        )
                    elif mutation == "complete":
                        future = pool.submit(runtime.store.complete_goal_step, step_id)
                    else:
                        future = pool.submit(runtime.store.set_goal_status, goal_id, "paused")
                    if not begin_entered.wait(timeout=5):
                        raise SystemExit(f"{mutation} mutation never reached its write lock")
                    if timestamp_sampled.is_set():
                        release_begin.set()
                        raise SystemExit(f"{mutation} mutation sampled its timestamp before the write lock")
                    release_begin.set()
                    if future.result(timeout=5) is None or not timestamp_sampled.is_set():
                        raise SystemExit(f"{mutation} mutation did not complete after its write lock")


def test_uncertain_goal_targets_fence_changed_arguments() -> None:
    cases = (
        (CASES[0], lambda goal_id: {"goal_id": goal_id, "body": "owner body"}, lambda goal_id: {"goal_id": goal_id, "body": "changed body"}),
        (CASES[2], lambda goal_id: {"goal_id": goal_id, "status": "paused"}, lambda goal_id: {"goal_id": goal_id, "status": "done"}),
    )
    for case, owner_args, changed_args in cases:
        with TemporaryDirectory(prefix=f"jarvis-goal-target-fence-{case.tool_name}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            goal_id = runtime.store.create_goal(GoalRecord("Uncertain target", "custody proof"))
            _set_action(runtime, case, owner_args(goal_id))
            calls = _install_counted_handler(runtime, case.tool_name)

            def fail_publication(
                _goal: Any,
                _steps: Any,
                *,
                store_identity: str,
                expected_prior_content_digest: str | None = None,
            ) -> tuple[Path, str, str]:
                raise OSError("representative goal projection failure")

            runtime.vault.write_goal_with_evidence = fail_publication  # type: ignore[method-assign]
            owner = runtime.handle(
                "goal mutation becomes uncertain",
                request_token=f"goal-target-uncertain-{case.tool_name}",
            )
            receipts = _receipt_rows(runtime)
            if (
                owner.tool_results[0].ok
                or calls[0] != 1
                or len(receipts) != 1
                or receipts[0]["state"] != "uncertain"
            ):
                raise SystemExit(f"{case.tool_name} did not establish uncertain custody")

            _set_action(runtime, case, changed_args(goal_id))
            blocked = runtime.handle(
                "changed arguments target unresolved goal mutation",
                request_token=f"goal-target-contender-{case.tool_name}",
            )
            if (
                blocked.tool_results[0].metadata.get("failure_kind")
                != "auto_mutation_unresolved_action"
                or calls[0] != 1
                or len(_receipt_rows(runtime)) != 1
            ):
                raise SystemExit(f"{case.tool_name} changed arguments bypassed unresolved custody")


def test_missing_targets_stop_before_receipt_and_allow_corrected_same_token() -> None:
    for case in CASES:
        with TemporaryDirectory(prefix=f"jarvis-goal-target-missing-{case.tool_name}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            goal_id, step_id = _fixture(runtime, case)
            missing_id = 900_000 + goal_id + step_id
            _set_action(runtime, case, case.missing_args(missing_id))
            calls = _install_counted_handler(runtime, case.tool_name)
            publications = _install_mock_publication(runtime)
            goals_before = _goal_rows(runtime)
            steps_before = _step_rows(runtime)
            token = f"goal-target-corrected-{case.tool_name}"

            refused = runtime.handle("missing goal mutation target", request_token=token)
            item = refused.tool_results[0]
            _assert_recovery_metadata(item, case, missing_id, semantic_preflight=True)
            if (
                calls[0] != 0
                or publications
                or _receipt_rows(runtime)
                or _goal_rows(runtime) != goals_before
                or _step_rows(runtime) != steps_before
            ):
                raise SystemExit(f"{case.tool_name} missing target produced an effect: {item}")
            failed_runs = _rows(runtime, "SELECT * FROM tool_runs ORDER BY id")
            if not failed_runs or any(row["ok"] == 1 or row["approved"] == 1 for row in failed_runs):
                raise SystemExit(f"{case.tool_name} missing target created success evidence: {failed_runs}")
            for row in failed_runs:
                metadata = json.loads(row["metadata"] or "{}")
                if (
                    metadata.get("executed_handler") is not False
                    or metadata.get("handler_invoked") is not False
                ):
                    raise SystemExit(f"{case.tool_name} missing target audit looked executed: {row}")

            corrected_args = case.corrected_args(goal_id, step_id)
            _set_action(runtime, case, corrected_args)
            succeeded = runtime.handle("corrected goal mutation target", request_token=token)
            if not succeeded.tool_results[0].ok or calls[0] != 1:
                raise SystemExit(f"{case.tool_name} corrected same-token request failed: {succeeded.tool_results}")
            _assert_success_effect(runtime, case, goal_id, step_id, publications, 1)
            _assert_completed_receipts(runtime, case, 1)

            replay = runtime.handle("completed goal mutation replay", request_token=token)
            _assert_completed_replay(replay, case, calls)
            if len(publications) != 1:
                raise SystemExit(f"{case.tool_name} completed replay republished the goal")

            fresh = runtime.handle(
                "intentional fresh-token goal mutation",
                request_token=f"{token}-fresh",
            )
            if not fresh.tool_results[0].ok or calls[0] != 2:
                raise SystemExit(f"{case.tool_name} fresh-token execution failed: {fresh.tool_results}")
            expected_publications = 2
            _assert_success_effect(
                runtime,
                case,
                goal_id,
                step_id,
                publications,
                expected_publications,
            )
            if case.tool_name in {"complete_goal_step", "set_goal_status"}:
                fresh_item = fresh.tool_results[0]
                handoff = fresh_item.metadata.get("goal_mutation_handoff") or {}
                boundaries = handoff.get("boundaries") or {}
                if (
                    fresh_item.metadata.get("state_changed") is not False
                    or fresh_item.metadata.get("changed") != []
                    or fresh_item.metadata.get("writes_database") is not False
                    or fresh_item.metadata.get("writes_files") is not True
                    or handoff.get("state_changed") is not False
                    or handoff.get("changed") != []
                    or boundaries.get("writes_files") is not True
                    or boundaries.get("writes_notes") is not True
                ):
                    raise SystemExit(f"{case.tool_name} converged repeat claimed a state change: {fresh_item}")
            _assert_completed_receipts(runtime, case, 2)


def test_post_preflight_convergence_reports_no_change() -> None:
    for case in (CASES[1], CASES[2]):
        with TemporaryDirectory(prefix=f"jarvis-goal-target-converged-{case.tool_name}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            goal_id, step_id = _fixture(runtime, case)
            args = case.corrected_args(goal_id, step_id)
            _set_action(runtime, case, args)
            calls = _install_counted_handler(runtime, case.tool_name)
            publications = _install_mock_publication(runtime)
            original_prepare = runtime.store.prepare_auto_mutation_receipts

            def prepare_then_converge(*prepare_args: Any, **prepare_kwargs: Any):
                prepared = original_prepare(*prepare_args, **prepare_kwargs)
                if case.tool_name == "complete_goal_step":
                    runtime.store.complete_goal_step(step_id)
                else:
                    runtime.store.set_goal_status(goal_id, "paused")
                return prepared

            runtime.store.prepare_auto_mutation_receipts = prepare_then_converge  # type: ignore[method-assign]
            result = runtime.handle(
                "goal target converges after preflight",
                request_token=f"goal-target-converged-{case.tool_name}",
            )
            runtime.store.prepare_auto_mutation_receipts = original_prepare  # type: ignore[method-assign]
            item = result.tool_results[0]
            if (
                not item.ok
                or calls[0] != 1
                or len(publications) != 1
                or item.metadata.get("state_changed") is not False
                or item.metadata.get("changed") != []
                or item.metadata.get("writes_database") is not False
                or item.metadata.get("writes_files") is not True
            ):
                raise SystemExit(f"{case.tool_name} convergence race claimed a mutation: {item}")
            _assert_completed_receipts(runtime, case, 1)


def test_post_preflight_target_disappearance_releases_receipt_for_exact_restore() -> None:
    for case in CASES:
        with TemporaryDirectory(prefix=f"jarvis-goal-target-race-{case.tool_name}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            goal_id, step_id = _fixture(runtime, case)
            args = case.corrected_args(goal_id, step_id)
            _set_action(runtime, case, args)
            calls = _install_counted_handler(runtime, case.tool_name)
            publications = _install_mock_publication(runtime)
            target_table = "goal_steps" if case.target_kind == "step" else "goals"
            target_id = step_id if case.target_kind == "step" else goal_id
            target_row = _rows(
                runtime,
                f"SELECT * FROM {target_table} WHERE id = ?",
                (target_id,),
            )[0]
            original_prepare = runtime.store.prepare_auto_mutation_receipts

            def prepare_then_remove(*prepare_args: Any, **prepare_kwargs: Any):
                prepared = original_prepare(*prepare_args, **prepare_kwargs)
                with runtime.store.connect() as conn:
                    conn.execute(f"DELETE FROM {target_table} WHERE id = ?", (target_id,))
                return prepared

            runtime.store.prepare_auto_mutation_receipts = prepare_then_remove  # type: ignore[method-assign]
            token = f"goal-target-restored-{case.tool_name}"
            refused = runtime.handle("goal target disappears after preflight", request_token=token)
            runtime.store.prepare_auto_mutation_receipts = original_prepare  # type: ignore[method-assign]
            item = refused.tool_results[0]
            _assert_recovery_metadata(item, case, target_id, semantic_preflight=False)
            failed_runs = _rows(runtime, "SELECT * FROM tool_runs ORDER BY id")
            if (
                calls[0] != 1
                or publications
                or item.metadata.get("auto_mutation_effects_started") is not False
                or item.metadata.get("auto_mutation_definite_no_effect") is not True
                or item.metadata.get("auto_mutation_outcome_uncertain") is not False
                or _receipt_rows(runtime)
                or not failed_runs
                or any(row["ok"] == 1 or row["approved"] == 1 for row in failed_runs)
            ):
                raise SystemExit(f"{case.tool_name} post-preflight refusal retained custody: {item}")
            for row in failed_runs:
                metadata = json.loads(row["metadata"] or "{}")
                if (
                    metadata.get("executed_handler") is not True
                    or metadata.get("handler_invoked") is not True
                ):
                    raise SystemExit(f"{case.tool_name} definite-no-effect audit lost execution truth: {row}")

            _restore_row(runtime, target_table, target_row)
            restored = _rows(
                runtime,
                f"SELECT * FROM {target_table} WHERE id = ?",
                (target_id,),
            )[0]
            if restored != target_row:
                raise SystemExit(f"{case.tool_name} target was not restored exactly: {restored}")

            succeeded = runtime.handle("retry after exact goal target restore", request_token=token)
            if not succeeded.tool_results[0].ok or calls[0] != 2:
                raise SystemExit(f"{case.tool_name} same-token restore retry failed: {succeeded.tool_results}")
            _assert_success_effect(runtime, case, goal_id, step_id, publications, 1)
            _assert_completed_receipts(runtime, case, 1)


def main() -> None:
    test_target_identity_contracts_and_invalid_unicode_preflight()
    test_numeric_string_targets_reach_goal_specific_handlers()
    test_goal_mutation_timestamps_are_sampled_after_write_lock()
    test_uncertain_goal_targets_fence_changed_arguments()
    test_missing_targets_stop_before_receipt_and_allow_corrected_same_token()
    test_post_preflight_convergence_reports_no_change()
    test_post_preflight_target_disappearance_releases_receipt_for_exact_restore()
    print("auto-mutation goal target preflight smoke passed")


if __name__ == "__main__":
    main()
