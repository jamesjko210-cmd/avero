from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import threading
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.memory.store import TaskRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import tasks as task_tools
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    TOOL_ARGUMENT_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    ToolArgumentType,
)


TOOLS = ("complete_task", "update_task_status")
ZERO_WRITE_FLAGS = (
    "requires_confirmation",
    "requires_approval",
    "executed_handler",
    "handler_invoked",
    "authorizes_retry",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "queues_approval",
    "state_changed",
    "auto_mutation_effects_started",
    "writes_files",
    "writes_database",
    "writes_memory",
    "writes_notes",
    "external_side_effect",
    "controls_computer",
)


class StaticPlanner:
    def __init__(self, tool_name: str, args: dict[str, Any]):
        self.tool_name = tool_name
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise task-status auto-mutation fencing.",
            [
                PlannedAction(
                    self.tool_name,
                    dict(self.args),
                    "task-status fencing smoke",
                )
            ],
            needs_model=False,
        )


def _runtime(root: Path, tool_name: str, args: dict[str, Any]) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(tool_name, args)
    return runtime


def _set_action(
    runtime: JarvisRuntime,
    tool_name: str,
    args: dict[str, Any],
) -> None:
    runtime.planner = StaticPlanner(tool_name, args)


def _rows(
    runtime: JarvisRuntime,
    sql: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute(sql, params)]


def _receipt_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM auto_mutation_receipts ORDER BY id")


def _tool_run_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM tool_runs ORDER BY id")


def _projection_bytes(runtime: JarvisRuntime) -> bytes | None:
    path = runtime.vault.root_path / "Tasks" / "Open Tasks.md"
    return path.read_bytes() if path.is_file() else None


def _task(runtime: JarvisRuntime, task_id: int) -> dict[str, Any]:
    task = runtime.store.get_task(task_id)
    if task is None:
        raise SystemExit(f"task-status fixture lost task #{task_id}")
    return dict(task)


def _install_counted_handler(
    runtime: JarvisRuntime,
    tool_name: str,
) -> list[int]:
    tool = runtime.registry.get(tool_name)
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools[tool_name] = replace(tool, handler=counted)
    return calls


def _assert_failure_kind(result: Any, expected: str, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} returned an unexpected result count: {result.tool_results}")
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != expected:
        raise SystemExit(f"{label} returned the wrong failure state: {item}")


def _assert_completed_receipt_audits(
    runtime: JarvisRuntime,
    expected: int,
) -> None:
    receipts = _receipt_rows(runtime)
    successful_runs = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
    runs_by_id = {row["id"]: row for row in successful_runs}
    if len(receipts) != expected or len(successful_runs) != expected:
        raise SystemExit(
            f"expected {expected} completed task-status receipt/audit pairs: "
            f"{receipts} / {successful_runs}"
        )
    linked: set[int] = set()
    for receipt in receipts:
        run = runs_by_id.get(receipt["tool_run_id"])
        if (
            receipt["tool_name"] not in TOOLS
            or receipt["state"] != "completed"
            or receipt["result"] != "succeeded"
            or receipt["resolution"] != "recorded"
            or run is None
            or run["tool_name"] != receipt["tool_name"]
            or run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
        ):
            raise SystemExit(
                f"task-status receipt lost ordinary approved=0 audit linkage: {receipt} / {run}"
            )
        linked.add(int(run["id"]))
    if len(linked) != expected:
        raise SystemExit("task-status receipts reused one ordinary audit row")


def _assert_private_receipt_data_hidden(
    runtime: JarvisRuntime,
    results: list[Any],
    private_values: list[str],
) -> None:
    receipts = _receipt_rows(runtime)
    receipt_json = json.dumps(receipts, sort_keys=True, default=str)
    for private in private_values:
        if private and private in receipt_json:
            raise SystemExit(f"task-status receipt exposed raw private data: {private!r}")
    digests: set[str] = set()
    for receipt in receipts:
        for key in (
            "request_digest",
            "action_digest",
            "operation_digest",
            "uncertainty_digest",
            "run_token",
        ):
            value = receipt.get(key)
            if isinstance(value, str) and value:
                digests.add(value)
    public_surfaces = json.dumps(
        {
            "results": [repr(result) for result in results],
            "audit": [
                {"output": row.get("output"), "metadata": row.get("metadata")}
                for row in _tool_run_rows(runtime)
            ],
        },
        sort_keys=True,
        default=str,
    )
    leaked = [value for value in [*private_values, *digests] if value in public_surfaces]
    if leaked:
        raise SystemExit(f"task-status result/audit exposed receipt-private data: {leaked}")


def test_exact_registry_contract_and_shared_target_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-task-status-contract-") as temp:
        runtime = _runtime(Path(temp), "complete_task", {"task_id": 1})
        expected_effects = frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        )
        expected_builders = {
            "complete_task": task_tools.complete_task_auto_mutation_preflight_result,
            "update_task_status": task_tools.update_task_status_auto_mutation_preflight_result,
        }
        for tool_name in TOOLS:
            tool = runtime.registry.get(tool_name)
            contract = tool.auto_mutation_contract
            if (
                tool.toolset != "tasks"
                or contract is None
                or contract.version != AUTO_MUTATION_CONTRACT_VERSION
                or contract.effects != expected_effects
                or contract.replay_policy
                is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
                or contract.crash_policy
                is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
                or getattr(contract, "operation_scope", None) != "task_status"
                or contract.operation_key_builder
                is not task_tools.task_status_auto_mutation_operation_key
                or contract.semantic_preflight is None
                or contract.semantic_preflight_result_builder
                is not expected_builders[tool_name]
                or contract.definite_no_effect_failure_reasons
                != frozenset({"missing_task"})
            ):
                raise SystemExit(f"{tool_name} exact task-status contract drifted: {tool}")

        integer = frozenset({ToolArgumentType.INTEGER})
        string = frozenset({ToolArgumentType.STRING})
        expected_shapes = {
            "complete_task": (("task_id", integer, True),),
            "update_task_status": (
                ("status", string, True),
                ("task_id", integer, True),
            ),
        }
        for tool_name, expected_shape in expected_shapes.items():
            arguments = runtime.registry.get(tool_name).argument_contract
            shape = (
                tuple(
                    (field.name, field.types, field.required)
                    for field in arguments.fields
                )
                if arguments is not None
                else ()
            )
            if (
                arguments is None
                or arguments.version != TOOL_ARGUMENT_CONTRACT_VERSION
                or arguments.allow_unknown
                or shape != expected_shape
            ):
                raise SystemExit(f"{tool_name} strict argument contract drifted: {arguments}")

        complete_contract = runtime.registry.get("complete_task").auto_mutation_contract
        update_contract = runtime.registry.get("update_task_status").auto_mutation_contract
        assert complete_contract is not None and update_contract is not None
        keys = [
            complete_contract.operation_key_builder({"task_id": 17}),
            update_contract.operation_key_builder({"task_id": 17, "status": "open"}),
            update_contract.operation_key_builder({"task_id": 17, "status": "done"}),
            update_contract.operation_key_builder({"task_id": 17, "status": "paused"}),
            update_contract.operation_key_builder({"task_id": 17, "status": "dropped"}),
        ]
        if any(key != {"task_id": 17} for key in keys):
            raise SystemExit(f"task-status identity retained tool or changed-status data: {keys}")
        if update_contract.operation_key_builder({"task_id": 18, "status": "done"}) == keys[0]:
            raise SystemExit("task-status identity did not distinguish task targets")

        task_id = runtime.store.add_task(TaskRecord("contract preflight target"))
        if (
            complete_contract.semantic_preflight({"task_id": 0}) != "bad_task_id"
            or complete_contract.semantic_preflight({"task_id": -1}) != "bad_task_id"
            or complete_contract.semantic_preflight({"task_id": "bad"}) != "bad_task_id"
            or complete_contract.semantic_preflight({"task_id": 999999}) != "missing_task"
            or complete_contract.semantic_preflight({"task_id": task_id}) is not None
            or update_contract.semantic_preflight(
                {"task_id": task_id, "status": "unsupported"}
            )
            != "bad_status"
            or update_contract.semantic_preflight(
                {"task_id": task_id, "status": " PAUSED "}
            )
            is not None
        ):
            raise SystemExit("task-status semantic preflight reason ordering drifted")


def test_invalid_and_missing_preflight_is_zero_write() -> None:
    cases = (
        ("complete_task", {"task_id": 0}, "bad_task_id"),
        ("complete_task", {"task_id": -7}, "bad_task_id"),
        ("complete_task", {"task_id": 999999}, "missing_task"),
        ("update_task_status", {"task_id": 0, "status": "done"}, "bad_task_id"),
        ("update_task_status", {"task_id": -7, "status": "done"}, "bad_task_id"),
        (
            "update_task_status",
            {"task_id": 1, "status": "/\x55sers/private/CLIENT-STATUS-SECRET"},
            "bad_status",
        ),
        (
            "update_task_status",
            {"task_id": 999999, "status": "paused"},
            "missing_task",
        ),
    )
    for index, (tool_name, raw_args, reason) in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-task-status-preflight-") as temp:
            runtime = _runtime(Path(temp), tool_name, raw_args)
            task_id = runtime.store.add_task(TaskRecord("preflight remains unchanged"))
            runtime.vault.sync_open_tasks(runtime.store)
            args = dict(raw_args)
            if reason == "bad_status":
                args["task_id"] = task_id
                _set_action(runtime, tool_name, args)
            tasks_before = _rows(runtime, "SELECT * FROM tasks ORDER BY id")
            projection_before = _projection_bytes(runtime)
            calls = _install_counted_handler(runtime, tool_name)
            token = f"PRIVATE-TASK-STATUS-PREFLIGHT-{index}"
            result = runtime.handle("reject task-status preflight", request_token=token)
            item = result.tool_results[0]
            if (
                item.ok
                or item.metadata.get("failure_kind")
                != "auto_mutation_semantic_preflight_rejected"
                or item.metadata.get("reason") != reason
                or any(item.metadata.get(flag) is not False for flag in ZERO_WRITE_FLAGS)
                or calls[0] != 0
                or _rows(runtime, "SELECT * FROM tasks ORDER BY id") != tasks_before
                or _projection_bytes(runtime) != projection_before
                or _receipt_rows(runtime)
                or any(row["ok"] == 1 or row["approved"] == 1 for row in _tool_run_rows(runtime))
            ):
                raise SystemExit(f"task-status preflight case {index} crossed a boundary: {item}")
            _assert_private_receipt_data_hidden(runtime, [result], [token, str(temp)])
            if "CLIENT-STATUS-SECRET" in json.dumps(item.metadata, default=str):
                raise SystemExit("task-status semantic refusal exposed a raw private status")


def test_same_token_replay_collision_fresh_repeat_audit_and_privacy() -> None:
    owner = "PRIVATE-TASK-STATUS-SUCCESS-OWNER"
    fresh_token = "PRIVATE-TASK-STATUS-SUCCESS-FRESH"
    with TemporaryDirectory(prefix="jarvis-task-status-success-") as temp:
        runtime = _runtime(Path(temp), "update_task_status", {})
        task_id = runtime.store.add_task(TaskRecord("status replay target"))
        runtime.vault.sync_open_tasks(runtime.store)
        args = {"task_id": task_id, "status": "paused"}
        _set_action(runtime, "update_task_status", args)
        calls = _install_counted_handler(runtime, "update_task_status")

        first = runtime.handle("pause task once", request_token=owner)
        if not first.tool_results[0].ok or calls[0] != 1 or _task(runtime, task_id)["status"] != "paused":
            raise SystemExit(f"initial task-status mutation drifted: {first.tool_results}")
        _assert_completed_receipt_audits(runtime, 1)

        replay = runtime.handle("same-token status replay", request_token=owner)
        _assert_failure_kind(replay, "auto_mutation_completed_replay", "same-token replay")
        if calls[0] != 1:
            raise SystemExit("same-token completed replay reached the status handler")

        _set_action(runtime, "update_task_status", {"task_id": task_id, "status": "done"})
        collision = runtime.handle("same-token changed status", request_token=owner)
        _assert_failure_kind(
            collision,
            "auto_mutation_request_collision",
            "same-token changed-status collision",
        )
        if calls[0] != 1 or _task(runtime, task_id)["status"] != "paused":
            raise SystemExit("same-token changed-status collision reached the handler")

        _set_action(runtime, "update_task_status", args)
        fresh = runtime.handle("fresh intentional completed repeat", request_token=fresh_token)
        if not fresh.tool_results[0].ok or calls[0] != 2 or _task(runtime, task_id)["status"] != "paused":
            raise SystemExit(f"fresh completed status repeat did not converge: {fresh.tool_results}")
        _assert_completed_receipt_audits(runtime, 2)
        _assert_private_receipt_data_hidden(
            runtime,
            [first, replay, collision, fresh],
            [owner, fresh_token],
        )


def _assert_uncertain_owner_and_blocked_contender(
    *,
    owner_tool: str,
    owner_args: dict[str, Any],
    contender_tool: str,
    contender_args: dict[str, Any],
    expected_status: str,
    label: str,
) -> None:
    owner_token = f"PRIVATE-{label}-OWNER"
    contender_token = f"PRIVATE-{label}-CONTENDER"
    with TemporaryDirectory(prefix=f"jarvis-{label.lower()}-") as temp:
        runtime = _runtime(Path(temp), owner_tool, {})
        task_id = runtime.store.add_task(TaskRecord(f"{label} stale projection"))
        runtime.vault.sync_open_tasks(runtime.store)
        projection_before = _projection_bytes(runtime)
        owner_args = {**owner_args, "task_id": task_id}
        contender_args = {**contender_args, "task_id": task_id}
        _set_action(runtime, owner_tool, owner_args)
        owner_calls = _install_counted_handler(runtime, owner_tool)
        contender_calls = (
            owner_calls
            if contender_tool == owner_tool
            else _install_counted_handler(runtime, contender_tool)
        )
        publications = [0]

        def fail_projection(*_args: Any, **_kwargs: Any) -> Any:
            publications[0] += 1
            raise OSError("injected task-status projection failure")

        runtime.vault.sync_open_tasks = fail_projection  # type: ignore[method-assign]
        failed = runtime.handle("fail task-status projection", request_token=owner_token)
        receipt = _receipt_rows(runtime)
        if (
            failed.tool_results[0].ok
            or failed.tool_results[0].metadata.get("auto_mutation_outcome_uncertain") is not True
            or owner_calls[0] != 1
            or publications[0] != 1
            or _task(runtime, task_id)["status"] != expected_status
            or _projection_bytes(runtime) != projection_before
            or len(receipt) != 1
            or receipt[0]["state"] != "uncertain"
            or receipt[0]["result"] != "unknown"
            or any(row["ok"] == 1 for row in _tool_run_rows(runtime))
        ):
            raise SystemExit(f"{label} did not retain uncertain status custody: {failed.tool_results}")

        same = runtime.handle("same-token uncertain replay", request_token=owner_token)
        _assert_failure_kind(same, "auto_mutation_outcome_uncertain", f"{label} same-token replay")
        _set_action(runtime, contender_tool, contender_args)
        blocked = runtime.handle("same-target contender", request_token=contender_token)
        _assert_failure_kind(
            blocked,
            "auto_mutation_unresolved_action",
            f"{label} contender",
        )
        expected_owner_calls = 1
        expected_contender_calls = 1 if contender_calls is owner_calls else 0
        if (
            owner_calls[0] != expected_owner_calls
            or contender_calls[0] != expected_contender_calls
            or publications[0] != 1
            or _task(runtime, task_id)["status"] != expected_status
            or len(_receipt_rows(runtime)) != 1
        ):
            raise SystemExit(f"{label} contender bypassed unresolved task-status fencing")
        _assert_private_receipt_data_hidden(
            runtime,
            [failed, same, blocked],
            [owner_token, contender_token],
        )


def test_changed_status_and_cross_tool_unresolved_fencing() -> None:
    _assert_uncertain_owner_and_blocked_contender(
        owner_tool="update_task_status",
        owner_args={"status": "paused"},
        contender_tool="update_task_status",
        contender_args={"status": "done"},
        expected_status="paused",
        label="TASK-STATUS-CHANGED-STATUS",
    )
    _assert_uncertain_owner_and_blocked_contender(
        owner_tool="complete_task",
        owner_args={},
        contender_tool="update_task_status",
        contender_args={"status": "open"},
        expected_status="done",
        label="TASK-STATUS-COMPLETE-TO-UPDATE",
    )
    _assert_uncertain_owner_and_blocked_contender(
        owner_tool="update_task_status",
        owner_args={"status": "paused"},
        contender_tool="complete_task",
        contender_args={},
        expected_status="paused",
        label="TASK-STATUS-UPDATE-TO-COMPLETE",
    )


def _restore_task(runtime: JarvisRuntime, task: dict[str, Any]) -> None:
    with runtime.store.connect() as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, body, source, due, priority, status,
                created_at, updated_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task["id"],
                task["body"],
                task["source"],
                task["due"],
                task["priority"],
                task["status"],
                task["created_at"],
                task["updated_at"],
                task["completed_at"],
            ),
        )


def test_post_preflight_disappearance_abandons_receipt() -> None:
    cases = (
        ("complete_task", {}, "done"),
        ("update_task_status", {"status": "paused"}, "paused"),
    )
    for index, (tool_name, extra_args, expected_status) in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-task-status-disappearance-") as temp:
            runtime = _runtime(Path(temp), tool_name, {})
            task_id = runtime.store.add_task(TaskRecord("post-preflight disappearance"))
            original_task = _task(runtime, task_id)
            runtime.vault.sync_open_tasks(runtime.store)
            projection_before = _projection_bytes(runtime)
            args = {"task_id": task_id, **extra_args}
            _set_action(runtime, tool_name, args)
            original_prepare = runtime.store.prepare_auto_mutation_receipts

            def prepare_then_remove(*prepare_args: Any, **prepare_kwargs: Any):
                prepared = original_prepare(*prepare_args, **prepare_kwargs)
                with runtime.store.connect() as conn:
                    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                return prepared

            runtime.store.prepare_auto_mutation_receipts = prepare_then_remove  # type: ignore[method-assign]
            token = f"PRIVATE-TASK-STATUS-DISAPPEARANCE-{index}"
            failed = runtime.handle("task disappears after preflight", request_token=token)
            runtime.store.prepare_auto_mutation_receipts = original_prepare  # type: ignore[method-assign]
            item = failed.tool_results[0]
            if (
                item.ok
                or item.metadata.get("reason") != "missing_task"
                or item.metadata.get("auto_mutation_effects_started") is not False
                or item.metadata.get("writes_database") is not False
                or item.metadata.get("auto_mutation_definite_no_effect") is not True
                or item.metadata.get("auto_mutation_outcome_uncertain") is not False
                or _receipt_rows(runtime)
                or _projection_bytes(runtime) != projection_before
            ):
                raise SystemExit(
                    f"{tool_name} post-preflight disappearance retained receipt custody: {item}"
                )

            _restore_task(runtime, original_task)
            retried = runtime.handle("retry after target refresh", request_token=token)
            if (
                not retried.tool_results[0].ok
                or _task(runtime, task_id)["status"] != expected_status
            ):
                raise SystemExit(
                    f"{tool_name} definite-no-effect disappearance did not release retry"
                )
            _assert_completed_receipt_audits(runtime, 1)
            _assert_private_receipt_data_hidden(runtime, [failed, retried], [token])


def test_concurrent_cross_tool_same_target_has_one_winner() -> None:
    with TemporaryDirectory(prefix="jarvis-task-status-concurrent-") as temp:
        runtime = _runtime(Path(temp), "complete_task", {})
        task_id = runtime.store.add_task(TaskRecord("concurrent status target"))
        runtime.vault.sync_open_tasks(runtime.store)
        _set_action(runtime, "complete_task", {"task_id": task_id})
        complete_tool = runtime.registry.get("complete_task")
        update_calls = _install_counted_handler(runtime, "update_task_status")
        entered = threading.Event()
        release = threading.Event()
        complete_calls = [0]

        def delayed_complete(args: dict[str, Any]) -> ToolResult:
            complete_calls[0] += 1
            entered.set()
            if not release.wait(10):
                raise TimeoutError("concurrent task-status owner was not released")
            return complete_tool.handler(args)

        runtime.registry._tools["complete_task"] = replace(
            complete_tool,
            handler=delayed_complete,
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            winner_future = pool.submit(
                runtime.handle,
                "concurrent complete owner",
                request_token="PRIVATE-TASK-STATUS-CONCURRENT-OWNER",
            )
            if not entered.wait(10):
                release.set()
                raise SystemExit("concurrent complete owner did not enter the handler")
            _set_action(
                runtime,
                "update_task_status",
                {"task_id": task_id, "status": "paused"},
            )
            try:
                blocked = runtime.handle(
                    "concurrent update contender",
                    request_token="PRIVATE-TASK-STATUS-CONCURRENT-CONTENDER",
                )
            finally:
                release.set()
            winner = winner_future.result(timeout=15)

        if (
            not winner.tool_results[0].ok
            or complete_calls[0] != 1
            or update_calls[0] != 0
            or _task(runtime, task_id)["status"] != "done"
        ):
            raise SystemExit(f"concurrent task-status winner drifted: {winner.tool_results}")
        _assert_failure_kind(
            blocked,
            "auto_mutation_unresolved_action",
            "concurrent cross-tool contender",
        )
        _assert_completed_receipt_audits(runtime, 1)
        _assert_private_receipt_data_hidden(
            runtime,
            [winner, blocked],
            [
                "PRIVATE-TASK-STATUS-CONCURRENT-OWNER",
                "PRIVATE-TASK-STATUS-CONCURRENT-CONTENDER",
            ],
        )


def main() -> None:
    test_exact_registry_contract_and_shared_target_identity()
    test_invalid_and_missing_preflight_is_zero_write()
    test_same_token_replay_collision_fresh_repeat_audit_and_privacy()
    test_changed_status_and_cross_tool_unresolved_fencing()
    test_post_preflight_disappearance_abandons_receipt()
    test_concurrent_cross_tool_same_target_has_one_winner()
    print("Task-status auto-mutation fencing smoke passed")


if __name__ == "__main__":
    main()
