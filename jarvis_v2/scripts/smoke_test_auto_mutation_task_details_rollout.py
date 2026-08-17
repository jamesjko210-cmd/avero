from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import threading
from tempfile import TemporaryDirectory
from typing import Any, Callable

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.memory.store import TaskRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    TOOL_ARGUMENT_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    ToolArgumentType,
)


TOOL_NAME = "update_task_details"
PROCESS_DEATH_TOKEN = "PRIVATE-TASK-DETAILS-PROCESS-DEATH-OWNER"
PROCESS_DEATH_BODY = "process death task details committed"


class StaticPlanner:
    def __init__(self, args: Any):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise the aggregate-wired task-details mutation.",
            [PlannedAction(TOOL_NAME, self.args, "task details rollout smoke")],
            needs_model=False,
        )


def _runtime(root: Path, args: Any) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(args)
    return runtime


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


def _task(runtime: JarvisRuntime, task_id: int) -> dict[str, Any]:
    row = runtime.store.get_task(task_id)
    if row is None:
        raise SystemExit(f"task-details fixture lost task #{task_id}")
    return dict(row)


def _projection_bytes(runtime: JarvisRuntime) -> bytes | None:
    path = runtime.vault.root_path / "Tasks" / "Open Tasks.md"
    return path.read_bytes() if path.is_file() else None


def _install_counted_handler(runtime: JarvisRuntime) -> list[int]:
    tool = runtime.registry.get(TOOL_NAME)
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools[TOOL_NAME] = replace(tool, handler=counted)
    return calls


def _install_projection_failure(
    runtime: JarvisRuntime,
    callback: Callable[..., Any],
) -> None:
    runtime.vault.sync_open_tasks = callback  # type: ignore[method-assign]
    runtime.vault.sync_open_tasks_with_evidence = callback  # type: ignore[method-assign]


def _assert_replay(result: Any, failure_kind: str, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} did not return one tool result: {result.tool_results}")
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != failure_kind:
        raise SystemExit(f"{label} returned the wrong replay state: {item}")


def _assert_completed_receipt_audits(runtime: JarvisRuntime, expected: int) -> None:
    receipts = _receipt_rows(runtime)
    successful_runs = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
    runs_by_id = {row["id"]: row for row in successful_runs}
    if len(receipts) != expected or len(successful_runs) != expected:
        raise SystemExit(
            f"expected {expected} task-details receipt/audit pairs: "
            f"{receipts} / {successful_runs}"
        )
    linked: set[int] = set()
    for receipt in receipts:
        run = runs_by_id.get(receipt["tool_run_id"])
        if (
            receipt["tool_name"] != TOOL_NAME
            or receipt["state"] != "completed"
            or receipt["result"] != "succeeded"
            or receipt["resolution"] != "recorded"
            or run is None
            or run["tool_name"] != TOOL_NAME
            or run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
        ):
            raise SystemExit(f"task-details receipt lost ordinary audit linkage: {receipt} / {run}")
        linked.add(int(run["id"]))
    if len(linked) != expected:
        raise SystemExit("task-details receipts reused one ordinary audit row")


def _assert_receipt_privacy(
    runtime: JarvisRuntime,
    *,
    private_values: list[str],
) -> None:
    receipts = _receipt_rows(runtime)
    serialized = json.dumps(receipts, ensure_ascii=False, sort_keys=True, default=str)
    for private in private_values:
        if private and private in serialized:
            raise SystemExit(f"task-details receipt exposed private input: {private!r}")
    for receipt in receipts:
        for key in ("request_digest", "action_digest", "operation_digest"):
            value = receipt.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise SystemExit(f"task-details receipt lost a bounded digest: {receipt}")
    if str(runtime.vault.root_path) in serialized or str(runtime.store.db_path.parent) in serialized:
        raise SystemExit("task-details receipts exposed an absolute temporary path")


def _assert_no_absolute_path_metadata(runtime: JarvisRuntime, results: list[Any]) -> None:
    metadata = [item.metadata for result in results for item in result.tool_results]
    metadata.extend(row.get("metadata") for row in _tool_run_rows(runtime))
    serialized = json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)
    if str(runtime.vault.root_path) in serialized or str(runtime.store.db_path.parent) in serialized:
        raise SystemExit("task-details result or audit metadata exposed an absolute local path")
    for item in metadata:
        if isinstance(item, dict) and any(key in item for key in ("path", "tasks_path")):
            raise SystemExit(f"task-details metadata retained a raw path key: {item}")


def _assert_no_private_path_output(
    runtime: JarvisRuntime,
    results: list[Any],
    private_path: str,
) -> None:
    surfaces = [
        result.response
        for result in results
    ]
    surfaces.extend(
        json.dumps(result.metadata, ensure_ascii=False, sort_keys=True, default=str)
        for result in results
    )
    surfaces.extend(
        item.output
        for result in results
        for item in result.tool_results
    )
    surfaces.extend(str(row.get("output") or "") for row in _tool_run_rows(runtime))
    surfaces.extend(
        str(row.get("metadata") or "")
        for row in _rows(
            runtime,
            "SELECT metadata FROM messages WHERE role = 'assistant' ORDER BY id",
        )
    )
    surfaces.append((_projection_bytes(runtime) or b"").decode("utf-8"))
    private_fragments = [private_path]
    if ";" in private_path:
        private_fragments.append(private_path.split(";", 1)[1])
    if any(
        fragment in str(surface)
        for surface in surfaces
        for fragment in private_fragments
    ):
        raise SystemExit("task-details output, audit, or projection leaked a local path fragment")


def _assert_no_boundary_crossing(
    runtime: JarvisRuntime,
    *,
    task_id: int,
    task_before: dict[str, Any],
    projection_before: bytes | None,
    label: str,
) -> None:
    if _task(runtime, task_id) != task_before:
        raise SystemExit(f"{label} changed the task row")
    if _projection_bytes(runtime) != projection_before:
        raise SystemExit(f"{label} changed the task projection")
    if _receipt_rows(runtime):
        raise SystemExit(f"{label} created an auto-mutation receipt")
    if any(row["ok"] == 1 or row["approved"] == 1 for row in _tool_run_rows(runtime)):
        raise SystemExit(f"{label} created success or approval evidence")


def _crash_after_task_details_database_commit(root_text: str, task_id: int) -> None:
    args = {
        "task_id": task_id,
        "body": PROCESS_DEATH_BODY,
        "due": "next month",
        "priority": "high",
    }
    runtime = _runtime(Path(root_text), args)

    def exit_before_projection(*_args: Any, **_kwargs: Any) -> Any:
        row = _task(runtime, task_id)
        if (
            row["body"] == PROCESS_DEATH_BODY
            and row["due"] == "next month"
            and row["priority"] == "high"
        ):
            os._exit(73)
        os._exit(74)

    _install_projection_failure(runtime, exit_before_projection)
    runtime.handle(
        "update task details before representative process death",
        request_token=PROCESS_DEATH_TOKEN,
    )
    os._exit(75)


def test_exact_contract_schema_and_target_only_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-task-details-contract-") as temp:
        runtime = _runtime(Path(temp), {"task_id": 17, "body": "contract"})
        tool = runtime.registry.get(TOOL_NAME)
        contract = tool.auto_mutation_contract
        if (
            tool.toolset != "tasks"
            or contract is None
            or contract.version != AUTO_MUTATION_CONTRACT_VERSION
            or contract.effects
            != frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT})
            or contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
            or contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
            or contract.operation_key_builder is None
            or contract.semantic_preflight is None
            or contract.semantic_preflight_result_builder is None
            or contract.definite_no_effect_failure_reasons != frozenset({"missing_task"})
        ):
            raise SystemExit(f"{TOOL_NAME} auto-mutation contract drifted: {tool}")

        arguments = tool.argument_contract
        string = frozenset({ToolArgumentType.STRING})
        integer = frozenset({ToolArgumentType.INTEGER})
        shape = (
            tuple((field.name, field.types, field.required) for field in arguments.fields)
            if arguments is not None
            else ()
        )
        if (
            arguments is None
            or arguments.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or arguments.allow_unknown
            or shape
            != (
                ("body", string, False),
                ("due", string, False),
                ("priority", string, False),
                ("task_id", integer, True),
            )
        ):
            raise SystemExit(f"{TOOL_NAME} strict argument contract drifted: {arguments}")

        builder = contract.operation_key_builder
        equivalent = [
            builder({"task_id": 17, "body": "first"}),
            builder({"task_id": 17, "due": "tomorrow"}),
            builder({"task_id": 17, "priority": "high"}),
            builder(
                {
                    "task_id": 17,
                    "body": "changed",
                    "due": "next week",
                    "priority": "low",
                }
            ),
        ]
        if any(item != {"task_id": 17} for item in equivalent):
            raise SystemExit(f"task-details operation identity retained changed values: {equivalent}")
        if builder({"task_id": 18, "body": "first"}) == equivalent[0]:
            raise SystemExit("task-details operation identity did not distinguish task targets")


def test_typed_and_semantic_refusals_precede_handler_and_receipt() -> None:
    typed_cases: list[Any] = [
        {},
        {"task_id": None, "body": "typed"},
        {"task_id": "1", "body": "typed"},
        {"task_id": True, "body": "typed"},
        {"task_id": 1, "body": None},
        {"task_id": 1, "due": 3},
        {"task_id": 1, "priority": False},
        {"task_id": 1, "body": "typed", "unknown": "rejected"},
        None,
        [],
    ]
    for index, args in enumerate(typed_cases):
        with TemporaryDirectory(prefix="jarvis-task-details-typed-") as temp:
            runtime = _runtime(Path(temp), args)
            task_id = runtime.store.add_task(TaskRecord("typed refusal original"))
            runtime.vault.sync_open_tasks(runtime.store)
            task_before = _task(runtime, task_id)
            projection_before = _projection_bytes(runtime)
            calls = _install_counted_handler(runtime)
            result = runtime.handle(
                "invalid typed task details",
                request_token=f"task-details-typed-{index}",
            )
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
            ):
                raise SystemExit(f"task-details typed case {index} drifted: {item}")
            _assert_no_boundary_crossing(
                runtime,
                task_id=task_id,
                task_before=task_before,
                projection_before=projection_before,
                label=f"task-details typed case {index}",
            )

    semantic_cases = [
        lambda task_id: {"task_id": task_id},
        lambda task_id: {"task_id": task_id, "body": "   "},
        lambda task_id: {"task_id": task_id, "priority": "urgent"},
        lambda task_id: {"task_id": task_id, "body": "invalid\ud800body"},
        lambda task_id: {"task_id": task_id, "due": "invalid\udfff due"},
        lambda _task_id: {"task_id": 999999, "body": "missing target"},
    ]
    for index, make_args in enumerate(semantic_cases):
        with TemporaryDirectory(prefix="jarvis-task-details-semantic-") as temp:
            runtime = _runtime(Path(temp), {})
            task_id = runtime.store.add_task(TaskRecord("semantic refusal original"))
            runtime.vault.sync_open_tasks(runtime.store)
            runtime.planner = StaticPlanner(make_args(task_id))
            task_before = _task(runtime, task_id)
            projection_before = _projection_bytes(runtime)
            calls = _install_counted_handler(runtime)
            result = runtime.handle(
                "invalid semantic task details",
                request_token=f"task-details-semantic-{index}",
            )
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind")
                != "auto_mutation_semantic_preflight_rejected"
                or item.metadata.get("handler_invoked") is not False
                or item.metadata.get("state_changed") is not False
                or calls[0] != 0
            ):
                raise SystemExit(f"task-details semantic case {index} drifted: {item}")
            _assert_no_boundary_crossing(
                runtime,
                task_id=task_id,
                task_before=task_before,
                projection_before=projection_before,
                label=f"task-details semantic case {index}",
            )
            _assert_no_absolute_path_metadata(runtime, [result])


def test_refusal_retry_commands_use_deterministic_planner_grammar() -> None:
    fixtures = [
        ({"task_id": 1}, "missing_update"),
        ({"task_id": 1, "body": "   "}, "missing_body"),
        ({"task_id": 1, "priority": "urgent"}, "bad_priority"),
        ({"task_id": 1, "body": "invalid\ud800body"}, "invalid_unicode"),
        ({"task_id": 1, "due": "invalid\udfff due"}, "invalid_unicode"),
        ({"task_id": 1, "body": "   ", "priority": "high"}, "missing_body"),
        (
            {"task_id": 1, "body": "invalid\ud800body", "priority": "high"},
            "invalid_unicode",
        ),
        (
            {"task_id": 1, "due": "invalid\udfff due", "priority": "high"},
            "invalid_unicode",
        ),
        ({"task_id": 999999, "body": "missing target"}, "missing_task"),
    ]
    with TemporaryDirectory(prefix="jarvis-task-details-recovery-routing-") as temp:
        runtime = _runtime(Path(temp), {})
        runtime.store.add_task(TaskRecord("recovery routing target"))
        contract = runtime.registry.get(TOOL_NAME).auto_mutation_contract
        if contract is None or contract.semantic_preflight_result_builder is None:
            raise SystemExit("task-details recovery routing lost its preflight result builder")
        for args, reason in fixtures:
            result = contract.semantic_preflight_result_builder(dict(args), reason)
            recovery_commands = list(result.metadata.get("recovery_commands") or [])
            refusal_handoff = result.metadata.get("task_refusal_handoff") or {}
            handoff_next = refusal_handoff.get("next_commands") or {}
            template = (
                recovery_commands[-1]
                if recovery_commands
                else str(
                    handoff_next.get("retry")
                    or result.metadata.get("next_safe_command")
                    or ""
                )
            )
            materialized = (
                template.replace("<correct task id>", "1")
                .replace("<task id>", "1")
                .replace("<low|normal|high>", "high")
                .replace("<due>", "tomorrow")
                .replace("<body>", "safe replacement body")
            )
            plan = RuleBasedPlanner().plan(materialized)
            if (
                len(plan.actions) != 1
                or plan.actions[0].tool_name != TOOL_NAME
                or plan.needs_model
            ):
                raise SystemExit(
                    f"task-details refusal advertised a non-routable recovery command: "
                    f"{template!r} -> {materialized!r} -> {plan}"
                )


def test_combined_success_replay_fresh_convergence_audit_and_privacy() -> None:
    owner = "PRIVATE-TASK-DETAILS-SUCCESS-OWNER"
    fresh_token = "PRIVATE-TASK-DETAILS-SUCCESS-FRESH"
    private_body = "/users/PRIVATE/account/;CLIENT-ALPHA-secret-task-8841.txt"
    private_due = "PRIVATE Friday 8842"
    with TemporaryDirectory(prefix="jarvis-task-details-success-") as temp:
        root = Path(temp)
        runtime = _runtime(root, {})
        task_id = runtime.store.add_task(TaskRecord("combined update original"))
        runtime.vault.sync_open_tasks(runtime.store)
        args = {
            "task_id": task_id,
            "body": f"  {private_body}  ",
            "due": f"  {private_due}  ",
            "priority": " HIGH ",
        }
        runtime.planner = StaticPlanner(args)
        calls = _install_counted_handler(runtime)

        first = runtime.handle("update all task details", request_token=owner)
        first_item = first.tool_results[0]
        row = _task(runtime, task_id)
        projection = (_projection_bytes(runtime) or b"").decode("utf-8")
        expected_line = f"- [ ] #{task_id} <local-path> due {private_due} priority high"
        if (
            not first_item.ok
            or calls[0] != 1
            or row["body"] != private_body
            or row["due"] != private_due
            or row["priority"] != "high"
            or expected_line not in projection.splitlines()
            or "combined update original" in projection
            or first_item.metadata.get("changed") != ["body", "due", "priority"]
        ):
            raise SystemExit(f"combined task-details update drifted: {first_item} / {row}")
        _assert_completed_receipt_audits(runtime, 1)

        same = runtime.handle("same-token task-details replay", request_token=owner)
        _assert_replay(same, "auto_mutation_completed_replay", "same-token task-details replay")
        if calls[0] != 1 or _task(runtime, task_id) != row:
            raise SystemExit("same-token task-details replay reached the handler")

        fresh = runtime.handle("fresh-token exact task-details update", request_token=fresh_token)
        fresh_item = fresh.tool_results[0]
        fresh_projection = (_projection_bytes(runtime) or b"").decode("utf-8")
        if (
            not fresh_item.ok
            or calls[0] != 2
            or _task(runtime, task_id)["body"] != private_body
            or fresh_projection.splitlines().count(expected_line) != 1
        ):
            raise SystemExit(f"fresh-token task-details update did not converge: {fresh_item}")
        _assert_completed_receipt_audits(runtime, 2)
        _assert_receipt_privacy(
            runtime,
            private_values=[owner, fresh_token, private_body, private_due, str(root)],
        )
        _assert_no_absolute_path_metadata(runtime, [first, same, fresh])
        _assert_no_private_path_output(runtime, [first, same, fresh], private_body)


def test_post_preflight_missing_task_releases_single_action_receipt() -> None:
    args = {"task_id": 1, "body": "retry after target refresh", "priority": "high"}
    with TemporaryDirectory(prefix="jarvis-task-details-post-preflight-missing-") as temp:
        runtime = _runtime(Path(temp), args)
        task_id = runtime.store.add_task(TaskRecord("target disappears after preflight"))
        original_task = _task(runtime, task_id)
        runtime.planner = StaticPlanner({**args, "task_id": task_id})
        runtime.vault.sync_open_tasks(runtime.store)
        original_prepare = runtime.store.prepare_auto_mutation_receipts

        def prepare_then_remove(*prepare_args: Any, **prepare_kwargs: Any):
            prepared = original_prepare(*prepare_args, **prepare_kwargs)
            with runtime.store.connect() as conn:
                conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            return prepared

        runtime.store.prepare_auto_mutation_receipts = prepare_then_remove  # type: ignore[method-assign]
        token = "task-details-post-preflight-missing-owner"
        failed = runtime.handle(
            "task disappears after task-details preflight",
            request_token=token,
        )
        runtime.store.prepare_auto_mutation_receipts = original_prepare  # type: ignore[method-assign]
        item = failed.tool_results[0]
        if (
            item.ok
            or item.metadata.get("reason") != "missing_task"
            or item.metadata.get("auto_mutation_effects_started") is not False
            or item.metadata.get("auto_mutation_definite_no_effect") is not True
            or item.metadata.get("auto_mutation_outcome_uncertain") is not False
            or _receipt_rows(runtime)
        ):
            raise SystemExit(f"post-preflight missing task retained mutation custody: {item}")

        with runtime.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks(
                    id, body, source, due, priority, status,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    original_task["id"],
                    original_task["body"],
                    original_task["source"],
                    original_task["due"],
                    original_task["priority"],
                    original_task["status"],
                    original_task["created_at"],
                    original_task["updated_at"],
                    original_task["completed_at"],
                ),
            )
        retried = runtime.handle(
            "retry task details after target refresh",
            request_token=token,
        )
        if not retried.tool_results[0].ok or _task(runtime, task_id)["body"] != args["body"]:
            raise SystemExit("definite-no-effect task-details refusal did not release same-token retry")


def test_changed_values_are_fenced_after_database_before_projection_failure() -> None:
    owner = "PRIVATE-TASK-DETAILS-PROJECTION-OWNER-7F3C9A1E5D8B42A6"
    contender = "PRIVATE-TASK-DETAILS-PROJECTION-CONTENDER-C4E8B2076A1D593F"
    owner_body = "PRIVATE-TASK-DETAILS-PROJECTION-BODY-91A6D3F8C2E74B50"
    owner_due = "PRIVATE-TASK-DETAILS-PROJECTION-DUE-2B7E4C90A5D168F3"
    contender_body = "PRIVATE-TASK-DETAILS-CONTENDER-BODY-E5C1A793D8246F0B"
    contender_due = "PRIVATE-TASK-DETAILS-CONTENDER-DUE-68D2F4A1C9B705E3"
    with TemporaryDirectory(prefix="jarvis-task-details-projection-failure-") as temp:
        root = Path(temp)
        runtime = _runtime(root, {})
        task_id = runtime.store.add_task(TaskRecord("projection remains stale"))
        runtime.vault.sync_open_tasks(runtime.store)
        projection_before = _projection_bytes(runtime)
        owner_args = {
            "task_id": task_id,
            "body": owner_body,
            "due": owner_due,
            "priority": "low",
        }
        runtime.planner = StaticPlanner(owner_args)
        calls = _install_counted_handler(runtime)
        publication_calls = [0]

        def fail_projection(*_args: Any, **_kwargs: Any) -> Any:
            publication_calls[0] += 1
            raise OSError("injected task-details projection failure")

        _install_projection_failure(runtime, fail_projection)
        failed = runtime.handle("fail task-details projection", request_token=owner)
        item = failed.tool_results[0]
        receipt = _receipt_rows(runtime)
        row = _task(runtime, task_id)
        if (
            item.ok
            or item.metadata.get("auto_mutation_outcome_uncertain") is not True
            or calls[0] != 1
            or publication_calls[0] != 1
            or row["body"] != owner_body
            or row["due"] != owner_due
            or row["priority"] != "low"
            or _projection_bytes(runtime) != projection_before
            or len(receipt) != 1
            or receipt[0]["state"] != "uncertain"
            or receipt[0]["result"] != "unknown"
            or any(run["ok"] == 1 for run in _tool_run_rows(runtime))
        ):
            raise SystemExit(f"task-details projection failure lost uncertain custody: {item}")

        same = runtime.handle("same-token projection replay", request_token=owner)
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token projection replay")
        changed_args = {
            "task_id": task_id,
            "body": contender_body,
            "due": contender_due,
            "priority": "high",
        }
        runtime.planner = StaticPlanner(changed_args)
        blocked = runtime.handle("changed-value task-details contender", request_token=contender)
        _assert_replay(blocked, "auto_mutation_unresolved_action", "changed-value contender")
        if calls[0] != 1 or publication_calls[0] != 1 or _task(runtime, task_id) != row:
            raise SystemExit("changed-value contender bypassed target-only uncertain fencing")
        _assert_receipt_privacy(
            runtime,
            private_values=[
                owner,
                contender,
                owner_body,
                owner_due,
                contender_body,
                contender_due,
                str(root),
            ],
        )
        _assert_no_absolute_path_metadata(runtime, [failed, same, blocked])


def test_completion_audit_failure_is_uncertain_and_fenced() -> None:
    owner = "PRIVATE-TASK-DETAILS-COMPLETION-OWNER"
    contender = "PRIVATE-TASK-DETAILS-COMPLETION-CONTENDER"
    with TemporaryDirectory(prefix="jarvis-task-details-completion-failure-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("completion failure original"))
        runtime.vault.sync_open_tasks(runtime.store)
        args = {"task_id": task_id, "body": "completion committed", "priority": "high"}
        runtime.planner = StaticPlanner(args)
        original_complete = runtime.store.complete_auto_mutation_receipt

        def fail_completion(**_kwargs: Any) -> int:
            raise RuntimeError("injected task-details completion/audit failure")

        runtime.store.complete_auto_mutation_receipt = fail_completion  # type: ignore[method-assign]
        failed = runtime.handle("fail task-details completion", request_token=owner)
        runtime.store.complete_auto_mutation_receipt = original_complete  # type: ignore[method-assign]
        _assert_replay(failed, "auto_mutation_completion_failed", "task-details completion failure")
        projection = (_projection_bytes(runtime) or b"").decode("utf-8")
        receipts = _receipt_rows(runtime)
        if (
            _task(runtime, task_id)["body"] != "completion committed"
            or f"- [ ] #{task_id} completion committed priority high" not in projection.splitlines()
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
            or any(row["ok"] == 1 for row in _tool_run_rows(runtime))
        ):
            raise SystemExit("task-details completion failure left false success evidence")

        same = runtime.handle("same-token completion replay", request_token=owner)
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token completion replay")
        runtime.planner = StaticPlanner(
            {"task_id": task_id, "body": "completion contender", "priority": "low"}
        )
        cross = runtime.handle("cross-token completion replay", request_token=contender)
        _assert_replay(cross, "auto_mutation_unresolved_action", "cross-token completion replay")
        if _task(runtime, task_id)["body"] != "completion committed" or len(_receipt_rows(runtime)) != 1:
            raise SystemExit("completion-failure replay changed task details")


def test_process_death_after_database_commit_is_recovered_and_fenced() -> None:
    with TemporaryDirectory(prefix="jarvis-task-details-process-death-") as temp:
        root = Path(temp)
        setup = _runtime(root, {})
        task_id = setup.store.add_task(TaskRecord("process death stale projection"))
        setup.vault.sync_open_tasks(setup.store)
        projection_before = _projection_bytes(setup)
        process = multiprocessing.get_context("spawn").Process(
            target=_crash_after_task_details_database_commit,
            args=(str(root), task_id),
        )
        process.start()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise SystemExit("task-details process-death fixture did not terminate")
        if process.exitcode != 73:
            raise SystemExit(f"task-details process-death fixture exited unexpectedly: {process.exitcode}")

        receipts = _receipt_rows(setup)
        if (
            _task(setup, task_id)["body"] != PROCESS_DEATH_BODY
            or _projection_bytes(setup) != projection_before
            or len(receipts) != 1
            or receipts[0]["state"] != "running"
        ):
            raise SystemExit("task-details process death lost database or running-receipt custody")
        with setup.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET running_at = ?, updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z", receipts[0]["id"]),
            )

        args = {
            "task_id": task_id,
            "body": PROCESS_DEATH_BODY,
            "due": "next month",
            "priority": "high",
        }
        recovered = _runtime(root, args)
        recovered_receipts = _receipt_rows(recovered)
        if (
            len(recovered_receipts) != 1
            or recovered_receipts[0]["state"] != "uncertain"
            or recovered_receipts[0]["result"] != "unknown"
            or recovered_receipts[0]["resolution"] != "stale_recovery"
        ):
            raise SystemExit("startup did not fence the stale task-details process death")
        same = recovered.handle(
            "same-token task-details process-death replay",
            request_token=PROCESS_DEATH_TOKEN,
        )
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token process death")
        recovered.planner = StaticPlanner(
            {"task_id": task_id, "body": "process death changed contender"}
        )
        cross_token = "PRIVATE-TASK-DETAILS-PROCESS-DEATH-CROSS"
        cross = recovered.handle(
            "cross-token task-details process-death replay",
            request_token=cross_token,
        )
        _assert_replay(cross, "auto_mutation_unresolved_action", "cross-token process death")
        if _task(recovered, task_id)["body"] != PROCESS_DEATH_BODY:
            raise SystemExit("process-death replay overwrote the committed task details")
        _assert_receipt_privacy(
            recovered,
            private_values=[PROCESS_DEATH_TOKEN, cross_token, PROCESS_DEATH_BODY, str(root)],
        )


def test_concurrent_same_target_is_fenced() -> None:
    with TemporaryDirectory(prefix="jarvis-task-details-concurrent-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("concurrent original"))
        runtime.vault.sync_open_tasks(runtime.store)
        owner_args = {"task_id": task_id, "body": "concurrent owner", "priority": "high"}
        runtime.planner = StaticPlanner(owner_args)
        tool = runtime.registry.get(TOOL_NAME)
        entered = threading.Event()
        release = threading.Event()
        calls = [0]

        def delayed(args: dict[str, Any]) -> ToolResult:
            calls[0] += 1
            entered.set()
            if not release.wait(10):
                raise TimeoutError("concurrent task-details fixture was not released")
            return tool.handler(args)

        runtime.registry._tools[TOOL_NAME] = replace(tool, handler=delayed)
        with ThreadPoolExecutor(max_workers=1) as pool:
            winner_future = pool.submit(
                runtime.handle,
                "concurrent task-details owner",
                request_token="task-details-concurrent-owner",
            )
            if not entered.wait(10):
                release.set()
                raise SystemExit("concurrent task-details owner did not enter the handler")
            runtime.planner = StaticPlanner(
                {"task_id": task_id, "body": "concurrent contender", "priority": "low"}
            )
            try:
                blocked = runtime.handle(
                    "concurrent task-details contender",
                    request_token="task-details-concurrent-contender",
                )
            finally:
                release.set()
            winner = winner_future.result(timeout=15)

        if not winner.tool_results[0].ok or calls[0] != 1:
            raise SystemExit(f"concurrent task-details owner did not complete once: {winner.tool_results}")
        _assert_replay(blocked, "auto_mutation_unresolved_action", "concurrent task-details contender")
        row = _task(runtime, task_id)
        if row["body"] != "concurrent owner" or row["priority"] != "high":
            raise SystemExit("concurrent same-target task-details contender changed the winner")
        _assert_completed_receipt_audits(runtime, 1)


def main() -> None:
    test_exact_contract_schema_and_target_only_identity()
    test_typed_and_semantic_refusals_precede_handler_and_receipt()
    test_refusal_retry_commands_use_deterministic_planner_grammar()
    test_combined_success_replay_fresh_convergence_audit_and_privacy()
    test_post_preflight_missing_task_releases_single_action_receipt()
    test_changed_values_are_fenced_after_database_before_projection_failure()
    test_completion_audit_failure_is_uncertain_and_fenced()
    test_process_death_after_database_commit_is_recovered_and_fenced()
    test_concurrent_same_target_is_fenced()
    print("Task details auto-mutation rollout smoke passed")


if __name__ == "__main__":
    main()
