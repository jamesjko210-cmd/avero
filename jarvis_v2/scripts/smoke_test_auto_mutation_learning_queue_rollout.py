from __future__ import annotations

from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    TOOL_ARGUMENT_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    ToolArgumentType,
)


TOOL_NAME = "queue_learning_tasks"
PROCESS_DEATH_TOKEN = "PRIVATE-LEARNING-QUEUE-PROCESS-DEATH-OWNER"
PRIVATE_SOURCE = "PRIVATE learning source 84917"
PRIVATE_CANDIDATE_SOURCE = "/Volumes/Private/PRIVATE-learning-candidate-84917/tool"


def _expected_candidates(failed_run_id: int) -> list[str]:
    return [
        "Review Jarvis feedback report and decide whether to add a preference, skill, or smoke test.",
        "Consider adding one explicit preference from repeated Jarvis feedback.",
        "Consider drafting one reviewed skill from repeated successful Jarvis workflows.",
        (
            f"Review after-action learning for failed run #{failed_run_id} &lt;local-path&gt; "
            "before promoting memories, skills, or tests."
        ),
    ]


class StaticPlanner:
    def __init__(self, args: Any):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise aggregate-wired learning-task queue custody.",
            [PlannedAction(TOOL_NAME, self.args, "learning queue rollout smoke")],
            needs_model=False,
        )


def _runtime(root: Path, args: Any) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(args)
    return runtime


def _seed_candidates(runtime: JarvisRuntime) -> int:
    runtime.store.add_memory(
        MemoryRecord(
            category="feedback",
            title="Queue rollout feedback",
            body=PRIVATE_SOURCE,
            source="learning-queue-rollout",
        )
    )
    return runtime.store.log_tool_run(
        "learning-queue-private-source",
        PRIVATE_CANDIDATE_SOURCE,
        "LOCAL_SAFE",
        False,
        False,
        "private candidate source failure",
    )


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


def _task_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM tasks ORDER BY id")


def _projection_bytes(runtime: JarvisRuntime) -> bytes | None:
    path = runtime.vault.root_path / "Tasks" / "Open Tasks.md"
    return path.read_bytes() if path.is_file() else None


def _projection_without_updated_timestamp(content: bytes | None) -> tuple[str, ...]:
    text = (content or b"").decode("utf-8")
    return tuple(line for line in text.splitlines() if not line.startswith("Updated: "))


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
    successful_runs = [
        row
        for row in _tool_run_rows(runtime)
        if row["tool_name"] == TOOL_NAME and row["ok"] == 1
    ]
    runs_by_id = {row["id"]: row for row in successful_runs}
    if len(receipts) != expected or len(successful_runs) != expected:
        raise SystemExit(
            f"expected {expected} learning-queue receipt/audit pairs: "
            f"{receipts} / {successful_runs}"
        )
    for receipt in receipts:
        run = runs_by_id.get(receipt["tool_run_id"])
        if (
            receipt["tool_name"] != TOOL_NAME
            or receipt["state"] != "completed"
            or receipt["result"] != "succeeded"
            or receipt["resolution"] != "recorded"
            or run is None
            or run["tool_name"] != TOOL_NAME
            or run["risk"] != "LOCAL_SAFE"
            or run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
        ):
            raise SystemExit(
                f"learning queue lost ordinary receipt/audit linkage: {receipt} / {run}"
            )


def _assert_receipt_and_audit_privacy(
    runtime: JarvisRuntime,
    *,
    private_values: list[str],
) -> None:
    receipts = _receipt_rows(runtime)
    audit_rows = [row for row in _tool_run_rows(runtime) if row["tool_name"] == TOOL_NAME]
    receipt_text = json.dumps(receipts, ensure_ascii=False, sort_keys=True, default=str)
    audit_text = json.dumps(audit_rows, ensure_ascii=False, sort_keys=True, default=str)
    for private in private_values:
        if private and (private in receipt_text or private in audit_text):
            raise SystemExit(f"learning queue receipt/audit exposed private source: {private!r}")
    for receipt in receipts:
        for key in ("request_digest", "action_digest", "operation_digest"):
            value = receipt.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise SystemExit(f"learning queue receipt lost a bounded digest: {receipt}")
    absolute_roots = (str(runtime.vault.root_path), str(runtime.store.db_path.parent))
    if any(root in receipt_text or root in audit_text for root in absolute_roots):
        raise SystemExit("learning queue receipt/audit exposed an absolute temporary path")


def _crash_after_learning_queue_database_commit(root_text: str) -> None:
    runtime = _runtime(Path(root_text), {"limit": 1})
    expected = _expected_candidates(_seed_candidates(runtime))

    def exit_before_projection(*_args: Any, **_kwargs: Any) -> Any:
        if [row["body"] for row in _task_rows(runtime)] == expected:
            os._exit(73)
        os._exit(74)

    _install_projection_failure(runtime, exit_before_projection)
    runtime.handle(
        "queue learning tasks before representative process death",
        request_token=PROCESS_DEATH_TOKEN,
    )
    os._exit(75)


def test_exact_contract_schema_and_global_target_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-learning-queue-contract-") as temp:
        runtime = _runtime(Path(temp), {})
        tool = runtime.registry.get(TOOL_NAME)
        contract = tool.auto_mutation_contract
        if (
            tool.toolset != "learning"
            or contract is None
            or contract.version != AUTO_MUTATION_CONTRACT_VERSION
            or contract.effects
            != frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT})
            or contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
            or contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
            or contract.operation_key_builder is None
            or contract.semantic_preflight is not None
            or contract.semantic_preflight_result_builder is not None
            or contract.definite_no_effect_failure_reasons != frozenset()
        ):
            raise SystemExit(f"{TOOL_NAME} auto-mutation contract drifted: {tool}")

        arguments = tool.argument_contract
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
            or shape != (("limit", integer, False),)
        ):
            raise SystemExit(f"{TOOL_NAME} strict argument contract drifted: {arguments}")

        builder = contract.operation_key_builder
        keys = [builder({}), builder({"limit": 1}), builder({"limit": 200})]
        if keys != [{"target": "learning_task_queue"}] * 3:
            raise SystemExit(f"learning queue operation identity is not fixed and global: {keys}")


def test_typed_invalid_inputs_precede_receipts_and_writes() -> None:
    invalid_args: list[Any] = [
        {"limit": None},
        {"limit": "12"},
        {"limit": True},
        {"limit": False},
        {"limit": 1.5},
        {"limit": []},
        {"limit": 1, "unknown": PRIVATE_SOURCE},
        None,
        [],
    ]
    for index, args in enumerate(invalid_args):
        with TemporaryDirectory(prefix="jarvis-learning-queue-typed-") as temp:
            runtime = _runtime(Path(temp), args)
            _seed_candidates(runtime)
            projection_before = _projection_bytes(runtime)
            calls = _install_counted_handler(runtime)
            result = runtime.handle(
                "reject invalid learning queue arguments",
                request_token=f"learning-queue-typed-{index}",
            )
            item = result.tool_results[0]
            if (
                item.ok
                or item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
                or _task_rows(runtime)
                or _receipt_rows(runtime)
                or _projection_bytes(runtime) != projection_before
            ):
                raise SystemExit(f"learning queue typed case {index} crossed a boundary: {item}")


def test_multi_candidate_database_failure_is_all_or_none() -> None:
    with TemporaryDirectory(prefix="jarvis-learning-queue-atomic-") as temp:
        runtime = _runtime(Path(temp), {"limit": 1})
        expected = _expected_candidates(_seed_candidates(runtime))
        projection_before = _projection_bytes(runtime)
        failing_body = expected[1].replace("'", "''")
        with runtime.store.connect() as conn:
            conn.execute(
                f"""
                CREATE TRIGGER learning_queue_atomic_failure
                BEFORE INSERT ON tasks
                WHEN NEW.body = '{failing_body}'
                BEGIN
                    SELECT RAISE(ABORT, 'injected learning queue batch failure');
                END
                """
            )
        result = runtime.handle(
            "inject learning queue SQL failure",
            request_token="learning-queue-atomic-owner",
        )
        item = result.tool_results[0]
        receipts = _receipt_rows(runtime)
        if (
            item.ok
            or item.metadata.get("handler_invoked") is not True
            or _task_rows(runtime)
            or _projection_bytes(runtime) != projection_before
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
            or any(row["ok"] == 1 for row in _tool_run_rows(runtime))
        ):
            raise SystemExit(f"learning queue SQL failure was not atomic: {item} / {receipts}")


def test_success_replay_fresh_convergence_receipt_audit_and_privacy() -> None:
    owner = "PRIVATE-LEARNING-QUEUE-SUCCESS-OWNER"
    fresh_token = "PRIVATE-LEARNING-QUEUE-SUCCESS-FRESH"
    with TemporaryDirectory(prefix="jarvis-learning-queue-success-") as temp:
        runtime = _runtime(Path(temp), {"limit": 1})
        expected = _expected_candidates(_seed_candidates(runtime))
        calls = _install_counted_handler(runtime)
        first = runtime.handle("queue deterministic learning batch", request_token=owner)
        item = first.tool_results[0]
        rows = _task_rows(runtime)
        projection = _projection_bytes(runtime)
        projection_text = (projection or b"").decode("utf-8")
        if (
            not item.ok
            or calls[0] != 1
            or item.metadata.get("added") != len(expected)
            or item.metadata.get("skipped") != 0
            or [row["body"] for row in rows] != expected
            or any(row["status"] != "open" for row in rows)
            or any(body not in projection_text for body in expected)
        ):
            raise SystemExit(f"successful learning queue batch drifted: {item} / {rows}")
        returned_text = f"{item.output}\n{json.dumps(item.metadata, ensure_ascii=False, default=str)}"
        for private in (
            PRIVATE_SOURCE,
            PRIVATE_CANDIDATE_SOURCE,
            str(runtime.vault.root_path),
            str(runtime.store.db_path.parent),
        ):
            if private in returned_text:
                raise SystemExit(
                    f"learning queue result exposed private content or an absolute path: {private!r}"
                )
        _assert_completed_receipt_audits(runtime, 1)

        same = runtime.handle("same-token learning queue replay", request_token=owner)
        _assert_replay(same, "auto_mutation_completed_replay", "same-token learning queue")
        if calls[0] != 1 or _task_rows(runtime) != rows:
            raise SystemExit("same-token learning queue replay reran the handler")

        runtime.planner = StaticPlanner({"limit": 200})
        fresh = runtime.handle("fresh-token learning queue convergence", request_token=fresh_token)
        fresh_item = fresh.tool_results[0]
        if (
            not fresh_item.ok
            or fresh_item.metadata.get("added") != 0
            or fresh_item.metadata.get("skipped") != len(expected)
            or calls[0] != 2
            or _task_rows(runtime) != rows
            or _projection_without_updated_timestamp(_projection_bytes(runtime))
            != _projection_without_updated_timestamp(projection)
        ):
            raise SystemExit(f"fresh-token learning queue did not converge: {fresh_item}")
        _assert_completed_receipt_audits(runtime, 2)
        _assert_receipt_and_audit_privacy(
            runtime,
            private_values=[
                owner,
                fresh_token,
                PRIVATE_SOURCE,
                PRIVATE_CANDIDATE_SOURCE,
                str(Path(temp)),
            ],
        )


def test_projection_failure_is_uncertain_and_globally_fenced() -> None:
    owner = "PRIVATE-LEARNING-QUEUE-PROJECTION-OWNER"
    cross_token = "PRIVATE-LEARNING-QUEUE-PROJECTION-CROSS"
    with TemporaryDirectory(prefix="jarvis-learning-queue-projection-") as temp:
        runtime = _runtime(Path(temp), {"limit": 1})
        expected = _expected_candidates(_seed_candidates(runtime))
        projection_before = _projection_bytes(runtime)
        calls = _install_counted_handler(runtime)
        publications = [0]

        def fail_projection(*_args: Any, **_kwargs: Any) -> Any:
            publications[0] += 1
            raise OSError("injected learning queue projection failure")

        _install_projection_failure(runtime, fail_projection)
        failed = runtime.handle("fail learning queue projection", request_token=owner)
        item = failed.tool_results[0]
        receipts = _receipt_rows(runtime)
        if (
            item.ok
            or item.metadata.get("auto_mutation_outcome_uncertain") is not True
            or calls[0] != 1
            or publications[0] != 1
            or [row["body"] for row in _task_rows(runtime)] != expected
            or _projection_bytes(runtime) != projection_before
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
        ):
            raise SystemExit(f"learning queue projection failure lost custody: {item}")

        same = runtime.handle("same-token exact projection replay", request_token=owner)
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token exact queue replay")
        runtime.planner = StaticPlanner({"limit": 200})
        same_changed = runtime.handle("same-token changed-limit replay", request_token=owner)
        cross_changed = runtime.handle("cross-token changed-limit replay", request_token=cross_token)
        _assert_replay(
            same_changed,
            "auto_mutation_request_collision",
            "same-token changed-limit queue replay",
        )
        _assert_replay(
            cross_changed,
            "auto_mutation_unresolved_action",
            "cross-token changed-limit queue replay",
        )
        if (
            calls[0] != 1
            or publications[0] != 1
            or len(_task_rows(runtime)) != len(expected)
        ):
            raise SystemExit("learning queue uncertainty replay reran the mutation")
        _assert_receipt_and_audit_privacy(
            runtime,
            private_values=[
                owner,
                cross_token,
                PRIVATE_SOURCE,
                PRIVATE_CANDIDATE_SOURCE,
                str(Path(temp)),
            ],
        )


def test_process_death_after_database_commit_is_recovered_and_fenced() -> None:
    cross_token = "PRIVATE-LEARNING-QUEUE-PROCESS-DEATH-CROSS"
    with TemporaryDirectory(prefix="jarvis-learning-queue-process-death-") as temp:
        root = Path(temp)
        process = multiprocessing.get_context("spawn").Process(
            target=_crash_after_learning_queue_database_commit,
            args=(str(root),),
        )
        process.start()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise SystemExit("learning queue process-death fixture did not terminate")
        if process.exitcode != 73:
            raise SystemExit(f"learning queue process-death fixture exited unexpectedly: {process.exitcode}")

        runtime = _runtime(root, {"limit": 1})
        receipts = _receipt_rows(runtime)
        source_runs = [
            row
            for row in _tool_run_rows(runtime)
            if row["session_id"] == "learning-queue-private-source"
            and row["tool_name"] == PRIVATE_CANDIDATE_SOURCE
            and row["ok"] == 0
        ]
        if len(source_runs) != 1:
            raise SystemExit(
                f"process-death fixture lost its unique seeded failed run: {source_runs}"
            )
        expected = _expected_candidates(int(source_runs[0]["id"]))
        if (
            [row["body"] for row in _task_rows(runtime)] != expected
            or _projection_bytes(runtime) is not None
            or len(receipts) != 1
            or receipts[0]["state"] != "running"
        ):
            raise SystemExit("learning queue process death lost database or running receipt custody")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET running_at = ?, updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z", receipts[0]["id"]),
            )

        recovered = _runtime(root, {"limit": 200})
        recovered_receipts = _receipt_rows(recovered)
        if (
            len(recovered_receipts) != 1
            or recovered_receipts[0]["state"] != "uncertain"
            or recovered_receipts[0]["result"] != "unknown"
            or recovered_receipts[0]["resolution"] != "stale_recovery"
        ):
            raise SystemExit("startup did not fence stale learning queue process death")
        recovered.planner = StaticPlanner({"limit": 1})
        same = recovered.handle(
            "same-token learning queue process-death replay",
            request_token=PROCESS_DEATH_TOKEN,
        )
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token process death")
        recovered.planner = StaticPlanner({"limit": 200})
        cross = recovered.handle(
            "cross-token changed-limit process-death replay",
            request_token=cross_token,
        )
        _assert_replay(cross, "auto_mutation_unresolved_action", "cross-token process death")
        if (
            len(_task_rows(recovered)) != len(expected)
            or _projection_bytes(recovered) is not None
        ):
            raise SystemExit("process-death replay duplicated tasks or published an unowned projection")
        _assert_receipt_and_audit_privacy(
            recovered,
            private_values=[
                PROCESS_DEATH_TOKEN,
                cross_token,
                PRIVATE_SOURCE,
                PRIVATE_CANDIDATE_SOURCE,
                str(root),
            ],
        )


def main() -> None:
    test_exact_contract_schema_and_global_target_identity()
    test_typed_invalid_inputs_precede_receipts_and_writes()
    test_multi_candidate_database_failure_is_all_or_none()
    test_success_replay_fresh_convergence_receipt_audit_and_privacy()
    test_projection_failure_is_uncertain_and_globally_fenced()
    test_process_death_after_database_commit_is_recovered_and_fenced()
    print("Learning queue auto-mutation rollout smoke passed")


if __name__ == "__main__":
    main()
