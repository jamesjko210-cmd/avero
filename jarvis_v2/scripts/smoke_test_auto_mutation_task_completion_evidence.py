from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from tempfile import TemporaryDirectory
from typing import Any, Callable

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.memory.store import MemoryStore, TaskRecord
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


TOOL_NAME = "complete_task_with_evidence"
TASK_STATUS_TOOLS = (
    "complete_task",
    TOOL_NAME,
    "update_task_status",
)
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
    def __init__(self, tool_name: str, args: Any):
        self.tool_name = tool_name
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise evidence-backed task completion custody.",
            [
                PlannedAction(
                    self.tool_name,
                    self.args,
                    "task completion evidence smoke",
                )
            ],
            needs_model=False,
        )


def _runtime(root: Path, args: Any) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(TOOL_NAME, args)
    return runtime


def _set_action(
    runtime: JarvisRuntime,
    tool_name: str,
    args: Any,
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


def _task(runtime: JarvisRuntime, task_id: int) -> dict[str, Any]:
    row = runtime.store.get_task(task_id)
    if row is None:
        raise SystemExit(f"task-completion fixture lost task #{task_id}")
    return dict(row)


def _evidence(runtime: JarvisRuntime, task_id: int) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in runtime.store.list_task_completion_evidence(task_id, limit=100)
    ]


def _projection_bytes(runtime: JarvisRuntime) -> bytes | None:
    path = runtime.vault.root_path / "Tasks" / "Open Tasks.md"
    return path.read_bytes() if path.is_file() else None


def _install_counted_handler(
    runtime: JarvisRuntime,
    tool_name: str = TOOL_NAME,
) -> list[int]:
    tool = runtime.registry.get(tool_name)
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools[tool_name] = replace(tool, handler=counted)
    return calls


def _install_projection_failure(
    runtime: JarvisRuntime,
    callback: Callable[..., Any],
) -> None:
    runtime.vault.sync_open_tasks = callback  # type: ignore[method-assign]
    runtime.vault.sync_open_tasks_with_evidence = callback  # type: ignore[method-assign]


def _assert_failure_kind(result: Any, expected: str, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} returned an unexpected result count: {result.tool_results}")
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != expected:
        raise SystemExit(f"{label} returned the wrong failure state: {item}")


def _assert_no_mutation(
    runtime: JarvisRuntime,
    *,
    task_id: int,
    task_before: dict[str, Any],
    projection_before: bytes | None,
    label: str,
) -> None:
    if _task(runtime, task_id) != task_before:
        raise SystemExit(f"{label} changed the task row")
    if _evidence(runtime, task_id):
        raise SystemExit(f"{label} persisted completion evidence")
    if _projection_bytes(runtime) != projection_before:
        raise SystemExit(f"{label} changed the open-task projection")
    if _receipt_rows(runtime):
        raise SystemExit(f"{label} created an auto-mutation receipt")
    if any(row["ok"] == 1 or row["approved"] == 1 for row in _tool_run_rows(runtime)):
        raise SystemExit(f"{label} created success or approval evidence")


def _assert_completed_receipt_audit(runtime: JarvisRuntime) -> None:
    receipts = _receipt_rows(runtime)
    successful_runs = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
    if len(receipts) != 1 or len(successful_runs) != 1:
        raise SystemExit(
            "expected one evidence-completion receipt/audit pair: "
            f"{receipts} / {successful_runs}"
        )
    receipt = receipts[0]
    run = successful_runs[0]
    if (
        receipt["tool_name"] != TOOL_NAME
        or receipt["operation_scope"] != "task_status"
        or receipt["state"] != "completed"
        or receipt["result"] != "succeeded"
        or receipt["resolution"] != "recorded"
        or receipt["tool_run_id"] != run["id"]
        or run["tool_name"] != TOOL_NAME
        or run["approved"] != 0
        or run["approval_id"] is not None
        or run["approval_action_digest"] is not None
    ):
        raise SystemExit(f"evidence completion lost ordinary audit linkage: {receipt} / {run}")


def _assert_uncertain_receipt(runtime: JarvisRuntime, label: str) -> None:
    receipts = _receipt_rows(runtime)
    if (
        len(receipts) != 1
        or receipts[0]["tool_name"] != TOOL_NAME
        or receipts[0]["operation_scope"] != "task_status"
        or receipts[0]["state"] != "uncertain"
        or receipts[0]["result"] != "unknown"
        or receipts[0]["resolution"] != "manual_review"
        or receipts[0]["tool_run_id"] is not None
    ):
        raise SystemExit(f"{label} lost uncertain receipt custody: {receipts}")


def _assert_uncertain_privacy(
    runtime: JarvisRuntime,
    results: list[Any],
    private_values: list[str],
) -> None:
    receipts = _receipt_rows(runtime)
    receipt_json = json.dumps(receipts, ensure_ascii=False, sort_keys=True, default=str)
    public = {
        "runtime_results": [
            {
                "response": result.response,
                "metadata": result.metadata,
                "tool_results": [
                    {
                        "output": item.output,
                        "metadata": item.metadata,
                    }
                    for item in result.tool_results
                ],
            }
            for result in results
        ],
        "tool_runs": [
            {
                "output": row.get("output"),
                "metadata": row.get("metadata"),
            }
            for row in _tool_run_rows(runtime)
        ],
        "messages": _rows(
            runtime,
            "SELECT role, content, metadata FROM messages ORDER BY id",
        ),
    }
    public_json = json.dumps(public, ensure_ascii=False, sort_keys=True, default=str)
    for private in private_values:
        if not private:
            continue
        if private in receipt_json:
            raise SystemExit(f"uncertain receipt exposed private evidence: {private!r}")
        if private in public_json:
            raise SystemExit(f"public uncertain result exposed private evidence: {private!r}")
    for receipt in receipts:
        for key in (
            "request_digest",
            "action_digest",
            "operation_digest",
            "uncertainty_digest",
            "run_token",
        ):
            value = receipt.get(key)
            if isinstance(value, str) and value and value in public_json:
                raise SystemExit(f"public uncertain result exposed private receipt field {key}")


def _seed_mock_recovery_blocker(runtime: JarvisRuntime) -> int:
    return runtime.store.log_tool_run(
        runtime.session_id,
        "mock_external_action",
        "HIGH_RISK",
        False,
        False,
        "Mock action held for approval; no action executed.",
        metadata={
            "failure_kind": "approval_required",
            "requires_confirmation": True,
            "executed_handler": False,
            "handler_invoked": False,
        },
    )


def _log_mock_recovery_proofs(
    runtime: JarvisRuntime,
    target_run_id: int,
    *,
    ok: bool,
) -> list[int]:
    return [
        runtime.store.log_tool_run(
            runtime.session_id,
            tool_name,
            "READ_ONLY",
            ok,
            False,
            "mock recovery proof",
            metadata={"run_id": target_run_id},
        )
        for tool_name in (
            "verification_receipt",
            "execution_recovery_packet",
            "after_action_learning_packet",
        )
    ]


def test_exact_schema_contract_and_shared_task_status_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-task-evidence-contract-") as temp:
        runtime = _runtime(Path(temp), {"task_id": 17, "evidence": "contract"})
        tool = runtime.registry.get(TOOL_NAME)
        contract = tool.auto_mutation_contract
        aliases = frozenset(TASK_STATUS_TOOLS)
        if (
            tool.toolset != "tasks"
            or tool.risk.name != "LOCAL_SAFE"
            or contract is None
            or contract.version != AUTO_MUTATION_CONTRACT_VERSION
            or contract.effects
            != frozenset(
                {
                    AutoMutationEffect.LOCAL_DATABASE,
                    AutoMutationEffect.OBSIDIAN_VAULT,
                }
            )
            or contract.replay_policy
            is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
            or contract.crash_policy
            is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
            or contract.operation_key_builder
            is not task_tools.task_status_auto_mutation_operation_key
            or contract.semantic_preflight is None
            or contract.semantic_preflight_result_builder
            is not task_tools.complete_task_with_evidence_auto_mutation_preflight_result
            or contract.definite_no_effect_failure_reasons
            != frozenset({"missing_task", "recovery_state_changed"})
            or contract.operation_scope != "task_status"
            or contract.legacy_operation_aliases != aliases
        ):
            raise SystemExit(f"{TOOL_NAME} exact auto-mutation contract drifted: {tool}")

        string = frozenset({ToolArgumentType.STRING})
        integer = frozenset({ToolArgumentType.INTEGER})
        arguments = tool.argument_contract
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
                ("evidence", string, False),
                ("verification_run_id", string, False),
                ("task_id", integer, True),
            )
        ):
            raise SystemExit(f"{TOOL_NAME} strict argument contract drifted: {arguments}")

        for status_tool in TASK_STATUS_TOOLS:
            status_contract = runtime.registry.get(status_tool).auto_mutation_contract
            if (
                status_contract is None
                or status_contract.operation_scope != "task_status"
                or status_contract.legacy_operation_aliases != aliases
                or status_contract.operation_key_builder
                is not task_tools.task_status_auto_mutation_operation_key
            ):
                raise SystemExit(f"{status_tool} lost shared task-status custody: {status_contract}")

        builder = contract.operation_key_builder
        equivalent = [
            builder({"task_id": 17, "evidence": "first"}),
            builder({"task_id": 17, "evidence": "changed"}),
            builder({"task_id": 17, "verification_run_id": "run-17"}),
        ]
        if any(item != {"task_id": 17} for item in equivalent):
            raise SystemExit(f"evidence completion identity retained private evidence: {equivalent}")
        if builder({"task_id": 18, "evidence": "first"}) == equivalent[0]:
            raise SystemExit("evidence completion identity did not distinguish task targets")

        columns = [
            row["name"]
            for row in _rows(runtime, "PRAGMA table_info(task_completion_evidence)")
        ]
        if columns != [
            "id",
            "task_id",
            "task_body",
            "task_body_digest",
            "evidence",
            "verification_run_id",
            "evidence_digest",
            "created_at",
        ]:
            raise SystemExit(f"task completion evidence schema drifted: {columns}")
        indexes = _rows(runtime, "PRAGMA index_list(task_completion_evidence)")
        if (
            any(row["unique"] for row in indexes)
            or not any(
                row["name"] == "task_completion_evidence_task_created_idx"
                for row in indexes
            )
        ):
            raise SystemExit(f"task completion evidence indexes drifted: {indexes}")
        triggers = {
            row["name"]
            for row in _rows(
                runtime,
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND tbl_name = 'task_completion_evidence'",
            )
        }
        if triggers != {
            "task_completion_evidence_immutable_update",
            "task_completion_evidence_immutable_delete",
        }:
            raise SystemExit(f"task completion evidence guards drifted: {triggers}")


def test_invalid_and_unknown_arguments_precede_handler() -> None:
    cases: list[Any] = [
        {},
        {"task_id": "1", "evidence": "typed"},
        {"task_id": True, "evidence": "typed"},
        {"task_id": 1, "evidence": None},
        {"task_id": 1, "verification_run_id": 7},
        {"task_id": 1, "proof": "legacy alias must be rejected"},
        {"task_id": 1, "run_id": "legacy alias must be rejected"},
        {"task_id": 1, "evidence": "typed", "unknown": "/private/typed-secret"},
        None,
        [],
    ]
    for index, args in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-task-evidence-typed-") as temp:
            runtime = _runtime(Path(temp), args)
            task_id = runtime.store.add_task(TaskRecord("typed refusal remains open"))
            runtime.vault.sync_open_tasks(runtime.store)
            before = _task(runtime, task_id)
            projection_before = _projection_bytes(runtime)
            calls = _install_counted_handler(runtime)
            result = runtime.handle(
                "reject invalid evidence completion arguments",
                request_token=f"task-evidence-typed-{index}",
            )
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
            ):
                raise SystemExit(f"typed evidence case {index} reached the handler: {item}")
            _assert_no_mutation(
                runtime,
                task_id=task_id,
                task_before=before,
                projection_before=projection_before,
                label=f"typed evidence case {index}",
            )


def test_semantic_preflights_are_zero_write() -> None:
    cases = (
        ("bad_task_id", lambda _task_id: {"task_id": 0, "evidence": "proof"}, False),
        (
            "bad_task_id",
            lambda _task_id: {"task_id": 9223372036854775808, "evidence": "proof"},
            False,
        ),
        ("missing_evidence", lambda task_id: {"task_id": task_id}, False),
        (
            "evidence_too_long",
            lambda task_id: {"task_id": task_id, "evidence": "x" * 801},
            False,
        ),
        (
            "verification_run_id_too_long",
            lambda task_id: {"task_id": task_id, "verification_run_id": "r" * 81},
            False,
        ),
        (
            "invalid_evidence",
            lambda task_id: {"task_id": task_id, "evidence": "\u200b\u2060"},
            False,
        ),
        (
            "invalid_evidence",
            lambda task_id: {"task_id": task_id, "evidence": "visible\ud800"},
            False,
        ),
        (
            "invalid_evidence",
            lambda task_id: {"task_id": task_id, "evidence": "visible\u200bproof"},
            False,
        ),
        (
            "invalid_evidence",
            lambda task_id: {"task_id": task_id, "evidence": "visible\u2028proof"},
            False,
        ),
        (
            "invalid_evidence",
            lambda task_id: {"task_id": task_id, "evidence": "visible\x00proof"},
            False,
        ),
        (
            "invalid_evidence",
            lambda task_id: {
                "task_id": task_id,
                "verification_run_id": "run\u2060id",
            },
            False,
        ),
        (
            "missing_task",
            lambda _task_id: {"task_id": 999999, "evidence": "proof"},
            False,
        ),
        (
            "recovery_closure_required",
            lambda task_id: {"task_id": task_id, "evidence": "mock proof"},
            True,
        ),
    )
    for index, (reason, make_args, seed_recovery) in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-task-evidence-semantic-") as temp:
            runtime = _runtime(Path(temp), {})
            task_id = runtime.store.add_task(TaskRecord("semantic refusal remains open"))
            runtime.vault.sync_open_tasks(runtime.store)
            if seed_recovery:
                _seed_mock_recovery_blocker(runtime)
            _set_action(runtime, TOOL_NAME, make_args(task_id))
            before = _task(runtime, task_id)
            projection_before = _projection_bytes(runtime)
            calls = _install_counted_handler(runtime)
            result = runtime.handle(
                "reject semantic evidence completion",
                request_token=f"task-evidence-semantic-{index}",
            )
            item = result.tool_results[0]
            if (
                item.ok
                or item.metadata.get("failure_kind")
                != "auto_mutation_semantic_preflight_rejected"
                or item.metadata.get("reason") != reason
                or any(item.metadata.get(flag) is not False for flag in ZERO_WRITE_FLAGS)
                or calls[0] != 0
            ):
                raise SystemExit(f"semantic evidence case {reason} crossed preflight: {item}")
            _assert_no_mutation(
                runtime,
                task_id=task_id,
                task_before=before,
                projection_before=projection_before,
                label=f"semantic evidence case {reason}",
            )


def test_store_completion_rolls_back_evidence_if_status_update_fails() -> None:
    with TemporaryDirectory(prefix="jarvis-task-evidence-atomic-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("atomic rollback remains open"))
        with runtime.store.connect() as conn:
            conn.execute(
                f"""
                CREATE TRIGGER reject_mock_task_completion
                BEFORE UPDATE OF status ON tasks
                WHEN OLD.id = NEW.id AND NEW.id = {task_id} AND NEW.status = 'done'
                BEGIN
                    SELECT RAISE(ABORT, 'injected status update failure');
                END
                """
            )
        try:
            runtime.store.complete_task_with_evidence(
                task_id,
                evidence="must roll back with status",
                verification_run_id="atomic-run",
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise SystemExit("injected status update failure did not abort evidence completion")
        if _task(runtime, task_id)["status"] != "open" or _evidence(runtime, task_id):
            raise SystemExit("task status and evidence were not committed atomically")

    with TemporaryDirectory(prefix="jarvis-task-evidence-ignored-update-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("ignored completion remains open"))
        with runtime.store.connect() as conn:
            conn.execute(
                f"""
                CREATE TRIGGER ignore_mock_task_completion
                BEFORE UPDATE OF status ON tasks
                WHEN OLD.id = NEW.id AND NEW.id = {task_id} AND NEW.status = 'done'
                BEGIN
                    SELECT RAISE(IGNORE);
                END
                """
            )
        try:
            runtime.store.complete_task_with_evidence(
                task_id,
                evidence="must roll back when the update is suppressed",
                verification_run_id="ignored-update-run",
            )
        except RuntimeError as exc:
            if str(exc) != "task completion update was not applied":
                raise
        else:
            raise SystemExit("suppressed status update reported successful completion")
        if _task(runtime, task_id)["status"] != "open" or _evidence(runtime, task_id):
            raise SystemExit("suppressed task update committed completion evidence")


def test_store_rejects_mixed_invalid_unicode_without_writes() -> None:
    variants = (
        {"evidence": "visible\u200bproof"},
        {"evidence": "visible\u2028proof"},
        {"evidence": "visible\x00proof"},
        {"verification_run_id": "run\u2060id"},
        {"evidence": "visible\ud800proof"},
    )
    with TemporaryDirectory(prefix="jarvis-task-evidence-invalid-unicode-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("invalid unicode remains open"))
        for values in variants:
            try:
                runtime.store.complete_task_with_evidence(task_id, **values)
            except ValueError:
                pass
            else:
                raise SystemExit(f"store accepted invalid task evidence: {values!r}")
            if _task(runtime, task_id)["status"] != "open" or _evidence(runtime, task_id):
                raise SystemExit(f"invalid task evidence crossed the store boundary: {values!r}")


def test_store_preserves_exact_proof_and_records_each_completion_event() -> None:
    evidence = "  focused assertion\npassed exactly  "
    run_id = "  run-id-42  "
    with TemporaryDirectory(prefix="jarvis-task-evidence-exact-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("exact evidence event target"))
        first = runtime.store.complete_task_with_evidence(
            task_id,
            evidence=evidence,
            verification_run_id=run_id,
        )
        runtime.store.set_task_status(task_id, "open")
        second = runtime.store.complete_task_with_evidence(
            task_id,
            evidence=evidence,
            verification_run_id=run_id,
        )
        rows = _evidence(runtime, task_id)
        if (
            first is None
            or second is None
            or second["status"] != "done"
            or len(rows) != 2
            or any(row["evidence"] != evidence for row in rows)
            or any(row["verification_run_id"] != run_id for row in rows)
            or any(row["task_body"] != "exact evidence event target" for row in rows)
            or any(len(row["task_body_digest"]) != 64 for row in rows)
            or rows[0]["id"] == rows[1]["id"]
        ):
            raise SystemExit(f"exact or repeated completion evidence was lost: {rows}")
        runtime.store.update_task_fields(task_id, body="renamed after completion")
        if any(
            row["task_body"] != "exact evidence event target"
            for row in _evidence(runtime, task_id)
        ):
            raise SystemExit("completion evidence was rebound to a later task body")

        invalid = (
            ("x" * 801, ""),
            ("", "r" * 81),
            ("\u200b", ""),
            ("valid", "\u2060"),
        )
        for bad_evidence, bad_run_id in invalid:
            try:
                runtime.store.complete_task_with_evidence(
                    task_id,
                    evidence=bad_evidence,
                    verification_run_id=bad_run_id,
                )
            except ValueError:
                pass
            else:
                raise SystemExit("store accepted malformed completion evidence")
        try:
            runtime.store.list_task_completion_evidence(9223372036854775808)
        except ValueError:
            pass
        else:
            raise SystemExit("evidence read accepted an out-of-range SQLite task id")


def test_evidence_ledger_is_immutable_and_verifies_digest_on_read() -> None:
    with TemporaryDirectory(prefix="jarvis-task-evidence-integrity-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("immutable evidence target"))
        runtime.store.complete_task_with_evidence(task_id, evidence="trusted proof")
        for statement in (
            "UPDATE task_completion_evidence SET evidence = 'changed'",
            "DELETE FROM task_completion_evidence",
        ):
            try:
                with runtime.store.connect() as conn:
                    conn.execute(statement)
            except sqlite3.IntegrityError:
                pass
            else:
                raise SystemExit(f"evidence ledger allowed mutation: {statement}")

        with runtime.store.connect() as conn:
            conn.execute("DROP TRIGGER task_completion_evidence_immutable_update")
            conn.execute(
                "UPDATE task_completion_evidence SET evidence_digest = ?",
                ("0" * 64,),
            )
        try:
            runtime.store.list_task_completion_evidence(task_id)
        except RuntimeError as exc:
            if "integrity" not in str(exc):
                raise
        else:
            raise SystemExit("evidence read accepted a mismatched digest")


def test_incompatible_existing_evidence_schema_fails_closed_atomically() -> None:
    with TemporaryDirectory(prefix="jarvis-task-evidence-schema-") as temp:
        db_path = Path(temp) / "memory.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE task_completion_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL UNIQUE,
                    evidence TEXT NOT NULL DEFAULT '',
                    verification_run_id TEXT NOT NULL DEFAULT '',
                    evidence_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        store = MemoryStore(db_path)
        try:
            store.init()
        except RuntimeError as exc:
            if "task completion evidence" not in str(exc):
                raise
        else:
            raise SystemExit("incompatible evidence ledger did not fail closed")
        with sqlite3.connect(db_path) as conn:
            created = conn.execute(
                "SELECT name FROM sqlite_master WHERE name IN ("
                "'task_completion_evidence_task_created_idx', "
                "'task_completion_evidence_immutable_update', "
                "'task_completion_evidence_immutable_delete')"
            ).fetchall()
        if created:
            raise SystemExit(f"failed evidence migration left partial schema objects: {created}")


def test_legacy_unique_evidence_schema_migrates_without_losing_proof() -> None:
    evidence = "legacy verified proof"
    run_id = "legacy-run-7"
    digest = hashlib.sha256(
        json.dumps(
            {"evidence": evidence, "verification_run_id": run_id},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with TemporaryDirectory(prefix="jarvis-task-evidence-legacy-") as temp:
        db_path = Path(temp) / "memory.db"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    body TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    due TEXT NOT NULL DEFAULT '',
                    priority TEXT NOT NULL DEFAULT 'normal',
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE task_completion_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE RESTRICT,
                    evidence TEXT NOT NULL DEFAULT '',
                    verification_run_id TEXT NOT NULL DEFAULT '',
                    evidence_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, evidence_digest)
                );
                CREATE INDEX task_completion_evidence_task_created_idx
                ON task_completion_evidence(task_id, created_at, id);
                """
            )
            conn.execute(
                "INSERT INTO tasks(id, body, status, created_at, updated_at) "
                "VALUES (1, 'legacy task target', 'done', '2026-01-01T00:00:00Z', "
                "'2026-01-01T00:00:00Z')"
            )
            conn.execute(
                "INSERT INTO task_completion_evidence("
                "id, task_id, evidence, verification_run_id, evidence_digest, created_at"
                ") VALUES (1, 1, ?, ?, ?, '2026-01-01T00:00:00Z')",
                (evidence, run_id, digest),
            )
        store = MemoryStore(db_path)
        store.init()
        rows = [dict(row) for row in store.list_task_completion_evidence(1)]
        if (
            len(rows) != 1
            or rows[0]["evidence"] != evidence
            or rows[0]["verification_run_id"] != run_id
            or rows[0]["task_body"] != "legacy task target"
            or len(rows[0]["task_body_digest"]) != 64
        ):
            raise SystemExit(f"legacy completion evidence migration lost proof: {rows}")


def test_old_recovery_failure_cannot_age_out_or_race_completion() -> None:
    with TemporaryDirectory(prefix="jarvis-task-evidence-recovery-age-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("aged recovery target"))
        _seed_mock_recovery_blocker(runtime)
        for index in range(30):
            runtime.store.log_tool_run(
                runtime.session_id,
                f"mock_success_{index}",
                "READ_ONLY",
                True,
                False,
                "mock success",
            )
        _set_action(runtime, TOOL_NAME, {"task_id": task_id, "evidence": "proof"})
        calls = _install_counted_handler(runtime)
        blocked = runtime.handle(
            "complete task with an aged recovery blocker",
            request_token="task-evidence-aged-recovery",
        )
        item = blocked.tool_results[0]
        if (
            item.metadata.get("reason") != "recovery_closure_required"
            or calls[0] != 0
            or _task(runtime, task_id)["status"] != "open"
            or _evidence(runtime, task_id)
        ):
            raise SystemExit(f"old recovery failure aged out of completion custody: {item}")

    with TemporaryDirectory(prefix="jarvis-task-evidence-recovery-race-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("raced recovery target"))
        runtime.vault.sync_open_tasks(runtime.store)
        _set_action(runtime, TOOL_NAME, {"task_id": task_id, "evidence": "race proof"})
        original_complete = runtime.store.complete_task_with_evidence

        def inject_new_failure(*args: Any, **kwargs: Any) -> Any:
            runtime.store.log_tool_run(
                runtime.session_id,
                "mock_concurrent_failure",
                "HIGH_RISK",
                False,
                False,
                "concurrent failure",
            )
            return original_complete(*args, **kwargs)

        runtime.store.complete_task_with_evidence = inject_new_failure  # type: ignore[method-assign]
        try:
            raced = runtime.handle(
                "complete task with a concurrent recovery change",
                request_token="task-evidence-recovery-race",
            )
        finally:
            runtime.store.complete_task_with_evidence = original_complete  # type: ignore[method-assign]
        item = raced.tool_results[0]
        if (
            item.metadata.get("reason") != "recovery_state_changed"
            or item.metadata.get("auto_mutation_definite_no_effect") is not True
            or item.metadata.get("auto_mutation_outcome_uncertain") is not False
            or _task(runtime, task_id)["status"] != "open"
            or _evidence(runtime, task_id)
            or _receipt_rows(runtime)
        ):
            raise SystemExit(f"recovery race did not fail closed before effects: {item}")


def test_recovery_proofs_require_success_and_are_fenced_through_commit() -> None:
    with TemporaryDirectory(prefix="jarvis-task-evidence-proof-success-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("failed proof target"))
        runtime.vault.sync_open_tasks(runtime.store)
        target_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "mock_local_failure",
            "READ_ONLY",
            False,
            False,
            "mock failure",
        )
        _log_mock_recovery_proofs(runtime, target_run_id, ok=False)
        _set_action(runtime, TOOL_NAME, {"task_id": task_id, "evidence": "proof"})
        calls = _install_counted_handler(runtime)
        blocked = runtime.handle(
            "failed proof rows must not authorize completion",
            request_token="task-evidence-failed-proof-rows",
        )
        if (
            blocked.tool_results[0].metadata.get("reason")
            != "recovery_closure_required"
            or calls[0] != 0
            or _task(runtime, task_id)["status"] != "open"
            or _evidence(runtime, task_id)
        ):
            raise SystemExit(f"failed recovery proof authorized completion: {blocked}")

        _log_mock_recovery_proofs(runtime, target_run_id, ok=True)
        completed = runtime.handle(
            "successful proof rows authorize reviewed completion",
            request_token="task-evidence-success-proof-rows",
        )
        if (
            not completed.tool_results[0].ok
            or calls[0] != 1
            or _task(runtime, task_id)["status"] != "done"
            or len(_evidence(runtime, task_id)) != 1
        ):
            raise SystemExit(f"successful recovery proof did not close custody: {completed}")

    with TemporaryDirectory(prefix="jarvis-task-evidence-proof-race-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("proof fence race target"))
        runtime.vault.sync_open_tasks(runtime.store)
        target_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "mock_local_failure",
            "READ_ONLY",
            False,
            False,
            "mock failure",
        )
        proof_ids = _log_mock_recovery_proofs(runtime, target_run_id, ok=True)
        _set_action(runtime, TOOL_NAME, {"task_id": task_id, "evidence": "race proof"})
        original_complete = runtime.store.complete_task_with_evidence

        def delete_proofs_before_commit(*args: Any, **kwargs: Any) -> Any:
            with runtime.store.connect() as conn:
                placeholders = ", ".join("?" for _proof_id in proof_ids)
                conn.execute(
                    f"DELETE FROM tool_runs WHERE id IN ({placeholders})",
                    tuple(proof_ids),
                )
            return original_complete(*args, **kwargs)

        runtime.store.complete_task_with_evidence = delete_proofs_before_commit  # type: ignore[method-assign]
        try:
            raced = runtime.handle(
                "delete recovery proof rows before completion commit",
                request_token="task-evidence-proof-fence-race",
            )
        finally:
            runtime.store.complete_task_with_evidence = original_complete  # type: ignore[method-assign]
        item = raced.tool_results[0]
        if (
            item.metadata.get("reason") != "recovery_state_changed"
            or item.metadata.get("auto_mutation_definite_no_effect") is not True
            or _task(runtime, task_id)["status"] != "open"
            or _evidence(runtime, task_id)
            or _receipt_rows(runtime)
        ):
            raise SystemExit(f"recovery proof deletion crossed the commit fence: {item}")


def test_approval_chain_proof_can_close_approval_recovery() -> None:
    with TemporaryDirectory(prefix="jarvis-task-evidence-approval-proof-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("approval proof target"))
        runtime.vault.sync_open_tasks(runtime.store)
        target_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "mock_risky_failure",
            "HIGH_RISK",
            False,
            False,
            "Mock action queued as approval #11; no action executed.",
            metadata={"approval_id": 11},
        )
        _log_mock_recovery_proofs(runtime, target_run_id, ok=True)
        invalid_proof = ToolResult(
            "approval_chain_proof",
            True,
            "invalid approval chain proof",
            {
                "approval_id": 11,
                "valid_execution_proof": False,
                "verdict": "APPROVAL_EXECUTION_FAILED",
            },
        )
        invalid_audit_metadata = runtime._tool_run_audit_metadata(invalid_proof)
        runtime.store.log_tool_run(
            runtime.session_id,
            "approval_chain_proof",
            "READ_ONLY",
            True,
            False,
            invalid_proof.output,
            metadata=invalid_audit_metadata,
        )
        _set_action(
            runtime,
            TOOL_NAME,
            {"task_id": task_id, "evidence": "approval closure proof"},
        )
        blocked = runtime.handle(
            "invalid approval proof must not close recovery",
            request_token="task-evidence-invalid-approval-proof",
        )
        if (
            blocked.tool_results[0].metadata.get("reason")
            != "recovery_closure_required"
            or _task(runtime, task_id)["status"] != "open"
            or _evidence(runtime, task_id)
        ):
            raise SystemExit(f"invalid approval proof closed recovery: {blocked}")

        valid_proof = ToolResult(
            "approval_chain_proof",
            True,
            "valid approval chain proof",
            {
                "approval_id": 11,
                "valid_execution_proof": True,
                "verdict": "APPROVAL_CHAIN_PROVEN",
            },
        )
        valid_audit_metadata = runtime._tool_run_audit_metadata(valid_proof)
        if valid_audit_metadata != valid_proof.metadata:
            raise SystemExit(
                f"approval proof audit allowlist lost custody: {valid_audit_metadata}"
            )
        runtime.store.log_tool_run(
            runtime.session_id,
            "approval_chain_proof",
            "READ_ONLY",
            True,
            False,
            valid_proof.output,
            metadata=valid_audit_metadata,
        )
        completed = runtime.handle(
            "complete after approval proof closure",
            request_token="task-evidence-approval-proof",
        )
        if (
            not completed.tool_results[0].ok
            or _task(runtime, task_id)["status"] != "done"
            or len(_evidence(runtime, task_id)) != 1
        ):
            raise SystemExit(f"approval chain proof remained permanently blocked: {completed}")


def test_legacy_alias_receipt_matches_only_the_same_task() -> None:
    with TemporaryDirectory(prefix="jarvis-task-evidence-legacy-alias-") as temp:
        runtime = _runtime(Path(temp), {})
        first = runtime.store.add_task(TaskRecord("legacy alias first target"))
        second = runtime.store.add_task(TaskRecord("legacy alias second target"))
        prepared = runtime.store.prepare_auto_mutation_receipts(
            "legacy-task-status-first",
            [("update_task_status", {"task_id": first, "status": "paused"})],
        )
        if prepared.status != "PREPARED":
            raise SystemExit(f"could not seed legacy alias receipt: {prepared}")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET operation_scope = NULL "
                "WHERE id = ?",
                (prepared.receipts[0].receipt_id,),
            )
        aliases = tuple(sorted(TASK_STATUS_TOOLS))
        same = runtime.store.inspect_auto_mutation_receipts(
            "legacy-task-status-same",
            [
                (
                    TOOL_NAME,
                    {"task_id": first, "evidence": "proof"},
                    "task_status",
                    {"task_id": first},
                    aliases,
                )
            ],
        )
        same_changed_status = runtime.store.inspect_auto_mutation_receipts(
            "legacy-task-status-same-changed",
            [
                (
                    "update_task_status",
                    {"task_id": first, "status": "dropped"},
                    "task_status",
                    {"task_id": first},
                    aliases,
                )
            ],
        )
        other = runtime.store.inspect_auto_mutation_receipts(
            "legacy-task-status-other",
            [
                (
                    TOOL_NAME,
                    {"task_id": second, "evidence": "proof"},
                    "task_status",
                    {"task_id": second},
                    aliases,
                )
            ],
        )
        if (
            same.status != "UNRESOLVED_ACTION_BLOCKED"
            or same_changed_status.status != "UNRESOLVED_ACTION_BLOCKED"
            or other.status != "ABSENT"
        ):
            raise SystemExit(
                "legacy alias receipt crossed task identities: "
                f"same={same}, changed={same_changed_status}, other={other}"
            )


def test_ambiguous_legacy_status_spelling_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-task-evidence-ambiguous-legacy-") as temp:
        runtime = _runtime(Path(temp), {})
        first = runtime.store.add_task(TaskRecord("ambiguous legacy first target"))
        second = runtime.store.add_task(TaskRecord("ambiguous legacy second target"))
        token = "ambiguous-legacy-status-owner"
        prepared = runtime.store.prepare_auto_mutation_receipts(
            token,
            [("update_task_status", {"task_id": first, "status": " PAUSED "})],
        )
        if prepared.status != "PREPARED" or prepared.receipts[0].receipt_id is None:
            raise SystemExit(f"could not seed ambiguous legacy receipt: {prepared}")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET operation_scope = NULL WHERE id = ?",
                (prepared.receipts[0].receipt_id,),
            )
        aliases = tuple(sorted(TASK_STATUS_TOOLS))
        same_alias = runtime.store.inspect_auto_mutation_receipts(
            "ambiguous-legacy-same-alias",
            [
                (
                    TOOL_NAME,
                    {"task_id": first, "evidence": "proof"},
                    "task_status",
                    {"task_id": first},
                    aliases,
                )
            ],
        )
        same_changed = runtime.store.classify_auto_mutation_receipt(
            "ambiguous-legacy-same-changed",
            0,
            "update_task_status",
            {"task_id": first, "status": "dropped"},
            operation_args={"task_id": first},
            operation_scope="task_status",
            legacy_operation_aliases=aliases,
        )
        other = runtime.store.prepare_auto_mutation_receipts(
            "ambiguous-legacy-other-target",
            [
                (
                    "update_task_status",
                    {"task_id": second, "status": "open"},
                    "task_status",
                    {"task_id": second},
                    aliases,
                )
            ],
        )
        if (
            same_alias.status != "UNRESOLVED_ACTION_BLOCKED"
            or same_changed.status != "UNRESOLVED_ACTION_BLOCKED"
            or other.status != "UNRESOLVED_ACTION_BLOCKED"
        ):
            raise SystemExit(
                "ambiguous unscoped legacy receipt was not quarantined: "
                f"alias={same_alias}, changed={same_changed}, other={other}"
            )


def test_success_persists_evidence_done_status_and_audit() -> None:
    evidence = "all focused assertions passed"
    verification_run_id = "verification-run-42"
    private_task_path = "/\x55sers/private/client/completion-target.md"
    task_body = f"durable evidence success target at {private_task_path}"
    with TemporaryDirectory(prefix="jarvis-task-evidence-success-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord(task_body))
        runtime.vault.sync_open_tasks(runtime.store)
        args = {
            "task_id": task_id,
            "evidence": evidence,
            "verification_run_id": verification_run_id,
        }
        _set_action(runtime, TOOL_NAME, args)
        calls = _install_counted_handler(runtime)
        result = runtime.handle(
            f"complete task {task_id} with evidence: {evidence}",
            request_token="task-evidence-success-owner",
        )
        item = result.tool_results[0]
        rows = _evidence(runtime, task_id)
        projection = (_projection_bytes(runtime) or b"").decode("utf-8")
        if (
            not item.ok
            or calls[0] != 1
            or _task(runtime, task_id)["status"] != "done"
            or _task(runtime, task_id)["completed_at"] is None
            or len(rows) != 1
            or rows[0]["task_id"] != task_id
            or rows[0]["evidence"] != evidence
            or rows[0]["verification_run_id"] != verification_run_id
            or rows[0]["task_body"] != task_body
            or len(rows[0]["evidence_digest"]) != 64
            or f"#{task_id} durable evidence success target" in projection
        ):
            raise SystemExit(f"evidence-backed completion lost durable custody: {item} / {rows}")
        public_json = json.dumps(
            {
                "user_input": result.user_input,
                "response": result.response,
                "metadata": result.metadata,
                "tool_result": {
                    "output": item.output,
                    "metadata": item.metadata,
                },
                "messages": _rows(
                    runtime,
                    "SELECT role, content, metadata FROM messages ORDER BY id",
                ),
                "tool_runs": _tool_run_rows(runtime),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if (
            evidence in public_json
            or private_task_path in public_json
            or str(runtime.vault.root_path) in public_json
        ):
            raise SystemExit(
                "successful evidence completion exposed private proof, task path, or vault path"
            )
        _assert_completed_receipt_audit(runtime)

        replay = runtime.handle(
            "same-token completed evidence replay",
            request_token="task-evidence-success-owner",
        )
        _assert_failure_kind(replay, "auto_mutation_completed_replay", "completed replay")
        if calls[0] != 1 or len(_evidence(runtime, task_id)) != 1:
            raise SystemExit("completed same-token replay reached the evidence handler")


def test_completion_packet_redacts_private_evidence_command() -> None:
    private_path = "/\x55sers/private/client/packet-proof.txt"
    markers = [
        "PRIVATE-PACKET-EVIDENCE-91C7-A",
        "PRIVATE-PACKET-EVIDENCE-91C7-B",
        "PRIVATE-PACKET-EVIDENCE-91C7-C",
        "PRIVATE-PACKET-EVIDENCE-91C7-D",
    ]
    with TemporaryDirectory(prefix="jarvis-task-evidence-packet-privacy-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(
            TaskRecord(f"inspect private completion target {private_path}")
        )
        runtime.planner = RuleBasedPlanner()
        commands = [
            f"task completion packet {task_id}: {markers[0]} at {private_path}",
            f"   task completion packet {task_id}: {markers[1]} at {private_path}",
            f"Jarvis, task completion packet {task_id}: {markers[2]} at {private_path}",
            f"completion packet for task {task_id}: {markers[3]} at {private_path}",
        ]
        results = [
            runtime.handle(
                command,
                request_token=f"task-evidence-packet-privacy-{index}",
            )
            for index, command in enumerate(commands)
        ]
        public_json = json.dumps(
            {
                "runtime_results": [
                    {
                        "user_input": result.user_input,
                        "plan": result.plan,
                        "response": result.response,
                        "metadata": result.metadata,
                        "tool_results": [
                            {"output": item.output, "metadata": item.metadata}
                            for item in result.tool_results
                        ],
                    }
                    for result in results
                ],
                "messages": _rows(
                    runtime,
                    "SELECT role, content, metadata FROM messages ORDER BY id",
                ),
                "tool_runs": _tool_run_rows(runtime),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if (
            any(
                len(result.tool_results) != 1
                or not result.tool_results[0].ok
                or result.tool_results[0].tool_name != "task_completion_packet"
                or "<completion-evidence>" not in result.user_input
                for result in results
            )
            or any(marker in public_json for marker in markers)
            or private_path in public_json
        ):
            raise SystemExit(f"completion packet exposed private command evidence: {public_json}")


def test_rule_planner_preserves_explicit_evidence_boundaries() -> None:
    planner = RuleBasedPlanner()
    labelled = planner.plan("complete task 17 with evidence:   exact proof   ")
    run_bound = planner.plan(
        "complete task 18 with verification run 42:  exact run proof  "
    )
    polite = planner.plan(
        "Please complete task 19 with evidence:proof ending please   "
    )
    if (
        len(labelled.actions) != 1
        or labelled.actions[0].tool_name != TOOL_NAME
        or labelled.actions[0].args
        != {
            "task_id": 17,
            "verification_run_id": "",
            "evidence": "   exact proof   ",
        }
        or len(run_bound.actions) != 1
        or run_bound.actions[0].tool_name != TOOL_NAME
        or run_bound.actions[0].args
        != {
            "task_id": 18,
            "verification_run_id": "42",
            "evidence": "  exact run proof  ",
        }
        or len(polite.actions) != 1
        or polite.actions[0].tool_name != TOOL_NAME
        or polite.actions[0].args
        != {
            "task_id": 19,
            "verification_run_id": "",
            "evidence": "proof ending please   ",
        }
    ):
        raise SystemExit(
            "rule planner normalized explicit task-completion evidence boundaries: "
            f"{labelled.actions} / {run_bound.actions} / {polite.actions}"
        )


def test_runtime_preserves_exact_evidence_but_redacts_public_history() -> None:
    marker = "PRIVATE-EXACT-EVIDENCE-2D91"
    evidence = f"  {marker} alpha      beta\nfinal please   "
    with TemporaryDirectory(prefix="jarvis-task-evidence-exact-runtime-") as temp:
        runtime = _runtime(Path(temp), {})
        runtime.planner = RuleBasedPlanner()
        task_id = runtime.store.add_task(TaskRecord("exact runtime evidence target"))
        runtime.vault.sync_open_tasks(runtime.store)
        result = runtime.handle(
            f"complete task {task_id} with evidence:{evidence}",
            request_token="task-evidence-exact-runtime",
        )
        rows = _evidence(runtime, task_id)
        public_json = json.dumps(
            {
                "user_input": result.user_input,
                "plan": result.plan,
                "response": result.response,
                "metadata": result.metadata,
                "tool_results": [
                    {"output": item.output, "metadata": item.metadata}
                    for item in result.tool_results
                ],
                "messages": _rows(
                    runtime,
                    "SELECT role, content, metadata FROM messages ORDER BY id",
                ),
                "tool_runs": _tool_run_rows(runtime),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if (
            len(rows) != 1
            or rows[0]["evidence"] != evidence
            or not result.tool_results[0].ok
            or marker in public_json
            or "<completion-evidence>" not in result.user_input
        ):
            raise SystemExit(
                "runtime did not preserve exact private evidence custody: "
                f"rows={rows}, public={public_json}"
            )


def test_projection_failure_keeps_evidence_done_and_fences_all_replays() -> None:
    owner_token = "PRIVATE-TASK-EVIDENCE-PROJECTION-OWNER"
    fresh_token = "PRIVATE-TASK-EVIDENCE-PROJECTION-FRESH"
    complete_token = "PRIVATE-TASK-EVIDENCE-PROJECTION-COMPLETE"
    update_token = "PRIVATE-TASK-EVIDENCE-PROJECTION-UPDATE"
    evidence_marker = "PRIVATE-EVIDENCE-7F4D1A9C"
    private_path = "/\x55sers/private/client/task-proof.txt"
    evidence = f"{evidence_marker} stored at {private_path}"
    run_id = "PRIVATE-VERIFICATION-RUN-7F4D1A9C"
    with TemporaryDirectory(prefix="jarvis-task-evidence-projection-") as temp:
        root = Path(temp)
        runtime = _runtime(root, {})
        task_id = runtime.store.add_task(TaskRecord("projection failure target"))
        runtime.vault.sync_open_tasks(runtime.store)
        projection_before = _projection_bytes(runtime)
        args = {
            "task_id": task_id,
            "evidence": evidence,
            "verification_run_id": run_id,
        }
        _set_action(runtime, TOOL_NAME, args)
        calls = _install_counted_handler(runtime)
        publication_calls = [0]

        def fail_projection(*_args: Any, **_kwargs: Any) -> Any:
            publication_calls[0] += 1
            raise OSError("injected evidence-completion projection failure")

        _install_projection_failure(runtime, fail_projection)
        failed = runtime.handle(
            "inject projection failure after durable completion",
            request_token=owner_token,
        )
        item = failed.tool_results[0]
        rows = _evidence(runtime, task_id)
        if (
            item.ok
            or item.metadata.get("auto_mutation_outcome_uncertain") is not True
            or item.metadata.get("handler_invoked") is not True
            or calls[0] != 1
            or publication_calls[0] != 1
            or _task(runtime, task_id)["status"] != "done"
            or len(rows) != 1
            or rows[0]["evidence"] != evidence
            or rows[0]["verification_run_id"] != run_id
            or _projection_bytes(runtime) != projection_before
            or any(row["ok"] == 1 for row in _tool_run_rows(runtime))
        ):
            raise SystemExit(f"projection failure lost evidence completion custody: {item} / {rows}")
        _assert_uncertain_receipt(runtime, "projection failure")

        same = runtime.handle(
            "same-token uncertain evidence replay",
            request_token=owner_token,
        )
        _assert_failure_kind(same, "auto_mutation_outcome_uncertain", "same-token replay")

        _set_action(
            runtime,
            TOOL_NAME,
            {"task_id": task_id, "evidence": "changed private evidence"},
        )
        fresh = runtime.handle(
            "fresh-token uncertain evidence replay",
            request_token=fresh_token,
        )
        _assert_failure_kind(fresh, "auto_mutation_unresolved_action", "fresh-token replay")

        _set_action(runtime, "complete_task", {"task_id": task_id})
        complete = runtime.handle(
            "cross-tool complete replay",
            request_token=complete_token,
        )
        _assert_failure_kind(complete, "auto_mutation_unresolved_action", "complete_task replay")

        _set_action(
            runtime,
            "update_task_status",
            {"task_id": task_id, "status": "open"},
        )
        update = runtime.handle(
            "cross-tool status replay",
            request_token=update_token,
        )
        _assert_failure_kind(
            update,
            "auto_mutation_unresolved_action",
            "update_task_status replay",
        )
        if (
            calls[0] != 1
            or publication_calls[0] != 1
            or _task(runtime, task_id)["status"] != "done"
            or len(_evidence(runtime, task_id)) != 1
            or len(_receipt_rows(runtime)) != 1
        ):
            raise SystemExit("an unresolved replay bypassed evidence-completion fencing")
        _assert_uncertain_privacy(
            runtime,
            [failed, same, fresh, complete, update],
            [
                owner_token,
                fresh_token,
                complete_token,
                update_token,
                evidence_marker,
                private_path,
                run_id,
                str(root),
                str(runtime.vault.root_path),
                str(runtime.store.db_path),
            ],
        )


def test_audit_finalization_failure_keeps_evidence_done_and_uncertain() -> None:
    token = "PRIVATE-TASK-EVIDENCE-AUDIT-OWNER"
    evidence_marker = "PRIVATE-AUDIT-EVIDENCE-5B3E8C1D"
    private_path = "/private/tmp/client-audit-proof.txt"
    evidence = f"{evidence_marker} stored at {private_path}"
    with TemporaryDirectory(prefix="jarvis-task-evidence-audit-") as temp:
        root = Path(temp)
        runtime = _runtime(root, {})
        task_id = runtime.store.add_task(TaskRecord("audit failure target"))
        runtime.vault.sync_open_tasks(runtime.store)
        _set_action(
            runtime,
            TOOL_NAME,
            {"task_id": task_id, "evidence": evidence},
        )
        calls = _install_counted_handler(runtime)
        original_complete = runtime.store.complete_auto_mutation_receipt

        def fail_completion(**_kwargs: Any) -> int:
            raise OSError("injected evidence-completion audit finalization failure")

        runtime.store.complete_auto_mutation_receipt = fail_completion  # type: ignore[method-assign]
        try:
            failed = runtime.handle(
                "inject audit finalization failure after durable completion",
                request_token=token,
            )
        finally:
            runtime.store.complete_auto_mutation_receipt = original_complete  # type: ignore[method-assign]
        _assert_failure_kind(failed, "auto_mutation_completion_failed", "audit finalization")
        item = failed.tool_results[0]
        rows = _evidence(runtime, task_id)
        projection = (_projection_bytes(runtime) or b"").decode("utf-8")
        if (
            item.metadata.get("handler_invoked") is not True
            or item.metadata.get("audit_logging_failed") is not True
            or calls[0] != 1
            or _task(runtime, task_id)["status"] != "done"
            or len(rows) != 1
            or rows[0]["evidence"] != evidence
            or f"#{task_id} audit failure target" in projection
            or any(row["ok"] == 1 for row in _tool_run_rows(runtime))
        ):
            raise SystemExit(f"audit finalization failure lost durable evidence: {item} / {rows}")
        _assert_uncertain_receipt(runtime, "audit finalization failure")
        _assert_uncertain_privacy(
            runtime,
            [failed],
            [
                token,
                evidence_marker,
                private_path,
                str(root),
                str(runtime.vault.root_path),
                str(runtime.store.db_path),
            ],
        )


def test_concurrent_same_target_has_one_handler_and_evidence_row() -> None:
    with TemporaryDirectory(prefix="jarvis-task-evidence-concurrent-") as temp:
        runtime = _runtime(Path(temp), {})
        task_id = runtime.store.add_task(TaskRecord("concurrent evidence target"))
        runtime.vault.sync_open_tasks(runtime.store)
        owner_args = {"task_id": task_id, "evidence": "concurrent owner evidence"}
        contender_args = {"task_id": task_id, "evidence": "concurrent contender evidence"}
        _set_action(runtime, TOOL_NAME, owner_args)
        calls = _install_counted_handler(runtime)
        original_complete = runtime.store.complete_task_with_evidence
        entered = threading.Event()
        release = threading.Event()

        def delayed_complete(
            delayed_task_id: int,
            *,
            evidence: str = "",
            verification_run_id: str = "",
            expected_recovery_problem_run_id: int | None = None,
            expected_recovery_proof_fence: str | None = None,
            recovery_meta_tool_names: Any = (),
        ) -> Any:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("concurrent evidence-completion owner was not released")
            return original_complete(
                delayed_task_id,
                evidence=evidence,
                verification_run_id=verification_run_id,
                expected_recovery_problem_run_id=expected_recovery_problem_run_id,
                expected_recovery_proof_fence=expected_recovery_proof_fence,
                recovery_meta_tool_names=recovery_meta_tool_names,
            )

        runtime.store.complete_task_with_evidence = delayed_complete  # type: ignore[method-assign]
        with ThreadPoolExecutor(max_workers=1) as pool:
            owner_future = pool.submit(
                runtime.handle,
                "concurrent evidence owner",
                request_token="task-evidence-concurrent-owner",
            )
            if not entered.wait(10):
                release.set()
                raise SystemExit("concurrent evidence owner did not enter the handler")
            _set_action(runtime, TOOL_NAME, contender_args)
            try:
                blocked = runtime.handle(
                    "concurrent evidence contender",
                    request_token="task-evidence-concurrent-contender",
                )
            finally:
                release.set()
            owner = owner_future.result(timeout=15)
        runtime.store.complete_task_with_evidence = original_complete  # type: ignore[method-assign]

        _assert_failure_kind(
            blocked,
            "auto_mutation_unresolved_action",
            "concurrent evidence contender",
        )
        rows = _evidence(runtime, task_id)
        if (
            not owner.tool_results[0].ok
            or calls[0] != 1
            or _task(runtime, task_id)["status"] != "done"
            or len(rows) != 1
            or rows[0]["evidence"] != owner_args["evidence"]
        ):
            raise SystemExit(f"concurrent evidence completion did not have one winner: {owner.tool_results}")
        _assert_completed_receipt_audit(runtime)


def main() -> None:
    test_exact_schema_contract_and_shared_task_status_identity()
    test_invalid_and_unknown_arguments_precede_handler()
    test_semantic_preflights_are_zero_write()
    test_store_completion_rolls_back_evidence_if_status_update_fails()
    test_store_rejects_mixed_invalid_unicode_without_writes()
    test_store_preserves_exact_proof_and_records_each_completion_event()
    test_evidence_ledger_is_immutable_and_verifies_digest_on_read()
    test_incompatible_existing_evidence_schema_fails_closed_atomically()
    test_legacy_unique_evidence_schema_migrates_without_losing_proof()
    test_old_recovery_failure_cannot_age_out_or_race_completion()
    test_recovery_proofs_require_success_and_are_fenced_through_commit()
    test_approval_chain_proof_can_close_approval_recovery()
    test_legacy_alias_receipt_matches_only_the_same_task()
    test_ambiguous_legacy_status_spelling_fails_closed()
    test_success_persists_evidence_done_status_and_audit()
    test_completion_packet_redacts_private_evidence_command()
    test_rule_planner_preserves_explicit_evidence_boundaries()
    test_runtime_preserves_exact_evidence_but_redacts_public_history()
    test_projection_failure_keeps_evidence_done_and_fences_all_replays()
    test_audit_finalization_failure_keeps_evidence_done_and_uncertain()
    test_concurrent_same_target_has_one_handler_and_evidence_row()
    print("Task completion evidence auto-mutation smoke passed")


if __name__ == "__main__":
    main()
