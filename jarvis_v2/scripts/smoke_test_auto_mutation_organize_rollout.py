from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import threading
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.memory.goal_projection import reconcile_goal_projection
from jarvis_v2.memory.store import (
    GoalRecord,
    OrganizedNoteEntryRecord,
    auto_mutation_request_digest,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.organize import (
    MAX_ORGANIZE_LINES,
    MAX_ORGANIZE_TEXT_CHARS,
    organize_note_auto_mutation_operation_key,
    organize_note_auto_mutation_preflight,
    organize_note_auto_mutation_preflight_result,
)
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    TOOL_ARGUMENT_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    ToolArgumentType,
)


class StaticPlanner:
    def __init__(self, args: Any):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise the aggregate-wired organize-note mutation.",
            [PlannedAction("organize_note", self.args, "organize rollout smoke")],
            needs_model=False,
        )


class SimulatedOrganizerProcessDeath(BaseException):
    pass


def _runtime(root: Path, args: Any) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(args)
    return runtime


def _crash_after_organized_memory_publication(root_text: str) -> None:
    text = "remember: Process death organizer custody remains convergent."
    runtime = _runtime(Path(root_text), {"text": text})
    original_write = runtime.vault.write_organized_memory

    def write_then_exit(record: Any, memory_id: int, created_at: str) -> tuple[Path, bool]:
        original_write(record, memory_id, created_at)
        os._exit(37)

    runtime.vault.write_organized_memory = write_then_exit  # type: ignore[method-assign]
    runtime.handle(
        "organize before representative process death",
        request_token="PRIVATE-ORGANIZE-PROCESS-DEATH-OWNER",
    )
    os._exit(38)


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


def _batch_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM organized_note_batches ORDER BY source_key")


def _entry_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(
        runtime,
        "SELECT * FROM organized_note_entries ORDER BY source_key, entry_index",
    )


def _domain_snapshot(runtime: JarvisRuntime) -> dict[str, list[dict[str, Any]]]:
    return {
        table: _rows(runtime, f"SELECT * FROM {table} ORDER BY {order}")
        for table, order in (
            ("tasks", "id"),
            ("goals", "id"),
            ("memories", "id"),
            ("decisions", "id"),
            ("organized_note_batches", "source_key"),
            ("organized_note_entries", "source_key, entry_index"),
        )
    }


def _vault_snapshot(runtime: JarvisRuntime) -> dict[str, bytes]:
    return {
        str(path.relative_to(runtime.vault.root_path)): path.read_bytes()
        for path in runtime.vault.root_path.rglob("*.md")
    }


def _assert_no_organize_writes(
    runtime: JarvisRuntime,
    label: str,
    vault_before: dict[str, bytes],
) -> None:
    snapshot = _domain_snapshot(runtime)
    if any(snapshot.values()) or _receipt_rows(runtime):
        raise SystemExit(f"{label} crossed the receipt or domain-write boundary: {snapshot}")
    if _vault_snapshot(runtime) != vault_before:
        raise SystemExit(f"{label} changed the Obsidian baseline")


def _assert_replay(result: Any, failure_kind: str, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} did not return one tool result: {result.tool_results}")
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != failure_kind:
        raise SystemExit(f"{label} returned the wrong replay state: {item}")


def _assert_completed_receipt_audits(runtime: JarvisRuntime, expected: int) -> None:
    receipts = _receipt_rows(runtime)
    runs = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
    runs_by_id = {row["id"]: row for row in runs}
    if len(receipts) != expected or len(runs) != expected:
        raise SystemExit(f"expected {expected} organize receipt/audit pairs: {receipts} / {runs}")
    linked: set[int] = set()
    for receipt in receipts:
        run = runs_by_id.get(receipt["tool_run_id"])
        if (
            receipt["tool_name"] != "organize_note"
            or receipt["state"] != "completed"
            or receipt["result"] != "succeeded"
            or receipt["resolution"] != "recorded"
            or run is None
            or run["tool_name"] != "organize_note"
            or run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
        ):
            raise SystemExit(f"organize receipt did not link one ordinary audit: {receipt} / {run}")
        linked.add(int(run["id"]))
    if len(linked) != expected:
        raise SystemExit(f"organize receipts reused an audit row: {receipts}")


def _install_counted_handler(runtime: JarvisRuntime) -> list[int]:
    tool = runtime.registry.get("organize_note")
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools["organize_note"] = replace(tool, handler=counted)
    return calls


def test_exact_contract_typed_args_and_semantic_preflight() -> None:
    with TemporaryDirectory(prefix="jarvis-organize-contract-") as temp:
        runtime = _runtime(Path(temp), {"text": "remember: contract fixture"})
        tool = runtime.registry.get("organize_note")
        contract = tool.auto_mutation_contract
        if (
            tool.toolset != "organize"
            or contract is None
            or contract.version != AUTO_MUTATION_CONTRACT_VERSION
            or contract.effects
            != frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT})
            or contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
            or contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
            or contract.operation_key_builder is not organize_note_auto_mutation_operation_key
            or contract.semantic_preflight is not organize_note_auto_mutation_preflight
            or contract.semantic_preflight_result_builder
            is not organize_note_auto_mutation_preflight_result
        ):
            raise SystemExit(f"organize_note auto-mutation contract drifted: {tool}")
        arguments = tool.argument_contract
        string = frozenset({ToolArgumentType.STRING})
        shape = (
            tuple((field.name, field.types, field.required) for field in arguments.fields)
            if arguments is not None
            else ()
        )
        if (
            arguments is None
            or arguments.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or arguments.allow_unknown
            or shape != (("text", string, True),)
        ):
            raise SystemExit(f"organize_note strict argument contract drifted: {arguments}")
        equivalent_keys = {
            json.dumps(
                organize_note_auto_mutation_operation_key({"text": text}),
                sort_keys=True,
                ensure_ascii=False,
            )
            for text in (
                "TASK: normalize organizer spacing",
                "- todo: normalize   organizer\tspacing",
                "\r\n1. task: normalize organizer spacing\r\n",
            )
        }
        changed_key = json.dumps(
            organize_note_auto_mutation_operation_key(
                {"text": "task: normalize organizer spacing tomorrow"}
            ),
            sort_keys=True,
            ensure_ascii=False,
        )
        if len(equivalent_keys) != 1 or changed_key in equivalent_keys:
            raise SystemExit("organize operation identity lost semantic normalization or change detection")

    typed_cases: list[Any] = [
        {},
        {"text": None},
        {"text": 1},
        {"text": True},
        {"text": []},
        {"text": {}},
        {"text": "remember: safe", "unknown": "rejected"},
        None,
        [],
    ]
    for index, args in enumerate(typed_cases):
        with TemporaryDirectory(prefix="jarvis-organize-typed-") as temp:
            runtime = _runtime(Path(temp), args)
            vault_before = _vault_snapshot(runtime)
            calls = _install_counted_handler(runtime)
            result = runtime.handle("invalid typed organize note", request_token=f"typed-{index}")
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
            ):
                raise SystemExit(f"organize typed case {index} drifted: {item}")
            _assert_no_organize_writes(runtime, f"organize typed case {index}", vault_before)

    path_sentinel = "/\x55sers/example/PRIVATE-ORGANIZE-PREFLIGHT-PATH"
    semantic_cases = [
        ({"text": " \r\n "}, "missing_text"),
        ({"text": "x" * (MAX_ORGANIZE_TEXT_CHARS + 1)}, "text_too_large"),
        ({"text": "an unrecognized organizer line"}, "no_recognizable_lines"),
        ({"text": f"remember: {path_sentinel}"}, "no_recognizable_lines"),
        (
            {"text": "\n".join(f"task: overflow fixture {index}" for index in range(MAX_ORGANIZE_LINES + 1))},
            "too_many_lines",
        ),
    ]
    for index, (args, reason) in enumerate(semantic_cases):
        with TemporaryDirectory(prefix="jarvis-organize-semantic-") as temp:
            runtime = _runtime(Path(temp), args)
            vault_before = _vault_snapshot(runtime)
            calls = _install_counted_handler(runtime)
            result = runtime.handle("invalid semantic organize note", request_token=f"semantic-{index}")
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
                or item.metadata.get("reason") != reason
                or item.metadata.get("handler_invoked") is not False
                or item.metadata.get("organize_refusal_handoff_ready") is not True
                or item.metadata.get("state_changed") is not False
                or item.metadata.get("writes_database") is not False
                or item.metadata.get("writes_files") is not False
                or calls[0] != 0
            ):
                raise SystemExit(f"organize semantic case {index} drifted: {item}")
            _assert_no_organize_writes(runtime, f"organize semantic case {index}", vault_before)
            exposed = json.dumps(
                {"response": result.response, "output": item.output, "metadata": item.metadata},
                default=str,
            )
            if path_sentinel in exposed:
                raise SystemExit("organize semantic preflight exposed a raw local path")


def test_mixed_success_replay_convergence_and_decision_memory_truth() -> None:
    text = "\n".join(
        (
            "task: Ship organizer rollout due 2026-07-20 priority high",
            "goal: Reliable organizer | Prove convergence | this quarter",
            "remember: Organizer inputs remain deterministic.",
            "decision: Use reservation batches | Preserve exact IDs across retry | Eliminate duplicate domain truth",
        )
    )
    with TemporaryDirectory(prefix="jarvis-organize-success-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        first_token = "PRIVATE-ORGANIZE-SUCCESS-TOKEN-ONE"
        second_token = "PRIVATE-ORGANIZE-SUCCESS-TOKEN-TWO"
        first = runtime.handle("organize mixed fixture", request_token=first_token)
        item = first.tool_results[0]
        if (
            not item.ok
            or item.metadata.get("organized_batch_created") is not True
            or item.metadata.get("organized_batch_completed") is not True
            or item.metadata.get("converged") is not False
            or [item.metadata.get(key) for key in ("tasks", "goals", "memories", "decisions")]
            != [1, 1, 2, 1]
        ):
            raise SystemExit(f"mixed organize mutation failed: {item}")

        tasks = _rows(runtime, "SELECT * FROM tasks ORDER BY id")
        goals = _rows(runtime, "SELECT * FROM goals ORDER BY id")
        memories = _rows(runtime, "SELECT * FROM memories ORDER BY id")
        decisions = _rows(runtime, "SELECT * FROM decisions ORDER BY id")
        batches = _batch_rows(runtime)
        entries = _entry_rows(runtime)
        if (
            len(tasks) != 1
            or tasks[0]["body"] != "Ship organizer rollout"
            or tasks[0]["due"] != "2026-07-20"
            or tasks[0]["priority"] != "high"
            or len(goals) != 1
            or goals[0]["title"] != "Reliable organizer"
            or goals[0]["purpose"] != "Prove convergence"
            or goals[0]["horizon"] != "this quarter"
            or len(memories) != 2
            or len(decisions) != 1
            or len(batches) != 1
            or batches[0]["state"] != "completed"
            or len(entries) != 4
            or [entry["kind"] for entry in entries] != ["task", "goal", "memory", "decision"]
        ):
            raise SystemExit("mixed organize domain truth drifted")

        decision = decisions[0]
        decision_entry = entries[3]
        decision_memory = next(
            memory for memory in memories if memory["id"] == decision_entry["decision_memory_id"]
        )
        expected_body = (
            f"{decision['title']}\n\nRationale: {decision['rationale']}"
            f"\n\nImpact: {decision['impact']}"
        )
        if (
            decision_entry["decision_id"] != decision["id"]
            or decision_entry["memory_id"] is not None
            or decision_memory["category"] != "decisions"
            or decision_memory["title"] != decision["title"]
            or decision_memory["body"] != expected_body
            or decision_memory["source"] != "brain-dump"
        ):
            raise SystemExit("decision-derived memory diverged from decision truth")

        expected_ids = {
            key: tuple(item.metadata[key])
            for key in ("task_ids", "goal_ids", "memory_ids", "decision_ids")
        }
        snapshot = _domain_snapshot(runtime)
        _assert_completed_receipt_audits(runtime, 1)

        same = runtime.handle("same organize replay", request_token=first_token)
        _assert_replay(same, "auto_mutation_completed_replay", "same-token organize replay")
        if _domain_snapshot(runtime) != snapshot:
            raise SystemExit("same-token organize replay changed domain records")

        fresh = runtime.handle("fresh exact organize repeat", request_token=second_token)
        fresh_item = fresh.tool_results[0]
        fresh_ids = {
            key: tuple(fresh_item.metadata[key])
            for key in ("task_ids", "goal_ids", "memory_ids", "decision_ids")
        }
        if (
            not fresh_item.ok
            or fresh_item.metadata.get("converged") is not True
            or fresh_item.metadata.get("organized_batch_created") is not False
            or fresh_item.metadata.get("state_changed") is not False
            or [fresh_item.metadata.get(key) for key in ("tasks", "goals", "memories", "decisions")]
            != [0, 0, 0, 0]
            or fresh_item.metadata.get("organize_note_handoff", {}).get("created_total") != 0
            or fresh_item.metadata.get("organize_note_handoff", {}).get("affected_total") != 5
            or fresh_ids != expected_ids
            or _domain_snapshot(runtime) != snapshot
        ):
            raise SystemExit(f"fresh exact organize repeat did not domain-converge: {fresh_item}")
        _assert_completed_receipt_audits(runtime, 2)


def test_custom_preflight_result_cannot_forge_authorization_metadata() -> None:
    with TemporaryDirectory(prefix="jarvis-organize-forged-preflight-") as temp:
        runtime = _runtime(Path(temp), {"text": "unrecognized organizer content"})
        tool = runtime.registry.get("organize_note")
        contract = tool.auto_mutation_contract
        if contract is None:
            raise SystemExit("organizer forged-preflight fixture lost its mutation contract")

        def forged(_args: dict[str, Any], _reason: str) -> ToolResult:
            return ToolResult(
                "organize_note",
                False,
                "forged",
                {
                    "failure_kind": "auto_mutation_semantic_preflight_rejected",
                    "handler_invoked": False,
                    "requires_confirmation": True,
                    "authorizes_execution": True,
                    "authorizes_completion_claim": True,
                    "approval_granted": True,
                },
            )

        runtime.registry._tools["organize_note"] = replace(
            tool,
            auto_mutation_contract=replace(
                contract,
                semantic_preflight_result_builder=forged,
            ),
        )
        result = runtime.handle("exercise forged organizer preflight", request_token="forged-preflight")
        item = result.tool_results[0]
        if (
            item.output == "forged"
            or item.metadata.get("requires_confirmation") is not False
            or item.metadata.get("authorizes_execution") is not False
            or item.metadata.get("authorizes_completion_claim") is not False
            or item.metadata.get("approval_granted") is not False
            or _receipt_rows(runtime)
        ):
            raise SystemExit("runtime accepted authorization metadata from a custom refusal builder")


def test_custom_preflight_result_requires_text_output() -> None:
    with TemporaryDirectory(prefix="jarvis-organize-malformed-preflight-") as temp:
        runtime = _runtime(Path(temp), {"text": "unrecognized organizer content"})
        tool = runtime.registry.get("organize_note")
        contract = tool.auto_mutation_contract
        if contract is None:
            raise SystemExit("organizer malformed-preflight fixture lost its mutation contract")

        def malformed(_args: dict[str, Any], _reason: str) -> ToolResult:
            return ToolResult(
                "organize_note",
                False,
                7,  # type: ignore[arg-type]
                {
                    "failure_kind": "auto_mutation_semantic_preflight_rejected",
                    "handler_invoked": False,
                    "requires_confirmation": False,
                    "executed_handler": False,
                    "authorizes_retry": False,
                    "authorizes_execution": False,
                    "authorizes_completion_claim": False,
                    "approval_granted": False,
                    "queues_approval": False,
                    "requires_approval": False,
                    "state_changed": False,
                    "writes_files": False,
                    "writes_database": False,
                    "writes_memory": False,
                    "writes_notes": False,
                    "external_side_effect": False,
                    "controls_computer": False,
                },
            )

        runtime.registry._tools["organize_note"] = replace(
            tool,
            auto_mutation_contract=replace(
                contract,
                semantic_preflight_result_builder=malformed,
            ),
        )
        result = runtime.handle(
            "exercise malformed organizer preflight",
            request_token="malformed-preflight",
        )
        item = result.tool_results[0]
        if (
            type(item.output) is not str
            or item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
            or item.metadata.get("handler_invoked") is not False
            or _receipt_rows(runtime)
        ):
            raise SystemExit("runtime accepted a non-text custom refusal result")


def test_mirror_failure_blocks_replay_and_direct_retry_repairs_pending_batch() -> None:
    initial_text = "remember: Recover the organizer mirror without duplicate records."
    equivalent_text = "memory: Recover the organizer mirror without duplicate records."
    with TemporaryDirectory(prefix="jarvis-organize-recovery-") as temp:
        runtime = _runtime(Path(temp), {"text": initial_text})
        original_write = runtime.vault.write_organized_memory
        mirror_calls = [0]

        def fail_memory_mirror(
            _record: Any,
            _memory_id: int,
            _created_at: str,
        ) -> tuple[Path, bool]:
            mirror_calls[0] += 1
            raise OSError("representative organizer mirror failure")

        runtime.vault.write_organized_memory = fail_memory_mirror  # type: ignore[method-assign]
        owner_token = "PRIVATE-ORGANIZE-UNCERTAIN-OWNER"
        equivalent_token = "PRIVATE-ORGANIZE-UNCERTAIN-EQUIVALENT"
        failed = runtime.handle("organize with mirror failure", request_token=owner_token)
        if failed.tool_results[0].metadata.get("failure_kind") != "tool_error":
            raise SystemExit(f"organize mirror failure did not become uncertain: {failed.tool_results}")

        receipts = _receipt_rows(runtime)
        batches = _batch_rows(runtime)
        entries = _entry_rows(runtime)
        memories = _rows(runtime, "SELECT * FROM memories ORDER BY id")
        if (
            mirror_calls[0] != 1
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
            or len(batches) != 1
            or batches[0]["state"] != "pending"
            or len(entries) != 1
            or entries[0]["kind"] != "memory"
            or len(memories) != 1
            or entries[0]["memory_id"] != memories[0]["id"]
            or any(row["ok"] == 1 for row in _tool_run_rows(runtime))
        ):
            raise SystemExit("organize mirror failure lost pending single-record custody")
        original_id = int(memories[0]["id"])

        same = runtime.handle("same uncertain organize retry", request_token=owner_token)
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token uncertain organize replay")
        runtime.planner = StaticPlanner({"text": equivalent_text})
        equivalent = runtime.handle(
            "equivalent uncertain organize retry",
            request_token=equivalent_token,
        )
        _assert_replay(
            equivalent,
            "auto_mutation_unresolved_action",
            "equivalent-token uncertain organize replay",
        )
        if mirror_calls[0] != 1 or len(_receipt_rows(runtime)) != 1:
            raise SystemExit("blocked uncertain organize replay reran or created a receipt")

        runtime.vault.write_organized_memory = original_write  # type: ignore[method-assign]
        repaired = runtime.registry.get("organize_note").handler({"text": equivalent_text})
        repaired_memories = _rows(runtime, "SELECT * FROM memories ORDER BY id")
        repaired_batches = _batch_rows(runtime)
        repaired_entries = _entry_rows(runtime)
        if (
            not repaired.ok
            or repaired.metadata.get("organized_batch_created") is not False
            or repaired.metadata.get("organized_batch_recovered") is not True
            or repaired.metadata.get("organized_batch_completed") is not True
            or repaired.metadata.get("converged") is not False
            or repaired.metadata.get("memory_ids") != [original_id]
            or len(repaired_memories) != 1
            or repaired_memories[0]["id"] != original_id
            or len(repaired_batches) != 1
            or repaired_batches[0]["state"] != "completed"
            or len(repaired_entries) != 1
            or repaired_entries[0]["memory_id"] != original_id
            or len(_receipt_rows(runtime)) != 1
        ):
            raise SystemExit(f"direct organize retry did not repair the pending batch: {repaired}")


def test_late_entry_trigger_failure_rolls_back_atomic_reservation() -> None:
    with TemporaryDirectory(prefix="jarvis-organize-rollback-") as temp:
        runtime = _runtime(
            Path(temp),
            {"text": "task: rollback first row\ngoal: rollback later row"},
        )
        with runtime.store.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER fail_later_organized_entry
                BEFORE INSERT ON organized_note_entries
                WHEN NEW.entry_index = 1
                BEGIN
                    SELECT RAISE(ABORT, 'representative later-entry failure');
                END
                """
            )
        vault_before = _vault_snapshot(runtime)
        try:
            runtime.registry.get("organize_note").handler(
                {"text": "task: rollback first row\ngoal: rollback later row"}
            )
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("later organized-note entry trigger did not abort the reservation")
        _assert_no_organize_writes(runtime, "later organized-note entry trigger", vault_before)


def test_direct_concurrent_identical_calls_create_one_record_set() -> None:
    text = "remember: Concurrent organizer calls share one durable record."
    with TemporaryDirectory(prefix="jarvis-organize-concurrent-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        handler = runtime.registry.get("organize_note").handler
        workers = 6
        barrier = threading.Barrier(workers)

        def invoke(_index: int) -> ToolResult:
            barrier.wait(timeout=5)
            return handler({"text": text})

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(invoke, range(workers)))
        batches = _batch_rows(runtime)
        entries = _entry_rows(runtime)
        memories = _rows(runtime, "SELECT * FROM memories ORDER BY id")
        returned_ids = {tuple(result.metadata.get("memory_ids", [])) for result in results}
        if (
            any(not result.ok for result in results)
            or len(batches) != 1
            or batches[0]["state"] != "completed"
            or len(entries) != 1
            or len(memories) != 1
            or entries[0]["memory_id"] != memories[0]["id"]
            or returned_ids != {(memories[0]["id"],)}
            or sum(result.metadata.get("organized_batch_created") is True for result in results) != 1
            or not 1 <= sum(result.metadata.get("state_changed") is True for result in results) <= 2
            or sum(result.metadata.get("writes_files") is True for result in results) != 1
            or any(result.metadata.get("organized_batch_completed") is not True for result in results)
            or _receipt_rows(runtime)
            or _tool_run_rows(runtime)
        ):
            raise SystemExit(
                "concurrent direct organize calls did not converge to one record set: "
                f"results={results!r} batches={batches!r} entries={entries!r} memories={memories!r}"
            )


def test_direct_concurrent_goal_calls_converge_after_completion_race() -> None:
    text = "goal: Concurrent organizer goal | share durable projection | this month"
    with TemporaryDirectory(prefix="jarvis-organize-concurrent-goal-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        handler = runtime.registry.get("organize_note").handler
        workers = 6
        barrier = threading.Barrier(workers)

        def invoke(_index: int) -> ToolResult:
            barrier.wait(timeout=5)
            return handler({"text": text})

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(invoke, range(workers)))
        goals = _rows(runtime, "SELECT * FROM goals ORDER BY id")
        job = runtime.store.get_goal_projection_job(1)
        goal_files = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if (
            any(not result.ok for result in results)
            or len(_batch_rows(runtime)) != 1
            or _batch_rows(runtime)[0]["state"] != "completed"
            or len(goals) != 1
            or job is None
            or job["state"] != "completed"
            or len(goal_files) != 1
            or sum(
                result.metadata.get("organized_batch_created") is True
                for result in results
            )
            != 1
        ):
            raise SystemExit(
                "concurrent organizer goal calls did not converge after completion race"
            )


def test_unowned_regular_destinations_are_not_overwritten() -> None:
    cases = (
        (
            "remember: Organizer collision memory",
            Path("Memory Tree/Facts/000001 Organizer collision memory.md"),
        ),
        (
            "decision: Organizer collision decision | preserve file ownership | fail closed",
            Path("Decisions/0001 Organizer collision decision.md"),
        ),
    )
    for index, (text, relative_path) in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-organize-collision-") as temp:
            runtime = _runtime(Path(temp), {"text": text})
            destination = runtime.vault.root_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            sentinel = f"UNOWNED-ORGANIZE-DESTINATION-{index}"
            destination.write_text(sentinel, encoding="utf-8")
            handler = runtime.registry.get("organize_note").handler
            for attempt in range(2):
                try:
                    handler({"text": text})
                except FileExistsError:
                    pass
                else:
                    raise SystemExit("organize overwrote or accepted an unowned regular destination")
                if destination.read_text(encoding="utf-8") != sentinel:
                    raise SystemExit("organize changed an unowned regular destination")
                if len(_batch_rows(runtime)) != 1 or _batch_rows(runtime)[0]["state"] != "pending":
                    raise SystemExit("organize collision lost its pending batch custody")
                if len(_entry_rows(runtime)) != 1:
                    raise SystemExit("organize collision changed its reserved entry identity")
            snapshot = _domain_snapshot(runtime)
            if len(snapshot["memories"]) != 1:
                raise SystemExit("organize collision did not preserve one reserved memory identity")
            if len(snapshot["decisions"]) != (0 if index == 0 else 1):
                raise SystemExit("organize collision did not preserve one reserved decision identity")


def test_process_death_preserves_one_repairable_batch() -> None:
    text = "remember: Process death organizer custody remains convergent."
    token = "PRIVATE-ORGANIZE-PROCESS-DEATH-OWNER"
    with TemporaryDirectory(prefix="jarvis-organize-process-death-") as temp:
        root = Path(temp)
        process = multiprocessing.get_context("spawn").Process(
            target=_crash_after_organized_memory_publication,
            args=(str(root),),
        )
        process.start()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise SystemExit("organize process-death fixture did not terminate")
        if process.exitcode != 37:
            raise SystemExit(f"organize process-death fixture exited unexpectedly: {process.exitcode}")

        runtime = _runtime(root, {"text": text})
        receipts = _receipt_rows(runtime)
        batches = _batch_rows(runtime)
        memories = _rows(runtime, "SELECT * FROM memories ORDER BY id")
        notes = list((runtime.vault.root_path / "Memory Tree" / "Facts").glob("*.md"))
        if (
            len(receipts) != 1
            or receipts[0]["state"] != "running"
            or len(batches) != 1
            or batches[0]["state"] != "pending"
            or len(memories) != 1
            or len(notes) != 1
        ):
            raise SystemExit("organize process death lost its durable row/file custody")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET running_at = ?, updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z", receipts[0]["id"]),
            )

        recovered = _runtime(root, {"text": text})
        recovered_receipts = _receipt_rows(recovered)
        if (
            len(recovered_receipts) != 1
            or recovered_receipts[0]["state"] != "uncertain"
            or recovered_receipts[0]["result"] != "unknown"
            or recovered_receipts[0]["resolution"] != "stale_recovery"
        ):
            raise SystemExit("startup did not fence the stale organizer process-death receipt")
        replay = recovered.handle("replay process-death organizer", request_token=token)
        _assert_replay(replay, "auto_mutation_outcome_uncertain", "process-death organizer replay")
        if len(_rows(recovered, "SELECT * FROM memories")) != 1:
            raise SystemExit("process-death organizer replay duplicated its memory")

        repaired = recovered.registry.get("organize_note").handler({"text": text})
        if (
            not repaired.ok
            or repaired.metadata.get("organized_batch_recovered") is not True
            or len(_rows(recovered, "SELECT * FROM memories")) != 1
            or _batch_rows(recovered)[0]["state"] != "completed"
            or len(list((recovered.vault.root_path / "Memory Tree" / "Facts").glob("*.md"))) != 1
        ):
            raise SystemExit("process-death organizer batch did not repair without duplication")


def test_completed_ledger_does_not_block_supported_memory_deletion() -> None:
    text = "remember: Organizer deletion ledger remains a non-owning tombstone."
    with TemporaryDirectory(prefix="jarvis-organize-delete-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        first = runtime.handle("organize deletable memory", request_token="organize-delete-owner")
        memory_id = first.tool_results[0].metadata.get("memory_ids", [None])[0]
        if type(memory_id) is not int or not runtime.store.delete_memory(memory_id):
            raise SystemExit("organizer ledger blocked supported memory deletion")
        entries = _entry_rows(runtime)
        if len(entries) != 1 or entries[0]["memory_id"] != memory_id:
            raise SystemExit("organizer deletion lost its non-owning historical identity")
        replay = runtime.handle("organize deleted memory replay", request_token="organize-delete-fresh")
        item = replay.tool_results[0]
        if (
            not item.ok
            or item.metadata.get("converged") is not True
            or item.metadata.get("state_changed") is not False
            or _rows(runtime, "SELECT * FROM memories")
        ):
            raise SystemExit("completed organizer replay recreated a deliberately deleted memory")


def test_owned_target_foreign_key_upgrade_becomes_non_owning() -> None:
    text = "remember: Organizer schema upgrade preserves deletable memory truth."
    with TemporaryDirectory(prefix="jarvis-organize-schema-upgrade-") as temp:
        root = Path(temp)
        runtime = _runtime(root, {"text": text})
        first = runtime.handle("organize before schema upgrade", request_token="organize-upgrade-owner")
        memory_id = first.tool_results[0].metadata.get("memory_ids", [None])[0]
        if type(memory_id) is not int:
            raise SystemExit("organizer schema upgrade fixture did not create a memory")
        with runtime.store.connect() as conn:
            conn.execute("ALTER TABLE organized_note_entries RENAME TO organized_note_entries_current")
            conn.execute(
                """
                CREATE TABLE organized_note_entries (
                    source_key TEXT NOT NULL REFERENCES organized_note_batches(source_key) ON DELETE RESTRICT,
                    entry_index INTEGER NOT NULL CHECK(entry_index >= 0),
                    kind TEXT NOT NULL CHECK(kind IN ('task', 'goal', 'memory', 'decision')),
                    task_id INTEGER UNIQUE REFERENCES tasks(id) ON DELETE RESTRICT,
                    goal_id INTEGER UNIQUE REFERENCES goals(id) ON DELETE RESTRICT,
                    memory_id INTEGER UNIQUE REFERENCES memories(id) ON DELETE RESTRICT,
                    decision_id INTEGER UNIQUE REFERENCES decisions(id) ON DELETE RESTRICT,
                    decision_memory_id INTEGER UNIQUE REFERENCES memories(id) ON DELETE RESTRICT,
                    PRIMARY KEY(source_key, entry_index)
                )
                """
            )
            conn.execute(
                "INSERT INTO organized_note_entries SELECT * FROM organized_note_entries_current"
            )
            conn.execute("DROP TABLE organized_note_entries_current")

        upgraded = _runtime(root, {"text": text})
        foreign_keys = _rows(upgraded, "PRAGMA foreign_key_list(organized_note_entries)")
        if [row["table"] for row in foreign_keys] != ["organized_note_batches"]:
            raise SystemExit(f"organizer schema upgrade retained target-owning foreign keys: {foreign_keys}")
        if not upgraded.store.delete_memory(memory_id):
            raise SystemExit("organizer schema upgrade still blocked supported memory deletion")


def test_legacy_completed_batch_rebuilds_missing_projection_manifest() -> None:
    text = "remember: Legacy organizer completion can rebuild projection custody."
    with TemporaryDirectory(prefix="jarvis-organize-legacy-manifest-") as temp:
        root = Path(temp)
        runtime = _runtime(root, {"text": text})
        first = runtime.handle(
            "organize before legacy manifest simulation",
            request_token="organize-legacy-manifest-owner",
        )
        memory_id = first.tool_results[0].metadata.get("memory_ids", [None])[0]
        if type(memory_id) is not int:
            raise SystemExit("legacy organizer manifest fixture did not create a memory")
        projection = next((runtime.vault.root_path / "Memory Tree" / "Facts").glob("*.md"))
        projection.unlink()
        with sqlite3.connect(runtime.store.db_path) as conn:
            conn.execute("PRAGMA ignore_check_constraints = ON")
            conn.execute(
                "UPDATE organized_note_batches SET projection_manifest = NULL WHERE state = 'completed'"
            )

        replay = runtime.handle(
            "replay legacy organizer completion",
            request_token="organize-legacy-manifest-fresh",
        )
        item = replay.tool_results[0]
        batches = _batch_rows(runtime)
        if (
            not item.ok
            or item.metadata.get("organized_batch_completed") is not True
            or not projection.exists()
            or len(batches) != 1
            or not batches[0].get("projection_manifest")
            or len(_rows(runtime, "SELECT * FROM memories")) != 1
        ):
            raise SystemExit("legacy completed organizer batch did not rebuild manifest custody")


def test_legacy_manifest_repair_tombstones_deleted_target_without_resurrection() -> None:
    text = "remember: Legacy organizer deletion remains deleted during manifest repair."
    with TemporaryDirectory(prefix="jarvis-organize-legacy-deleted-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        first = runtime.handle(
            "organize before legacy deletion simulation",
            request_token="organize-legacy-delete-owner",
        )
        memory_id = first.tool_results[0].metadata.get("memory_ids", [None])[0]
        projection = next((runtime.vault.root_path / "Memory Tree" / "Facts").glob("*.md"))
        projection.unlink()
        with sqlite3.connect(runtime.store.db_path) as conn:
            conn.execute("PRAGMA ignore_check_constraints = ON")
            conn.execute(
                "UPDATE organized_note_batches SET projection_manifest = NULL WHERE state = 'completed'"
            )
        if type(memory_id) is not int or not runtime.store.delete_memory(memory_id):
            raise SystemExit("legacy organizer deletion fixture did not delete its memory")

        replay = runtime.handle(
            "replay legacy deleted organizer completion",
            request_token="organize-legacy-delete-fresh",
        )
        item = replay.tool_results[0]
        batches = _batch_rows(runtime)
        manifest = json.loads(str(batches[0].get("projection_manifest") or "null"))
        if (
            not item.ok
            or item.metadata.get("organized_batch_recovered") is not True
            or _rows(runtime, "SELECT * FROM memories")
            or projection.exists()
            or not isinstance(manifest, list)
            or len(manifest) != 1
            or manifest[0].get("legacy_lifecycle_changed") is not True
        ):
            raise SystemExit("legacy manifest repair resurrected or lost a deleted target")

        repeated = runtime.handle(
            "repeat legacy deleted organizer completion",
            request_token="organize-legacy-delete-repeat",
        )
        if not repeated.tool_results[0].ok or _rows(runtime, "SELECT * FROM memories"):
            raise SystemExit("legacy deletion tombstone did not remain replay-convergent")


def test_legacy_lifecycle_tombstone_self_heals_after_value_restore() -> None:
    text = "remember: Restored legacy organizer values regain projection verification."
    with TemporaryDirectory(prefix="jarvis-organize-legacy-restore-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        first = runtime.handle(
            "organize before legacy restore simulation",
            request_token="organize-legacy-restore-owner",
        )
        memory_id = first.tool_results[0].metadata.get("memory_ids", [None])[0]
        original = runtime.store.get_memory(memory_id) if type(memory_id) is int else None
        if original is None:
            raise SystemExit("legacy organizer restore fixture did not create a memory")
        with sqlite3.connect(runtime.store.db_path) as conn:
            conn.execute("PRAGMA ignore_check_constraints = ON")
            conn.execute(
                "UPDATE organized_note_batches SET projection_manifest = NULL WHERE state = 'completed'"
            )
        if not runtime.store.update_memory(
            memory_id,
            str(original["category"]),
            str(original["title"]),
            "temporarily changed after the legacy completion",
            float(original["confidence"]),
        ):
            raise SystemExit("legacy organizer restore fixture could not edit its memory")
        tombstoned = runtime.handle(
            "repair legacy manifest after edit",
            request_token="organize-legacy-restore-tombstone",
        )
        tombstone_manifest = json.loads(
            str(_batch_rows(runtime)[0].get("projection_manifest") or "null")
        )
        if (
            not tombstoned.tool_results[0].ok
            or tombstone_manifest[0].get("legacy_lifecycle_changed") is not True
        ):
            raise SystemExit("legacy organizer edit did not produce a lifecycle tombstone")

        if not runtime.store.update_memory(
            memory_id,
            str(original["category"]),
            str(original["title"]),
            str(original["body"]),
            float(original["confidence"]),
        ):
            raise SystemExit("legacy organizer restore fixture could not restore its memory")
        restored = runtime.handle(
            "repair restored legacy organizer projection",
            request_token="organize-legacy-restore-republish",
        )
        restored_manifest = json.loads(
            str(_batch_rows(runtime)[0].get("projection_manifest") or "null")
        )
        if (
            not restored.tool_results[0].ok
            or restored.tool_results[0].metadata.get("organized_batch_recovered") is not True
            or restored_manifest[0].get("legacy_lifecycle_changed") is not None
            or not restored_manifest[0].get("path_display")
            or not restored_manifest[0].get("content_sha256")
        ):
            raise SystemExit("restored legacy organizer value did not regain projection evidence")
        converged = runtime.handle(
            "verify restored legacy organizer projection",
            request_token="organize-legacy-restore-converged",
        )
        if not converged.tool_results[0].ok:
            raise SystemExit("restored legacy organizer projection did not remain replayable")


def test_late_completion_failure_repairs_full_batch_without_duplicates() -> None:
    text = "\n".join(
        (
            "remember: Late organizer completion keeps its first mirror.",
            "decision: Late organizer completion | retain fixed IDs | repair projections",
            "task: Verify late organizer completion",
        )
    )
    with TemporaryDirectory(prefix="jarvis-organize-late-completion-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        handler = runtime.registry.get("organize_note").handler
        original_publication = runtime.store.organized_note_batch_publication

        @contextmanager
        def fail_after_publication(*args: Any, **kwargs: Any):
            with original_publication(*args, **kwargs) as publication:
                yield publication
                raise RuntimeError("representative organizer completion failure")

        runtime.store.organized_note_batch_publication = fail_after_publication  # type: ignore[method-assign]
        try:
            handler({"text": text})
        except RuntimeError as exc:
            if "completion failure" not in str(exc):
                raise
        else:
            raise SystemExit("late organizer completion failure was not injected")
        before = _domain_snapshot(runtime)
        before_vault = _vault_snapshot(runtime)
        if (
            len(before["tasks"]) != 1
            or len(before["memories"]) != 2
            or len(before["decisions"]) != 1
            or before["organized_note_batches"][0]["state"] != "pending"
        ):
            raise SystemExit("late organizer failure lost its atomically reserved domain rows")

        runtime.store.organized_note_batch_publication = original_publication  # type: ignore[method-assign]
        repaired = handler({"text": text})
        after = _domain_snapshot(runtime)
        after_vault = _vault_snapshot(runtime)
        if (
            not repaired.ok
            or repaired.metadata.get("organized_batch_recovered") is not True
            or after["organized_note_batches"][0]["state"] != "completed"
            or len(after["tasks"]) != 1
            or len(after["memories"]) != 2
            or len(after["decisions"]) != 1
        ):
            raise SystemExit("late organizer completion repair duplicated or lost domain state")
        if set(after_vault) != set(before_vault):
            raise SystemExit("late organizer repair created or removed a projection path")
        for path, content in before_vault.items():
            if not path.startswith("Tasks/") and after_vault[path] != content:
                raise SystemExit("late organizer repair rewrote an immutable projection")


def test_projection_tamper_after_verification_is_detected_on_replay() -> None:
    text = "goal: Organizer tamper guard | preserve external edits | fail closed"
    with TemporaryDirectory(prefix="jarvis-organize-tamper-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        handler = runtime.registry.get("organize_note").handler
        original_verify = runtime.vault.verify_organized_projection
        sentinel = "EXTERNAL-ORGANIZER-EDIT-MUST-SURVIVE"
        injected = [False]

        def tamper_after_verify(path: Path, expected_sha256: str) -> None:
            original_verify(path, expected_sha256)
            if not injected[0] and path.parent.name == "Projects":
                injected[0] = True
                path.write_text(path.read_text(encoding="utf-8") + f"\n{sentinel}\n", encoding="utf-8")

        runtime.vault.verify_organized_projection = tamper_after_verify  # type: ignore[method-assign]
        try:
            handler({"text": text})
        except RuntimeError as exc:
            if "could not be reconciled safely" not in str(exc):
                raise
        else:
            raise SystemExit("organizer did not detect the post-verification goal edit immediately")
        if _batch_rows(runtime)[0]["state"] != "completed":
            raise SystemExit("organizer post-verification fixture did not retain batch custody")

        runtime.vault.verify_organized_projection = original_verify  # type: ignore[method-assign]
        try:
            handler({"text": text})
        except RuntimeError as exc:
            if "changed before batch completion" not in str(exc):
                raise
        else:
            raise SystemExit("organizer replay falsely reported convergence for a tampered projection")
        goal_files = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if len(goal_files) != 1 or sentinel not in goal_files[0].read_text(encoding="utf-8"):
            raise SystemExit("organizer replay did not preserve the post-verification external edit")


def test_completed_manifest_rejects_newline_only_byte_tamper() -> None:
    text = "remember: Organizer byte-exact verification remains durable."
    with TemporaryDirectory(prefix="jarvis-organize-byte-exact-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        handler = runtime.registry.get("organize_note").handler
        created = handler({"text": text})
        memory_files = list(
            (runtime.vault.root_path / "Memory Tree" / "Facts").glob("*.md")
        )
        if not created.ok or len(memory_files) != 1:
            raise SystemExit("organizer byte-exact fixture did not publish")
        tampered = memory_files[0].read_bytes().replace(b"\n", b"\r\n")
        memory_files[0].write_bytes(tampered)
        try:
            handler({"text": text})
        except RuntimeError as exc:
            if "changed before batch completion" not in str(exc):
                raise
        else:
            raise SystemExit("organizer accepted newline-normalized false evidence")
        if memory_files[0].read_bytes() != tampered:
            raise SystemExit("organizer changed newline-tampered projection bytes")


def test_goal_update_waits_for_organizer_publication_fence() -> None:
    text = "goal: Organizer publication fence | keep goal state ordered | this month"
    with TemporaryDirectory(prefix="jarvis-organize-goal-fence-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        organize_handler = runtime.registry.get("organize_note").handler
        update_handler = runtime.registry.get("set_goal_status").handler
        original_write = runtime.vault.write_organized_goal
        published = threading.Event()
        update_attempting = threading.Event()
        update_result: list[ToolResult] = []

        def pause_goal() -> None:
            if not published.wait(timeout=5):
                return
            update_attempting.set()
            update_result.append(update_handler({"goal_id": 1, "status": "paused"}))

        def hold_after_goal_write(
            goal: Any,
            steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | tuple[str, ...] | None = None,
        ) -> tuple[Path, bool, str]:
            result = original_write(
                goal,
                steps,
                store_identity=store_identity,
                expected_prior_content_digest=expected_prior_content_digest,
            )
            published.set()
            if not update_attempting.wait(timeout=5):
                raise RuntimeError("goal update did not reach the publication fence")
            return result

        runtime.vault.write_organized_goal = hold_after_goal_write  # type: ignore[method-assign]
        updater = threading.Thread(target=pause_goal, daemon=True)
        updater.start()
        organized = organize_handler({"text": text})
        updater.join(timeout=10)
        if updater.is_alive() or not organized.ok or len(update_result) != 1 or not update_result[0].ok:
            raise SystemExit("goal update did not serialize behind organizer publication")
        goal = runtime.store.get_goal(1)
        goal_files = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if (
            goal is None
            or goal["status"] != "paused"
            or _batch_rows(runtime)[0]["state"] != "completed"
            or len(goal_files) != 1
            or 'status: "paused"' not in goal_files[0].read_text(encoding="utf-8")
        ):
            raise SystemExit("organizer publication fence left stale goal state or projection")


def test_goal_post_file_crash_retains_repair_evidence_across_source_advance() -> None:
    text = "goal: Organizer crash evidence | retain durable ownership | this month"
    with TemporaryDirectory(prefix="jarvis-organize-goal-crash-evidence-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        handler = runtime.registry.get("organize_note").handler
        original_write = runtime.vault.write_organized_goal

        def die_after_goal_write(
            goal: Any,
            steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | tuple[str, ...] | None = None,
        ) -> tuple[Path, bool, str]:
            original_write(
                goal,
                steps,
                store_identity=store_identity,
                expected_prior_content_digest=expected_prior_content_digest,
            )
            raise SimulatedOrganizerProcessDeath(
                "injected organizer death after goal publication"
            )

        runtime.vault.write_organized_goal = die_after_goal_write  # type: ignore[method-assign]
        try:
            handler({"text": text})
        except SimulatedOrganizerProcessDeath:
            pass
        else:
            raise SystemExit("organizer goal crash injection did not interrupt publication")
        crashed_job = runtime.store.get_goal_projection_job(1)
        goal_files = list((runtime.vault.root_path / "Projects").glob("*.md"))
        if (
            crashed_job is None
            or crashed_job["state"] != "pending"
            or type(crashed_job["prepared_content_digest"]) is not str
            or crashed_job["prepared_content_digest"]
            not in json.loads(crashed_job["prepared_content_digests"])
            or len(goal_files) != 1
            or hashlib.sha256(goal_files[0].read_bytes()).hexdigest()
            != crashed_job["prepared_content_digest"]
            or _batch_rows(runtime)[0]["state"] != "pending"
        ):
            raise SystemExit("organizer crash lost durable goal publication evidence")

        runtime.vault.write_organized_goal = original_write  # type: ignore[method-assign]
        advanced = runtime.store.set_goal_status_result(1, "paused")
        if not advanced.found or not advanced.changed:
            raise SystemExit("organizer crash fixture did not advance the goal source")
        outcome = reconcile_goal_projection(runtime.store, runtime.vault, 1)
        repaired_job = runtime.store.get_goal_projection_job(1)
        if (
            outcome.status not in {"completed", "superseded"}
            or repaired_job is None
            or repaired_job["state"] != "completed"
            or repaired_job["prepared_content_digest"] is not None
            or repaired_job["prepared_content_digests"] != "[]"
            or 'status: "paused"' not in goal_files[0].read_text(encoding="utf-8")
        ):
            raise SystemExit(
                "organizer crash evidence did not repair the advanced goal projection"
            )
        replay = handler({"text": text})
        batch = _batch_rows(runtime)[0]
        manifest = json.loads(batch["projection_manifest"])
        if (
            not replay.ok
            or batch["state"] != "completed"
            or len(manifest) != 1
            or manifest[0].get("role") != "goal"
            or manifest[0].get("legacy_lifecycle_changed") is not True
        ):
            raise SystemExit(
                "advanced organizer source did not complete with a lifecycle tombstone"
            )


def test_goal_post_file_crash_adoption_allows_batch_replay() -> None:
    text = "goal: Organizer adopted evidence | complete replay safely | this month"
    with TemporaryDirectory(prefix="jarvis-organize-goal-adoption-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        handler = runtime.registry.get("organize_note").handler
        original_write = runtime.vault.write_organized_goal

        def die_after_goal_write(
            goal: Any,
            steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | tuple[str, ...] | None = None,
        ) -> tuple[Path, bool, str]:
            original_write(
                goal,
                steps,
                store_identity=store_identity,
                expected_prior_content_digest=expected_prior_content_digest,
            )
            raise SimulatedOrganizerProcessDeath(
                "injected organizer death before batch completion"
            )

        runtime.vault.write_organized_goal = die_after_goal_write  # type: ignore[method-assign]
        try:
            handler({"text": text})
        except SimulatedOrganizerProcessDeath:
            pass
        else:
            raise SystemExit("organizer adoption crash injection did not interrupt")
        runtime.vault.write_organized_goal = original_write  # type: ignore[method-assign]
        goal_file = list((runtime.vault.root_path / "Projects").glob("*.md"))[0]
        bytes_before = goal_file.read_bytes()
        adopted = reconcile_goal_projection(runtime.store, runtime.vault, 1)
        replay = handler({"text": text})
        job = runtime.store.get_goal_projection_job(1)
        if (
            adopted.status not in {"completed", "superseded"}
            or not replay.ok
            or job is None
            or job["state"] != "completed"
            or _batch_rows(runtime)[0]["state"] != "completed"
            or goal_file.read_bytes() != bytes_before
        ):
            raise SystemExit(
                "adopted organizer goal evidence stranded or rewrote the pending batch"
            )


def test_goal_post_file_crash_external_edit_fails_closed() -> None:
    text = "goal: Organizer edited evidence | preserve external bytes | this month"
    sentinel = b"\nEXTERNAL-ORGANIZER-GOAL-EDIT\n"
    with TemporaryDirectory(prefix="jarvis-organize-goal-external-edit-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        handler = runtime.registry.get("organize_note").handler
        original_write = runtime.vault.write_organized_goal

        def die_after_goal_write(
            goal: Any,
            steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | tuple[str, ...] | None = None,
        ) -> tuple[Path, bool, str]:
            original_write(
                goal,
                steps,
                store_identity=store_identity,
                expected_prior_content_digest=expected_prior_content_digest,
            )
            raise SimulatedOrganizerProcessDeath("injected external-edit fixture crash")

        runtime.vault.write_organized_goal = die_after_goal_write  # type: ignore[method-assign]
        try:
            handler({"text": text})
        except SimulatedOrganizerProcessDeath:
            pass
        else:
            raise SystemExit("organizer external-edit crash injection did not interrupt")
        runtime.vault.write_organized_goal = original_write  # type: ignore[method-assign]
        goal_file = list((runtime.vault.root_path / "Projects").glob("*.md"))[0]
        edited = goal_file.read_bytes() + sentinel
        goal_file.write_bytes(edited)
        outcome = reconcile_goal_projection(runtime.store, runtime.vault, 1)
        try:
            handler({"text": text})
        except (FileExistsError, RuntimeError):
            pass
        else:
            raise SystemExit("organizer replay accepted externally edited crash evidence")
        if (
            outcome.status != "pending_error"
            or goal_file.read_bytes() != edited
            or _batch_rows(runtime)[0]["state"] != "pending"
        ):
            raise SystemExit("organizer crash recovery changed external goal bytes")


def test_goal_batch_completion_requires_durable_preparation() -> None:
    with TemporaryDirectory(prefix="jarvis-organize-goal-preparation-binding-") as temp:
        runtime = _runtime(Path(temp), {"text": "unused"})
        entries = (
            OrganizedNoteEntryRecord(
                "goal",
                goal=GoalRecord(
                    "Preparation binding",
                    "refuse unprepared completion",
                    "this month",
                ),
            ),
        )
        source_key = "organize-note:v1:" + "a" * 64
        runtime.store.reserve_organized_note_batch(source_key, entries)
        evidence: list[dict[str, Any]] = []
        try:
            with runtime.store.organized_note_batch_publication(
                source_key,
                entries,
                evidence,
            ):
                evidence.append(
                    {
                        "entry_index": 0,
                        "role": "goal",
                        "path_display": "Projects/Goal 1.md",
                        "content_sha256": "b" * 64,
                    }
                )
        except RuntimeError as exc:
            if "completion was lost" not in str(exc):
                raise
        else:
            raise SystemExit("unprepared organizer goal completed its durable ledger")
        job = runtime.store.get_goal_projection_job(1)
        if (
            job is None
            or job["state"] != "pending"
            or _batch_rows(runtime)[0]["state"] != "pending"
        ):
            raise SystemExit("unprepared organizer completion changed durable custody")


def test_task_projection_uses_sidecar_then_database_lock_order() -> None:
    text = "task: Organizer task lock order remains deadlock free"
    with TemporaryDirectory(prefix="jarvis-organize-task-lock-order-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        handler = runtime.registry.get("organize_note").handler
        sidecar_held = threading.Event()
        organizer_started = threading.Event()
        normal_result: list[bool] = []
        organize_result: list[ToolResult] = []
        store_identity = runtime.store.get_store_identity()

        def normal_task_sync() -> None:
            with runtime.vault.open_task_projection_publication():
                sidecar_held.set()
                if not organizer_started.wait(timeout=5):
                    return
                with runtime.store.open_task_mirror_snapshot() as tasks:
                    runtime.vault.write_tasks_with_evidence_under_publication_lock(
                        tasks,
                        store_identity=store_identity,
                    )
                normal_result.append(True)

        def run_organizer() -> None:
            if not sidecar_held.wait(timeout=5):
                return
            organizer_started.set()
            organize_result.append(handler({"text": text}))

        normal = threading.Thread(target=normal_task_sync, daemon=True)
        organizer = threading.Thread(target=run_organizer, daemon=True)
        normal.start()
        organizer.start()
        normal.join(timeout=10)
        organizer.join(timeout=10)
        if (
            normal.is_alive()
            or organizer.is_alive()
            or normal_result != [True]
            or len(organize_result) != 1
            or not organize_result[0].ok
            or _batch_rows(runtime)[0]["state"] != "completed"
        ):
            raise SystemExit("organizer and normal task projection entered a lock-order cycle")


def test_raw_content_token_and_path_receipt_privacy() -> None:
    content_sentinel = "PRIVATE-ORGANIZE-CONTENT-SENTINEL-7F2C"
    path_sentinel = "/\x55sers/example/PRIVATE-ORGANIZE-PATH-SENTINEL-9A1D"
    token_sentinel = "PRIVATE-ORGANIZE-REQUEST-TOKEN-SENTINEL-4B8E"
    text = f"remember: {content_sentinel}\nremember: {path_sentinel}"
    with TemporaryDirectory(prefix="jarvis-organize-privacy-") as temp:
        runtime = _runtime(Path(temp), {"text": text})
        result = runtime.handle("organize privacy fixture", request_token=token_sentinel)
        item = result.tool_results[0]
        if not item.ok or item.metadata.get("skipped") != 1:
            raise SystemExit(f"organize privacy fixture did not process only safe content: {item}")

        receipts = _receipt_rows(runtime)
        receipt_text = json.dumps(receipts, ensure_ascii=False, sort_keys=True, default=str)
        for private in (content_sentinel, path_sentinel, token_sentinel):
            if private in receipt_text:
                raise SystemExit(f"organize receipt stored raw private evidence: {private}")

        exposed = json.dumps(
            {
                "response": result.response,
                "runtime_metadata": result.metadata,
                "tool_output": item.output,
                "tool_metadata": item.metadata,
                "audits": _tool_run_rows(runtime),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        private_evidence = [
            token_sentinel,
            path_sentinel,
            auto_mutation_request_digest(token_sentinel),
        ]
        private_evidence.extend(
            str(receipt[key])
            for receipt in receipts
            for key in (
                "request_digest",
                "action_digest",
                "operation_digest",
                "uncertainty_digest",
                "run_token",
            )
            if receipt.get(key)
        )
        if any(private and private in exposed for private in private_evidence):
            raise SystemExit("organize runtime or audit exposed receipt-private evidence")
        if path_sentinel in item.output or path_sentinel in json.dumps(item.metadata, default=str):
            raise SystemExit("organize result exposed an unredacted local path")
        _assert_completed_receipt_audits(runtime, 1)


def main() -> None:
    test_exact_contract_typed_args_and_semantic_preflight()
    test_mixed_success_replay_convergence_and_decision_memory_truth()
    test_custom_preflight_result_cannot_forge_authorization_metadata()
    test_custom_preflight_result_requires_text_output()
    test_mirror_failure_blocks_replay_and_direct_retry_repairs_pending_batch()
    test_late_entry_trigger_failure_rolls_back_atomic_reservation()
    test_direct_concurrent_identical_calls_create_one_record_set()
    test_direct_concurrent_goal_calls_converge_after_completion_race()
    test_unowned_regular_destinations_are_not_overwritten()
    test_process_death_preserves_one_repairable_batch()
    test_completed_ledger_does_not_block_supported_memory_deletion()
    test_owned_target_foreign_key_upgrade_becomes_non_owning()
    test_legacy_completed_batch_rebuilds_missing_projection_manifest()
    test_legacy_manifest_repair_tombstones_deleted_target_without_resurrection()
    test_legacy_lifecycle_tombstone_self_heals_after_value_restore()
    test_late_completion_failure_repairs_full_batch_without_duplicates()
    test_projection_tamper_after_verification_is_detected_on_replay()
    test_completed_manifest_rejects_newline_only_byte_tamper()
    test_goal_update_waits_for_organizer_publication_fence()
    test_goal_post_file_crash_retains_repair_evidence_across_source_advance()
    test_goal_post_file_crash_adoption_allows_batch_replay()
    test_goal_post_file_crash_external_edit_fails_closed()
    test_goal_batch_completion_requires_durable_preparation()
    test_task_projection_uses_sidecar_then_database_lock_order()
    test_raw_content_token_and_path_receipt_privacy()
    print("Auto mutation organize rollout smoke passed")


if __name__ == "__main__":
    main()
