from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType
from typing import Any, Callable

import jarvis_v2.agent.runtime as runtime_module
from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.agent.verifier import Verifier
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore, auto_mutation_request_digest
from jarvis_v2.tools import tasks
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    AutoMutationContract,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    Tool,
    ToolRegistry,
)


AUTO_CONTRACT = AutoMutationContract(
    version=AUTO_MUTATION_CONTRACT_VERSION,
    effects=frozenset({AutoMutationEffect.LOCAL_DATABASE}),
    replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
    crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
)


class StaticPlanner:
    def __init__(self, factory: Callable[[str], Plan]):
        self.factory = factory

    def plan(self, user_input: str) -> Plan:
        return self.factory(user_input)


def _runtime(
    temp: str,
    tools: list[Tool],
    factory: Callable[[str], Plan],
) -> JarvisRuntime:
    runtime = object.__new__(JarvisRuntime)
    runtime.session_id = uuid.uuid4().hex[:8]
    runtime.storage_fallback = None
    runtime.store = MemoryStore(Path(temp) / "runtime.sqlite")
    runtime.store.init()
    runtime.registry = ToolRegistry()
    for tool in tools:
        runtime.registry.register(tool)
    runtime.planner = StaticPlanner(factory)
    runtime.executor = Executor(runtime.registry, PermissionPolicy())
    runtime.verifier = Verifier()
    runtime.chat = None
    runtime.vault = None
    return runtime


def _plan(*actions: PlannedAction) -> Plan:
    return Plan("Exercise runtime auto-mutation receipts.", list(actions), needs_model=False)


def _receipt_rows(store: MemoryStore) -> list[dict[str, Any]]:
    with store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM auto_mutation_receipts ORDER BY id")]


def _tool_run_rows(store: MemoryStore) -> list[dict[str, Any]]:
    with store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM tool_runs ORDER BY id")]


def test_concurrent_exact_once_replay_and_intentional_repeat() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-runtime-once-") as temp:
        calls = 0
        lock = threading.Lock()

        def mutate(_: dict[str, Any]) -> ToolResult:
            nonlocal calls
            with lock:
                calls += 1
            time.sleep(0.08)
            return ToolResult("mutate", True, "mutation committed")

        action = PlannedAction("mutate", {"value": 1}, "mutate once")
        runtime = _runtime(
            temp,
            [Tool("mutate", "fixture", RiskLevel.LOCAL_SAFE, mutate, "fixture", AUTO_CONTRACT)],
            lambda _text: _plan(action),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda _index: runtime.handle("concurrent fixture request", request_token="shared-request"),
                    range(2),
                )
            )
        if calls != 1 or sum(result.tool_results[0].ok for result in results) != 1:
            raise SystemExit(f"same-token concurrent handlers did not execute exactly once: calls={calls}, {results}")
        blocked = [result.tool_results[0] for result in results if not result.tool_results[0].ok]
        if len(blocked) != 1 or blocked[0].metadata.get("failure_kind") not in {
            "auto_mutation_running_replay",
            "auto_mutation_completed_replay",
        }:
            raise SystemExit(f"concurrent loser was not a bounded replay state: {blocked}")

        replay = runtime.handle("completed replay fixture", request_token="shared-request")
        if calls != 1 or replay.tool_results[0].metadata.get("failure_kind") != "auto_mutation_completed_replay":
            raise SystemExit(f"completed same-token replay invoked again: calls={calls}, {replay.tool_results}")
        repeated = runtime.handle("intentional repeat fixture", request_token="new-request")
        if calls != 2 or not repeated.tool_results[0].ok:
            raise SystemExit(f"new-token intentional repeat did not execute: calls={calls}, {repeated.tool_results}")
        derived_one = runtime.handle("derived token one")
        derived_two = runtime.handle("derived token two")
        if calls != 4 or not derived_one.tool_results[0].ok or not derived_two.tool_results[0].ok:
            raise SystemExit("per-turn derived request keys did not remain unique")

        runs = _tool_run_rows(runtime.store)
        successful = [row for row in runs if row["ok"] == 1]
        if len(successful) != 4 or any(row["approved"] != 0 for row in successful):
            raise SystemExit(f"successful finalized mutations were double-audited or authorized: {runs}")


def test_collision_and_multi_action_prepare_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-runtime-prepare-") as temp:
        calls: list[str] = []

        def handler(args: dict[str, Any]) -> ToolResult:
            calls.append(str(args["value"]))
            return ToolResult("mutate", True, "done")

        runtime = _runtime(
            temp,
            [Tool("mutate", "fixture", RiskLevel.LOCAL_SAFE, handler, "fixture", AUTO_CONTRACT)],
            lambda text: _plan(PlannedAction("mutate", {"value": text}, "dynamic mutation")),
        )
        first = runtime.handle("alpha", request_token="collision-key")
        collision = runtime.handle("beta", request_token="collision-key")
        if not first.tool_results[0].ok or calls != ["alpha"]:
            raise SystemExit(f"collision setup failed: {calls}, {first.tool_results}")
        if collision.tool_results[0].metadata.get("failure_kind") != "auto_mutation_request_collision":
            raise SystemExit(f"token collision did not fail closed: {collision.tool_results}")

    with TemporaryDirectory(prefix="jarvis-auto-runtime-batch-") as temp:
        calls = []

        def named(name: str) -> Callable[[dict[str, Any]], ToolResult]:
            def handler(_: dict[str, Any]) -> ToolResult:
                calls.append(name)
                return ToolResult(name, True, "done")

            return handler

        tools = [
            Tool("first", "fixture", RiskLevel.LOCAL_SAFE, named("first"), "fixture", AUTO_CONTRACT),
            Tool("second", "fixture", RiskLevel.LOCAL_SAFE, named("second"), "fixture", AUTO_CONTRACT),
        ]
        actions = [PlannedAction("first", {"id": 1}), PlannedAction("second", {"id": 2})]
        runtime = _runtime(temp, tools, lambda _text: _plan(*actions))
        blocker = runtime.store.prepare_auto_mutation_receipts("other-request", [("second", {"id": 2})])
        if blocker.status != "PREPARED":
            raise SystemExit("multi-action blocker setup failed")
        result = runtime.handle("blocked batch", request_token="batch-request")
        if calls or len(result.tool_results) != 2:
            raise SystemExit(f"blocked batch invoked a handler or lost result binding: {calls}, {result.tool_results}")
        if any(item.metadata.get("failure_kind") != "auto_mutation_unresolved_action" for item in result.tool_results):
            raise SystemExit(f"blocked batch did not return one bound result per action: {result.tool_results}")
        with runtime.store.connect() as conn:
            prepared_count = conn.execute(
                "SELECT COUNT(*) FROM auto_mutation_receipts WHERE request_digest = ?",
                (auto_mutation_request_digest("batch-request"),),
            ).fetchone()[0]
        if prepared_count != 0:
            raise SystemExit("all-or-none prepare left a partial batch")


def test_intra_batch_duplicate_real_and_fixture_actions_fail_before_execution() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-runtime-duplicate-fixture-") as temp:
        calls = 0
        contract = AutoMutationContract(
            version=AUTO_MUTATION_CONTRACT_VERSION,
            effects=frozenset({AutoMutationEffect.LOCAL_DATABASE}),
            replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
            crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            operation_key_builder=lambda args: {"record_id": args["record_id"]},
        )

        def mutate(_: dict[str, Any]) -> ToolResult:
            nonlocal calls
            calls += 1
            return ToolResult("mutate", True, "mutation committed")

        private_value = "PRIVATE-RUNTIME-DUPLICATE-CONTENT"
        actions = (
            PlannedAction("mutate", {"record_id": 7, "value": private_value}),
            PlannedAction("mutate", {"record_id": 7, "value": "different"}),
        )
        runtime = _runtime(
            temp,
            [Tool("mutate", "fixture", RiskLevel.LOCAL_SAFE, mutate, "fixture", contract)],
            lambda _text: _plan(*actions),
        )
        result = runtime.handle("duplicate fixture batch", request_token="duplicate-fixture")
        if calls != 0 or _receipt_rows(runtime.store):
            raise SystemExit("duplicate fixture operations invoked a handler or inserted receipts")
        if any(
            item.metadata.get("failure_kind") != "auto_mutation_duplicate_operation"
            or item.metadata.get("handler_invoked") is not False
            for item in result.tool_results
        ):
            raise SystemExit(f"duplicate fixture failure was not bound to every action: {result.tool_results}")
        if private_value in repr(result):
            raise SystemExit("duplicate fixture runtime failure exposed private action content")

    with TemporaryDirectory(prefix="jarvis-auto-runtime-duplicate-real-") as temp:
        store = MemoryStore(Path(temp) / "runtime.sqlite")
        store.init()
        vault = ObsidianVault(Path(temp) / "vault")
        vault.init()
        add_task = tasks.make_task_tools(store, vault)[0]
        real_tool = Tool(
            "add_task",
            "Capture a lightweight open task.",
            RiskLevel.LOCAL_SAFE,
            add_task,
            "tasks",
            AUTO_CONTRACT,
        )
        action = PlannedAction("add_task", {"body": "must exist zero times"})
        runtime = _runtime(temp, [real_tool], lambda _text: _plan(action, action))
        result = runtime.handle("duplicate real batch", request_token="duplicate-real")
        if _receipt_rows(runtime.store) or runtime.store.list_tasks(status=None):
            raise SystemExit("duplicate real operations inserted receipts or mutated task state")
        if any(
            item.metadata.get("failure_kind") != "auto_mutation_duplicate_operation"
            or item.metadata.get("handler_invoked") is not False
            for item in result.tool_results
        ):
            raise SystemExit(f"duplicate real failure was not content-private: {result.tool_results}")


def test_cross_request_running_and_uncertain_block() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-runtime-cross-") as temp:
        calls = 0

        def mutate(_: dict[str, Any]) -> ToolResult:
            nonlocal calls
            calls += 1
            return ToolResult("mutate", True, "done")

        action = PlannedAction("mutate", {"id": 8})
        runtime = _runtime(
            temp,
            [Tool("mutate", "fixture", RiskLevel.LOCAL_SAFE, mutate, "fixture", AUTO_CONTRACT)],
            lambda _text: _plan(action),
        )
        prepared = runtime.store.prepare_auto_mutation_receipts("owner-request", [("mutate", {"id": 8})])
        claim = runtime.store.claim_auto_mutation_receipt(int(prepared.receipts[0].receipt_id), "owner-request")
        running = runtime.handle("cross request running", request_token="cross-running")
        if calls or running.tool_results[0].metadata.get("failure_kind") != "auto_mutation_unresolved_action":
            raise SystemExit(f"cross-request running action did not block: {running.tool_results}")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET running_at = '2020-01-01T00:00:00Z' WHERE id = ?",
                (claim.receipt_id,),
            )
        uncertain = runtime.handle("cross request uncertain", request_token="cross-uncertain")
        if calls or uncertain.tool_results[0].metadata.get("failure_kind") != "auto_mutation_unresolved_action":
            raise SystemExit(f"cross-request uncertain action did not block: {uncertain.tool_results}")
        owner = _receipt_rows(runtime.store)[0]
        if owner["state"] != "uncertain" or owner["resolution"] != "stale_recovery":
            raise SystemExit(f"turn-boundary stale recovery did not stop as uncertain: {owner}")


def test_handler_error_and_completion_failure_become_uncertain() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-runtime-handler-error-") as temp:
        calls = 0

        def commit_then_error(_: dict[str, Any]) -> ToolResult:
            nonlocal calls
            calls += 1
            raise RuntimeError("private handler detail")

        action = PlannedAction("mutate", {"id": 9})
        runtime = _runtime(
            temp,
            [Tool("mutate", "fixture", RiskLevel.LOCAL_SAFE, commit_then_error, "fixture", AUTO_CONTRACT)],
            lambda _text: _plan(action),
        )
        failed = runtime.handle("commit then error", request_token="handler-error")
        replay = runtime.handle("commit then error replay", request_token="handler-error")
        rows = _receipt_rows(runtime.store)
        if calls != 1 or failed.tool_results[0].metadata.get("failure_kind") != "tool_error":
            raise SystemExit(f"handler error was not recorded as invoked once: {calls}, {failed.tool_results}")
        if rows[0]["state"] != "uncertain" or replay.tool_results[0].metadata.get("failure_kind") != "auto_mutation_outcome_uncertain":
            raise SystemExit(f"handler error did not stop replay as uncertain: {rows}, {replay.tool_results}")

    with TemporaryDirectory(prefix="jarvis-auto-runtime-audit-error-") as temp:
        calls = 0

        def mutate(_: dict[str, Any]) -> ToolResult:
            nonlocal calls
            calls += 1
            return ToolResult("mutate", True, "handler committed")

        runtime = _runtime(
            temp,
            [Tool("mutate", "fixture", RiskLevel.LOCAL_SAFE, mutate, "fixture", AUTO_CONTRACT)],
            lambda _text: _plan(PlannedAction("mutate", {"id": 10})),
        )

        def fail_completion(**_: Any) -> int:
            raise RuntimeError("private completion detail")

        runtime.store.complete_auto_mutation_receipt = fail_completion  # type: ignore[method-assign]
        failed = runtime.handle("audit completion failure", request_token="audit-error")
        replay = runtime.handle("audit completion failure replay", request_token="audit-error")
        rows = _receipt_rows(runtime.store)
        if calls != 1 or failed.tool_results[0].metadata.get("failure_kind") != "auto_mutation_completion_failed":
            raise SystemExit(f"audit completion failure did not fail closed: {calls}, {failed.tool_results}")
        if rows[0]["state"] != "uncertain" or replay.tool_results[0].metadata.get("failure_kind") != "auto_mutation_outcome_uncertain":
            raise SystemExit(f"audit completion failure was auto-retried: {rows}, {replay.tool_results}")
        if any(row["ok"] == 1 for row in _tool_run_rows(runtime.store)):
            raise SystemExit("failed atomic completion left a successful ordinary audit row")


def test_noneligible_unchanged_and_token_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-runtime-noneligible-") as temp:
        calls = 0

        def read_only(_: dict[str, Any]) -> ToolResult:
            nonlocal calls
            calls += 1
            return ToolResult("plain", True, "plain result")

        runtime = _runtime(
            temp,
            [Tool("plain", "fixture", RiskLevel.READ_ONLY, read_only, "fixture")],
            lambda _text: _plan(PlannedAction("plain")),
        )
        runtime.handle("plain one", request_token="same-noneligible-key")
        runtime.handle("plain two", request_token="same-noneligible-key")
        if calls != 2 or _receipt_rows(runtime.store):
            raise SystemExit("noneligible execution behavior changed")

    with TemporaryDirectory(prefix="jarvis-auto-runtime-privacy-") as temp:
        secret = "PRIVATE-RUNTIME-REQUEST-TOKEN-DO-NOT-LEAK"
        digest = auto_mutation_request_digest(secret)
        runtime = _runtime(
            temp,
            [
                Tool(
                    "mutate",
                    "fixture",
                    RiskLevel.LOCAL_SAFE,
                    lambda _: ToolResult("mutate", True, "safe output"),
                    "fixture",
                    AUTO_CONTRACT,
                )
            ],
            lambda _text: _plan(PlannedAction("mutate", {"public": True})),
        )
        result = runtime.handle("privacy fixture", request_token=secret)
        with runtime.store.connect() as conn:
            messages = [dict(row) for row in conn.execute("SELECT content, metadata FROM messages")]
        protected_surfaces = json.dumps(
            {
                "runtime_result": repr(result),
                "tool_runs": _tool_run_rows(runtime.store),
                "messages": messages,
            },
            sort_keys=True,
        )
        if secret in protected_surfaces or digest in protected_surfaces:
            raise SystemExit("request token or digest leaked into runtime, output, metadata, messages, or audit")
        invalid = runtime.handle("invalid key fixture", request_token="x" * 513)
        if invalid.tool_results[0].metadata.get("failure_kind") != "auto_mutation_request_key_invalid":
            raise SystemExit(f"oversized request key did not fail closed: {invalid.tool_results}")


def test_approved_rerun_bypasses_receipt_authority() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-runtime-approval-") as temp:
        calls = 0

        def mutate(_: dict[str, Any]) -> ToolResult:
            nonlocal calls
            calls += 1
            return ToolResult("mutate", True, "approved result")

        action = PlannedAction("mutate", {"id": 12})
        runtime = _runtime(
            temp,
            [Tool("mutate", "fixture", RiskLevel.LOCAL_SAFE, mutate, "fixture", AUTO_CONTRACT)],
            lambda _text: _plan(action),
        )

        def approved_plan(
            _self: JarvisRuntime,
            _user_input: str,
            _approval_id: int | None,
        ) -> tuple[Plan, str, None]:
            return _plan(action), "", None

        runtime._approved_rerun_plan = MethodType(approved_plan, runtime)  # type: ignore[method-assign]
        result = runtime.handle(
            "approved fixture",
            approved=True,
            approved_approval_id=77,
            request_token="must-not-enter-ledger",
        )
        runs = _tool_run_rows(runtime.store)
        if calls != 1 or not result.tool_results[0].ok or _receipt_rows(runtime.store):
            raise SystemExit("approved rerun entered or depended on the auto-mutation ledger")
        if len(runs) != 1 or runs[0]["approved"] != 1 or runs[0]["approval_id"] != 77:
            raise SystemExit(f"approved rerun audit proof was replaced by receipt authority: {runs}")


def main() -> None:
    original_suggest_command = runtime_module.suggest_command
    runtime_module.suggest_command = lambda _text: None
    try:
        test_concurrent_exact_once_replay_and_intentional_repeat()
        test_collision_and_multi_action_prepare_fail_closed()
        test_intra_batch_duplicate_real_and_fixture_actions_fail_before_execution()
        test_cross_request_running_and_uncertain_block()
        test_handler_error_and_completion_failure_become_uncertain()
        test_noneligible_unchanged_and_token_privacy()
        test_approved_rerun_bypasses_receipt_authority()
    finally:
        runtime_module.suggest_command = original_suggest_command
    print("Auto mutation runtime smoke passed")


if __name__ == "__main__":
    main()
