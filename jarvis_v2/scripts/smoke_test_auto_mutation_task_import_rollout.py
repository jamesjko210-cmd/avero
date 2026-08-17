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

import jarvis_v2.tools.tasks as tasks_module
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    TOOL_ARGUMENT_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    ToolArgumentType,
)


TOOL_NAME = "import_tasks_from_note"
PROCESS_DEATH_PATH = "Inbox/Process Death Import.md"
PROCESS_DEATH_TOKEN = "PRIVATE-TASK-IMPORT-PROCESS-DEATH-OWNER"


class StaticPlanner:
    def __init__(self, args: Any):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise the aggregate-wired note task import mutation.",
            [PlannedAction(TOOL_NAME, self.args, "task import rollout smoke")],
            needs_model=False,
        )


class MultiStaticPlanner:
    def __init__(self, actions: list[dict[str, Any]]):
        self.actions = actions

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise multi-action task import receipt custody.",
            [
                PlannedAction(TOOL_NAME, args, "task import batch rollout smoke")
                for args in self.actions
            ],
            needs_model=False,
        )


def _runtime(root: Path, args: Any) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(args)
    return runtime


def _write_note(runtime: JarvisRuntime, relative_path: str, body: str) -> Path:
    path = runtime.vault.root_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


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
            f"expected {expected} task-import receipt/audit pairs: "
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
            raise SystemExit(f"task-import receipt lost ordinary audit linkage: {receipt} / {run}")
        linked.add(int(run["id"]))
    if len(linked) != expected:
        raise SystemExit("task-import receipts reused one ordinary audit row")


def _assert_receipt_privacy(
    runtime: JarvisRuntime,
    *,
    private_values: list[str],
) -> None:
    receipts = _receipt_rows(runtime)
    serialized = json.dumps(receipts, ensure_ascii=False, sort_keys=True, default=str)
    for private in private_values:
        if private and private in serialized:
            raise SystemExit(f"task-import receipt exposed private input: {private!r}")
    for receipt in receipts:
        for key in ("request_digest", "action_digest", "operation_digest"):
            value = receipt.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise SystemExit(f"task-import receipt lost a bounded digest: {receipt}")
    if str(runtime.vault.root_path) in serialized or str(runtime.store.db_path.parent) in serialized:
        raise SystemExit("task-import receipts exposed an absolute temporary path")


def _assert_no_mutation_boundary(
    runtime: JarvisRuntime,
    *,
    label: str,
    projection_before: bytes | None,
) -> None:
    if _task_rows(runtime) or _receipt_rows(runtime):
        raise SystemExit(f"{label} crossed the database or receipt boundary")
    if _projection_bytes(runtime) != projection_before:
        raise SystemExit(f"{label} changed the open-task projection")


def _install_projection_failure(
    runtime: JarvisRuntime,
    callback: Callable[..., Any],
) -> None:
    runtime.vault.sync_open_tasks = callback  # type: ignore[method-assign]
    runtime.vault.sync_open_tasks_with_evidence = callback  # type: ignore[method-assign]


def _oversized_note_chars() -> int:
    return tasks_module.MAX_TASK_IMPORT_NOTE_CHARS + 1


def _crash_after_task_import_database_commit(root_text: str) -> None:
    runtime = _runtime(
        Path(root_text),
        {"path": PROCESS_DEATH_PATH, "priority": "normal"},
    )
    _write_note(
        runtime,
        PROCESS_DEATH_PATH,
        "- [ ] Process death import first\n- [ ] Process death import second\n",
    )

    def exit_before_projection(*_args: Any, **_kwargs: Any) -> Any:
        if len(_task_rows(runtime)) == 2:
            os._exit(73)
        os._exit(74)

    _install_projection_failure(runtime, exit_before_projection)
    runtime.handle(
        "import tasks before representative process death",
        request_token=PROCESS_DEATH_TOKEN,
    )
    os._exit(75)


def test_exact_contract_schema_and_target_only_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-task-import-contract-") as temp:
        runtime = _runtime(
            Path(temp),
            {"path": "Projects/Contract.md", "priority": "normal"},
        )
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
            or contract.definite_no_effect_failure_reasons
            != frozenset(
                {
                    "note_not_found",
                    "note_not_regular",
                    "note_read_failed",
                    "note_too_large",
                    "too_many_tasks",
                    "unsafe_path",
                }
            )
        ):
            raise SystemExit(f"{TOOL_NAME} auto-mutation contract drifted: {tool}")

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
            or shape != (("path", string, True), ("priority", string, False))
        ):
            raise SystemExit(f"{TOOL_NAME} strict argument contract drifted: {arguments}")

        builder = contract.operation_key_builder
        equivalent = [
            builder({"path": "Projects/ＦＯＯ", "priority": "low"}),
            builder({"path": "projects/foo.md", "priority": "normal"}),
            builder({"path": "PROJECTS/FOO.MD", "priority": "high"}),
        ]
        canonical = {
            json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            for item in equivalent
        }
        changed = builder({"path": "projects/other.md", "priority": "low"})
        if len(canonical) != 1 or changed == equivalent[0]:
            raise SystemExit(
                "task-import operation identity lost target-only lexical NFKC/casefold/.md semantics"
            )
        serialized_key = next(iter(canonical))
        if "priority" in serialized_key or any(value in serialized_key for value in ('"low"', '"high"')):
            raise SystemExit("task-import operation identity retained priority")


def test_typed_and_semantic_preflight_precede_receipts_and_writes() -> None:
    typed_cases: list[Any] = [
        {},
        {"path": None},
        {"path": 3},
        {"path": True},
        {"path": []},
        {"path": "Inbox/Typed.md", "priority": None},
        {"path": "Inbox/Typed.md", "priority": 1},
        {"path": "Inbox/Typed.md", "priority": False},
        {"path": "Inbox/Typed.md", "unknown": "rejected"},
        None,
        [],
    ]
    for index, args in enumerate(typed_cases):
        with TemporaryDirectory(prefix="jarvis-task-import-typed-") as temp:
            runtime = _runtime(Path(temp), args)
            projection_before = _projection_bytes(runtime)
            calls = _install_counted_handler(runtime)
            result = runtime.handle(
                "invalid typed task import",
                request_token=f"task-import-typed-{index}",
            )
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
            ):
                raise SystemExit(f"task-import typed case {index} drifted: {item}")
            _assert_no_mutation_boundary(
                runtime,
                label=f"task-import typed case {index}",
                projection_before=projection_before,
            )

    semantic_cases: list[
        tuple[dict[str, Any], Callable[[JarvisRuntime], None] | None]
    ] = [
        ({"path": "   ", "priority": "normal"}, None),
        ({"path": "Inbox/Invalid Priority.md", "priority": "urgent"}, None),
        ({"path": "../Outside.md", "priority": "normal"}, None),
        ({"path": "Inbox/Missing.md", "priority": "normal"}, None),
        ({"path": "x" * (tasks_module.MAX_NOTE_PATH_CHARS + 1), "priority": "normal"}, None),
        (
            {"path": "Inbox/Alias.md", "priority": "normal"},
            lambda runtime: (
                _write_note(runtime, "Inbox/Real.md", "- [ ] symlink victim\n"),
                (runtime.vault.root_path / "Inbox" / "Alias.md").symlink_to("Real.md"),
            ),
        ),
    ]
    if hasattr(os, "mkfifo"):
        def make_fifo(runtime: JarvisRuntime) -> None:
            path = runtime.vault.root_path / "Inbox" / "Pipe.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            os.mkfifo(path)

        semantic_cases.append(
            ({"path": "Inbox/Pipe.md", "priority": "normal"}, make_fifo)
        )

    for index, (args, setup) in enumerate(semantic_cases):
        with TemporaryDirectory(prefix="jarvis-task-import-semantic-") as temp:
            runtime = _runtime(Path(temp), args)
            if setup is not None:
                setup(runtime)
            projection_before = _projection_bytes(runtime)
            calls = _install_counted_handler(runtime)
            result = runtime.handle(
                "invalid semantic task import",
                request_token=f"task-import-semantic-{index}",
            )
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind")
                != "auto_mutation_semantic_preflight_rejected"
                or item.metadata.get("handler_invoked") is not False
                or item.metadata.get("state_changed") is not False
                or calls[0] != 0
            ):
                raise SystemExit(f"task-import semantic case {index} drifted: {item}")
            _assert_no_mutation_boundary(
                runtime,
                label=f"task-import semantic case {index}",
                projection_before=projection_before,
            )
            exposed = json.dumps(
                {"response": result.response, "output": item.output, "metadata": item.metadata},
                ensure_ascii=False,
                default=str,
            )
            if str(runtime.vault.root_path) in exposed or str(runtime.store.db_path.parent) in exposed:
                raise SystemExit("task-import preflight exposed an absolute temporary path")


def test_control_character_path_is_rejected_without_audit_injection() -> None:
    hostile_path = "Inbox/Visible\n\x1b[31mFAKE\u2028FORGED\u2029LINE.md"
    args = {"path": hostile_path, "priority": "normal"}
    with TemporaryDirectory(prefix="jarvis-task-import-control-path-") as temp:
        runtime = _runtime(Path(temp), args)
        projection_before = _projection_bytes(runtime)
        calls = _install_counted_handler(runtime)
        result = runtime.handle(
            "reject control-character task import path",
            request_token="task-import-control-path-owner",
        )
        item = result.tool_results[0]
        if (
            item.ok
            or item.metadata.get("failure_kind")
            != "auto_mutation_semantic_preflight_rejected"
            or item.metadata.get("handler_invoked") is not False
            or calls[0] != 0
        ):
            raise SystemExit(f"control-character task import did not fail preflight: {item}")
        _assert_no_mutation_boundary(
            runtime,
            label="control-character task import",
            projection_before=projection_before,
        )
        outward_values = [item.output, *[str(value) for value in item.metadata.values()]]
        for row in _tool_run_rows(runtime):
            outward_values.extend([str(row.get("output") or ""), str(row.get("metadata") or "")])
        if any(
            "\n" in value or "\x1b" in value or "\u2028" in value or "\u2029" in value
            for value in outward_values
        ):
            raise SystemExit("control-character task path reached output or audit metadata")


def test_nonregular_note_reason_matches_preview_direct_and_runtime_import() -> None:
    args = {"path": "Inbox/Folder.md", "priority": "normal"}
    with TemporaryDirectory(prefix="jarvis-task-import-nonregular-parity-") as temp:
        runtime = _runtime(Path(temp), args)
        folder = runtime.vault.root_path / "Inbox" / "Folder.md"
        folder.mkdir(parents=True)
        preview = runtime.registry.get("preview_tasks_from_note").handler(
            {"path": args["path"]}
        )
        direct = runtime.registry.get(TOOL_NAME).handler(args)
        routed = runtime.handle(
            "runtime import of a nonregular note",
            request_token="task-import-nonregular-parity-owner",
        )
        routed_item = routed.tool_results[0]
        if (
            preview.ok
            or direct.ok
            or routed_item.ok
            or preview.metadata.get("reason") != "note_not_regular"
            or direct.metadata.get("reason") != "note_not_regular"
            or routed_item.metadata.get("reason") != "note_not_regular"
            or routed_item.metadata.get("handler_invoked") is not False
            or _receipt_rows(runtime)
            or _task_rows(runtime)
        ):
            raise SystemExit(
                "nonregular note classification diverged across preview/direct/runtime import: "
                f"{preview} / {direct} / {routed_item}"
            )


def test_post_claim_definite_no_effect_refusals_release_custody() -> None:
    with TemporaryDirectory(prefix="jarvis-task-import-oversized-release-") as temp:
        args = {"path": "Inbox/Oversized.md", "priority": "normal"}
        runtime = _runtime(Path(temp), args)
        note_path = _write_note(
            runtime,
            "Inbox/Oversized.md",
            "x" * _oversized_note_chars(),
        )
        preview = runtime.registry.get("preview_tasks_from_note").handler(
            {"path": "Inbox/Oversized.md"}
        )
        if preview.ok or preview.metadata.get("reason") != "note_too_large":
            raise SystemExit(f"oversized note preview disagreed with import boundary: {preview}")
        token = "task-import-oversized-release"
        result = runtime.handle("oversized task import", request_token=token)
        item = result.tool_results[0]
        if (
            item.ok
            or item.metadata.get("reason") != "note_too_large"
            or item.metadata.get("auto_mutation_effects_started") is not False
            or item.metadata.get("auto_mutation_definite_no_effect") is not True
            or item.metadata.get("auto_mutation_outcome_uncertain") is not False
            or _receipt_rows(runtime)
            or _task_rows(runtime)
        ):
            raise SystemExit(f"oversized pre-effect refusal retained mutation custody: {item}")
        note_path.write_text("- [ ] retry after bounded refusal\n", encoding="utf-8")
        retried = runtime.handle("retry bounded task import", request_token=token)
        if not retried.tool_results[0].ok or len(_task_rows(runtime)) != 1:
            raise SystemExit("definite-no-effect oversized refusal did not release same-token retry")

    with TemporaryDirectory(prefix="jarvis-task-import-drift-release-") as temp:
        args = {"path": "Inbox/Drift.md", "priority": "normal"}
        runtime = _runtime(Path(temp), args)
        note_path = _write_note(runtime, "Inbox/Drift.md", "- [ ] original drift task\n")
        original_prepare = runtime.store.prepare_auto_mutation_receipts

        def prepare_then_remove(*prepare_args: Any, **prepare_kwargs: Any):
            prepared = original_prepare(*prepare_args, **prepare_kwargs)
            note_path.unlink()
            return prepared

        runtime.store.prepare_auto_mutation_receipts = prepare_then_remove  # type: ignore[method-assign]
        token = "task-import-source-drift-release"
        try:
            result = runtime.handle("task import with source drift", request_token=token)
        finally:
            runtime.store.prepare_auto_mutation_receipts = original_prepare  # type: ignore[method-assign]
        item = result.tool_results[0]
        if (
            item.ok
            or item.metadata.get("reason") != "note_not_found"
            or item.metadata.get("auto_mutation_effects_started") is not False
            or item.metadata.get("auto_mutation_definite_no_effect") is not True
            or _receipt_rows(runtime)
            or _task_rows(runtime)
        ):
            raise SystemExit(f"source-drift pre-effect refusal retained mutation custody: {item}")
        _write_note(runtime, "Inbox/Drift.md", "- [ ] replacement drift task\n")
        retried = runtime.handle("retry task import after source drift", request_token=token)
        rows = _task_rows(runtime)
        if not retried.tool_results[0].ok or [row["body"] for row in rows] != ["replacement drift task"]:
            raise SystemExit("source-drift refusal did not release same-token retry")


def test_definite_no_effect_requires_explicit_prewrite_marker() -> None:
    args = {"path": "Inbox/Marker Required.md", "priority": "normal"}
    with TemporaryDirectory(prefix="jarvis-task-import-marker-required-") as temp:
        runtime = _runtime(Path(temp), args)
        _write_note(runtime, "Inbox/Marker Required.md", "- [ ] marker required task\n")
        tool = runtime.registry.get(TOOL_NAME)

        def unmarked_failure(_args: dict[str, Any]) -> ToolResult:
            return ToolResult(
                TOOL_NAME,
                False,
                "unmarked deterministic-looking failure",
                {"reason": "note_not_found", "state_changed": False},
            )

        runtime.registry._tools[TOOL_NAME] = replace(tool, handler=unmarked_failure)
        result = runtime.handle(
            "unmarked task-import failure",
            request_token="task-import-marker-required-owner",
        )
        item = result.tool_results[0]
        receipts = _receipt_rows(runtime)
        if (
            item.metadata.get("auto_mutation_outcome_uncertain") is not True
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
        ):
            raise SystemExit("reason alone released mutation custody without a pre-effect marker")


def test_empty_note_postread_symlink_drift_uses_lexical_display() -> None:
    args = {"path": "Inbox/Empty Drift.md", "priority": "normal"}
    with TemporaryDirectory(prefix="jarvis-task-import-empty-drift-") as temp:
        root = Path(temp)
        runtime = _runtime(root, args)
        note_path = _write_note(runtime, "Inbox/Empty Drift.md", "# No open tasks\n")
        outside = root / "outside-empty-drift.md"
        outside.write_text("outside remains unchanged\n", encoding="utf-8")
        original_read = runtime.vault.read_note_bounded
        reads = [0]

        def read_then_swap(*read_args: Any, **read_kwargs: Any):
            content = original_read(*read_args, **read_kwargs)
            reads[0] += 1
            if reads[0] == 2:
                note_path.unlink()
                note_path.symlink_to(outside)
            return content

        runtime.vault.read_note_bounded = read_then_swap  # type: ignore[method-assign]
        try:
            result = runtime.handle(
                "import empty note during post-read source drift",
                request_token="task-import-empty-drift-owner",
            )
        finally:
            runtime.vault.read_note_bounded = original_read  # type: ignore[method-assign]
        item = result.tool_results[0]
        receipts = _receipt_rows(runtime)
        if (
            not item.ok
            or item.metadata.get("imported") != 0
            or item.metadata.get("skipped") != 0
            or len(receipts) != 1
            or receipts[0]["state"] != "completed"
            or _task_rows(runtime)
            or outside.read_text(encoding="utf-8") != "outside remains unchanged\n"
        ):
            raise SystemExit(f"empty-note post-read drift corrupted receipt custody: {item}")


def test_stale_prepared_receipt_cleanup_releases_unstarted_import() -> None:
    args = {"path": "Inbox/Stale Prepared.md", "priority": "normal"}
    sibling_args = {"path": "Inbox/Stale Prepared Sibling.md", "priority": "high"}
    with TemporaryDirectory(prefix="jarvis-task-import-stale-prepared-") as temp:
        root = Path(temp)
        runtime = _runtime(root, args)
        _write_note(runtime, "Inbox/Stale Prepared.md", "- [ ] stale prepared retry task\n")
        _write_note(
            runtime,
            "Inbox/Stale Prepared Sibling.md",
            "- [ ] stale prepared sibling retry task\n",
        )
        contract = runtime.registry.get(TOOL_NAME).auto_mutation_contract
        if contract is None or contract.operation_key_builder is None:
            raise SystemExit("stale-prepared fixture lost operation identity")
        owner = "task-import-stale-prepared-owner"
        prepared = runtime.store.prepare_auto_mutation_receipts(
            owner,
            [
                (TOOL_NAME, args, contract.operation_key_builder(dict(args))),
                (
                    TOOL_NAME,
                    sibling_args,
                    contract.operation_key_builder(dict(sibling_args)),
                ),
            ],
        )
        if prepared.status != "PREPARED":
            raise SystemExit(f"could not create stale prepared fixture: {prepared}")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET prepared_at = ?, updated_at = ?",
                ("2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z"),
            )

        recovered = _runtime(root, args)
        if _receipt_rows(recovered):
            raise SystemExit("startup did not release untouched stale prepared receipt custody")
        result = recovered.handle(
            "fresh request after stale prepared cleanup",
            request_token="task-import-stale-prepared-fresh",
        )
        if not result.tool_results[0].ok or len(_task_rows(recovered)) != 1:
            raise SystemExit("stale prepared cleanup did not release the operation target")
        recovered.planner = StaticPlanner(sibling_args)
        sibling = recovered.handle(
            "fresh sibling request after stale prepared cleanup",
            request_token="task-import-stale-prepared-sibling-fresh",
        )
        if not sibling.tool_results[0].ok or len(_task_rows(recovered)) != 2:
            raise SystemExit("stale prepared cleanup did not release the whole request batch")

    first_args = {"path": "Inbox/Mixed Running.md", "priority": "normal"}
    second_args = {"path": "Inbox/Mixed Prepared.md", "priority": "normal"}
    with TemporaryDirectory(prefix="jarvis-task-import-mixed-prepared-") as temp:
        root = Path(temp)
        runtime = _runtime(root, first_args)
        _write_note(runtime, "Inbox/Mixed Running.md", "- [ ] must remain unexecuted\n")
        _write_note(runtime, "Inbox/Mixed Prepared.md", "- [ ] released prepared sibling\n")
        contract = runtime.registry.get(TOOL_NAME).auto_mutation_contract
        if contract is None or contract.operation_key_builder is None:
            raise SystemExit("mixed stale-prepared fixture lost operation identity")
        owner = "task-import-mixed-prepared-owner"
        prepared = runtime.store.prepare_auto_mutation_receipts(
            owner,
            [
                (TOOL_NAME, first_args, contract.operation_key_builder(dict(first_args))),
                (TOOL_NAME, second_args, contract.operation_key_builder(dict(second_args))),
            ],
        )
        if prepared.status != "PREPARED" or any(
            receipt.receipt_id is None for receipt in prepared.receipts
        ):
            raise SystemExit(f"could not create mixed stale-prepared fixture: {prepared}")
        claim = runtime.store.claim_auto_mutation_receipt(
            int(prepared.receipts[0].receipt_id),
            owner,
        )
        if claim.status != "CLAIMED":
            raise SystemExit(f"could not claim mixed stale-prepared owner: {claim}")
        with runtime.store.connect() as conn:
            conn.execute(
                """
                UPDATE auto_mutation_receipts
                SET prepared_at = ?, running_at = CASE WHEN state = 'running' THEN ? ELSE running_at END,
                    updated_at = ?
                """,
                (
                    "2000-01-01T00:00:00Z",
                    "2000-01-01T00:00:00Z",
                    "2000-01-01T00:00:00Z",
                ),
            )

        recovered = _runtime(root, second_args)
        remaining = _receipt_rows(recovered)
        if (
            len(remaining) != 2
            or [row["state"] for row in remaining] != ["uncertain", "prepared"]
            or [row["action_index"] for row in remaining] != [0, 1]
        ):
            raise SystemExit(f"mixed startup cleanup damaged request-batch shape: {remaining}")
        result = recovered.handle(
            "attempt previously prepared sibling with a fresh request",
            request_token="task-import-mixed-prepared-fresh",
        )
        if result.tool_results[0].ok or _task_rows(recovered) or len(_receipt_rows(recovered)) != 2:
            raise SystemExit("mixed request batch did not remain conservatively fenced")


def test_multi_action_no_effect_failure_preserves_receipt_batch_shape() -> None:
    actions = [
        {"path": "Inbox/Missing Batch One.md", "priority": "normal"},
        {"path": "Inbox/Missing Batch Two.md", "priority": "high"},
    ]
    with TemporaryDirectory(prefix="jarvis-task-import-multi-no-effect-") as temp:
        runtime = _runtime(Path(temp), actions[0])
        _write_note(runtime, actions[0]["path"], "- [ ] first batch task\n")
        _write_note(runtime, actions[1]["path"], "- [ ] second batch task\n")
        runtime.planner = MultiStaticPlanner(actions)
        original_read = runtime.vault.read_note_bounded
        reads = [0]

        def disappear_after_preflight(*read_args: Any, **read_kwargs: Any):
            reads[0] += 1
            if reads[0] <= len(actions):
                return original_read(*read_args, **read_kwargs)
            return None

        runtime.vault.read_note_bounded = disappear_after_preflight  # type: ignore[method-assign]
        result = runtime.handle(
            "import two missing notes as one request",
            request_token="task-import-multi-no-effect-owner",
        )
        receipts = _receipt_rows(runtime)
        if (
            len(result.tool_results) != 2
            or any(item.ok for item in result.tool_results)
            or len(receipts) != 2
            or [row["action_index"] for row in receipts] != [0, 1]
            or any(row["state"] != "uncertain" for row in receipts)
            or _task_rows(runtime)
        ):
            raise SystemExit(f"multi-action no-effect failure damaged receipt batch: {receipts}")


def test_unicode_equivalent_tasks_dedupe_within_atomic_batch() -> None:
    args = {"path": "Inbox/Unicode Identity.md", "priority": "normal"}
    with TemporaryDirectory(prefix="jarvis-task-import-unicode-") as temp:
        runtime = _runtime(Path(temp), args)
        _write_note(
            runtime,
            "Inbox/Unicode Identity.md",
            "- [ ] Café review\n- [ ] Cafe\u0301 review\n",
        )
        preview = runtime.registry.get("preview_tasks_from_note").handler(
            {"path": "Inbox/Unicode Identity.md"}
        )
        if (
            not preview.ok
            or preview.metadata.get("open_checkboxes") != 1
            or preview.metadata.get("new_tasks") != 1
            or preview.metadata.get("duplicates") != 0
        ):
            raise SystemExit(f"Unicode-equivalent preview identity diverged: {preview}")
        result = runtime.handle(
            "import Unicode-equivalent task identities",
            request_token="task-import-unicode-owner",
        )
        item = result.tool_results[0]
        if (
            not item.ok
            or item.metadata.get("imported") != 1
            or item.metadata.get("skipped") != 0
            or len(_task_rows(runtime)) != 1
        ):
            raise SystemExit(f"Unicode-equivalent task identities duplicated: {item}")


def test_import_body_boundary_precedes_identity_and_insertion() -> None:
    path = "Inbox/Task Body Boundary.md"
    args = {"path": path, "priority": "normal"}
    exact_body = "a" * tasks_module.MAX_TASK_BODY_CHARS
    oversized_body = "b" * (tasks_module.MAX_TASK_BODY_CHARS + 1)
    collision_prefix = "c" * tasks_module.MAX_TASK_BODY_CHARS
    collision_a = collision_prefix + "x"
    collision_b = collision_prefix + "y"
    expected_oversized = (
        "b" * (tasks_module.MAX_TASK_BODY_CHARS - 1) + "…"
    )
    expected_collision = (
        "c" * (tasks_module.MAX_TASK_BODY_CHARS - 1) + "…"
    )
    with TemporaryDirectory(prefix="jarvis-task-import-body-boundary-") as temp:
        runtime = _runtime(Path(temp), args)
        _write_note(
            runtime,
            path,
            "".join(
                f"- [ ] {body}\n"
                for body in (
                    exact_body,
                    oversized_body,
                    collision_a,
                    collision_b,
                )
            ),
        )
        preview = runtime.registry.get("preview_tasks_from_note").handler(
            {"path": path}
        )
        result = runtime.handle(
            "import bounded checkbox bodies",
            request_token="task-import-body-boundary-owner",
        )
        item = result.tool_results[0]
        rows = _task_rows(runtime)
        expected = [exact_body, expected_oversized, expected_collision]
        if (
            not preview.ok
            or preview.metadata.get("open_checkboxes") != 3
            or not item.ok
            or item.metadata.get("imported") != 3
            or item.metadata.get("skipped") != 0
            or [row["body"] for row in rows] != expected
            or any(len(row["body"]) > tasks_module.MAX_TASK_BODY_CHARS for row in rows)
        ):
            raise SystemExit(
                "task-import body normalization did not precede identity/insertion: "
                f"{preview} / {item} / {rows}"
            )


def test_task_candidate_cap_preflight_allows_200_and_refuses_201() -> None:
    allowed_path = "Inbox/Exactly 200 Tasks.md"
    with TemporaryDirectory(prefix="jarvis-task-import-cap-allowed-") as temp:
        runtime = _runtime(
            Path(temp),
            {"path": allowed_path, "priority": "normal"},
        )
        allowed_bodies = [f"allowed import task {index:03d}" for index in range(200)]
        _write_note(
            runtime,
            allowed_path,
            "".join(f"- [ ] {body}\n" for body in allowed_bodies),
        )
        result = runtime.handle(
            "import exactly two hundred tasks",
            request_token="task-import-cap-200-owner",
        )
        item = result.tool_results[0]
        if (
            not item.ok
            or item.metadata.get("imported") != tasks_module.MAX_TASK_IMPORT_CANDIDATES
            or [row["body"] for row in _task_rows(runtime)] != allowed_bodies
        ):
            raise SystemExit(f"exactly 200 task candidates were not allowed: {item}")
        _assert_completed_receipt_audits(runtime, 1)

    refused_path = "Inbox/Private 201 Tasks.md"
    private_marker = "PRIVATE-CAP-REFUSAL-BODY"
    with TemporaryDirectory(prefix="jarvis-task-import-cap-refused-") as temp:
        runtime = _runtime(
            Path(temp),
            {"path": refused_path, "priority": "high"},
        )
        refused_bodies = [f"{private_marker}-{index:03d}" for index in range(201)]
        _write_note(
            runtime,
            refused_path,
            "".join(f"- [ ] {body}\n" for body in refused_bodies),
        )
        projection_before = _projection_bytes(runtime)
        calls = _install_counted_handler(runtime)
        result = runtime.handle(
            "refuse more than two hundred tasks in preflight",
            request_token="PRIVATE-TASK-IMPORT-CAP-201-OWNER",
        )
        item = result.tool_results[0]
        if (
            item.ok
            or item.metadata.get("failure_kind")
            != "auto_mutation_semantic_preflight_rejected"
            or item.metadata.get("reason") != "too_many_tasks"
            or item.metadata.get("open_task_candidates") != 201
            or item.metadata.get("max_open_task_candidates")
            != tasks_module.MAX_TASK_IMPORT_CANDIDATES
            or item.metadata.get("handler_invoked") is not False
            or item.metadata.get("auto_mutation_effects_started") is not False
            or calls[0] != 0
        ):
            raise SystemExit(f"201 task candidates did not fail semantic preflight: {item}")
        _assert_no_mutation_boundary(
            runtime,
            label="201-candidate semantic preflight refusal",
            projection_before=projection_before,
        )
        outward = json.dumps(
            {
                "output": item.output,
                "metadata": item.metadata,
                "tool_runs": _tool_run_rows(runtime),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if private_marker in outward or str(Path(temp)) in outward:
            raise SystemExit("task-cap semantic refusal exposed private task or local path data")


def test_task_candidate_cap_post_preflight_race_releases_receipt() -> None:
    path = "Inbox/Private Candidate Race.md"
    args = {"path": path, "priority": "normal"}
    private_marker = "PRIVATE-CAP-RACE-BODY"
    initial_bodies = [f"initial race task {index:03d}" for index in range(200)]
    raced_bodies = [f"{private_marker}-{index:03d}" for index in range(201)]
    with TemporaryDirectory(prefix="jarvis-task-import-cap-race-") as temp:
        runtime = _runtime(Path(temp), args)
        note_path = _write_note(
            runtime,
            path,
            "".join(f"- [ ] {body}\n" for body in initial_bodies),
        )
        projection_before = _projection_bytes(runtime)
        calls = _install_counted_handler(runtime)
        original_prepare = runtime.store.prepare_auto_mutation_receipts

        def prepare_then_expand(*prepare_args: Any, **prepare_kwargs: Any):
            prepared = original_prepare(*prepare_args, **prepare_kwargs)
            note_path.write_text(
                "".join(f"- [ ] {body}\n" for body in raced_bodies),
                encoding="utf-8",
            )
            return prepared

        runtime.store.prepare_auto_mutation_receipts = prepare_then_expand  # type: ignore[method-assign]
        token = "PRIVATE-TASK-IMPORT-CAP-RACE-OWNER"
        try:
            result = runtime.handle(
                "refuse task import after note candidate race",
                request_token=token,
            )
        finally:
            runtime.store.prepare_auto_mutation_receipts = original_prepare  # type: ignore[method-assign]
        item = result.tool_results[0]
        if (
            item.ok
            or item.metadata.get("reason") != "too_many_tasks"
            or item.metadata.get("open_task_candidates") != 201
            or item.metadata.get("max_open_task_candidates")
            != tasks_module.MAX_TASK_IMPORT_CANDIDATES
            or item.metadata.get("handler_invoked") is not True
            or item.metadata.get("auto_mutation_effects_started") is not False
            or item.metadata.get("auto_mutation_definite_no_effect") is not True
            or item.metadata.get("auto_mutation_outcome_uncertain") is not False
            or calls[0] != 1
            or _task_rows(runtime)
            or _receipt_rows(runtime)
            or _projection_bytes(runtime) != projection_before
        ):
            raise SystemExit(f"post-preflight task-cap race retained effects or custody: {item}")
        outward = json.dumps(
            {"output": item.output, "metadata": item.metadata},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if private_marker in outward or str(Path(temp)) in outward:
            raise SystemExit("task-cap race refusal exposed private task or local path data")

        retry_body = "retry after task candidate cap refusal"
        note_path.write_text(f"- [ ] {retry_body}\n", encoding="utf-8")
        retried = runtime.handle(
            "retry task import after candidate cap refusal",
            request_token=token,
        )
        if (
            not retried.tool_results[0].ok
            or [row["body"] for row in _task_rows(runtime)] != [retry_body]
            or calls[0] != 2
        ):
            raise SystemExit("task-cap race refusal did not release same-token retry")
        _assert_completed_receipt_audits(runtime, 1)
        _assert_receipt_privacy(
            runtime,
            private_values=[token, path, private_marker, *raced_bodies, str(Path(temp))],
        )


def test_success_replay_fresh_convergence_audit_and_privacy() -> None:
    path = "Inbox/PRIVATE  Task Import.md"
    args = {"path": path, "priority": "high"}
    task_bodies = [
        "PRIVATE import alpha 7731",
        "PRIVATE import beta 7732",
        "PRIVATE import gamma 7733",
    ]
    owner = "PRIVATE-TASK-IMPORT-SUCCESS-OWNER"
    fresh_token = "PRIVATE-TASK-IMPORT-SUCCESS-FRESH"
    with TemporaryDirectory(prefix="jarvis-task-import-success-") as temp:
        runtime = _runtime(Path(temp), args)
        _write_note(
            runtime,
            path,
            "\n".join(
                [
                    "# Import fixture",
                    *(f"- [ ] {body}" for body in task_bodies),
                    "- [x] completed item must stay ignored",
                ]
            )
            + "\n",
        )
        calls = _install_counted_handler(runtime)
        first = runtime.handle("import three tasks", request_token=owner)
        item = first.tool_results[0]
        rows = _task_rows(runtime)
        projection = (_projection_bytes(runtime) or b"").decode("utf-8")
        if (
            not item.ok
            or calls[0] != 1
            or len(rows) != 3
            or [row["body"] for row in rows] != task_bodies
            or any(row["priority"] != "high" for row in rows)
            or any(body not in projection for body in task_bodies)
            or "completed item must stay ignored" in projection
        ):
            raise SystemExit(f"successful multi-task import drifted: {item} / {rows}")
        handoff = item.metadata.get("note_task_import_handoff")
        if (
            not isinstance(handoff, dict)
            or handoff.get("note_path_display") != path
            or handoff.get("next_commands", {}).get("preview_again")
            != f"preview tasks from note {path}"
        ):
            raise SystemExit(f"task-import handoff did not preserve a round-trippable note path: {handoff}")
        preview_command = str(handoff["next_commands"]["preview_again"])
        preview_plan = RuleBasedPlanner().plan(preview_command)
        if (
            len(preview_plan.actions) != 1
            or preview_plan.actions[0].tool_name != "preview_tasks_from_note"
            or preview_plan.actions[0].args != {"path": path}
        ):
            raise SystemExit(f"task-import preview command did not preserve repeated spaces: {preview_plan}")
        preview = runtime.registry.get("preview_tasks_from_note").handler(
            preview_plan.actions[0].args
        )
        if not preview.ok or preview.metadata.get("duplicates") != 3:
            raise SystemExit(f"task-import preview command did not round-trip: {preview}")
        first_snapshot = rows
        _assert_completed_receipt_audits(runtime, 1)

        same = runtime.handle("same-token task import replay", request_token=owner)
        _assert_replay(same, "auto_mutation_completed_replay", "same-token task import replay")
        if calls[0] != 1 or _task_rows(runtime) != first_snapshot:
            raise SystemExit("same-token task import replay reached the handler or duplicated rows")

        fresh = runtime.handle("fresh-token exact task import", request_token=fresh_token)
        fresh_item = fresh.tool_results[0]
        if (
            not fresh_item.ok
            or fresh_item.metadata.get("imported") != 0
            or fresh_item.metadata.get("skipped") != 3
            or calls[0] != 2
            or _task_rows(runtime) != first_snapshot
            or _projection_without_updated_timestamp(_projection_bytes(runtime))
            != _projection_without_updated_timestamp(projection.encode("utf-8"))
        ):
            raise SystemExit(f"fresh-token task import did not converge: {fresh_item}")
        _assert_completed_receipt_audits(runtime, 2)
        _assert_receipt_privacy(
            runtime,
            private_values=[owner, fresh_token, path, *task_bodies, str(Path(temp))],
        )
        outward = json.dumps(
            [first.tool_results[0].metadata, fresh.tool_results[0].metadata],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if (
            str(runtime.vault.root_path) in outward
            or str(runtime.store.db_path.parent) in outward
            or any(
                key in first.tool_results[0].metadata
                for key in ("path", "tasks_path")
            )
        ):
            raise SystemExit("task-import success metadata exposed an absolute local path")


def test_changed_priority_unresolved_target_is_fenced() -> None:
    path = "Inbox/Changed Priority Fence.md"
    low_args = {"path": path, "priority": "low"}
    with TemporaryDirectory(prefix="jarvis-task-import-priority-fence-") as temp:
        runtime = _runtime(Path(temp), low_args)
        _write_note(runtime, path, "- [ ] priority fence task\n")
        tool = runtime.registry.get(TOOL_NAME)
        contract = tool.auto_mutation_contract
        if contract is None or contract.operation_key_builder is None:
            raise SystemExit("task-import priority fence lost its operation-key builder")
        operation_args = contract.operation_key_builder(dict(low_args))
        owner = "task-import-priority-fence-owner"
        prepared = runtime.store.prepare_auto_mutation_receipts(
            owner,
            [(TOOL_NAME, low_args, operation_args)],
        )
        if prepared.status != "PREPARED" or prepared.receipts[0].receipt_id is None:
            raise SystemExit(f"could not prepare task-import priority fence: {prepared}")
        claim = runtime.store.claim_auto_mutation_receipt(
            int(prepared.receipts[0].receipt_id),
            owner,
        )
        if claim.status != "CLAIMED":
            raise SystemExit(f"could not claim task-import priority fence: {claim}")

        runtime.planner = StaticPlanner({"path": path, "priority": "high"})
        calls = _install_counted_handler(runtime)
        blocked = runtime.handle(
            "changed-priority task import contender",
            request_token="task-import-priority-fence-contender",
        )
        _assert_replay(
            blocked,
            "auto_mutation_unresolved_action",
            "changed-priority task import contender",
        )
        if calls[0] != 0 or _task_rows(runtime) or len(_receipt_rows(runtime)) != 1:
            raise SystemExit("changed-priority contender bypassed target-only unresolved fencing")


def test_atomic_mid_batch_database_failure_leaves_zero_new_tasks() -> None:
    path = "Inbox/Atomic Batch Failure.md"
    failing_body = "atomic import second must fail"
    args = {"path": path, "priority": "normal"}
    with TemporaryDirectory(prefix="jarvis-task-import-atomic-") as temp:
        runtime = _runtime(Path(temp), args)
        _write_note(
            runtime,
            path,
            "- [ ] atomic import first\n"
            f"- [ ] {failing_body}\n"
            "- [ ] atomic import third\n",
        )
        projection_before = _projection_bytes(runtime)
        failing_body_sql = failing_body.replace("'", "''")
        with runtime.store.connect() as conn:
            conn.execute(
                f"""
                CREATE TRIGGER task_import_atomic_failure
                BEFORE INSERT ON tasks
                WHEN NEW.body = '{failing_body_sql}'
                BEGIN
                    SELECT RAISE(ABORT, 'injected task import batch failure');
                END
                """
            )
        result = runtime.handle(
            "inject a task import batch failure",
            request_token="task-import-atomic-failure-owner",
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
            raise SystemExit(f"mid-batch task import failure was not atomic: {item} / {receipts}")


def test_database_before_projection_failure_is_uncertain_and_fenced() -> None:
    path = "Inbox/Projection Failure.md"
    args = {"path": path, "priority": "normal"}
    owner = "PRIVATE-TASK-IMPORT-PROJECTION-OWNER"
    cross_token = "PRIVATE-TASK-IMPORT-PROJECTION-CROSS"
    with TemporaryDirectory(prefix="jarvis-task-import-projection-failure-") as temp:
        runtime = _runtime(Path(temp), args)
        bodies = ["projection failure first", "projection failure second"]
        _write_note(runtime, path, "".join(f"- [ ] {body}\n" for body in bodies))
        projection_before = _projection_bytes(runtime)
        publication_calls = [0]

        def fail_projection(*_args: Any, **_kwargs: Any) -> Any:
            publication_calls[0] += 1
            raise OSError("injected task import projection failure")

        _install_projection_failure(runtime, fail_projection)
        failed = runtime.handle("fail task import projection", request_token=owner)
        item = failed.tool_results[0]
        receipts = _receipt_rows(runtime)
        if (
            item.ok
            or item.metadata.get("auto_mutation_outcome_uncertain") is not True
            or [row["body"] for row in _task_rows(runtime)] != bodies
            or _projection_bytes(runtime) != projection_before
            or publication_calls[0] != 1
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
            or any(row["ok"] == 1 for row in _tool_run_rows(runtime))
        ):
            raise SystemExit(f"post-database projection failure lost uncertain custody: {item}")

        same = runtime.handle("same-token projection failure replay", request_token=owner)
        cross = runtime.handle("cross-token projection failure replay", request_token=cross_token)
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token projection failure")
        _assert_replay(cross, "auto_mutation_unresolved_action", "cross-token projection failure")
        if publication_calls[0] != 1 or len(_task_rows(runtime)) != 2 or len(_receipt_rows(runtime)) != 1:
            raise SystemExit("projection-failure replay reran the task import")
        _assert_receipt_privacy(
            runtime,
            private_values=[owner, cross_token, path, *bodies, str(Path(temp))],
        )


def test_completion_audit_failure_is_uncertain_and_fenced() -> None:
    path = "Inbox/Completion Failure.md"
    args = {"path": path, "priority": "normal"}
    owner = "PRIVATE-TASK-IMPORT-COMPLETION-OWNER"
    cross_token = "PRIVATE-TASK-IMPORT-COMPLETION-CROSS"
    with TemporaryDirectory(prefix="jarvis-task-import-completion-failure-") as temp:
        runtime = _runtime(Path(temp), args)
        bodies = ["completion failure first", "completion failure second"]
        _write_note(runtime, path, "".join(f"- [ ] {body}\n" for body in bodies))
        original_complete = runtime.store.complete_auto_mutation_receipt

        def fail_completion(**_kwargs: Any) -> int:
            raise RuntimeError("injected task import completion/audit failure")

        runtime.store.complete_auto_mutation_receipt = fail_completion  # type: ignore[method-assign]
        failed = runtime.handle("fail task import completion", request_token=owner)
        runtime.store.complete_auto_mutation_receipt = original_complete  # type: ignore[method-assign]
        _assert_replay(
            failed,
            "auto_mutation_completion_failed",
            "task-import completion/audit failure",
        )
        projection = (_projection_bytes(runtime) or b"").decode("utf-8")
        receipts = _receipt_rows(runtime)
        if (
            [row["body"] for row in _task_rows(runtime)] != bodies
            or any(body not in projection for body in bodies)
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
            or any(row["ok"] == 1 for row in _tool_run_rows(runtime))
        ):
            raise SystemExit("task-import completion failure left false success evidence")

        same = runtime.handle("same-token completion replay", request_token=owner)
        cross = runtime.handle("cross-token completion replay", request_token=cross_token)
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token completion failure")
        _assert_replay(cross, "auto_mutation_unresolved_action", "cross-token completion failure")
        if len(_task_rows(runtime)) != 2 or len(_receipt_rows(runtime)) != 1:
            raise SystemExit("completion-failure replay duplicated task import rows")


def test_process_death_after_database_commit_is_recovered_and_fenced() -> None:
    args = {"path": PROCESS_DEATH_PATH, "priority": "normal"}
    cross_token = "PRIVATE-TASK-IMPORT-PROCESS-DEATH-CROSS"
    with TemporaryDirectory(prefix="jarvis-task-import-process-death-") as temp:
        root = Path(temp)
        process = multiprocessing.get_context("spawn").Process(
            target=_crash_after_task_import_database_commit,
            args=(str(root),),
        )
        process.start()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise SystemExit("task-import process-death fixture did not terminate")
        if process.exitcode != 73:
            raise SystemExit(f"task-import process-death fixture exited unexpectedly: {process.exitcode}")

        runtime = _runtime(root, args)
        receipts = _receipt_rows(runtime)
        if (
            len(_task_rows(runtime)) != 2
            or _projection_bytes(runtime) is not None
            or len(receipts) != 1
            or receipts[0]["state"] != "running"
        ):
            raise SystemExit("task-import process death lost its database or running receipt custody")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET running_at = ?, updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z", receipts[0]["id"]),
            )

        recovered = _runtime(root, args)
        recovered_receipts = _receipt_rows(recovered)
        if (
            len(recovered_receipts) != 1
            or recovered_receipts[0]["state"] != "uncertain"
            or recovered_receipts[0]["result"] != "unknown"
            or recovered_receipts[0]["resolution"] != "stale_recovery"
        ):
            raise SystemExit("startup did not fence the stale task-import process death")
        same = recovered.handle(
            "same-token task-import process-death replay",
            request_token=PROCESS_DEATH_TOKEN,
        )
        cross = recovered.handle(
            "cross-token task-import process-death replay",
            request_token=cross_token,
        )
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token process death")
        _assert_replay(cross, "auto_mutation_unresolved_action", "cross-token process death")
        if len(_task_rows(recovered)) != 2 or _projection_bytes(recovered) is not None:
            raise SystemExit("process-death replay duplicated rows or published an unowned projection")
        _assert_receipt_privacy(
            recovered,
            private_values=[PROCESS_DEATH_TOKEN, cross_token, PROCESS_DEATH_PATH, str(root)],
        )


def test_concurrent_same_target_is_fenced() -> None:
    path = "Inbox/Concurrent Import.md"
    args = {"path": path, "priority": "normal"}
    with TemporaryDirectory(prefix="jarvis-task-import-concurrent-") as temp:
        runtime = _runtime(Path(temp), args)
        bodies = ["concurrent import first", "concurrent import second"]
        _write_note(runtime, path, "".join(f"- [ ] {body}\n" for body in bodies))
        tool = runtime.registry.get(TOOL_NAME)
        entered = threading.Event()
        release = threading.Event()
        calls = [0]

        def delayed(args: dict[str, Any]) -> ToolResult:
            calls[0] += 1
            entered.set()
            if not release.wait(10):
                raise TimeoutError("concurrent task-import fixture was not released")
            return tool.handler(args)

        runtime.registry._tools[TOOL_NAME] = replace(tool, handler=delayed)
        with ThreadPoolExecutor(max_workers=1) as pool:
            winner_future = pool.submit(
                runtime.handle,
                "concurrent task-import owner",
                request_token="task-import-concurrent-owner",
            )
            if not entered.wait(10):
                release.set()
                raise SystemExit("concurrent task-import owner did not enter the handler")
            try:
                blocked = runtime.handle(
                    "concurrent task-import contender",
                    request_token="task-import-concurrent-contender",
                )
            finally:
                release.set()
            winner = winner_future.result(timeout=15)

        if not winner.tool_results[0].ok or calls[0] != 1:
            raise SystemExit(f"concurrent task-import owner did not complete once: {winner.tool_results}")
        _assert_replay(blocked, "auto_mutation_unresolved_action", "concurrent task import")
        if [row["body"] for row in _task_rows(runtime)] != bodies:
            raise SystemExit("concurrent same-target task imports duplicated or lost rows")
        _assert_completed_receipt_audits(runtime, 1)


def main() -> None:
    test_exact_contract_schema_and_target_only_identity()
    test_typed_and_semantic_preflight_precede_receipts_and_writes()
    test_control_character_path_is_rejected_without_audit_injection()
    test_nonregular_note_reason_matches_preview_direct_and_runtime_import()
    test_post_claim_definite_no_effect_refusals_release_custody()
    test_definite_no_effect_requires_explicit_prewrite_marker()
    test_empty_note_postread_symlink_drift_uses_lexical_display()
    test_stale_prepared_receipt_cleanup_releases_unstarted_import()
    test_multi_action_no_effect_failure_preserves_receipt_batch_shape()
    test_unicode_equivalent_tasks_dedupe_within_atomic_batch()
    test_import_body_boundary_precedes_identity_and_insertion()
    test_task_candidate_cap_preflight_allows_200_and_refuses_201()
    test_task_candidate_cap_post_preflight_race_releases_receipt()
    test_success_replay_fresh_convergence_audit_and_privacy()
    test_changed_priority_unresolved_target_is_fenced()
    test_atomic_mid_batch_database_failure_leaves_zero_new_tasks()
    test_database_before_projection_failure_is_uncertain_and_fenced()
    test_completion_audit_failure_is_uncertain_and_fenced()
    test_process_death_after_database_commit_is_recovered_and_fenced()
    test_concurrent_same_target_is_fenced()
    print("Task import auto-mutation rollout smoke passed")


if __name__ == "__main__":
    main()
