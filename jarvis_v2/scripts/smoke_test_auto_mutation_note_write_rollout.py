from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import threading
from tempfile import TemporaryDirectory
from typing import Any

import jarvis_v2.memory.obsidian as obsidian_module
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.memory.store import (
    auto_mutation_action_digest,
    auto_mutation_request_digest,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.notes import (
    MAX_NOTE_PATH_CHARS,
    MAX_NOTE_WRITE_CHARS,
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
            "Exercise the aggregate-wired note-write mutation.",
            [PlannedAction("write_jarvis_note", self.args, "note-write rollout smoke")],
            needs_model=False,
        )


def _runtime(root: Path, args: Any) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(args)
    return runtime


def _crash_after_note_append(root_text: str) -> None:
    args = {
        "path": "Projects/Process Death Note",
        "body": "process-death note body remains single",
        "mode": "append",
    }
    runtime = _runtime(Path(root_text), args)
    original_append = runtime.vault.append_note

    def append_then_exit(
        note_path: str | Path,
        body: str,
        *,
        heading: str,
    ) -> tuple[Path, bool]:
        original_append(note_path, body, heading=heading)
        os._exit(67)

    runtime.vault.append_note = append_then_exit  # type: ignore[method-assign]
    runtime.handle(
        "append before representative process death",
        request_token="PRIVATE-NOTE-PROCESS-DEATH-OWNER",
    )
    os._exit(68)


def _rows(runtime: JarvisRuntime, table: str) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id")]


def _receipt_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "auto_mutation_receipts")


def _tool_run_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "tool_runs")


def _vault_snapshot(runtime: JarvisRuntime) -> dict[str, bytes]:
    return {
        str(path.relative_to(runtime.vault.root_path)): path.read_bytes()
        for path in runtime.vault.root_path.rglob("*")
        if path.is_file()
    }


def _vault_tree(runtime: JarvisRuntime) -> tuple[str, ...]:
    return tuple(
        sorted(str(path.relative_to(runtime.vault.root_path)) for path in runtime.vault.root_path.rglob("*"))
    )


def _note_text(runtime: JarvisRuntime, relative_path: str) -> str:
    path = runtime.vault.root_path / relative_path
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _assert_replay(result: Any, failure_kind: str, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} did not return exactly one tool result: {result.tool_results}")
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != failure_kind:
        raise SystemExit(f"{label} returned the wrong replay state: {item}")


def _install_counted_handler(runtime: JarvisRuntime) -> list[int]:
    tool = runtime.registry.get("write_jarvis_note")
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools["write_jarvis_note"] = replace(tool, handler=counted)
    return calls


def _assert_completed_receipt_audits(runtime: JarvisRuntime, expected: int) -> None:
    receipts = _receipt_rows(runtime)
    successful_runs = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
    runs_by_id = {row["id"]: row for row in successful_runs}
    if len(receipts) != expected or len(successful_runs) != expected:
        raise SystemExit(f"expected {expected} note-write receipt/audit pairs: {receipts} / {successful_runs}")
    linked: set[int] = set()
    for receipt in receipts:
        run = runs_by_id.get(receipt["tool_run_id"])
        if (
            receipt["tool_name"] != "write_jarvis_note"
            or receipt["state"] != "completed"
            or receipt["result"] != "succeeded"
            or receipt["resolution"] != "recorded"
            or run is None
            or run["tool_name"] != "write_jarvis_note"
            or run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
        ):
            raise SystemExit(f"note-write receipt did not link one ordinary approved=0 audit: {receipt} / {run}")
        linked.add(int(run["id"]))
    if len(linked) != expected:
        raise SystemExit("note-write receipts reused an ordinary audit row")


def _receipt_runtime_fields(results: list[Any]) -> list[dict[str, Any]]:
    private_names = {
        "request_digest",
        "action_digest",
        "operation_digest",
        "uncertainty_digest",
        "run_token",
        "receipt_id",
        "request_token",
    }
    collected: list[dict[str, Any]] = []
    for result in results:
        fields: dict[str, Any] = {}
        for source in (result.metadata, result.tool_results[0].metadata):
            for key, value in source.items():
                lowered = str(key).casefold()
                if (
                    lowered in private_names
                    or lowered.startswith("auto_mutation")
                    or "receipt" in lowered
                ):
                    fields[str(key)] = value
        collected.append(fields)
    return collected


def _assert_receipt_privacy(
    runtime: JarvisRuntime,
    results: list[Any],
    *,
    actions: list[dict[str, Any]],
    request_tokens: list[str],
) -> None:
    receipts = _receipt_rows(runtime)
    receipt_text = json.dumps(receipts, ensure_ascii=False, sort_keys=True, default=str)
    raw_values = [
        value
        for action in actions
        for key, value in action.items()
        if key in {"path", "body"} and type(value) is str and value
    ]
    for private in [*request_tokens, *raw_values]:
        if private in receipt_text:
            raise SystemExit("note-write receipt stored a raw body, path, or request token")

    digests = [auto_mutation_request_digest(token) for token in request_tokens]
    digests.extend(auto_mutation_action_digest("write_jarvis_note", action) for action in actions)
    digests.extend(
        str(row[key])
        for row in receipts
        for key in (
            "request_digest",
            "action_digest",
            "operation_digest",
            "uncertainty_digest",
            "run_token",
        )
        if row.get(key)
    )
    private_runtime_text = json.dumps(
        _receipt_runtime_fields(results),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    for private in [*request_tokens, *raw_values, *digests]:
        if private and private in private_runtime_text:
            raise SystemExit("note-write receipt-private runtime fields exposed mutation evidence")


def _assert_no_boundary_crossing(
    runtime: JarvisRuntime,
    *,
    label: str,
    vault_before: dict[str, bytes],
    receipt_count: int = 0,
) -> None:
    if len(_receipt_rows(runtime)) != receipt_count:
        raise SystemExit(f"{label} created an auto-mutation receipt")
    if _vault_snapshot(runtime) != vault_before:
        raise SystemExit(f"{label} changed the Obsidian vault")


def test_exact_contract_schema_preflight_and_target_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-note-write-contract-") as temp:
        runtime = _runtime(Path(temp), {"path": "Projects/Contract", "body": "contract"})
        tool = runtime.registry.get("write_jarvis_note")
        contract = tool.auto_mutation_contract
        if (
            tool.risk is not RiskLevel.LOCAL_SAFE
            or tool.toolset != "notes"
            or contract is None
            or contract.version != AUTO_MUTATION_CONTRACT_VERSION
            or contract.effects != frozenset({AutoMutationEffect.OBSIDIAN_VAULT})
            or contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
            or contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
            or not callable(contract.operation_key_builder)
            or getattr(contract.operation_key_builder, "__name__", "") != "operation_key"
            or not callable(contract.semantic_preflight)
            or getattr(contract.semantic_preflight, "__name__", "") != "preflight"
            or not callable(contract.semantic_preflight_result_builder)
            or getattr(contract.semantic_preflight_result_builder, "__name__", "")
            != "refusal"
        ):
            raise SystemExit(f"write_jarvis_note auto-mutation contract drifted: {tool}")

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
            or shape
            != (
                ("path", string, True),
                ("body", string, True),
                ("mode", string, False),
            )
        ):
            raise SystemExit(f"write_jarvis_note strict argument contract drifted: {arguments}")

        variants = (
            {"path": "Projects/\uff21lpha", "body": "first"},
            {"path": "projects/alpha.md", "body": "different", "mode": "append"},
            {"path": "PROJECTS/ALPHA.MD", "body": "another", "mode": "create"},
        )
        keys = [contract.operation_key_builder(args) for args in variants]
        if keys[0] != keys[1] or keys[1] != keys[2]:
            raise SystemExit(f"note-write identity is not canonical target-only identity: {keys}")
        if set(keys[0]) != {"target"}:
            raise SystemExit(f"note-write operation identity contains mutable request content: {keys[0]}")
        changed = contract.operation_key_builder(
            {"path": "Projects/Beta", "body": "first"}
        )
        if changed == keys[0]:
            raise SystemExit("different note targets collapsed to one operation identity")

        tree_before = _vault_tree(runtime)
        preflight_result = contract.semantic_preflight(
            {
                "path": "Never/Created/By/Preflight/Note",
                "body": "classification must stay read-only",
                "mode": "append",
            }
        )
        if preflight_result is not None or _vault_tree(runtime) != tree_before:
            raise SystemExit("note-write semantic preflight created a missing parent directory")


def test_append_success_replay_intentional_repeat_privacy_and_audit() -> None:
    args = {
        "path": "Projects/PRIVATE-NEBULA-NOTE",
        "body": "PRIVATE-COPPER-BODY-APPEND-7419",
        "mode": "append",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-append-") as temp:
        runtime = _runtime(Path(temp), args)
        calls = _install_counted_handler(runtime)
        token_one = "PRIVATE-NOTE-WRITE-TOKEN-APPEND-ONE"
        token_two = "PRIVATE-NOTE-WRITE-TOKEN-APPEND-TWO"

        first = runtime.handle("append note first", request_token=token_one)
        if not first.tool_results[0].ok or calls[0] != 1:
            raise SystemExit(f"initial aggregate-wired note append failed: {first.tool_results}")
        relative_path = "Projects/PRIVATE-NEBULA-NOTE.md"
        if _note_text(runtime, relative_path).count(args["body"]) != 1:
            raise SystemExit("initial append did not publish exactly one body")

        same = runtime.handle("append note same-token replay", request_token=token_one)
        _assert_replay(same, "auto_mutation_completed_replay", "same-token note append replay")
        if calls[0] != 1 or _note_text(runtime, relative_path).count(args["body"]) != 1:
            raise SystemExit("same-token note append replay ran the handler or appended again")

        fresh = runtime.handle("append note intentional repeat", request_token=token_two)
        if not fresh.tool_results[0].ok or calls[0] != 2:
            raise SystemExit(f"fresh-token note append did not intentionally repeat: {fresh.tool_results}")
        if _note_text(runtime, relative_path).count(args["body"]) != 2:
            raise SystemExit("fresh-token append did not publish the body a second time")

        _assert_completed_receipt_audits(runtime, 2)
        _assert_receipt_privacy(
            runtime,
            [first, same, fresh],
            actions=[args],
            request_tokens=[token_one, token_two],
        )


def test_unresolved_canonical_target_fences_changed_body_and_mode() -> None:
    owner_args = {
        "path": "Projects/\uff26ence Target",
        "body": "PRIVATE-OWNER-BODY-1157",
        "mode": "append",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-fence-") as temp:
        runtime = _runtime(Path(temp), owner_args)
        contract = runtime.registry.get("write_jarvis_note").auto_mutation_contract
        if contract is None or contract.operation_key_builder is None:
            raise SystemExit("note-write fencing fixture lost its operation-key builder")
        operation_args = contract.operation_key_builder(dict(owner_args))
        owner_token = "PRIVATE-NOTE-FENCE-OWNER"
        prepared = runtime.store.prepare_auto_mutation_receipts(
            owner_token,
            [("write_jarvis_note", owner_args, operation_args)],
        )
        if prepared.status != "PREPARED" or prepared.receipts[0].receipt_id is None:
            raise SystemExit(f"could not prepare note-write fencing owner: {prepared}")
        claim = runtime.store.claim_auto_mutation_receipt(
            int(prepared.receipts[0].receipt_id), owner_token
        )
        if claim.status != "CLAIMED":
            raise SystemExit(f"could not claim note-write fencing owner: {claim}")

        calls = _install_counted_handler(runtime)
        vault_before = _vault_snapshot(runtime)
        explicit_args = {
            "path": "projects/fence target.md",
            "body": "PRIVATE-DIFFERENT-BODY-2268",
            "mode": "append",
        }
        runtime.planner = StaticPlanner(explicit_args)
        explicit = runtime.handle(
            "explicit extension equivalent target",
            request_token="PRIVATE-NOTE-FENCE-EXPLICIT",
        )
        _assert_replay(
            explicit,
            "auto_mutation_unresolved_action",
            "omitted-extension versus explicit-extension note target",
        )

        changed_intent_args = {
            "path": "PROJECTS/FENCE TARGET.MD",
            "body": "PRIVATE-CREATE-BODY-3379",
            "mode": "create",
        }
        runtime.planner = StaticPlanner(changed_intent_args)
        changed = runtime.handle(
            "changed body and mode for unresolved target",
            request_token="PRIVATE-NOTE-FENCE-CHANGED",
        )
        _assert_replay(
            changed,
            "auto_mutation_unresolved_action",
            "changed body and mode for unresolved note target",
        )
        if calls[0] != 0:
            raise SystemExit("same-target unresolved fencing invoked the note handler")
        _assert_no_boundary_crossing(
            runtime,
            label="same-target unresolved note requests",
            vault_before=vault_before,
            receipt_count=1,
        )
        _assert_receipt_privacy(
            runtime,
            [explicit, changed],
            actions=[owner_args, explicit_args, changed_intent_args],
            request_tokens=[
                owner_token,
                "PRIVATE-NOTE-FENCE-EXPLICIT",
                "PRIVATE-NOTE-FENCE-CHANGED",
            ],
        )


def test_post_append_vault_failure_is_uncertain_and_blocks_replay() -> None:
    args = {
        "path": "Projects/PRIVATE-UNCERTAIN-NOTE",
        "body": "PRIVATE-UNCERTAIN-BODY-4480",
        "mode": "append",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-uncertain-") as temp:
        runtime = _runtime(Path(temp), args)
        original_append = runtime.vault.append_note
        calls = [0]

        def append_then_fail(note_path: str | Path, body: str, *, heading: str) -> tuple[Path, bool]:
            calls[0] += 1
            original_append(note_path, body, heading=heading)
            raise OSError("representative private failure after note publication")

        runtime.vault.append_note = append_then_fail  # type: ignore[method-assign]
        owner = "PRIVATE-NOTE-UNCERTAIN-OWNER"
        failed = runtime.handle("append before representative vault failure", request_token=owner)
        item = failed.tool_results[0]
        receipts = _receipt_rows(runtime)
        if (
            item.ok
            or item.metadata.get("auto_mutation_outcome_uncertain") is not True
            or calls[0] != 1
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
        ):
            raise SystemExit(f"post-append note failure did not retain uncertain custody: {item} / {receipts}")
        if _note_text(runtime, "Projects/PRIVATE-UNCERTAIN-NOTE.md").count(args["body"]) != 1:
            raise SystemExit("post-append failure fixture did not publish exactly once before failing")

        same = runtime.handle("same-token uncertain replay", request_token=owner)
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token uncertain note replay")
        cross_token = "PRIVATE-NOTE-UNCERTAIN-CROSS"
        cross = runtime.handle("cross-token uncertain replay", request_token=cross_token)
        _assert_replay(cross, "auto_mutation_unresolved_action", "cross-token uncertain note replay")
        if calls[0] != 1 or len(_receipt_rows(runtime)) != 1:
            raise SystemExit("uncertain same/cross-token note replay reran or created another receipt")
        if any(row["ok"] == 1 for row in _tool_run_rows(runtime)):
            raise SystemExit("uncertain post-append note failure created an ordinary success audit")
        _assert_receipt_privacy(
            runtime,
            [failed, same, cross],
            actions=[args],
            request_tokens=[owner, cross_token],
        )


def test_completion_failure_after_append_stops_all_replay() -> None:
    args = {
        "path": "Projects/PRIVATE-COMPLETION-FAILURE-NOTE",
        "body": "PRIVATE-COMPLETION-FAILURE-BODY-5591",
        "mode": "append",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-completion-failure-") as temp:
        runtime = _runtime(Path(temp), args)
        original_complete = runtime.store.complete_auto_mutation_receipt

        def fail_completion(**_kwargs: Any) -> int:
            raise RuntimeError("representative private receipt completion failure")

        runtime.store.complete_auto_mutation_receipt = fail_completion  # type: ignore[method-assign]
        owner = "PRIVATE-NOTE-COMPLETION-OWNER"
        failed = runtime.handle("append before audit completion failure", request_token=owner)
        runtime.store.complete_auto_mutation_receipt = original_complete  # type: ignore[method-assign]
        _assert_replay(
            failed,
            "auto_mutation_completion_failed",
            "note append audit completion failure",
        )
        relative_path = "Projects/PRIVATE-COMPLETION-FAILURE-NOTE.md"
        before = _note_text(runtime, relative_path)
        if before.count(args["body"]) != 1:
            raise SystemExit("note append did not complete before audit finalization failed")

        same = runtime.handle("same-token completion failure replay", request_token=owner)
        cross_token = "PRIVATE-NOTE-COMPLETION-CROSS"
        cross = runtime.handle("cross-token completion failure replay", request_token=cross_token)
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token completion failure")
        _assert_replay(cross, "auto_mutation_unresolved_action", "cross-token completion failure")
        receipts = _receipt_rows(runtime)
        if (
            _note_text(runtime, relative_path) != before
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
            or any(row["ok"] == 1 for row in _tool_run_rows(runtime))
        ):
            raise SystemExit("note completion failure replayed or left false success evidence")
        _assert_receipt_privacy(
            runtime,
            [failed, same, cross],
            actions=[args],
            request_tokens=[owner, cross_token],
        )


def test_process_death_after_append_recovers_uncertain_without_duplicate() -> None:
    args = {
        "path": "Projects/Process Death Note",
        "body": "process-death note body remains single",
        "mode": "append",
    }
    owner = "PRIVATE-NOTE-PROCESS-DEATH-OWNER"
    cross_token = "PRIVATE-NOTE-PROCESS-DEATH-CROSS"
    with TemporaryDirectory(prefix="jarvis-note-write-process-death-") as temp:
        root = Path(temp)
        process = multiprocessing.get_context("spawn").Process(
            target=_crash_after_note_append,
            args=(str(root),),
        )
        process.start()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise SystemExit("note process-death fixture did not terminate")
        if process.exitcode != 67:
            raise SystemExit(f"note process-death fixture exited unexpectedly: {process.exitcode}")

        runtime = _runtime(root, args)
        receipts = _receipt_rows(runtime)
        relative_path = "Projects/Process Death Note.md"
        before = _note_text(runtime, relative_path)
        if (
            len(receipts) != 1
            or receipts[0]["state"] != "running"
            or before.count(args["body"]) != 1
        ):
            raise SystemExit("note process death lost its running receipt or published content")
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
            raise SystemExit("startup did not fence the stale note-write process death")
        same = recovered.handle("same-token process-death replay", request_token=owner)
        cross = recovered.handle("cross-token process-death replay", request_token=cross_token)
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token process-death replay")
        _assert_replay(cross, "auto_mutation_unresolved_action", "cross-token process-death replay")
        if _note_text(recovered, relative_path) != before:
            raise SystemExit("process-death note write replay appended duplicate content")
        _assert_receipt_privacy(
            recovered,
            [same, cross],
            actions=[args],
            request_tokens=[owner, cross_token],
        )


def test_preflight_and_typed_refusals_precede_receipts_and_writes() -> None:
    semantic_cases = [
        ({"path": "", "body": "safe"}, "missing_path"),
        ({"path": "Projects/Missing Body", "body": " \r\n "}, "missing_body"),
        ({"path": "Projects/Oversized", "body": "x" * (MAX_NOTE_WRITE_CHARS + 1)}, "body_too_large"),
        ({"path": "Projects/Bad Mode", "body": "safe", "mode": "overwrite"}, "bad_mode"),
        ({"path": "../PRIVATE-OUTSIDE.md", "body": "safe"}, "unsafe_path"),
        ({"path": "Memory Tree/Current Context", "body": "safe"}, "managed_projection"),
    ]
    typed_cases: list[Any] = [
        {},
        {"path": "Projects/Missing Body"},
        {"body": "missing path"},
        {"path": None, "body": "safe"},
        {"path": 1, "body": "safe"},
        {"path": "Projects/Typed", "body": None},
        {"path": "Projects/Typed", "body": True},
        {"path": "Projects/Typed", "body": []},
        {"path": "Projects/Typed", "body": "safe", "mode": False},
        {"path": "Projects/Typed", "body": "safe", "unknown": "rejected"},
        None,
        [],
    ]

    for index, (args, reason) in enumerate(semantic_cases):
        with TemporaryDirectory(prefix="jarvis-note-write-semantic-") as temp:
            runtime = _runtime(Path(temp), args)
            calls = _install_counted_handler(runtime)
            vault_before = _vault_snapshot(runtime)
            result = runtime.handle("invalid semantic note write", request_token=f"semantic-{index}")
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
                or item.metadata.get("reason") != reason
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
            ):
                raise SystemExit(f"note-write semantic preflight case {index} drifted: {item}")
            _assert_no_boundary_crossing(
                runtime,
                label=f"note-write semantic preflight case {index}",
                vault_before=vault_before,
            )

    for index, args in enumerate(typed_cases):
        with TemporaryDirectory(prefix="jarvis-note-write-typed-") as temp:
            runtime = _runtime(Path(temp), args)
            calls = _install_counted_handler(runtime)
            vault_before = _vault_snapshot(runtime)
            result = runtime.handle("invalid typed note write", request_token=f"typed-{index}")
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
            ):
                raise SystemExit(f"note-write typed argument case {index} drifted: {item}")
            _assert_no_boundary_crossing(
                runtime,
                label=f"note-write typed argument case {index}",
                vault_before=vault_before,
            )


def test_oversized_path_preflight_precedes_receipt_and_write() -> None:
    args = {
        "path": "Projects/" + "p" * (MAX_NOTE_PATH_CHARS + 1),
        "body": "safe",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-oversized-path-") as temp:
        runtime = _runtime(Path(temp), args)
        calls = _install_counted_handler(runtime)
        vault_before = _vault_snapshot(runtime)
        result = runtime.handle("oversized note path", request_token="oversized-path")
        item = result.tool_results[0]
        if (
            item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
            or item.metadata.get("reason") != "path_too_large"
            or item.metadata.get("handler_invoked") is not False
            or calls[0] != 0
        ):
            raise SystemExit(f"note-write oversized-path preflight drifted: {item}")
        _assert_no_boundary_crossing(
            runtime,
            label="note-write oversized-path preflight",
            vault_before=vault_before,
        )


def test_concurrent_same_target_operation_is_fenced() -> None:
    args = {
        "path": "Projects/Concurrent Fence",
        "body": "concurrent note body",
        "mode": "append",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-concurrent-") as temp:
        runtime = _runtime(Path(temp), args)
        original_append = runtime.vault.append_note
        entered = threading.Event()
        release = threading.Event()
        calls = [0]

        def delayed_append(note_path: str | Path, body: str, *, heading: str) -> tuple[Path, bool]:
            calls[0] += 1
            entered.set()
            if not release.wait(10):
                raise TimeoutError("concurrent note-write fixture was not released")
            return original_append(note_path, body, heading=heading)

        runtime.vault.append_note = delayed_append  # type: ignore[method-assign]
        with ThreadPoolExecutor(max_workers=1) as pool:
            winner_future = pool.submit(
                runtime.handle,
                "concurrent note-write owner",
                request_token="note-concurrent-owner",
            )
            if not entered.wait(10):
                release.set()
                raise SystemExit("concurrent note-write owner did not enter the vault handler")
            try:
                blocked = runtime.handle(
                    "concurrent note-write contender",
                    request_token="note-concurrent-contender",
                )
            finally:
                release.set()
            winner = winner_future.result(timeout=15)

        if not winner.tool_results[0].ok or calls[0] != 1:
            raise SystemExit(f"concurrent note-write owner did not complete exactly once: {winner.tool_results}")
        _assert_replay(blocked, "auto_mutation_unresolved_action", "concurrent same-target contender")
        if _note_text(runtime, "Projects/Concurrent Fence.md").count(args["body"]) != 1:
            raise SystemExit("concurrent same-target note requests published more than one body")
        _assert_completed_receipt_audits(runtime, 1)


def test_create_replay_exact_convergence_and_differing_existing_refusal() -> None:
    args = {
        "path": "Projects/Create Convergence",
        "body": "Canonical create body.",
        "mode": "create",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-create-") as temp:
        runtime = _runtime(Path(temp), args)
        calls = _install_counted_handler(runtime)
        owner = "note-create-owner"
        fresh_token = "note-create-fresh-exact"

        created = runtime.handle("create canonical note", request_token=owner)
        if not created.tool_results[0].ok or calls[0] != 1:
            raise SystemExit(f"initial create-mode note write failed: {created.tool_results}")
        canonical = _note_text(runtime, "Projects/Create Convergence.md")
        if canonical != "# Create Convergence\n\nCanonical create body.\n":
            raise SystemExit(f"create-mode note content was not canonical: {canonical!r}")

        same = runtime.handle("same-token create replay", request_token=owner)
        _assert_replay(same, "auto_mutation_completed_replay", "same-token create replay")
        if calls[0] != 1:
            raise SystemExit("same-token completed create replay reached the handler")

        fresh = runtime.handle("fresh exact create convergence", request_token=fresh_token)
        fresh_item = fresh.tool_results[0]
        if (
            not fresh_item.ok
            or fresh_item.metadata.get("state_changed") is not False
            or calls[0] != 2
            or _note_text(runtime, "Projects/Create Convergence.md") != canonical
        ):
            raise SystemExit(f"fresh exact existing create did not converge without a write: {fresh_item}")
        _assert_completed_receipt_audits(runtime, 2)

        differing_args = dict(args, body="Different create body must be refused.")
        runtime.planner = StaticPlanner(differing_args)
        vault_before = _vault_snapshot(runtime)
        receipts_before = len(_receipt_rows(runtime))
        differing = runtime.handle(
            "fresh differing create refusal",
            request_token="note-create-fresh-different",
        )
        differing_item = differing.tool_results[0]
        if (
            differing_item.metadata.get("failure_kind")
            != "auto_mutation_semantic_preflight_rejected"
            or differing_item.metadata.get("reason") != "already_exists"
            or differing_item.metadata.get("handler_invoked") is not False
            or calls[0] != 2
        ):
            raise SystemExit(f"differing existing create was not deterministically refused: {differing_item}")
        _assert_no_boundary_crossing(
            runtime,
            label="differing existing create preflight",
            vault_before=vault_before,
            receipt_count=receipts_before,
        )


def test_durable_receipt_state_precedes_mutable_create_preflight() -> None:
    args = {
        "path": "Projects/Create Receipt Precedence",
        "body": "Original canonical body.",
        "mode": "create",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-receipt-precedence-") as temp:
        runtime = _runtime(Path(temp), args)
        calls = _install_counted_handler(runtime)
        owner = "note-create-precedence-owner"
        created = runtime.handle("create before external drift", request_token=owner)
        if not created.tool_results[0].ok or calls[0] != 1:
            raise SystemExit(f"receipt-precedence fixture did not create its note: {created.tool_results}")

        note_path = runtime.vault.root_path / "Projects/Create Receipt Precedence.md"
        note_path.write_text("# External edit\n\nChanged after completion.\n", encoding="utf-8")
        replay = runtime.handle("same-token replay after external drift", request_token=owner)
        _assert_replay(
            replay,
            "auto_mutation_completed_replay",
            "completed create receipt after mutable destination drift",
        )
        if calls[0] != 1 or "Changed after completion" not in note_path.read_text(encoding="utf-8"):
            raise SystemExit("completed receipt replay reran the handler or rewrote external content")

        fenced_args = {
            "path": "Projects/Create Unresolved Precedence",
            "body": "Owner body.",
            "mode": "create",
        }
        contract = runtime.registry.get("write_jarvis_note").auto_mutation_contract
        if contract is None or contract.operation_key_builder is None:
            raise SystemExit("create receipt-precedence fixture lost its operation-key builder")
        operation_args = contract.operation_key_builder(dict(fenced_args))
        unresolved_owner = "note-create-unresolved-owner"
        prepared = runtime.store.prepare_auto_mutation_receipts(
            unresolved_owner,
            [("write_jarvis_note", fenced_args, operation_args)],
        )
        if prepared.status != "PREPARED" or prepared.receipts[0].receipt_id is None:
            raise SystemExit(f"could not prepare unresolved create owner: {prepared}")
        claim = runtime.store.claim_auto_mutation_receipt(
            int(prepared.receipts[0].receipt_id), unresolved_owner
        )
        if claim.status != "CLAIMED":
            raise SystemExit(f"could not claim unresolved create owner: {claim}")

        unresolved_path = runtime.vault.root_path / "Projects/Create Unresolved Precedence.md"
        unresolved_path.write_text("# External winner\n\nDifferent content.\n", encoding="utf-8")
        runtime.planner = StaticPlanner(fenced_args)
        before = unresolved_path.read_bytes()
        blocked = runtime.handle(
            "cross-token create after destination drift",
            request_token="note-create-unresolved-contender",
        )
        _assert_replay(
            blocked,
            "auto_mutation_unresolved_action",
            "unresolved create receipt before mutable preflight",
        )
        if calls[0] != 1 or unresolved_path.read_bytes() != before:
            raise SystemExit("unresolved create contender reached the handler or changed the destination")


def test_managed_open_tasks_projection_is_protected() -> None:
    for mode in ("append", "create"):
        args = {
            "path": "Tasks/Open Tasks.md",
            "body": "must not overwrite the generated task projection",
            "mode": mode,
        }
        with TemporaryDirectory(prefix=f"jarvis-note-write-managed-{mode}-") as temp:
            runtime = _runtime(Path(temp), args)
            calls = _install_counted_handler(runtime)
            vault_before = _vault_snapshot(runtime)
            result = runtime.handle(
                f"{mode} managed open-tasks projection",
                request_token=f"managed-open-tasks-{mode}",
            )
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind")
                != "auto_mutation_semantic_preflight_rejected"
                or item.metadata.get("reason") != "managed_projection"
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
            ):
                raise SystemExit(f"managed Open Tasks {mode} protection drifted: {item}")
            _assert_no_boundary_crossing(
                runtime,
                label=f"managed Open Tasks {mode}",
                vault_before=vault_before,
            )


def test_symlink_alias_and_fifo_targets_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-note-write-symlink-") as temp:
        runtime = _runtime(
            Path(temp),
            {
                "path": "Aliases/Victim",
                "body": "must never reach the symlink victim",
                "mode": "append",
            },
        )
        projects = runtime.vault.root_path / "Projects"
        projects.mkdir(parents=True, exist_ok=True)
        victim = projects / "Victim.md"
        victim.write_text("# Victim\n\nOriginal.\n", encoding="utf-8")
        (runtime.vault.root_path / "Aliases").symlink_to(projects, target_is_directory=True)
        before = victim.read_bytes()
        result = runtime.handle("append through internal symlink alias", request_token="symlink-alias")
        item = result.tool_results[0]
        if (
            item.ok
            or item.metadata.get("failure_kind")
            != "auto_mutation_semantic_preflight_rejected"
            or item.metadata.get("reason") != "unsafe_path"
            or _receipt_rows(runtime)
            or victim.read_bytes() != before
        ):
            raise SystemExit(f"internal symlink alias did not fail closed: {result.tool_results}")

    if not hasattr(os, "mkfifo"):
        return
    with TemporaryDirectory(prefix="jarvis-note-write-fifo-") as temp:
        args = {
            "path": "Projects/Existing Pipe",
            "body": "must not block on a non-regular destination",
            "mode": "create",
        }
        runtime = _runtime(Path(temp), args)
        projects = runtime.vault.root_path / "Projects"
        projects.mkdir(parents=True, exist_ok=True)
        fifo = projects / "Existing Pipe.md"
        os.mkfifo(fifo)
        result = runtime.handle("create where a FIFO exists", request_token="fifo-create")
        item = result.tool_results[0]
        if (
            item.metadata.get("failure_kind")
            != "auto_mutation_semantic_preflight_rejected"
            or item.metadata.get("reason") != "already_exists"
            or not fifo.exists()
            or _receipt_rows(runtime)
        ):
            raise SystemExit(f"FIFO create target did not refuse before mutation custody: {item}")

    with TemporaryDirectory(prefix="jarvis-note-write-fifo-append-") as temp:
        args = {
            "path": "Projects/Existing Append Pipe",
            "body": "must not block while reading a non-regular append target",
            "mode": "append",
        }
        runtime = _runtime(Path(temp), args)
        projects = runtime.vault.root_path / "Projects"
        projects.mkdir(parents=True, exist_ok=True)
        fifo = projects / "Existing Append Pipe.md"
        os.mkfifo(fifo)
        result = runtime.handle("append where a FIFO exists", request_token="fifo-append")
        item = result.tool_results[0]
        if (
            item.ok
            or item.metadata.get("failure_kind")
            != "auto_mutation_semantic_preflight_rejected"
            or item.metadata.get("reason") != "unsafe_path"
            or _receipt_rows(runtime)
            or not fifo.exists()
        ):
            raise SystemExit(f"FIFO append target did not fail closed: {result.tool_results}")


def test_durable_receipts_precede_out_of_vault_symlink_drift() -> None:
    args = {
        "path": "Projects/Symlink Drift Receipt",
        "body": "canonical receipt body",
        "mode": "create",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-symlink-drift-") as temp:
        root = Path(temp)
        runtime = _runtime(root, args)
        owner = "symlink-drift-completed-owner"
        created = runtime.handle("create before symlink drift", request_token=owner)
        if not created.tool_results[0].ok:
            raise SystemExit(f"symlink-drift fixture did not create its note: {created.tool_results}")
        note_path = runtime.vault.root_path / "Projects/Symlink Drift Receipt.md"
        outside = root / "outside-target.md"
        outside.write_text("outside must remain unchanged\n", encoding="utf-8")
        note_path.unlink()
        note_path.symlink_to(outside)
        replay = runtime.handle("same-token replay after symlink drift", request_token=owner)
        _assert_replay(
            replay,
            "auto_mutation_completed_replay",
            "completed receipt before out-of-vault symlink drift",
        )
        if outside.read_text(encoding="utf-8") != "outside must remain unchanged\n":
            raise SystemExit("completed receipt replay followed a drifted out-of-vault symlink")

        unresolved_args = {
            "path": "Projects/Unresolved Symlink Drift",
            "body": "unresolved body",
            "mode": "append",
        }
        contract = runtime.registry.get("write_jarvis_note").auto_mutation_contract
        if contract is None or contract.operation_key_builder is None:
            raise SystemExit("symlink-drift fixture lost operation identity")
        unresolved_owner = "symlink-drift-unresolved-owner"
        prepared = runtime.store.prepare_auto_mutation_receipts(
            unresolved_owner,
            [
                (
                    "write_jarvis_note",
                    unresolved_args,
                    contract.operation_key_builder(dict(unresolved_args)),
                )
            ],
        )
        if prepared.status != "PREPARED" or prepared.receipts[0].receipt_id is None:
            raise SystemExit(f"could not prepare symlink-drift unresolved receipt: {prepared}")
        claim = runtime.store.claim_auto_mutation_receipt(
            int(prepared.receipts[0].receipt_id), unresolved_owner
        )
        if claim.status != "CLAIMED":
            raise SystemExit(f"could not claim symlink-drift unresolved receipt: {claim}")
        unresolved_path = runtime.vault.root_path / "Projects/Unresolved Symlink Drift.md"
        unresolved_path.symlink_to(outside)
        runtime.planner = StaticPlanner(unresolved_args)
        blocked = runtime.handle(
            "cross-token request after symlink drift",
            request_token="symlink-drift-unresolved-contender",
        )
        _assert_replay(
            blocked,
            "auto_mutation_unresolved_action",
            "unresolved receipt before out-of-vault symlink drift",
        )
        if outside.read_text(encoding="utf-8") != "outside must remain unchanged\n":
            raise SystemExit("unresolved receipt contender followed a drifted out-of-vault symlink")


def test_atomic_publication_failures_preserve_prior_state() -> None:
    append_args = {
        "path": "Projects/Atomic Append",
        "body": "candidate append body",
        "mode": "append",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-atomic-append-") as temp:
        runtime = _runtime(Path(temp), append_args)
        note_path = runtime.vault.root_path / "Projects/Atomic Append.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("# Atomic Append\n\nOriginal body.\n", encoding="utf-8")
        before = note_path.read_bytes()
        original_exchange = obsidian_module._rename_exchange

        def fail_exchange(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("injected atomic append publication failure")

        obsidian_module._rename_exchange = fail_exchange
        try:
            result = runtime.handle("append with publication failure", request_token="atomic-append")
        finally:
            obsidian_module._rename_exchange = original_exchange
        if result.tool_results[0].ok or note_path.read_bytes() != before:
            raise SystemExit("failed atomic append changed the previously published note")

    create_args = {
        "path": "Projects/Atomic Create",
        "body": "candidate create body",
        "mode": "create",
    }
    with TemporaryDirectory(prefix="jarvis-note-write-atomic-create-") as temp:
        runtime = _runtime(Path(temp), create_args)
        note_path = runtime.vault.root_path / "Projects/Atomic Create.md"
        original_noreplace = obsidian_module._rename_noreplace

        def fail_noreplace(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("injected atomic create publication failure")

        obsidian_module._rename_noreplace = fail_noreplace
        try:
            result = runtime.handle("create with publication failure", request_token="atomic-create")
        finally:
            obsidian_module._rename_noreplace = original_noreplace
        if result.tool_results[0].ok or note_path.exists():
            raise SystemExit("failed atomic create left a partially published destination")


def test_committed_append_recovers_when_journal_cleanup_was_interrupted() -> None:
    with TemporaryDirectory(prefix="jarvis-note-write-journal-recovery-") as temp:
        runtime = _runtime(Path(temp), {"path": "unused", "body": "unused"})
        note_path = runtime.vault.root_path / "Projects/Journal Recovery.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("# Journal Recovery\n\nOriginal.\n", encoding="utf-8")
        original_unlink = obsidian_module.os.unlink

        def fail_journal_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
            if str(path).endswith(".exchange.json"):
                raise OSError("injected journal cleanup interruption")
            original_unlink(path, *args, **kwargs)

        obsidian_module.os.unlink = fail_journal_unlink
        try:
            try:
                runtime.vault.append_note(note_path, "Committed once.", heading="Journal Recovery")
            except OSError:
                pass
            else:
                raise SystemExit("journal cleanup failure fixture did not interrupt publication cleanup")
        finally:
            obsidian_module.os.unlink = original_unlink

        committed = note_path.read_text(encoding="utf-8")
        if committed.count("Committed once.") != 1:
            raise SystemExit("journal cleanup interruption did not leave the committed candidate visible")
        runtime.vault.append_note(note_path, "Recovered follow-up.", heading="Journal Recovery")
        recovered = note_path.read_text(encoding="utf-8")
        if recovered.count("Committed once.") != 1 or recovered.count("Recovered follow-up.") != 1:
            raise SystemExit("journal recovery duplicated or lost committed append content")
        leftovers = [
            path.name
            for path in note_path.parent.iterdir()
            if path.name.startswith(f".{note_path.name}.")
            and (path.name.endswith(".tmp") or path.name.endswith(".exchange.json"))
        ]
        if leftovers:
            raise SystemExit(f"journal recovery left atomic publication artifacts: {leftovers}")


def test_aborted_append_recovers_when_journal_cleanup_was_interrupted() -> None:
    with TemporaryDirectory(prefix="jarvis-note-write-aborted-journal-") as temp:
        runtime = _runtime(Path(temp), {"path": "unused", "body": "unused"})
        note_path = runtime.vault.root_path / "Projects/Aborted Journal Recovery.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        original = "# Aborted Journal Recovery\n\nOriginal.\n"
        candidate = original + "\n\nAborted candidate.\n"
        note_path.write_text(original, encoding="utf-8")
        token = "a" * 16
        temp_name = f".{note_path.name}.{token}.tmp"
        journal_path = note_path.parent / f".{note_path.name}.{token}.exchange.json"
        journal_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "temp_name": temp_name,
                    "expected_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                    "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

        runtime.vault.append_note(note_path, "Recovered after abort.", heading="Aborted Journal Recovery")
        recovered = note_path.read_text(encoding="utf-8")
        if (
            recovered.count("Original.") != 1
            or "Aborted candidate." in recovered
            or recovered.count("Recovered after abort.") != 1
            or journal_path.exists()
        ):
            raise SystemExit("aborted journal recovery lost, duplicated, or published the wrong content")


def main() -> None:
    test_exact_contract_schema_preflight_and_target_identity()
    test_append_success_replay_intentional_repeat_privacy_and_audit()
    test_unresolved_canonical_target_fences_changed_body_and_mode()
    test_post_append_vault_failure_is_uncertain_and_blocks_replay()
    test_completion_failure_after_append_stops_all_replay()
    test_process_death_after_append_recovers_uncertain_without_duplicate()
    test_preflight_and_typed_refusals_precede_receipts_and_writes()
    test_concurrent_same_target_operation_is_fenced()
    test_create_replay_exact_convergence_and_differing_existing_refusal()
    test_durable_receipt_state_precedes_mutable_create_preflight()
    test_managed_open_tasks_projection_is_protected()
    test_symlink_alias_and_fifo_targets_fail_closed()
    test_durable_receipts_precede_out_of_vault_symlink_drift()
    test_atomic_publication_failures_preserve_prior_state()
    test_committed_append_recovers_when_journal_cleanup_was_interrupted()
    test_aborted_append_recovers_when_journal_cleanup_was_interrupted()
    test_oversized_path_preflight_precedes_receipt_and_write()
    print("Auto-mutation note-write rollout smoke passed")


if __name__ == "__main__":
    main()
