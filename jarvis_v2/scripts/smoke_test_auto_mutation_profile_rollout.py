from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Lock
from typing import Any

import jarvis_v2.memory.obsidian as obsidian_module
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.memory.store import (
    auto_mutation_action_digest,
    auto_mutation_request_digest,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.profile import (
    MAX_PROFILE_CATEGORY_CHARS,
    MAX_PROFILE_HEADING_CHARS,
    MAX_PROFILE_WRITE_CHARS,
    PROFILE_GENERATION_STALE_ERROR,
    add_profile_note_auto_mutation_operation_key,
)
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
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
            "Exercise the aggregate-wired profile mutation.",
            [PlannedAction("add_profile_note", self.args, "profile rollout smoke")],
            needs_model=False,
        )


def _runtime(root: Path, args: Any) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(args)
    return runtime


def _crash_after_profile_append(root_text: str) -> None:
    runtime = _runtime(
        Path(root_text),
        {
            "heading": "Process crash profile note",
            "body": "Survive a process exit after projection publication.",
        },
    )
    original_append = runtime.vault.append_profile_once

    def append_then_exit(source_key: str, heading: str, body: str) -> tuple[Path, bool]:
        original_append(source_key, heading, body)
        os._exit(23)

    runtime.vault.append_profile_once = append_then_exit  # type: ignore[method-assign]
    runtime.handle(
        "crash during profile projection",
        request_token="profile-process-crash-private-owner",
    )
    os._exit(24)


def _crash_during_profile_temp_write(root_text: str) -> None:
    runtime = _runtime(
        Path(root_text),
        {
            "heading": "Atomic profile note",
            "body": "Never publish a torn profile section.",
        },
    )
    original_exchange = obsidian_module._rename_exchange

    def exit_before_atomic_swap(*_args: Any, **_kwargs: Any) -> None:
        os._exit(25)

    obsidian_module._rename_exchange = exit_before_atomic_swap
    runtime.handle(
        "crash inside atomic profile publication",
        request_token="profile-atomic-crash-private-owner",
    )
    obsidian_module._rename_exchange = original_exchange
    os._exit(26)


def _crash_after_exchange_with_external_edit(root_text: str) -> None:
    runtime = _runtime(
        Path(root_text),
        {
            "heading": "Journaled profile exchange",
            "body": "Recover the displaced external edit after process death.",
        },
    )
    original_exchange = obsidian_module._rename_exchange
    injected = [False]

    def exchange_then_exit(
        source_name: str,
        destination_name: str,
        *,
        source_dir_fd: int,
        destination_dir_fd: int,
    ) -> None:
        if destination_name == "Profile.md" and not injected[0]:
            injected[0] = True
            profile_path = runtime.vault.root_path / destination_name
            existing = profile_path.read_text(encoding="utf-8")
            profile_path.write_text(
                existing + "\n## Displaced external edit\n\nPreserve me after restart.\n",
                encoding="utf-8",
            )
        original_exchange(
            source_name,
            destination_name,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )
        os._exit(27)

    obsidian_module._rename_exchange = exchange_then_exit
    runtime.handle(
        "crash after profile exchange",
        request_token="profile-exchange-crash-private-owner",
    )
    os._exit(28)


def _receipt_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [
            dict(row)
            for row in conn.execute("SELECT * FROM auto_mutation_receipts ORDER BY id")
        ]


def _tool_run_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM tool_runs ORDER BY id")]


def _profile_source_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM ingested_sources WHERE source_type = 'profile_note' ORDER BY source_key"
            )
        ]


def _profile_memories(runtime: JarvisRuntime) -> list[Any]:
    with runtime.store.connect() as conn:
        return list(
            conn.execute(
                """
                SELECT memory.*
                FROM memories AS memory
                JOIN ingested_sources AS source ON source.memory_id = memory.id
                WHERE source.source_type = 'profile_note'
                ORDER BY memory.id
                LIMIT 100
                """
            )
        )


def _raw_profile(runtime: JarvisRuntime) -> str:
    path = runtime.vault.root_path / "Profile.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _assert_one_visible_section(runtime: JarvisRuntime, heading: str, body: str) -> None:
    raw = _raw_profile(runtime)
    visible = runtime.vault.read_profile(MAX_PROFILE_WRITE_CHARS + 1000)
    if (
        raw.count(f"## {heading}") != 1
        or raw.count(body) != 1
        or raw.count("<!-- jarvis-profile-note-start:v1:") != 1
        or raw.count("<!-- jarvis-profile-note:v1:") != 1
        or f"## {heading}" in visible
        or body in visible
        or "jarvis-profile-note:v1:" in visible
    ):
        raise SystemExit(
            "profile projection was not exactly-once on disk and hidden from the "
            f"unverified direct reader: {raw!r} / {visible!r}"
        )
    marker_start = raw.index("<!-- jarvis-profile-note:v1:")
    marker_line = raw[marker_start : raw.index("\n", marker_start)]
    original_read_text = obsidian_module._read_text
    try:
        for cut in range(1, len(marker_line)):
            truncated_source = raw[: marker_start + cut]

            def read_truncated(*_args: Any, **_kwargs: Any) -> str:
                return truncated_source

            obsidian_module._read_text = read_truncated
            partial_marker_read = runtime.vault.read_profile(MAX_PROFILE_WRITE_CHARS + 1000)
            if "<!--" in partial_marker_read or "jarvis-profile" in partial_marker_read:
                raise SystemExit(
                    f"truncated profile read exposed marker bytes at cut {cut}: {partial_marker_read!r}"
                )
    finally:
        obsidian_module._read_text = original_read_text


def _assert_replay(result: Any, failure_kind: str, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} did not return one tool result: {result.tool_results}")
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != failure_kind:
        raise SystemExit(f"{label} returned the wrong replay state: {item}")


def _assert_contract(runtime: JarvisRuntime) -> None:
    tool = runtime.registry.get("add_profile_note")
    contract = tool.auto_mutation_contract
    if (
        contract is None
        or contract.version != AUTO_MUTATION_CONTRACT_VERSION
        or contract.effects
        != frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT})
        or contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
        or contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
        or contract.operation_key_builder is None
        or contract.semantic_preflight is None
    ):
        raise SystemExit(f"add_profile_note auto-mutation contract drifted: {contract}")

    argument_contract = tool.argument_contract
    if argument_contract is None or argument_contract.allow_unknown:
        raise SystemExit("add_profile_note lost its strict argument contract")
    shape = tuple(
        (field.name, field.types, field.required) for field in argument_contract.fields
    )
    string = frozenset({ToolArgumentType.STRING})
    if shape != (
        ("body", string, True),
        ("heading", string, False),
        ("category", string, False),
    ):
        raise SystemExit(f"add_profile_note strict argument shape drifted: {shape}")


def _install_counted_handler(runtime: JarvisRuntime) -> list[int]:
    tool = runtime.registry.get("add_profile_note")
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools["add_profile_note"] = replace(tool, handler=counted)
    return calls


def _assert_completed_receipt_audits(runtime: JarvisRuntime, expected: int) -> None:
    receipts = _receipt_rows(runtime)
    runs = _tool_run_rows(runtime)
    successful_runs = [row for row in runs if row["ok"] == 1]
    if len(receipts) != expected or len(successful_runs) != expected:
        raise SystemExit(f"expected {expected} receipt/audit pairs: {receipts} / {runs}")
    runs_by_id = {row["id"]: row for row in successful_runs}
    linked: set[int] = set()
    for receipt in receipts:
        run = runs_by_id.get(receipt["tool_run_id"])
        if (
            receipt["state"] != "completed"
            or receipt["result"] != "succeeded"
            or run is None
            or run["ok"] != 1
            or run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
        ):
            raise SystemExit(f"profile receipt did not link one ordinary approved=0 audit: {receipt} / {run}")
        linked.add(int(run["id"]))
    if len(linked) != expected:
        raise SystemExit(f"profile receipts reused an audit row: {linked}")


def _assert_private_evidence_not_exposed(
    runtime: JarvisRuntime,
    results: list[Any],
    *,
    request_tokens: list[str],
    actions: list[dict[str, Any]],
) -> None:
    receipts = _receipt_rows(runtime)
    receipt_text = json.dumps(receipts, ensure_ascii=False, sort_keys=True, default=str)
    raw_action_values = [
        value
        for action in actions
        for value in action.values()
        if type(value) is str and len(value) >= 12
    ]
    for private in [*request_tokens, *raw_action_values]:
        if private in receipt_text:
            raise SystemExit("profile receipt stored raw request or profile content")

    private_values = [auto_mutation_request_digest(token) for token in request_tokens]
    private_values.extend(
        auto_mutation_action_digest("add_profile_note", action) for action in actions
    )
    private_values.extend(
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
    exposed_items = []
    for result in results:
        tool_metadata = dict(result.tool_results[0].metadata)
        if tool_metadata.pop("path", None) is not None:
            raise SystemExit("profile success metadata exposed a raw local path")
        exposed_items.append(
            {
                "response": result.response,
                "runtime_metadata": result.metadata,
                "tool_output": result.tool_results[0].output,
                "tool_metadata_without_internal_path": tool_metadata,
            }
        )
    exposed = json.dumps(
        exposed_items,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    audits = json.dumps(_tool_run_rows(runtime), ensure_ascii=False, sort_keys=True, default=str)
    for private in [*request_tokens, *private_values]:
        if private and (private in exposed or private in audits):
            raise SystemExit(f"profile runtime/audit exposed private mutation evidence: {private!r}")
    if str(runtime.vault.root_path) in exposed or str(runtime.store.db_path.parent) in exposed:
        raise SystemExit("profile runtime result exposed an absolute temporary path")


def test_contract_replay_convergence_privacy_and_audit() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-rollout-success-") as temp:
        runtime = _runtime(Path(temp), {"body": "  Keep status updates brief.  "})
        _assert_contract(runtime)
        token_one = "profile-success-private-token-one"
        token_two = "profile-success-private-token-two"

        first = runtime.handle("first profile note", request_token=token_one)
        if not first.tool_results[0].ok:
            raise SystemExit(f"first profile mutation failed: {first.tool_results}")
        _assert_one_visible_section(runtime, "Profile Note", "Keep status updates brief.")
        if len(_profile_memories(runtime)) != 1:
            raise SystemExit("first profile mutation did not create exactly one memory")

        same = runtime.handle("same request replay", request_token=token_one)
        _assert_replay(same, "auto_mutation_completed_replay", "same-token profile replay")
        _assert_one_visible_section(runtime, "Profile Note", "Keep status updates brief.")
        if len(_profile_memories(runtime)) != 1:
            raise SystemExit("same-token profile replay changed the memory count")

        equivalent = {
            "heading": "  Profile   Note ",
            "body": "Keep status updates brief.",
            "category": " identity ",
        }
        runtime.planner = StaticPlanner(equivalent)
        fresh = runtime.handle("fresh equivalent profile note", request_token=token_two)
        fresh_item = fresh.tool_results[0]
        if (
            not fresh_item.ok
            or fresh_item.metadata.get("converged") is not True
            or fresh_item.metadata.get("source_new") is not False
            or fresh_item.metadata.get("profile_appended") is not False
        ):
            raise SystemExit(f"fresh equivalent profile request did not converge: {fresh_item}")
        _assert_one_visible_section(runtime, "Profile Note", "Keep status updates brief.")
        sources = _profile_source_rows(runtime)
        if len(_profile_memories(runtime)) != 1 or len(sources) != 1 or sources[0]["mirror_state"] != "completed":
            raise SystemExit(f"converged profile mutation duplicated durable state: {sources}")
        _assert_completed_receipt_audits(runtime, 2)
        _assert_private_evidence_not_exposed(
            runtime,
            [first, same, fresh],
            request_tokens=[token_one, token_two],
            actions=[{"body": "  Keep status updates brief.  "}, equivalent],
        )


def test_typed_and_semantic_preflight_before_receipts_or_handler() -> None:
    semantic_cases = [
        ({"body": ""}, "missing_body"),
        ({"body": "   \r\n  "}, "missing_body"),
        ({"body": "x" * (MAX_PROFILE_WRITE_CHARS + 1)}, "oversized_body"),
        ({"heading": "/tmp/private-heading", "body": "safe"}, "invalid_heading"),
        ({"category": "/private/tmp/private-category", "body": "safe"}, "invalid_category"),
        ({"body": "read /\x55sers/example/private/profile"}, "invalid_body"),
        ({"body": "reserved jarvis-profile-note:v1: namespace"}, "reserved_marker_namespace"),
    ]
    typed_cases: list[Any] = [
        {},
        {"body": None},
        {"body": 1},
        {"body": True},
        {"body": []},
        {"body": {}},
        {"body": "safe", "heading": 1},
        {"body": "safe", "category": False},
        {"body": "safe", "unknown": "rejected"},
        None,
        [],
    ]

    for index, (args, reason) in enumerate(semantic_cases):
        with TemporaryDirectory(prefix="jarvis-profile-rollout-semantic-") as temp:
            runtime = _runtime(Path(temp), args)
            calls = _install_counted_handler(runtime)
            result = runtime.handle("invalid semantic profile note", request_token=f"semantic-{index}")
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
                or item.metadata.get("reason") != reason
                or item.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(f"profile semantic preflight case {index} drifted: {item}")
            if calls[0] or _receipt_rows(runtime) or _profile_memories(runtime) or _profile_source_rows(runtime):
                raise SystemExit(f"profile semantic preflight case {index} crossed the write boundary")
            exposed = json.dumps(
                {"response": result.response, "output": item.output, "metadata": item.metadata},
                default=str,
            )
            if any(path in exposed for path in ("/\x55sers/", "/private/", "/tmp/")):
                raise SystemExit("profile semantic preflight exposed a local path")

    for index, args in enumerate(typed_cases):
        with TemporaryDirectory(prefix="jarvis-profile-rollout-typed-") as temp:
            runtime = _runtime(Path(temp), args)
            calls = _install_counted_handler(runtime)
            result = runtime.handle("invalid typed profile note", request_token=f"typed-{index}")
            item = result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(f"profile strict argument case {index} drifted: {item}")
            if calls[0] or _receipt_rows(runtime) or _profile_memories(runtime) or _profile_source_rows(runtime):
                raise SystemExit(f"profile strict argument case {index} crossed the write boundary")


def _install_append_then_fail(runtime: JarvisRuntime) -> tuple[list[int], Any]:
    original = runtime.vault.append_profile_once
    calls = [0]

    def append_then_fail(source_key: str, heading: str, body: str) -> tuple[Path, bool]:
        calls[0] += 1
        original(source_key, heading, body)
        raise OSError("representative profile mirror completion failure")

    runtime.vault.append_profile_once = append_then_fail  # type: ignore[method-assign]
    return calls, original


def _assert_uncertain_profile_state(
    runtime: JarvisRuntime,
    label: str,
    *,
    source_state: str = "pending",
) -> int:
    receipts = _receipt_rows(runtime)
    sources = _profile_source_rows(runtime)
    if (
        len(receipts) != 1
        or receipts[0]["state"] != "uncertain"
        or receipts[0]["result"] != "unknown"
        or len(sources) != 1
        or sources[0]["mirror_state"] != source_state
        or len(_profile_memories(runtime)) != 1
    ):
        raise SystemExit(f"{label} did not retain one pending source and uncertain receipt: {receipts} / {sources}")
    if any(row["ok"] == 1 for row in _tool_run_rows(runtime)):
        raise SystemExit(f"{label} created a successful ordinary audit")
    return int(receipts[0]["id"])


def _repair_profile_projection(runtime: JarvisRuntime, label: str) -> None:
    repaired = runtime.registry.get("repair_memory_projections").handler({"limit": 20})
    if (
        not repaired.ok
        or repaired.metadata.get("pending") != 0
        or "profile:completed" not in repaired.metadata.get("outcome_statuses", [])
    ):
        raise SystemExit(f"{label} did not repair reviewed profile custody: {repaired}")


def test_uncertain_mirror_reconciliation_repairs_once() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-rollout-recovery-") as temp:
        runtime = _runtime(Path(temp), {"body": "  Recover this profile mirror. \r\n"})
        append_calls, original_append = _install_append_then_fail(runtime)
        owner = "profile-recovery-private-owner"
        equivalent_token = "profile-recovery-private-equivalent"

        failed = runtime.handle("profile mirror crash", request_token=owner)
        if failed.tool_results[0].metadata.get("failure_kind") != "tool_error":
            raise SystemExit(f"profile mirror crash did not become uncertain: {failed.tool_results}")
        receipt_id = _assert_uncertain_profile_state(runtime, "profile mirror crash")
        _assert_one_visible_section(runtime, "Profile Note", "Recover this profile mirror.")
        if append_calls[0] != 1:
            raise SystemExit("profile mirror crash did not execute exactly one append attempt")

        same = runtime.handle("same uncertain request", request_token=owner)
        _assert_replay(same, "auto_mutation_outcome_uncertain", "same-token uncertain profile replay")
        runtime.planner = StaticPlanner(
            {
                "heading": " Profile Note ",
                "category": "identity",
                "body": "Recover this profile mirror.",
            }
        )
        equivalent = runtime.handle("fresh equivalent while unresolved", request_token=equivalent_token)
        _assert_replay(equivalent, "auto_mutation_unresolved_action", "cross-token uncertain profile replay")
        if append_calls[0] != 1 or len(_profile_memories(runtime)) != 1:
            raise SystemExit("unresolved equivalent profile request reran the handler")

        runtime.vault.append_profile_once = original_append  # type: ignore[method-assign]
        inspected = runtime.handle(f"uncertain mutation {receipt_id}")
        if inspected.tool_results[0].metadata.get("review_receipt_created") is not True:
            raise SystemExit("profile uncertain receipt inspection did not create review evidence")
        resolved = runtime.handle(f"resolve uncertain mutation {receipt_id} as applied")
        if (
            not resolved.verified
            or resolved.tool_results[0].metadata.get("reconciliation_status") != "resolved"
            or resolved.tool_results[0].metadata.get("reconciliation_disposition")
            != "confirmed_applied"
        ):
            raise SystemExit(f"profile uncertain receipt did not reconcile as applied: {resolved.tool_results}")

        repaired_token = "profile-recovery-private-repaired"
        repaired = runtime.handle("repair pending profile mirror", request_token=repaired_token)
        item = repaired.tool_results[0]
        if (
            not item.ok
            or item.metadata.get("source_new") is not False
            or item.metadata.get("profile_appended") is not False
            or item.metadata.get("profile_mirror_completed") is not True
            or item.metadata.get("state_changed") is not True
            or item.metadata.get("changed") != ["profile_mirror"]
            or item.metadata.get("converged") is not False
            or item.metadata.get("writes_database") is not True
            or item.metadata.get("writes_files") is not False
            or item.metadata.get("writes_notes") is not False
            or item.metadata.get("writes_memory") is not False
        ):
            raise SystemExit(f"profile pending mirror did not converge after reconciliation: {item}")
        sources = _profile_source_rows(runtime)
        if len(sources) != 1 or sources[0]["mirror_state"] != "completed":
            raise SystemExit(f"profile pending mirror did not complete: {sources}")
        if len(_profile_memories(runtime)) != 1:
            raise SystemExit("profile pending mirror repair created a second database row")
        _assert_one_visible_section(runtime, "Profile Note", "Recover this profile mirror.")
        _assert_private_evidence_not_exposed(
            runtime,
            [failed, same, equivalent, inspected, resolved, repaired],
            request_tokens=[owner, equivalent_token, repaired_token],
            actions=[
                {"body": "  Recover this profile mirror. \r\n"},
                {
                    "heading": " Profile Note ",
                    "category": "identity",
                    "body": "Recover this profile mirror.",
                },
            ],
        )


def test_truncated_labels_share_unresolved_operation_identity() -> None:
    heading_prefix = "H" * (MAX_PROFILE_HEADING_CHARS + 20)
    category_prefix = "C" * (MAX_PROFILE_CATEGORY_CHARS + 20)
    first_args = {
        "heading": heading_prefix + " FIRST",
        "category": category_prefix + " FIRST",
        "body": "  Canonical truncated identity.  ",
    }
    second_args = {
        "heading": heading_prefix + " SECOND",
        "category": category_prefix + " SECOND",
        "body": "Canonical truncated identity.",
    }
    if (
        add_profile_note_auto_mutation_operation_key(first_args)
        != add_profile_note_auto_mutation_operation_key(second_args)
    ):
        raise SystemExit("profile truncated-label fixture did not normalize to one operation")

    with TemporaryDirectory(prefix="jarvis-profile-rollout-truncated-") as temp:
        runtime = _runtime(Path(temp), first_args)
        append_calls, original_append = _install_append_then_fail(runtime)
        failed = runtime.handle("long-label profile crash", request_token="profile-long-one")
        runtime.vault.append_profile_once = original_append  # type: ignore[method-assign]
        if failed.tool_results[0].metadata.get("failure_kind") != "tool_error":
            raise SystemExit("long-label profile fixture did not become uncertain")
        runtime.planner = StaticPlanner(second_args)
        blocked = runtime.handle("equivalent long-label retry", request_token="profile-long-two")
        _assert_replay(blocked, "auto_mutation_unresolved_action", "truncated-label profile replay")
        if append_calls[0] != 1 or len(_profile_memories(runtime)) != 1:
            raise SystemExit("truncated-label equivalent request bypassed unresolved operation fencing")


def test_process_exit_after_projection_recovers_without_duplicate() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-rollout-process-crash-") as temp:
        root = Path(temp)
        process = multiprocessing.get_context("spawn").Process(
            target=_crash_after_profile_append,
            args=(str(root),),
        )
        process.start()
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(10)
            raise SystemExit("profile process-crash child did not exit")
        if process.exitcode != 23:
            raise SystemExit(f"profile process-crash child exited unexpectedly: {process.exitcode}")

        args = {
            "heading": "Process crash profile note",
            "body": "Survive a process exit after projection publication.",
        }
        runtime = _runtime(root, args)
        receipts = _receipt_rows(runtime)
        if len(receipts) != 1 or receipts[0]["state"] != "running":
            raise SystemExit(f"process crash did not leave one running receipt: {receipts}")
        if runtime.store.recover_stale_auto_mutation_receipts("9999-12-31T23:59:59Z") != 1:
            raise SystemExit("process-crash receipt was not recovered as uncertain")
        receipt_id = _assert_uncertain_profile_state(
            runtime,
            "profile process crash",
            source_state="pending",
        )

        blocked = runtime.handle(
            "retry process-crashed profile note",
            request_token="profile-process-crash-private-retry",
        )
        _assert_replay(blocked, "auto_mutation_unresolved_action", "process-crash profile replay")
        runtime.handle(f"uncertain mutation {receipt_id}")
        resolved = runtime.handle(f"resolve uncertain mutation {receipt_id} as applied")
        if (
            not resolved.verified
            or resolved.tool_results[0].metadata.get("reconciliation_disposition")
            != "confirmed_applied"
        ):
            raise SystemExit("process-crash profile receipt did not reconcile as applied")
        _repair_profile_projection(runtime, "process-crash profile")

        _assert_one_visible_section(
            runtime,
            "Process crash profile note",
            "Survive a process exit after projection publication.",
        )
        if (
            len(_profile_memories(runtime)) != 1
            or len(_profile_source_rows(runtime)) != 1
            or runtime.store.count_pending_profile_projection_sources() != 0
        ):
            raise SystemExit("process-crash recovery duplicated profile durable state")


def test_process_exit_during_atomic_write_never_publishes_torn_section() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-rollout-atomic-crash-") as temp:
        root = Path(temp)
        process = multiprocessing.get_context("spawn").Process(
            target=_crash_during_profile_temp_write,
            args=(str(root),),
        )
        process.start()
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(10)
            raise SystemExit("atomic profile crash child did not exit")
        if process.exitcode != 25:
            raise SystemExit(f"atomic profile crash child exited unexpectedly: {process.exitcode}")

        args = {
            "heading": "Atomic profile note",
            "body": "Never publish a torn profile section.",
        }
        runtime = _runtime(root, args)
        raw_before = _raw_profile(runtime)
        if "Atomic profile note" in raw_before or "Never publish a torn profile section." in raw_before:
            raise SystemExit(f"atomic profile crash published an incomplete section: {raw_before!r}")
        if runtime.store.recover_stale_auto_mutation_receipts("9999-12-31T23:59:59Z") != 1:
            raise SystemExit("atomic profile crash receipt was not recovered as uncertain")
        receipt_id = _assert_uncertain_profile_state(
            runtime,
            "atomic profile crash",
            source_state="pending",
        )
        runtime.handle(f"uncertain mutation {receipt_id}")
        runtime.handle(f"resolve uncertain mutation {receipt_id} as applied")
        _repair_profile_projection(runtime, "atomic profile crash")
        _assert_one_visible_section(
            runtime,
            "Atomic profile note",
            "Never publish a torn profile section.",
        )
        leftovers = list((runtime.vault.root_path).glob(".Profile.md.*.tmp"))
        if leftovers:
            raise SystemExit("atomic profile recovery left stale publication temp files")


def test_exchange_journal_restores_displaced_edit_after_process_exit() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-rollout-exchange-journal-") as temp:
        root = Path(temp)
        process = multiprocessing.get_context("spawn").Process(
            target=_crash_after_exchange_with_external_edit,
            args=(str(root),),
        )
        process.start()
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(10)
            raise SystemExit("journaled profile exchange child did not exit")
        if process.exitcode != 27:
            raise SystemExit(f"journaled profile exchange child exited unexpectedly: {process.exitcode}")

        args = {
            "heading": "Journaled profile exchange",
            "body": "Recover the displaced external edit after process death.",
        }
        runtime = _runtime(root, args)
        if runtime.store.recover_stale_auto_mutation_receipts("9999-12-31T23:59:59Z") != 1:
            raise SystemExit("journaled profile exchange receipt was not recovered as uncertain")
        receipt_id = _assert_uncertain_profile_state(
            runtime,
            "journaled profile exchange",
            source_state="pending",
        )
        runtime.handle(f"uncertain mutation {receipt_id}")
        runtime.handle(f"resolve uncertain mutation {receipt_id} as applied")
        _repair_profile_projection(runtime, "journaled profile exchange")
        raw = _raw_profile(runtime)
        if "Preserve me after restart." not in raw:
            raise SystemExit("journaled profile exchange lost the displaced external edit")
        _assert_one_visible_section(runtime, args["heading"], args["body"])
        if list(runtime.vault.root_path.glob(".Profile.md.*.exchange.json")):
            raise SystemExit("journaled profile exchange recovery left an active journal")


def test_post_exchange_exception_preserves_displaced_edit() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-rollout-post-exchange-error-") as temp:
        args = {
            "heading": "Post-exchange recovery",
            "body": "Keep displaced content when exchange returns with an error.",
        }
        runtime = _runtime(Path(temp), args)
        original_exchange = obsidian_module._rename_exchange
        injected = [False]

        def exchange_then_raise(
            source_name: str,
            destination_name: str,
            *,
            source_dir_fd: int,
            destination_dir_fd: int,
        ) -> None:
            if destination_name == "Profile.md" and not injected[0]:
                injected[0] = True
                profile_path = runtime.vault.root_path / destination_name
                existing = profile_path.read_text(encoding="utf-8")
                profile_path.write_text(
                    existing + "\n## Exchange wrapper edit\n\nKeep after raised exchange.\n",
                    encoding="utf-8",
                )
                original_exchange(
                    source_name,
                    destination_name,
                    source_dir_fd=source_dir_fd,
                    destination_dir_fd=destination_dir_fd,
                )
                raise OSError("exchange wrapper raised after the atomic swap")
            original_exchange(
                source_name,
                destination_name,
                source_dir_fd=source_dir_fd,
                destination_dir_fd=destination_dir_fd,
            )

        obsidian_module._rename_exchange = exchange_then_raise
        try:
            failed = runtime.handle(
                "raise after profile exchange",
                request_token="profile-post-exchange-private-owner",
            )
        finally:
            obsidian_module._rename_exchange = original_exchange
        if failed.tool_results[0].metadata.get("failure_kind") != "tool_error":
            raise SystemExit("post-exchange exception did not stop as uncertain")
        receipt_id = _assert_uncertain_profile_state(runtime, "post-exchange exception")
        runtime.handle(f"uncertain mutation {receipt_id}")
        runtime.handle(f"resolve uncertain mutation {receipt_id} as applied")
        repaired = runtime.handle(
            "recover after raised profile exchange",
            request_token="profile-post-exchange-private-repair",
        )
        if not repaired.tool_results[0].ok:
            raise SystemExit("post-exchange exception did not recover")
        raw = _raw_profile(runtime)
        if "Keep after raised exchange." not in raw:
            raise SystemExit("post-exchange exception deleted the displaced edit")
        _assert_one_visible_section(runtime, args["heading"], args["body"])


def test_completed_profile_evidence_repairs_both_stores() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-rollout-completed-repair-") as temp:
        args = {
            "heading": "Canonical profile evidence",
            "body": "Keep both durable stores convergent.\n\n## Details\n\nSubheading body.",
            "category": "Identity",
        }
        runtime = _runtime(Path(temp), args)
        first = runtime.handle("create canonical profile evidence", request_token="profile-evidence-one")
        if not first.tool_results[0].ok:
            raise SystemExit("completed-evidence fixture did not create")
        memory_id = int(_profile_memories(runtime)[0]["id"])

        with runtime.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            mutated = conn.execute(
                """
                UPDATE memories
                SET category = 'drifted', title = 'Mutated title',
                    body = 'Mutated body', confidence = 0.2,
                    revision = revision + 1
                WHERE id = ?
                """,
                (memory_id,),
            )
            if mutated.rowcount != 1:
                raise SystemExit("completed-evidence fixture could not corrupt linked memory")
        repaired_memory = runtime.handle(
            "repair canonical profile memory",
            request_token="profile-evidence-two",
        )
        memory_item = repaired_memory.tool_results[0]
        with runtime.store.connect() as conn:
            memory = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if (
            memory is None
            or memory["category"] != "identity"
            or memory["title"] != args["heading"]
            or memory["body"] != args["body"]
            or float(memory["confidence"]) != 1.0
            or memory_item.metadata.get("memory_repaired") is not True
            or memory_item.metadata.get("writes_memory") is not True
            or "Restored the memory index" not in memory_item.output
        ):
            raise SystemExit(f"completed source did not repair canonical memory: {memory_item} / {memory}")

        profile_path = runtime.vault.root_path / "Profile.md"
        profile_path.write_text("# Profile\n\n", encoding="utf-8")
        repaired_projection = runtime.handle(
            "repair canonical profile projection",
            request_token="profile-evidence-three",
        )
        projection_item = repaired_projection.tool_results[0]
        if (
            not projection_item.ok
            or projection_item.metadata.get("profile_appended") is not True
            or projection_item.metadata.get("writes_files") is not True
            or projection_item.metadata.get("writes_memory") is not False
            or "Restored profile note" not in projection_item.output
        ):
            raise SystemExit(f"completed source did not repair missing projection: {projection_item}")
        _assert_one_visible_section(runtime, args["heading"], args["body"])

        altered = _raw_profile(runtime).replace(
            args["body"],
            "A manually altered body that retained the internal marker.",
        )
        profile_path.write_text(altered, encoding="utf-8")
        repaired_owned_section = runtime.handle(
            "repair altered marker-owned profile projection",
            request_token="profile-evidence-four",
        )
        owned_item = repaired_owned_section.tool_results[0]
        repaired_raw = _raw_profile(runtime)
        if (
            not owned_item.ok
            or owned_item.metadata.get("profile_appended") is not True
            or "manually altered body" in repaired_raw
            or repaired_raw.count("<!-- jarvis-profile-note:v1:") != 1
        ):
            raise SystemExit(f"altered marker-owned section did not repair in place: {owned_item}")
        _assert_one_visible_section(runtime, args["heading"], args["body"])


def test_concurrent_external_profile_edit_is_preserved() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-rollout-external-edit-") as temp:
        args = {
            "heading": "Concurrent profile publication",
            "body": "Preserve an external Obsidian edit.",
        }
        runtime = _runtime(Path(temp), args)
        original_exchange = obsidian_module._rename_exchange
        exchange_calls = [0]

        def exchange_after_external_edit(
            source_name: str,
            destination_name: str,
            *,
            source_dir_fd: int,
            destination_dir_fd: int,
        ) -> None:
            if destination_name == "Profile.md":
                exchange_calls[0] += 1
                profile_path = runtime.vault.root_path / destination_name
                existing = profile_path.read_text(encoding="utf-8")
                if exchange_calls[0] == 1:
                    profile_path.write_text(
                        existing + "\n## External edit\n\nKeep this text.\n",
                        encoding="utf-8",
                    )
                elif exchange_calls[0] == 2:
                    profile_path.write_text(
                        existing + "\n## Second external edit\n\nQuarantine, do not delete.\n",
                        encoding="utf-8",
                    )
            original_exchange(
                source_name,
                destination_name,
                source_dir_fd=source_dir_fd,
                destination_dir_fd=destination_dir_fd,
            )

        obsidian_module._rename_exchange = exchange_after_external_edit
        try:
            failed = runtime.handle(
                "profile write racing an external edit",
                request_token="profile-external-edit-private-owner",
            )
        finally:
            obsidian_module._rename_exchange = original_exchange
        if failed.tool_results[0].metadata.get("failure_kind") != "tool_error":
            raise SystemExit("concurrent external profile edit did not stop publication")
        receipt_id = _assert_uncertain_profile_state(runtime, "concurrent external profile edit")
        raw = _raw_profile(runtime)
        if "Keep this text." not in raw or args["body"] in raw:
            raise SystemExit("concurrent external profile edit was overwritten")
        conflict_files = [
            path
            for path in runtime.vault.root_path.glob(".Profile.md.jarvis-conflict-*")
            if not path.name.endswith(".exchange.json")
        ]
        if not conflict_files or not any(
            "Quarantine, do not delete." in path.read_text(encoding="utf-8")
            for path in conflict_files
        ):
            raise SystemExit("second external edit was not preserved in conflict quarantine")

        runtime.handle(f"uncertain mutation {receipt_id}")
        runtime.handle(f"resolve uncertain mutation {receipt_id} as applied")
        repaired = runtime.handle(
            "retry profile after external edit",
            request_token="profile-external-edit-private-repair",
        )
        if not repaired.tool_results[0].ok:
            raise SystemExit("profile did not recover after concurrent external edit")
        raw = _raw_profile(runtime)
        if "Keep this text." not in raw:
            raise SystemExit("profile recovery discarded the external edit")
        _assert_one_visible_section(runtime, args["heading"], args["body"])


def test_expanding_unicode_category_normalizes_before_bound() -> None:
    folded = {
        "heading": "Unicode category identity",
        "body": "Category case folding must happen before truncation.",
        "category": "ß" * 40,
    }
    expanded = {**folded, "category": "ss" * 40}
    if add_profile_note_auto_mutation_operation_key(folded) != add_profile_note_auto_mutation_operation_key(expanded):
        raise SystemExit("expanding Unicode category variants did not share operation identity")

    with TemporaryDirectory(prefix="jarvis-profile-rollout-unicode-category-") as temp:
        runtime = _runtime(Path(temp), folded)
        first = runtime.handle("folded Unicode category", request_token="profile-unicode-one")
        runtime.planner = StaticPlanner(expanded)
        second = runtime.handle("expanded Unicode category", request_token="profile-unicode-two")
        if not first.tool_results[0].ok or second.tool_results[0].metadata.get("converged") is not True:
            raise SystemExit("expanding Unicode category variants did not converge")
        if len(_profile_memories(runtime)) != 1 or len(_profile_source_rows(runtime)) != 1:
            raise SystemExit("expanding Unicode category variants duplicated durable state")


def test_concurrent_category_case_variants_converge() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-rollout-category-case-") as temp:
        runtime = make_temp_runtime(Path(temp))
        handler = runtime.registry.get("add_profile_note").handler
        first_generation_barrier = Barrier(2)
        trace_lock = Lock()
        generations: list[int] = []
        completion_attempts: list[tuple[int | None, str, Any]] = []
        reopened_generations: list[int | None] = []
        original_ensure = runtime.store.ensure_profile_note_memory_with_projection
        original_complete = runtime.store.complete_ingested_source_projection
        original_mark_pending = runtime.store.mark_ingested_source_projection_pending

        def synchronized_ensure(record, source_key):
            result = original_ensure(record, source_key)
            with trace_lock:
                generations.append(result[2].attempt_generation)
                call_number = len(generations)
            if call_number <= 2:
                first_generation_barrier.wait(timeout=5.0)
            return result

        def traced_complete(
            source_key: str,
            memory_id: int,
            expected_revision: int,
            expected_source_digest: str,
            *,
            expected_generation: int | None = None,
        ) -> dict[str, Any]:
            try:
                result = original_complete(
                    source_key,
                    memory_id,
                    expected_revision,
                    expected_source_digest,
                    expected_generation=expected_generation,
                )
            except RuntimeError as exc:
                with trace_lock:
                    completion_attempts.append(
                        (expected_generation, "error", str(exc))
                    )
                raise
            with trace_lock:
                completion_attempts.append(
                    (expected_generation, "ok", result["completed_now"] is True)
                )
            return result

        def traced_mark_pending(
            source_key: str,
            memory_id: int,
            expected_revision: int,
            expected_source_digest: str,
            *,
            expected_generation: int | None = None,
        ) -> bool:
            with trace_lock:
                reopened_generations.append(expected_generation)
            return original_mark_pending(
                source_key,
                memory_id,
                expected_revision,
                expected_source_digest,
                expected_generation=expected_generation,
            )

        runtime.store.ensure_profile_note_memory_with_projection = synchronized_ensure  # type: ignore[method-assign]
        runtime.store.complete_ingested_source_projection = traced_complete  # type: ignore[method-assign]
        runtime.store.mark_ingested_source_projection_pending = traced_mark_pending  # type: ignore[method-assign]
        base = {
            "heading": "Casefolded category identity",
            "body": "Concurrent category variants must publish once.",
        }

        def run(category: str) -> ToolResult:
            return handler({**base, "category": category})

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, ("identity", "IDENTITY")))
        if not all(result.ok for result in results):
            raise SystemExit(f"concurrent profile category variants failed: {results}")
        with trace_lock:
            generation_snapshot = list(generations)
            completion_snapshot = list(completion_attempts)
            reopened_snapshot = list(reopened_generations)
        if (
            sorted(generation_snapshot[:2]) != [1, 2]
            or not 3 <= len(generation_snapshot) <= 4
            or reopened_snapshot
            or not any(
                status == "error" and detail == PROFILE_GENERATION_STALE_ERROR
                for _generation, status, detail in completion_snapshot
            )
        ):
            raise SystemExit(
                "case-variant profile concurrency did not exercise one bounded stale-generation retry: "
                f"{generation_snapshot} / {completion_snapshot} / {reopened_snapshot}"
            )
        sources = _profile_source_rows(runtime)
        latest_generation = max(generation_snapshot)
        if len(_profile_memories(runtime)) != 1 or len(sources) != 1:
            raise SystemExit("case-variant profile categories created duplicate database identity")
        if (
            sources[0]["mirror_state"] != "completed"
            or sources[0]["mirror_generation"] != latest_generation
            or not any(
                generation == latest_generation and status == "ok"
                for generation, status, _detail in completion_snapshot
            )
        ):
            raise SystemExit(
                "case-variant profile concurrency did not complete current-generation custody: "
                f"{sources} / {completion_snapshot}"
            )
        _assert_one_visible_section(
            runtime,
            "Casefolded category identity",
            "Concurrent category variants must publish once.",
        )
        if (
            sum(result.metadata.get("memory_created") is True for result in results) != 1
            or sum(result.metadata.get("profile_appended") is True for result in results) != 1
            or sum(result.metadata.get("profile_mirror_completed") is True for result in results) != 1
            or not all(result.metadata.get("state_changed") is True for result in results if not result.metadata.get("converged"))
        ):
            raise SystemExit("case-variant profile concurrency did not publish each durable effect once")
        for result in results:
            metadata = result.metadata
            if (
                (metadata.get("memory_created") is True or metadata.get("memory_repaired") is True)
                and (
                    metadata.get("writes_memory") is not True
                    or metadata.get("writes_database") is not True
                )
            ):
                raise SystemExit(f"concurrent memory writes were not accumulated truthfully: {result}")
            if metadata.get("profile_appended") is True and (
                metadata.get("writes_files") is not True
                or metadata.get("writes_notes") is not True
            ):
                raise SystemExit(f"concurrent profile writes were not accumulated truthfully: {result}")
            if metadata.get("profile_mirror_completed") is True and metadata.get("writes_database") is not True:
                raise SystemExit(f"concurrent mirror completion was not accumulated truthfully: {result}")


def test_deleted_profile_memory_is_recreated_without_duplicate_projection() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-rollout-deleted-memory-") as temp:
        runtime = _runtime(
            Path(temp),
            {"heading": "Durable profile identity", "body": "Recreate a deleted index row."},
        )
        first = runtime.handle("create profile before deletion", request_token="profile-delete-one")
        if not first.tool_results[0].ok:
            raise SystemExit("profile deletion fixture did not create its first note")
        original_memory_id = int(_profile_memories(runtime)[0]["id"])
        if runtime.store.delete_memory(original_memory_id):
            raise SystemExit("profile-owned memory was not protected from generic deletion")
        with runtime.store.connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (original_memory_id,))
        source_after_delete = _profile_source_rows(runtime)[0]
        if source_after_delete["memory_id"] is not None:
            raise SystemExit("profile source did not expose its deleted memory link")

        repaired = runtime.handle("recreate deleted profile index", request_token="profile-delete-two")
        item = repaired.tool_results[0]
        memories = _profile_memories(runtime)
        if (
            not item.ok
            or item.metadata.get("source_new") is not True
            or item.metadata.get("profile_appended") is not False
            or item.metadata.get("profile_mirror_completed") is not True
            or len(memories) != 1
            or int(memories[0]["id"]) == original_memory_id
        ):
            raise SystemExit(f"deleted profile memory was not repaired truthfully: {item} / {memories}")
        source = _profile_source_rows(runtime)[0]
        if source["memory_id"] != memories[0]["id"] or source["mirror_state"] != "completed":
            raise SystemExit("recreated profile memory did not relink its source ledger")
        _assert_one_visible_section(
            runtime,
            "Durable profile identity",
            "Recreate a deleted index row.",
        )

        repaired_memory_id = int(memories[0]["id"])
        with sqlite3.connect(runtime.store.db_path) as legacy_conn:
            legacy_conn.execute("PRAGMA foreign_keys = OFF")
            legacy_conn.execute("DELETE FROM memories WHERE id = ?", (repaired_memory_id,))
        stale_source = _profile_source_rows(runtime)[0]
        if stale_source["memory_id"] != repaired_memory_id:
            raise SystemExit("legacy stale-link fixture did not preserve its missing memory id")
        repaired_legacy = runtime.handle(
            "repair legacy stale profile index",
            request_token="profile-delete-three",
        )
        legacy_item = repaired_legacy.tool_results[0]
        legacy_memories = _profile_memories(runtime)
        if (
            not legacy_item.ok
            or legacy_item.metadata.get("source_new") is not True
            or len(legacy_memories) != 1
            or int(legacy_memories[0]["id"]) == repaired_memory_id
        ):
            raise SystemExit("legacy stale profile memory link was not recreated")
        _assert_one_visible_section(
            runtime,
            "Durable profile identity",
            "Recreate a deleted index row.",
        )


def main() -> None:
    test_contract_replay_convergence_privacy_and_audit()
    test_typed_and_semantic_preflight_before_receipts_or_handler()
    test_uncertain_mirror_reconciliation_repairs_once()
    test_truncated_labels_share_unresolved_operation_identity()
    test_process_exit_after_projection_recovers_without_duplicate()
    test_process_exit_during_atomic_write_never_publishes_torn_section()
    test_exchange_journal_restores_displaced_edit_after_process_exit()
    test_post_exchange_exception_preserves_displaced_edit()
    test_completed_profile_evidence_repairs_both_stores()
    test_concurrent_external_profile_edit_is_preserved()
    test_expanding_unicode_category_normalizes_before_bound()
    test_concurrent_category_case_variants_converge()
    test_deleted_profile_memory_is_recreated_without_duplicate_projection()
    print("auto-mutation profile rollout smoke passed")


if __name__ == "__main__":
    main()
