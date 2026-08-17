from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import ingest as ingest_module
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
)


TOOL_NAME = "ingest_obsidian_inbox"


class StaticPlanner:
    def __init__(self, args: dict[str, Any]):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise the aggregate-wired Inbox ingestion mutation.",
            [PlannedAction(TOOL_NAME, dict(self.args), "ingest rollout smoke")],
            needs_model=False,
        )


class MultiActionPlanner:
    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise a multi-action Inbox mutation batch.",
            [
                PlannedAction(TOOL_NAME, {"limit": 1}, "first ingest mutation"),
                PlannedAction(TOOL_NAME, {"limit": 2}, "second ingest mutation"),
            ],
            needs_model=False,
        )


def _runtime(root: Path, args: dict[str, Any] | None = None) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(args or {})
    return runtime


def _write_inbox(runtime: JarvisRuntime, content: str) -> Path:
    path = runtime.vault.root_path / "Inbox.md"
    path.write_text(content, encoding="utf-8")
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


def _ingested_memory_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(
        runtime,
        """
        SELECT memory.*, source.source_key
        FROM memories AS memory
        JOIN ingested_sources AS source ON source.memory_id = memory.id
        WHERE source.source_type = 'obsidian-inbox'
        ORDER BY memory.id
        """,
    )


def _projection_snapshot(runtime: JarvisRuntime) -> dict[str, bytes]:
    root = runtime.vault.root_path
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "Inbox.md"
    }


def _install_counted_handler(runtime: JarvisRuntime) -> list[int]:
    tool = runtime.registry.get(TOOL_NAME)
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools[TOOL_NAME] = replace(tool, handler=counted)
    return calls


def _assert_failure(result: Any, failure_kind: str, label: str) -> ToolResult:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} did not return one tool result: {result.tool_results}")
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != failure_kind:
        raise SystemExit(f"{label} returned the wrong failure: {item}")
    return item


def _assert_no_ingest_effects(
    runtime: JarvisRuntime,
    *,
    projection_before: dict[str, bytes],
    label: str,
) -> None:
    if _ingested_memory_rows(runtime):
        raise SystemExit(f"{label} created Inbox memories or source projections")
    if _projection_snapshot(runtime) != projection_before:
        raise SystemExit(f"{label} published an Obsidian memory projection")


def _assert_contract(runtime: JarvisRuntime) -> None:
    tool = runtime.registry.get(TOOL_NAME)
    contract = tool.auto_mutation_contract
    if (
        contract is None
        or contract.version != AUTO_MUTATION_CONTRACT_VERSION
        or contract.effects
        != frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT})
        or contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
        or contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
        or contract.operation_key_builder is None
        or contract.execution_args_builder is not ingest_module.inbox_ingest_auto_mutation_execution_args
        or contract.definite_no_effect_failure_reasons
        != frozenset({"inbox_source_changed", "inbox_source_too_large"})
    ):
        raise SystemExit(f"Inbox ingest auto-mutation contract drifted: {contract}")


def _assert_receipt_audits(runtime: JarvisRuntime, expected: int) -> None:
    receipts = _receipt_rows(runtime)
    successful_runs = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
    runs_by_id = {row["id"]: row for row in successful_runs}
    if len(receipts) != expected or len(successful_runs) != expected:
        raise SystemExit(
            f"expected {expected} Inbox ingest receipt/audit pairs: "
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
            or run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
        ):
            raise SystemExit(f"Inbox ingest receipt lost ordinary audit linkage: {receipt} / {run}")


def _assert_private_metadata_absent(
    runtime: JarvisRuntime,
    result: Any,
    *,
    private_values: list[str],
) -> None:
    surfaces = {
        "receipts": _receipt_rows(runtime),
        "tool_runs": _tool_run_rows(runtime),
        "runtime_metadata": result.metadata,
    }
    rendered = {
        name: json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        for name, value in surfaces.items()
    }
    for private in private_values:
        if not private:
            continue
        exposed = [name for name, text in rendered.items() if private in text]
        if exposed:
            raise SystemExit(
                f"Inbox ingest exposed private source/request binding in {exposed}: {private!r}"
            )
    for receipt in surfaces["receipts"]:
        for key in ("request_digest", "action_digest", "operation_digest"):
            value = receipt.get(key)
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise SystemExit(f"Inbox ingest receipt lost a bounded digest: {receipt}")


def test_exact_once_replay_and_changed_input_collision() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-rollout-replay-") as temp:
        runtime = _runtime(Path(temp), {"limit": 50})
        _assert_contract(runtime)
        body = "PRIVATE exact-once Inbox memory 7f42 must be ingested once."
        raw_source = f"# PRIVATE exact-once heading 61bc\n\n- {body}\n"
        _write_inbox(runtime, raw_source)
        operation = ingest_module.make_inbox_ingest_auto_mutation_operation_key(runtime.vault)(
            {"limit": 50}
        )
        binding = str(operation["snapshot_binding"])
        calls = _install_counted_handler(runtime)
        owner = "PRIVATE-INGEST-EXACT-ONCE-OWNER"

        first = runtime.handle("ingest exact Inbox snapshot", request_token=owner)
        if (
            not first.tool_results[0].ok
            or calls[0] != 1
            or [row["body"] for row in _ingested_memory_rows(runtime)] != [body]
        ):
            raise SystemExit(f"first Inbox ingestion did not publish exactly once: {first}")
        first_projection = _projection_snapshot(runtime)
        _assert_receipt_audits(runtime, 1)
        _assert_private_metadata_absent(
            runtime,
            first,
            private_values=[raw_source, body, owner, binding, str(Path(temp))],
        )

        replay = runtime.handle("replay exact Inbox snapshot", request_token=owner)
        _assert_failure(replay, "auto_mutation_completed_replay", "same-token Inbox replay")
        if (
            calls[0] != 1
            or [row["body"] for row in _ingested_memory_rows(runtime)] != [body]
            or _projection_snapshot(runtime) != first_projection
        ):
            raise SystemExit("same-token Inbox replay reached effects")

        runtime.planner = StaticPlanner({"limit": 1})
        collision = runtime.handle("reuse token with changed limit", request_token=owner)
        _assert_failure(
            collision,
            "auto_mutation_request_collision",
            "same-token changed Inbox request",
        )
        if calls[0] != 1 or len(_receipt_rows(runtime)) != 1:
            raise SystemExit("same-token changed input reached the handler or created a receipt")


def test_stale_snapshot_abandons_then_fresh_source_succeeds() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-rollout-stale-") as temp:
        runtime = _runtime(Path(temp))
        old_body = "PRIVATE stale Inbox snapshot memory 19ab."
        new_body = "PRIVATE fresh Inbox snapshot memory 28cd."
        inbox = _write_inbox(runtime, f"# Inbox\n\n- {old_body}\n")
        projection_before = _projection_snapshot(runtime)
        calls = _install_counted_handler(runtime)
        original_prepare = runtime.store.prepare_auto_mutation_receipts

        def prepare_then_change(*args: Any, **kwargs: Any):
            prepared = original_prepare(*args, **kwargs)
            inbox.write_text(f"# Inbox\n\n- {new_body}\n", encoding="utf-8")
            return prepared

        runtime.store.prepare_auto_mutation_receipts = prepare_then_change  # type: ignore[method-assign]
        stale_token = "PRIVATE-INGEST-STALE-SNAPSHOT-OWNER"
        try:
            stale = runtime.handle("ingest stale Inbox snapshot", request_token=stale_token)
        finally:
            runtime.store.prepare_auto_mutation_receipts = original_prepare  # type: ignore[method-assign]
        item = stale.tool_results[0]
        if (
            item.ok
            or item.metadata.get("reason") != "inbox_source_changed"
            or item.metadata.get("auto_mutation_effects_started") is not False
            or item.metadata.get("auto_mutation_definite_no_effect") is not True
            or item.metadata.get("auto_mutation_outcome_uncertain") is not False
            or calls[0] != 1
            or _receipt_rows(runtime)
        ):
            raise SystemExit(f"stale Inbox snapshot did not abandon its receipt: {item}")
        _assert_no_ingest_effects(
            runtime,
            projection_before=projection_before,
            label="stale Inbox snapshot",
        )

        fresh_token = "PRIVATE-INGEST-FRESH-SNAPSHOT-OWNER"
        fresh = runtime.handle("ingest changed Inbox with fresh token", request_token=fresh_token)
        if (
            not fresh.tool_results[0].ok
            or calls[0] != 2
            or [row["body"] for row in _ingested_memory_rows(runtime)] != [new_body]
        ):
            raise SystemExit(f"fresh token did not ingest the changed Inbox source: {fresh}")
        _assert_receipt_audits(runtime, 1)


def test_oversized_source_has_no_effects_and_no_receipt() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-rollout-oversize-") as temp:
        runtime = _runtime(Path(temp))
        private_marker = "PRIVATE-INGEST-OVERSIZE-5ee8"
        content = (
            "# Inbox\n\n- "
            + private_marker
            + "x" * (ingest_module.MAX_INBOX_SOURCE_CHARS + 1)
            + "\n"
        )
        _write_inbox(runtime, content)
        projection_before = _projection_snapshot(runtime)
        result = runtime.handle(
            "refuse oversized Inbox ingestion",
            request_token="PRIVATE-INGEST-OVERSIZE-OWNER",
        )
        item = result.tool_results[0]
        if (
            item.ok
            or item.metadata.get("reason") != "inbox_source_too_large"
            or item.metadata.get("auto_mutation_effects_started") is not False
            or item.metadata.get("auto_mutation_definite_no_effect") is not True
            or item.metadata.get("auto_mutation_outcome_uncertain") is not False
            or _receipt_rows(runtime)
        ):
            raise SystemExit(f"oversized Inbox source retained effects or custody: {item}")
        _assert_no_ingest_effects(
            runtime,
            projection_before=projection_before,
            label="oversized Inbox source",
        )


def test_source_change_after_effects_is_uncertain() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-rollout-source-race-") as temp:
        runtime = _runtime(Path(temp))
        original_body = "PRIVATE source-race Inbox content 1337."
        changed_body = "PRIVATE replacement Inbox content 7331."
        inbox = _write_inbox(runtime, f"# Inbox\n\n- {original_body}\n")
        original_add = runtime.store.add_memory_if_source_new_with_projection

        def add_then_change(*args: Any, **kwargs: Any):
            outcome = original_add(*args, **kwargs)
            inbox.write_text(f"# Inbox\n\n- {changed_body}\n", encoding="utf-8")
            return outcome

        runtime.store.add_memory_if_source_new_with_projection = add_then_change  # type: ignore[method-assign]
        token = "PRIVATE-INGEST-SOURCE-RACE-OWNER"
        try:
            raced = runtime.handle("ingest Inbox while source changes", request_token=token)
        finally:
            runtime.store.add_memory_if_source_new_with_projection = original_add  # type: ignore[method-assign]
        _assert_failure(raced, "auto_mutation_handler_failed", "post-effect Inbox source change")
        receipts = _receipt_rows(runtime)
        if (
            [row["body"] for row in _ingested_memory_rows(runtime)] != [original_body]
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
            or receipts[0]["resolution"] != "manual_review"
        ):
            raise SystemExit(f"post-effect Inbox source change lost uncertainty custody: {receipts}")

        replay = runtime.handle("replay source-changed Inbox", request_token=token)
        _assert_failure(replay, "auto_mutation_request_collision", "source-changed Inbox replay")
        if len(_ingested_memory_rows(runtime)) != 1:
            raise SystemExit("source-changed Inbox replay duplicated a guarded effect")


def test_projection_write_then_raise_stays_uncertain() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-rollout-projection-race-") as temp:
        runtime = _runtime(Path(temp))
        body = "PRIVATE projection write-then-raise Inbox content 647a."
        inbox = _write_inbox(runtime, f"# Inbox\n\n- {body}\n")
        original_reconcile = ingest_module.reconcile_memory_projection

        def write_then_change_and_raise(*args: Any, **kwargs: Any):
            original_reconcile(*args, **kwargs)
            inbox.write_text("# Inbox\n\n- PRIVATE replacement after projection 58cd.\n", encoding="utf-8")
            raise RuntimeError("fixture projection write then raise")

        ingest_module.reconcile_memory_projection = write_then_change_and_raise
        try:
            result = runtime.handle(
                "ingest projection write then raise",
                request_token="PRIVATE-INGEST-PROJECTION-OWNER",
            )
        finally:
            ingest_module.reconcile_memory_projection = original_reconcile
        _assert_failure(
            result,
            "auto_mutation_handler_failed",
            "projection write-then-raise source change",
        )
        receipts = _receipt_rows(runtime)
        if (
            len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
            or not _projection_snapshot(runtime)
        ):
            raise SystemExit(f"projection write-then-raise lost receipt custody: {receipts}")


def test_private_execution_binding_cannot_echo_or_split_a_batch() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-rollout-binding-") as temp:
        runtime = _runtime(Path(temp))
        private_body = "PRIVATE binding echo Inbox content e32b."
        _write_inbox(runtime, f"# Inbox\n\n- {private_body}\n")
        binding = str(
            ingest_module.make_inbox_ingest_auto_mutation_operation_key(runtime.vault)({})[
                "snapshot_binding"
            ]
        )
        tool = runtime.registry.get(TOOL_NAME)

        def echo(args: dict[str, Any]) -> ToolResult:
            return ToolResult(TOOL_NAME, True, f"echo: {args}", {"echo": args})

        runtime.registry._tools[TOOL_NAME] = replace(tool, handler=echo)
        token = "PRIVATE-INGEST-BINDING-ECHO-OWNER"
        echoed = runtime.handle("ingest binding echo fixture", request_token=token)
        _assert_failure(echoed, "internal_mutation_binding_leak", "binding echo refusal")
        receipts = _receipt_rows(runtime)
        if len(receipts) != 1 or receipts[0]["state"] != "uncertain":
            raise SystemExit(f"binding echo was not conservatively fenced: {receipts}")
        _assert_private_metadata_absent(
            runtime,
            echoed,
            private_values=[binding, token, private_body, str(runtime.vault.root_path)],
        )

    with TemporaryDirectory(prefix="jarvis-ingest-rollout-batch-") as temp:
        runtime = _runtime(Path(temp))
        _write_inbox(runtime, "# Inbox\n\n- PRIVATE batch binding source 2a4e.\n")
        calls = _install_counted_handler(runtime)
        tool = runtime.registry.get(TOOL_NAME)
        contract = tool.auto_mutation_contract
        if contract is None or contract.execution_args_builder is None:
            raise SystemExit("Inbox contract lost its execution builder")
        original_builder = contract.execution_args_builder

        def fail_second(args: dict[str, Any], operation_args: dict[str, Any]) -> dict[str, Any]:
            if args.get("limit") == 2:
                raise ValueError("fixture binding failure")
            return original_builder(args, operation_args)

        runtime.registry._tools[TOOL_NAME] = replace(
            tool,
            auto_mutation_contract=replace(contract, execution_args_builder=fail_second),
        )
        runtime.planner = MultiActionPlanner()
        batch = runtime.handle("reject mixed binding batch", request_token="PRIVATE-INGEST-BATCH-OWNER")
        if (
            len(batch.tool_results) != 2
            or any(
                item.metadata.get("failure_kind") != "auto_mutation_execution_binding_failed"
                or item.metadata.get("handler_invoked") is not False
                for item in batch.tool_results
            )
            or calls[0] != 0
            or _receipt_rows(runtime)
        ):
            raise SystemExit(f"binding failure split or claimed the Inbox batch: {batch.tool_results}")

    with TemporaryDirectory(prefix="jarvis-ingest-rollout-policy-") as temp:
        runtime = _runtime(Path(temp))
        _write_inbox(runtime, "# Inbox\n\n- PRIVATE policy refusal source 71a6.\n")
        calls = _install_counted_handler(runtime)
        runtime.executor.policy = PermissionPolicy(RiskLevel.READ_ONLY)
        blocked = runtime.handle("policy-blocked Inbox ingest", request_token="PRIVATE-INGEST-POLICY-OWNER")
        _assert_failure(blocked, "auto_mutation_policy_blocked", "policy preflight")
        if calls[0] != 0 or _receipt_rows(runtime):
            raise SystemExit("policy-blocked Inbox ingest claimed a receipt or invoked its handler")


def test_post_effect_failure_is_uncertain_and_metadata_is_private() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-rollout-uncertain-") as temp:
        runtime = _runtime(Path(temp))
        body = "PRIVATE uncertain Inbox content 83de must stay out of receipt metadata."
        raw_source = f"# PRIVATE Inbox heading b70c\n\n- {body}\n"
        _write_inbox(runtime, raw_source)
        operation = ingest_module.make_inbox_ingest_auto_mutation_operation_key(runtime.vault)({})
        binding = str(operation["snapshot_binding"])
        if re.fullmatch(r"[0-9a-f]{64}", binding) is None:
            raise SystemExit(f"Inbox privacy fixture did not produce an opaque binding: {binding!r}")
        token = "PRIVATE-INGEST-POST-EFFECT-OWNER"
        original_complete = runtime.store.complete_auto_mutation_receipt

        def fail_completion(**_kwargs: Any) -> int:
            raise RuntimeError("injected Inbox ingest completion failure")

        runtime.store.complete_auto_mutation_receipt = fail_completion  # type: ignore[method-assign]
        try:
            failed = runtime.handle("ingest before audit failure", request_token=token)
        finally:
            runtime.store.complete_auto_mutation_receipt = original_complete  # type: ignore[method-assign]
        _assert_failure(
            failed,
            "auto_mutation_completion_failed",
            "post-effect Inbox completion failure",
        )
        receipts = _receipt_rows(runtime)
        projection = _projection_snapshot(runtime)
        if (
            [row["body"] for row in _ingested_memory_rows(runtime)] != [body]
            or not projection
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
            or receipts[0]["resolution"] != "manual_review"
            or receipts[0]["tool_run_id"] is not None
            or any(row["ok"] == 1 for row in _tool_run_rows(runtime))
        ):
            raise SystemExit(
                "post-effect Inbox failure lost effects or uncertain receipt custody: "
                f"{receipts} / {_tool_run_rows(runtime)}"
            )

        replay = runtime.handle("replay uncertain Inbox ingestion", request_token=token)
        _assert_failure(replay, "auto_mutation_outcome_uncertain", "uncertain Inbox replay")
        if len(_ingested_memory_rows(runtime)) != 1 or _projection_snapshot(runtime) != projection:
            raise SystemExit("uncertain Inbox replay duplicated effects")

        _assert_private_metadata_absent(
            runtime,
            failed,
            private_values=[
                raw_source,
                body,
                token,
                binding,
                str(Path(temp)),
                str(runtime.vault.root_path),
                str(runtime.store.db_path.parent),
            ],
        )


def main() -> None:
    test_exact_once_replay_and_changed_input_collision()
    test_stale_snapshot_abandons_then_fresh_source_succeeds()
    test_oversized_source_has_no_effects_and_no_receipt()
    test_source_change_after_effects_is_uncertain()
    test_projection_write_then_raise_stays_uncertain()
    test_private_execution_binding_cannot_echo_or_split_a_batch()
    test_post_effect_failure_is_uncertain_and_metadata_is_private()
    print("Auto-mutation Inbox ingest rollout smoke passed")


if __name__ == "__main__":
    main()
