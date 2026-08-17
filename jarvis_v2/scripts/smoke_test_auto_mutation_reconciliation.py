from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from typing import Any

import jarvis_v2.agent.model_planner as model_planner_module
import jarvis_v2.tools.auto_mutation_reconciliation as reconciliation_module
from jarvis_v2.agent.model_planner import ModelBackedPlanner
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime


class StaticPlanner:
    def __init__(self, action: PlannedAction):
        self.action = action

    def plan(self, _user_input: str) -> Plan:
        return Plan("Reconciliation replay fixture.", [self.action], needs_model=False)


class ModelOnlyPlanner:
    def plan(self, user_input: str) -> Plan:
        return Plan(user_input, [], needs_model=True)


def _make_uncertain(runtime: Any, request_token: str, action: PlannedAction) -> int:
    contract = runtime.registry.get(action.tool_name).auto_mutation_contract
    operation_args = (
        contract.operation_key_builder(dict(action.args))
        if contract is not None and contract.operation_key_builder is not None
        else dict(action.args)
    )
    prepared = runtime.store.prepare_auto_mutation_receipts(
        request_token,
        [(action.tool_name, action.args, operation_args)],
    )
    if prepared.status != "PREPARED" or prepared.receipts[0].receipt_id is None:
        raise SystemExit("uncertain receipt setup did not prepare")
    receipt_id = int(prepared.receipts[0].receipt_id)
    claim = runtime.store.claim_auto_mutation_receipt(receipt_id, request_token)
    if claim.status != "CLAIMED" or not claim.run_token:
        raise SystemExit("uncertain receipt setup did not claim")
    if not runtime.store.mark_auto_mutation_uncertain(receipt_id, claim.run_token):
        raise SystemExit("uncertain receipt setup did not become uncertain")
    return receipt_id


def _reconciliation_rows(runtime: Any) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM auto_mutation_reconciliations ORDER BY receipt_id")]


def _tool_runs(runtime: Any, tool_name: str) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM tool_runs WHERE tool_name = ? ORDER BY id",
                (tool_name,),
            )
        ]


def _resolve_store(runtime: Any, receipt_id: int, digest: str, disposition: str, session: str):
    return runtime.store.resolve_auto_mutation_receipt(
        receipt_id,
        disposition,
        expected_uncertainty_digest=digest,
        session_id=session,
        reviewed_at="2026-07-11T00:00:00Z",
        output=f"resolved receipt #{receipt_id} without rerun",
        metadata={
            "receipt_id": receipt_id,
            "reconciliation_disposition": disposition,
            "mutation_rerun": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
        },
    )


def _assert_no_authority(result: Any, label: str) -> None:
    item = result.tool_results[0]
    for key in (
        "authorizes_retry",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if item.metadata.get(key) is not False:
            raise SystemExit(f"{label} granted authority through {key}")
    if result.metadata["runtime_trace"].get("approval_queue_delta") != 0:
        raise SystemExit(f"{label} changed the approval queue")


def _assert_recovery(
    item: ToolResult,
    *,
    action: str,
    commands: list[str],
    label: str,
) -> None:
    if item.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": commands,
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {item.metadata}")
    if (
        item.metadata.get("next_command") != commands[0]
        or item.metadata.get("recovery_commands") != commands
        or item.metadata.get("retry_safe") is not False
        or item.metadata.get("automatic_retry_allowed") is not False
        or item.metadata.get("retry_requires_reconciliation_review") is not True
        or item.metadata.get("mutation_rerun") is not False
        or item.metadata.get("authorizes_retry") is not False
    ):
        raise SystemExit(f"{label} recovery/no-rerun aliases drifted: {item.metadata}")


def test_operator_review_resolution_and_replay_contract() -> None:
    marker = "PRIVATE_RECONCILIATION_TARGET_CONTENT"
    action = PlannedAction(
        "remember",
        {"category": "facts", "title": "Reconciliation fixture", "body": marker},
        "private planner reason",
    )
    with TemporaryDirectory(prefix="jarvis-reconciliation-lifecycle-") as temp:
        runtime = make_temp_runtime(Path(temp))
        receipt_id = _make_uncertain(runtime, "original-request", action)
        original_tool = runtime.registry.get("remember")
        calls = 0

        def counted_handler(args: dict[str, Any]) -> ToolResult:
            nonlocal calls
            calls += 1
            return original_tool.handler(args)

        runtime.registry._tools["remember"] = replace(original_tool, handler=counted_handler)
        runtime.planner = StaticPlanner(action)

        queue = runtime.handle("uncertain mutations")
        if not queue.verified or queue.plan.actions[0].tool_name != "auto_mutation_reconciliation_queue":
            raise SystemExit("uncertain mutation queue did not use command-first routing")
        if f"receipt #{receipt_id}" not in queue.response or marker in queue.response:
            raise SystemExit("uncertain mutation queue missed the receipt or exposed content")
        _assert_no_authority(queue, "reconciliation queue")

        unreviewed = runtime.handle(f"resolve uncertain mutation {receipt_id} as applied")
        if unreviewed.tool_results[0].metadata.get("failure_kind") != "reconciliation_review_required":
            raise SystemExit("unreviewed resolution did not fail closed")
        _assert_recovery(
            unreviewed.tool_results[0],
            action=(
                "Run `uncertain mutation <id>` immediately before resolving. Do not reuse an "
                "expired review or rerun the mutation."
            ),
            commands=["uncertain mutation <id>"],
            label="unreviewed reconciliation",
        )
        if _reconciliation_rows(runtime) or calls:
            raise SystemExit("unreviewed resolution mutated state or reran target")

        inspected = runtime.handle(f"uncertain mutation {receipt_id}")
        if not inspected.verified or inspected.plan.actions[0].tool_name != "auto_mutation_reconciliation_status":
            raise SystemExit("uncertain mutation inspection did not use command-first routing")
        if inspected.tool_results[0].metadata.get("review_receipt_created") is not True:
            raise SystemExit("inspection did not create a private review receipt")
        _assert_no_authority(inspected, "reconciliation inspection")

        resolved = runtime.handle(f"resolve uncertain mutation {receipt_id} as applied")
        item = resolved.tool_results[0]
        if not resolved.verified or item.metadata.get("reconciliation_status") != "resolved":
            raise SystemExit("reviewed uncertain mutation did not resolve")
        if calls or item.metadata.get("mutation_rerun") is not False:
            raise SystemExit("resolution reran the target mutation")
        rows = _reconciliation_rows(runtime)
        if len(rows) != 1 or rows[0]["disposition"] != "confirmed_applied":
            raise SystemExit("resolution ledger did not preserve the operator disposition")
        runs = _tool_runs(runtime, "resolve_auto_mutation_receipt")
        if len(runs) != 2:
            raise SystemExit("resolution audit should contain one failed review gate and one atomic success")
        successful = [row for row in runs if row["ok"] == 1]
        if len(successful) != 1 or successful[0]["approved"] != 0:
            raise SystemExit("atomic resolution audit was missing or approval-linked")
        if successful[0]["approval_id"] is not None or successful[0]["approval_action_digest"] is not None:
            raise SystemExit("resolution audit entered approval proof")
        _assert_no_authority(resolved, "reconciliation resolution")

        same_request = runtime.handle("original replay", request_token="original-request")
        if calls or same_request.tool_results[0].metadata.get("failure_kind") != "auto_mutation_reconciled_applied":
            raise SystemExit("original request replay ran or lost reconciled state")
        fresh = runtime.handle("intentional fresh request", request_token="fresh-request")
        if calls != 1 or not fresh.tool_results[0].ok:
            raise SystemExit("reconciliation did not release a genuinely new request")

        exposed = json.dumps(
            {
                "queue": queue.tool_results[0].metadata,
                "inspect": inspected.tool_results[0].metadata,
                "resolve": resolved.tool_results[0].metadata,
                "queue_output": queue.response,
                "inspect_output": inspected.response,
                "resolve_output": resolved.response,
            },
            ensure_ascii=False,
            default=str,
        )
        receipt_row = runtime.store.get_auto_mutation_reconciliation_receipt(receipt_id)
        private_values = [
            marker,
            action.reason,
            str(receipt_row["uncertainty_digest"]),
        ]
        with runtime.store.connect() as conn:
            raw_receipt = conn.execute(
                "SELECT request_digest, action_digest, operation_digest, run_token "
                "FROM auto_mutation_receipts WHERE id = ?",
                (receipt_id,),
            ).fetchone()
        private_values.extend(
            str(raw_receipt[key])
            for key in ("request_digest", "action_digest", "operation_digest", "run_token")
        )
        if any(value and value in exposed for value in private_values):
            raise SystemExit("operator reconciliation surface exposed private receipt evidence")


def test_not_applied_resolution_and_exact_review_binding() -> None:
    action = PlannedAction("remember", {"body": "not-applied fixture"})
    with TemporaryDirectory(prefix="jarvis-reconciliation-not-applied-") as temp:
        runtime = make_temp_runtime(Path(temp))
        receipt_id = _make_uncertain(runtime, "not-applied-original", action)
        runtime.planner = StaticPlanner(action)
        runtime.handle(f"uncertain mutation {receipt_id}")

        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET uncertainty_digest = ? WHERE id = ?",
                ("f" * 64, receipt_id),
            )
        stale_review = runtime.handle(f"resolve uncertain mutation {receipt_id} as not applied")
        if stale_review.tool_results[0].metadata.get("failure_kind") != "reconciliation_review_mismatch":
            raise SystemExit("changed uncertainty proof did not invalidate review")
        _assert_recovery(
            stale_review.tool_results[0],
            action=(
                "Run `uncertain mutation <id>` to create a fresh bound review, then decide again. "
                "Do not reuse the stale resolution or rerun the mutation."
            ),
            commands=["uncertain mutation <id>"],
            label="stale reconciliation review",
        )
        if _reconciliation_rows(runtime):
            raise SystemExit("stale review created a reconciliation row")

        runtime.handle(f"uncertain mutation {receipt_id}")
        resolved = runtime.handle(f"resolve uncertain mutation {receipt_id} as not-applied")
        if not resolved.verified or _reconciliation_rows(runtime)[0]["disposition"] != "confirmed_not_applied":
            raise SystemExit("not-applied resolution failed")
        old = runtime.handle("not-applied old replay", request_token="not-applied-original")
        if old.tool_results[0].metadata.get("failure_kind") != "auto_mutation_reconciled_not_applied":
            raise SystemExit("not-applied original request did not remain non-replayable")


def test_concurrent_resolution_is_single_winner_and_conflict_safe() -> None:
    action = PlannedAction("remember", {"body": "race fixture"})
    with TemporaryDirectory(prefix="jarvis-reconciliation-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        receipt_id = _make_uncertain(runtime, "race-request", action)
        row = runtime.store.get_auto_mutation_reconciliation_receipt(receipt_id)
        digest = str(row["uncertainty_digest"])
        with ThreadPoolExecutor(max_workers=12) as pool:
            statuses = list(
                pool.map(
                    lambda index: _resolve_store(
                        runtime,
                        receipt_id,
                        digest,
                        "confirmed_applied" if index % 2 == 0 else "confirmed_not_applied",
                        f"race-{index}",
                    ).status,
                    range(12),
                )
            )
        if statuses.count("RESOLVED") != 1:
            raise SystemExit("concurrent reconciliation did not have exactly one winner")
        if any(status not in {"RESOLVED", "ALREADY_RESOLVED_SAME", "RESOLUTION_CONFLICT"} for status in statuses):
            raise SystemExit("concurrent reconciliation returned an invalid state")
        rows = _reconciliation_rows(runtime)
        if len(rows) != 1:
            raise SystemExit("concurrent reconciliation created duplicate decisions")
        runs = _tool_runs(runtime, "resolve_auto_mutation_receipt")
        if len(runs) != 1:
            raise SystemExit("concurrent reconciliation created duplicate atomic audits")


def test_malformed_proof_and_public_conflict_have_bounded_recovery() -> None:
    action = PlannedAction("remember", {"body": "PRIVATE reconciliation fixture"})
    with TemporaryDirectory(prefix="jarvis-reconciliation-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        malformed_id = _make_uncertain(runtime, "malformed-proof", action)
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET uncertainty_digest = ? WHERE id = ?",
                ("z" * 64, malformed_id),
            )
        malformed = runtime.handle(f"uncertain mutation {malformed_id}")
        malformed_item = malformed.tool_results[0]
        if malformed_item.metadata.get("failure_kind") != "malformed_uncertainty_proof":
            raise SystemExit(f"malformed proof failure kind drifted: {malformed_item}")
        malformed_action = (
            "This uncertain receipt has malformed proof. Run `auto mutation reconciliation "
            "queue`, then `recent tool runs` and `setup check`; repair storage before "
            "resolving. Do not resolve or rerun the mutation."
        )
        _assert_recovery(
            malformed_item,
            action=malformed_action,
            commands=[
                "auto mutation reconciliation queue",
                "recent tool runs",
                "setup check",
            ],
            label="malformed reconciliation proof",
        )
        if (
            _reconciliation_rows(runtime)
            or "PRIVATE reconciliation fixture" in malformed.response
            or ("z" * 64) in json.dumps(malformed_item.metadata, default=str)
        ):
            raise SystemExit("malformed reconciliation proof leaked evidence or changed state")

        conflict_action = PlannedAction(
            "remember",
            {"body": "separate conflict fixture"},
        )
        conflict_id = _make_uncertain(
            runtime,
            "conflicting-resolution",
            conflict_action,
        )
        inspected = runtime.handle(f"uncertain mutation {conflict_id}")
        if inspected.tool_results[0].metadata.get("review_receipt_created") is not True:
            raise SystemExit("conflict fixture did not create a bound review")
        row = runtime.store.get_auto_mutation_reconciliation_receipt(conflict_id)
        digest = str(row["uncertainty_digest"])
        stored = _resolve_store(
            runtime,
            conflict_id,
            digest,
            "confirmed_applied",
            "external-conflict-fixture",
        )
        if stored.status != "RESOLVED":
            raise SystemExit(f"conflict fixture did not record the first decision: {stored}")
        conflict = runtime.handle(
            f"resolve uncertain mutation {conflict_id} as not applied"
        )
        conflict_item = conflict.tool_results[0]
        if conflict_item.metadata.get("failure_kind") != "reconciliation_conflict":
            raise SystemExit(f"public reconciliation conflict drifted: {conflict_item}")
        _assert_recovery(
            conflict_item,
            action=(
                "Run `uncertain mutation <id>` and investigate the recorded disposition. Do not "
                "submit another resolution or rerun the mutation."
            ),
            commands=["uncertain mutation <id>"],
            label="reconciliation conflict",
        )
        rows = _reconciliation_rows(runtime)
        if (
            len(rows) != 1
            or rows[0]["receipt_id"] != conflict_id
            or rows[0]["disposition"] != "confirmed_applied"
        ):
            raise SystemExit(f"conflicting resolution changed recorded evidence: {rows}")


def test_resolution_audit_failure_rolls_back_and_wrong_states_refuse() -> None:
    action = PlannedAction("remember", {"body": "rollback fixture"})
    with TemporaryDirectory(prefix="jarvis-reconciliation-rollback-") as temp:
        runtime = make_temp_runtime(Path(temp))
        receipt_id = _make_uncertain(runtime, "rollback-request", action)
        row = runtime.store.get_auto_mutation_reconciliation_receipt(receipt_id)
        digest = str(row["uncertainty_digest"])
        with runtime.store.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER fail_reconciliation_insert
                BEFORE INSERT ON auto_mutation_reconciliations
                BEGIN
                    SELECT RAISE(ABORT, 'fixture failure');
                END
                """
            )
        before_runs = len(_tool_runs(runtime, "resolve_auto_mutation_receipt"))
        try:
            _resolve_store(runtime, receipt_id, digest, "confirmed_applied", "rollback")
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("injected reconciliation audit failure did not raise")
        if _reconciliation_rows(runtime):
            raise SystemExit("failed reconciliation insert left a decision")
        if len(_tool_runs(runtime, "resolve_auto_mutation_receipt")) != before_runs:
            raise SystemExit("failed reconciliation insert left an orphan audit")

        completed_action = PlannedAction("remember", {"body": "completed state fixture"})
        completed = runtime.store.prepare_auto_mutation_receipts(
            "completed-request",
            [(completed_action.tool_name, completed_action.args)],
        )
        completed_id = int(completed.receipts[0].receipt_id)
        claim = runtime.store.claim_auto_mutation_receipt(completed_id, "completed-request")
        runtime.store.complete_auto_mutation_receipt(
            receipt_id=completed_id,
            run_token=claim.run_token,
            session_id="fixture",
            risk="LOCAL_SAFE",
            ok=True,
            output="done",
            metadata={},
        )
        for invalid_id in (True, 0, -1):
            try:
                runtime.store.resolve_auto_mutation_receipt(  # type: ignore[arg-type]
                    invalid_id,
                    "confirmed_applied",
                    expected_uncertainty_digest=digest,
                    session_id="fixture",
                    reviewed_at="2026-07-11T00:00:00Z",
                    output="no",
                    metadata={},
                )
            except ValueError:
                pass
            else:
                raise SystemExit("invalid reconciliation id was accepted")
        wrong_state = _resolve_store(runtime, completed_id, digest, "confirmed_applied", "wrong-state")
        if wrong_state.status != "NOT_UNCERTAIN":
            raise SystemExit("completed receipt accepted reconciliation")


def test_resolution_is_model_denied_and_tools_are_typed() -> None:
    with TemporaryDirectory(prefix="jarvis-reconciliation-model-") as temp:
        runtime = make_temp_runtime(Path(temp))
        resolver = runtime.registry.get("resolve_auto_mutation_receipt")
        if resolver.auto_mutation_contract is not None or resolver.argument_contract is None:
            raise SystemExit("resolver entered mutation replay or lacked typed arguments")
        for name in (
            "auto_mutation_reconciliation_queue",
            "auto_mutation_reconciliation_status",
            "resolve_auto_mutation_receipt",
        ):
            if runtime.registry.get(name).argument_contract is None:
                raise SystemExit("reconciliation tool lacked typed arguments")
        planner = ModelBackedPlanner("fixture", runtime.registry, base=ModelOnlyPlanner())
        if "resolve_auto_mutation_receipt" in planner._tool_descriptions():
            raise SystemExit("model planner advertised the mutating reconciliation tool")
        original_generate = model_planner_module.generate_model_text
        try:
            model_planner_module.generate_model_text = lambda **_kwargs: json.dumps(  # type: ignore[assignment]
                {
                    "mode": "tool",
                    "goal": "forged resolution",
                    "actions": [
                        {
                            "tool_name": "resolve_auto_mutation_receipt",
                            "args": {"receipt_id": 1, "disposition": "confirmed_applied"},
                        }
                    ],
                }
            )
            plan = planner.plan("resolve something")
        finally:
            model_planner_module.generate_model_text = original_generate  # type: ignore[assignment]
        if plan.actions or "resolve_auto_mutation_receipt" not in plan.metadata.get("model_planner_ignored_disallowed_tools", []):
            raise SystemExit("model planner did not deny reconciliation mutation")
        runtime.planner = StaticPlanner(
            PlannedAction(
                "resolve_auto_mutation_receipt",
                {"receipt_id": 1, "disposition": "confirmed_applied"},
            )
        )
        forged = runtime.handle("forged planner reconciliation")
        if forged.tool_results[0].metadata.get("failure_kind") != "reconciliation_explicit_route_required":
            raise SystemExit("non-explicit reconciliation plan reached the resolver")


def test_review_receipt_expiry_and_additive_migration() -> None:
    action = PlannedAction("remember", {"body": "expiry fixture"})
    with TemporaryDirectory(prefix="jarvis-reconciliation-expiry-") as temp:
        runtime = make_temp_runtime(Path(temp))
        receipt_id = _make_uncertain(runtime, "expiry-request", action)
        runtime.handle(f"uncertain mutation {receipt_id}")
        original_monotonic = reconciliation_module.time.monotonic
        try:
            now = original_monotonic()
            reconciliation_module.time.monotonic = lambda: now + 1000  # type: ignore[assignment]
            expired = runtime.handle(f"resolve uncertain mutation {receipt_id} as applied")
        finally:
            reconciliation_module.time.monotonic = original_monotonic  # type: ignore[assignment]
        if expired.tool_results[0].metadata.get("failure_kind") != "reconciliation_review_required":
            raise SystemExit("expired reconciliation review receipt did not fail closed")
        _assert_recovery(
            expired.tool_results[0],
            action=(
                "Run `uncertain mutation <id>` immediately before resolving. Do not reuse an "
                "expired review or rerun the mutation."
            ),
            commands=["uncertain mutation <id>"],
            label="expired reconciliation review",
        )
        if _reconciliation_rows(runtime):
            raise SystemExit("expired reconciliation review changed state")

    with TemporaryDirectory(prefix="jarvis-reconciliation-migration-") as temp:
        runtime = make_temp_runtime(Path(temp))
        receipt_id = _make_uncertain(runtime, "migration-request", action)
        with runtime.store.connect() as conn:
            conn.execute("DROP TABLE auto_mutation_reconciliations")
        runtime.store.init()
        runtime.store.init()
        row = runtime.store.get_auto_mutation_reconciliation_receipt(receipt_id)
        if row is None or str(row["state"]) != "uncertain" or row["reconciliation_disposition"] is not None:
            raise SystemExit("additive reconciliation migration lost existing uncertain evidence")


def main() -> None:
    test_operator_review_resolution_and_replay_contract()
    test_not_applied_resolution_and_exact_review_binding()
    test_concurrent_resolution_is_single_winner_and_conflict_safe()
    test_malformed_proof_and_public_conflict_have_bounded_recovery()
    test_resolution_audit_failure_rolls_back_and_wrong_states_refuse()
    test_resolution_is_model_denied_and_tools_are_typed()
    test_review_receipt_expiry_and_additive_migration()
    print("Auto mutation reconciliation smoke passed")


if __name__ == "__main__":
    main()
