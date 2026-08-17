from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier

from jarvis_v2.memory.store import (
    MemoryStore,
    TaskRecord,
    auto_mutation_action_digest,
    auto_mutation_request_digest,
)


def _store(temp: str, name: str = "receipts.sqlite") -> MemoryStore:
    store = MemoryStore(Path(temp) / name)
    store.init()
    return store


def test_canonical_digests_and_privacy_shape() -> None:
    request_token = "private-request-token-never-store"
    args = {
        "path": "/\x55sers/private/secret.txt",
        "contact": "private@example.invalid",
        "metadata": {"secret": "do-not-store"},
    }
    if auto_mutation_request_digest(request_token) == request_token:
        raise SystemExit("request token was not digested")
    first = auto_mutation_action_digest("mutate_local_state", args)
    second = auto_mutation_action_digest(
        "mutate_local_state",
        {"metadata": {"secret": "do-not-store"}, "contact": args["contact"], "path": args["path"]},
    )
    if first != second or len(first) != 64:
        raise SystemExit("auto mutation action digest is not canonical SHA-256")

    with TemporaryDirectory(prefix="jarvis-auto-mutation-privacy-") as temp:
        store = _store(temp)
        prepared = store.prepare_auto_mutation_receipts(
            request_token,
            [("mutate_local_state", args)],
        )
        if prepared.status != "PREPARED" or len(prepared.receipts) != 1:
            raise SystemExit(f"privacy preparation failed: {prepared}")
        allowed_columns = {
            "id",
            "request_digest",
            "action_digest",
            "operation_digest",
            "operation_scope",
            "uncertainty_digest",
            "tool_name",
            "action_index",
            "state",
            "result",
            "resolution",
            "run_token",
            "tool_run_id",
            "prepared_at",
            "running_at",
            "completed_at",
            "uncertain_at",
            "updated_at",
        }
        with store.connect() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(auto_mutation_receipts)")}
            row = conn.execute("SELECT * FROM auto_mutation_receipts").fetchone()
        if columns != allowed_columns:
            raise SystemExit(f"receipt schema contains a forbidden or missing column: {sorted(columns)}")
        serialized = repr(dict(row))
        for secret in (request_token, args["path"], args["contact"], "do-not-store"):
            if secret in serialized:
                raise SystemExit(f"receipt leaked raw request/action data: {secret!r}")
        if "run_token" in prepared.__dataclass_fields__ or request_token in repr(prepared):
            raise SystemExit("batch preparation exposed a private token")


def test_atomic_prepare_resume_collision_and_cross_request_block() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-mutation-prepare-") as temp:
        store = _store(temp)
        actions = [("task_add", {"body": "alpha"}), ("preference_set", {"key": "k", "value": "v"})]
        prepared = store.prepare_auto_mutation_receipts("request-a", actions)
        resumed = store.prepare_auto_mutation_receipts("request-a", actions)
        if prepared.status != "PREPARED" or resumed.status != "RESUMED":
            raise SystemExit(f"prepared resume did not coalesce: {prepared}, {resumed}")
        if [row.receipt_id for row in prepared.receipts] != [row.receipt_id for row in resumed.receipts]:
            raise SystemExit("prepared resume changed receipt identities")

        target_store = _store(temp, "logical-targets.sqlite")
        target_one = target_store.prepare_auto_mutation_receipts(
            "target-request-a",
            [("preference_set", {"key": "Voice Tone", "value": "warm"}, {"key": "voice tone"})],
        )
        target_blocked = target_store.prepare_auto_mutation_receipts(
            "target-request-b",
            [("preference_set", {"key": "voice tone", "value": "direct"}, {"key": "voice tone"})],
        )
        target_distinct = target_store.prepare_auto_mutation_receipts(
            "target-request-c",
            [("preference_set", {"key": "planning depth", "value": "short"}, {"key": "planning depth"})],
        )
        if target_one.status != "PREPARED" or target_blocked.status != "UNRESOLVED_ACTION_BLOCKED":
            raise SystemExit(
                f"logical operation target did not fence value/casing variants: {target_one}, {target_blocked}"
            )
        if target_distinct.status != "PREPARED":
            raise SystemExit(f"distinct logical operation target was over-blocked: {target_distinct}")

        collision = store.prepare_auto_mutation_receipts(
            "request-a",
            [("task_add", {"body": "different"}), actions[1]],
        )
        if collision.status != "REQUEST_TOKEN_COLLISION":
            raise SystemExit(f"same token with a different action did not collide: {collision}")
        blocked = store.prepare_auto_mutation_receipts(
            "request-b",
            [("unrelated_first", {"value": 1}), actions[0]],
        )
        if blocked.status != "UNRESOLVED_ACTION_BLOCKED":
            raise SystemExit(f"cross-request unresolved action was not blocked: {blocked}")
        with store.connect() as conn:
            count_b = conn.execute(
                "SELECT COUNT(*) FROM auto_mutation_receipts WHERE request_digest = ?",
                (auto_mutation_request_digest("request-b"),),
            ).fetchone()[0]
        if count_b != 0:
            raise SystemExit("blocked batch left a partially prepared receipt")


def test_intra_batch_duplicate_operations_fail_before_receipt_insertion() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-mutation-intra-batch-") as temp:
        store = _store(temp)
        private_value = "PRIVATE-DUPLICATE-CONTENT-DO-NOT-EXPOSE"
        exact_actions = [
            ("task_add", {"body": private_value}),
            ("task_add", {"body": private_value}),
        ]
        logical_actions = [
            ("preference_set", {"key": "Voice Tone", "value": private_value}, {"key": "voice tone"}),
            ("preference_set", {"key": "voice tone", "value": "direct"}, {"key": "voice tone"}),
        ]
        outcomes = (
            store.inspect_auto_mutation_receipts("duplicate-inspect", exact_actions),
            store.prepare_auto_mutation_receipts("duplicate-exact", exact_actions),
            store.prepare_auto_mutation_receipts("duplicate-logical", logical_actions),
        )
        if any(outcome.status != "DUPLICATE_OPERATION_DIGEST" for outcome in outcomes):
            raise SystemExit(f"intra-batch duplicate operation was not rejected: {outcomes}")
        if any(outcome.receipts for outcome in outcomes):
            raise SystemExit("duplicate operation rejection returned receipt identities")
        if private_value in repr(outcomes):
            raise SystemExit("duplicate operation status exposed private action content")
        with store.connect() as conn:
            receipt_count = conn.execute("SELECT COUNT(*) FROM auto_mutation_receipts").fetchone()[0]
        if receipt_count != 0:
            raise SystemExit("duplicate operation batch inserted a receipt")

        legacy_token = "legacy-duplicate-replay"
        legacy_request_digest = auto_mutation_request_digest(legacy_token)
        legacy_action_digest = auto_mutation_action_digest(
            "task_add", {"body": "legacy duplicate"}
        )
        with store.connect() as conn:
            for action_index in (0, 1):
                conn.execute(
                    """
                    INSERT INTO auto_mutation_receipts(
                        request_digest, action_digest, operation_digest, tool_name,
                        action_index, state, result, resolution, prepared_at, updated_at
                    ) VALUES (?, ?, ?, 'task_add', ?, 'prepared', 'pending', 'none', ?, ?)
                    """,
                    (
                        legacy_request_digest,
                        legacy_action_digest,
                        legacy_action_digest,
                        action_index,
                        "2026-07-11T00:00:00Z",
                        "2026-07-11T00:00:00Z",
                    ),
                )
        legacy_actions = [
            ("task_add", {"body": "legacy duplicate"}),
            ("task_add", {"body": "legacy duplicate"}),
        ]
        legacy_inspected = store.inspect_auto_mutation_receipts(
            legacy_token, legacy_actions
        )
        legacy_resumed = store.prepare_auto_mutation_receipts(
            legacy_token, legacy_actions
        )
        if (
            legacy_inspected.status != "RESUMED"
            or legacy_resumed.status != "RESUMED"
            or len(legacy_resumed.receipts) != 2
        ):
            raise SystemExit(
                "legacy duplicate receipts did not preserve exact same-request replay: "
                f"{legacy_inspected}, {legacy_resumed}"
            )
        if store.cleanup_prepared_auto_mutation_receipts(
            legacy_token, "9999-12-31T23:59:59Z"
        ) != 2:
            raise SystemExit("legacy duplicate replay fixture did not release its broad alias fence")

        valid = [("task_add", {"body": "first"}), ("task_add", {"body": "second"})]
        prepared = store.prepare_auto_mutation_receipts("valid-replay", valid)
        inspected = store.inspect_auto_mutation_receipts("valid-replay", valid)
        resumed = store.prepare_auto_mutation_receipts("valid-replay", valid)
        if (
            prepared.status != "PREPARED"
            or inspected.status != "RESUMED"
            or resumed.status != "RESUMED"
            or [receipt.receipt_id for receipt in prepared.receipts]
            != [receipt.receipt_id for receipt in resumed.receipts]
        ):
            raise SystemExit(f"same-request replay semantics changed: {prepared}, {inspected}, {resumed}")


def test_concurrent_claim_has_one_private_winner() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-mutation-claim-") as temp:
        store = _store(temp)
        receipt_id = store.prepare_auto_mutation_receipts(
            "claim-request",
            [("task_add", {"body": "once"})],
        ).receipts[0].receipt_id
        if receipt_id is None:
            raise SystemExit("claim smoke did not receive a receipt id")
        callers = 12
        barrier = Barrier(callers)

        def claim() -> object:
            barrier.wait(timeout=10)
            return store.claim_auto_mutation_receipt(receipt_id, "claim-request")

        with ThreadPoolExecutor(max_workers=callers) as pool:
            results = list(pool.map(lambda _index: claim(), range(callers)))
        winners = [result for result in results if result.status == "CLAIMED"]
        if len(winners) != 1 or not winners[0].run_token:
            raise SystemExit(f"concurrent claim did not have exactly one private winner: {results}")
        if any(result.run_token for result in results if result.status != "CLAIMED"):
            raise SystemExit("a losing claimant received the private run token")
        classified = store.classify_auto_mutation_receipt(
            "claim-request", 0, "task_add", {"body": "once"}
        )
        if classified.status != "RUNNING_COALESCED" or "run_token" in classified.__dataclass_fields__:
            raise SystemExit(f"running classification leaked or diverged: {classified}")
        cross = store.prepare_auto_mutation_receipts(
            "claim-request-other",
            [("task_add", {"body": "once"})],
        )
        if cross.status != "UNRESOLVED_ACTION_BLOCKED":
            raise SystemExit("running action did not block a different request")


def test_uncertain_recovery_and_safe_prepared_cleanup() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-mutation-recovery-") as temp:
        store = _store(temp)
        prepared = store.prepare_auto_mutation_receipts(
            "stale-request",
            [("task_update", {"id": 7})],
        )
        receipt_id = prepared.receipts[0].receipt_id
        claim = store.claim_auto_mutation_receipt(int(receipt_id), "stale-request")
        with store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET running_at = '2020-01-01T00:00:00Z' WHERE id = ?",
                (receipt_id,),
            )
        if store.recover_stale_auto_mutation_receipts("2021-01-01T00:00:00Z") != 1:
            raise SystemExit("stale running receipt was not recovered as uncertain")
        classified = store.classify_auto_mutation_receipt(
            "stale-request", 0, "task_update", {"id": 7}
        )
        if (
            classified.status != "UNCERTAIN_BLOCKED"
            or classified.result != "unknown"
            or classified.resolution != "stale_recovery"
        ):
            raise SystemExit(f"stale receipt classification diverged: {classified}")
        if store.mark_auto_mutation_uncertain(int(receipt_id), claim.run_token):
            raise SystemExit("an already uncertain receipt transitioned twice")
        if store.cleanup_prepared_auto_mutation_receipts("stale-request", "9999-12-31T23:59:59Z") != 0:
            raise SystemExit("prepared cleanup deleted a non-prepared batch")
        blocked = store.prepare_auto_mutation_receipts(
            "stale-request-other",
            [("task_update", {"id": 7})],
        )
        if blocked.status != "UNRESOLVED_ACTION_BLOCKED":
            raise SystemExit("uncertain action did not block a different request")

        manual = store.prepare_auto_mutation_receipts(
            "manual-uncertain-request",
            [("decision_update", {"id": 9})],
        )
        manual_id = int(manual.receipts[0].receipt_id)
        manual_claim = store.claim_auto_mutation_receipt(manual_id, "manual-uncertain-request")
        if not store.mark_auto_mutation_uncertain(manual_id, manual_claim.run_token):
            raise SystemExit("explicit running-to-uncertain transition failed")
        with store.connect() as conn:
            manual_row = conn.execute(
                "SELECT state, result, resolution, uncertainty_digest FROM auto_mutation_receipts "
                "WHERE id = ?",
                (manual_id,),
            ).fetchone()
        if (
            manual_row["state"] != "uncertain"
            or manual_row["result"] != "unknown"
            or manual_row["resolution"] != "manual_review"
            or len(manual_row["uncertainty_digest"] or "") != 64
        ):
            raise SystemExit(f"explicit uncertain receipt was malformed: {dict(manual_row)}")

        cleanup = store.prepare_auto_mutation_receipts(
            "cleanup-request",
            [("preference_set", {"key": "clean", "value": True})],
        )
        if store.prepare_auto_mutation_receipts(
            "cleanup-request",
            [("preference_set", {"key": "clean", "value": True})],
        ).status != "RESUMED":
            raise SystemExit("safe prepared batch could not resume before cleanup")
        if store.cleanup_prepared_auto_mutation_receipts(
            "cleanup-request", "9999-12-31T23:59:59Z"
        ) != len(cleanup.receipts):
            raise SystemExit("safe prepared cleanup did not delete the untouched batch")
        replacement = store.prepare_auto_mutation_receipts(
            "cleanup-replacement",
            [("preference_set", {"key": "clean", "value": True})],
        )
        if replacement.status != "PREPARED":
            raise SystemExit(f"prepared cleanup did not release the action: {replacement}")


def test_atomic_completion_repeat_and_approval_non_authorization() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-mutation-complete-") as temp:
        store = _store(temp)
        action = ("task_add", {"body": "completed once"})
        prepared = store.prepare_auto_mutation_receipts("completed-request", [action])
        receipt_id = int(prepared.receipts[0].receipt_id)
        claim = store.claim_auto_mutation_receipt(receipt_id, "completed-request")
        run_id = store.complete_auto_mutation_receipt(
            receipt_id=receipt_id,
            run_token=claim.run_token,
            session_id="auto-mutation-smoke",
            risk="LOCAL_SAFE",
            ok=True,
            output="ordinary audit output",
            metadata={"kind": "smoke"},
        )
        classified = store.classify_auto_mutation_receipt(
            "completed-request", 0, action[0], action[1]
        )
        if classified.status != "COMPLETED_COALESCED" or classified.tool_run_id != run_id:
            raise SystemExit(f"completion linkage was not classified: {classified}")
        with store.connect() as conn:
            run = conn.execute("SELECT * FROM tool_runs WHERE id = ?", (run_id,)).fetchone()
            receipt = conn.execute(
                "SELECT * FROM auto_mutation_receipts WHERE id = ?", (receipt_id,)
            ).fetchone()
        if (
            run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
            or receipt["tool_run_id"] != run_id
            or receipt["state"] != "completed"
            or receipt["result"] != "succeeded"
        ):
            raise SystemExit("completion was not an ordinary approved=0 linked tool run")
        repeated = store.prepare_auto_mutation_receipts("completed-request-new-token", [action])
        if repeated.status != "PREPARED":
            raise SystemExit(f"completed identical action blocked a new request: {repeated}")

        approval_id = store.add_pending_approval(
            "approval-smoke",
            "run the completed action",
            action[0],
            "proof isolation",
            planned_args=action[1],
        )
        with store.connect() as conn:
            conn.execute(
                "UPDATE pending_approvals SET status = 'approved' WHERE id = ?",
                (approval_id,),
            )
        evidence = store.classify_approval_execution_evidence(approval_id)
        if evidence.valid_execution_proof or evidence.linked_runs:
            raise SystemExit(f"auto mutation receipt authorized approval proof: {evidence}")


def test_malformed_booleans_fail_before_writes() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-mutation-bool-") as temp:
        store = _store(temp)
        prepared = store.prepare_auto_mutation_receipts(
            "bool-request",
            [("task_add", {"body": "bool"})],
        )
        receipt_id = int(prepared.receipts[0].receipt_id)
        claim = store.claim_auto_mutation_receipt(receipt_id, "bool-request")
        before = len(store.recent_tool_runs(limit=100))
        for malformed in (1, 0, "true", None):
            try:
                store.complete_auto_mutation_receipt(
                    receipt_id=receipt_id,
                    run_token=claim.run_token,
                    session_id="bool-smoke",
                    risk="LOCAL_SAFE",
                    ok=malformed,  # type: ignore[arg-type]
                    output="must not persist",
                )
            except TypeError:
                pass
            else:
                raise SystemExit(f"malformed boolean was accepted: {malformed!r}")
        if len(store.recent_tool_runs(limit=100)) != before:
            raise SystemExit("malformed boolean inserted a tool run")
        if store.classify_auto_mutation_receipt(
            "bool-request", 0, "task_add", {"body": "bool"}
        ).status != "RUNNING_COALESCED":
            raise SystemExit("malformed completion changed receipt state")


def test_completion_insert_failure_rolls_back_for_uncertain_resolution() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-mutation-audit-failure-") as temp:
        store = _store(temp)
        prepared = store.prepare_auto_mutation_receipts(
            "audit-failure-request",
            [("task_add", {"body": "handler already returned"})],
        )
        receipt_id = int(prepared.receipts[0].receipt_id)
        claim = store.claim_auto_mutation_receipt(receipt_id, "audit-failure-request")
        before = len(store.recent_tool_runs(limit=100))
        try:
            store.complete_auto_mutation_receipt(
                receipt_id=receipt_id,
                run_token=claim.run_token,
                session_id=None,  # type: ignore[arg-type]
                risk="LOCAL_SAFE",
                ok=True,
                output="must roll back",
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise SystemExit("invalid audit insertion unexpectedly completed")
        if len(store.recent_tool_runs(limit=100)) != before:
            raise SystemExit("failed atomic completion left a stray tool run")
        classified = store.classify_auto_mutation_receipt(
            "audit-failure-request", 0, "task_add", {"body": "handler already returned"}
        )
        if classified.status != "RUNNING_COALESCED":
            raise SystemExit(f"failed atomic completion did not preserve running: {classified}")
        if not store.mark_auto_mutation_uncertain(receipt_id, claim.run_token):
            raise SystemExit("audit-failed running receipt could not become uncertain")


def test_supported_tuple_forms_upgrade_and_fence_shared_scope_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-mutation-tuple-scope-") as temp:
        for tuple_size in (2, 3, 4):
            store = _store(temp, f"tuple-{tuple_size}.sqlite")
            legacy_tool = f"legacy_scope_tool_{tuple_size}"
            alias_tool = f"shared_scope_tool_{tuple_size}"
            shared_scope = f"shared_scope_{tuple_size}"
            aliases = (legacy_tool, alias_tool)
            request_token = f"PRIVATE-TUPLE-{tuple_size}-REQUEST-TOKEN"
            args = {
                "target_id": tuple_size,
                "value": f"PRIVATE-TUPLE-{tuple_size}-VALUE",
            }
            operation_args = {"target_id": tuple_size}
            if tuple_size == 2:
                legacy_action = (legacy_tool, args)
                upgraded_operation_args = args
            elif tuple_size == 3:
                legacy_action = (legacy_tool, args, operation_args)
                upgraded_operation_args = operation_args
            else:
                legacy_action = (
                    legacy_tool,
                    args,
                    legacy_tool,
                    operation_args,
                )
                upgraded_operation_args = operation_args

            prepared = store.prepare_auto_mutation_receipts(
                request_token, [legacy_action]
            )
            if prepared.status != "PREPARED" or prepared.receipts[0].receipt_id is None:
                raise SystemExit(
                    f"{tuple_size}-tuple unresolved fixture was not prepared: {prepared}"
                )
            receipt_id = int(prepared.receipts[0].receipt_id)
            claim = store.claim_auto_mutation_receipt(receipt_id, request_token)
            if claim.status != "CLAIMED" or not claim.run_token:
                raise SystemExit(
                    f"{tuple_size}-tuple unresolved fixture was not claimed: {claim}"
                )
            if not store.mark_auto_mutation_uncertain(receipt_id, claim.run_token):
                raise SystemExit(
                    f"{tuple_size}-tuple unresolved fixture did not become uncertain"
                )

            upgraded_action = (
                legacy_tool,
                args,
                shared_scope,
                upgraded_operation_args,
                aliases,
            )
            upgraded_inspection = store.inspect_auto_mutation_receipts(
                request_token, [upgraded_action]
            )
            upgraded_classification = store.classify_auto_mutation_receipt(
                request_token,
                0,
                legacy_tool,
                args,
                operation_args=upgraded_operation_args,
                operation_scope=shared_scope,
                legacy_operation_aliases=aliases,
            )
            upgraded_preparation = store.prepare_auto_mutation_receipts(
                request_token, [upgraded_action]
            )
            if (
                upgraded_inspection.status != "RESUMED"
                or upgraded_preparation.status != "RESUMED"
                or upgraded_inspection.receipts[0].status != "UNCERTAIN_BLOCKED"
                or upgraded_preparation.receipts[0].status != "UNCERTAIN_BLOCKED"
                or upgraded_classification.status != "UNCERTAIN_BLOCKED"
                or upgraded_inspection.receipts[0].receipt_id != receipt_id
                or upgraded_preparation.receipts[0].receipt_id != receipt_id
                or upgraded_classification.receipt_id != receipt_id
            ):
                raise SystemExit(
                    f"{tuple_size}-tuple same-token shared-scope upgrade diverged: "
                    f"{upgraded_inspection}, {upgraded_classification}, "
                    f"{upgraded_preparation}"
                )

            alias_token = f"PRIVATE-TUPLE-{tuple_size}-ALIAS-TOKEN"
            alias_args = {
                "target_id": tuple_size,
                "value": f"PRIVATE-TUPLE-{tuple_size}-ALIAS-VALUE",
            }
            alias_action = (
                alias_tool,
                alias_args,
                shared_scope,
                upgraded_operation_args,
                aliases,
            )
            alias_inspection = store.inspect_auto_mutation_receipts(
                alias_token, [alias_action]
            )
            alias_classification = store.classify_auto_mutation_receipt(
                alias_token,
                0,
                alias_tool,
                alias_args,
                operation_args=upgraded_operation_args,
                operation_scope=shared_scope,
                legacy_operation_aliases=aliases,
            )
            alias_preparation = store.prepare_auto_mutation_receipts(
                alias_token, [alias_action]
            )
            if (
                alias_inspection.status != "UNRESOLVED_ACTION_BLOCKED"
                or alias_classification.status != "UNRESOLVED_ACTION_BLOCKED"
                or alias_preparation.status != "UNRESOLVED_ACTION_BLOCKED"
            ):
                raise SystemExit(
                    f"{tuple_size}-tuple unresolved per-tool scope did not fence "
                    f"the shared-scope alias: {alias_inspection}, "
                    f"{alias_classification}, {alias_preparation}"
                )

            with store.connect() as conn:
                rows = list(
                    conn.execute(
                        "SELECT id, operation_scope FROM auto_mutation_receipts"
                    )
                )
            if len(rows) != 1 or tuple(rows[0]) != (receipt_id, legacy_tool):
                raise SystemExit(
                    f"{tuple_size}-tuple alias fence changed its per-tool receipt: "
                    f"{[tuple(row) for row in rows]}"
                )


def test_legacy_alias_fencing_reconciliation_and_scoped_operations() -> None:
    aliases = ("complete_task", "update_task_status")

    def scoped_action(
        tool_name: str,
        task_id: int,
        status: str | None = None,
    ) -> tuple[str, dict[str, object], str, dict[str, int], tuple[str, ...]]:
        args: dict[str, object] = {"task_id": task_id}
        if status is not None:
            args["status"] = status
        return (tool_name, args, "task_status", {"task_id": task_id}, aliases)

    def classify(
        store: MemoryStore,
        request_token: str,
        action: tuple[str, dict[str, object], str, dict[str, int], tuple[str, ...]],
    ) -> object:
        tool_name, args, operation_scope, operation_args, legacy_aliases = action
        return store.classify_auto_mutation_receipt(
            request_token,
            0,
            tool_name,
            args,
            operation_args=operation_args,
            operation_scope=operation_scope,
            legacy_operation_aliases=legacy_aliases,
        )

    def assert_new_request_statuses(
        store: MemoryStore,
        request_token: str,
        action: tuple[str, dict[str, object], str, dict[str, int], tuple[str, ...]],
        expected_inspection: str,
        expected_classification: str,
        expected_preparation: str,
        label: str,
    ) -> object:
        inspected = store.inspect_auto_mutation_receipts(request_token, [action])
        classified = classify(store, request_token, action)
        prepared = store.prepare_auto_mutation_receipts(request_token, [action])
        if (
            inspected.status != expected_inspection
            or classified.status != expected_classification
            or prepared.status != expected_preparation
        ):
            raise SystemExit(
                f"{label} inspect/classify/prepare diverged: "
                f"{inspected}, {classified}, {prepared}"
            )
        return prepared

    with TemporaryDirectory(prefix="jarvis-auto-mutation-legacy-alias-") as temp:
        store = _store(temp)
        task_a = store.add_task(TaskRecord("legacy alias task A"))
        task_b = store.add_task(TaskRecord("legacy alias task B"))
        legacy_token = "PRIVATE-LEGACY-TASK-STATUS-TOKEN"
        legacy_status = "paused"
        legacy_args = {"task_id": task_a, "status": legacy_status}
        legacy = store.prepare_auto_mutation_receipts(
            legacy_token,
            [("update_task_status", legacy_args)],
        )
        if legacy.status != "PREPARED" or legacy.receipts[0].receipt_id is None:
            raise SystemExit(f"legacy full-args receipt setup failed: {legacy}")
        legacy_id = int(legacy.receipts[0].receipt_id)
        claim = store.claim_auto_mutation_receipt(legacy_id, legacy_token)
        if claim.status != "CLAIMED" or not claim.run_token:
            raise SystemExit(f"legacy full-args receipt claim failed: {claim}")
        if not store.mark_auto_mutation_uncertain(legacy_id, claim.run_token):
            raise SystemExit("legacy full-args receipt did not become unresolved")
        with store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET operation_scope = NULL WHERE id = ?",
                (legacy_id,),
            )
            legacy_row = conn.execute(
                "SELECT action_digest, operation_digest, operation_scope, uncertainty_digest "
                "FROM auto_mutation_receipts WHERE id = ?",
                (legacy_id,),
            ).fetchone()
        full_args_digest = auto_mutation_action_digest("update_task_status", legacy_args)
        if (
            legacy_row["action_digest"] != full_args_digest
            or legacy_row["operation_digest"] != full_args_digest
            or legacy_row["operation_scope"] is not None
            or not legacy_row["uncertainty_digest"]
        ):
            raise SystemExit(f"legacy full-args receipt fixture was malformed: {dict(legacy_row)}")

        exact_actions = [("update_task_status", legacy_args)]
        exact_inspected = store.inspect_auto_mutation_receipts(legacy_token, exact_actions)
        exact_classified = store.classify_auto_mutation_receipt(
            legacy_token,
            0,
            "update_task_status",
            legacy_args,
        )
        exact_prepared = store.prepare_auto_mutation_receipts(legacy_token, exact_actions)
        if (
            exact_inspected.status != "RESUMED"
            or exact_classified.status != "UNCERTAIN_BLOCKED"
            or exact_prepared.status != "RESUMED"
            or exact_inspected.receipts[0].status != "UNCERTAIN_BLOCKED"
            or exact_prepared.receipts[0].receipt_id != legacy_id
        ):
            raise SystemExit(
                "same-token exact legacy replay did not resume consistently: "
                f"{exact_inspected}, {exact_classified}, {exact_prepared}"
            )

        changed_same_token = scoped_action(
            "update_task_status",
            task_a,
            "PRIVATE-SAME-TOKEN-CHANGED-STATUS",
        )
        assert_new_request_statuses(
            store,
            legacy_token,
            changed_same_token,
            "REQUEST_TOKEN_COLLISION",
            "REQUEST_TOKEN_COLLISION",
            "REQUEST_TOKEN_COLLISION",
            "same-token changed legacy action",
        )

        blocked_cases = (
            (
                "PRIVATE-LEGACY-BLOCK-EXACT-TOKEN",
                scoped_action(
                    "update_task_status",
                    task_a,
                    legacy_status,
                ),
                "legacy exact-action target",
            ),
        )
        for request_token, action, label in blocked_cases:
            assert_new_request_statuses(
                store,
                request_token,
                action,
                "UNRESOLVED_ACTION_BLOCKED",
                "UNRESOLVED_ACTION_BLOCKED",
                "UNRESOLVED_ACTION_BLOCKED",
                label,
            )

        other_task_action = scoped_action(
            "update_task_status",
            task_b,
            "PRIVATE-LEGACY-OTHER-TASK-STATUS",
        )
        other_task_inspection = store.inspect_auto_mutation_receipts(
            "PRIVATE-LEGACY-OTHER-TASK-TOKEN",
            [other_task_action],
        )
        other_task_classification = classify(
            store,
            "PRIVATE-LEGACY-OTHER-TASK-TOKEN",
            other_task_action,
        )
        if (
            other_task_inspection.status != "ABSENT"
            or other_task_classification.status != "NOT_PREPARED"
        ):
            raise SystemExit(
                "legacy receipt blocked a different task target: "
                f"{other_task_inspection}, {other_task_classification}"
            )

        resolved = store.resolve_auto_mutation_receipt(
            legacy_id,
            "confirmed_not_applied",
            expected_uncertainty_digest=str(legacy_row["uncertainty_digest"]),
            session_id="legacy-alias-reconciliation-smoke",
            reviewed_at="2026-07-13T00:00:00Z",
            output="legacy task-status receipt reviewed without rerun",
            metadata={"mutation_rerun": False},
        )
        if resolved.status != "RESOLVED":
            raise SystemExit(f"legacy task-status reconciliation failed: {resolved}")

        scoped_a_token = "PRIVATE-SCOPED-TASK-A-TOKEN"
        scoped_a = scoped_action(
            "update_task_status",
            task_a,
            "PRIVATE-SCOPED-TASK-A-STATUS",
        )
        prepared_a = assert_new_request_statuses(
            store,
            scoped_a_token,
            scoped_a,
            "ABSENT",
            "NOT_PREPARED",
            "PREPARED",
            "reconciled legacy fence release",
        )
        if prepared_a.receipts[0].receipt_id is None:
            raise SystemExit("released scoped task A did not receive a receipt")

        for request_token, action, label in (
            (
                "PRIVATE-SCOPED-A-STATUS-BLOCK-TOKEN",
                scoped_action(
                    "update_task_status",
                    task_a,
                    "PRIVATE-SCOPED-TASK-A-CHANGED-STATUS",
                ),
                "scoped task A status variant",
            ),
            (
                "PRIVATE-SCOPED-A-ALIAS-BLOCK-TOKEN",
                scoped_action("complete_task", task_a),
                "scoped task A tool alias",
            ),
        ):
            assert_new_request_statuses(
                store,
                request_token,
                action,
                "UNRESOLVED_ACTION_BLOCKED",
                "UNRESOLVED_ACTION_BLOCKED",
                "UNRESOLVED_ACTION_BLOCKED",
                label,
            )

        scoped_b_token = "PRIVATE-SCOPED-TASK-B-TOKEN"
        scoped_b = scoped_action(
            "complete_task",
            task_b,
        )
        assert_new_request_statuses(
            store,
            scoped_b_token,
            scoped_b,
            "ABSENT",
            "NOT_PREPARED",
            "PREPARED",
            "scoped task A versus task B",
        )

        private_values = (
            legacy_token,
            legacy_status,
            "PRIVATE-SAME-TOKEN-CHANGED-STATUS",
            "PRIVATE-LEGACY-CHANGED-STATUS",
            "PRIVATE-LEGACY-OTHER-TASK-STATUS",
            "PRIVATE-SCOPED-TASK-A-STATUS",
            "PRIVATE-SCOPED-TASK-A-CHANGED-STATUS",
            *(request_token for request_token, _action, _label in blocked_cases),
            scoped_a_token,
            scoped_b_token,
            "PRIVATE-SCOPED-A-STATUS-BLOCK-TOKEN",
            "PRIVATE-SCOPED-A-ALIAS-BLOCK-TOKEN",
        )
        with store.connect() as conn:
            receipt_rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM auto_mutation_receipts ORDER BY id"
                )
            ]
        serialized = repr(receipt_rows)
        if any(value in serialized for value in private_values):
            raise SystemExit("legacy/scoped receipt rows exposed raw request or action inputs")
        scopes = [row["operation_scope"] for row in receipt_rows]
        if scopes != [None, "task_status", "task_status"]:
            raise SystemExit(f"legacy/scoped operation scopes were not preserved: {scopes}")


def test_legacy_ambiguity_reconstruction_is_cached_per_batch() -> None:
    aliases = ("complete_task", "complete_task_with_evidence", "update_task_status")
    with TemporaryDirectory(prefix="jarvis-auto-mutation-legacy-cache-") as temp:
        store = _store(temp)
        task_ids = [
            store.add_task(TaskRecord(f"legacy cache task {index}"))
            for index in range(50)
        ]
        legacy_token = "PRIVATE-LEGACY-CACHE-TOKEN"
        legacy = store.prepare_auto_mutation_receipts(
            legacy_token,
            [
                (
                    "update_task_status",
                    {"task_id": task_ids[-1], "status": "paused"},
                )
            ],
        )
        if legacy.status != "PREPARED" or legacy.receipts[0].receipt_id is None:
            raise SystemExit(f"legacy cache receipt setup failed: {legacy}")
        with store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET operation_scope = NULL WHERE id = ?",
                (legacy.receipts[0].receipt_id,),
            )

        scan_count = 0
        original_connect = store.connect

        def traced_connect() -> sqlite3.Connection:
            nonlocal scan_count
            conn = original_connect()

            def trace(statement: str) -> None:
                nonlocal scan_count
                if "SELECT id FROM tasks WHERE typeof(id)" in statement:
                    scan_count += 1

            conn.set_trace_callback(trace)
            return conn

        store.connect = traced_connect  # type: ignore[method-assign]
        try:
            result = store.inspect_auto_mutation_receipts(
                "PRIVATE-LEGACY-CACHE-CONTENDER",
                [
                    (
                        "complete_task",
                        {"task_id": task_id},
                        "task_status",
                        {"task_id": task_id},
                        aliases,
                    )
                    for task_id in task_ids
                ],
            )
        finally:
            store.connect = original_connect  # type: ignore[method-assign]
        if result.status != "UNRESOLVED_ACTION_BLOCKED" or scan_count != 1:
            raise SystemExit(
                "legacy ambiguity reconstruction repeated inside one batch: "
                f"result={result}, scans={scan_count}"
            )


def test_classifier_rejects_invalid_legacy_alias_sets() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-mutation-classifier-aliases-") as temp:
        store = _store(temp)
        invalid_alias_sets = (
            tuple(f"fixture_alias_{index}" for index in range(33)),
            ("other_tool",),
        )
        for aliases in invalid_alias_sets:
            try:
                store.classify_auto_mutation_receipt(
                    "classifier-alias-validation",
                    0,
                    "fixture_tool",
                    {"value": 1},
                    operation_args={"target": 1},
                    operation_scope="shared_fixture",
                    legacy_operation_aliases=aliases,
                )
            except ValueError:
                continue
            raise SystemExit(
                f"classifier accepted invalid legacy alias set of size {len(aliases)}"
            )


def test_legacy_migration_current_schema_and_read_only_access() -> None:
    with TemporaryDirectory(prefix="jarvis-auto-mutation-migration-") as temp:
        db_path = Path(temp) / "legacy.sqlite"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE legacy_marker(value TEXT NOT NULL)")
            conn.execute("INSERT INTO legacy_marker(value) VALUES ('preserved')")
            conn.execute(
                """
                CREATE TABLE auto_mutation_receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_digest TEXT NOT NULL,
                    action_digest TEXT NOT NULL,
                    uncertainty_digest TEXT,
                    tool_name TEXT NOT NULL,
                    action_index INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    result TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    run_token TEXT UNIQUE,
                    tool_run_id INTEGER UNIQUE,
                    prepared_at TEXT NOT NULL,
                    running_at TEXT,
                    completed_at TEXT,
                    uncertain_at TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE(request_digest, action_index)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO auto_mutation_receipts (
                    request_digest, action_digest, uncertainty_digest, tool_name,
                    action_index, state, result, resolution, run_token, tool_run_id,
                    prepared_at, running_at, completed_at, uncertain_at, updated_at
                ) VALUES (?, ?, NULL, 'remember', 0, 'prepared', 'pending', 'none',
                          NULL, NULL, '2026-01-01T00:00:00Z', NULL, NULL, NULL,
                          '2026-01-01T00:00:00Z')
                """,
                ("a" * 64, "b" * 64),
            )
        store = MemoryStore(db_path)
        store.init()
        store.init()
        with store.connect() as conn:
            marker = conn.execute("SELECT value FROM legacy_marker").fetchone()[0]
            table_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'auto_mutation_receipts'"
            ).fetchone()[0]
            migrated = conn.execute(
                "SELECT action_digest, operation_digest, operation_scope, state "
                "FROM auto_mutation_receipts"
            ).fetchone()
            scope_column = conn.execute(
                "SELECT * FROM pragma_table_info('auto_mutation_receipts') "
                "WHERE name = 'operation_scope'"
            ).fetchone()
        if marker != "preserved" or "approval_id" in table_sql or "planned_args" in table_sql:
            raise SystemExit("additive migration damaged legacy data or coupled receipts to approval proof")
        if scope_column is None or scope_column["notnull"] != 0:
            raise SystemExit("operation_scope migration was missing or not nullable")
        if tuple(migrated) != ("b" * 64, "b" * 64, None, "prepared"):
            raise SystemExit("legacy receipt identity/scope was not migrated safely")
        read_only = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            count = read_only.execute("SELECT COUNT(*) FROM auto_mutation_receipts").fetchone()[0]
        finally:
            read_only.close()
        if count != 1:
            raise SystemExit("legacy migrated receipt ledger did not preserve its row")


def main() -> None:
    test_canonical_digests_and_privacy_shape()
    test_atomic_prepare_resume_collision_and_cross_request_block()
    test_intra_batch_duplicate_operations_fail_before_receipt_insertion()
    test_concurrent_claim_has_one_private_winner()
    test_uncertain_recovery_and_safe_prepared_cleanup()
    test_atomic_completion_repeat_and_approval_non_authorization()
    test_malformed_booleans_fail_before_writes()
    test_completion_insert_failure_rolls_back_for_uncertain_resolution()
    test_supported_tuple_forms_upgrade_and_fence_shared_scope_aliases()
    test_legacy_alias_fencing_reconciliation_and_scoped_operations()
    test_legacy_ambiguity_reconstruction_is_cached_per_batch()
    test_classifier_rejects_invalid_legacy_alias_sets()
    test_legacy_migration_current_schema_and_read_only_access()
    print("Auto mutation receipt smoke passed")


if __name__ == "__main__":
    main()
