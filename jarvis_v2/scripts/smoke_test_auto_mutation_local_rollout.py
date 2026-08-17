from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import jarvis_v2.agent.runtime as runtime_module
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction
from jarvis_v2.memory.decision_projection import reconcile_pending_decision_projections
from jarvis_v2.memory.preference_projection import reconcile_pending_preference_projections
from jarvis_v2.memory.store import (
    MemoryRecord,
    PersonRecord,
    auto_mutation_action_digest,
    auto_mutation_request_digest,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
)


class StaticPlanner:
    def __init__(self, action: PlannedAction):
        self.action = action

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise a contracted local auto mutation.",
            [self.action],
            needs_model=False,
        )


class InjectedProcessDeath(BaseException):
    pass


def _runtime(root: Path, tool_name: str, args: dict[str, Any]) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(PlannedAction(tool_name, args, "local rollout smoke"))
    return runtime


def _receipt_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [
            dict(row)
            for row in conn.execute("SELECT * FROM auto_mutation_receipts ORDER BY id")
        ]


def _tool_run_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM tool_runs ORDER BY id")]


def _projection_graph_counts(runtime: JarvisRuntime, *, kind: str) -> dict[str, int]:
    if kind not in {"decision", "preference"}:
        raise ValueError(f"unsupported projection graph kind: {kind}")
    with runtime.store.connect() as conn:
        return {
            "links": int(
                conn.execute(f"SELECT COUNT(*) FROM {kind}_memory_links").fetchone()[0]
            ),
            "pending": int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {kind}_projection_jobs WHERE state = 'pending'"
                ).fetchone()[0]
            ),
            "completed": int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {kind}_projection_jobs WHERE state = 'completed'"
                ).fetchone()[0]
            ),
        }


def _assert_contracts(runtime: JarvisRuntime) -> None:
    expected_effects = {
        "record_feedback": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "record_decision": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "set_preference": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "add_person": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "log_interaction": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "save_skill": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "remember": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "memory_tree_summary": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
    }
    for name, effects in expected_effects.items():
        tool = runtime.registry.get(name)
        contract = tool.auto_mutation_contract
        if (
            contract is None
            or contract.version != AUTO_MUTATION_CONTRACT_VERSION
            or contract.effects != effects
            or contract.replay_policy
            is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
            or contract.crash_policy
            is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
        ):
            raise SystemExit(f"{name} registry auto-mutation contract diverged: {contract}")


def _assert_one_linked_ordinary_audit(runtime: JarvisRuntime, *, label: str) -> None:
    receipts = _receipt_rows(runtime)
    runs = _tool_run_rows(runtime)
    successful_runs = [row for row in runs if row["ok"] == 1]
    if len(receipts) != 1 or len(successful_runs) != 1:
        raise SystemExit(f"{label} should have one receipt and one audit row: {receipts} / {runs}")
    receipt = receipts[0]
    run = successful_runs[0]
    if (
        receipt["state"] != "completed"
        or receipt["result"] != "succeeded"
        or receipt["tool_run_id"] != run["id"]
        or run["approved"] != 0
        or run["approval_id"] is not None
        or run["approval_action_digest"] is not None
    ):
        raise SystemExit(f"{label} receipt was not linked to one ordinary approved=0 audit: {receipt} / {run}")


def _assert_completed_receipts_link_ordinary_audits(
    runtime: JarvisRuntime, *, expected: int, label: str
) -> None:
    receipts = _receipt_rows(runtime)
    runs = _tool_run_rows(runtime)
    successful_runs = [row for row in runs if row["ok"] == 1]
    if len(receipts) != expected or len(successful_runs) != expected:
        raise SystemExit(f"{label} expected {expected} completed receipts/audits: {receipts} / {runs}")
    runs_by_id = {row["id"]: row for row in successful_runs}
    linked_ids = []
    for receipt in receipts:
        run = runs_by_id.get(receipt["tool_run_id"])
        if (
            receipt["state"] != "completed"
            or receipt["result"] != "succeeded"
            or run is None
            or run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
        ):
            raise SystemExit(f"{label} malformed receipt/audit linkage: {receipt} / {run}")
        linked_ids.append(run["id"])
    if len(set(linked_ids)) != expected:
        raise SystemExit(f"{label} receipts did not link to distinct ordinary audits: {linked_ids}")


def _assert_receipt_privacy(
    runtime: JarvisRuntime,
    *,
    request_tokens: list[str],
    actions: list[tuple[str, dict[str, Any]]],
    raw_content: list[str],
    exposed_metadata: list[dict[str, Any]],
    label: str,
) -> None:
    rows = _receipt_rows(runtime)
    rows_text = json.dumps(rows, sort_keys=True, default=str)
    for secret in [*request_tokens, *raw_content]:
        if secret and secret in rows_text:
            raise SystemExit(f"{label} receipt row leaked raw private content: {secret!r}")

    digests = [auto_mutation_request_digest(token) for token in request_tokens]
    digests.extend(auto_mutation_action_digest(name, args) for name, args in actions)
    private_receipt_values = [
        str(row[key])
        for row in rows
        for key in ("request_digest", "action_digest", "operation_digest", "uncertainty_digest", "run_token")
        if row.get(key)
    ]
    exposed_text = json.dumps(exposed_metadata, sort_keys=True, default=str)
    for secret in [*request_tokens, *digests, *private_receipt_values, *raw_content]:
        if secret and secret in exposed_text:
            raise SystemExit(f"{label} exposed receipt metadata leaked token, digest, or content: {secret!r}")


def _assert_replay(result: Any, expected_kind: str, *, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} should return one replay result: {result.tool_results}")
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != expected_kind:
        raise SystemExit(f"{label} returned the wrong replay state: {item}")


def _assert_no_absolute_path_metadata(
    metadata_items: list[dict[str, Any]], *, root: Path, label: str
) -> None:
    for metadata in metadata_items:
        if "path" in metadata:
            raise SystemExit(f"{label} exposed absolute path metadata: {metadata}")
    exposed = json.dumps(metadata_items, sort_keys=True, default=str)
    for fragment in (str(root), "/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in exposed:
            raise SystemExit(f"{label} leaked a local path in metadata: {fragment!r}")


def test_record_feedback_coalescing_repeat_audit_and_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-feedback-") as temp:
        root = Path(temp)
        token_one = "feedback-private-request-token-one"
        token_two = "feedback-private-request-token-two"
        body = "RAW-FEEDBACK-CONTENT-LOCAL-ROLLOUT-ALPHA"
        args = {"title": "Local rollout feedback", "theme": "safety", "body": body}
        runtime = _runtime(root, "record_feedback", args)
        _assert_contracts(runtime)

        first = runtime.handle("record feedback first", request_token=token_one)
        if not first.tool_results[0].ok:
            raise SystemExit(f"record_feedback first execution failed: {first.tool_results}")
        replay = runtime.handle("record feedback replay", request_token=token_one)
        _assert_replay(replay, "auto_mutation_completed_replay", label="record_feedback same token")
        if len(runtime.store.list_memories(limit=20)) != 1:
            raise SystemExit("record_feedback same-token replay created an additional memory")
        _assert_one_linked_ordinary_audit(runtime, label="record_feedback same-token coalescing")

        repeated = runtime.handle("record feedback intentional repeat", request_token=token_two)
        if not repeated.tool_results[0].ok:
            raise SystemExit(f"record_feedback new-token repeat failed: {repeated.tool_results}")
        rows = runtime.store.list_memories(limit=20)
        if len(rows) != 2 or any(row["body"] != body for row in rows):
            raise SystemExit(f"record_feedback new token did not create exactly one new memory: {rows}")
        notes = list((runtime.vault.root_path / "Memory Tree" / "Records").glob("*.md"))
        if len(notes) != 2 or any(body not in path.read_text(encoding="utf-8") for path in notes):
            raise SystemExit(f"record_feedback new token did not create exactly one new note: {notes}")
        _assert_completed_receipts_link_ordinary_audits(
            runtime, expected=2, label="record_feedback intentional repeat"
        )
        _assert_receipt_privacy(
            runtime,
            request_tokens=[token_one, token_two],
            actions=[("record_feedback", args)],
            raw_content=[body],
            exposed_metadata=[item.metadata for item in [first.tool_results[0], replay.tool_results[0], repeated.tool_results[0]]],
            label="record_feedback",
        )


def test_record_decision_coalescing_repeat_audit_and_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-decision-") as temp:
        root = Path(temp)
        token_one = "decision-private-request-token-one"
        token_two = "decision-private-request-token-two"
        rationale = "RAW-DECISION-RATIONALE-LOCAL-ROLLOUT-EPSILON"
        impact = "RAW-DECISION-IMPACT-LOCAL-ROLLOUT-ZETA"
        args = {
            "title": "Use the durable decision path",
            "rationale": rationale,
            "impact": impact,
        }
        runtime = _runtime(root, "record_decision", args)

        first = runtime.handle("record decision first", request_token=token_one)
        if not first.tool_results[0].ok:
            raise SystemExit(f"record_decision first execution failed: {first.tool_results}")
        replay = runtime.handle("record decision replay", request_token=token_one)
        _assert_replay(replay, "auto_mutation_completed_replay", label="record_decision same token")
        if len(runtime.store.list_decisions(status=None, limit=20)) != 1:
            raise SystemExit("record_decision same-token replay created an additional decision")
        if len(runtime.store.list_memories(limit=20)) != 1:
            raise SystemExit("record_decision same-token replay created an additional memory")
        _assert_one_linked_ordinary_audit(runtime, label="record_decision same-token coalescing")

        repeated = runtime.handle("record decision intentional repeat", request_token=token_two)
        if not repeated.tool_results[0].ok:
            raise SystemExit(f"record_decision new-token repeat failed: {repeated.tool_results}")
        decisions = runtime.store.list_decisions(status=None, limit=20)
        memories = runtime.store.list_memories(limit=20)
        if len(decisions) != 2 or any(row["rationale"] != rationale for row in decisions):
            raise SystemExit(f"record_decision new token did not create exactly one new decision: {decisions}")
        if len(memories) != 2 or any(rationale not in row["body"] for row in memories):
            raise SystemExit(f"record_decision new token did not create exactly one new memory: {memories}")
        decision_notes = list((runtime.vault.root_path / "Decisions").glob("*.md"))
        if len(decision_notes) != 2 or any(rationale not in path.read_text(encoding="utf-8") for path in decision_notes):
            raise SystemExit(f"record_decision new token did not create exactly one new decision note: {decision_notes}")
        _assert_completed_receipts_link_ordinary_audits(
            runtime, expected=2, label="record_decision intentional repeat"
        )
        _assert_receipt_privacy(
            runtime,
            request_tokens=[token_one, token_two],
            actions=[("record_decision", args)],
            raw_content=[rationale, impact],
            exposed_metadata=[item.metadata for item in [first.tool_results[0], replay.tool_results[0], repeated.tool_results[0]]],
            label="record_decision",
        )


def test_record_decision_typed_and_semantic_preflight() -> None:
    cases = [
        ({}, "tool_arguments_invalid"),
        ({"title": None}, "tool_arguments_invalid"),
        ({"title": 0}, "tool_arguments_invalid"),
        ({"title": 1}, "tool_arguments_invalid"),
        ({"title": True}, "tool_arguments_invalid"),
        ({"title": []}, "tool_arguments_invalid"),
        ({"title": {}}, "tool_arguments_invalid"),
        ({"title": "valid", "rationale": None}, "tool_arguments_invalid"),
        ({"title": "valid", "impact": 1}, "tool_arguments_invalid"),
        ({"title": "valid", "extra": "rejected"}, "tool_arguments_invalid"),
        ({"title": "", "rationale": "none"}, "auto_mutation_semantic_preflight_rejected"),
        ({"title": "   ", "rationale": "none"}, "auto_mutation_semantic_preflight_rejected"),
        (
            {"title": "/\x55sers/example/private/decision", "impact": "none"},
            "auto_mutation_semantic_preflight_rejected",
        ),
    ]
    for index, (args, expected_failure) in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-local-rollout-decision-preflight-") as temp:
            runtime = _runtime(Path(temp), "record_decision", args)
            result = runtime.handle("invalid decision", request_token=f"decision-preflight-{index}")
            item = result.tool_results[0]
            if item.metadata.get("failure_kind") != expected_failure:
                raise SystemExit(f"record_decision preflight drifted for case {index}: {item}")
            if _receipt_rows(runtime) or runtime.store.list_decisions(status=None, limit=20):
                raise SystemExit(f"record_decision invalid case {index} crossed the receipt/write boundary")
            if runtime.store.list_memories(limit=20):
                raise SystemExit(f"record_decision invalid case {index} wrote memory")
            exposed = json.dumps(
                {"response": result.response, "metadata": item.metadata},
                ensure_ascii=False,
                default=str,
            )
            if "/\x55sers/example/private" in exposed:
                raise SystemExit("record_decision preflight exposed a local path")

    with TemporaryDirectory(prefix="jarvis-local-rollout-decision-optional-") as temp:
        runtime = _runtime(Path(temp), "record_decision", {"title": "Keep optional fields optional"})
        result = runtime.handle("minimal decision", request_token="decision-optional-fields")
        if not result.tool_results[0].ok:
            raise SystemExit(f"record_decision rejected omitted optional fields: {result.tool_results}")
        decisions = runtime.store.list_decisions(status=None, limit=20)
        if len(decisions) != 1 or decisions[0]["rationale"] or decisions[0]["impact"]:
            raise SystemExit(f"record_decision optional-field compatibility drifted: {decisions}")

    with TemporaryDirectory(prefix="jarvis-local-rollout-decision-operation-") as temp:
        runtime = _runtime(Path(temp), "record_decision", {"title": "Equivalent decision shape"})
        original_write_decision = runtime.vault.write_decision_with_evidence

        def fail_after_database_commit(
            _decision: Any, *, store_identity: str
        ) -> tuple[Path, str]:
            del store_identity
            raise OSError("representative decision mirror failure")

        runtime.vault.write_decision_with_evidence = fail_after_database_commit  # type: ignore[method-assign]
        failed = runtime.handle("minimal decision failure", request_token="decision-operation-one")
        runtime.vault.write_decision_with_evidence = original_write_decision  # type: ignore[method-assign]
        if failed.tool_results[0].metadata.get("failure_kind") != "auto_mutation_handler_failed":
            raise SystemExit(f"record_decision partial write did not become uncertain: {failed.tool_results}")
        if len(runtime.store.list_memories(limit=20)) != 1:
            raise SystemExit("record_decision mirror failure did not atomically preserve its memory")
        graph = _projection_graph_counts(runtime, kind="decision")
        if graph != {"links": 1, "pending": 1, "completed": 0}:
            raise SystemExit(f"record_decision mirror failure left a partial graph: {graph}")
        repaired = reconcile_pending_decision_projections(runtime.store, runtime.vault)
        if repaired.pending != 0 or _projection_graph_counts(
            runtime, kind="decision"
        ) != {"links": 1, "pending": 0, "completed": 1}:
            raise SystemExit(f"record_decision pending projection did not reconcile: {repaired}")
        runtime.planner = StaticPlanner(
            PlannedAction(
                "record_decision",
                {"title": "Equivalent decision shape", "rationale": "", "impact": ""},
                "equivalent canonical decision payload",
            )
        )
        blocked = runtime.handle("full-shape decision retry", request_token="decision-operation-two")
        _assert_replay(blocked, "auto_mutation_unresolved_action", label="record_decision operation identity")
        if (
            len(runtime.store.list_decisions(status=None, limit=20)) != 1
            or len(runtime.store.list_memories(limit=20)) != 1
            or _projection_graph_counts(runtime, kind="decision")["links"] != 1
        ):
            raise SystemExit("record_decision equivalent payload bypassed uncertain operation fencing")


def test_set_preference_target_fencing_and_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-preference-") as temp:
        root = Path(temp)
        args = {
            "key": "Rollout Tone",
            "value": "RAW-PREFERENCE-VALUE-LOCAL-ROLLOUT-ETA",
            "category": "Communication",
        }
        runtime = _runtime(root, "set_preference", args)
        first = runtime.handle("set preference first", request_token="preference-request-one")
        replay = runtime.handle("set preference replay", request_token="preference-request-one")
        if not first.tool_results[0].ok:
            raise SystemExit(f"set_preference first execution failed: {first.tool_results}")
        _assert_replay(replay, "auto_mutation_completed_replay", label="set_preference same token")
        repeated = runtime.handle("set preference repeat", request_token="preference-request-two")
        if not repeated.tool_results[0].ok:
            raise SystemExit(f"set_preference intentional repeat failed: {repeated.tool_results}")
        rows = runtime.store.list_preferences(status=None, limit=20)
        memories = runtime.store.list_memories(limit=20)
        graph = _projection_graph_counts(runtime, kind="preference")
        if (
            len(rows) != 1
            or len(memories) != 1
            or graph != {"links": 1, "pending": 0, "completed": 1}
        ):
            raise SystemExit(f"set_preference replay/upsert effects diverged: {rows} / {memories}")
        _assert_completed_receipts_link_ordinary_audits(
            runtime, expected=2, label="set_preference intentional repeat"
        )
        _assert_receipt_privacy(
            runtime,
            request_tokens=["preference-request-one", "preference-request-two"],
            actions=[("set_preference", args)],
            raw_content=[args["value"]],
            exposed_metadata=[
                first.tool_results[0].metadata,
                replay.tool_results[0].metadata,
                repeated.tool_results[0].metadata,
            ],
            label="set_preference",
        )


def test_set_preference_semantic_preflight_and_uncertain_target_fence() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-preference-failure-") as temp:
        root = Path(temp)
        first_args = {"key": "Voice Tone", "value": "warm", "category": "Communication"}
        runtime = _runtime(root, "set_preference", first_args)
        original_write_preferences = runtime.vault.write_preferences_with_evidence

        def fail_after_database_commit(
            _preferences: Any, *, store_identity: str, generation: int
        ) -> tuple[Path, str]:
            del store_identity, generation
            raise OSError("representative preference mirror failure")

        runtime.vault.write_preferences_with_evidence = fail_after_database_commit  # type: ignore[method-assign]
        failed = runtime.handle("preference mirror failure", request_token="preference-failure-one")
        runtime.vault.write_preferences_with_evidence = original_write_preferences  # type: ignore[method-assign]
        if failed.tool_results[0].metadata.get("failure_kind") != "auto_mutation_handler_failed":
            raise SystemExit(f"set_preference partial write did not become uncertain: {failed.tool_results}")
        memories = runtime.store.list_memories(limit=20)
        graph = _projection_graph_counts(runtime, kind="preference")
        if (
            len(runtime.store.list_preferences(status=None, limit=20)) != 1
            or len(memories) != 1
            or graph != {"links": 1, "pending": 1, "completed": 0}
        ):
            raise SystemExit(
                "set_preference mirror failure did not preserve its atomic source graph: "
                f"memories={memories} graph={graph}"
            )
        repaired = reconcile_pending_preference_projections(runtime.store, runtime.vault)
        if repaired.pending != 0 or _projection_graph_counts(
            runtime, kind="preference"
        ) != {"links": 1, "pending": 0, "completed": 1}:
            raise SystemExit(f"set_preference pending projection did not reconcile: {repaired}")
        changed_args = {"key": " voice tone ", "value": "direct", "category": "communication"}
        _set_action = PlannedAction("set_preference", changed_args, "same logical preference target")
        runtime.planner = StaticPlanner(_set_action)
        blocked = runtime.handle("preference target retry", request_token="preference-failure-two")
        _assert_replay(blocked, "auto_mutation_unresolved_action", label="set_preference target fence")
        rows = runtime.store.list_preferences(status=None, limit=20)
        if (
            len(rows) != 1
            or rows[0]["value"] != "warm"
            or len(runtime.store.list_memories(limit=20)) != 1
            or _projection_graph_counts(runtime, kind="preference")["links"] != 1
        ):
            raise SystemExit(f"case/value variant bypassed uncertain preference target fence: {rows}")

    with TemporaryDirectory(prefix="jarvis-local-rollout-preference-preflight-") as temp:
        runtime = _runtime(Path(temp), "set_preference", {"key": "", "value": "warm"})
        refused = runtime.handle("invalid preference", request_token="preference-preflight")
        item = refused.tool_results[0]
        if item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected":
            raise SystemExit(f"set_preference semantic refusal crossed receipt claim: {item}")
        if _receipt_rows(runtime):
            raise SystemExit("set_preference semantic preflight refusal created a receipt")


def test_add_person_retry_safety_preflight_and_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-person-repeat-") as temp:
        root = Path(temp)
        args = {
            "name": "Retry Safe Person",
            "relation": "collaborator",
            "notes": "RAW-PERSON-NOTE-LOCAL-ROLLOUT-THETA",
        }
        runtime = _runtime(root, "add_person", args)
        first = runtime.handle("add person first", request_token="person-request-one")
        replay = runtime.handle("add person replay", request_token="person-request-one")
        if not first.tool_results[0].ok:
            raise SystemExit(f"add_person first execution failed: {first.tool_results}")
        _assert_replay(replay, "auto_mutation_completed_replay", label="add_person same token")
        repeated = runtime.handle("add person repeat", request_token="person-request-two")
        if not repeated.tool_results[0].ok:
            raise SystemExit(f"add_person new-token repeat failed: {repeated.tool_results}")
        rows = runtime.store.list_people(limit=20)
        if len(rows) != 1 or rows[0]["notes"] != args["notes"]:
            raise SystemExit(f"add_person repeat duplicated its row or trailing note: {rows}")
        memories = runtime.store.list_memories(limit=20)
        with runtime.store.connect() as conn:
            person_memory_links = int(
                conn.execute("SELECT COUNT(*) FROM person_memory_links").fetchone()[0]
            )
        if len(memories) != 1 or person_memory_links != 1 or int(memories[0]["revision"]) != 2:
            raise SystemExit(
                "add_person repeat did not converge on one revisioned profile memory: "
                f"{memories} / links={person_memory_links}"
            )
        person_notes = list((runtime.vault.root_path / "People").glob("*.md"))
        if len(person_notes) != 1 or person_notes[0].read_text(encoding="utf-8").count(args["notes"]) != 1:
            raise SystemExit(f"add_person repeat did not leave one mirrored note copy: {person_notes}")
        _assert_completed_receipts_link_ordinary_audits(
            runtime, expected=2, label="add_person intentional repeat"
        )
        _assert_receipt_privacy(
            runtime,
            request_tokens=["person-request-one", "person-request-two"],
            actions=[("add_person", args)],
            raw_content=[args["name"], args["relation"], args["notes"]],
            exposed_metadata=[
                first.tool_results[0].metadata,
                replay.tool_results[0].metadata,
                repeated.tool_results[0].metadata,
            ],
            label="add_person",
        )
        _assert_no_absolute_path_metadata(
            [
                first.tool_results[0].metadata,
                replay.tool_results[0].metadata,
                repeated.tool_results[0].metadata,
            ],
            root=root,
            label="add_person",
        )

    with TemporaryDirectory(prefix="jarvis-local-rollout-person-failure-") as temp:
        initial_args = {
            "name": "Case Retry Person",
            "relation": "friend",
            "notes": "one durable note",
        }
        runtime = _runtime(Path(temp), "add_person", initial_args)
        original_record = runtime.store.record_person_with_projections
        original_write_person = runtime.vault.write_person_with_evidence
        handler_calls = 0

        def counted_record(record: Any) -> Any:
            nonlocal handler_calls
            handler_calls += 1
            return original_record(record)

        def fail_after_database_commit(_person: Any, _interactions: Any, **_kwargs: Any) -> Path:
            raise OSError("representative person mirror failure")

        runtime.store.record_person_with_projections = counted_record  # type: ignore[method-assign]
        runtime.vault.write_person_with_evidence = fail_after_database_commit  # type: ignore[method-assign]
        failed = runtime.handle("person mirror failure", request_token="person-failure-owner")
        runtime.store.record_person_with_projections = original_record  # type: ignore[method-assign]
        runtime.vault.write_person_with_evidence = original_write_person  # type: ignore[method-assign]
        if failed.tool_results[0].metadata.get("failure_kind") != "auto_mutation_handler_failed":
            raise SystemExit(f"add_person partial write did not become uncertain: {failed.tool_results}")

        runtime.planner = StaticPlanner(
            PlannedAction(
                "add_person",
                {**initial_args, "name": "  case retry person  "},
                "same-token equivalent person identity",
            )
        )
        same = runtime.handle("same-token person retry", request_token="person-failure-owner")
        _assert_replay(same, "auto_mutation_request_collision", label="add_person same-token target fence")
        runtime.planner = StaticPlanner(
            PlannedAction(
                "add_person",
                {**initial_args, "name": "CASE RETRY PERSON"},
                "new-token equivalent person identity",
            )
        )
        cross = runtime.handle("new-token person retry", request_token="person-failure-other")
        _assert_replay(cross, "auto_mutation_unresolved_action", label="add_person new-token target fence")
        rows = runtime.store.list_people(limit=20)
        if handler_calls != 1 or len(rows) != 1 or rows[0]["notes"] != initial_args["notes"]:
            raise SystemExit(f"add_person uncertain retry reran or duplicated state: {handler_calls} / {rows}")
        if runtime.store.count_pending_person_projection_jobs() != 1:
            raise SystemExit("add_person mirror failure did not retain exactly one repairable person projection")
        memories = runtime.store.list_memories(limit=20)
        if len(memories) != 1:
            raise SystemExit("add_person mirror failure lost its atomically committed profile memory")
        memory_job = runtime.store.get_memory_projection_job(int(memories[0]["id"]))
        if memory_job is None or memory_job["state"] != "completed":
            raise SystemExit(f"add_person mirror failure did not independently publish memory: {memory_job}")
        _assert_receipt_privacy(
            runtime,
            request_tokens=["person-failure-owner", "person-failure-other"],
            actions=[
                ("add_person", initial_args),
                ("add_person", {**initial_args, "name": "  case retry person  "}),
                ("add_person", {**initial_args, "name": "CASE RETRY PERSON"}),
            ],
            raw_content=[initial_args["notes"]],
            exposed_metadata=[
                failed.tool_results[0].metadata,
                same.tool_results[0].metadata,
                cross.tool_results[0].metadata,
            ],
            label="add_person uncertain failure",
        )

    with TemporaryDirectory(prefix="jarvis-local-rollout-person-process-death-") as temp:
        root = Path(temp)
        args = {
            "name": "Private Atomic Person",
            "relation": "collaborator",
            "notes": "RAW-ADD-PERSON-PROCESS-DEATH-NOTE",
        }
        owner_token = "add-person-process-death-owner-token"
        cross_token = "add-person-process-death-cross-token"
        runtime = _runtime(root, "add_person", args)
        original_record = runtime.store.record_person_with_projections

        def commit_person_then_die(record: Any) -> Any:
            original_record(record)
            raise InjectedProcessDeath("injected after atomic add_person commit")

        runtime.store.record_person_with_projections = commit_person_then_die  # type: ignore[method-assign]
        try:
            runtime.handle("add person across process death", request_token=owner_token)
        except InjectedProcessDeath:
            pass
        else:
            raise SystemExit("add_person process death did not escape JarvisRuntime.handle")
        finally:
            runtime.store.record_person_with_projections = original_record  # type: ignore[method-assign]

        with runtime.store.connect() as conn:
            counts = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "people",
                    "memories",
                    "person_memory_links",
                    "person_projection_jobs",
                    "memory_projection_jobs",
                )
            }
        if counts != {
            "people": 1,
            "memories": 1,
            "person_memory_links": 1,
            "person_projection_jobs": 1,
            "memory_projection_jobs": 1,
        }:
            raise SystemExit(f"add_person process death lost atomic custody: {counts}")
        receipts = _receipt_rows(runtime)
        if len(receipts) != 1 or receipts[0]["state"] != "running":
            raise SystemExit(f"add_person process death missed running receipt: {receipts}")
        if (
            runtime.store.count_pending_person_projection_jobs() != 1
            or runtime.store.count_pending_memory_projection_jobs() != 1
        ):
            raise SystemExit("add_person process death did not retain both pending projection jobs")

        restarted = _runtime(root, "add_person", args)
        if (
            restarted.person_projection_recovery_status
            != {"status": "completed", "attempted": 1, "completed": 1, "pending": 0}
            or restarted.memory_projection_recovery_status
            != {"status": "completed", "attempted": 1, "completed": 1, "pending": 0}
        ):
            raise SystemExit(
                "add_person startup projection recovery diverged: "
                f"{restarted.person_projection_recovery_status} / "
                f"{restarted.memory_projection_recovery_status}"
            )
        same = restarted.handle("same add_person process-death replay", request_token=owner_token)
        _assert_replay(same, "auto_mutation_running_replay", label="add_person process-death owner replay")
        cross = restarted.handle("fresh add_person process-death replay", request_token=cross_token)
        _assert_replay(cross, "auto_mutation_unresolved_action", label="add_person process-death cross replay")
        if len(restarted.store.list_people(limit=20)) != 1 or len(restarted.store.list_memories(limit=20)) != 1:
            raise SystemExit("add_person process-death replay duplicated committed business state")
        _assert_receipt_privacy(
            restarted,
            request_tokens=[owner_token, cross_token],
            actions=[("add_person", args)],
            raw_content=[args["name"], args["relation"], args["notes"]],
            exposed_metadata=[same.tool_results[0].metadata, cross.tool_results[0].metadata],
            label="add_person process death",
        )

    for index, name in enumerate(("", "   ", "/\x55sers/example/private/person")):
        with TemporaryDirectory(prefix="jarvis-local-rollout-person-preflight-") as temp:
            runtime = _runtime(Path(temp), "add_person", {"name": name})
            refused = runtime.handle("invalid person", request_token=f"person-preflight-{index}")
            item = refused.tool_results[0]
            if item.metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected":
                raise SystemExit(f"add_person semantic preflight drifted: {item}")
            if _receipt_rows(runtime) or runtime.store.list_people(limit=20):
                raise SystemExit("add_person semantic preflight crossed the receipt/write boundary")
            exposed = json.dumps(
                {"response": refused.response, "metadata": item.metadata},
                ensure_ascii=False,
                default=str,
            )
            if "/\x55sers/example/private" in exposed:
                raise SystemExit("add_person semantic preflight exposed a local path")


def test_log_interaction_replay_failure_and_privacy_matrix() -> None:
    privacy_failures: list[str] = []

    def record_privacy(
        runtime: JarvisRuntime,
        *,
        root: Path,
        request_tokens: list[str],
        actions: list[tuple[str, dict[str, Any]]],
        raw_values: list[str],
        metadata_items: list[dict[str, Any]],
        label: str,
    ) -> None:
        try:
            _assert_receipt_privacy(
                runtime,
                request_tokens=request_tokens,
                actions=actions,
                raw_content=raw_values,
                exposed_metadata=metadata_items,
                label=label,
            )
        except SystemExit as exc:
            privacy_failures.append(str(exc))
        try:
            _assert_no_absolute_path_metadata(metadata_items, root=root, label=label)
        except SystemExit as exc:
            privacy_failures.append(str(exc))
        receipt_text = json.dumps(
            _receipt_rows(runtime), sort_keys=True, ensure_ascii=False, default=str
        )
        metadata_text = json.dumps(
            metadata_items, sort_keys=True, ensure_ascii=False, default=str
        )
        for raw_value in [*request_tokens, *raw_values]:
            if raw_value and raw_value in receipt_text:
                privacy_failures.append(
                    f"{label} receipt row leaked raw interaction content: {raw_value!r}"
                )
            if raw_value and raw_value in metadata_text:
                privacy_failures.append(
                    f"{label} outward metadata leaked raw interaction content: {raw_value!r}"
                )

    def assert_semantic_refusal(result: Any, reason: str, *, label: str) -> None:
        if len(result.tool_results) != 1:
            raise SystemExit(f"{label} should return one refusal: {result.tool_results}")
        item = result.tool_results[0]
        if (
            item.ok
            or item.metadata.get("failure_kind")
            != "auto_mutation_semantic_preflight_rejected"
            or item.metadata.get("reason") != reason
            or item.metadata.get("handler_invoked") is not False
        ):
            raise SystemExit(f"{label} crossed semantic preflight: {item}")

    with TemporaryDirectory(prefix="jarvis-local-rollout-interaction-success-") as temp:
        root = Path(temp)
        name = "Existing Interaction Privacy Person"
        summary = "RAW-INTERACTION-SUMMARY-NORMAL-ALPHA"
        happened_at = "2026-07-12T09:15:00+09:00"
        owner_token = "interaction-normal-owner-token"
        repeat_token = "interaction-normal-repeat-token"
        args = {"name": name, "summary": summary, "happened_at": happened_at}
        runtime = _runtime(root, "log_interaction", args)
        person_id = runtime.store.upsert_person(PersonRecord(name=name, relation="friend"))
        contract = runtime.registry.get("log_interaction").auto_mutation_contract
        if contract is None or contract.operation_key_builder is None:
            raise SystemExit("log_interaction is missing its operation-key builder")
        name_key = contract.operation_key_builder(args)
        id_args = {
            "person_id": person_id,
            "summary": summary,
            "happened_at": happened_at,
        }
        id_key = contract.operation_key_builder(id_args)
        string_id_args = {
            "person_id": str(person_id),
            "summary": summary,
            "happened_at": happened_at,
        }
        string_id_key = contract.operation_key_builder(string_id_args)
        if name_key != id_key or id_key != string_id_key:
            raise SystemExit(
                "existing-person name, integer ID, and numeric-string ID operation keys "
                f"diverged: {name_key} / {id_key} / {string_id_key}"
            )

        first = runtime.handle("log interaction by name", request_token=owner_token)
        if not first.tool_results[0].ok:
            raise SystemExit(f"log_interaction name-target first execution failed: {first.tool_results}")
        replay = runtime.handle("log interaction same-token replay", request_token=owner_token)
        _assert_replay(replay, "auto_mutation_completed_replay", label="log_interaction same token")
        if len(runtime.store.list_person_interactions(person_id, limit=20)) != 1:
            raise SystemExit("log_interaction same-token replay created another interaction")

        runtime.planner = StaticPlanner(
            PlannedAction(
                "log_interaction",
                string_id_args,
                "intentional repeat through canonical numeric-string person id",
            )
        )
        repeated = runtime.handle(
            "log interaction numeric-string intentional repeat", request_token=repeat_token
        )
        if not repeated.tool_results[0].ok:
            raise SystemExit(f"log_interaction fresh-token repeat failed: {repeated.tool_results}")
        interactions = runtime.store.list_person_interactions(person_id, limit=20)
        memories = runtime.store.list_memories(limit=20)
        if len(interactions) != 2 or any(row["summary"] != summary for row in interactions):
            raise SystemExit(f"log_interaction fresh token did not add exactly one interaction: {interactions}")
        if len(memories) != 2 or any(row["body"] != summary for row in memories):
            raise SystemExit(f"log_interaction fresh token did not add exactly one memory: {memories}")
        if runtime.store.count_pending_memory_projection_jobs() != 0:
            raise SystemExit("successful log_interaction left a pending memory projection")
        listed = runtime.registry.get("list_people").handler({"limit": 20})
        inspected = runtime.registry.get("get_person").handler(
            {"person_id": person_id, "limit": 20}
        )
        if not listed.ok or name not in listed.output or not inspected.ok or name not in inspected.output:
            raise SystemExit("people read tools lost useful user-facing person names")
        _assert_completed_receipts_link_ordinary_audits(
            runtime, expected=2, label="log_interaction intentional repeat"
        )
        record_privacy(
            runtime,
            root=root,
            request_tokens=[owner_token, repeat_token],
            actions=[
                ("log_interaction", args),
                ("log_interaction", id_args),
                ("log_interaction", string_id_args),
            ],
            raw_values=[name, summary, happened_at],
            metadata_items=[
                first.tool_results[0].metadata,
                replay.tool_results[0].metadata,
                repeated.tool_results[0].metadata,
                listed.metadata,
                inspected.metadata,
            ],
            label="log_interaction normal replay",
        )

    with TemporaryDirectory(prefix="jarvis-local-rollout-interaction-create-") as temp:
        root = Path(temp)
        name = "Created Interaction Identity Person"
        summary = "RAW-INTERACTION-SUMMARY-CREATED-BETA"
        happened_at = "2026-07-12T10:30:00+09:00"
        token = "interaction-created-person-token"
        args = {"name": name, "summary": summary, "happened_at": happened_at}
        runtime = _runtime(root, "log_interaction", args)
        contract = runtime.registry.get("log_interaction").auto_mutation_contract
        if contract is None or contract.operation_key_builder is None:
            raise SystemExit("log_interaction created-person case lost its operation-key builder")
        name_key = contract.operation_key_builder(args)
        first = runtime.handle("create person through interaction", request_token=token)
        if not first.tool_results[0].ok:
            raise SystemExit(f"name-created log_interaction failed: {first.tool_results}")
        people = runtime.store.list_people(limit=20)
        if len(people) != 1:
            raise SystemExit(f"name-created log_interaction did not create one person: {people}")
        person_id = int(people[0]["id"])
        id_args = {
            "person_id": person_id,
            "summary": summary,
            "happened_at": happened_at,
        }
        id_key = contract.operation_key_builder(id_args)
        if name_key != id_key:
            raise SystemExit(
                f"name-created person did not converge to its person_id operation key: {name_key} / {id_key}"
            )
        if len(runtime.store.list_person_interactions(person_id, limit=20)) != 1:
            raise SystemExit("name-created log_interaction did not preserve one interaction")
        record_privacy(
            runtime,
            root=root,
            request_tokens=[token],
            actions=[("log_interaction", args), ("log_interaction", id_args)],
            raw_values=[name, summary, happened_at],
            metadata_items=[first.tool_results[0].metadata],
            label="log_interaction name-created identity",
        )

    with TemporaryDirectory(prefix="jarvis-local-rollout-interaction-preflight-") as temp:
        root = Path(temp)
        first_name = "Interaction Preflight First Person"
        other_name = "Interaction Preflight Other Person"
        runtime = _runtime(
            root,
            "log_interaction",
            {
                "person_id": 999999,
                "summary": "RAW-INTERACTION-MISSING-ID-GAMMA",
                "happened_at": "2026-07-12T11:00:00+09:00",
            },
        )
        first_id = runtime.store.upsert_person(PersonRecord(name=first_name))
        runtime.store.upsert_person(PersonRecord(name=other_name))
        missing_args = {
            "person_id": 999999,
            "summary": "RAW-INTERACTION-MISSING-ID-GAMMA",
            "happened_at": "2026-07-12T11:00:00+09:00",
        }
        missing_token = "interaction-missing-id-token"
        missing = runtime.handle("missing person id", request_token=missing_token)
        assert_semantic_refusal(missing, "missing_person_id", label="log_interaction missing id")
        if _receipt_rows(runtime) or runtime.store.list_person_interactions(first_id, limit=20):
            raise SystemExit("missing person_id semantic refusal created state or a receipt")

        corrected_missing_args = {
            "person_id": first_id,
            "summary": missing_args["summary"],
            "happened_at": missing_args["happened_at"],
        }
        runtime.planner = StaticPlanner(
            PlannedAction("log_interaction", corrected_missing_args, "correct missing person id")
        )
        corrected_missing = runtime.handle(
            "correct missing person id with same token", request_token=missing_token
        )
        if not corrected_missing.tool_results[0].ok:
            raise SystemExit(
                f"corrected missing person_id did not run with the same token: {corrected_missing.tool_results}"
            )

        mismatch_args = {
            "person_id": first_id,
            "name": other_name,
            "summary": "RAW-INTERACTION-MISMATCH-DELTA",
            "happened_at": "2026-07-12T11:30:00+09:00",
        }
        mismatch_token = "interaction-mismatch-token"
        runtime.planner = StaticPlanner(
            PlannedAction("log_interaction", mismatch_args, "mismatched person targets")
        )
        before_receipts = len(_receipt_rows(runtime))
        before_interactions = len(runtime.store.list_person_interactions(first_id, limit=20))
        mismatch = runtime.handle("mismatched person targets", request_token=mismatch_token)
        assert_semantic_refusal(
            mismatch, "person_target_mismatch", label="log_interaction target mismatch"
        )
        if (
            len(_receipt_rows(runtime)) != before_receipts
            or len(runtime.store.list_person_interactions(first_id, limit=20)) != before_interactions
        ):
            raise SystemExit("mismatched name/person_id semantic refusal created state or a receipt")

        corrected_mismatch_args = {
            "person_id": first_id,
            "name": first_name,
            "summary": mismatch_args["summary"],
            "happened_at": mismatch_args["happened_at"],
        }
        runtime.planner = StaticPlanner(
            PlannedAction("log_interaction", corrected_mismatch_args, "correct person targets")
        )
        corrected_mismatch = runtime.handle(
            "correct mismatch with same token", request_token=mismatch_token
        )
        if not corrected_mismatch.tool_results[0].ok:
            raise SystemExit(
                f"corrected target mismatch did not run with the same token: {corrected_mismatch.tool_results}"
            )
        if len(_receipt_rows(runtime)) != 2:
            raise SystemExit("corrected semantic refusals did not create exactly two receipts")
        if len(runtime.store.list_person_interactions(first_id, limit=20)) != 2:
            raise SystemExit("corrected semantic refusals did not create exactly two interactions")

        invalid_results: list[Any] = []
        for index, invalid_id in enumerate((1.5, True)):
            invalid_args = {
                "person_id": invalid_id,
                "summary": f"RAW-INTERACTION-INVALID-ID-{index}",
                "happened_at": f"2026-07-12T11:4{index}:00+09:00",
            }
            runtime.planner = StaticPlanner(
                PlannedAction("log_interaction", invalid_args, "reject non-integral person id")
            )
            before_receipts = len(_receipt_rows(runtime))
            before_interactions = len(runtime.store.list_person_interactions(first_id, limit=20))
            invalid = runtime.handle(
                "reject non-integral person id",
                request_token=f"interaction-invalid-id-token-{index}",
            )
            invalid_item = invalid.tool_results[0]
            if (
                invalid_item.ok
                or invalid_item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or invalid_item.metadata.get("handler_invoked") is not False
                or "person_id" not in invalid_item.metadata.get("type_mismatch_arg_keys", [])
            ):
                raise SystemExit(f"non-integral person_id crossed the typed boundary: {invalid_item}")
            if (
                len(_receipt_rows(runtime)) != before_receipts
                or len(runtime.store.list_person_interactions(first_id, limit=20))
                != before_interactions
            ):
                raise SystemExit("non-integral person_id created a receipt or interaction")
            invalid_results.append(invalid)

        oversized_results: list[Any] = []
        for index, oversized_id in enumerate((2**63, str(2**63), "9" * 5000)):
            oversized_args = {
                "person_id": oversized_id,
                "summary": f"RAW-INTERACTION-OVERSIZED-ID-{index}",
                "happened_at": f"2026-07-12T11:5{index}:00+09:00",
            }
            runtime.planner = StaticPlanner(
                PlannedAction("log_interaction", oversized_args, "reject oversized person id")
            )
            before_receipts = len(_receipt_rows(runtime))
            before_interactions = len(runtime.store.list_person_interactions(first_id, limit=20))
            oversized = runtime.handle(
                "reject oversized person id",
                request_token=f"interaction-oversized-id-token-{index}",
            )
            assert_semantic_refusal(
                oversized, "bad_person_id", label="log_interaction oversized person id"
            )
            if (
                len(_receipt_rows(runtime)) != before_receipts
                or len(runtime.store.list_person_interactions(first_id, limit=20))
                != before_interactions
            ):
                raise SystemExit("oversized person_id created a receipt or interaction")
            metadata_text = json.dumps(
                oversized.tool_results[0].metadata,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            if str(oversized_id) in metadata_text:
                raise SystemExit("oversized person_id leaked into refusal metadata")
            oversized_results.append(oversized)
        record_privacy(
            runtime,
            root=root,
            request_tokens=[missing_token, mismatch_token],
            actions=[
                ("log_interaction", missing_args),
                ("log_interaction", corrected_missing_args),
                ("log_interaction", mismatch_args),
                ("log_interaction", corrected_mismatch_args),
                (
                    "log_interaction",
                    {
                        "person_id": 1.5,
                        "summary": "RAW-INTERACTION-INVALID-ID-0",
                        "happened_at": "2026-07-12T11:40:00+09:00",
                    },
                ),
                (
                    "log_interaction",
                    {
                        "person_id": True,
                        "summary": "RAW-INTERACTION-INVALID-ID-1",
                        "happened_at": "2026-07-12T11:41:00+09:00",
                    },
                ),
            ],
            raw_values=[
                first_name,
                other_name,
                str(missing_args["summary"]),
                str(missing_args["happened_at"]),
                str(mismatch_args["summary"]),
                str(mismatch_args["happened_at"]),
                "RAW-INTERACTION-INVALID-ID-0",
                "RAW-INTERACTION-INVALID-ID-1",
                "2026-07-12T11:40:00+09:00",
                "2026-07-12T11:41:00+09:00",
            ],
            metadata_items=[
                missing.tool_results[0].metadata,
                corrected_missing.tool_results[0].metadata,
                mismatch.tool_results[0].metadata,
                corrected_mismatch.tool_results[0].metadata,
                *[item.tool_results[0].metadata for item in invalid_results],
                *[item.tool_results[0].metadata for item in oversized_results],
            ],
            label="log_interaction semantic preflight",
        )

    with TemporaryDirectory(prefix="jarvis-local-rollout-interaction-person-failure-") as temp:
        root = Path(temp)
        name = "Interaction Person Mirror Failure"
        summary = "RAW-INTERACTION-PERSON-MIRROR-EPSILON"
        happened_at = "2026-07-12T12:00:00+09:00"
        owner_token = "interaction-person-failure-owner"
        cross_token = "interaction-person-failure-cross"
        args = {"name": name, "summary": summary, "happened_at": happened_at}
        runtime = _runtime(root, "log_interaction", args)
        person_id = runtime.store.upsert_person(PersonRecord(name=name))
        original_write_person = runtime.vault.write_person

        def fail_person_mirror(*_args: Any, **_kwargs: Any) -> Path:
            raise OSError("representative person mirror failure after interaction commit")

        runtime.vault.write_person_with_evidence = fail_person_mirror  # type: ignore[method-assign]
        try:
            failed = runtime.handle("person mirror failure", request_token=owner_token)
        finally:
            runtime.vault.write_person_with_evidence = original_write_person  # type: ignore[method-assign]
        failed_item = failed.tool_results[0]
        if failed_item.ok or failed_item.metadata.get("auto_mutation_outcome_uncertain") is not True:
            raise SystemExit(f"person mirror failure was not fenced as uncertain: {failed.tool_results}")
        if len(runtime.store.list_person_interactions(person_id, limit=20)) != 1:
            raise SystemExit("person mirror failure did not preserve exactly one committed interaction")
        memories = runtime.store.list_memories(limit=20)
        if len(memories) != 1 or memories[0]["body"] != summary:
            raise SystemExit(
                f"person mirror failure did not atomically preserve its linked memory: {memories}"
            )
        if runtime.store.count_pending_person_projection_jobs() != 1:
            raise SystemExit("person mirror failure did not retain one pending person projection")
        if runtime.store.count_pending_memory_projection_jobs() != 0:
            raise SystemExit("person mirror failure prevented independent memory-note completion")
        with runtime.store.connect() as conn:
            links = conn.execute("SELECT COUNT(*) FROM interaction_memory_links").fetchone()[0]
        if links != 1:
            raise SystemExit("person mirror failure missed atomic interaction-memory custody")

        same = runtime.handle("person mirror same-token retry", request_token=owner_token)
        _assert_replay(same, "auto_mutation_outcome_uncertain", label="log_interaction person failure same token")
        cross_args = {
            "person_id": person_id,
            "summary": summary,
            "happened_at": happened_at,
        }
        runtime.planner = StaticPlanner(
            PlannedAction("log_interaction", cross_args, "equivalent person_id retry")
        )
        cross = runtime.handle("person mirror equivalent cross-token retry", request_token=cross_token)
        _assert_replay(cross, "auto_mutation_unresolved_action", label="log_interaction person failure cross token")
        if (
            len(runtime.store.list_person_interactions(person_id, limit=20)) != 1
            or len(runtime.store.list_memories(limit=20)) != 1
            or runtime.store.count_pending_person_projection_jobs() != 1
        ):
            raise SystemExit("person mirror uncertain retry changed custodied interaction state")
        receipts = _receipt_rows(runtime)
        if len(receipts) != 1 or receipts[0]["state"] != "uncertain" or receipts[0]["result"] != "unknown":
            raise SystemExit(f"person mirror failure did not leave one uncertain receipt: {receipts}")
        record_privacy(
            runtime,
            root=root,
            request_tokens=[owner_token, cross_token],
            actions=[("log_interaction", args), ("log_interaction", cross_args)],
            raw_values=[name, summary, happened_at],
            metadata_items=[failed_item.metadata, same.tool_results[0].metadata, cross.tool_results[0].metadata],
            label="log_interaction person mirror failure",
        )

    with TemporaryDirectory(prefix="jarvis-local-rollout-interaction-memory-failure-") as temp:
        root = Path(temp)
        name = "Interaction Memory Projection Failure"
        summary = "RAW-INTERACTION-MEMORY-PROJECTION-ZETA"
        happened_at = "2026-07-12T12:30:00+09:00"
        owner_token = "interaction-memory-failure-owner"
        cross_token = "interaction-memory-failure-cross"
        args = {"name": name, "summary": summary, "happened_at": happened_at}
        runtime = _runtime(root, "log_interaction", args)
        person_id = runtime.store.upsert_person(PersonRecord(name=name))
        original_write_memory = runtime.vault.write_memory_projection_with_evidence

        def fail_memory_projection(*_args: Any, **_kwargs: Any) -> Path:
            raise OSError("representative memory projection failure after person note")

        runtime.vault.write_memory_projection_with_evidence = fail_memory_projection  # type: ignore[method-assign]
        try:
            failed = runtime.handle("memory projection failure", request_token=owner_token)
        finally:
            runtime.vault.write_memory_projection_with_evidence = original_write_memory  # type: ignore[method-assign]
        failed_item = failed.tool_results[0]
        if failed_item.ok or failed_item.metadata.get("auto_mutation_outcome_uncertain") is not True:
            raise SystemExit(f"memory projection failure was not fenced as uncertain: {failed.tool_results}")
        interactions = runtime.store.list_person_interactions(person_id, limit=20)
        memories = runtime.store.list_memories(limit=20)
        pending = runtime.store.list_pending_memory_projection_jobs(limit=20)
        if len(interactions) != 1 or interactions[0]["summary"] != summary:
            raise SystemExit(f"memory projection failure lost or duplicated its interaction: {interactions}")
        if len(memories) != 1 or memories[0]["body"] != summary or len(pending) != 1:
            raise SystemExit(
                f"memory projection failure did not preserve one memory and pending job: {memories} / {pending}"
            )

        same = runtime.handle("memory projection same-token retry", request_token=owner_token)
        _assert_replay(same, "auto_mutation_outcome_uncertain", label="log_interaction memory failure same token")
        cross_args = {
            "person_id": person_id,
            "summary": summary,
            "happened_at": happened_at,
        }
        runtime.planner = StaticPlanner(
            PlannedAction("log_interaction", cross_args, "equivalent pending-memory retry")
        )
        cross = runtime.handle("memory projection equivalent cross-token retry", request_token=cross_token)
        _assert_replay(cross, "auto_mutation_unresolved_action", label="log_interaction memory failure cross token")
        if (
            len(runtime.store.list_person_interactions(person_id, limit=20)) != 1
            or len(runtime.store.list_memories(limit=20)) != 1
            or runtime.store.count_pending_memory_projection_jobs() != 1
        ):
            raise SystemExit("pending memory projection retry repeated committed interaction state")
        receipts = _receipt_rows(runtime)
        if len(receipts) != 1 or receipts[0]["state"] != "uncertain" or receipts[0]["result"] != "unknown":
            raise SystemExit(f"memory projection failure did not leave one uncertain receipt: {receipts}")
        record_privacy(
            runtime,
            root=root,
            request_tokens=[owner_token, cross_token],
            actions=[("log_interaction", args), ("log_interaction", cross_args)],
            raw_values=[name, summary, happened_at],
            metadata_items=[failed_item.metadata, same.tool_results[0].metadata, cross.tool_results[0].metadata],
            label="log_interaction memory projection failure",
        )

    for created_person in (False, True):
        case_label = "name-created" if created_person else "existing-person"
        with TemporaryDirectory(
            prefix=f"jarvis-local-rollout-interaction-process-death-{case_label}-"
        ) as temp:
            root = Path(temp)
            name = f"Interaction Process Death {case_label} Private Person"
            summary = f"RAW-INTERACTION-PROCESS-DEATH-{case_label.upper()}"
            happened_at = (
                "2026-07-12T13:30:00+09:00"
                if created_person
                else "2026-07-12T13:00:00+09:00"
            )
            owner_token = f"interaction-process-death-{case_label}-owner-token"
            canonical_token = f"interaction-process-death-{case_label}-canonical-token"
            args = {"name": name, "summary": summary, "happened_at": happened_at}
            runtime = _runtime(root, "log_interaction", args)
            if not created_person:
                runtime.store.upsert_person(PersonRecord(name=name))

            original_record = runtime.store.record_person_interaction_with_projections

            def commit_then_die(**kwargs: Any) -> Any:
                original_record(**kwargs)
                raise InjectedProcessDeath("injected after atomic interaction commit")

            runtime.store.record_person_interaction_with_projections = commit_then_die  # type: ignore[method-assign]
            try:
                runtime.handle(
                    "log interaction across injected process death",
                    request_token=owner_token,
                )
            except InjectedProcessDeath:
                pass
            else:
                raise SystemExit(
                    f"log_interaction {case_label} process death did not escape JarvisRuntime.handle"
                )
            finally:
                runtime.store.record_person_interaction_with_projections = original_record  # type: ignore[method-assign]

            people = runtime.store.list_people(limit=20)
            if len(people) != 1:
                raise SystemExit(
                    f"log_interaction {case_label} process death did not preserve one person row"
                )
            person_id = int(people[0]["id"])

            def business_counts(active_runtime: JarvisRuntime) -> dict[str, int]:
                with active_runtime.store.connect() as conn:
                    return {
                        "people": int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]),
                        "interactions": int(
                            conn.execute("SELECT COUNT(*) FROM person_interactions").fetchone()[0]
                        ),
                        "memories": int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]),
                        "links": int(
                            conn.execute("SELECT COUNT(*) FROM interaction_memory_links").fetchone()[0]
                        ),
                        "person_jobs": int(
                            conn.execute("SELECT COUNT(*) FROM person_projection_jobs").fetchone()[0]
                        ),
                        "memory_jobs": int(
                            conn.execute("SELECT COUNT(*) FROM memory_projection_jobs").fetchone()[0]
                        ),
                        "successful_audits": int(
                            conn.execute("SELECT COUNT(*) FROM tool_runs WHERE ok = 1").fetchone()[0]
                        ),
                    }

            def raw_projection_jobs(active_runtime: JarvisRuntime) -> list[dict[str, Any]]:
                with active_runtime.store.connect() as conn:
                    return [
                        *[
                            {"kind": "person", **dict(row)}
                            for row in conn.execute(
                                "SELECT * FROM person_projection_jobs ORDER BY person_id"
                            )
                        ],
                        *[
                            {"kind": "memory", **dict(row)}
                            for row in conn.execute(
                                "SELECT * FROM memory_projection_jobs ORDER BY memory_id"
                            )
                        ],
                    ]

            def assert_private_process_death_metadata(
                active_runtime: JarvisRuntime,
                metadata_items: list[dict[str, Any]],
                *,
                phase: str,
            ) -> None:
                receipts = _receipt_rows(active_runtime)
                jobs = raw_projection_jobs(active_runtime)
                receipt_and_job_text = json.dumps(
                    {"receipts": receipts, "jobs": jobs},
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                outward_text = json.dumps(
                    metadata_items, sort_keys=True, ensure_ascii=False, default=str
                )
                private_values = [name, summary, happened_at, owner_token, canonical_token]
                for private_value in private_values:
                    if private_value in receipt_and_job_text:
                        privacy_failures.append(
                            f"log_interaction {case_label} {phase} receipt/job metadata leaked private content"
                        )
                    if private_value in outward_text:
                        privacy_failures.append(
                            f"log_interaction {case_label} {phase} outward metadata leaked private content"
                        )
                private_digests = [
                    str(row[key])
                    for row in [*receipts, *jobs]
                    for key in (
                        "request_digest",
                        "action_digest",
                        "operation_digest",
                        "uncertainty_digest",
                        "run_token",
                        "source_digest",
                        "content_digest",
                        "prior_content_digest",
                    )
                    if row.get(key)
                ]
                if any(value in outward_text for value in private_digests):
                    privacy_failures.append(
                        f"log_interaction {case_label} {phase} outward metadata leaked a private digest"
                    )
                for fragment in (str(root), "/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
                    if fragment in receipt_and_job_text or fragment in outward_text:
                        privacy_failures.append(
                            f"log_interaction {case_label} {phase} metadata leaked an absolute path"
                        )

            committed_counts = business_counts(runtime)
            expected_counts = {
                "people": 1,
                "interactions": 1,
                "memories": 1,
                "links": 1,
                "person_jobs": 1,
                "memory_jobs": 1,
                "successful_audits": 0,
            }
            if committed_counts != expected_counts:
                raise SystemExit(
                    f"log_interaction {case_label} process death custody counts diverged: "
                    f"{committed_counts}"
                )
            receipts = _receipt_rows(runtime)
            if (
                len(receipts) != 1
                or receipts[0]["state"] != "running"
                or receipts[0]["result"] != "pending"
                or receipts[0]["tool_run_id"] is not None
            ):
                raise SystemExit(
                    f"log_interaction {case_label} process death did not leave one running receipt"
                )
            if (
                runtime.store.count_pending_person_projection_jobs() != 1
                or runtime.store.count_pending_memory_projection_jobs() != 1
            ):
                raise SystemExit(
                    f"log_interaction {case_label} process death did not retain both pending jobs"
                )
            assert_private_process_death_metadata(runtime, [], phase="after death")

            restarted = _runtime(root, "log_interaction", args)
            if business_counts(restarted) != committed_counts:
                raise SystemExit(
                    f"log_interaction {case_label} startup recovery recreated business rows or jobs"
                )
            if (
                restarted.person_projection_recovery_status
                != {"status": "completed", "attempted": 1, "completed": 1, "pending": 0}
                or restarted.memory_projection_recovery_status
                != {"status": "completed", "attempted": 1, "completed": 1, "pending": 0}
                or restarted.store.count_pending_person_projection_jobs() != 0
                or restarted.store.count_pending_memory_projection_jobs() != 0
            ):
                raise SystemExit(
                    f"log_interaction {case_label} bounded startup projection recovery diverged: "
                    f"{restarted.person_projection_recovery_status} / "
                    f"{restarted.memory_projection_recovery_status}"
                )

            same_running = restarted.handle(
                "same request while process-death receipt is running",
                request_token=owner_token,
            )
            _assert_replay(
                same_running,
                "auto_mutation_running_replay",
                label=f"log_interaction {case_label} process death running owner replay",
            )
            canonical_args = {
                "person_id": person_id,
                "summary": summary,
                "happened_at": happened_at,
            }
            restarted.planner = StaticPlanner(
                PlannedAction(
                    "log_interaction",
                    canonical_args,
                    "canonical person id replay after process death",
                )
            )
            cross_running = restarted.handle(
                "canonical fresh request while process-death receipt is running",
                request_token=canonical_token,
            )
            _assert_replay(
                cross_running,
                "auto_mutation_unresolved_action",
                label=f"log_interaction {case_label} process death running canonical replay",
            )
            if business_counts(restarted) != committed_counts:
                raise SystemExit(
                    f"log_interaction {case_label} running replay fences changed custodied state"
                )
            assert_private_process_death_metadata(
                restarted,
                [
                    same_running.tool_results[0].metadata,
                    cross_running.tool_results[0].metadata,
                ],
                phase="running replays",
            )

            with restarted.store.connect() as conn:
                conn.execute(
                    "UPDATE auto_mutation_receipts "
                    "SET running_at = '2020-01-01T00:00:00Z' WHERE state = 'running'"
                )
            restarted.planner = StaticPlanner(
                PlannedAction("log_interaction", args, "owner replay after stale recovery")
            )
            same_uncertain = restarted.handle(
                "same request after process-death receipt becomes stale",
                request_token=owner_token,
            )
            _assert_replay(
                same_uncertain,
                "auto_mutation_outcome_uncertain",
                label=f"log_interaction {case_label} process death stale owner replay",
            )
            restarted.planner = StaticPlanner(
                PlannedAction(
                    "log_interaction",
                    canonical_args,
                    "canonical person id replay after stale recovery",
                )
            )
            cross_uncertain = restarted.handle(
                "canonical fresh request after process-death receipt becomes stale",
                request_token=canonical_token,
            )
            _assert_replay(
                cross_uncertain,
                "auto_mutation_unresolved_action",
                label=f"log_interaction {case_label} process death stale canonical replay",
            )
            final_receipts = _receipt_rows(restarted)
            if (
                len(final_receipts) != 1
                or final_receipts[0]["state"] != "uncertain"
                or final_receipts[0]["result"] != "unknown"
                or final_receipts[0]["resolution"] != "stale_recovery"
            ):
                raise SystemExit(
                    f"log_interaction {case_label} stale running receipt did not become uncertain"
                )
            if business_counts(restarted) != committed_counts:
                raise SystemExit(
                    f"log_interaction {case_label} uncertain replay fences changed custodied state"
                )
            assert_private_process_death_metadata(
                restarted,
                [
                    same_uncertain.tool_results[0].metadata,
                    cross_uncertain.tool_results[0].metadata,
                ],
                phase="stale uncertain replays",
            )

    if privacy_failures:
        raise SystemExit(
            "log_interaction privacy matrix found production metadata leaks:\n- "
            + "\n- ".join(privacy_failures)
        )


def test_save_skill_coalescing_upsert_audit_and_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-skill-") as temp:
        root = Path(temp)
        token_one = "skill-private-request-token-one"
        token_two = "skill-private-request-token-two"
        body = "RAW-SKILL-CONTENT-LOCAL-ROLLOUT-BETA"
        args = {
            "name": "Local Rollout Stable Skill",
            "trigger": "when testing local receipt rollout",
            "body": body,
            "tags": "smoke,local",
        }
        runtime = _runtime(root, "save_skill", args)

        first = runtime.handle("save skill first", request_token=token_one)
        if not first.tool_results[0].ok:
            raise SystemExit(f"save_skill first execution failed: {first.tool_results}")
        replay = runtime.handle("save skill replay", request_token=token_one)
        _assert_replay(replay, "auto_mutation_completed_replay", label="save_skill same token")
        if len(runtime.store.list_skills(limit=20)) != 1:
            raise SystemExit("save_skill same-token replay duplicated the skill")
        _assert_one_linked_ordinary_audit(runtime, label="save_skill same-token coalescing")

        repeated = runtime.handle("save skill intentional repeat", request_token=token_two)
        if not repeated.tool_results[0].ok:
            raise SystemExit(f"save_skill new-token repeat failed: {repeated.tool_results}")
        skills = runtime.store.list_skills(limit=20)
        if len(skills) != 1 or skills[0]["body"] != body:
            raise SystemExit(f"save_skill new token did not upsert one stable row: {skills}")
        skill_notes = list((runtime.vault.root_path / "Skills").glob("*.md"))
        if len(skill_notes) != 1 or body not in skill_notes[0].read_text(encoding="utf-8"):
            raise SystemExit(f"save_skill new token did not replace one stable note: {skill_notes}")
        _assert_completed_receipts_link_ordinary_audits(
            runtime, expected=2, label="save_skill intentional repeat"
        )
        _assert_receipt_privacy(
            runtime,
            request_tokens=[token_one, token_two],
            actions=[("save_skill", args)],
            raw_content=[body],
            exposed_metadata=[item.metadata for item in [first.tool_results[0], replay.tool_results[0], repeated.tool_results[0]]],
            label="save_skill",
        )


def test_memory_tree_snapshot_coalescing_replace_audit_and_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-tree-") as temp:
        root = Path(temp)
        token_one = "tree-private-request-token-one"
        token_two = "tree-private-request-token-two"
        first_body = "RAW-MEMORY-CONTENT-LOCAL-ROLLOUT-GAMMA"
        second_body = "RAW-MEMORY-CONTENT-LOCAL-ROLLOUT-DELTA"
        args = {"limit": 200}
        runtime = _runtime(root, "memory_tree_summary", args)
        runtime.store.add_memory(MemoryRecord("project", "First rollout memory", first_body, "smoke"))
        snapshot = runtime.vault.root_path / "Memory Tree" / "Memory Tree Snapshot.md"

        first = runtime.handle("write tree first", request_token=token_one)
        if not first.tool_results[0].ok or first_body not in snapshot.read_text(encoding="utf-8"):
            raise SystemExit(f"memory_tree_summary first snapshot failed: {first.tool_results}")
        runtime.store.add_memory(MemoryRecord("project", "Second rollout memory", second_body, "smoke"))
        replay = runtime.handle("write tree replay", request_token=token_one)
        _assert_replay(replay, "auto_mutation_completed_replay", label="memory_tree_summary same token")
        if second_body in snapshot.read_text(encoding="utf-8"):
            raise SystemExit("memory_tree_summary same-token replay rewrote the snapshot")
        _assert_one_linked_ordinary_audit(runtime, label="memory_tree_summary same-token coalescing")

        repeated = runtime.handle("write tree intentional repeat", request_token=token_two)
        snapshot_text = snapshot.read_text(encoding="utf-8")
        if not repeated.tool_results[0].ok or first_body not in snapshot_text or second_body not in snapshot_text:
            raise SystemExit(f"memory_tree_summary new token did not replace the fixed snapshot: {repeated.tool_results}")
        snapshots = list((runtime.vault.root_path / "Memory Tree").glob("Memory Tree Snapshot.md"))
        if snapshots != [snapshot]:
            raise SystemExit(f"memory_tree_summary created duplicate fixed snapshots: {snapshots}")
        _assert_completed_receipts_link_ordinary_audits(
            runtime, expected=2, label="memory_tree_summary intentional repeat"
        )
        _assert_receipt_privacy(
            runtime,
            request_tokens=[token_one, token_two],
            actions=[("memory_tree_summary", args)],
            raw_content=[first_body, second_body],
            exposed_metadata=[item.metadata for item in [first.tool_results[0], replay.tool_results[0], repeated.tool_results[0]]],
            label="memory_tree_summary",
        )


def test_record_feedback_vault_failure_stops_replay_as_uncertain() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-feedback-failure-") as temp:
        root = Path(temp)
        body = "RAW-FEEDBACK-CONTENT-POST-COMMIT-FAILURE"
        args = {"title": "Post commit failure", "theme": "safety", "body": body}
        runtime = _runtime(root, "record_feedback", args)
        original_write_memory = runtime.vault.write_memory_projection_with_evidence

        def fail_after_database_commit(*_args: Any, **_kwargs: Any) -> Path:
            raise OSError("representative vault write failure")

        runtime.vault.write_memory_projection_with_evidence = fail_after_database_commit  # type: ignore[method-assign]
        failed = runtime.handle("feedback vault failure", request_token="feedback-failure-owner")
        runtime.vault.write_memory_projection_with_evidence = original_write_memory  # type: ignore[method-assign]
        if failed.tool_results[0].metadata.get("failure_kind") != "auto_mutation_handler_failed":
            raise SystemExit(f"record_feedback vault failure did not surface as handler uncertainty: {failed.tool_results}")
        if len(runtime.store.list_memories(limit=20)) != 1:
            raise SystemExit("record_feedback vault failure did not preserve the already-committed memory")

        same = runtime.handle("feedback vault same-token replay", request_token="feedback-failure-owner")
        cross = runtime.handle("feedback vault cross-token replay", request_token="feedback-failure-other")
        _assert_replay(same, "auto_mutation_outcome_uncertain", label="record_feedback failure same token")
        _assert_replay(cross, "auto_mutation_unresolved_action", label="record_feedback failure cross token")
        if len(runtime.store.list_memories(limit=20)) != 1:
            raise SystemExit("record_feedback uncertain outcome was replayed automatically")
        rows = _receipt_rows(runtime)
        if len(rows) != 1 or rows[0]["state"] != "uncertain" or rows[0]["result"] != "unknown":
            raise SystemExit(f"record_feedback vault failure receipt was not uncertain: {rows}")
        if any(row["ok"] == 1 for row in _tool_run_rows(runtime)):
            raise SystemExit("record_feedback uncertain failure created an ordinary success audit")
        _assert_receipt_privacy(
            runtime,
            request_tokens=["feedback-failure-owner", "feedback-failure-other"],
            actions=[("record_feedback", args)],
            raw_content=[body],
            exposed_metadata=[
                failed.tool_results[0].metadata,
                same.tool_results[0].metadata,
                cross.tool_results[0].metadata,
            ],
            label="record_feedback uncertain failure",
        )


def test_remember_effect_equivalent_retry_is_fenced() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-remember-failure-") as temp:
        body = "Remember retries with   equivalent\n whitespace stay fenced."
        runtime = _runtime(Path(temp), "remember", {"body": body})
        contract = runtime.registry.get("remember").auto_mutation_contract
        if contract is None or contract.operation_key_builder is None:
            raise SystemExit("remember is missing its semantic operation-key builder")
        operation_key = contract.operation_key_builder(
            {
                "body": f"  {'x' * 1300}  ",
                "category": f"  {'c' * 90}  ",
                "title": f"  {'t' * 180}  ",
            }
        )
        if (
            len(operation_key["body"]) != 1200
            or len(operation_key["category"]) != 80
            or len(operation_key["title"]) != 160
            or not operation_key["body"].endswith("…")
            or not operation_key["category"].endswith("…")
            or not operation_key["title"].endswith("…")
        ):
            raise SystemExit(f"remember operation-key bounds diverged from the handler: {operation_key}")

        original_write_memory = runtime.vault.write_memory_projection_with_evidence

        def fail_after_database_commit(*_args: Any, **_kwargs: Any) -> Path:
            raise OSError("representative remember mirror failure")

        runtime.vault.write_memory_projection_with_evidence = fail_after_database_commit  # type: ignore[method-assign]
        failed = runtime.handle("remember mirror failure", request_token="remember-failure-owner")
        runtime.vault.write_memory_projection_with_evidence = original_write_memory  # type: ignore[method-assign]
        if failed.tool_results[0].metadata.get("failure_kind") != "auto_mutation_handler_failed":
            raise SystemExit(f"remember partial write did not become uncertain: {failed.tool_results}")

        runtime.planner = StaticPlanner(
            PlannedAction(
                "remember",
                {
                    "body": " Remember retries with equivalent whitespace stay fenced. ",
                    "category": " facts ",
                    "title": " Untitled ",
                },
                "effect-equivalent remember retry with explicit defaults",
            )
        )
        blocked = runtime.handle(
            "remember equivalent retry",
            request_token="remember-failure-other",
        )
        _assert_replay(blocked, "auto_mutation_unresolved_action", label="remember operation identity")
        rows = runtime.store.list_memories(limit=20)
        if (
            len(rows) != 1
            or rows[0]["body"] != "Remember retries with equivalent whitespace stay fenced."
            or rows[0]["category"] != "facts"
            or rows[0]["title"] != "Untitled"
        ):
            raise SystemExit(f"remember equivalent retry bypassed uncertain operation fencing: {rows}")


def test_record_decision_vault_failure_stops_replay_as_uncertain() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-decision-failure-") as temp:
        root = Path(temp)
        rationale = "RAW-DECISION-RATIONALE-POST-COMMIT-FAILURE"
        args = {
            "title": "Preserve uncertain decision outcome",
            "rationale": rationale,
            "impact": "Bound duplicate writes",
        }
        runtime = _runtime(root, "record_decision", args)
        original_write_decision = runtime.vault.write_decision_with_evidence

        def fail_after_database_commit(
            _decision: Any, *, store_identity: str
        ) -> tuple[Path, str]:
            del store_identity
            raise OSError("representative decision vault write failure")

        runtime.vault.write_decision_with_evidence = fail_after_database_commit  # type: ignore[method-assign]
        failed = runtime.handle("decision vault failure", request_token="decision-failure-owner")
        runtime.vault.write_decision_with_evidence = original_write_decision  # type: ignore[method-assign]
        if failed.tool_results[0].metadata.get("failure_kind") != "auto_mutation_handler_failed":
            raise SystemExit(f"record_decision vault failure did not surface as handler uncertainty: {failed.tool_results}")
        if len(runtime.store.list_decisions(status=None, limit=20)) != 1:
            raise SystemExit("record_decision vault failure did not preserve the already-committed decision")
        memories = runtime.store.list_memories(limit=20)
        graph = _projection_graph_counts(runtime, kind="decision")
        if len(memories) != 1 or graph != {"links": 1, "pending": 1, "completed": 0}:
            raise SystemExit(
                "record_decision vault failure did not preserve its atomic source graph: "
                f"memories={memories} graph={graph}"
            )

        repaired = reconcile_pending_decision_projections(runtime.store, runtime.vault)
        notes = list((runtime.vault.root_path / "Decisions").glob("*.md"))
        if (
            repaired.pending != 0
            or len(notes) != 1
            or rationale not in notes[0].read_text(encoding="utf-8")
            or _projection_graph_counts(runtime, kind="decision")
            != {"links": 1, "pending": 0, "completed": 1}
        ):
            raise SystemExit(f"record_decision pending projection did not reconcile: {repaired}")

        same = runtime.handle("decision vault same-token replay", request_token="decision-failure-owner")
        cross = runtime.handle("decision vault cross-token replay", request_token="decision-failure-other")
        _assert_replay(same, "auto_mutation_outcome_uncertain", label="record_decision failure same token")
        _assert_replay(cross, "auto_mutation_unresolved_action", label="record_decision failure cross token")
        if (
            len(runtime.store.list_decisions(status=None, limit=20)) != 1
            or len(runtime.store.list_memories(limit=20)) != 1
            or len(list((runtime.vault.root_path / "Decisions").glob("*.md"))) != 1
            or _projection_graph_counts(runtime, kind="decision")["links"] != 1
        ):
            raise SystemExit("record_decision uncertain outcome was replayed automatically")
        rows = _receipt_rows(runtime)
        if len(rows) != 1 or rows[0]["state"] != "uncertain" or rows[0]["result"] != "unknown":
            raise SystemExit(f"record_decision vault failure receipt was not uncertain: {rows}")
        if any(row["ok"] == 1 for row in _tool_run_rows(runtime)):
            raise SystemExit("record_decision uncertain failure created an ordinary success audit")
        _assert_receipt_privacy(
            runtime,
            request_tokens=["decision-failure-owner", "decision-failure-other"],
            actions=[("record_decision", args)],
            raw_content=[rationale],
            exposed_metadata=[
                failed.tool_results[0].metadata,
                same.tool_results[0].metadata,
                cross.tool_results[0].metadata,
            ],
            label="record_decision uncertain failure",
        )


def test_completion_failure_after_skill_write_stops_all_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-completion-failure-") as temp:
        root = Path(temp)
        body = "RAW-SKILL-CONTENT-COMPLETION-FAILURE"
        args = {
            "name": "Completion Failure Stable Skill",
            "trigger": "when audit completion fails",
            "body": body,
            "tags": "smoke",
        }
        runtime = _runtime(root, "save_skill", args)
        original_complete = runtime.store.complete_auto_mutation_receipt

        def fail_completion(**_kwargs: Any) -> int:
            raise RuntimeError("representative receipt completion failure")

        runtime.store.complete_auto_mutation_receipt = fail_completion  # type: ignore[method-assign]
        failed = runtime.handle("skill completion failure", request_token="skill-failure-owner")
        runtime.store.complete_auto_mutation_receipt = original_complete  # type: ignore[method-assign]
        _assert_replay(failed, "auto_mutation_completion_failed", label="save_skill completion failure")
        skills = runtime.store.list_skills(limit=20)
        notes = list((runtime.vault.root_path / "Skills").glob("*.md"))
        if len(skills) != 1 or skills[0]["body"] != body or len(notes) != 1:
            raise SystemExit(f"save_skill local write did not complete before audit failure: {skills} / {notes}")
        before_note = notes[0].read_text(encoding="utf-8")

        same = runtime.handle("skill completion same-token replay", request_token="skill-failure-owner")
        cross = runtime.handle("skill completion cross-token replay", request_token="skill-failure-other")
        _assert_replay(same, "auto_mutation_outcome_uncertain", label="save_skill failure same token")
        _assert_replay(cross, "auto_mutation_unresolved_action", label="save_skill failure cross token")
        if len(runtime.store.list_skills(limit=20)) != 1 or notes[0].read_text(encoding="utf-8") != before_note:
            raise SystemExit("save_skill uncertain completion was replayed automatically")
        rows = _receipt_rows(runtime)
        if len(rows) != 1 or rows[0]["state"] != "uncertain" or rows[0]["result"] != "unknown":
            raise SystemExit(f"save_skill completion failure receipt was not uncertain: {rows}")
        if any(row["ok"] == 1 for row in _tool_run_rows(runtime)):
            raise SystemExit("save_skill completion failure left a successful ordinary audit")
        _assert_receipt_privacy(
            runtime,
            request_tokens=["skill-failure-owner", "skill-failure-other"],
            actions=[("save_skill", args)],
            raw_content=[body],
            exposed_metadata=[
                failed.tool_results[0].metadata,
                same.tool_results[0].metadata,
                cross.tool_results[0].metadata,
            ],
            label="save_skill uncertain failure",
        )


def test_empty_memory_tree_is_successful_and_read_only() -> None:
    with TemporaryDirectory(prefix="jarvis-local-rollout-empty-tree-") as temp:
        root = Path(temp)
        runtime = _runtime(root, "memory_tree_summary", {"limit": 200})
        snapshot = runtime.vault.root_path / "Memory Tree" / "Memory Tree Snapshot.md"
        before = snapshot.read_text(encoding="utf-8") if snapshot.exists() else None
        result = runtime.handle("empty memory tree", request_token="empty-tree-request")
        item = result.tool_results[0]
        if not item.ok or "No memories to summarize." not in item.output:
            raise SystemExit(f"empty memory_tree_summary did not follow handler semantics: {item}")
        for key in ("writes_files", "writes_database", "writes_memory", "writes_notes"):
            if item.metadata.get(key) is not False:
                raise SystemExit(f"empty memory_tree_summary should expose {key}=False: {item.metadata}")
        after = snapshot.read_text(encoding="utf-8") if snapshot.exists() else None
        if after != before:
            raise SystemExit("empty memory_tree_summary modified the fixed snapshot")
        _assert_one_linked_ordinary_audit(runtime, label="empty memory_tree_summary")


def test_browser_history_writers_have_no_auto_mutation_contract() -> None:
    with TemporaryDirectory(prefix="jarvis-browser-no-auto-mutation-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for name in ("fetch_page", "extract_links", "web_search"):
            tool = runtime.registry.get(name)
            if tool.auto_mutation_contract is not None:
                raise SystemExit(f"{name} must not opt into the local auto-mutation receipt protocol: {tool.auto_mutation_contract}")


def main() -> None:
    original_suggest_command = runtime_module.suggest_command
    runtime_module.suggest_command = lambda _text: None
    try:
        test_record_feedback_coalescing_repeat_audit_and_privacy()
        test_record_decision_coalescing_repeat_audit_and_privacy()
        test_record_decision_typed_and_semantic_preflight()
        test_set_preference_target_fencing_and_replay()
        test_set_preference_semantic_preflight_and_uncertain_target_fence()
        test_add_person_retry_safety_preflight_and_privacy()
        test_log_interaction_replay_failure_and_privacy_matrix()
        test_save_skill_coalescing_upsert_audit_and_privacy()
        test_memory_tree_snapshot_coalescing_replace_audit_and_privacy()
        test_record_feedback_vault_failure_stops_replay_as_uncertain()
        test_remember_effect_equivalent_retry_is_fenced()
        test_record_decision_vault_failure_stops_replay_as_uncertain()
        test_completion_failure_after_skill_write_stops_all_replay()
        test_empty_memory_tree_is_successful_and_read_only()
        test_browser_history_writers_have_no_auto_mutation_contract()
    finally:
        runtime_module.suggest_command = original_suggest_command
    print("Auto mutation local rollout smoke passed")


if __name__ == "__main__":
    main()
