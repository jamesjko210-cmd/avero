from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.profile_projection import reconcile_profile_projection_candidate
from jarvis_v2.memory.store import (
    MemoryProfilePromotionResult,
    MemoryRecord,
    MemoryStore,
    profile_note_source_key,
)
from jarvis_v2.scripts.test_runtime import (
    approve_pending_runtime_approval,
    make_temp_runtime,
)


def _setup(root: Path) -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / "store.sqlite")
    store.init()
    vault = ObsidianVault(root / "vault")
    vault.init()
    return store, vault


def test_exact_profile_promotion_preserves_memory_identity_and_custody() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-promotion-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = store.add_memory(
            MemoryRecord(
                category="identity",
                title="Unreviewed identity note",
                body="This generic note must become an owned profile note only after review.",
                source="promotion-smoke",
                confidence=0.73,
            )
        )
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None or target.target_kind != "profile":
            raise SystemExit("profile candidate did not resolve for exact review")
        source_key = profile_note_source_key(
            "the operator", "Prefers a grounded personal assistant.", "identity"
        )
        result = store.promote_memory_to_profile_note_exact(
            memory_id,
            target.revision,
            target.binding,
            MemoryRecord(
                category="identity",
                title="the operator",
                body="Prefers a grounded personal assistant.",
                source="profile",
                confidence=1.0,
            ),
            source_key,
        )
        if (
            type(result) is not MemoryProfilePromotionResult
            or result.status != "promoted"
            or result.memory_id != memory_id
            or result.source_key != source_key
            or result.memory_projection_target is None
        ):
            raise SystemExit(f"profile promotion did not return exact custody evidence: {result!r}")
        with store.connect() as conn:
            memory = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            source = conn.execute(
                "SELECT * FROM ingested_sources WHERE source_key = ?", (source_key,)
            ).fetchone()
        if (
            memory is None
            or memory["id"] != memory_id
            or memory["category"] != "identity"
            or memory["title"] != "the operator"
            or memory["source"] != "profile"
            or float(memory["confidence"]) != 1.0
            or memory["revision"] != target.revision + 1
            or source is None
            or source["memory_id"] != memory_id
            or source["source_type"] != "profile_note"
            or source["memory_link_required"] != 1
            or source["mirror_state"] != "pending"
        ):
            raise SystemExit("profile promotion did not preserve one exact source-owned memory")

        pending = store.list_pending_profile_projection_sources(limit=1)
        if len(pending) != 1 or pending[0]["source_key"] != source_key:
            raise SystemExit("profile promotion did not create one recoverable pending projection")
        outcome = reconcile_profile_projection_candidate(store, vault, pending[0])
        if outcome.status != "completed":
            raise SystemExit(f"profile promotion custody did not converge: {outcome!r}")
        profile = store.read_profile_knowledge_snapshot()
        if len(profile.notes) != 1 or profile.notes[0].memory_id != memory_id:
            raise SystemExit("completed profile promotion was not visible through trusted profile custody")
        if store.resolve_knowledge_promotion_approval_target(memory_id) is not None:
            raise SystemExit("profile-owned memory remained promotion-eligible")

        duplicate = store.promote_memory_to_profile_note_exact(
            memory_id,
            target.revision,
            target.binding,
            MemoryRecord(
                category="identity",
                title="the operator",
                body="Prefers a grounded personal assistant.",
                source="profile",
                confidence=1.0,
            ),
            source_key,
        )
        if duplicate.status not in {"stale", "not_candidate"}:
            raise SystemExit(f"replayed profile promotion was not rejected: {duplicate!r}")


def test_profile_promotion_command_is_approval_bound_end_to_end() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-promotion-runtime-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = runtime.store.add_memory(
            MemoryRecord(
                category="identity",
                title="Reviewed personal context",
                body="This should only become profile-owned after its exact approval runs.",
                source="promotion-runtime-smoke",
                confidence=0.74,
            )
        )
        target = runtime.store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None or target.target_kind != "profile":
            raise SystemExit("runtime profile candidate did not resolve")
        packet = runtime.registry.get("knowledge_promotion_packet").handler(
            {"memory_id": memory_id}
        )
        if (
            not packet.ok
            or "to profile:" not in packet.output
            or target.binding not in packet.output
        ):
            raise SystemExit("profile candidate packet omitted its exact approval template")
        command = (
            f"promote memory {memory_id} revision {target.revision} token {target.binding} "
            "to profile: the operator | identity | Prefers grounded, approval-gated assistance."
        )
        plan = RuleBasedPlanner().plan(command)
        expected_raw = {
            "memory_id": memory_id,
            "reviewed_revision": target.revision,
            "review_token": target.binding,
            "heading": "the operator",
            "category": "identity",
            "body": "Prefers grounded, approval-gated assistance.",
        }
        if len(plan.actions) != 1 or plan.actions[0].tool_name != "promote_memory_to_profile":
            raise SystemExit(f"profile promotion command did not route exactly: {plan!r}")
        if plan.actions[0].args != expected_raw:
            raise SystemExit(f"profile promotion planner changed reviewed fields: {plan.actions[0].args!r}")
        placeholder_plan = RuleBasedPlanner().plan(
            f"promote memory {memory_id} revision {target.revision} token {target.binding} "
            "to profile: <explicit heading> | identity | reviewed body"
        )
        if any(
            action.tool_name == "promote_memory_to_profile"
            for action in placeholder_plan.actions
        ):
            raise SystemExit("placeholder profile fields reached mutation planning")
        tool = runtime.registry.get("promote_memory_to_profile")
        if (
            tool.risk is not RiskLevel.HIGH_RISK
            or tool.approval_argument_resolver is None
            or tool.approval_argument_contract is None
        ):
            raise SystemExit("profile promotion lost its approval-gated high-risk contract")
        held = runtime.handle(command)
        approvals = [
            result
            for result in held.tool_results
            if result.tool_name == "promote_memory_to_profile"
            and result.metadata.get("requires_confirmation") is True
        ]
        if len(approvals) != 1:
            raise SystemExit(f"profile promotion did not queue exactly one approval: {held!r}")
        approval_id = approvals[0].metadata.get("approval_id")
        if type(approval_id) is not int:
            raise SystemExit("profile promotion approval was missing its id")
        pending = runtime.store.get_pending_approval(approval_id)
        stored_args = json.loads(pending["planned_args"]) if pending is not None else None
        if (
            type(stored_args) is not dict
            or "review_token" in stored_args
            or stored_args.get("review_binding") != target.binding
            or stored_args.get("target_binding") != target.binding
        ):
            raise SystemExit("profile promotion did not persist only the bound review identity")
        transition = approve_pending_runtime_approval(runtime, approval_id)
        if transition.metadata.get("rerun_user_input") != command:
            raise SystemExit("profile promotion approval changed the reviewed command")
        completed = runtime.handle(command, approved=True, approved_approval_id=approval_id)
        results = [
            result
            for result in completed.tool_results
            if result.tool_name == "promote_memory_to_profile"
        ]
        if len(results) != 1 or not results[0].ok:
            raise SystemExit(f"approved profile promotion did not complete: {completed!r}")
        metadata = results[0].metadata
        if (
            metadata.get("promotion_committed") is not True
            or metadata.get("profile_integrity_verified") is not True
            or metadata.get("projection_pending") is not False
            or target.binding in str(metadata)
        ):
            raise SystemExit(f"profile promotion completion metadata was unsafe or incomplete: {metadata!r}")
        snapshot = runtime.store.read_profile_knowledge_snapshot()
        if (
            len(snapshot.notes) != 1
            or snapshot.notes[0].memory_id != memory_id
            or snapshot.notes[0].heading != "the operator"
        ):
            raise SystemExit("approved profile promotion was not visible through profile custody")


def main() -> None:
    test_exact_profile_promotion_preserves_memory_identity_and_custody()
    test_profile_promotion_command_is_approval_bound_end_to_end()
    print("Knowledge profile promotion smoke passed")


if __name__ == "__main__":
    main()
