from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.execution_outcome import classify_approved_execution_outcome
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    DecisionProjectionTarget,
    DecisionRecord,
    KnowledgePromotionCandidateTarget,
    MemoryApprovalTarget,
    MemoryDecisionPromotionResult,
    MemoryProjectionTarget,
    MemoryRecord,
    MemoryStore,
    memory_projection_source_digest,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.knowledge_promotion import make_knowledge_promotion_tools


GRAPH_TABLES = (
    "decisions",
    "decision_memory_links",
    "decision_projection_jobs",
    "memory_projection_jobs",
)


def _setup(root: Path) -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / "store.sqlite")
    store.init()
    vault = ObsidianVault(root / "vault")
    vault.init()
    return store, vault


def _seed(
    store: MemoryStore,
    title: str,
    body: str,
    *,
    source: str = "promotion-smoke-private-source",
) -> int:
    return store.add_memory(
        MemoryRecord(
            category="decisions",
            title=title,
            body=body,
            source=source,
            confidence=0.73,
        )
    )


def _row_tuple(row: sqlite3.Row | None) -> tuple[Any, ...] | None:
    return None if row is None else tuple(row)


def _memory(store: MemoryStore, memory_id: int) -> tuple[Any, ...] | None:
    with store.connect() as conn:
        return _row_tuple(
            conn.execute(
                "SELECT id, category, title, body, source, confidence, revision, "
                "created_at, updated_at FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        )


def _counts(store: MemoryStore, tables: Iterable[str] = GRAPH_TABLES) -> dict[str, int]:
    with store.connect() as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }


def _database_rows(store: MemoryStore) -> dict[str, list[tuple[Any, ...]]]:
    tables = ("memories",) + GRAPH_TABLES
    with store.connect() as conn:
        return {
            table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in tables
        }


def _vault_files(vault: ObsidianVault) -> dict[str, bytes]:
    return {
        str(path.relative_to(vault.root_path)): path.read_bytes()
        for path in sorted(vault.root_path.rglob("*"))
        if path.is_file()
    }


def _status(result: Any) -> str:
    if isinstance(result, str):
        return result
    value = getattr(result, "status", "")
    return value if isinstance(value, str) else ""


def _mutation(result: Any) -> Any | None:
    if (
        type(getattr(result, "decision_id", None)) is int
        and getattr(result, "decision_id") > 0
        and type(getattr(result, "memory_id", None)) is int
        and getattr(result, "memory_id") > 0
    ):
        return result
    for field in ("mutation", "target", "mutation_target", "decision_mutation"):
        value = getattr(result, field, None)
        if value is not None:
            return value
    return None


def _promote(
    store: MemoryStore,
    memory_id: int,
    target: Any,
    *,
    title: str = "Reviewed decision title",
    rationale: str = "Reviewed decision rationale",
    impact: str = "Reviewed decision impact",
) -> Any:
    return store.promote_memory_to_decision_exact(
        memory_id,
        target.revision,
        target.binding,
        DecisionRecord(title=title, rationale=rationale, impact=impact),
    )


def _assert_promoted(result: Any, label: str) -> Any:
    mutation = _mutation(result)
    if _status(result) not in {"promoted", "transferred"} or mutation is None:
        raise SystemExit(f"{label} did not return a successful exact transfer: {result!r}")
    return mutation


def _factory_handlers(
    store: MemoryStore, vault: ObsidianVault
) -> tuple[
    Callable[[dict[str, Any]], Any],
    Callable[[dict[str, Any]], Any],
    Callable[[dict[str, Any]], Any],
]:
    made = make_knowledge_promotion_tools(store, vault)
    if isinstance(made, dict):
        items = list(made.items())
    else:
        items = [("", item) for item in made]
    handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
    for supplied_name, item in items:
        handler = getattr(item, "handler", item)
        if not callable(handler):
            continue
        name = str(
            supplied_name
            or getattr(item, "name", "")
            or getattr(handler, "__name__", "")
        ).lower()
        if "packet" in name or "review" in name:
            handlers["packet"] = handler
        elif (
            name.startswith("resolve_")
            and ("promote" in name or "transfer" in name)
            and "decision" in name
        ):
            handlers["resolver"] = handler
        elif name in {"promote_memory_to_decision", "transfer_memory_to_decision"}:
            handlers["promote"] = handler
    if set(handlers) != {"packet", "promote", "resolver"}:
        raise SystemExit(
            "knowledge promotion factory must expose one packet reader, transfer handler, "
            "and approval resolver: "
            f"{sorted(handlers)}"
        )
    return handlers["packet"], handlers["promote"], handlers["resolver"]


def _tool_promote(
    handler: Callable[[dict[str, Any]], Any],
    memory_id: int,
    target: Any,
    *,
    title: str,
    rationale: str,
    impact: str,
):
    return handler(
        {
            "memory_id": memory_id,
            "reviewed_revision": target.revision,
            "review_binding": target.binding,
            "title": title,
            "rationale": rationale,
            "impact": impact,
            "target_revision": target.revision,
            "target_binding": target.binding,
        }
    )


def _assert_content_free_metadata(
    metadata: dict[str, Any],
    *,
    forbidden: Iterable[str],
    label: str,
) -> None:
    rendered = json.dumps(metadata, sort_keys=True, default=str)
    leaked = [marker for marker in forbidden if marker and marker in rendered]
    if leaked:
        raise SystemExit(f"{label} metadata leaked private or approval material: {leaked}")


def test_exact_candidate_binding_and_preapproval_read_only() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-binding-") as temp:
        store, vault = _setup(Path(temp))
        title = "Private promotion title marker"
        body = "Private promotion body marker /\x55sers/the operator/Secret/decision.txt"
        source = "private-promotion-source-marker"
        memory_id = _seed(store, title, body, source=source)
        before_database = _database_rows(store)
        before_vault = _vault_files(vault)

        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if (
            target is None
            or target.memory_id != memory_id
            or target.revision != 1
            or re.fullmatch(r"[0-9a-f]{64}", target.binding) is None
        ):
            raise SystemExit(f"exact decision candidate did not resolve an opaque binding: {target!r}")
        packet, _promote_handler, resolver = _factory_handlers(store, vault)
        original_get_memory = store.get_memory
        store.get_memory = lambda _memory_id: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("packet performed a second memory read")
        )
        try:
            result = packet({"memory_id": memory_id})
        finally:
            store.get_memory = original_get_memory  # type: ignore[method-assign]
        if (
            not result.ok
            or title not in result.output
            or body not in result.output
            or target.binding not in result.output
        ):
            raise SystemExit(f"promotion packet did not privately render the exact candidate: {result}")
        metadata = result.metadata
        if (
            metadata.get("reads_private_data") is not True
            or metadata.get("reads_personal_data") is not True
            or metadata.get("writes_database") is not False
            or metadata.get("writes_files") is not False
            or metadata.get("writes_memory") is not False
            or metadata.get("queues_approval") is not False
        ):
            raise SystemExit(f"promotion packet reported the wrong read-only boundaries: {metadata}")
        _assert_content_free_metadata(
            metadata,
            forbidden=(target.binding, str(vault.root_path), title, body, source, "decision.txt"),
            label="promotion packet",
        )
        explicit_fields = {
            "memory_id": memory_id,
            "reviewed_revision": target.revision,
            "review_token": target.binding,
            "title": "Explicit reviewed title",
            "rationale": "Explicit reviewed rationale",
            "impact": "Explicit reviewed impact",
        }
        resolution = resolver(explicit_fields)
        expected_bound = {
            **{key: value for key, value in explicit_fields.items() if key != "review_token"},
            "review_binding": target.binding,
            "target_revision": target.revision,
            "target_binding": target.binding,
        }
        if getattr(resolution, "args", None) != expected_bound:
            raise SystemExit(f"approval resolver did not bind the exact candidate: {resolution}")
        resolution_metadata = getattr(resolution, "metadata", None)
        if type(resolution_metadata) is not dict:
            raise SystemExit("approval resolver did not return bounded public metadata")
        _assert_content_free_metadata(
            resolution_metadata,
            forbidden=(
                target.binding,
                str(vault.root_path),
                title,
                body,
                source,
                explicit_fields["title"],
                explicit_fields["rationale"],
                explicit_fields["impact"],
            ),
            label="promotion approval resolver",
        )
        if _database_rows(store) != before_database or _vault_files(vault) != before_vault:
            raise SystemExit(
                "binding resolution, packet read, or approval resolution mutated durable "
                "state before approval"
            )

        for index, category in enumerate(
            ("\tdecisions\n", "\u00a0decisions\u00a0", "\u2003decisions\u2003"),
            start=1,
        ):
            whitespace_id = _seed(
                store,
                f"Whitespace category {index}",
                "category boundary whitespace should normalize",
            )
            with store.connect() as conn:
                conn.execute(
                    "UPDATE memories SET category = ? WHERE id = ?",
                    (category, whitespace_id),
                )
            whitespace_target = store.resolve_knowledge_promotion_approval_target(whitespace_id)
            if whitespace_target is None or whitespace_target.target_kind != "decision":
                raise SystemExit(
                    f"Unicode-whitespace category did not resolve consistently: {category!r}"
                )


def test_all_owned_candidate_classes_are_excluded() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-owned-") as temp:
        store, _vault = _setup(Path(temp))
        ids = {
            name: _seed(store, f"Owned {name}", f"owned {name} body")
            for name in (
                "ingested",
                "bootstrap",
                "organized",
                "decision",
                "preference",
                "person",
                "interaction",
                "provenance",
                "compaction",
                "compaction_source",
            )
        }
        ids["bootstrap_ollama_source"] = _seed(
            store,
            "Owned bootstrap Ollama source",
            "legacy source-only bootstrap body",
            source="jarvis-ollama-import",
        )
        ids["bootstrap_life_source"] = _seed(
            store,
            "Owned bootstrap life-model source",
            "legacy source-only bootstrap body",
            source="life-model-import",
        )
        ids["bootstrap_ollama_source_whitespace"] = _seed(
            store,
            "Owned whitespace bootstrap Ollama source",
            "legacy source-only bootstrap body",
            source="\tjarvis-ollama-import\n",
        )
        ids["bootstrap_life_source_nbsp"] = _seed(
            store,
            "Owned NBSP bootstrap life-model source",
            "legacy source-only bootstrap body",
            source="\u00a0life-model-import\u00a0",
        )
        ids["ingested_source_only"] = _seed(
            store,
            "Owned source-only Inbox memory",
            "legacy source-only ingest body",
            source="obsidian-inbox",
        )
        ids["profile_source_whitespace"] = _seed(
            store,
            "Owned whitespace profile source",
            "legacy source-only profile body",
            source="\tprofile\n",
        )
        now = "2026-07-13T00:00:00Z"
        organized_key = "organize-note:v1:" + "a" * 64
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO ingested_sources(source_key, source_type, title, memory_id, "
                "memory_link_required, mirror_state, mirror_completed_at, created_at) "
                "VALUES ('ingested-owned', 'markdown', 'owned', ?, 1, 'completed', ?, ?)",
                (ids["ingested"], now, now),
            )
            conn.execute(
                "INSERT INTO bootstrap_memory_occurrences(source_key, memory_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                ("bootstrap-memory:v1:" + "b" * 64, ids["bootstrap"], now, now),
            )
            conn.execute(
                "INSERT INTO organized_note_batches(source_key, state, created_at, updated_at, "
                "completed_at, projection_manifest) VALUES (?, 'completed', ?, ?, ?, '{}')",
                (organized_key, now, now, now),
            )
            conn.execute(
                "INSERT INTO organized_note_entries(source_key, entry_index, kind, memory_id) "
                "VALUES (?, 0, 'memory', ?)",
                (organized_key, ids["organized"]),
            )
            decision = conn.execute(
                "INSERT INTO decisions(title, rationale, impact, status, revision, created_at, updated_at) "
                "VALUES ('owned', '', '', 'active', 1, ?, ?)",
                (now, now),
            )
            conn.execute(
                "INSERT INTO decision_memory_links(decision_id, memory_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (int(decision.lastrowid), ids["decision"], now, now),
            )
            preference = conn.execute(
                "INSERT INTO preferences(category, key, value, status, revision, created_at, updated_at) "
                "VALUES ('general', 'owned', 'yes', 'active', 1, ?, ?)",
                (now, now),
            )
            conn.execute(
                "INSERT INTO preference_memory_links(preference_id, memory_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (int(preference.lastrowid), ids["preference"], now, now),
            )
            person = conn.execute(
                "INSERT INTO people(name, relation, notes, revision, created_at, updated_at) "
                "VALUES ('Owned Person', '', '', 1, ?, ?)",
                (now, now),
            )
            person_id = int(person.lastrowid)
            conn.execute(
                "INSERT INTO person_memory_links(memory_id, person_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (ids["person"], person_id, now, now),
            )
            interaction = conn.execute(
                "INSERT INTO person_interactions(person_id, summary, happened_at, created_at) "
                "VALUES (?, 'owned', ?, ?)",
                (person_id, now, now),
            )
            conn.execute(
                "INSERT INTO interaction_memory_links(interaction_id, memory_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (int(interaction.lastrowid), ids["interaction"], now, now),
            )
            epoch = conn.execute(
                "SELECT epoch.epoch_id, epoch.destination_class "
                "FROM active_history_policy_epoch active "
                "JOIN history_policy_epochs epoch ON epoch.epoch_id = active.active_epoch_id "
                "WHERE active.singleton_id = 1"
            ).fetchone()
            if epoch is None:
                raise SystemExit("provenance exclusion fixture lacks an active policy epoch")
            conn.execute(
                "INSERT INTO memory_provenance(memory_id, lineage_state, source_count, "
                "source_digest, content_digest, policy_epoch_id, destination_class, "
                "remote_eligible, created_at) VALUES (?, 'complete', 1, ?, ?, ?, ?, 0, ?)",
                (
                    ids["provenance"],
                    "a" * 64,
                    "b" * 64,
                    epoch["epoch_id"],
                    epoch["destination_class"],
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO conversation_compaction_batches(batch_key, first_message_id, "
                "last_message_id, outcome, memory_id, mirror_state, created_at, updated_at) "
                "VALUES ('promotion-exclusion-compaction', 1, 1, 'digest', ?, 'completed', ?, ?)",
                (ids["compaction"], now, now),
            )
            conn.execute(
                "UPDATE memories SET source = 'conversation_compaction' WHERE id = ?",
                (ids["compaction_source"],),
            )

        before = _database_rows(store)
        for owner, memory_id in ids.items():
            if store.resolve_knowledge_promotion_approval_target(memory_id) is not None:
                raise SystemExit(f"{owner}-owned memory resolved as a transferable candidate")
        if _database_rows(store) != before:
            raise SystemExit("owned-candidate exclusion checks mutated durable state")


def test_exact_transfer_reuses_source_and_refuses_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-exact-") as temp:
        store, _vault = _setup(Path(temp))
        title = "Choose the exact transfer"
        body = "Because this exact private rationale was reviewed."
        source = "manual-private-source"
        memory_id = _seed(store, title, body, source=source)
        before_memory = _memory(store, memory_id)
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("exact transfer fixture did not resolve")

        decision_title = "Reviewed exact decision title"
        decision_rationale = "Reviewed exact decision rationale"
        decision_impact = "Reviewed exact decision impact"
        result = _promote(
            store,
            memory_id,
            target,
            title=decision_title,
            rationale=decision_rationale,
            impact=decision_impact,
        )
        mutation = _assert_promoted(result, "exact promotion")
        if mutation.memory_id != memory_id:
            raise SystemExit("promotion replaced the reviewed source memory identity")
        with store.connect() as conn:
            decision = conn.execute(
                "SELECT * FROM decisions WHERE id = ?", (mutation.decision_id,)
            ).fetchone()
            links = list(conn.execute("SELECT decision_id, memory_id FROM decision_memory_links"))
            memories = list(conn.execute("SELECT id FROM memories"))
        if (
            decision is None
            or decision["title"] != decision_title
            or decision["rationale"] != decision_rationale
            or decision["impact"] != decision_impact
            or decision["status"] != "active"
            or int(decision["revision"]) != 1
        ):
            raise SystemExit(f"promotion changed the exact decision field mapping: {decision}")
        if [(int(row[0]), int(row[1])) for row in links] != [
            (mutation.decision_id, memory_id)
        ]:
            raise SystemExit(f"source memory was not the sole decision custody link: {links}")
        if [int(row[0]) for row in memories] != [memory_id]:
            raise SystemExit(f"promotion duplicated or replaced the source memory: {memories}")
        after_memory = _memory(store, memory_id)
        expected_body = (
            f"{decision_title}\n\nRationale: {decision_rationale}"
            f"\n\nImpact: {decision_impact}"
        )
        if (
            before_memory is None
            or after_memory is None
            or after_memory[:7]
            != (
                memory_id,
                "decisions",
                decision_title,
                expected_body,
                "decision-log",
                1.0,
                target.revision + 1,
            )
            or after_memory[7] != before_memory[7]
        ):
            raise SystemExit(
                "promotion did not preserve the source ID while applying exact canonical "
                f"fields/source/revision: before={before_memory}, after={after_memory}"
            )

        replay = _promote(
            store,
            memory_id,
            target,
            title=decision_title,
            rationale=decision_rationale,
            impact=decision_impact,
        )
        if _status(replay) in {"promoted", "transferred"} or _mutation(replay) is not None:
            raise SystemExit(f"second transfer was not refused: {replay!r}")
        if _counts(store) != {
            "decisions": 1,
            "decision_memory_links": 1,
            "decision_projection_jobs": 1,
            "memory_projection_jobs": 1,
        }:
            raise SystemExit(f"refused second transfer changed the graph: {_counts(store)}")
        if not store.decision_mutation_source_integrity(
            mutation.decision_id,
            memory_id,
            mutation.decision_projection_target,
            mutation.memory_projection_target,
        ):
            raise SystemExit("canonical promoted source graph failed integrity verification")
        with store.connect() as conn:
            preference = conn.execute(
                "INSERT INTO preferences(category, key, value, status, revision, created_at, updated_at) "
                "VALUES ('general', 'conflicting custody', 'yes', 'active', 1, ?, ?)",
                ("2026-07-13T00:00:00Z", "2026-07-13T00:00:00Z"),
            )
            preference_id = int(preference.lastrowid)
            conn.execute(
                "INSERT INTO preference_memory_links(preference_id, memory_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    preference_id,
                    memory_id,
                    "2026-07-13T00:00:00Z",
                    "2026-07-13T00:00:00Z",
                ),
            )
        if store.decision_mutation_source_integrity(
            mutation.decision_id,
            memory_id,
            mutation.decision_projection_target,
            mutation.memory_projection_target,
        ):
            raise SystemExit("conflicting structured custody passed source integrity")
        with store.connect() as conn:
            conn.execute(
                "DELETE FROM preference_memory_links WHERE preference_id = ?",
                (preference_id,),
            )
            conn.execute("DELETE FROM preferences WHERE id = ?", (preference_id,))
        with store.connect() as conn:
            conn.execute(
                "UPDATE memories SET body = 'tampered after promotion' WHERE id = ?",
                (memory_id,),
            )
            tampered = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        if tampered is None:
            raise SystemExit("tampered promoted memory disappeared")
        forged_memory_target = type(mutation.memory_projection_target)(
            memory_id=memory_id,
            revision=int(tampered["revision"]),
            operation="publish",
            source_digest=memory_projection_source_digest(
                memory_id,
                int(tampered["revision"]),
                tampered["category"],
                tampered["title"],
                tampered["body"],
                tampered["source"],
                tampered["confidence"],
                tampered["created_at"],
                "publish",
            ),
        )
        if store.decision_mutation_source_integrity(
            mutation.decision_id,
            memory_id,
            mutation.decision_projection_target,
            forged_memory_target,
        ):
            raise SystemExit("self-consistent noncanonical memory passed source integrity")


def test_storage_rejects_invalid_fields_and_oversized_identifiers() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-validation-") as temp:
        store, _vault = _setup(Path(temp))
        memory_id = _seed(store, "Validation target", "reviewed source")
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("validation fixture did not resolve")
        before = _database_rows(store)
        invalid_records = (
            DecisionRecord(title="", rationale="r", impact="i"),
            DecisionRecord(title=" whitespace ", rationale="r", impact="i"),
            DecisionRecord(title="valid", rationale="", impact="i"),
            DecisionRecord(title="valid", rationale=" whitespace ", impact="i"),
            DecisionRecord(title="valid", rationale="r", impact=""),
            DecisionRecord(title="valid", rationale="r", impact=" whitespace "),
            DecisionRecord(title="control\x00", rationale="r", impact="i"),
            DecisionRecord(title="valid", rationale="r", impact="i", status=""),
            DecisionRecord(title="valid", rationale="r", impact="i", status=" whitespace "),
            DecisionRecord(title="valid", rationale="r", impact="i", status="unknown"),
        )
        for record in invalid_records:
            try:
                store.promote_memory_to_decision_exact(
                    memory_id, target.revision, target.binding, record
                )
            except ValueError:
                pass
            else:
                raise SystemExit(f"invalid decision fields reached storage: {record!r}")
        for oversized_id, oversized_revision in (
            (9223372036854775808, target.revision),
            (memory_id, 9223372036854775808),
        ):
            try:
                store.promote_memory_to_decision_exact(
                    oversized_id,
                    oversized_revision,
                    target.binding,
                    DecisionRecord(title="valid", rationale="r", impact="i"),
                )
            except ValueError:
                pass
            else:
                raise SystemExit("oversized SQLite identifier was not rejected cleanly")
        if _database_rows(store) != before:
            raise SystemExit("invalid promotion inputs changed durable state")

        max_revision_id = _seed(store, "Max revision target", "cannot increment safely")
        with store.connect() as conn:
            conn.execute(
                "UPDATE memories SET revision = 9223372036854775807 WHERE id = ?",
                (max_revision_id,),
            )
        if store.resolve_knowledge_promotion_approval_target(max_revision_id) is not None:
            raise SystemExit("unincrementable max-int revision resolved as a promotion target")

        malformed_id = _seed(store, "Malformed storage title", "malformed storage body")
        with store.connect() as conn:
            conn.execute(
                "UPDATE memories SET title = CAST(X'00FF' AS BLOB) WHERE id = ?",
                (malformed_id,),
            )
        if store.resolve_knowledge_promotion_approval_target(malformed_id) is not None:
            raise SystemExit("malformed stored text resolved as a promotion target")
        oversized_title_id = _seed(store, "x" * 4097, "oversized title body")
        oversized_category_id = _seed(store, "oversized category title", "oversized category body")
        with store.connect() as conn:
            conn.execute(
                "UPDATE memories SET category = ? WHERE id = ?",
                ("x" * 1001, oversized_category_id),
            )
        for label, candidate_id in (
            ("oversized legacy title", oversized_title_id),
            ("oversized legacy category", oversized_category_id),
        ):
            if store.resolve_knowledge_promotion_approval_target(candidate_id) is not None:
                raise SystemExit(f"{label} resolved despite projection locator limits")
        if malformed_id in {
            int(row["id"])
            for row in store.list_knowledge_promotion_candidates(200)
        }:
            raise SystemExit("malformed stored text entered the learning-review candidate list")


def test_candidate_custody_indexes_are_migration_safe_and_used() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-indexes-") as temp:
        store, _vault = _setup(Path(temp))
        with store.connect() as conn:
            conn.execute("DROP INDEX IF EXISTS ingested_sources_memory_idx")
            conn.execute("DROP INDEX IF EXISTS conversation_compaction_batches_memory_idx")
        store.init()
        with store.connect() as conn:
            indexes = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' AND name IN (?, ?)",
                    (
                        "ingested_sources_memory_idx",
                        "conversation_compaction_batches_memory_idx",
                    ),
                )
            }
            if indexes != {
                "ingested_sources_memory_idx",
                "conversation_compaction_batches_memory_idx",
            }:
                raise SystemExit(f"candidate custody indexes were not recreated: {indexes}")
            plan_rows = conn.execute(
                "EXPLAIN QUERY PLAN "
                + store._knowledge_promotion_candidate_cte()
                + " SELECT id FROM unowned ORDER BY id LIMIT 10"
            ).fetchall()
        plan = "\n".join(str(row[3]) for row in plan_rows)
        for index_name in (
            "ingested_sources_memory_idx",
            "conversation_compaction_batches_memory_idx",
        ):
            if index_name not in plan:
                raise SystemExit(f"candidate query did not use {index_name}: {plan}")


def test_stale_edit_and_sql_body_drift_refuse_without_effects() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-stale-edit-") as temp:
        store, _vault = _setup(Path(temp))
        memory_id = _seed(store, "Stale edit", "before")
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("stale edit fixture did not resolve")
        if not store.update_memory(memory_id, "decisions", "Stale edit", "after", 0.73):
            raise SystemExit("stale edit fixture could not update its memory")
        before = _database_rows(store)
        result = _promote(store, memory_id, target)
        if _status(result) in {"promoted", "transferred"} or _database_rows(store) != before:
            raise SystemExit(f"ordinary edit did not stale promotion atomically: {result!r}")

    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-sql-drift-") as temp:
        store, _vault = _setup(Path(temp))
        memory_id = _seed(store, "SQL drift", "before direct drift")
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("SQL drift fixture did not resolve")
        with store.connect() as conn:
            conn.execute(
                "UPDATE memories SET body = 'after direct drift' WHERE id = ?",
                (memory_id,),
            )
        before = _database_rows(store)
        result = _promote(store, memory_id, target)
        if _status(result) in {"promoted", "transferred"} or _database_rows(store) != before:
            raise SystemExit(f"body drift without a revision bump bypassed binding: {result!r}")


def test_competing_transfers_have_one_winner() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-race-") as temp:
        store, _vault = _setup(Path(temp))
        memory_id = _seed(store, "Concurrent promotion", "one exact winner")
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("concurrent transfer fixture did not resolve")
        barrier = threading.Barrier(2)
        results: list[Any] = []
        errors: list[BaseException] = []

        def run() -> None:
            try:
                barrier.wait(timeout=5)
                results.append(_promote(store, memory_id, target))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        if errors or any(thread.is_alive() for thread in threads):
            raise SystemExit(f"competing transfer calls failed: {errors}")
        winners = [result for result in results if _status(result) in {"promoted", "transferred"}]
        if len(winners) != 1 or _mutation(winners[0]) is None:
            raise SystemExit(f"competing transfers did not produce one winner: {results!r}")
        if _counts(store) != {
            "decisions": 1,
            "decision_memory_links": 1,
            "decision_projection_jobs": 1,
            "memory_projection_jobs": 1,
        } or len(store.list_memories(limit=10)) != 1:
            raise SystemExit(f"competing transfers committed more than one graph: {_counts(store)}")


def test_trigger_failure_rolls_back_the_whole_transfer() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-rollback-") as temp:
        store, _vault = _setup(Path(temp))
        memory_id = _seed(store, "Atomic rollback", "retain only this source row")
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("rollback fixture did not resolve")
        with store.connect() as conn:
            conn.execute(
                "CREATE TRIGGER fail_knowledge_promotion BEFORE INSERT ON decision_memory_links "
                "BEGIN SELECT RAISE(ABORT, 'promotion link failure'); END"
            )
        before = _database_rows(store)
        try:
            _promote(store, memory_id, target)
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("promotion link trigger did not abort the transfer")
        if _database_rows(store) != before:
            raise SystemExit("trigger failure left a partial decision, link, or projection job")


def test_tool_projection_completion_pending_truth_and_metadata_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-tool-complete-") as temp:
        store, vault = _setup(Path(temp))
        title = "Tool private title marker"
        body = "Tool private body marker /private/tmp/promotion-secret.md"
        source = "tool-private-source-marker"
        memory_id = _seed(store, title, body, source=source)
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("tool completion fixture did not resolve")
        _packet, promote, _resolver = _factory_handlers(store, vault)
        decision_title = "Reviewed tool decision title"
        decision_rationale = "Reviewed tool decision rationale"
        decision_impact = "Reviewed tool decision impact"
        result = _tool_promote(
            promote,
            memory_id,
            target,
            title=decision_title,
            rationale=decision_rationale,
            impact=decision_impact,
        )
        if not result.ok or result.metadata.get("projection_pending") is True:
            raise SystemExit(f"completed promotion projections were not certified: {result}")
        with store.connect() as conn:
            decision_states = [row[0] for row in conn.execute("SELECT state FROM decision_projection_jobs")]
            memory_states = [row[0] for row in conn.execute("SELECT state FROM memory_projection_jobs")]
        if decision_states != ["completed"] or memory_states != ["completed"]:
            raise SystemExit(
                f"promotion tool left completed projections pending: {decision_states}, {memory_states}"
            )
        _assert_content_free_metadata(
            result.metadata,
            forbidden=(
                target.binding,
                str(vault.root_path),
                str(Path(temp)),
                title,
                body,
                source,
                decision_title,
                decision_rationale,
                decision_impact,
                "promotion-secret.md",
            ),
            label="successful promotion",
        )

    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-tool-pending-") as temp:
        store, vault = _setup(Path(temp))
        title = "Pending private title marker"
        body = "Pending private body marker /\x55sers/the operator/private/pending.md"
        source = "pending-private-source-marker"
        memory_id = _seed(store, title, body, source=source)
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("pending projection fixture did not resolve")
        _packet, promote, _resolver = _factory_handlers(store, vault)
        original = vault.write_decision_with_evidence

        def fail_projection(*_args: Any, **_kwargs: Any):
            raise OSError("private mock path /\x55sers/the operator/private/pending.md")

        vault.write_decision_with_evidence = fail_projection
        try:
            result = _tool_promote(
                promote,
                memory_id,
                target,
                title="Reviewed pending decision title",
                rationale="Reviewed pending decision rationale",
                impact="Reviewed pending decision impact",
            )
        finally:
            vault.write_decision_with_evidence = original
        if result.ok or result.metadata.get("projection_pending") is not True:
            raise SystemExit(f"mocked projection failure was not reported as pending: {result}")
        if _counts(store)["decisions"] != 1 or len(store.list_memories(limit=10)) != 1:
            raise SystemExit("pending projection result duplicated or lost durable source state")
        _assert_content_free_metadata(
            result.metadata,
            forbidden=(
                target.binding,
                str(vault.root_path),
                str(Path(temp)),
                title,
                body,
                source,
                "Reviewed pending decision title",
                "Reviewed pending decision rationale",
                "Reviewed pending decision impact",
                "pending.md",
            ),
            label="pending promotion",
        )


def test_review_binding_and_unknown_outcomes_fail_closed_without_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-reviewed-revision-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = _seed(store, "Reviewed revision", "before review drift")
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("reviewed-revision fixture did not resolve")
        _packet, promote, resolver = _factory_handlers(store, vault)
        raw_args = {
            "memory_id": memory_id,
            "reviewed_revision": target.revision,
            "review_token": target.binding,
            "title": "Explicit title",
            "rationale": "Explicit rationale",
            "impact": "Explicit impact",
        }
        if not store.update_memory(
            memory_id, "decisions", "Reviewed revision", "after review drift", 0.73
        ):
            raise SystemExit("reviewed-revision fixture did not drift")
        resolution = resolver(raw_args)
        if getattr(resolution, "args", None) is not None:
            raise SystemExit("stale reviewed revision reached approval binding")
        if _counts(store)["decisions"] != 0:
            raise SystemExit("stale reviewed revision changed decision state")

        for placeholder_fields in (
            {
                "title": "Decision <explicit impact>",
                "rationale": "<explicit rationale>",
                "impact": "<explicit impact>",
            },
            {
                "title": "Decision <title>",
                "rationale": "<rationale>",
                "impact": "<impact>",
            },
            {
                "title": "Decision < title >",
                "rationale": "<explicit  rationale>",
                "impact": "＜explicit impact＞",
            },
            {
                "title": "Decision < t i t l e >",
                "rationale": "<explicit r a t i o n a l e>",
                "impact": "Explicit impact",
            },
        ):
            placeholder = resolver(
                {
                    "memory_id": memory_id,
                    "reviewed_revision": 2,
                    "review_token": "0" * 64,
                    **placeholder_fields,
                }
            )
            if getattr(placeholder, "args", None) is not None:
                raise SystemExit("partial or reordered packet placeholders reached approval binding")

        for path_value in (
            "path=/\x55sers/the operator/private.txt",
            "file:///private/tmp/private.txt",
            r"target=C:\Users\the operator\private.txt",
            "/etc/passwd",
            "/root/.ssh/id_ed25519",
            "/opt/private/data.db",
            "//server/share/private.txt",
            "file:///etc/passwd",
            "file://server/share/private.txt",
            "secrets/config.yaml",
            r"secrets\config.yaml",
            "line\u2028separator",
            "paragraph\u2029separator",
            "surrogate\ud800value",
        ):
            path_refusal = resolver(
                {
                    "memory_id": memory_id,
                    "reviewed_revision": 2,
                    "review_token": "0" * 64,
                    "title": "Explicit title",
                    "rationale": path_value,
                    "impact": "Explicit impact",
                }
            )
            if getattr(path_refusal, "args", None) is not None:
                raise SystemExit(f"delimited local path bypassed field validation: {path_value}")

        current_target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if current_target is None:
            raise SystemExit("URL validation fixture did not resolve after revision drift")
        for reviewed_url in (
            "https://example.com/reviewed-source",
            "https://example.com/private/review",
            "https://example.com:8443/\x55sers/review",
            "https://[::1]/private/review",
            "https://例え.テスト/private/review",
            "https://example.com.",
            "(https://example.com)",
        ):
            url_resolution = resolver(
                {
                    "memory_id": memory_id,
                    "reviewed_revision": current_target.revision,
                    "review_token": current_target.binding,
                    "title": "Use the reviewed documentation",
                    "rationale": f"The reference is {reviewed_url}",
                    "impact": "The decision remains explicit",
                }
            )
            if getattr(url_resolution, "args", None) is None:
                raise SystemExit(f"ordinary HTTPS URL was misclassified as a local path: {reviewed_url}")
        malformed_url_path = resolver(
            {
                "memory_id": memory_id,
                "reviewed_revision": current_target.revision,
                "review_token": current_target.binding,
                "title": "Use a malformed reference",
                "rationale": "The reference is https:///private/review",
                "impact": "This must be refused",
            }
        )
        if getattr(malformed_url_path, "args", None) is not None:
            raise SystemExit("malformed HTTP URL hid a local path")
        for disguised_path in (
            "https://file:///private/review",
            "http://C:/\x55sers/the operator/private.txt",
            "https://:443/review",
            "https://example.com:99999/private/review",
            "https://example.com)/\x55sers/the operator/private.txt",
            "https://999.999.999.999/a",
        ):
            disguised_refusal = resolver(
                {
                    "memory_id": memory_id,
                    "reviewed_revision": current_target.revision,
                    "review_token": current_target.binding,
                    "title": "Use a malformed reference",
                    "rationale": f"The reference is {disguised_path}",
                    "impact": "This must be refused",
                }
            )
            if getattr(disguised_refusal, "args", None) is not None:
                raise SystemExit(f"HTTP-shaped input hid a local path: {disguised_path}")

        for active_embed in (
            "![review](https://example.com/tracker.png)",
            '<img src="https://example.com/tracker.png">',
            '<iframe src="https://example.com/review"></iframe>',
        ):
            embed_refusal = resolver(
                {
                    "memory_id": memory_id,
                    "reviewed_revision": current_target.revision,
                    "review_token": current_target.binding,
                    "title": "Explicit title",
                    "rationale": active_embed,
                    "impact": "This must be refused",
                }
            )
            if getattr(embed_refusal, "args", None) is not None:
                raise SystemExit(
                    f"active remote-resource embed reached approval binding: {active_embed}"
                )

        malformed_target = KnowledgePromotionCandidateTarget(
            memory_target=MemoryApprovalTarget(
                memory_id=memory_id,
                revision=current_target.revision,
                binding="not-a-valid-review-token\n",
            ),
            target_kind="decision",
            category="decisions",
            title="Malformed binding title",
            body="Malformed binding body",
        )
        packet, _promote, _resolver = _factory_handlers(store, vault)
        original_target_resolver = store.resolve_knowledge_promotion_approval_target
        store.resolve_knowledge_promotion_approval_target = (  # type: ignore[method-assign]
            lambda _memory_id: malformed_target
        )
        try:
            malformed_packet = packet({"memory_id": memory_id})
        finally:
            store.resolve_knowledge_promotion_approval_target = original_target_resolver  # type: ignore[method-assign]
        if malformed_packet.ok or "malformed" not in malformed_packet.output.lower():
            raise SystemExit("malformed opaque candidate token reached a successful packet")

        malformed_kind_target = KnowledgePromotionCandidateTarget(
            memory_target=MemoryApprovalTarget(
                memory_id=memory_id,
                revision=current_target.revision,
                binding=current_target.binding,
            ),
            target_kind="decision\n/private/review",
            category="decisions",
            title="Malformed kind title",
            body="Malformed kind body",
        )
        store.resolve_knowledge_promotion_approval_target = (  # type: ignore[method-assign]
            lambda _memory_id: malformed_kind_target
        )
        try:
            malformed_kind_packet = packet({"memory_id": memory_id})
        finally:
            store.resolve_knowledge_promotion_approval_target = original_target_resolver  # type: ignore[method-assign]
        if malformed_kind_packet.ok or "malformed" not in malformed_kind_packet.output.lower():
            raise SystemExit("malformed target kind reached a successful packet")

        for malformed_snapshot_target in (
            KnowledgePromotionCandidateTarget(
                memory_target=MemoryApprovalTarget(
                    memory_id=memory_id,
                    revision=current_target.revision,
                    binding=current_target.binding,
                ),
                target_kind="decision",
                category=object(),  # type: ignore[arg-type]
                title="Malformed snapshot title",
                body="Malformed snapshot body",
            ),
            KnowledgePromotionCandidateTarget(
                memory_target=MemoryApprovalTarget(
                    memory_id=memory_id,
                    revision=current_target.revision,
                    binding=current_target.binding,
                ),
                target_kind="decision",
                category="profile",
                title="Mismatched snapshot title",
                body="Mismatched snapshot body",
            ),
        ):
            store.resolve_knowledge_promotion_approval_target = (  # type: ignore[method-assign]
                lambda _memory_id, target=malformed_snapshot_target: target
            )
            try:
                malformed_snapshot_resolution = resolver(
                    {
                        "memory_id": memory_id,
                        "reviewed_revision": current_target.revision,
                        "review_token": current_target.binding,
                        "title": "Explicit title",
                        "rationale": "Explicit rationale",
                        "impact": "Explicit impact",
                    }
                )
            finally:
                store.resolve_knowledge_promotion_approval_target = original_target_resolver  # type: ignore[method-assign]
            if (
                getattr(malformed_snapshot_resolution, "args", None) is not None
                or getattr(malformed_snapshot_resolution, "metadata", {}).get(
                    "approval_argument_resolution_status"
                )
                != "malformed_candidate_snapshot"
            ):
                raise SystemExit("malformed candidate snapshot reached approval binding")

        class ExplodingCandidate:
            @property
            def target_kind(self) -> str:
                raise RuntimeError("private target property failure")

        store.resolve_knowledge_promotion_approval_target = (  # type: ignore[method-assign]
            lambda _memory_id: ExplodingCandidate()
        )
        try:
            exploding_packet = packet({"memory_id": memory_id})
        finally:
            store.resolve_knowledge_promotion_approval_target = original_target_resolver  # type: ignore[method-assign]
        if (
            exploding_packet.ok
            or "malformed" not in exploding_packet.output.lower()
            or "private target property failure" in exploding_packet.output
            or "private target property failure" in repr(exploding_packet.metadata)
        ):
            raise SystemExit("candidate property failure escaped the packet boundary")

        class RaisingEquality:
            def __eq__(self, _other: Any) -> bool:
                raise RuntimeError("private equality failure")

        hostile_identity_target = KnowledgePromotionCandidateTarget(
            memory_target=MemoryApprovalTarget(
                memory_id=RaisingEquality(),  # type: ignore[arg-type]
                revision=current_target.revision,
                binding=current_target.binding,
            ),
            target_kind="decision",
            category="decisions",
            title="Hostile identity title",
            body="Hostile identity body",
        )
        store.resolve_knowledge_promotion_approval_target = (  # type: ignore[method-assign]
            lambda _memory_id: hostile_identity_target
        )
        try:
            hostile_identity_packet = packet({"memory_id": memory_id})
        finally:
            store.resolve_knowledge_promotion_approval_target = original_target_resolver  # type: ignore[method-assign]
        if hostile_identity_packet.ok or "private equality failure" in repr(hostile_identity_packet):
            raise SystemExit("hostile candidate equality escaped fail-closed validation")

        oversized_bound = promote(
            {
                "memory_id": memory_id,
                "reviewed_revision": 9223372036854775807,
                "review_binding": current_target.binding,
                "title": "Explicit title",
                "rationale": "Explicit rationale",
                "impact": "Explicit impact",
                "target_revision": 9223372036854775807,
                "target_binding": current_target.binding,
            }
        )
        if (
            oversized_bound.ok
            or oversized_bound.metadata.get("reason") != "unbound_target"
            or oversized_bound.metadata.get("execution_outcome_unknown") is not None
            or oversized_bound.metadata.get("handler_invoked") is not True
        ):
            raise SystemExit(f"oversized bound revision did not refuse deterministically: {oversized_bound}")

    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-review-token-drift-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = _seed(store, "Review token drift", "packet-reviewed body")
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("review-token drift fixture did not resolve")
        packet, _promote, resolver = _factory_handlers(store, vault)
        packet_result = packet({"memory_id": memory_id})
        if not packet_result.ok or target.binding not in packet_result.output:
            raise SystemExit("packet did not expose its opaque full-row review token")
        with store.connect() as conn:
            conn.execute(
                "UPDATE memories SET body = 'direct drift without revision' WHERE id = ?",
                (memory_id,),
            )
        resolution = resolver(
            {
                "memory_id": memory_id,
                "reviewed_revision": target.revision,
                "review_token": target.binding,
                "title": "Explicit title",
                "rationale": "Explicit rationale",
                "impact": "Explicit impact",
            }
        )
        if getattr(resolution, "args", None) is not None:
            raise SystemExit("direct body drift reused a stale packet review token")

    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-executor-refusal-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(runtime.store, "Executor refusal", "reviewed body")
        target = runtime.store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("executor-refusal fixture did not resolve")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE memories SET body = 'drift before executor' WHERE id = ?",
                (memory_id,),
            )
        command = (
            f"promote memory {memory_id} revision {target.revision} token {target.binding} "
            "to decision: Explicit title | Explicit rationale | Explicit impact"
        )
        before_pending = len(runtime.store.list_pending_approvals())
        runtime_result = runtime.handle(command)
        after_pending = len(runtime.store.list_pending_approvals())
        if len(runtime_result.tool_results) != 1:
            raise SystemExit("executor-refusal path did not return one tool result")
        refusal = runtime_result.tool_results[0]
        if (
            refusal.ok
            or refusal.metadata.get("requires_confirmation") is not False
            or refusal.metadata.get("approval_argument_resolution_status") != "review_token_stale"
            or refusal.metadata.get("failure_kind") == "approval_argument_resolution_failed"
            or before_pending != after_pending
        ):
            raise SystemExit(f"executor replaced or queued a safe resolver refusal: {refusal}")

    for mode in (
        "exception",
        "malformed",
        "stale_shaped",
        "promoted_shaped",
        "exact_promoted_malformed",
        "exact_promoted_decision_id_mismatch",
        "exact_promoted_decision_revision_malformed",
        "exact_promoted_decision_digest_malformed",
        "exact_promoted_memory_id_mismatch",
        "exact_promoted_memory_revision_malformed",
        "exact_promoted_memory_operation_malformed",
        "exact_promoted_memory_digest_malformed",
        "exact_promoted_status_malformed",
        "exact_promoted_target_fields_hostile",
    ):
        with TemporaryDirectory(prefix=f"jarvis-knowledge-promotion-{mode}-") as temp:
            store, vault = _setup(Path(temp))
            memory_id = _seed(store, f"{mode} outcome", "outcome must remain unknown")
            target = store.resolve_knowledge_promotion_approval_target(memory_id)
            if target is None:
                raise SystemExit(f"{mode} outcome fixture did not resolve")
            _packet, promote, _resolver = _factory_handlers(store, vault)
            original = store.promote_memory_to_decision_exact
            if mode == "exception":
                def replacement(*_args: Any, **_kwargs: Any):
                    raise sqlite3.DatabaseError("private storage outcome detail")
            elif mode == "stale_shaped":
                replacement = lambda *_args, **_kwargs: type(  # type: ignore[misc]
                    "MalformedStale", (), {"status": "stale"}
                )()
            elif mode == "promoted_shaped":
                replacement = lambda *_args, **_kwargs: type(  # type: ignore[misc]
                    "MalformedPromoted", (), {"status": "promoted"}
                )()
            elif mode == "exact_promoted_malformed":
                replacement = lambda *_args, **_kwargs: MemoryDecisionPromotionResult(
                    "promoted", 1, memory_id, object(), object()  # type: ignore[arg-type]
                )
            elif mode == "exact_promoted_decision_id_mismatch":
                replacement = lambda *_args, **_kwargs: MemoryDecisionPromotionResult(
                    "promoted",
                    1,
                    memory_id,
                    DecisionProjectionTarget(2, 1, "a" * 64),
                    MemoryProjectionTarget(memory_id, 1, "publish", "b" * 64),
                )
            elif mode == "exact_promoted_decision_revision_malformed":
                replacement = lambda *_args, **_kwargs: MemoryDecisionPromotionResult(
                    "promoted",
                    1,
                    memory_id,
                    DecisionProjectionTarget(1, 0, "a" * 64),
                    MemoryProjectionTarget(memory_id, 1, "publish", "b" * 64),
                )
            elif mode == "exact_promoted_decision_digest_malformed":
                replacement = lambda *_args, **_kwargs: MemoryDecisionPromotionResult(
                    "promoted",
                    1,
                    memory_id,
                    DecisionProjectionTarget(1, 1, "not-a-digest"),
                    MemoryProjectionTarget(memory_id, 1, "publish", "b" * 64),
                )
            elif mode == "exact_promoted_memory_id_mismatch":
                replacement = lambda *_args, **_kwargs: MemoryDecisionPromotionResult(
                    "promoted",
                    1,
                    memory_id,
                    DecisionProjectionTarget(1, 1, "a" * 64),
                    MemoryProjectionTarget(memory_id + 1, 1, "publish", "b" * 64),
                )
            elif mode == "exact_promoted_memory_revision_malformed":
                replacement = lambda *_args, **_kwargs: MemoryDecisionPromotionResult(
                    "promoted",
                    1,
                    memory_id,
                    DecisionProjectionTarget(1, 1, "a" * 64),
                    MemoryProjectionTarget(memory_id, 0, "publish", "b" * 64),
                )
            elif mode == "exact_promoted_memory_operation_malformed":
                replacement = lambda *_args, **_kwargs: MemoryDecisionPromotionResult(
                    "promoted",
                    1,
                    memory_id,
                    DecisionProjectionTarget(1, 1, "a" * 64),
                    MemoryProjectionTarget(memory_id, 1, "delete", "b" * 64),
                )
            elif mode == "exact_promoted_memory_digest_malformed":
                replacement = lambda *_args, **_kwargs: MemoryDecisionPromotionResult(
                    "promoted",
                    1,
                    memory_id,
                    DecisionProjectionTarget(1, 1, "a" * 64),
                    MemoryProjectionTarget(memory_id, 1, "publish", "not-a-digest"),
                )
            elif mode == "exact_promoted_status_malformed":
                replacement = lambda *_args, **_kwargs: MemoryDecisionPromotionResult(
                    [], 1, memory_id,  # type: ignore[arg-type]
                    DecisionProjectionTarget(1, 1, "a" * 64),
                    MemoryProjectionTarget(memory_id, 1, "publish", "b" * 64),
                )
            elif mode == "exact_promoted_target_fields_hostile":
                replacement = lambda *_args, **_kwargs: MemoryDecisionPromotionResult(
                    "promoted",
                    1,
                    memory_id,
                    DecisionProjectionTarget(object(), object(), object()),  # type: ignore[arg-type]
                    MemoryProjectionTarget(object(), object(), object(), object()),  # type: ignore[arg-type]
                )
            else:
                replacement = lambda *_args, **_kwargs: object()
            store.promote_memory_to_decision_exact = replacement  # type: ignore[method-assign]
            try:
                result = _tool_promote(
                    promote,
                    memory_id,
                    target,
                    title="Explicit title",
                    rationale="Explicit rationale",
                    impact="Explicit impact",
                )
            finally:
                store.promote_memory_to_decision_exact = original  # type: ignore[method-assign]
            if (
                result.ok
                or result.metadata.get("promotion_outcome") != "unknown"
                or result.metadata.get("projection_pending") is not None
                or result.metadata.get("projection_state") != "unknown"
                or classify_approved_execution_outcome(
                    ok=result.ok,
                    metadata=result.metadata,
                    risk=RiskLevel.HIGH_RISK,
                ).outcome_unknown
                is not True
                or "Do not promote the memory again" not in result.output
                or "private storage outcome detail" in result.output
                or "private storage outcome detail" in repr(result.metadata)
            ):
                raise SystemExit(f"{mode} mutation outcome was not fail-closed: {result}")

    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-bool-contract-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = _seed(store, "Boolean contract", "exact booleans only")
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("boolean completion fixture did not resolve")
        _packet, promote, _resolver = _factory_handlers(store, vault)
        original_completion = store.decision_mutation_projection_completion
        store.decision_mutation_projection_completion = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: ("completed", 1)
        )
        try:
            result = _tool_promote(
                promote,
                memory_id,
                target,
                title="Explicit title",
                rationale="Explicit rationale",
                impact="Explicit impact",
            )
        finally:
            store.decision_mutation_projection_completion = original_completion  # type: ignore[method-assign]
        if (
            result.ok
            or result.metadata.get("projection_completion_contract_valid") is not False
            or "durability verification contract was malformed" not in result.output
        ):
            raise SystemExit(f"truthy malformed completion values certified durability: {result}")

    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-source-integrity-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = _seed(store, "Source integrity", "source graph must verify")
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("source-integrity fixture did not resolve")
        _packet, promote, _resolver = _factory_handlers(store, vault)
        original_integrity = store.decision_mutation_source_integrity
        store.decision_mutation_source_integrity = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: False
        )
        try:
            result = _tool_promote(
                promote,
                memory_id,
                target,
                title="Explicit title",
                rationale="Explicit rationale",
                impact="Explicit impact",
            )
        finally:
            store.decision_mutation_source_integrity = original_integrity  # type: ignore[method-assign]
        if (
            result.ok
            or result.metadata.get("source_integrity_verified") is not False
            or "source integrity verification failed" not in result.output
            or "projection repair alone cannot resolve" not in result.output
        ):
            raise SystemExit(f"source-integrity failure looked projection-repairable: {result}")

    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-source-unavailable-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = _seed(store, "Source verification unavailable", "source graph read may fail")
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None:
            raise SystemExit("source-verification-unavailable fixture did not resolve")
        _packet, promote, _resolver = _factory_handlers(store, vault)
        original_integrity = store.decision_mutation_source_integrity

        def unavailable_integrity(*_args: Any, **_kwargs: Any) -> bool:
            raise sqlite3.DatabaseError("private verification outage")

        store.decision_mutation_source_integrity = unavailable_integrity  # type: ignore[method-assign]
        try:
            unavailable = _tool_promote(
                promote,
                memory_id,
                target,
                title="Explicit title",
                rationale="Explicit rationale",
                impact="Explicit impact",
            )
        finally:
            store.decision_mutation_source_integrity = original_integrity  # type: ignore[method-assign]
        if (
            unavailable.ok
            or unavailable.metadata.get("source_integrity_verified") is not None
            or unavailable.metadata.get("source_integrity_status") != "unavailable"
            or "verification was unavailable" not in unavailable.output
            or "verification failed" in unavailable.output
            or "private verification outage" in unavailable.output
            or "private verification outage" in repr(unavailable.metadata)
        ):
            raise SystemExit(f"unavailable source verification was misreported: {unavailable}")


def test_candidate_snapshot_fields_are_bounded_before_binding() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-bounds-") as temp:
        store, vault = _setup(Path(temp))
        packet, _promote, _resolver = _factory_handlers(store, vault)
        oversized_fields = (
            ("body", "b" * 1_000_001),
            ("source", "s" * 4097),
            ("created_at", "c" * 257),
            ("updated_at", "u" * 257),
        )
        for column, value in oversized_fields:
            memory_id = _seed(store, f"Oversized {column}", "bounded candidate")
            with store.connect() as conn:
                conn.execute(
                    f"UPDATE memories SET {column} = ? WHERE id = ?",
                    (value, memory_id),
                )
            if store.resolve_knowledge_promotion_approval_target(memory_id) is not None:
                raise SystemExit(
                    f"oversized candidate {column} reached approval-target hashing"
                )
            result = packet({"memory_id": memory_id})
            if result.ok or result.metadata.get("candidate_state") == "ready":
                raise SystemExit(
                    f"oversized candidate {column} reached a successful review packet"
                )


def test_planner_and_registry_keep_the_transfer_explicit_and_high_risk() -> None:
    planner = RuleBasedPlanner()
    review_token = "a" * 64
    packet_cases = {
        "knowledge promotion packet 12": 12,
        "review knowledge memory #12": 12,
    }
    for command, memory_id in packet_cases.items():
        plan = planner.plan(command)
        if len(plan.actions) != 1 or plan.actions[0].tool_name != "knowledge_promotion_packet":
            raise SystemExit(f"promotion packet command did not route exactly: {command!r}")
        if plan.actions[0].args != {"memory_id": memory_id}:
            raise SystemExit(f"promotion packet command changed its exact id: {plan.actions[0].args}")
    for invalid_packet in (
        "knowledge promotion packet 9223372036854775808",
        "knowledge promotion packet " + "9" * 5000,
    ):
        invalid_plan = planner.plan(invalid_packet)
        if invalid_plan.actions or invalid_plan.needs_model:
            raise SystemExit(f"invalid packet id did not fail closed: {invalid_packet[:80]!r}")

    promote_command = (
        f"promote memory 12 revision 3 token {review_token} to decision: "
        "Reviewed title | Reviewed rationale | "
        "Reviewed impact"
    )
    promote_plan = planner.plan(promote_command)
    if len(promote_plan.actions) != 1 or promote_plan.actions[0].tool_name != "promote_memory_to_decision":
        raise SystemExit("strict decision promotion command did not reach the approval-gated tool")
    if promote_plan.actions[0].args != {
        "memory_id": 12,
        "reviewed_revision": 3,
        "review_token": review_token,
        "title": "Reviewed title",
        "rationale": "Reviewed rationale",
        "impact": "Reviewed impact",
    }:
        raise SystemExit(f"strict decision fields changed during planning: {promote_plan.actions[0].args}")

    preference_command = (
        f"promote memory 12 revision 3 token {review_token} to preference: "
        "communication | daily brief language | Korean"
    )
    preference_plan = planner.plan(preference_command)
    if (
        len(preference_plan.actions) != 1
        or preference_plan.actions[0].tool_name != "promote_memory_to_preference"
        or preference_plan.actions[0].args
        != {
            "memory_id": 12,
            "reviewed_revision": 3,
            "review_token": review_token,
            "category": "communication",
            "key": "daily brief language",
            "value": "Korean",
        }
    ):
        raise SystemExit(
            "strict preference promotion command did not preserve its explicit fields"
        )

    long_rationale = "r" * 3900
    long_impact = "i" * 200
    long_command = (
        f"promote memory 12 revision 3 token {review_token} to decision: "
        f"Reviewed title | {long_rationale} | {long_impact}"
    )
    if len(long_command) <= 4000:
        raise SystemExit("long promotion fixture did not cross the global planner cap")
    long_plan = planner.plan(long_command)
    if (
        len(long_plan.actions) != 1
        or long_plan.actions[0].tool_name != "promote_memory_to_decision"
        or long_plan.actions[0].args.get("rationale") != long_rationale
        or long_plan.actions[0].args.get("impact") != long_impact
    ):
        raise SystemExit("long exact promotion fields were truncated before approval")

    oversized_command = (
        f"promote memory 12 revision 3 token {review_token} to decision: "
        f"Reviewed title | {'r' * 4001} | Reviewed impact"
    )
    oversized_plan = planner.plan(oversized_command)
    if (
        len(oversized_plan.actions) != 1
        or oversized_plan.actions[0].tool_name != "knowledge_promotion_packet"
    ):
        raise SystemExit("oversized promotion command was not rejected through a safe packet refresh")

    for malformed in (
        f"promote memory 12 revision 3 token {review_token} to decision: infer it from the memory",
        f"promote memory 12 revision 3 token {review_token} to decision: title | rationale",
        f"promote memory 12 revision 3 token {review_token} to decision: title | rationale | impact | extra",
        f"promote memory 12 revision 3 token {review_token} to preference: key | value",
        f"promote memory 12 revision 3 token {review_token} to preference: category | key | value | extra",
        f"promote memory 12 revision 3 token {review_token} to decision: <explicit title> | <explicit rationale> | <explicit impact>",
        f"promote memory 12 revision 3 token {review_token} to decision: <explicit impact> | <explicit title> | <explicit rationale>",
        f"promote memory 12 revision 3 token {review_token} to decision: Decision <explicit title> | rationale | impact",
        f"promote memory 12 revision 3 token {review_token} to decision: <title> | <rationale> | <impact>",
        f"promote memory 12 revision 3 token {review_token} to decision: Decision <title> | rationale | impact",
        f"promote memory 12 revision 3 token {review_token} to decision: < title > | <explicit  rationale> | ＜explicit impact＞",
        f"promote memory 12 revision 3 token {review_token} to decision: < t i t l e > | <explicit r a t i o n a l e> | impact",
        f"promote memory 12 revision 3 token {review_token} to decision:     | rationale | impact",
        f"promote memory 12 revision 3 token {review_token} to decision: title |     | impact",
        f"promote memory 12 revision 3 token {review_token} to decision: title | rationale |     ",
        "promote memory " + "9" * 5000 + f" revision 3 token {review_token} to decision: title | rationale | impact",
        f"promote memory 12 revision {'9' * 5000} token {review_token} to decision: title | rationale | impact",
    ):
        if any(
            action.tool_name in {
                "promote_memory_to_decision",
                "promote_memory_to_preference",
            }
            for action in planner.plan(malformed).actions
        ):
            raise SystemExit(f"ambiguous promotion syntax reached mutation planning: {malformed!r}")

    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-registry-") as temp:
        runtime = make_temp_runtime(Path(temp))
        packet_tool = runtime.registry.get("knowledge_promotion_packet")
        promotion_tool = runtime.registry.get("promote_memory_to_decision")
        preference_promotion_tool = runtime.registry.get(
            "promote_memory_to_preference"
        )
        if packet_tool.risk is not RiskLevel.READ_ONLY:
            raise SystemExit("knowledge promotion packet is not read-only")
        if (
            promotion_tool.risk is not RiskLevel.HIGH_RISK
            or promotion_tool.approval_argument_resolver is None
            or promotion_tool.approval_argument_contract is None
        ):
            raise SystemExit("decision promotion lost its high-risk approval boundary")
        if (
            preference_promotion_tool.risk is not RiskLevel.HIGH_RISK
            or preference_promotion_tool.approval_argument_resolver is None
            or preference_promotion_tool.approval_argument_contract is None
        ):
            raise SystemExit("preference promotion lost its high-risk approval boundary")


def main() -> None:
    test_exact_candidate_binding_and_preapproval_read_only()
    test_all_owned_candidate_classes_are_excluded()
    test_exact_transfer_reuses_source_and_refuses_replay()
    test_storage_rejects_invalid_fields_and_oversized_identifiers()
    test_candidate_custody_indexes_are_migration_safe_and_used()
    test_stale_edit_and_sql_body_drift_refuse_without_effects()
    test_competing_transfers_have_one_winner()
    test_trigger_failure_rolls_back_the_whole_transfer()
    test_tool_projection_completion_pending_truth_and_metadata_privacy()
    test_review_binding_and_unknown_outcomes_fail_closed_without_replay()
    test_candidate_snapshot_fields_are_bounded_before_binding()
    test_planner_and_registry_keep_the_transfer_explicit_and_high_risk()
    print("Knowledge promotion smoke passed")


if __name__ == "__main__":
    main()
