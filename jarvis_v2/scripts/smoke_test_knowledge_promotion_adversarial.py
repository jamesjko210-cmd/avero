from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    MAX_KNOWLEDGE_PROMOTION_SOURCE_REVISION,
    DecisionRecord,
    MemoryRecord,
    MemoryProjectionTarget,
    MemoryStore,
    PreferenceProjectionTarget,
    PreferenceRecord,
)
from jarvis_v2.tools.knowledge_promotion import (
    MAX_CANDIDATE_SOURCE_CHARS,
    MAX_CANDIDATE_TIMESTAMP_CHARS,
    MAX_PACKET_BODY_CHARS,
    MAX_PACKET_CATEGORY_CHARS,
    MAX_PACKET_TITLE_CHARS,
    make_knowledge_promotion_tools,
)


SQLITE_MAX_INTEGER = 9_223_372_036_854_775_807
REQUIRED_PROMOTION_HEADROOM = 1_024
GRAPH_TABLES = (
    "memories",
    "memory_projection_jobs",
    "decisions",
    "decision_memory_links",
    "decision_projection_jobs",
    "preferences",
    "preference_identity_owners",
    "preference_memory_links",
    "preference_projection_state",
    "preference_projection_jobs",
    "store_instance_state",
)


def _setup(root: Path) -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / "store.sqlite")
    store.init()
    vault = ObsidianVault(root / "vault")
    vault.init()
    return store, vault


def _seed(
    store: MemoryStore,
    *,
    category: str = "decisions",
    title: str = "Adversarial promotion candidate",
    body: str = "Candidate content must cross every eligibility gate intact.",
    source: str = "knowledge-promotion-adversarial",
    confidence: float = 0.875,
) -> int:
    return store.add_memory(
        MemoryRecord(category, title, body, source, confidence)
    )


def _handlers(
    store: MemoryStore, vault: ObsidianVault
) -> tuple[Callable[[dict[str, Any]], Any], ...]:
    handlers = tuple(make_knowledge_promotion_tools(store, vault))
    if len(handlers) != 5 or not all(callable(handler) for handler in handlers):
        raise AssertionError("knowledge promotion factory returned a malformed handler set")
    return handlers


def _database_snapshot(store: MemoryStore) -> dict[str, list[tuple[Any, ...]]]:
    with store.connect() as conn:
        return {
            table: [
                tuple(row)
                for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")
            ]
            for table in GRAPH_TABLES
        }


def _vault_snapshot(vault: ObsidianVault) -> dict[str, bytes]:
    return {
        str(path.relative_to(vault.root_path)): path.read_bytes()
        for path in sorted(vault.root_path.rglob("*"))
        if path.is_file()
    }


def _status(result: Any) -> str:
    status = getattr(result, "status", "")
    return status if type(status) is str else ""


def _different_hex(value: str, length: int) -> str:
    candidate = "f" * length
    return ("e" * length) if value == candidate else candidate


def _generic_target(store: MemoryStore, memory_id: int) -> Any:
    target = store.resolve_memory_approval_target(memory_id)
    if target is None:
        raise AssertionError(f"memory #{memory_id} had no generic exact target")
    return target


def _assert_candidate_rejected_before_binding(
    store: MemoryStore, memory_id: int, *, label: str
) -> None:
    original = store._memory_approval_binding
    binding_calls = 0

    def counted_binding(*args: Any, **kwargs: Any) -> str:
        nonlocal binding_calls
        binding_calls += 1
        return original(*args, **kwargs)

    store._memory_approval_binding = counted_binding  # type: ignore[method-assign]
    try:
        candidate = store.resolve_knowledge_promotion_approval_target(memory_id)
    finally:
        store._memory_approval_binding = original  # type: ignore[method-assign]
    if candidate is not None or binding_calls != 0:
        raise AssertionError(
            f"{label} reached candidate binding: candidate={candidate!r}, "
            f"binding_calls={binding_calls}"
        )


def _assert_decision_commit_rejected_without_effects(
    store: MemoryStore,
    vault: ObsidianVault,
    memory_id: int,
    target: Any,
    *,
    label: str,
) -> None:
    packet, promote, _resolve, _promote_preference, _resolve_preference = _handlers(
        store, vault
    )
    before_database = _database_snapshot(store)
    before_vault = _vault_snapshot(vault)
    candidate = store.resolve_knowledge_promotion_approval_target(memory_id)
    packet_result = packet({"memory_id": memory_id})
    tool_result = promote(
        {
            "memory_id": memory_id,
            "reviewed_revision": target.revision,
            "review_binding": target.binding,
            "title": "Reviewed adversarial decision",
            "rationale": "The source candidate was reviewed explicitly.",
            "impact": "Ineligible custody must have a definite no-effect result.",
            "target_revision": target.revision,
            "target_binding": target.binding,
        }
    )
    promotion = store.promote_memory_to_decision_exact(
        memory_id,
        target.revision,
        target.binding,
        DecisionRecord(
            title="Reviewed adversarial decision",
            rationale="The source candidate was reviewed explicitly.",
            impact="No mutation is allowed when candidate custody is ineligible.",
        ),
    )
    after_database = _database_snapshot(store)
    after_vault = _vault_snapshot(vault)
    if (
        candidate is not None
        or packet_result.ok
        or tool_result.ok
        or tool_result.metadata.get("reason") != "exact_promotion_refused"
        or tool_result.metadata.get("promotion_outcome") == "unknown"
        or tool_result.metadata.get("execution_outcome_unknown") is True
        or _status(promotion) != "not_candidate"
        or after_database != before_database
        or after_vault != before_vault
    ):
        raise AssertionError(
            f"{label} was promotable or mutated the graph: candidate={candidate!r}, "
            f"packet_ok={packet_result.ok!r}, tool_ok={tool_result.ok!r}, "
            f"tool_reason={tool_result.metadata.get('reason')!r}, "
            f"tool_outcome={tool_result.metadata.get('promotion_outcome')!r}, "
            f"status={_status(promotion)!r}, "
            f"database_changed={after_database != before_database}, "
            f"vault_changed={after_vault != before_vault}"
        )


def _complete_projection(
    store: MemoryStore, vault: ObsidianVault, target: Any
) -> Any:
    outcome = reconcile_memory_projection(
        store,
        vault,
        target.memory_id,
        expected_operation=target.operation,
        expected_revision=target.revision,
        expected_source_digest=target.source_digest,
    )
    if outcome.status != "completed":
        raise AssertionError(f"memory projection did not complete: {outcome!r}")
    return outcome


def test_projection_job_custody_controls_eligibility() -> None:
    with TemporaryDirectory(prefix="jarvis-promotion-job-custody-") as temp:
        root = Path(temp)

        pending_store, pending_vault = _setup(root / "completion-lost")
        pending_target = pending_store.add_memory_with_projection(
            MemoryRecord(
                "decisions",
                "Published before completion",
                "The file exists while the projection ledger remains pending.",
                "knowledge-promotion-adversarial",
                0.9,
            )
        )
        pending_job = pending_store.get_memory_projection_job(pending_target.memory_id)
        if pending_job is None:
            raise AssertionError("completion-lost fixture has no pending projection job")
        path, _digest = pending_vault.write_memory_projection_with_evidence(
            MemoryRecord(
                str(pending_job["category"]),
                str(pending_job["title"]),
                str(pending_job["body"]),
                str(pending_job["source"]),
                float(pending_job["confidence"]),
            ),
            memory_id=pending_target.memory_id,
            store_identity=str(pending_job["store_identity"]),
            memory_revision=pending_target.revision,
            source_digest=pending_target.source_digest,
            created_at=str(pending_job["memory_created_at"]),
        )
        if not path.is_file() or pending_store.get_memory_projection_job(
            pending_target.memory_id
        )["state"] != "pending":
            raise AssertionError("completion-lost fixture was not file-published and pending")
        _assert_decision_commit_rejected_without_effects(
            pending_store,
            pending_vault,
            pending_target.memory_id,
            _generic_target(pending_store, pending_target.memory_id),
            label="file-published completion-lost projection",
        )

        valid_store, valid_vault = _setup(root / "completed-valid")
        valid_target = valid_store.add_memory_with_projection(
            MemoryRecord(
                "decisions",
                "Completed projection",
                "Structurally valid completed custody remains reviewable.",
                "knowledge-promotion-adversarial",
                0.9,
            )
        )
        _complete_projection(valid_store, valid_vault, valid_target)
        valid_candidate = valid_store.resolve_knowledge_promotion_approval_target(
            valid_target.memory_id
        )
        packet = _handlers(valid_store, valid_vault)[0]
        valid_packet = packet({"memory_id": valid_target.memory_id})
        if valid_candidate is None or not valid_packet.ok:
            raise AssertionError(
                "a completed structurally valid projection job was not eligible"
            )

        corruptions = (
            ("malformed content digest", "content_digest", "malformed", True),
            ("incorrect source digest", "source_digest", None, False),
            ("foreign store identity", "store_identity", None, False),
            ("empty canonical path", "canonical_path_display", "", True),
            (
                "wrong projection revision",
                "memory_revision",
                valid_target.revision + 1,
                False,
            ),
        )
        for index, (label, column, configured_value, bypass_checks) in enumerate(
            corruptions
        ):
            store, vault = _setup(root / f"malformed-{index}")
            projection_target = store.add_memory_with_projection(
                MemoryRecord(
                    "decisions",
                    f"Malformed projection {index}",
                    "Malformed completed custody must fail closed.",
                    "knowledge-promotion-adversarial",
                    0.9,
                )
            )
            _complete_projection(store, vault, projection_target)
            exact_target = _generic_target(store, projection_target.memory_id)
            job = store.get_memory_projection_job(projection_target.memory_id)
            if job is None:
                raise AssertionError(f"{label} fixture had no completed job")
            value = configured_value
            if column == "source_digest":
                value = _different_hex(str(job["source_digest"]), 64)
            elif column == "store_identity":
                value = _different_hex(str(job["store_identity"]), 32)
            with store.connect() as conn:
                if bypass_checks:
                    conn.execute("PRAGMA ignore_check_constraints = ON")
                conn.execute(
                    f"UPDATE memory_projection_jobs SET {column} = ? WHERE memory_id = ?",
                    (value, projection_target.memory_id),
                )
            _assert_decision_commit_rejected_without_effects(
                store,
                vault,
                projection_target.memory_id,
                exact_target,
                label=label,
            )


def test_malformed_timestamps_fail_before_binding_and_commit() -> None:
    malformed_values = (
        "2026-07-13",
        "2026-07-13T01:02:03",
        "2026-07-13T01:02:03+09:00",
        "2026-13-01T00:00:00+00:00",
        "2026-02-30T01:02:03Z",
    )
    with TemporaryDirectory(prefix="jarvis-promotion-timestamps-") as temp:
        root = Path(temp)
        for index, (column, value) in enumerate(
            (column, value)
            for column in ("created_at", "updated_at")
            for value in malformed_values
        ):
            store, vault = _setup(root / str(index))
            memory_id = _seed(store, title=f"Malformed {column} {index}")
            with store.connect() as conn:
                conn.execute(
                    f"UPDATE memories SET {column} = ? WHERE id = ?",
                    (value, memory_id),
                )
            exact_target = _generic_target(store, memory_id)
            _assert_candidate_rejected_before_binding(
                store, memory_id, label=f"malformed {column} {value}"
            )
            _assert_decision_commit_rejected_without_effects(
                store,
                vault,
                memory_id,
                exact_target,
                label=f"malformed {column} {value}",
            )

        ordered_store, ordered_vault = _setup(root / "updated-before-created")
        ordered_id = _seed(ordered_store, title="Timestamp ordering")
        with ordered_store.connect() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
                (
                    "2026-07-13T02:00:00Z",
                    "2026-07-13T01:59:59Z",
                    ordered_id,
                ),
            )
        ordered_target = _generic_target(ordered_store, ordered_id)
        _assert_candidate_rejected_before_binding(
            ordered_store,
            ordered_id,
            label="updated_at earlier than created_at",
        )
        _assert_decision_commit_rejected_without_effects(
            ordered_store,
            ordered_vault,
            ordered_id,
            ordered_target,
            label="updated_at earlier than created_at",
        )


def _promote_preference(store: MemoryStore, memory_id: int) -> Any:
    candidate = store.resolve_knowledge_promotion_approval_target(memory_id)
    if candidate is None or candidate.target_kind != "preference":
        raise AssertionError("preference promotion fixture was not eligible")
    result = store.promote_memory_to_preference_exact(
        memory_id,
        candidate.revision,
        candidate.binding,
        PreferenceRecord(
            category="communication",
            key="adversarial response tone",
            value="direct and calm",
        ),
    )
    if (
        result.status != "promoted"
        or result.preference_id is None
        or result.memory_id != memory_id
        or result.preference_projection_target is None
        or result.memory_projection_target is None
        or re.fullmatch(r"[0-9a-f]{64}", result.source_integrity_binding or "")
        is None
    ):
        raise AssertionError(f"preference promotion fixture failed: {result!r}")
    return result


def test_preference_source_integrity_survives_legitimate_global_advance() -> None:
    corruptions = {
        "promoted custody graph": (
            "DELETE FROM preference_memory_links WHERE preference_id = ?",
            "preference_id",
        ),
        "promoted memory projection job": (
            "UPDATE memory_projection_jobs SET source_digest = ? WHERE memory_id = ?",
            "memory_job",
        ),
        "global preference projection job": (
            "UPDATE preference_projection_jobs SET source_digest = ? WHERE singleton_id = 1",
            "preference_job",
        ),
    }
    with TemporaryDirectory(prefix="jarvis-promotion-preference-integrity-") as temp:
        root = Path(temp)
        for index, (label, (statement, mode)) in enumerate(corruptions.items()):
            store, _vault = _setup(root / str(index))
            memory_id = _seed(
                store,
                category="preferences",
                title=f"Preference integrity {index}",
                body="A source preference awaits explicit review.",
            )
            promoted = _promote_preference(store, memory_id)
            preference_id = int(promoted.preference_id)
            preference_target = promoted.preference_projection_target
            memory_target = promoted.memory_projection_target
            if not store.preference_mutation_source_integrity(
                preference_id,
                memory_id,
                preference_target,
                memory_target,
                promoted.source_integrity_binding,
            ):
                raise AssertionError("fresh promoted preference graph failed source integrity")

            unrelated = store.set_preference_with_projections(
                PreferenceRecord(
                    category="display",
                    key=f"unrelated density {index}",
                    value="compact",
                )
            )
            if unrelated.preference_projection_target.generation <= preference_target.generation:
                raise AssertionError("unrelated preference did not advance global projection")
            if not store.preference_mutation_source_integrity(
                preference_id,
                memory_id,
                preference_target,
                memory_target,
                promoted.source_integrity_binding,
            ):
                raise AssertionError(
                    "legitimate unrelated preference advance invalidated source integrity"
                )

            with store.connect() as conn:
                if mode == "preference_id":
                    conn.execute(statement, (preference_id,))
                elif mode == "memory_job":
                    conn.execute(statement, ("f" * 64, memory_id))
                else:
                    conn.execute(statement, ("f" * 64,))
            if store.preference_mutation_source_integrity(
                preference_id,
                memory_id,
                preference_target,
                memory_target,
                promoted.source_integrity_binding,
            ):
                raise AssertionError(f"source integrity accepted corruption in {label}")


def test_forged_historical_projection_digests_fail_after_preference_supersession() -> None:
    with TemporaryDirectory(prefix="jarvis-promotion-historical-digest-") as temp:
        store, _vault = _setup(Path(temp))
        memory_id = _seed(
            store,
            category="preferences",
            title="Historical digest integrity",
            body="A source preference awaits explicit review.",
        )
        promoted = _promote_preference(store, memory_id)
        preference_id = int(promoted.preference_id)
        preference_target = promoted.preference_projection_target
        memory_target = promoted.memory_projection_target
        superseded = store.set_preference_with_projections(
            PreferenceRecord(
                category="communication",
                key="adversarial response tone",
                value="direct, calm, and concise",
            )
        )
        if (
            superseded.changed is not True
            or superseded.preference_id != preference_id
            or superseded.memory_id != memory_id
            or superseded.preference_projection_target.generation
            <= preference_target.generation
            or superseded.memory_projection_target.revision <= memory_target.revision
        ):
            raise AssertionError(
                f"legitimate set_preference did not supersede the promoted graph: {superseded!r}"
            )
        if not store.preference_mutation_source_integrity(
            preference_id,
            memory_id,
            preference_target,
            memory_target,
            promoted.source_integrity_binding,
        ):
            raise AssertionError("legitimate historical targets failed after supersession")

        forged_preference_target = PreferenceProjectionTarget(
            generation=preference_target.generation,
            source_digest=_different_hex(preference_target.source_digest, 64),
        )
        forged_memory_target = MemoryProjectionTarget(
            memory_id=memory_target.memory_id,
            revision=memory_target.revision,
            operation=memory_target.operation,
            source_digest=_different_hex(memory_target.source_digest, 64),
        )
        accepted_forgeries: list[str] = []
        for label, checked_preference_target, checked_memory_target in (
            ("historical preference digest", forged_preference_target, memory_target),
            ("historical memory digest", preference_target, forged_memory_target),
        ):
            if store.preference_mutation_source_integrity(
                preference_id,
                memory_id,
                checked_preference_target,
                checked_memory_target,
                promoted.source_integrity_binding,
            ):
                accepted_forgeries.append(label)
        if accepted_forgeries:
            raise AssertionError(
                f"source integrity accepted forged targets: {accepted_forgeries!r}"
            )

        foreign_store, _foreign_vault = _setup(Path(temp) / "foreign-store")
        foreign_memory_id = _seed(
            foreign_store,
            category="preferences",
            title="Historical digest integrity",
            body="A source preference awaits explicit review.",
        )
        foreign_promoted = _promote_preference(foreign_store, foreign_memory_id)
        if store.preference_mutation_source_integrity(
            preference_id,
            memory_id,
            preference_target,
            memory_target,
            foreign_promoted.source_integrity_binding,
        ):
            raise AssertionError("source integrity accepted another store's binding")


def _padded_value(prefix: str, suffix: str, length: int, fill: str) -> str:
    remaining = length - len(prefix) - len(suffix)
    if remaining < 0:
        raise AssertionError("packet fixture prefix exceeds its target length")
    return prefix + (fill * remaining) + suffix


def _packet_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("!", "\\!")
    )


def test_packet_is_complete_escaped_and_display_bounded() -> None:
    with TemporaryDirectory(prefix="jarvis-promotion-packet-") as temp:
        root = Path(temp)
        store, vault = _setup(root / "complete")
        category = "decisions"
        title = _padded_value(
            '![title](https://example.com/title.png) <img src="title.png"> ',
            "TITLE-END",
            MAX_PACKET_TITLE_CHARS,
            "t",
        )
        body = _padded_value(
            '![body](https://example.com/body.png) <iframe src="body.html"></iframe> ',
            "BODY-END",
            MAX_PACKET_BODY_CHARS,
            "b",
        )
        source = _padded_value(
            '![source](https://example.com/source.png) <img src="source.png"> ',
            "SOURCE-END",
            MAX_CANDIDATE_SOURCE_CHARS,
            "s",
        )
        created_at = "2026-07-01T01:02:03Z"
        updated_at = "2026-07-02T04:05:06Z"
        confidence = 0.875
        memory_id = _seed(
            store,
            category=category,
            title=title,
            body=body,
            source=source,
            confidence=confidence,
        )
        with store.connect() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
                (created_at, updated_at, memory_id),
            )
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        packet = _handlers(store, vault)[0]
        result = packet({"memory_id": memory_id})
        expected_lines = (
            f"Category label: {_packet_escape(category)}",
            f"Memory title: {_packet_escape(title)}",
            f"Memory body: {_packet_escape(body)}",
            f"Source: {_packet_escape(source)}",
            f"Confidence: {confidence:g}",
            f"Created: {created_at}",
            f"Updated: {updated_at}",
        )
        if (
            target is None
            or not result.ok
            or any(line not in result.output.splitlines() for line in expected_lines)
            or re.search(r"(?<!\\)!\[", result.output) is not None
            or "<img" in result.output.casefold()
            or "<iframe" in result.output.casefold()
            or "..." in result.output
        ):
            raise AssertionError(
                "eligible packet fields were truncated, omitted, or left render-active"
            )

        over_limit_cases: tuple[tuple[str, Any], ...] = (
            (
                "category",
                "decisions"
                + (" " * (MAX_PACKET_CATEGORY_CHARS - len("decisions") + 1)),
            ),
            ("title", "t" * (MAX_PACKET_TITLE_CHARS + 1)),
            ("body", "b" * (MAX_PACKET_BODY_CHARS + 1)),
            ("source", "s" * (MAX_CANDIDATE_SOURCE_CHARS + 1)),
            (
                "created_at",
                "2026-07-01T01:02:03Z"
                + (
                    " "
                    * (
                        MAX_CANDIDATE_TIMESTAMP_CHARS
                        - len("2026-07-01T01:02:03Z")
                        + 1
                    )
                ),
            ),
            (
                "updated_at",
                "2026-07-02T04:05:06Z"
                + (
                    " "
                    * (
                        MAX_CANDIDATE_TIMESTAMP_CHARS
                        - len("2026-07-02T04:05:06Z")
                        + 1
                    )
                ),
            ),
        )
        for index, (column, value) in enumerate(over_limit_cases):
            bounded_store, bounded_vault = _setup(root / f"over-{index}")
            bounded_id = _seed(
                bounded_store,
                title=f"Over-limit {column}",
            )
            with bounded_store.connect() as conn:
                conn.execute(
                    f"UPDATE memories SET {column} = ? WHERE id = ?",
                    (value, bounded_id),
                )
            exact_target = _generic_target(bounded_store, bounded_id)
            _assert_candidate_rejected_before_binding(
                bounded_store,
                bounded_id,
                label=f"over-display-limit {column}",
            )
            _assert_decision_commit_rejected_without_effects(
                bounded_store,
                bounded_vault,
                bounded_id,
                exact_target,
                label=f"over-display-limit {column}",
            )


def test_revision_headroom_is_preserved_before_binding_and_commit() -> None:
    expected_ceiling = SQLITE_MAX_INTEGER - REQUIRED_PROMOTION_HEADROOM
    if MAX_KNOWLEDGE_PROMOTION_SOURCE_REVISION != expected_ceiling:
        raise AssertionError(
            "knowledge promotion revision ceiling no longer preserves 1024 updates"
        )
    with TemporaryDirectory(prefix="jarvis-promotion-revision-headroom-") as temp:
        root = Path(temp)

        eligible_store, _eligible_vault = _setup(root / "eligible-boundary")
        eligible_id = _seed(eligible_store, title="Eligible revision boundary")
        with eligible_store.connect() as conn:
            conn.execute(
                "UPDATE memories SET revision = ? WHERE id = ?",
                (expected_ceiling, eligible_id),
            )
        eligible = eligible_store.resolve_knowledge_promotion_approval_target(eligible_id)
        if (
            eligible is None
            or SQLITE_MAX_INTEGER - eligible.revision < REQUIRED_PROMOTION_HEADROOM
        ):
            raise AssertionError("maximum eligible revision lost required update headroom")

        blocked_store, blocked_vault = _setup(root / "blocked-near-max")
        blocked_id = _seed(blocked_store, title="Blocked revision near SQLite max")
        with blocked_store.connect() as conn:
            conn.execute(
                "UPDATE memories SET revision = ? WHERE id = ?",
                (expected_ceiling + 1, blocked_id),
            )
        exact_target = _generic_target(blocked_store, blocked_id)
        _assert_candidate_rejected_before_binding(
            blocked_store,
            blocked_id,
            label="near-SQLite-max revision",
        )
        _assert_decision_commit_rejected_without_effects(
            blocked_store,
            blocked_vault,
            blocked_id,
            exact_target,
            label="near-SQLite-max revision",
        )


def _review_args(kind: str, target: Any) -> dict[str, Any]:
    common = {
        "memory_id": target.memory_id,
        "reviewed_revision": target.revision,
        "review_token": target.binding,
    }
    if kind == "decision":
        return {
            **common,
            "title": "Reviewer-bound decision",
            "rationale": "Every source field remains approval-bound.",
            "impact": "Stale approval material has no effect.",
        }
    return {
        **common,
        "category": "communication",
        "key": "reviewer-bound preference",
        "value": "direct and calm",
    }


def _exact_transfer(store: MemoryStore, kind: str, target: Any) -> Any:
    if kind == "decision":
        return store.promote_memory_to_decision_exact(
            target.memory_id,
            target.revision,
            target.binding,
            DecisionRecord(
                title="Reviewer-bound decision",
                rationale="Every source field remains approval-bound.",
                impact="Stale approval material has no effect.",
            ),
        )
    return store.promote_memory_to_preference_exact(
        target.memory_id,
        target.revision,
        target.binding,
        PreferenceRecord(
            category="communication",
            key="reviewer-bound preference",
            value="direct and calm",
        ),
    )


def _copy_bound_memory(
    source: MemoryStore,
    source_id: int,
    destination: MemoryStore,
    destination_id: int,
) -> None:
    columns = (
        "category",
        "title",
        "body",
        "source",
        "confidence",
        "revision",
        "created_at",
        "updated_at",
    )
    with source.connect() as conn:
        row = conn.execute(
            f"SELECT {', '.join(columns)} FROM memories WHERE id = ?",
            (source_id,),
        ).fetchone()
    if row is None:
        raise AssertionError("cross-store source memory disappeared")
    with destination.connect() as conn:
        conn.execute(
            "UPDATE memories SET "
            + ", ".join(f"{column} = ?" for column in columns)
            + " WHERE id = ?",
            (*tuple(row), destination_id),
        )


def test_cross_store_review_token_replay_has_zero_effects() -> None:
    with TemporaryDirectory(prefix="jarvis-promotion-cross-store-") as temp:
        root = Path(temp)
        for index, (kind, category) in enumerate(
            (("decision", "decisions"), ("preference", "preferences"))
        ):
            source_store, _source_vault = _setup(root / f"source-{index}")
            destination_store, destination_vault = _setup(
                root / f"destination-{index}"
            )
            source_id = _seed(
                source_store,
                category=category,
                title=f"Cross-store {kind}",
                body="The row is identical, but store identity is not.",
                source="cross-store-reviewer",
            )
            destination_id = _seed(
                destination_store,
                category=category,
                title="Temporary destination row",
                body="This row will receive the exact source snapshot.",
                source="cross-store-reviewer",
            )
            _copy_bound_memory(
                source_store,
                source_id,
                destination_store,
                destination_id,
            )
            source_target = source_store.resolve_knowledge_promotion_approval_target(
                source_id
            )
            destination_target = (
                destination_store.resolve_knowledge_promotion_approval_target(
                    destination_id
                )
            )
            if (
                source_target is None
                or destination_target is None
                or source_target.revision != destination_target.revision
                or source_target.binding == destination_target.binding
            ):
                raise AssertionError(
                    f"cross-store {kind} fixture was not identical except for binding"
                )

            resolver = _handlers(destination_store, destination_vault)[
                2 if kind == "decision" else 4
            ]
            before_database = _database_snapshot(destination_store)
            before_vault = _vault_snapshot(destination_vault)
            resolution = resolver(_review_args(kind, source_target))
            exact = _exact_transfer(destination_store, kind, source_target)
            reason = getattr(resolution, "metadata", {}).get(
                "approval_argument_resolution_status"
            )
            if (
                getattr(resolution, "args", None) is not None
                or reason != "review_token_stale"
                or _status(exact) != "stale"
                or _database_snapshot(destination_store) != before_database
                or _vault_snapshot(destination_vault) != before_vault
            ):
                raise AssertionError(
                    f"cross-store {kind} token replay was not a stale zero-effect refusal: "
                    f"resolver_reason={reason!r}, status={_status(exact)!r}"
                )


def test_same_revision_bound_field_drift_invalidates_both_promotions() -> None:
    baseline_created = "2026-07-13T01:00:00Z"
    baseline_updated = "2026-07-13T02:00:00Z"
    field_values = {
        "title": "Same-revision reviewer title changed",
        "body": "Same-revision reviewer body changed.",
        "source": "same-revision-reviewer-changed",
        "confidence": 0.876,
        "created_at": "2026-07-13T00:59:59Z",
        "updated_at": "2026-07-13T02:00:01Z",
    }
    with TemporaryDirectory(prefix="jarvis-promotion-same-revision-") as temp:
        root = Path(temp)
        fixture_index = 0
        for kind, category, changed_category in (
            ("decision", "decisions", "decision"),
            ("preference", "preferences", "preference"),
        ):
            for field in (
                "category",
                "title",
                "body",
                "source",
                "confidence",
                "created_at",
                "updated_at",
            ):
                store, vault = _setup(root / str(fixture_index))
                fixture_index += 1
                memory_id = _seed(
                    store,
                    category=category,
                    title=f"Bound {kind} {field}",
                    body="Every directly bound field must stale the review token.",
                    source="same-revision-reviewer",
                )
                with store.connect() as conn:
                    conn.execute(
                        "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
                        (baseline_created, baseline_updated, memory_id),
                    )
                reviewed = store.resolve_knowledge_promotion_approval_target(memory_id)
                if reviewed is None:
                    raise AssertionError(f"{kind} {field} fixture was not reviewable")
                changed_value = (
                    changed_category if field == "category" else field_values[field]
                )
                with store.connect() as conn:
                    conn.execute(
                        f"UPDATE memories SET {field} = ? WHERE id = ?",
                        (changed_value, memory_id),
                    )
                current = store.resolve_knowledge_promotion_approval_target(memory_id)
                if (
                    current is None
                    or current.revision != reviewed.revision
                    or current.binding == reviewed.binding
                ):
                    raise AssertionError(
                        f"same-revision {kind} {field} drift did not remain eligible "
                        "with a changed binding"
                    )

                resolver = _handlers(store, vault)[2 if kind == "decision" else 4]
                before_database = _database_snapshot(store)
                before_vault = _vault_snapshot(vault)
                resolution = resolver(_review_args(kind, reviewed))
                exact = _exact_transfer(store, kind, reviewed)
                reason = getattr(resolution, "metadata", {}).get(
                    "approval_argument_resolution_status"
                )
                if (
                    getattr(resolution, "args", None) is not None
                    or reason != "review_token_stale"
                    or _status(exact) != "stale"
                    or _database_snapshot(store) != before_database
                    or _vault_snapshot(vault) != before_vault
                ):
                    raise AssertionError(
                        f"same-revision {kind} {field} drift crossed approval or commit: "
                        f"resolver_reason={reason!r}, status={_status(exact)!r}"
                    )


def _resolution_args(target: Any, value: str) -> dict[str, Any]:
    return {
        "memory_id": target.memory_id,
        "reviewed_revision": target.revision,
        "review_token": target.binding,
        "category": "communication",
        "key": "adversarial slash value",
        "value": value,
    }


def test_preference_value_slashes_paths_and_embeds() -> None:
    benign_values = ("24/7", "yes/no", "input/output")
    local_paths = (
        "/\x55sers/the operator/Secret/preferences.txt",
        "~/Library/Application Support/Jarvis/preferences.md",
        "./local/preferences.md",
        "Secret/preferences.txt",
        r"Secret\preferences.txt",
        "C:Secret.txt",
        "file:///\x55sers/the operator/Secret/preferences.md",
        r"C:\Users\the operator\Secret\preferences.txt",
        r"C:Secret\preferences.txt",
        r"\Users\the operator\Secret.txt",
    )
    active_embeds = (
        "![tone](https://example.com/tone.png)",
        '<img src="https://example.com/tone.png">',
        '<iframe src="https://example.com/tone"></iframe>',
    )
    with TemporaryDirectory(prefix="jarvis-promotion-preference-values-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = _seed(
            store,
            category="preferences",
            title="Preference slash validation",
            body="Benign slash phrases are not local paths.",
        )
        target = store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None or target.target_kind != "preference":
            raise AssertionError("preference value fixture was not eligible")
        resolve_preference = _handlers(store, vault)[4]
        before_database = _database_snapshot(store)
        before_vault = _vault_snapshot(vault)

        for value in benign_values:
            resolution = resolve_preference(_resolution_args(target, value))
            if getattr(resolution, "args", {}).get("value") != value:
                raise AssertionError(f"benign preference value was refused: {value!r}")

        accepted_local_paths: list[str] = []
        for value in local_paths:
            refusal = resolve_preference(_resolution_args(target, value))
            reason = getattr(refusal, "metadata", {}).get(
                "approval_argument_resolution_status", ""
            )
            if getattr(refusal, "args", None) is not None or "local_path" not in reason:
                accepted_local_paths.append(value)

        for value in active_embeds:
            refusal = resolve_preference(_resolution_args(target, value))
            reason = getattr(refusal, "metadata", {}).get(
                "approval_argument_resolution_status", ""
            )
            if (
                getattr(refusal, "args", None) is not None
                or "active_render_content" not in reason
            ):
                raise AssertionError(f"active preference embed reached approval: {value!r}")

        if (
            _database_snapshot(store) != before_database
            or _vault_snapshot(vault) != before_vault
        ):
            raise AssertionError("preference value validation mutated temporary state")
        if accepted_local_paths:
            raise AssertionError(
                f"local preference paths reached approval: {accepted_local_paths!r}"
            )


def main() -> None:
    test_projection_job_custody_controls_eligibility()
    test_malformed_timestamps_fail_before_binding_and_commit()
    test_preference_source_integrity_survives_legitimate_global_advance()
    test_forged_historical_projection_digests_fail_after_preference_supersession()
    test_packet_is_complete_escaped_and_display_bounded()
    test_revision_headroom_is_preserved_before_binding_and_commit()
    test_cross_store_review_token_replay_has_zero_effects()
    test_same_revision_bound_field_drift_invalidates_both_promotions()
    test_preference_value_slashes_paths_and_embeds()
    print("Knowledge promotion adversarial smoke passed")


if __name__ == "__main__":
    main()
