from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    MemoryProjectionTarget,
    MemoryRecord,
    MemoryStore,
    memory_projection_source_digest,
    profile_note_source_key,
)


MAX_PROFILE_PROJECTION_RECOVERY_JOBS = 100
MAX_PROFILE_MEMORY_RECONCILE_PASSES = 4
MAX_PROFILE_WRITE_CHARS = 50_000
MAX_PROFILE_HEADING_CHARS = 120
MAX_PROFILE_CATEGORY_CHARS = 64
LOCAL_PATH_RE = re.compile(
    r"(?:(?<![:A-Za-z0-9])/(?:System/Volumes/Data/Users|Users|root|private|"
    r"var/(?:folders|tmp)|tmp|Volumes|home)/[^\n\r;]*"
    r"|~[/\\][^\n\r;]*|[A-Za-z]:[/\\][^\n\r;]*)",
    re.IGNORECASE,
)
PROFILE_MARKER_NAMESPACE_RE = re.compile(r"jarvis-profile-note\b", re.IGNORECASE)


@dataclass(frozen=True)
class ProfileProjectionOutcome:
    memory_id: int
    status: str
    file_written: bool = False


@dataclass(frozen=True)
class ProfileProjectionSummary:
    attempted: int
    completed: int
    pending: int
    outcomes: tuple[ProfileProjectionOutcome, ...]


@dataclass(frozen=True)
class _ProfileRecoveryCandidate:
    source_key: str
    memory_id: int
    revision: int
    source_digest: str
    mirror_generation: int
    record: MemoryRecord


def _candidate_from_row(row) -> _ProfileRecoveryCandidate | None:
    try:
        source_key = row["source_key"]
        source_type = row["source_type"]
        source_title = row["source_title"]
        memory_id = row["memory_id"]
        revision = row["revision"]
        category = row["category"]
        title = row["title"]
        body = row["body"]
        memory_source = row["source"]
        confidence = row["confidence"]
        created_at = row["created_at"]
        mirror_generation = row["mirror_generation"]
    except (KeyError, IndexError, TypeError):
        return None
    if (
        type(source_key) is not str
        or re.fullmatch(r"profile-note:v1:[0-9a-f]{64}", source_key) is None
        or source_type != "profile_note"
        or type(memory_id) is not int
        or memory_id < 1
        or type(revision) is not int
        or revision < 1
        or type(mirror_generation) is not int
        or mirror_generation < 0
        or any(
            type(value) is not str
            for value in (source_title, category, title, body, memory_source, created_at)
        )
        or source_title != title
        or memory_source != "profile"
        or type(confidence) not in {int, float}
        or isinstance(confidence, bool)
        or float(confidence) != 1.0
        or re.match(r"^\d{4}-\d{2}-\d{2}", created_at) is None
        or "\r" in created_at
        or "\n" in created_at
        or profile_note_source_key(title, body, category) != source_key
    ):
        return None
    canonical_title = " ".join(unicodedata.normalize("NFKC", title).split())
    canonical_category = " ".join(
        unicodedata.normalize("NFKC", category).casefold().split()
    )
    canonical_body = (
        unicodedata.normalize("NFKC", body)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
    if (
        not canonical_body
        or len(canonical_title) > MAX_PROFILE_HEADING_CHARS
        or len(canonical_category) > MAX_PROFILE_CATEGORY_CHARS
        or len(canonical_body) > MAX_PROFILE_WRITE_CHARS
        or canonical_title != title
        or canonical_category != category
        or canonical_body != body
        or any(LOCAL_PATH_RE.search(value) for value in (title, category, body))
        or any(
            PROFILE_MARKER_NAMESPACE_RE.search(value)
            for value in (title, category, body)
        )
    ):
        return None
    try:
        source_digest = memory_projection_source_digest(
            memory_id,
            revision,
            category,
            title,
            body,
            memory_source,
            confidence,
            created_at,
            "publish",
        )
        record = MemoryRecord(
            category=category,
            title=title,
            body=body,
            source=memory_source,
            confidence=float(confidence),
        )
    except (TypeError, ValueError):
        return None
    return _ProfileRecoveryCandidate(
        source_key=source_key,
        memory_id=memory_id,
        revision=revision,
        source_digest=source_digest,
        mirror_generation=mirror_generation,
        record=record,
    )


def _keep_pending(
    store: MemoryStore,
    candidate: _ProfileRecoveryCandidate,
    target: MemoryProjectionTarget,
) -> None:
    try:
        store.mark_ingested_source_projection_pending(
            candidate.source_key,
            candidate.memory_id,
            target.revision,
            target.source_digest,
            expected_generation=target.attempt_generation,
        )
    except Exception:
        pass


def reconcile_profile_projection_candidate(
    store: MemoryStore,
    vault: ObsidianVault,
    row,
) -> ProfileProjectionOutcome:
    """Recover one exact source-authoritative profile projection without user replay."""
    candidate = _candidate_from_row(row)
    if candidate is None:
        memory_id = 0
        try:
            if type(row["memory_id"]) is int and row["memory_id"] > 0:
                memory_id = int(row["memory_id"])
        except (KeyError, IndexError, TypeError):
            pass
        return ProfileProjectionOutcome(memory_id, "malformed")

    try:
        if store.profile_projection_has_unresolved_auto_mutation(
            heading=candidate.record.title,
            body=candidate.record.body,
            category=candidate.record.category,
        ):
            return ProfileProjectionOutcome(
                candidate.memory_id,
                "blocked_uncertain_mutation",
            )
    except Exception:
        return ProfileProjectionOutcome(candidate.memory_id, "pending_error")

    try:
        target = store.prepare_profile_projection_recovery(
            candidate.source_key,
            candidate.memory_id,
            candidate.revision,
            candidate.source_digest,
            candidate.mirror_generation,
        )
    except Exception:
        return ProfileProjectionOutcome(candidate.memory_id, "pending_error")
    if target is None:
        return ProfileProjectionOutcome(candidate.memory_id, "superseded")

    projection = None
    for _ in range(MAX_PROFILE_MEMORY_RECONCILE_PASSES):
        projection = reconcile_memory_projection(
            store,
            vault,
            candidate.memory_id,
            expected_operation=target.operation,
            expected_revision=target.revision,
            expected_source_digest=target.source_digest,
        )
        if projection.status == "completed":
            break
    if (
        projection is None
        or projection.status != "completed"
        or not projection.path_display
        or not projection.content_digest
    ):
        _keep_pending(store, candidate, target)
        return ProfileProjectionOutcome(
            candidate.memory_id,
            "pending_error",
            bool(projection and projection.file_written),
        )

    profile_written = False
    try:
        with store.profile_memory_ownership_egress_fence(()):
            legacy_candidates = tuple(
                recovered
                for source_row in store.list_profile_projection_sources(
                    limit=MAX_PROFILE_PROJECTION_RECOVERY_JOBS
                )
                if (recovered := _candidate_from_row(source_row)) is not None
            )
            legacy_written = vault.upgrade_profile_legacy_blocks(
                tuple(
                    (
                        recovered.source_key,
                        recovered.record.title,
                        recovered.record.body,
                    )
                    for recovered in legacy_candidates
                )
            )
            _path, appended = vault.append_profile_once(
                candidate.source_key,
                candidate.record.title,
                candidate.record.body,
            )
            profile_written = legacy_written or appended
            with (
                vault.canonical_profile_note_evidence_lock(
                    source_key=candidate.source_key,
                    heading=candidate.record.title,
                    body=candidate.record.body,
                ) as profile_evidence_current,
            ):
                if not profile_evidence_current:
                    raise RuntimeError("profile projection evidence is unavailable")
                with vault.canonical_memory_projection_evidence_lock(
                    memory_id=candidate.memory_id,
                    store_identity=store.get_store_identity(),
                    expected_relative_path=projection.path_display,
                    expected_content_digest=projection.content_digest,
                ) as memory_evidence_current:
                    if not memory_evidence_current:
                        raise RuntimeError("profile memory projection evidence is unavailable")
                    store.complete_ingested_source_projection(
                        candidate.source_key,
                        candidate.memory_id,
                        target.revision,
                        target.source_digest,
                        expected_generation=target.attempt_generation,
                    )
    except Exception:
        _keep_pending(store, candidate, target)
        return ProfileProjectionOutcome(
            candidate.memory_id,
            "pending_error",
            bool(profile_written or projection.file_written),
        )
    return ProfileProjectionOutcome(
        candidate.memory_id,
        "completed",
        bool(profile_written or projection.file_written),
    )


def reconcile_pending_profile_projections(
    store: MemoryStore,
    vault: ObsidianVault,
    *,
    limit: int = 20,
) -> ProfileProjectionSummary:
    if type(limit) is not int:
        raise TypeError("profile projection reconciliation limit must be an integer")
    limit = max(1, min(limit, MAX_PROFILE_PROJECTION_RECOVERY_JOBS))
    rows = store.list_pending_profile_projection_sources(
        limit=MAX_PROFILE_PROJECTION_RECOVERY_JOBS
    )
    valid_rows = [row for row in rows if _candidate_from_row(row) is not None]
    malformed_rows = [row for row in rows if _candidate_from_row(row) is None]
    selected_rows = (valid_rows + malformed_rows)[:limit]
    outcomes = tuple(
        reconcile_profile_projection_candidate(store, vault, row)
        for row in selected_rows
    )
    return ProfileProjectionSummary(
        attempted=len(outcomes),
        completed=sum(outcome.status == "completed" for outcome in outcomes),
        pending=store.count_pending_profile_projection_sources(),
        outcomes=outcomes,
    )
