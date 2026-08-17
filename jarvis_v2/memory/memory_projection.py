from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    MemoryRecord,
    MemoryStore,
    memory_projection_source_digest,
)


MAX_MEMORY_PROJECTION_RECONCILE_JOBS = 100
MAX_MEMORY_PROJECTION_RECONCILE_PASSES = 4


@dataclass(frozen=True)
class MemoryProjectionOutcome:
    memory_id: int
    operation: str
    status: str
    path_display: str = ""
    content_digest: str = ""
    completion_status: str = ""
    file_written: bool = False


@dataclass(frozen=True)
class MemoryProjectionSummary:
    attempted: int
    completed: int
    pending: int
    outcomes: tuple[MemoryProjectionOutcome, ...]


def _path_display(vault: ObsidianVault, path) -> str:
    try:
        return str(path.relative_to(vault.root_path))
    except (AttributeError, ValueError):
        return ""


def _valid_job_identity(row) -> tuple[int, int, str, str, str] | None:
    try:
        memory_id = row["memory_id"]
        revision = row["memory_revision"]
        operation = row["operation"]
        state = row["state"]
        store_identity = row["store_identity"]
        source_digest = row["source_digest"]
        legacy_category = row["legacy_category"]
        legacy_title = row["legacy_title"]
        prior_content_digest = row["prior_content_digest"]
    except (KeyError, IndexError, TypeError):
        return None
    if (
        type(memory_id) is not int
        or memory_id < 1
        or type(revision) is not int
        or revision < 1
        or operation not in {"publish", "delete"}
        or state not in {"pending", "completed"}
        or type(store_identity) is not str
        or re.fullmatch(r"[0-9a-f]{32}", store_identity) is None
        or type(source_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None
        or (legacy_category is None) != (legacy_title is None)
        or (legacy_category is not None and type(legacy_category) is not str)
        or (legacy_title is not None and type(legacy_title) is not str)
        or (
            prior_content_digest is not None
            and (
                type(prior_content_digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", prior_content_digest) is None
            )
        )
    ):
        return None
    return memory_id, revision, operation, store_identity, source_digest


def _valid_publish_snapshot(snapshot) -> tuple[MemoryRecord, str] | None:
    try:
        category = snapshot["category"]
        title = snapshot["title"]
        body = snapshot["body"]
        source = snapshot["source"]
        confidence = snapshot["confidence"]
        try:
            created_at = snapshot["memory_created_at"]
        except (KeyError, IndexError):
            created_at = snapshot["created_at"]
    except (KeyError, IndexError, TypeError):
        return None
    if (
        any(type(value) is not str for value in (category, title, body, source))
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or type(created_at) is not str
        or re.match(r"^\d{4}-\d{2}-\d{2}", created_at) is None
        or "\r" in created_at
        or "\n" in created_at
    ):
        return None
    return (
        MemoryRecord(
            category=category,
            title=title,
            body=body,
            source=source,
            confidence=float(confidence),
        ),
        created_at,
    )


def _mark_error(
    store: MemoryStore,
    memory_id: int,
    operation: str,
    revision: int,
    source_digest: str,
    error_code: str,
) -> None:
    store.mark_memory_projection_error(
        memory_id,
        operation,
        revision,
        source_digest,
        error_code,
    )


def _get_publish_snapshot(
    store: MemoryStore,
    memory_id: int,
    revision: int,
    source_digest: str,
):
    getter = getattr(store, "get_memory_projection_publish_snapshot", None)
    if getter is None:
        getter = getattr(store, "get_current_memory_projection_publish_snapshot")
    return getter(memory_id, revision, source_digest)


def reconcile_memory_projection(
    store: MemoryStore,
    vault: ObsidianVault,
    memory_id: int,
    *,
    expected_operation: str | None = None,
    expected_revision: int | None = None,
    expected_source_digest: str | None = None,
    effect_authority: Callable[[], None] | None = None,
) -> MemoryProjectionOutcome:
    if type(memory_id) is not int or memory_id < 1:
        raise ValueError("memory projection id must be a positive integer")
    expected_values = (expected_operation, expected_revision, expected_source_digest)
    if any(value is not None for value in expected_values):
        if (
            expected_operation not in {"publish", "delete"}
            or type(expected_revision) is not int
            or expected_revision < 1
            or type(expected_source_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_source_digest) is None
        ):
            raise ValueError("expected memory projection identity is incomplete or invalid")
        expected_identity = (
            memory_id,
            expected_revision,
            expected_operation,
            expected_source_digest,
        )
    else:
        expected_identity = None

    target_superseded = False
    repair_current = False
    reopened_repair_identity: tuple[int, int, str, str] | None = None
    for _ in range(MAX_MEMORY_PROJECTION_RECONCILE_PASSES):
        job = store.get_memory_projection_job(memory_id)
        if job is None:
            return MemoryProjectionOutcome(memory_id, "unknown", "missing")
        identity = _valid_job_identity(job)
        if identity is None:
            return MemoryProjectionOutcome(memory_id, "unknown", "malformed")
        current_id, revision, operation, store_identity, source_digest = identity
        current_target = (current_id, revision, operation, source_digest)
        if expected_identity is not None and current_target != expected_identity:
            target_superseded = True
            if not repair_current:
                return MemoryProjectionOutcome(current_id, operation, "superseded")
        if repair_current:
            expected_identity = None
        if job["state"] == "completed" and operation == "delete":
            return MemoryProjectionOutcome(
                current_id,
                operation,
                "superseded" if target_superseded else "completed",
                str(job["canonical_path_display"] or ""),
                str(job["content_digest"] or ""),
                str(job["completion_status"] or ""),
            )

        if job["state"] == "completed":
            try:
                retained_path = job["canonical_path_display"]
                retained_digest = job["content_digest"]
            except (KeyError, IndexError, TypeError):
                return MemoryProjectionOutcome(current_id, operation, "malformed")
            if (
                type(retained_path) is not str
                or not retained_path
                or type(retained_digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", retained_digest) is None
            ):
                return MemoryProjectionOutcome(current_id, operation, "malformed")
            if effect_authority is not None:
                effect_authority()
            try:
                verified = vault.verify_memory_projection_evidence(
                    memory_id=current_id,
                    store_identity=store_identity,
                    expected_relative_path=retained_path,
                    expected_content_digest=retained_digest,
                )
            except Exception:
                verified = False
            if verified:
                return MemoryProjectionOutcome(
                    current_id,
                    operation,
                    "superseded" if target_superseded else "completed",
                    retained_path,
                    retained_digest,
                    "verified",
                )
            if store.reopen_completed_memory_projection(
                current_id,
                revision,
                source_digest,
                retained_digest,
            ):
                reopened_repair_identity = current_target
                continue
            latest = store.get_memory_projection_job(current_id)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and latest_identity != identity:
                target_superseded = True
                repair_current = True
                continue
            if latest_identity is not None and latest["state"] == "pending":
                reopened_repair_identity = current_target
                continue
            if latest_identity is not None and latest["state"] == "completed":
                continue
            return MemoryProjectionOutcome(current_id, operation, "pending_error")

        if operation == "publish":
            snapshot = _get_publish_snapshot(
                store,
                current_id,
                revision,
                source_digest,
            )
            if snapshot is None:
                latest = store.get_memory_projection_job(current_id)
                latest_identity = _valid_job_identity(latest)
                if latest_identity is not None and latest_identity != identity:
                    target_superseded = True
                    repair_current = True
                    continue
                _mark_error(
                    store,
                    current_id,
                    operation,
                    revision,
                    source_digest,
                    "source_snapshot_unavailable",
                )
                return MemoryProjectionOutcome(current_id, operation, "pending_error")
            validated = _valid_publish_snapshot(snapshot)
            if validated is None:
                _mark_error(
                    store,
                    current_id,
                    operation,
                    revision,
                    source_digest,
                    "source_snapshot_malformed",
                )
                return MemoryProjectionOutcome(current_id, operation, "pending_error")
            record, created_at = validated
            try:
                computed = memory_projection_source_digest(
                    current_id,
                    revision,
                    record.category,
                    record.title,
                    record.body,
                    record.source,
                    record.confidence,
                    created_at,
                    "publish",
                )
            except (TypeError, ValueError):
                computed = ""
            if computed != source_digest:
                _mark_error(
                    store,
                    current_id,
                    operation,
                    revision,
                    source_digest,
                    "source_digest_mismatch",
                )
                return MemoryProjectionOutcome(current_id, operation, "pending_error")
            try:
                expected_path, expected_content_digest = (
                    vault.expected_memory_projection_evidence(
                        record,
                        memory_id=current_id,
                        store_identity=store_identity,
                        memory_revision=revision,
                        source_digest=source_digest,
                        created_at=created_at,
                    )
                )
            except (TypeError, ValueError):
                _mark_error(
                    store,
                    current_id,
                    operation,
                    revision,
                    source_digest,
                    "projection_render_invalid",
                )
                return MemoryProjectionOutcome(current_id, operation, "pending_error")
            if effect_authority is not None:
                effect_authority()
            try:
                path, content_digest = vault.write_memory_projection_with_evidence(
                    record,
                    memory_id=current_id,
                    store_identity=store_identity,
                    memory_revision=revision,
                    source_digest=source_digest,
                    created_at=created_at,
                    expected_prior_content_digest=job["prior_content_digest"],
                )
            except Exception:
                latest = store.get_memory_projection_job(current_id)
                latest_identity = _valid_job_identity(latest)
                if latest_identity is not None and latest_identity != identity:
                    target_superseded = True
                    repair_current = True
                    continue
                _mark_error(
                    store,
                    current_id,
                    operation,
                    revision,
                    source_digest,
                    "vault_publish_failed",
                )
                expected_display = _path_display(vault, expected_path)
                try:
                    durable_bytes_present = bool(
                        expected_display
                        and vault.verify_memory_projection_evidence(
                            memory_id=current_id,
                            store_identity=store_identity,
                            expected_relative_path=expected_display,
                            expected_content_digest=expected_content_digest,
                        )
                    )
                except Exception:
                    durable_bytes_present = False
                return MemoryProjectionOutcome(
                    current_id,
                    operation,
                    "pending_error",
                    expected_display if durable_bytes_present else "",
                    expected_content_digest if durable_bytes_present else "",
                    file_written=durable_bytes_present,
                )
            display = _path_display(vault, path)
            if not display:
                _mark_error(
                    store,
                    current_id,
                    operation,
                    revision,
                    source_digest,
                    "path_display_unavailable",
                )
                return MemoryProjectionOutcome(
                    current_id,
                    operation,
                    "pending_error",
                    content_digest=content_digest,
                    file_written=True,
                )
            if effect_authority is not None:
                effect_authority()
            try:
                legacy_cleanup = vault.delete_legacy_memory_projection(
                    record=record,
                    memory_id=current_id,
                    store_identity=store_identity,
                    legacy_category=job["legacy_category"],
                    legacy_title=job["legacy_title"],
                )
            except Exception:
                _mark_error(
                    store,
                    current_id,
                    operation,
                    revision,
                    source_digest,
                    "legacy_cleanup_failed",
                )
                return MemoryProjectionOutcome(
                    current_id,
                    operation,
                    "pending_error",
                    display,
                    content_digest,
                    file_written=True,
                )
            if legacy_cleanup.status in {"error", "ownership_mismatch"}:
                _mark_error(
                    store,
                    current_id,
                    operation,
                    revision,
                    source_digest,
                    (
                        "legacy_cleanup_ownership_mismatch"
                        if legacy_cleanup.status == "ownership_mismatch"
                        else "legacy_cleanup_failed"
                    ),
                )
                return MemoryProjectionOutcome(
                    current_id,
                    operation,
                    "pending_error",
                    display,
                    content_digest,
                    file_written=True,
                )
            if effect_authority is not None:
                effect_authority()
            try:
                with vault.canonical_memory_projection_evidence_lock(
                    memory_id=current_id,
                    store_identity=store_identity,
                    expected_relative_path=display,
                    expected_content_digest=content_digest,
                ) as verified:
                    if not verified:
                        _mark_error(
                            store,
                            current_id,
                            operation,
                            revision,
                            source_digest,
                            "canonical_evidence_mismatch",
                        )
                        return MemoryProjectionOutcome(
                            current_id,
                            operation,
                            "pending_error",
                            display,
                            content_digest,
                            file_written=True,
                        )
                    if store.complete_memory_projection_publish(
                        current_id,
                        revision,
                        source_digest,
                        display,
                        content_digest,
                    ):
                        return MemoryProjectionOutcome(
                            current_id,
                            operation,
                            "superseded" if target_superseded else "completed",
                            display,
                            content_digest,
                            (
                                "repaired"
                                if reopened_repair_identity == current_target
                                else "published"
                            ),
                        )
            except (OSError, TimeoutError, ValueError):
                _mark_error(
                    store,
                    current_id,
                    operation,
                    revision,
                    source_digest,
                    "canonical_evidence_verification_failed",
                )
                return MemoryProjectionOutcome(
                    current_id,
                    operation,
                    "pending_error",
                    display,
                    content_digest,
                    file_written=True,
                )
            target_superseded = True
            repair_current = True
            continue

        try:
            deleted = vault.delete_memory_projection(
                memory_id=current_id,
                store_identity=store_identity,
                legacy_category=job["legacy_category"],
                legacy_title=job["legacy_title"],
                expected_content_digest=job["prior_content_digest"],
            )
        except Exception:
            deleted = None
        if deleted is None or deleted.status == "error":
            _mark_error(
                store,
                current_id,
                operation,
                revision,
                source_digest,
                "vault_delete_failed",
            )
            return MemoryProjectionOutcome(current_id, operation, "pending_error")
        if deleted.status == "ownership_mismatch":
            _mark_error(
                store,
                current_id,
                operation,
                revision,
                source_digest,
                "delete_ownership_mismatch",
            )
            return MemoryProjectionOutcome(current_id, operation, "pending_error")
        if store.complete_memory_projection_delete(
            current_id,
            revision,
            source_digest,
            deleted.status,
            deleted.path_display or None,
        ):
            return MemoryProjectionOutcome(
                current_id,
                operation,
                "superseded" if target_superseded else "completed",
                deleted.path_display,
                "",
                deleted.status,
            )
        target_superseded = True
        repair_current = True

    job = store.get_memory_projection_job(memory_id)
    operation = str(job["operation"]) if job is not None else "unknown"
    return MemoryProjectionOutcome(memory_id, operation, "superseded_or_busy")


def reconcile_pending_memory_projections(
    store: MemoryStore,
    vault: ObsidianVault,
    *,
    limit: int = 20,
) -> MemoryProjectionSummary:
    if type(limit) is not int:
        raise TypeError("memory projection reconciliation limit must be an integer")
    limit = max(1, min(limit, MAX_MEMORY_PROJECTION_RECONCILE_JOBS))
    jobs = store.list_pending_memory_projection_jobs(limit=limit)
    outcomes = tuple(
        reconcile_memory_projection(store, vault, int(job["memory_id"]))
        for job in jobs
    )
    completed = sum(outcome.status == "completed" for outcome in outcomes)
    return MemoryProjectionSummary(
        attempted=len(outcomes),
        completed=completed,
        pending=store.count_pending_memory_projection_jobs(),
        outcomes=outcomes,
    )
