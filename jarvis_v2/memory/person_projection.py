from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    MemoryStore,
    PersonProjectionTarget,
    person_projection_source_digest,
)


MAX_PERSON_PROJECTION_RECONCILE_JOBS = 100
MAX_PERSON_PROJECTION_RECONCILE_PASSES = 4


@dataclass(frozen=True)
class PersonProjectionOutcome:
    person_id: int
    status: str
    path_display: str = ""
    content_digest: str = ""


@dataclass(frozen=True)
class PersonProjectionSummary:
    attempted: int
    completed: int
    pending: int
    outcomes: tuple[PersonProjectionOutcome, ...]


def _opaque_path_display(person_id: int) -> str:
    return f"People/person-{person_id}.md"


def _valid_target(target: PersonProjectionTarget) -> tuple[int, int, int, str] | None:
    try:
        person_id = target.person_id
        revision = target.revision
        interaction_high_watermark = target.interaction_high_watermark
        source_digest = target.source_digest
    except AttributeError:
        return None
    if (
        type(person_id) is not int
        or person_id < 1
        or type(revision) is not int
        or revision < 0
        or type(interaction_high_watermark) is not int
        or interaction_high_watermark < 0
        or type(source_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None
    ):
        return None
    return person_id, revision, interaction_high_watermark, source_digest


def _valid_job_identity(row) -> tuple[int, int, int, str, str] | None:
    try:
        person_id = row["person_id"]
        revision = row["person_revision"]
        interaction_high_watermark = row["interaction_high_watermark"]
        store_identity = row["store_identity"]
        source_digest = row["source_digest"]
        state = row["state"]
    except (KeyError, IndexError, TypeError):
        return None
    if (
        type(person_id) is not int
        or person_id < 1
        or type(revision) is not int
        or revision < 0
        or type(interaction_high_watermark) is not int
        or interaction_high_watermark < 0
        or type(store_identity) is not str
        or re.fullmatch(r"[0-9a-f]{32}", store_identity) is None
        or type(source_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None
        or state not in {"pending", "completed"}
    ):
        return None
    return person_id, revision, interaction_high_watermark, store_identity, source_digest


def _rotate_malformed_job(store: MemoryStore, row) -> None:
    try:
        person_id = row["person_id"]
    except (KeyError, IndexError, TypeError):
        return
    if type(person_id) is not int or person_id < 1:
        return
    try:
        store.rotate_malformed_person_projection_job(person_id)
    except (TypeError, ValueError):
        return


def _reconcile_person_projection_delete(
    store: MemoryStore,
    vault: ObsidianVault,
    row,
) -> PersonProjectionOutcome | None:
    try:
        person_id = row["person_id"]
        store_identity = row["store_identity"]
        prior_content_digest = row["prior_content_digest"]
        state = row["state"]
    except (KeyError, IndexError, TypeError):
        return None
    if (
        type(person_id) is not int
        or person_id < 1
        or type(store_identity) is not str
        or re.fullmatch(r"[0-9a-f]{32}", store_identity) is None
        or state != "pending"
        or (
            prior_content_digest is not None
            and (
                type(prior_content_digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", prior_content_digest) is None
            )
        )
    ):
        if type(person_id) is int and person_id > 0:
            try:
                store.mark_person_projection_delete_error(person_id, "malformed_job")
            except (TypeError, ValueError):
                pass
            return PersonProjectionOutcome(person_id, "pending_error")
        return None
    try:
        result = vault.delete_person_projection(
            person_id=person_id,
            store_identity=store_identity,
            expected_content_digest=prior_content_digest,
        )
    except Exception:
        result = None
    status = getattr(result, "status", "error")
    if status not in {"deleted", "absent", "ownership_mismatch"}:
        store.mark_person_projection_delete_error(person_id, "vault_delete_failed")
        return PersonProjectionOutcome(person_id, "pending_error")
    if not store.complete_person_projection_delete(person_id, status):
        return PersonProjectionOutcome(person_id, "superseded_or_busy")
    return PersonProjectionOutcome(person_id, "completed")


def _reconcile_reopened_person_delete(
    store: MemoryStore,
    vault: ObsidianVault,
    person_id: int,
) -> PersonProjectionOutcome | None:
    if not store.reopen_person_projection_delete(person_id):
        return None
    row = store.get_person_projection_delete_job(person_id)
    if row is None or row["state"] != "pending":
        return PersonProjectionOutcome(person_id, "pending_error")
    return _reconcile_person_projection_delete(store, vault, row)


def _snapshot_parts(snapshot) -> tuple[object, Sequence] | None:
    if isinstance(snapshot, tuple) and len(snapshot) == 2:
        person, interactions = snapshot
    elif isinstance(snapshot, Mapping):
        try:
            person = snapshot["person"]
            interactions = snapshot["interactions"]
        except KeyError:
            return None
    else:
        try:
            person = snapshot.person
            interactions = snapshot.interactions
        except AttributeError:
            return None
    if isinstance(interactions, (str, bytes)) or not isinstance(interactions, Sequence):
        return None
    return person, interactions


def _valid_snapshot(
    snapshot,
    identity: tuple[int, int, int, str, str],
) -> tuple[object, tuple[object, ...], tuple[str, ...]] | None:
    parts = _snapshot_parts(snapshot)
    if parts is None:
        return None
    person, raw_interactions = parts
    person_id, revision, interaction_high_watermark, _store_identity, _source_digest = identity
    try:
        fields = {
            "id": person["id"],
            "name": person["name"],
            "relation": person["relation"],
            "notes": person["notes"],
            "last_contact_at": person["last_contact_at"],
            "revision": person["revision"],
            "created_at": person["created_at"],
            "updated_at": person["updated_at"],
        }
    except (KeyError, IndexError, TypeError):
        return None
    if (
        fields["id"] != person_id
        or type(fields["id"]) is not int
        or fields["revision"] != revision
        or type(fields["revision"]) is not int
        or any(
            type(fields[key]) is not str
            for key in ("name", "relation", "notes", "created_at", "updated_at")
        )
        or (fields["last_contact_at"] is not None and type(fields["last_contact_at"]) is not str)
    ):
        return None

    interactions: list[object] = []
    rendered_interactions: list[str] = []
    for interaction in raw_interactions:
        try:
            interaction_id = interaction["id"]
            interaction_person_id = interaction["person_id"]
            summary = interaction["summary"]
            happened_at = interaction["happened_at"]
            created_at = interaction["created_at"]
        except (KeyError, IndexError, TypeError):
            return None
        if (
            type(interaction_id) is not int
            or interaction_id < 1
            or interaction_id > interaction_high_watermark
            or interaction_person_id != person_id
            or type(interaction_person_id) is not int
            or any(type(value) is not str for value in (summary, happened_at, created_at))
        ):
            return None
        interactions.append(interaction)
        rendered_interactions.append(f"- {happened_at}: {summary}")
    if not rendered_interactions:
        rendered_interactions.append("- No interactions captured.")
    return person, tuple(interactions), tuple(rendered_interactions)


def _mark_error(
    store: MemoryStore,
    identity: tuple[int, int, int, str, str],
    error_code: str,
) -> None:
    person_id, revision, interaction_high_watermark, _store_identity, source_digest = identity
    store.mark_person_projection_error(
        person_id,
        revision,
        interaction_high_watermark,
        source_digest,
        error_code,
    )


def reconcile_person_projection(
    store: MemoryStore,
    vault: ObsidianVault,
    person_id: int | PersonProjectionTarget,
    target: PersonProjectionTarget | None = None,
) -> PersonProjectionOutcome:
    if isinstance(person_id, PersonProjectionTarget):
        if target is not None:
            raise ValueError("person projection target was provided twice")
        target = person_id
        person_id = target.person_id
    if type(person_id) is not int or person_id < 1:
        raise ValueError("person projection id must be a positive integer")
    expected_identity = None
    if target is not None:
        expected_identity = _valid_target(target)
        if expected_identity is None or expected_identity[0] != person_id:
            raise ValueError("person projection target is invalid or belongs to another person")

    target_superseded = False
    for _ in range(MAX_PERSON_PROJECTION_RECONCILE_PASSES):
        job = store.get_person_projection_job(person_id)
        if job is None:
            try:
                store.ensure_current_person_projection_job(person_id)
            except ValueError:
                deleted = _reconcile_reopened_person_delete(store, vault, person_id)
                return deleted or PersonProjectionOutcome(person_id, "missing")
            job = store.get_person_projection_job(person_id)
            if job is None:
                return PersonProjectionOutcome(person_id, "missing")
        identity = _valid_job_identity(job)
        if identity is None:
            _rotate_malformed_job(store, job)
            return PersonProjectionOutcome(person_id, "malformed")
        current_id, revision, interaction_high_watermark, store_identity, source_digest = identity
        current_target = (current_id, revision, interaction_high_watermark, source_digest)
        if expected_identity is not None and current_target != expected_identity:
            target_superseded = True
            expected_identity = None

        if job["state"] == "completed":
            content_digest = job["content_digest"]
            if (
                job["path_display"] != _opaque_path_display(current_id)
                or type(content_digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", content_digest) is None
            ):
                return PersonProjectionOutcome(current_id, "malformed")
            snapshot = store.get_current_person_projection_snapshot(
                current_id,
                revision,
                interaction_high_watermark,
                source_digest,
            )
            validated = _valid_snapshot(snapshot, identity) if snapshot is not None else None
            if validated is None:
                return PersonProjectionOutcome(current_id, "completed_audit_error")
            person, _interactions, _rendered = validated
            try:
                verified = vault.verify_person_projection_evidence(
                    person_id=current_id,
                    person_name=person["name"],
                    store_identity=store_identity,
                    expected_content_digest=content_digest,
                )
            except Exception:
                verified = False
            current_target_object = PersonProjectionTarget(
                person_id=current_id,
                revision=revision,
                interaction_high_watermark=interaction_high_watermark,
                source_digest=source_digest,
            )
            if verified:
                store.mark_person_projection_audited(
                    current_target_object, content_digest
                )
                return PersonProjectionOutcome(
                    current_id,
                    "superseded" if target_superseded else "completed",
                    _opaque_path_display(current_id),
                    content_digest,
                )
            if store.reopen_completed_person_projection(
                current_target_object, content_digest
            ):
                continue
            latest = store.get_person_projection_job(current_id)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and latest_identity != identity:
                target_superseded = True
                continue
            return PersonProjectionOutcome(current_id, "completed_audit_error")

        snapshot = store.get_current_person_projection_snapshot(
            current_id,
            revision,
            interaction_high_watermark,
            source_digest,
        )
        if snapshot is None:
            store.ensure_current_person_projection_job(current_id)
            latest = store.get_person_projection_job(current_id)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and (
                latest_identity != identity or latest["state"] == "completed"
            ):
                target_superseded = target_superseded or latest_identity != identity
                continue
            _mark_error(store, identity, "source_snapshot_unavailable")
            return PersonProjectionOutcome(current_id, "pending_error")
        validated = _valid_snapshot(snapshot, identity)
        if validated is None:
            _mark_error(store, identity, "source_snapshot_malformed")
            return PersonProjectionOutcome(current_id, "pending_error")
        person, interactions, rendered_interactions = validated
        try:
            computed = person_projection_source_digest(
                current_id,
                revision,
                person["name"],
                person["relation"],
                person["notes"],
                person["last_contact_at"],
                person["created_at"],
                person["updated_at"],
                interaction_high_watermark,
                rendered_interactions,
            )
        except (TypeError, ValueError):
            computed = ""
        if computed != source_digest:
            _mark_error(store, identity, "source_digest_mismatch")
            return PersonProjectionOutcome(current_id, "pending_error")

        try:
            _path, content_digest = vault.write_person_with_evidence(
                person,
                interactions,
                store_identity=store_identity,
                interaction_high_watermark=interaction_high_watermark,
            )
        except Exception:
            try:
                store.ensure_current_person_projection_job(current_id)
            except ValueError:
                deleted = _reconcile_reopened_person_delete(store, vault, current_id)
                return deleted or PersonProjectionOutcome(current_id, "missing")
            latest = store.get_person_projection_job(current_id)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and (
                latest_identity != identity or latest["state"] == "completed"
            ):
                target_superseded = target_superseded or latest_identity != identity
                continue
            _mark_error(store, identity, "vault_publish_failed")
            return PersonProjectionOutcome(current_id, "pending_error")
        display = _opaque_path_display(current_id)
        if store.complete_person_projection(
            current_id,
            revision,
            interaction_high_watermark,
            source_digest,
            display,
            content_digest,
        ):
            return PersonProjectionOutcome(
                current_id,
                "superseded" if target_superseded else "completed",
                display,
                content_digest,
            )
        deleted = _reconcile_reopened_person_delete(store, vault, current_id)
        if deleted is not None:
            return deleted
        store.ensure_current_person_projection_job(current_id)
        latest = store.get_person_projection_job(current_id)
        latest_identity = _valid_job_identity(latest)
        if latest_identity is not None and latest_identity != identity:
            target_superseded = True

    return PersonProjectionOutcome(person_id, "superseded_or_busy")


def reconcile_pending_person_projections(
    store: MemoryStore,
    vault: ObsidianVault,
    *,
    limit: int = 20,
) -> PersonProjectionSummary:
    if type(limit) is not int:
        raise TypeError("person projection reconciliation limit must be an integer")
    limit = max(1, min(limit, MAX_PERSON_PROJECTION_RECONCILE_JOBS))
    delete_limit = max(1, limit // 2)
    delete_jobs = store.list_pending_person_projection_delete_jobs(limit=delete_limit)
    delete_outcomes = tuple(
        outcome
        for row in delete_jobs
        for outcome in (_reconcile_person_projection_delete(store, vault, row),)
        if outcome is not None
    )
    remaining = max(0, limit - len(delete_jobs))
    if not delete_jobs:
        remaining = limit
    jobs = []
    if remaining:
        store.ensure_missing_person_projection_jobs(limit=remaining)
        jobs = store.list_pending_person_projection_jobs(limit=remaining)
        if not jobs and not delete_jobs:
            jobs = store.list_completed_person_projection_jobs_for_audit(limit=remaining)
    publish_outcomes = tuple(
        reconcile_person_projection(store, vault, job["person_id"])
        for job in jobs
        if type(job["person_id"]) is int and job["person_id"] > 0
    )
    outcomes = delete_outcomes + publish_outcomes
    return PersonProjectionSummary(
        attempted=len(outcomes),
        completed=sum(outcome.status == "completed" for outcome in outcomes),
        pending=(
            store.count_pending_person_projection_jobs()
            + store.count_pending_person_projection_delete_jobs()
        ),
        outcomes=outcomes,
    )
