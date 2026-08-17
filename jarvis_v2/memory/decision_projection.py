from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    DecisionProjectionTarget,
    MAX_DECISION_OUTCOME_SUMMARY_CHARS,
    MAX_DECISION_PROJECTION_OUTCOMES,
    MemoryStore,
    decision_projection_source_digest,
)


MAX_DECISION_PROJECTION_RECONCILE_JOBS = 100
MAX_DECISION_PROJECTION_RECONCILE_PASSES = 4
_MAX_SQLITE_INTEGER = 9_223_372_036_854_775_807
_DECISION_OUTCOME_FIELDS = frozenset(
    {
        "id",
        "decision_id",
        "decision_revision",
        "summary",
        "provenance",
        "created_at",
    }
)


@dataclass(frozen=True)
class DecisionProjectionOutcome:
    decision_id: int
    status: str
    path_display: str = ""
    content_digest: str = ""


@dataclass(frozen=True)
class DecisionProjectionSummary:
    attempted: int
    completed: int
    pending: int
    outcomes: tuple[DecisionProjectionOutcome, ...]
    custody_backfilled: int = 0


def _opaque_path_display(decision_id: int) -> str:
    return f"Decisions/decision-{decision_id}.md"


def _valid_digest(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_target(target: DecisionProjectionTarget) -> tuple[int, int, str] | None:
    try:
        decision_id = target.decision_id
        revision = target.revision
        source_digest = target.source_digest
    except AttributeError:
        return None
    if (
        type(decision_id) is not int
        or decision_id < 1
        or type(revision) is not int
        or revision < 1
        or not _valid_digest(source_digest)
    ):
        return None
    return decision_id, revision, source_digest


def _target_from_row(row) -> DecisionProjectionTarget | None:
    try:
        target = DecisionProjectionTarget(
            decision_id=row["decision_id"],
            revision=row["decision_revision"],
            source_digest=row["source_digest"],
        )
    except (KeyError, IndexError, TypeError):
        return None
    return target if _valid_target(target) is not None else None


def _valid_job_identity(row) -> tuple[int, int, str, str] | None:
    try:
        decision_id = row["decision_id"]
        revision = row["decision_revision"]
        store_identity = row["store_identity"]
        state = row["state"]
        source_digest = row["source_digest"]
        path_display = row["path_display"]
        content_digest = row["content_digest"]
    except (KeyError, IndexError, TypeError):
        return None
    if (
        type(decision_id) is not int
        or decision_id < 1
        or type(revision) is not int
        or revision < 1
        or type(store_identity) is not str
        or re.fullmatch(r"[0-9a-f]{32}", store_identity) is None
        or state not in {"pending", "completed"}
        or not _valid_digest(source_digest)
        or (
            state == "pending"
            and (path_display is not None or content_digest is not None)
        )
        or (
            state == "completed"
            and (type(path_display) is not str or not _valid_digest(content_digest))
        )
    ):
        return None
    return decision_id, revision, store_identity, source_digest


def _mark_malformed_job(store: MemoryStore, row, requested_id: int) -> None:
    target = _target_from_row(row)
    if target is None or target.decision_id != requested_id:
        return
    try:
        store.mark_decision_projection_error(target, "malformed_job")
    except (TypeError, ValueError):
        return


def _valid_snapshot(snapshot, target: DecisionProjectionTarget):
    try:
        decision_id = snapshot["id"]
        revision = snapshot["revision"]
        title = snapshot["title"]
        rationale = snapshot["rationale"]
        impact = snapshot["impact"]
        status = snapshot["status"]
        created_at = snapshot["created_at"]
        updated_at = snapshot["updated_at"]
        raw_outcomes = snapshot["outcomes"]
    except (KeyError, IndexError, TypeError):
        return None
    if (
        type(decision_id) is not int
        or decision_id != target.decision_id
        or type(revision) is not int
        or revision != target.revision
        or any(
            type(value) is not str
            for value in (title, rationale, impact, status, created_at, updated_at)
        )
    ):
        return None

    if isinstance(raw_outcomes, (str, bytes, bytearray, dict)):
        return None
    try:
        outcome_iterator = iter(raw_outcomes)
    except Exception:
        return None

    outcomes: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    seen_revisions: set[int] = set()
    prior_order: tuple[str, int] | None = None
    try:
        for outcome in outcome_iterator:
            if len(outcomes) >= MAX_DECISION_PROJECTION_OUTCOMES:
                return None
            keys_method = getattr(outcome, "keys", None)
            if not callable(keys_method):
                return None
            keys: list[object] = []
            key_iterator = iter(keys_method())
            for _ in range(len(_DECISION_OUTCOME_FIELDS) + 1):
                try:
                    keys.append(next(key_iterator))
                except StopIteration:
                    break
            if (
                len(keys) != len(_DECISION_OUTCOME_FIELDS)
                or any(type(key) is not str for key in keys)
                or frozenset(keys) != _DECISION_OUTCOME_FIELDS
            ):
                return None
            item = {
                "id": outcome["id"],
                "decision_id": outcome["decision_id"],
                "decision_revision": outcome["decision_revision"],
                "summary": outcome["summary"],
                "provenance": outcome["provenance"],
                "created_at": outcome["created_at"],
            }
            outcome_id = item["id"]
            outcome_decision_id = item["decision_id"]
            outcome_revision = item["decision_revision"]
            summary = item["summary"]
            provenance = item["provenance"]
            outcome_created_at = item["created_at"]
            if (
                type(outcome_id) is not int
                or not 1 <= outcome_id <= _MAX_SQLITE_INTEGER
                or outcome_id in seen_ids
                or type(outcome_decision_id) is not int
                or outcome_decision_id != decision_id
                or type(outcome_revision) is not int
                or not 1 <= outcome_revision <= revision
                or outcome_revision in seen_revisions
                or type(summary) is not str
                or not summary.strip()
                or len(summary) > MAX_DECISION_OUTCOME_SUMMARY_CHARS
                or any(
                    unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
                    for character in summary
                )
                or provenance != "user_reported"
                or type(provenance) is not str
                or not _valid_outcome_timestamp(outcome_created_at)
            ):
                return None
            order = (outcome_created_at, outcome_id)
            if prior_order is not None and order <= prior_order:
                return None
            seen_ids.add(outcome_id)
            seen_revisions.add(outcome_revision)
            prior_order = order
            outcomes.append(item)
    except Exception:
        return None

    return {
        "id": decision_id,
        "revision": revision,
        "title": title,
        "rationale": rationale,
        "impact": impact,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at,
        "outcomes": tuple(outcomes),
    }


def _valid_outcome_timestamp(value: object) -> bool:
    if type(value) is not str or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _mark_error(
    store: MemoryStore,
    target: DecisionProjectionTarget,
    error_code: str,
) -> None:
    store.mark_decision_projection_error(target, error_code)


def _latest_job_after_ensure(store: MemoryStore, decision_id: int):
    try:
        store.ensure_current_decision_projection_job(decision_id)
    except ValueError:
        return None
    return store.get_decision_projection_job(decision_id)


def reconcile_decision_projection(
    store: MemoryStore,
    vault: ObsidianVault,
    decision_id: int | DecisionProjectionTarget,
    target: DecisionProjectionTarget | None = None,
) -> DecisionProjectionOutcome:
    if isinstance(decision_id, DecisionProjectionTarget):
        if target is not None:
            raise ValueError("decision projection target was provided twice")
        target = decision_id
        decision_id = target.decision_id
    if type(decision_id) is not int or decision_id < 1:
        raise ValueError("decision projection id must be a positive integer")

    expected_identity = None
    if target is not None:
        expected_identity = _valid_target(target)
        if expected_identity is None or expected_identity[0] != decision_id:
            raise ValueError(
                "decision projection target is invalid or belongs to another decision"
            )

    target_superseded = False
    for _ in range(MAX_DECISION_PROJECTION_RECONCILE_PASSES):
        job = store.get_decision_projection_job(decision_id)
        if job is None:
            job = _latest_job_after_ensure(store, decision_id)
            if job is None:
                return DecisionProjectionOutcome(decision_id, "missing")

        identity = _valid_job_identity(job)
        if identity is None or identity[0] != decision_id:
            _mark_malformed_job(store, job, decision_id)
            return DecisionProjectionOutcome(decision_id, "malformed")
        current_id, revision, store_identity, source_digest = identity
        state = job["state"]
        current_target = DecisionProjectionTarget(
            decision_id=current_id,
            revision=revision,
            source_digest=source_digest,
        )
        current_identity = (current_id, revision, source_digest)
        if expected_identity is not None and current_identity != expected_identity:
            target_superseded = True
            expected_identity = None

        if state == "completed":
            try:
                path_display = job["path_display"]
                content_digest = job["content_digest"]
            except (KeyError, IndexError, TypeError):
                return DecisionProjectionOutcome(current_id, "malformed")
            if (
                path_display != _opaque_path_display(current_id)
                or not _valid_digest(content_digest)
            ):
                return DecisionProjectionOutcome(current_id, "malformed")

            snapshot = store.get_current_decision_projection_snapshot(current_target)
            validated = _valid_snapshot(snapshot, current_target) if snapshot is not None else None
            if validated is None:
                latest = _latest_job_after_ensure(store, current_id)
                latest_identity = _valid_job_identity(latest)
                if latest_identity is not None and (
                    latest_identity[0],
                    latest_identity[1],
                    latest_identity[3],
                ) != current_identity:
                    target_superseded = True
                    continue
                return DecisionProjectionOutcome(current_id, "completed_audit_error")

            try:
                verified = vault.verify_decision_projection_evidence(
                    decision_id=current_id,
                    decision_title=validated["title"],
                    store_identity=store_identity,
                    expected_content_digest=content_digest,
                )
            except Exception:
                verified = False
            if verified is True:
                store.mark_decision_projection_audited(current_target, content_digest)
                return DecisionProjectionOutcome(
                    current_id,
                    "superseded" if target_superseded else "completed",
                    _opaque_path_display(current_id),
                    content_digest,
                )
            if store.reopen_completed_decision_projection(
                current_target,
                content_digest,
            ):
                continue
            latest = store.get_decision_projection_job(current_id)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and latest_identity != identity:
                target_superseded = True
                continue
            return DecisionProjectionOutcome(current_id, "completed_audit_error")

        snapshot = store.get_current_decision_projection_snapshot(current_target)
        if snapshot is None:
            latest = _latest_job_after_ensure(store, current_id)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and (
                latest_identity != identity or latest["state"] == "completed"
            ):
                target_superseded = target_superseded or latest_identity != identity
                continue
            if latest is None:
                return DecisionProjectionOutcome(current_id, "missing")
            _mark_error(store, current_target, "source_snapshot_unavailable")
            return DecisionProjectionOutcome(current_id, "pending_error")

        validated = _valid_snapshot(snapshot, current_target)
        if validated is None:
            _mark_error(store, current_target, "source_snapshot_malformed")
            return DecisionProjectionOutcome(current_id, "pending_error")
        try:
            computed = decision_projection_source_digest(
                current_id,
                revision,
                validated["title"],
                validated["rationale"],
                validated["impact"],
                validated["status"],
                validated["created_at"],
                validated["updated_at"],
                validated["outcomes"],
            )
        except (TypeError, ValueError):
            computed = ""
        if computed != source_digest:
            _mark_error(store, current_target, "source_digest_mismatch")
            return DecisionProjectionOutcome(current_id, "pending_error")

        try:
            _path, content_digest = vault.write_decision_with_evidence(
                validated,
                store_identity=store_identity,
            )
            if not _valid_digest(content_digest):
                raise ValueError("decision projection content digest is invalid")
        except Exception:
            latest = _latest_job_after_ensure(store, current_id)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and (
                latest_identity != identity or latest["state"] == "completed"
            ):
                target_superseded = target_superseded or latest_identity != identity
                continue
            if latest is None:
                return DecisionProjectionOutcome(current_id, "missing")
            _mark_error(store, current_target, "vault_publish_failed")
            return DecisionProjectionOutcome(current_id, "pending_error")

        display = _opaque_path_display(current_id)
        if store.complete_decision_projection(
            current_target,
            display,
            content_digest,
        ):
            return DecisionProjectionOutcome(
                current_id,
                "superseded" if target_superseded else "completed",
                display,
                content_digest,
            )

        latest = _latest_job_after_ensure(store, current_id)
        latest_identity = _valid_job_identity(latest)
        if latest_identity is not None and latest_identity != identity:
            target_superseded = True
        elif latest is None:
            return DecisionProjectionOutcome(current_id, "missing")

    return DecisionProjectionOutcome(decision_id, "superseded_or_busy")


def reconcile_pending_decision_projections(
    store: MemoryStore,
    vault: ObsidianVault,
    *,
    limit: int = 20,
) -> DecisionProjectionSummary:
    if type(limit) is not int:
        raise TypeError("decision projection reconciliation limit must be an integer")
    limit = max(1, min(limit, MAX_DECISION_PROJECTION_RECONCILE_JOBS))
    ensure_custody = getattr(store, "ensure_missing_decision_custody", None)
    custody_backfilled = int(ensure_custody(limit=limit)) if callable(ensure_custody) else 0
    store.ensure_missing_decision_projection_jobs(limit=limit)
    jobs = store.list_pending_decision_projection_jobs(limit=limit)
    audit_jobs = store.list_completed_decision_projection_jobs_for_audit(
        limit=1 if jobs else limit
    )
    if not jobs:
        jobs = list(audit_jobs)
    elif audit_jobs:
        audited_id = audit_jobs[0]["decision_id"]
        if not any(job["decision_id"] == audited_id for job in jobs):
            jobs.append(audit_jobs[0])

    outcomes: list[DecisionProjectionOutcome] = []
    for job in jobs:
        try:
            current_id = job["decision_id"]
        except (KeyError, IndexError, TypeError):
            continue
        if type(current_id) is int and current_id > 0:
            outcomes.append(reconcile_decision_projection(store, vault, current_id))
    frozen_outcomes = tuple(outcomes)
    count_missing_custody = getattr(store, "count_missing_decision_custody", None)
    missing_custody = int(count_missing_custody()) if callable(count_missing_custody) else 0
    return DecisionProjectionSummary(
        attempted=len(frozen_outcomes),
        completed=sum(
            outcome.status in {"completed", "superseded"}
            for outcome in frozen_outcomes
        ),
        pending=store.count_pending_decision_projection_jobs() + missing_custody,
        outcomes=frozen_outcomes,
        custody_backfilled=custody_backfilled,
    )
