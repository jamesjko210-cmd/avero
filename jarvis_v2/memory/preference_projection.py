from __future__ import annotations

import re
from dataclasses import dataclass

from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    MemoryStore,
    PreferenceProjectionTarget,
    preference_projection_source_digest,
)


MAX_PREFERENCE_PROJECTION_RECONCILE_PASSES = 4
PREFERENCE_PROJECTION_PATH_DISPLAY = "Memory Tree/Preferences.md"


@dataclass(frozen=True)
class PreferenceProjectionOutcome:
    generation: int | None
    status: str


@dataclass(frozen=True)
class PreferenceProjectionSummary:
    attempted: int
    completed: int
    pending: int
    outcomes: tuple[PreferenceProjectionOutcome, ...]
    custody_backfilled: int = 0
    custody_conflicts: int = 0


def _valid_digest(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_target(target: object) -> tuple[int, str] | None:
    if type(target) is not PreferenceProjectionTarget:
        return None
    generation = target.generation
    source_digest = target.source_digest
    if (
        type(generation) is not int
        or generation < 0
        or generation > 9223372036854775807
        or not _valid_digest(source_digest)
    ):
        return None
    return generation, source_digest


def _valid_job_identity(
    row,
) -> tuple[int, str, str, str, str | None, str | None] | None:
    try:
        singleton_id = row["singleton_id"]
        generation = row["generation"]
        store_identity = row["store_identity"]
        state = row["state"]
        source_digest = row["source_digest"]
        path_display = row["path_display"]
        content_digest = row["content_digest"]
    except (KeyError, IndexError, TypeError):
        return None
    if (
        type(singleton_id) is not int
        or singleton_id != 1
        or type(generation) is not int
        or generation < 0
        or generation > 9223372036854775807
        or type(store_identity) is not str
        or re.fullmatch(r"[0-9a-f]{32}", store_identity) is None
        or state not in {"pending", "completed"}
        or not _valid_digest(source_digest)
    ):
        return None
    if state == "pending":
        if path_display is not None or content_digest is not None:
            return None
    elif (
        path_display != PREFERENCE_PROJECTION_PATH_DISPLAY
        or not _valid_digest(content_digest)
    ):
        return None
    return (
        generation,
        store_identity,
        state,
        source_digest,
        path_display,
        content_digest,
    )


def _job_generation(row) -> int | None:
    try:
        generation = row["generation"]
    except (KeyError, IndexError, TypeError):
        return None
    if (
        type(generation) is not int
        or generation < 0
        or generation > 9223372036854775807
    ):
        return None
    return generation


def _path_display(vault: ObsidianVault, path) -> str:
    try:
        return str(path.relative_to(vault.root_path))
    except (AttributeError, TypeError, ValueError):
        return ""


def _mark_error(
    store: MemoryStore,
    target: PreferenceProjectionTarget,
    error_code: str,
) -> None:
    try:
        store.mark_preference_projection_error(target, error_code)
    except Exception:
        pass


def _refresh_job(store: MemoryStore):
    try:
        store.ensure_current_preference_projection_job()
        return store.get_preference_projection_job()
    except Exception:
        return None


def _snapshot_rows(snapshot) -> tuple[object, ...] | None:
    if type(snapshot) is not tuple:
        return None
    return snapshot


def reconcile_preference_projection(
    store: MemoryStore,
    vault: ObsidianVault,
    target: PreferenceProjectionTarget | None = None,
) -> PreferenceProjectionOutcome:
    expected_identity = None
    if target is not None:
        expected_identity = _valid_target(target)
        if expected_identity is None:
            raise ValueError("preference projection target is invalid")

    target_superseded = False
    last_generation: int | None = None
    for _ in range(MAX_PREFERENCE_PROJECTION_RECONCILE_PASSES):
        try:
            job = store.get_preference_projection_job()
        except Exception:
            job = None
        if job is None:
            job = _refresh_job(store)
            if job is None:
                return PreferenceProjectionOutcome(last_generation, "missing")
        identity = _valid_job_identity(job)
        if identity is None:
            return PreferenceProjectionOutcome(_job_generation(job), "malformed")
        (
            generation,
            store_identity,
            state,
            source_digest,
            _retained_path_display,
            retained_content_digest,
        ) = identity
        last_generation = generation
        current_identity = (generation, source_digest)
        current_target = PreferenceProjectionTarget(generation, source_digest)
        if expected_identity is not None and current_identity != expected_identity:
            target_superseded = True
            expected_identity = None

        if state == "completed":
            snapshot = store.get_current_preference_projection_snapshot(current_target)
            rows = _snapshot_rows(snapshot)
            try:
                computed_digest = (
                    preference_projection_source_digest(generation, rows)
                    if rows is not None
                    else ""
                )
            except (TypeError, ValueError):
                computed_digest = ""
            if computed_digest != source_digest:
                latest = _refresh_job(store)
                latest_identity = _valid_job_identity(latest)
                if latest_identity is not None and (
                    (latest_identity[0], latest_identity[3]) != current_identity
                    or latest_identity[2] != state
                ):
                    target_superseded = target_superseded or (
                        (latest_identity[0], latest_identity[3]) != current_identity
                    )
                    continue
                return PreferenceProjectionOutcome(generation, "completed_audit_error")

            try:
                verified = vault.verify_preferences_projection_evidence(
                    store_identity=store_identity,
                    generation=generation,
                    expected_content_digest=retained_content_digest,
                )
            except Exception:
                verified = False
            if verified:
                try:
                    audited = store.mark_preference_projection_audited(
                        current_target, retained_content_digest
                    )
                except Exception:
                    audited = False
                if audited:
                    return PreferenceProjectionOutcome(
                        generation,
                        "superseded" if target_superseded else "completed",
                    )
                latest = _refresh_job(store)
                latest_identity = _valid_job_identity(latest)
                if latest_identity is not None and (
                    (latest_identity[0], latest_identity[3]) != current_identity
                    or latest_identity[2] != state
                ):
                    target_superseded = target_superseded or (
                        (latest_identity[0], latest_identity[3]) != current_identity
                    )
                    continue
                return PreferenceProjectionOutcome(generation, "completed_audit_error")

            try:
                reopened = store.reopen_completed_preference_projection(
                    current_target, retained_content_digest
                )
            except Exception:
                reopened = False
            if reopened:
                continue
            latest = _refresh_job(store)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and (
                (latest_identity[0], latest_identity[3]) != current_identity
                or latest_identity[2] != state
            ):
                target_superseded = target_superseded or (
                    (latest_identity[0], latest_identity[3]) != current_identity
                )
                continue
            return PreferenceProjectionOutcome(generation, "completed_audit_error")

        snapshot = store.get_current_preference_projection_snapshot(current_target)
        rows = _snapshot_rows(snapshot)
        if rows is None:
            latest = _refresh_job(store)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and (
                (latest_identity[0], latest_identity[3]) != current_identity
                or latest_identity[2] == "completed"
            ):
                target_superseded = target_superseded or (
                    (latest_identity[0], latest_identity[3]) != current_identity
                )
                continue
            _mark_error(store, current_target, "source_snapshot_unavailable")
            return PreferenceProjectionOutcome(generation, "pending_error")
        try:
            computed_digest = preference_projection_source_digest(generation, rows)
        except (TypeError, ValueError):
            computed_digest = ""
        if computed_digest != source_digest:
            latest = _refresh_job(store)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and (
                (latest_identity[0], latest_identity[3]) != current_identity
                or latest_identity[2] == "completed"
            ):
                target_superseded = target_superseded or (
                    (latest_identity[0], latest_identity[3]) != current_identity
                )
                continue
            _mark_error(
                store,
                current_target,
                "source_snapshot_malformed"
                if computed_digest == ""
                else "source_digest_mismatch",
            )
            return PreferenceProjectionOutcome(generation, "pending_error")

        try:
            path, content_digest = vault.write_preferences_with_evidence(
                rows,
                store_identity=store_identity,
                generation=generation,
            )
        except Exception:
            latest = _refresh_job(store)
            latest_identity = _valid_job_identity(latest)
            if latest_identity is not None and (
                (latest_identity[0], latest_identity[3]) != current_identity
                or latest_identity[2] == "completed"
            ):
                target_superseded = target_superseded or (
                    (latest_identity[0], latest_identity[3]) != current_identity
                )
                continue
            _mark_error(store, current_target, "vault_publish_failed")
            return PreferenceProjectionOutcome(generation, "pending_error")

        if _path_display(vault, path) != PREFERENCE_PROJECTION_PATH_DISPLAY:
            _mark_error(store, current_target, "projection_path_mismatch")
            return PreferenceProjectionOutcome(generation, "pending_error")
        if not _valid_digest(content_digest):
            _mark_error(store, current_target, "content_digest_invalid")
            return PreferenceProjectionOutcome(generation, "pending_error")
        try:
            completed = store.complete_preference_projection(
                current_target,
                PREFERENCE_PROJECTION_PATH_DISPLAY,
                content_digest,
            )
        except Exception:
            completed = False
        if completed:
            return PreferenceProjectionOutcome(
                generation,
                "superseded" if target_superseded else "completed",
            )

        latest = _refresh_job(store)
        latest_identity = _valid_job_identity(latest)
        if latest_identity is not None and (
            (latest_identity[0], latest_identity[3]) != current_identity
            or latest_identity[2] == "completed"
        ):
            target_superseded = target_superseded or (
                (latest_identity[0], latest_identity[3]) != current_identity
            )
            continue
        _mark_error(store, current_target, "projection_completion_failed")
        return PreferenceProjectionOutcome(generation, "pending_error")

    return PreferenceProjectionOutcome(last_generation, "superseded_or_busy")


def reconcile_pending_preference_projections(
    store: MemoryStore,
    vault: ObsidianVault,
) -> PreferenceProjectionSummary:
    ensure_custody = getattr(store, "ensure_missing_preference_custody", None)
    custody_backfilled = int(ensure_custody(limit=20)) if callable(ensure_custody) else 0
    try:
        store.ensure_missing_preference_projection_job()
        job = store.get_preference_projection_job()
    except Exception:
        job = None
    if job is None:
        count_conflicts = getattr(store, "count_preference_identity_conflicts", None)
        custody_conflicts = int(count_conflicts()) if callable(count_conflicts) else 0
        return PreferenceProjectionSummary(
            attempted=0,
            completed=0,
            pending=store.count_pending_preference_projection_jobs()
            + (
                int(store.count_missing_preference_custody())
                if callable(getattr(store, "count_missing_preference_custody", None))
                else 0
            )
            + custody_conflicts,
            outcomes=(),
            custody_backfilled=custody_backfilled,
            custody_conflicts=custody_conflicts,
        )
    outcome = reconcile_preference_projection(store, vault)
    count_missing_custody = getattr(store, "count_missing_preference_custody", None)
    missing_custody = int(count_missing_custody()) if callable(count_missing_custody) else 0
    count_conflicts = getattr(store, "count_preference_identity_conflicts", None)
    custody_conflicts = int(count_conflicts()) if callable(count_conflicts) else 0
    return PreferenceProjectionSummary(
        attempted=1,
        completed=int(outcome.status == "completed"),
        pending=(
            store.count_pending_preference_projection_jobs()
            + missing_custody
            + custody_conflicts
        ),
        outcomes=(outcome,),
        custody_backfilled=custody_backfilled,
        custody_conflicts=custody_conflicts,
    )
