from __future__ import annotations

from dataclasses import dataclass

from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore, skill_projection_source_digest


MAX_SKILL_PROJECTION_RECONCILE_JOBS = 100
MAX_SKILL_PROJECTION_RECONCILE_PASSES = 4


@dataclass(frozen=True)
class SkillProjectionOutcome:
    skill_id: int
    operation: str
    status: str
    path_display: str = ""
    content_digest: str = ""
    completion_status: str = ""


@dataclass(frozen=True)
class SkillProjectionSummary:
    attempted: int
    completed: int
    pending: int
    outcomes: tuple[SkillProjectionOutcome, ...]


def _path_display(vault: ObsidianVault, path) -> str:
    try:
        return str(path.relative_to(vault.root_path))
    except (AttributeError, ValueError):
        return ""


def _valid_job_identity(row) -> tuple[int, int, str, str, str] | None:
    try:
        skill_id = row["skill_id"]
        revision = row["skill_revision"]
        operation = row["operation"]
        store_identity = row["store_identity"]
        source_digest = row["source_digest"]
    except (KeyError, IndexError, TypeError):
        return None
    if (
        type(skill_id) is not int
        or skill_id < 1
        or type(revision) is not int
        or revision < 1
        or operation not in {"publish", "delete"}
        or type(store_identity) is not str
        or len(store_identity) != 32
        or type(source_digest) is not str
        or len(source_digest) != 64
    ):
        return None
    return skill_id, revision, operation, store_identity, source_digest


def reconcile_skill_projection(
    store: MemoryStore,
    vault: ObsidianVault,
    skill_id: int,
    *,
    expected_operation: str | None = None,
    expected_revision: int | None = None,
    expected_source_digest: str | None = None,
) -> SkillProjectionOutcome:
    expected_values = (expected_operation, expected_revision, expected_source_digest)
    if any(value is not None for value in expected_values):
        if (
            expected_operation not in {"publish", "delete"}
            or type(expected_revision) is not int
            or expected_revision < 1
            or type(expected_source_digest) is not str
            or len(expected_source_digest) != 64
        ):
            raise ValueError("expected skill projection identity is incomplete or invalid")
        expected_identity = (
            skill_id,
            expected_revision,
            expected_operation,
            expected_source_digest,
        )
    else:
        expected_identity = None
    repair_current = False
    target_superseded = False
    for _ in range(MAX_SKILL_PROJECTION_RECONCILE_PASSES):
        job = store.get_skill_projection_job(skill_id)
        if job is None:
            return SkillProjectionOutcome(skill_id, "unknown", "missing")
        identity = _valid_job_identity(job)
        if identity is None:
            return SkillProjectionOutcome(skill_id, "unknown", "malformed")
        current_id, revision, operation, store_identity, source_digest = identity
        current_target = (current_id, revision, operation, source_digest)
        if expected_identity is not None and current_target != expected_identity:
            target_superseded = True
            if not repair_current:
                return SkillProjectionOutcome(current_id, operation, "superseded")
        if job["state"] == "completed":
            if repair_current:
                if operation == "publish":
                    snapshot = store.get_current_skill_projection_publish_snapshot(
                        current_id,
                        revision,
                        str(job["canonical_name"]),
                        source_digest,
                    )
                    if snapshot is None:
                        return SkillProjectionOutcome(current_id, operation, "repair_error")
                    try:
                        path, content_digest = vault.write_skill_with_evidence(
                            str(snapshot["canonical_name"]),
                            str(snapshot["trigger"]),
                            str(snapshot["body"]),
                            str(snapshot["tags"]),
                            skill_id=current_id,
                            store_identity=store_identity,
                            skill_revision=revision,
                            source_digest=source_digest,
                        )
                    except Exception:
                        return SkillProjectionOutcome(current_id, operation, "repair_error")
                    display = _path_display(vault, path)
                    if (
                        display != str(job["path_display"] or "")
                        or content_digest != str(job["content_digest"] or "")
                    ):
                        return SkillProjectionOutcome(current_id, operation, "repair_error")
                else:
                    try:
                        repaired_delete = vault.delete_skill_projection(
                            str(job["canonical_name"]),
                            skill_id=current_id,
                            store_identity=store_identity,
                        )
                    except Exception:
                        return SkillProjectionOutcome(current_id, operation, "repair_error")
                    if repaired_delete.status == "error":
                        return SkillProjectionOutcome(current_id, operation, "repair_error")
                latest_job = store.get_skill_projection_job(current_id)
                latest_identity = _valid_job_identity(latest_job)
                if (
                    latest_job is None
                    or latest_identity != identity
                    or latest_job["state"] != "completed"
                ):
                    repair_current = True
                    continue
            return SkillProjectionOutcome(
                current_id,
                operation,
                "superseded" if target_superseded else "completed",
                str(job["path_display"] or ""),
                str(job["content_digest"] or ""),
                str(job["completion_status"] or ""),
            )

        if operation == "publish":
            snapshot = store.get_skill_projection_publish_snapshot(
                current_id,
                revision,
                source_digest,
            )
            if snapshot is None:
                continue
            try:
                computed = skill_projection_source_digest(
                    current_id,
                    revision,
                    snapshot["canonical_name"],
                    snapshot["trigger"],
                    snapshot["body"],
                    snapshot["tags"],
                    "publish",
                )
            except (TypeError, ValueError):
                store.mark_skill_projection_error(
                    current_id,
                    operation,
                    source_digest,
                    "source_snapshot_malformed",
                )
                return SkillProjectionOutcome(current_id, operation, "pending_error")
            if computed != source_digest:
                store.mark_skill_projection_error(
                    current_id,
                    operation,
                    source_digest,
                    "source_digest_mismatch",
                )
                return SkillProjectionOutcome(current_id, operation, "pending_error")
            try:
                path, content_digest = vault.write_skill_with_evidence(
                    str(snapshot["canonical_name"]),
                    str(snapshot["trigger"]),
                    str(snapshot["body"]),
                    str(snapshot["tags"]),
                    skill_id=current_id,
                    store_identity=store_identity,
                    skill_revision=revision,
                    source_digest=source_digest,
                )
            except Exception:
                store.mark_skill_projection_error(
                    current_id,
                    operation,
                    source_digest,
                    "vault_publish_failed",
                )
                return SkillProjectionOutcome(current_id, operation, "pending_error")
            display = _path_display(vault, path)
            if not display:
                store.mark_skill_projection_error(
                    current_id,
                    operation,
                    source_digest,
                    "path_display_unavailable",
                )
                return SkillProjectionOutcome(current_id, operation, "pending_error")
            if store.complete_skill_projection_publish(
                current_id,
                revision,
                source_digest,
                display,
                content_digest,
            ):
                return SkillProjectionOutcome(
                    current_id,
                    operation,
                    "superseded" if target_superseded else "completed",
                    display,
                    content_digest,
                )
            repair_current = True
            continue

        try:
            deleted = vault.delete_skill_projection(
                str(job["canonical_name"]),
                skill_id=current_id,
                store_identity=store_identity,
            )
        except Exception:
            deleted = None
        if deleted is None or deleted.status == "error":
            store.mark_skill_projection_error(
                current_id,
                operation,
                source_digest,
                "vault_delete_failed",
            )
            return SkillProjectionOutcome(current_id, operation, "pending_error")
        if store.complete_skill_projection_delete(
            current_id,
            revision,
            source_digest,
            deleted.status,
            deleted.path_display or None,
        ):
            return SkillProjectionOutcome(
                current_id,
                operation,
                "superseded" if target_superseded else "completed",
                deleted.path_display,
                "",
                deleted.status,
            )
        repair_current = True
    job = store.get_skill_projection_job(skill_id)
    operation = str(job["operation"]) if job is not None else "unknown"
    return SkillProjectionOutcome(skill_id, operation, "superseded_or_busy")


def reconcile_pending_skill_projections(
    store: MemoryStore,
    vault: ObsidianVault,
    *,
    limit: int = 20,
) -> SkillProjectionSummary:
    if type(limit) is not int:
        raise TypeError("skill projection reconciliation limit must be an integer")
    limit = max(1, min(limit, MAX_SKILL_PROJECTION_RECONCILE_JOBS))
    jobs = store.list_pending_skill_projection_jobs(limit=limit)
    outcomes = tuple(
        reconcile_skill_projection(store, vault, int(job["skill_id"]))
        for job in jobs
    )
    completed = sum(outcome.status == "completed" for outcome in outcomes)
    return SkillProjectionSummary(
        attempted=len(outcomes),
        completed=completed,
        pending=store.count_pending_skill_projection_jobs(),
        outcomes=outcomes,
    )
