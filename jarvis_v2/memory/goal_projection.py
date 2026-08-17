from __future__ import annotations

import json
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    GoalProjectionTarget,
    MAX_GOAL_PROJECTION_STEPS,
    MemoryStore,
    goal_projection_source_digest,
)


MAX_GOAL_PROJECTION_RECONCILE_JOBS = 20
MAX_GOAL_PROJECTION_RECONCILE_PASSES = 4
_MAX_SQLITE_INTEGER = 9_223_372_036_854_775_807
_GOAL_FIELDS = frozenset(
    {
        "id",
        "title",
        "purpose",
        "horizon",
        "status",
        "revision",
        "created_at",
        "updated_at",
    }
)
_GOAL_STEP_FIELDS = frozenset(
    {
        "id",
        "goal_id",
        "body",
        "status",
        "created_at",
        "updated_at",
        "completed_at",
    }
)
_ERROR_CODES = frozenset(
    {
        "content_digest_invalid",
        "malformed_job",
        "projection_completion_failed",
        "published_source_digest_mismatch",
        "source_digest_mismatch",
        "source_snapshot_malformed",
        "source_snapshot_unavailable",
        "vault_publish_failed",
    }
)


@dataclass(frozen=True)
class GoalProjectionOutcome:
    goal_id: int
    status: str
    path_display: str = ""
    content_digest: str = ""


@dataclass(frozen=True)
class GoalProjectionSummary:
    attempted: int
    completed: int
    pending: int
    outcomes: tuple[GoalProjectionOutcome, ...]


@dataclass(frozen=True)
class _GoalProjectionJob:
    goal_id: int
    revision: int
    store_identity: str
    source_digest: str
    state: str
    path_display: str | None
    content_digest: str | None
    prepared_content_digest: str | None
    prepared_content_digests: tuple[str, ...]
    prior_content_digest: str | None

    @property
    def target_identity(self) -> tuple[int, int, str]:
        return self.goal_id, self.revision, self.source_digest

    @property
    def target(self) -> GoalProjectionTarget:
        return GoalProjectionTarget(
            goal_id=self.goal_id,
            revision=self.revision,
            source_digest=self.source_digest,
        )


def _opaque_path_display(goal_id: int) -> str:
    return f"Projects/Goal {goal_id}.md"


def _valid_digest(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_target(target: object) -> tuple[int, int, str] | None:
    if type(target) is not GoalProjectionTarget:
        return None
    if (
        type(target.goal_id) is not int
        or not 1 <= target.goal_id <= _MAX_SQLITE_INTEGER
        or type(target.revision) is not int
        or not 1 <= target.revision <= _MAX_SQLITE_INTEGER
        or not _valid_digest(target.source_digest)
    ):
        return None
    return target.goal_id, target.revision, target.source_digest


def _row_values(row: object, fields: frozenset[str]) -> dict[str, object] | None:
    if isinstance(row, (str, bytes, bytearray, list, tuple, set, frozenset)):
        return None
    keys_method = getattr(row, "keys", None)
    if not callable(keys_method):
        return None
    try:
        key_iterator = iter(keys_method())
        keys: list[object] = []
        for _ in range(len(fields) + 1):
            try:
                keys.append(next(key_iterator))
            except StopIteration:
                break
        if (
            len(keys) != len(fields)
            or any(type(key) is not str for key in keys)
            or frozenset(keys) != fields
        ):
            return None
        return {field: row[field] for field in fields}  # type: ignore[index]
    except Exception:
        return None


def _job_from_row(row: object) -> _GoalProjectionJob | None:
    try:
        goal_id = row["goal_id"]  # type: ignore[index]
        revision = row["goal_revision"]  # type: ignore[index]
        store_identity = row["store_identity"]  # type: ignore[index]
        source_digest = row["source_digest"]  # type: ignore[index]
        state = row["state"]  # type: ignore[index]
        path_display = row["path_display"]  # type: ignore[index]
        content_digest = row["content_digest"]  # type: ignore[index]
        prepared_content_digest = row["prepared_content_digest"]  # type: ignore[index]
        prepared_content_digests_raw = row["prepared_content_digests"]  # type: ignore[index]
        prior_content_digest = row["prior_content_digest"]  # type: ignore[index]
    except Exception:
        return None
    try:
        prepared_content_digests = tuple(json.loads(prepared_content_digests_raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        type(goal_id) is not int
        or not 1 <= goal_id <= _MAX_SQLITE_INTEGER
        or type(revision) is not int
        or not 1 <= revision <= _MAX_SQLITE_INTEGER
        or type(store_identity) is not str
        or re.fullmatch(r"[0-9a-f]{32}", store_identity) is None
        or not _valid_digest(source_digest)
        or type(state) is not str
        or state not in {"pending", "completed"}
        or (
            prepared_content_digest is not None
            and not _valid_digest(prepared_content_digest)
        )
        or type(prepared_content_digests_raw) is not str
        or len(prepared_content_digests) > 256
        or any(not _valid_digest(item) for item in prepared_content_digests)
        or len(set(prepared_content_digests)) != len(prepared_content_digests)
        or prepared_content_digests_raw
        != json.dumps(
            prepared_content_digests,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        or (
            prepared_content_digest is not None
            and prepared_content_digest not in prepared_content_digests
        )
        or (
            prior_content_digest is not None
            and not _valid_digest(prior_content_digest)
        )
    ):
        return None
    if state == "pending":
        if path_display is not None or content_digest is not None:
            return None
    elif path_display != _opaque_path_display(goal_id) or not _valid_digest(
        content_digest
    ) or prepared_content_digest is not None or prepared_content_digests:
        return None
    return _GoalProjectionJob(
        goal_id=goal_id,
        revision=revision,
        store_identity=store_identity,
        source_digest=source_digest,
        state=state,
        path_display=path_display,
        content_digest=content_digest,
        prepared_content_digest=prepared_content_digest,
        prepared_content_digests=prepared_content_digests,
        prior_content_digest=prior_content_digest,
    )


def _target_from_row(row: object) -> GoalProjectionTarget | None:
    try:
        target = GoalProjectionTarget(
            goal_id=row["goal_id"],  # type: ignore[index]
            revision=row["goal_revision"],  # type: ignore[index]
            source_digest=row["source_digest"],  # type: ignore[index]
        )
    except Exception:
        return None
    return target if _valid_target(target) is not None else None


def _mark_error(
    store: MemoryStore,
    target: GoalProjectionTarget,
    error_code: str,
) -> None:
    if error_code not in _ERROR_CODES or len(error_code) > 64:
        raise ValueError("invalid goal projection error code")
    try:
        store.mark_goal_projection_error(target, error_code)
    except Exception:
        pass


def _mark_malformed_job(store: MemoryStore, row: object, goal_id: int) -> None:
    target = _target_from_row(row)
    if target is not None and target.goal_id == goal_id:
        _mark_error(store, target, "malformed_job")


def _snapshot_parts(snapshot: object) -> tuple[object, object] | None:
    if type(snapshot) is tuple:
        if len(snapshot) != 2:
            return None
        return snapshot[0], snapshot[1]
    if not isinstance(snapshot, Mapping):
        return None

    nested = _row_values(snapshot, frozenset({"goal", "steps"}))
    if nested is not None:
        return nested["goal"], nested["steps"]

    flattened = _row_values(snapshot, _GOAL_FIELDS | {"steps"})
    if flattened is None:
        return None
    steps = flattened.pop("steps")
    return flattened, steps


def _valid_snapshot(
    snapshot: object,
    target: GoalProjectionTarget,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]] | None:
    parts = _snapshot_parts(snapshot)
    if parts is None:
        return None
    raw_goal, raw_steps = parts
    goal = _row_values(raw_goal, _GOAL_FIELDS)
    if goal is None:
        return None
    if (
        type(goal["id"]) is not int
        or goal["id"] != target.goal_id
        or type(goal["revision"]) is not int
        or goal["revision"] != target.revision
        or any(
            type(goal[field]) is not str
            for field in (
                "title",
                "purpose",
                "horizon",
                "status",
                "created_at",
                "updated_at",
            )
        )
    ):
        return None
    if (
        isinstance(raw_steps, (str, bytes, bytearray, Mapping))
        or not isinstance(raw_steps, Sequence)
    ):
        return None

    steps: list[dict[str, object]] = []
    prior_step_id = 0
    try:
        for raw_step in raw_steps:
            if len(steps) >= MAX_GOAL_PROJECTION_STEPS:
                return None
            step = _row_values(raw_step, _GOAL_STEP_FIELDS)
            if step is None:
                return None
            step_id = step["id"]
            if (
                type(step_id) is not int
                or not prior_step_id < step_id <= _MAX_SQLITE_INTEGER
                or type(step["goal_id"]) is not int
                or step["goal_id"] != target.goal_id
                or any(
                    type(step[field]) is not str
                    for field in ("body", "status", "created_at", "updated_at")
                )
                or (
                    step["completed_at"] is not None
                    and type(step["completed_at"]) is not str
                )
            ):
                return None
            prior_step_id = step_id
            steps.append(step)
    except Exception:
        return None
    return goal, tuple(steps)


def _snapshot_source_digest(
    goal: dict[str, object],
    steps: tuple[dict[str, object], ...],
) -> str:
    return goal_projection_source_digest(
        goal["id"],
        goal["revision"],
        goal["title"],
        goal["purpose"],
        goal["horizon"],
        goal["status"],
        goal["created_at"],
        goal["updated_at"],
        steps,
    )


def _latest_job_after_ensure(store: MemoryStore, goal_id: int):
    try:
        store.ensure_current_goal_projection_job(goal_id)
    except ValueError:
        return None
    return store.get_goal_projection_job(goal_id)


def _refresh_after_race(
    store: MemoryStore,
    current: _GoalProjectionJob,
) -> tuple[_GoalProjectionJob | None, bool, bool]:
    latest_row = _latest_job_after_ensure(store, current.goal_id)
    if latest_row is None:
        return None, False, False
    latest = _job_from_row(latest_row)
    if latest is None:
        _mark_malformed_job(store, latest_row, current.goal_id)
        return None, False, True
    return latest, latest.target_identity != current.target_identity, False


def _verify_completed_evidence(
    vault: ObsidianVault,
    job: _GoalProjectionJob,
    goal: dict[str, object],
) -> bool:
    if job.content_digest is None:
        return False
    try:
        verified = vault.verify_goal_projection_evidence(
            goal_id=job.goal_id,
            title=goal["title"],
            store_identity=job.store_identity,
            expected_content_digest=job.content_digest,
            expected_revision=job.revision,
            expected_source_digest=job.source_digest,
        )
    except Exception:
        return False
    return verified is True


def _completed_outcome(
    job: _GoalProjectionJob,
    *,
    target_superseded: bool,
) -> GoalProjectionOutcome:
    return GoalProjectionOutcome(
        job.goal_id,
        "superseded" if target_superseded else "completed",
        _opaque_path_display(job.goal_id),
        job.content_digest or "",
    )


def reconcile_goal_projection(
    store: MemoryStore,
    vault: ObsidianVault,
    goal_id: int | GoalProjectionTarget,
    target: GoalProjectionTarget | None = None,
) -> GoalProjectionOutcome:
    if type(goal_id) is GoalProjectionTarget:
        if target is not None:
            raise ValueError("goal projection target was provided twice")
        target = goal_id
        goal_id = target.goal_id
    if (
        type(goal_id) is not int
        or goal_id < 1
        or goal_id > _MAX_SQLITE_INTEGER
    ):
        raise ValueError("goal projection id must be a positive SQLite integer")

    expected_identity = None
    if target is not None:
        expected_identity = _valid_target(target)
        if expected_identity is None or expected_identity[0] != goal_id:
            raise ValueError(
                "goal projection target is invalid or belongs to another goal"
            )

    target_superseded = False
    for _ in range(MAX_GOAL_PROJECTION_RECONCILE_PASSES):
        job_row = store.get_goal_projection_job(goal_id)
        if job_row is None:
            job_row = _latest_job_after_ensure(store, goal_id)
            if job_row is None:
                return GoalProjectionOutcome(goal_id, "missing")

        job = _job_from_row(job_row)
        if job is None or job.goal_id != goal_id:
            _mark_malformed_job(store, job_row, goal_id)
            return GoalProjectionOutcome(goal_id, "malformed")
        current_target = job.target
        if expected_identity is not None and job.target_identity != expected_identity:
            target_superseded = True
            expected_identity = None

        if job.state == "completed":
            snapshot = store.get_current_goal_projection_snapshot(current_target)
            validated = (
                _valid_snapshot(snapshot, current_target)
                if snapshot is not None
                else None
            )
            if validated is None:
                latest, superseded, malformed = _refresh_after_race(store, job)
                if malformed:
                    return GoalProjectionOutcome(goal_id, "malformed")
                if latest is not None and (
                    superseded or latest.state != job.state
                ):
                    target_superseded = target_superseded or superseded
                    continue
                return GoalProjectionOutcome(goal_id, "completed_audit_error")
            goal, steps = validated
            computed_digest = _snapshot_source_digest(
                goal,
                steps,
            )
            if not secrets.compare_digest(computed_digest, job.source_digest):
                latest, superseded, malformed = _refresh_after_race(store, job)
                if malformed:
                    return GoalProjectionOutcome(goal_id, "malformed")
                if latest is not None and (
                    superseded or latest.state != job.state
                ):
                    target_superseded = target_superseded or superseded
                    continue
                return GoalProjectionOutcome(goal_id, "completed_audit_error")

            if _verify_completed_evidence(vault, job, goal):
                try:
                    audited = store.mark_goal_projection_audited(
                        current_target,
                        job.content_digest,
                    )
                except Exception:
                    audited = False
                if audited:
                    return _completed_outcome(
                        job,
                        target_superseded=target_superseded,
                    )
                latest, superseded, malformed = _refresh_after_race(store, job)
                if malformed:
                    return GoalProjectionOutcome(goal_id, "malformed")
                if latest is not None and (
                    superseded or latest.state != job.state
                ):
                    target_superseded = target_superseded or superseded
                    continue
                return GoalProjectionOutcome(goal_id, "completed_audit_error")

            try:
                reopened = store.reopen_completed_goal_projection(
                    current_target,
                    job.content_digest,
                )
            except Exception:
                reopened = False
            if reopened:
                continue
            latest, superseded, malformed = _refresh_after_race(store, job)
            if malformed:
                return GoalProjectionOutcome(goal_id, "malformed")
            if latest is not None and (
                superseded or latest.state != job.state
            ):
                target_superseded = target_superseded or superseded
                continue
            return GoalProjectionOutcome(goal_id, "completed_audit_error")

        try:
            preparation_snapshot = store.get_current_goal_projection_snapshot(
                current_target
            )
        except Exception:
            preparation_snapshot = None
        prepared_snapshot = (
            _valid_snapshot(preparation_snapshot, current_target)
            if preparation_snapshot is not None
            else None
        )
        if prepared_snapshot is None:
            latest, superseded, malformed = _refresh_after_race(store, job)
            if malformed:
                return GoalProjectionOutcome(goal_id, "malformed")
            if latest is not None and (superseded or latest.state == "completed"):
                target_superseded = target_superseded or superseded
                continue
            _mark_error(store, current_target, "source_snapshot_unavailable")
            return GoalProjectionOutcome(goal_id, "pending_error")
        prepared_goal, prepared_steps = prepared_snapshot
        if not secrets.compare_digest(
            _snapshot_source_digest(prepared_goal, prepared_steps),
            job.source_digest,
        ):
            _mark_error(store, current_target, "source_digest_mismatch")
            return GoalProjectionOutcome(goal_id, "pending_error")
        try:
            candidate = vault.goal_projection_candidate_evidence(
                prepared_goal,
                prepared_steps,
                store_identity=job.store_identity,
            )
        except Exception:
            candidate = None
        if (
            type(candidate) is not tuple
            or len(candidate) != 2
            or not _valid_digest(candidate[0])
            or not _valid_digest(candidate[1])
            or not secrets.compare_digest(candidate[1], job.source_digest)
        ):
            _mark_error(store, current_target, "vault_publish_failed")
            return GoalProjectionOutcome(goal_id, "pending_error")
        prepared_content_digest = candidate[0]
        try:
            prior_evidence = vault.inspect_goal_projection_prior_evidence(
                prepared_goal,
                prepared_steps,
                store_identity=job.store_identity,
            )
        except Exception:
            prior_evidence = None
        if (
            type(prior_evidence) is not tuple
            or len(prior_evidence) != 2
            or prior_evidence[0] not in {"absent", "legacy", "owned", "unverified"}
            or (
                prior_evidence[0] == "owned"
                and not _valid_digest(prior_evidence[1])
            )
            or (
                prior_evidence[0] != "owned"
                and prior_evidence[1] != ""
            )
            or prior_evidence[0] == "unverified"
        ):
            _mark_error(store, current_target, "vault_publish_failed")
            return GoalProjectionOutcome(goal_id, "pending_error")
        observed_content_digest = (
            prior_evidence[1] if prior_evidence[0] == "owned" else None
        )
        try:
            prepared = store.prepare_goal_projection_after_observed_content(
                current_target,
                observed_content_digest,
                prepared_content_digest,
            )
        except Exception:
            prepared = False
        if not prepared:
            latest, superseded, malformed = _refresh_after_race(store, job)
            if malformed:
                return GoalProjectionOutcome(goal_id, "malformed")
            if latest is not None and (superseded or latest.state == "completed"):
                target_superseded = target_superseded or superseded
                continue
            _mark_error(store, current_target, "vault_publish_failed")
            return GoalProjectionOutcome(goal_id, "pending_error")
        try:
            allowed_prior_content_digests = (
                store.goal_projection_prior_content_digests(current_target)
            )
        except Exception:
            allowed_prior_content_digests = None
        if allowed_prior_content_digests is None:
            latest, superseded, malformed = _refresh_after_race(store, job)
            if malformed:
                return GoalProjectionOutcome(goal_id, "malformed")
            if latest is not None and (superseded or latest.state == "completed"):
                target_superseded = target_superseded or superseded
                continue
            _mark_error(store, current_target, "projection_completion_failed")
            return GoalProjectionOutcome(goal_id, "pending_error")

        published = None
        publication_evidence: list[tuple[str, str]] = []
        publication_error = ""
        try:
            with store.goal_projection_publication(
                current_target,
                prepared_content_digest,
                publication_evidence,
            ) as snapshot:
                validated = _valid_snapshot(snapshot, current_target)
                if validated is None:
                    publication_error = (
                        "source_snapshot_unavailable"
                        if snapshot[0] is None
                        else "source_snapshot_malformed"
                    )
                else:
                    goal, steps = validated
                    computed_digest = _snapshot_source_digest(goal, steps)
                    if not secrets.compare_digest(computed_digest, job.source_digest):
                        publication_error = "source_digest_mismatch"
                    else:
                        published = vault.write_goal_with_evidence(
                            goal,
                            steps,
                            store_identity=job.store_identity,
                            expected_prior_content_digest=allowed_prior_content_digests,
                        )
                        if type(published) is not tuple or len(published) != 3:
                            publication_error = "vault_publish_failed"
                        else:
                            path, content_digest, published_source_digest = published
                            if not isinstance(path, Path):
                                publication_error = "vault_publish_failed"
                            elif not _valid_digest(content_digest):
                                publication_error = "content_digest_invalid"
                            elif (
                                not _valid_digest(published_source_digest)
                                or not secrets.compare_digest(
                                    published_source_digest,
                                    job.source_digest,
                                )
                            ):
                                publication_error = "published_source_digest_mismatch"
                            else:
                                publication_evidence.append(
                                    (_opaque_path_display(goal_id), content_digest)
                                )
        except Exception:
            publication_error = "vault_publish_failed"

        if publication_error:
            latest, superseded, malformed = _refresh_after_race(store, job)
            if malformed:
                return GoalProjectionOutcome(goal_id, "malformed")
            if latest is None:
                return GoalProjectionOutcome(goal_id, "missing")
            if superseded or latest.state == "completed":
                target_superseded = target_superseded or superseded
                continue
            _mark_error(store, current_target, publication_error)
            return GoalProjectionOutcome(goal_id, "pending_error")

        if published is None or not publication_evidence:
            _mark_error(store, current_target, "vault_publish_failed")
            return GoalProjectionOutcome(goal_id, "pending_error")
        path, content_digest, published_source_digest = published
        display = _opaque_path_display(goal_id)
        return GoalProjectionOutcome(
            goal_id,
            "superseded" if target_superseded else "completed",
            display,
            content_digest,
        )

    return GoalProjectionOutcome(goal_id, "superseded_or_busy")


def _bounded_job_rows(value: object, *, limit: int) -> tuple[object, ...]:
    if type(value) not in {list, tuple}:
        return ()
    return tuple(value[:limit])


def _job_row_goal_id(row: object) -> int | None:
    try:
        goal_id = row["goal_id"]  # type: ignore[index]
    except Exception:
        return None
    if (
        type(goal_id) is not int
        or goal_id < 1
        or goal_id > _MAX_SQLITE_INTEGER
    ):
        return None
    return goal_id


def _job_row_fairness_key(row: object) -> tuple[str, int]:
    try:
        updated_at = row["updated_at"]  # type: ignore[index]
    except Exception:
        updated_at = ""
    goal_id = _job_row_goal_id(row)
    return (
        updated_at if type(updated_at) is str else "",
        goal_id if goal_id is not None else 0,
    )


def reconcile_pending_goal_projections(
    store: MemoryStore,
    vault: ObsidianVault,
    *,
    limit: int = 20,
) -> GoalProjectionSummary:
    if type(limit) is not int:
        raise TypeError("goal projection reconciliation limit must be an integer")
    limit = max(1, min(limit, MAX_GOAL_PROJECTION_RECONCILE_JOBS))

    store.ensure_missing_goal_projection_jobs(limit=limit)
    pending_rows = _bounded_job_rows(
        store.list_pending_goal_projection_jobs(limit=limit),
        limit=limit,
    )
    audit_rows = _bounded_job_rows(
        store.list_completed_goal_projection_jobs_for_audit(limit=1),
        limit=1,
    )

    if pending_rows and audit_rows and limit == 1:
        selected_rows = (min(
            (pending_rows[0], audit_rows[0]),
            key=_job_row_fairness_key,
        ),)
    elif pending_rows and audit_rows:
        pending_selection = pending_rows[: limit - 1]
        audit_id = _job_row_goal_id(audit_rows[0])
        selected_rows = pending_selection
        if audit_id is not None and all(
            _job_row_goal_id(row) != audit_id for row in pending_selection
        ):
            selected_rows += (audit_rows[0],)
    elif pending_rows:
        selected_rows = pending_rows
    else:
        selected_rows = audit_rows

    outcomes: list[GoalProjectionOutcome] = []
    for row in selected_rows:
        current_id = _job_row_goal_id(row)
        if current_id is not None:
            outcomes.append(reconcile_goal_projection(store, vault, current_id))
    frozen_outcomes = tuple(outcomes)
    pending = store.count_pending_goal_projection_jobs()
    if type(pending) is not int or pending < 0:
        raise RuntimeError("goal projection pending count is invalid")
    return GoalProjectionSummary(
        attempted=len(frozen_outcomes),
        completed=sum(
            outcome.status in {"completed", "superseded"}
            for outcome in frozen_outcomes
        ),
        pending=pending,
        outcomes=frozen_outcomes,
    )
