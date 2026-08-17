from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.decision_projection import (
    reconcile_decision_projection,
    reconcile_pending_decision_projections,
)
from jarvis_v2.memory.store import (
    DecisionProjectionTarget,
    decision_projection_source_digest,
)


class FakeDecisionStore:
    def __init__(self) -> None:
        self.store_identity = "a" * 32
        self.sources: dict[int, dict[str, object]] = {}
        self.jobs: dict[int, dict[str, object]] = {}
        self._next_id = 1
        self._clock = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    @staticmethod
    def _target_for(source: dict[str, object]) -> DecisionProjectionTarget:
        decision_id = source["id"]
        revision = source["revision"]
        if type(decision_id) is not int or type(revision) is not int:
            raise AssertionError("fake source identity is malformed")
        digest = decision_projection_source_digest(
            decision_id,
            revision,
            source["title"],
            source["rationale"],
            source["impact"],
            source["status"],
            source["created_at"],
            source["updated_at"],
            source.get("outcomes", ()),
        )
        return DecisionProjectionTarget(decision_id, revision, digest)

    def _install_pending_job(self, target: DecisionProjectionTarget) -> None:
        self.jobs[target.decision_id] = {
            "decision_id": target.decision_id,
            "decision_revision": target.revision,
            "store_identity": self.store_identity,
            "state": "pending",
            "source_digest": target.source_digest,
            "path_display": None,
            "content_digest": None,
            "last_error_code": None,
            "updated_order": self._tick(),
            "audit_count": 0,
        }

    def add_decision(
        self,
        *,
        title: str,
        rationale: str,
        impact: str,
        status: str = "active",
        materialize_job: bool = True,
    ) -> DecisionProjectionTarget:
        decision_id = self._next_id
        self._next_id += 1
        source: dict[str, object] = {
            "id": decision_id,
            "title": title,
            "rationale": rationale,
            "impact": impact,
            "status": status,
            "revision": 1,
            "created_at": "2026-07-12T00:00:00Z",
            "updated_at": "2026-07-12T00:00:01Z",
            "outcomes": (),
        }
        self.sources[decision_id] = source
        target = self._target_for(source)
        if materialize_job:
            self._install_pending_job(target)
        return target

    def revise_decision(
        self,
        decision_id: int,
        *,
        rationale: str,
        impact: str,
    ) -> DecisionProjectionTarget:
        current = self.sources[decision_id]
        revision = int(current["revision"]) + 1
        revised = dict(current)
        revised.update(
            rationale=rationale,
            impact=impact,
            revision=revision,
            updated_at=f"2026-07-12T00:00:{revision + 1:02d}Z",
        )
        self.sources[decision_id] = revised
        target = self._target_for(revised)
        self._install_pending_job(target)
        return target

    @staticmethod
    def _job_matches_target(
        job: dict[str, object] | None,
        target: DecisionProjectionTarget,
    ) -> bool:
        return job is not None and (
            job.get("decision_id") == target.decision_id
            and job.get("decision_revision") == target.revision
            and job.get("source_digest") == target.source_digest
        )

    def get_decision_projection_job(self, decision_id: int):
        return self.jobs.get(decision_id)

    def ensure_current_decision_projection_job(
        self,
        decision_id: int,
    ) -> DecisionProjectionTarget:
        source = self.sources.get(decision_id)
        if source is None:
            raise ValueError("decision source does not exist")
        target = self._target_for(source)
        if not self._job_matches_target(self.jobs.get(decision_id), target):
            self._install_pending_job(target)
        return target

    def get_current_decision_projection_snapshot(
        self,
        target: DecisionProjectionTarget,
    ):
        source = self.sources.get(target.decision_id)
        if source is None or self._target_for(source) != target:
            return None
        if not self._job_matches_target(self.jobs.get(target.decision_id), target):
            return None
        return dict(source)

    def mark_decision_projection_error(
        self,
        target: DecisionProjectionTarget,
        error_code: str,
    ) -> bool:
        job = self.jobs.get(target.decision_id)
        if not self._job_matches_target(job, target) or job["state"] != "pending":
            return False
        job["last_error_code"] = error_code
        job["updated_order"] = self._tick()
        return True

    def complete_decision_projection(
        self,
        target: DecisionProjectionTarget,
        path_display: str,
        content_digest: str,
    ) -> bool:
        job = self.jobs.get(target.decision_id)
        source = self.sources.get(target.decision_id)
        if (
            not self._job_matches_target(job, target)
            or job["state"] != "pending"
            or source is None
            or self._target_for(source) != target
        ):
            return False
        if path_display != f"Decisions/decision-{target.decision_id}.md":
            raise ValueError("decision path display is not opaque")
        if re.fullmatch(r"[0-9a-f]{64}", content_digest) is None:
            raise ValueError("decision content digest is invalid")
        job.update(
            state="completed",
            path_display=path_display,
            content_digest=content_digest,
            last_error_code=None,
            updated_order=self._tick(),
        )
        return True

    def reopen_completed_decision_projection(
        self,
        target: DecisionProjectionTarget,
        expected_content_digest: str,
    ) -> bool:
        job = self.jobs.get(target.decision_id)
        source = self.sources.get(target.decision_id)
        if (
            not self._job_matches_target(job, target)
            or job["state"] != "completed"
            or job["content_digest"] != expected_content_digest
            or source is None
            or self._target_for(source) != target
        ):
            return False
        job.update(
            state="pending",
            path_display=None,
            content_digest=None,
            last_error_code="completed_projection_invalid",
            updated_order=self._tick(),
        )
        return True

    def mark_decision_projection_audited(
        self,
        target: DecisionProjectionTarget,
        expected_content_digest: str,
    ) -> bool:
        job = self.jobs.get(target.decision_id)
        if (
            not self._job_matches_target(job, target)
            or job["state"] != "completed"
            or job["content_digest"] != expected_content_digest
        ):
            return False
        job["audit_count"] = int(job["audit_count"]) + 1
        job["updated_order"] = self._tick()
        return True

    def ensure_missing_decision_projection_jobs(self, limit: int) -> int:
        missing = [
            decision_id
            for decision_id in sorted(self.sources)
            if decision_id not in self.jobs
        ][:limit]
        for decision_id in missing:
            self._install_pending_job(self._target_for(self.sources[decision_id]))
        return len(missing)

    def list_pending_decision_projection_jobs(self, limit: int):
        pending = [job for job in self.jobs.values() if job["state"] == "pending"]
        pending.sort(
            key=lambda job: (
                job["last_error_code"] is not None,
                job["updated_order"],
                job["decision_id"],
            )
        )
        return pending[:limit]

    def list_completed_decision_projection_jobs_for_audit(self, limit: int):
        completed = [
            job for job in self.jobs.values() if job["state"] == "completed"
        ]
        completed.sort(key=lambda job: (job["updated_order"], job["decision_id"]))
        return completed[:limit]

    def count_pending_decision_projection_jobs(self) -> int:
        missing = sum(decision_id not in self.jobs for decision_id in self.sources)
        pending = sum(job["state"] == "pending" for job in self.jobs.values())
        return missing + pending


class FakeDecisionVault:
    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path
        (self.root_path / "Decisions").mkdir(parents=True, exist_ok=True)
        self.write_calls = 0
        self.fail_writes = 0
        self.before_write = None

    @staticmethod
    def _filename_title(title: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-") or "decision"

    def path_for(self, decision_id: int, title: str) -> Path:
        return (
            self.root_path
            / "Decisions"
            / f"{decision_id:04d} {self._filename_title(title)}.md"
        )

    @staticmethod
    def _content(decision: dict[str, object], store_identity: str) -> str:
        return "\n".join(
            (
                "---",
                "jarvis_projection: decision",
                f"store_identity: {store_identity}",
                f"id: {decision['id']}",
                f"source_revision: {decision['revision']}",
                "---",
                "",
                f"# {decision['title']}",
                "",
                "## Rationale",
                "",
                str(decision["rationale"]),
                "",
                "## Impact",
                "",
                str(decision["impact"]),
                "",
                "## Status",
                "",
                str(decision["status"]),
                "",
                f"Created: {decision['created_at']}",
                f"Updated: {decision['updated_at']}",
                "",
            )
        )

    def write_decision_with_evidence(
        self,
        decision: dict[str, object],
        *,
        store_identity: str,
    ) -> tuple[Path, str]:
        self.write_calls += 1
        callback = self.before_write
        if callback is not None:
            callback()
        if self.fail_writes:
            self.fail_writes -= 1
            raise OSError(
                f"private write failure: {decision['title']} "
                f"{decision['rationale']} {self.root_path}"
            )
        path = self.path_for(int(decision["id"]), str(decision["title"]))
        content = self._content(decision, store_identity)
        path.write_text(content, encoding="utf-8")
        return path, hashlib.sha256(content.encode("utf-8")).hexdigest()

    def verify_decision_projection_evidence(
        self,
        *,
        decision_id: int,
        decision_title: str,
        store_identity: str,
        expected_content_digest: str,
    ) -> bool:
        path = self.path_for(decision_id, decision_title)
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return False
        owned = (
            "jarvis_projection: decision" in content
            and f"store_identity: {store_identity}" in content
            and f"id: {decision_id}" in content
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return owned and digest == expected_content_digest


def _assert_private(value: object, forbidden: tuple[str, ...], label: str) -> None:
    rendered = str(value)
    leaked = [secret for secret in forbidden if secret and secret in rendered]
    if leaked:
        raise SystemExit(f"{label} leaked decision content or a real path: {leaked}")


def _assert_source_unchanged(
    store: FakeDecisionStore,
    before: dict[int, dict[str, object]],
    label: str,
) -> None:
    if store.sources != before:
        raise SystemExit(f"{label} mutated decision source rows")


def test_startup_publish_idempotency_privacy_and_exact_target() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-projection-publish-") as temp:
        root = Path(temp)
        store = FakeDecisionStore()
        vault = FakeDecisionVault(root / "vault-private-root")
        title = "PRIVATE-DECISION-TITLE-PUBLISH"
        rationale = "PRIVATE-DECISION-RATIONALE-PUBLISH"
        impact = "PRIVATE-DECISION-IMPACT-PUBLISH"
        target = store.add_decision(
            title=title,
            rationale=rationale,
            impact=impact,
            materialize_job=False,
        )
        sources_before = copy.deepcopy(store.sources)

        summary = reconcile_pending_decision_projections(store, vault, limit=1)
        job = store.get_decision_projection_job(target.decision_id)
        path = vault.path_for(target.decision_id, title)
        if (
            summary.attempted != 1
            or summary.completed != 1
            or summary.pending != 0
            or job is None
            or job["state"] != "completed"
            or job["path_display"] != f"Decisions/decision-{target.decision_id}.md"
            or not path.exists()
        ):
            raise SystemExit(f"startup decision projection did not publish: {summary}")
        text = path.read_text(encoding="utf-8")
        if title not in text or rationale not in text or impact not in text:
            raise SystemExit("published decision projection omitted exact source content")
        _assert_source_unchanged(store, sources_before, "startup publish")
        forbidden = (title, rationale, impact, str(root), str(path))
        _assert_private(summary, forbidden, "startup decision projection summary")

        before_bytes = path.read_bytes()
        writes_before = vault.write_calls
        repeated = reconcile_decision_projection(store, vault, target)
        if (
            repeated.status != "completed"
            or vault.write_calls != writes_before
            or path.read_bytes() != before_bytes
            or job["audit_count"] != 1
        ):
            raise SystemExit(f"completed decision projection was not idempotent: {repeated}")
        _assert_private(repeated, forbidden, "idempotent decision projection outcome")
        _assert_source_unchanged(store, sources_before, "completed audit")

        invalid_targets = (
            (True, None),
            (
                target.decision_id,
                DecisionProjectionTarget(target.decision_id, 0, target.source_digest),
            ),
            (
                target.decision_id,
                DecisionProjectionTarget(target.decision_id + 1, 1, "0" * 64),
            ),
        )
        for invalid_id, invalid_target in invalid_targets:
            try:
                reconcile_decision_projection(
                    store,
                    vault,
                    invalid_id,
                    target=invalid_target,
                )
            except ValueError:
                pass
            else:
                raise SystemExit("invalid exact decision target was accepted")
        if vault.write_calls != writes_before:
            raise SystemExit("invalid exact target reached the decision vault")


def test_failure_retry_privacy_and_no_source_mutation() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-projection-retry-") as temp:
        root = Path(temp)
        store = FakeDecisionStore()
        vault = FakeDecisionVault(root / "vault-private-root")
        title = "PRIVATE-DECISION-TITLE-RETRY"
        rationale = "PRIVATE-DECISION-RATIONALE-RETRY"
        impact = "PRIVATE-DECISION-IMPACT-RETRY"
        target = store.add_decision(title=title, rationale=rationale, impact=impact)
        sources_before = copy.deepcopy(store.sources)
        vault.fail_writes = 1

        failed = reconcile_decision_projection(store, vault, target)
        job = store.get_decision_projection_job(target.decision_id)
        if (
            failed.status != "pending_error"
            or job["state"] != "pending"
            or job["last_error_code"] != "vault_publish_failed"
        ):
            raise SystemExit(f"decision write failure lost retry custody: {failed}")
        forbidden = (title, rationale, impact, str(root))
        _assert_private(failed, forbidden, "failed decision projection outcome")
        _assert_source_unchanged(store, sources_before, "failed decision publish")

        repaired = reconcile_decision_projection(store, vault, target)
        if repaired.status != "completed" or job["last_error_code"] is not None:
            raise SystemExit(f"decision projection retry did not converge: {repaired}")
        _assert_private(repaired, forbidden, "repaired decision projection outcome")
        _assert_source_unchanged(store, sources_before, "decision projection retry")


def test_pending_current_supersession() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-projection-supersession-") as temp:
        root = Path(temp)
        store = FakeDecisionStore()
        vault = FakeDecisionVault(root / "vault-private-root")
        title = "PRIVATE-DECISION-TITLE-SUPERSESSION"
        old_rationale = "PRIVATE-OLD-RATIONALE-MUST-NOT-WIN"
        old_impact = "PRIVATE-OLD-IMPACT-MUST-NOT-WIN"
        new_rationale = "PRIVATE-NEW-RATIONALE-MUST-WIN"
        new_impact = "PRIVATE-NEW-IMPACT-MUST-WIN"
        old_target = store.add_decision(
            title=title,
            rationale=old_rationale,
            impact=old_impact,
        )
        newest: list[DecisionProjectionTarget] = []

        def supersede_during_publish() -> None:
            vault.before_write = None
            newest.append(
                store.revise_decision(
                    old_target.decision_id,
                    rationale=new_rationale,
                    impact=new_impact,
                )
            )

        vault.before_write = supersede_during_publish
        summary = reconcile_pending_decision_projections(store, vault, limit=20)
        if len(summary.outcomes) != 1:
            raise SystemExit(f"decision supersession summary was malformed: {summary}")
        outcome = summary.outcomes[0]
        job = store.get_decision_projection_job(old_target.decision_id)
        path = vault.path_for(old_target.decision_id, title)
        text = path.read_text(encoding="utf-8")
        if (
            outcome.status != "superseded"
            or len(newest) != 1
            or job["state"] != "completed"
            or job["decision_revision"] != newest[0].revision
            or job["source_digest"] != newest[0].source_digest
            or vault.write_calls != 2
            or new_rationale not in text
            or new_impact not in text
            or old_rationale in text
            or old_impact in text
            or summary.attempted != 1
            or summary.completed != 1
            or summary.pending != 0
        ):
            raise SystemExit(f"decision supersession did not converge to current: {outcome}")
        _assert_private(
            outcome,
            (
                title,
                old_rationale,
                old_impact,
                new_rationale,
                new_impact,
                str(root),
                str(path),
            ),
            "superseded decision projection outcome",
        )


def test_completed_missing_and_corrupt_repair() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-projection-audit-") as temp:
        root = Path(temp)
        store = FakeDecisionStore()
        vault = FakeDecisionVault(root / "vault-private-root")
        targets = (
            store.add_decision(
                title="PRIVATE-MISSING-DECISION-TITLE",
                rationale="PRIVATE-MISSING-DECISION-RATIONALE",
                impact="PRIVATE-MISSING-DECISION-IMPACT",
            ),
            store.add_decision(
                title="PRIVATE-CORRUPT-DECISION-TITLE",
                rationale="PRIVATE-CORRUPT-DECISION-RATIONALE",
                impact="PRIVATE-CORRUPT-DECISION-IMPACT",
            ),
        )
        initial = [reconcile_decision_projection(store, vault, target) for target in targets]
        if any(outcome.status != "completed" for outcome in initial):
            raise SystemExit(f"decision audit fixtures did not complete: {initial}")
        sources_before = copy.deepcopy(store.sources)
        paths = [
            vault.path_for(target.decision_id, str(store.sources[target.decision_id]["title"]))
            for target in targets
        ]
        paths[0].unlink()
        paths[1].write_text(
            paths[1].read_text(encoding="utf-8") + "CORRUPTED-BY-SMOKE\n",
            encoding="utf-8",
        )

        repaired = reconcile_pending_decision_projections(store, vault, limit=2)
        if repaired.attempted != 2 or repaired.completed != 2 or repaired.pending != 0:
            raise SystemExit(f"completed decision audit did not repair both notes: {repaired}")
        for target, path in zip(targets, paths, strict=True):
            job = store.get_decision_projection_job(target.decision_id)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if (
                not path.exists()
                or job["state"] != "completed"
                or job["content_digest"] != digest
                or job["last_error_code"] is not None
            ):
                raise SystemExit("completed decision evidence repair was not durable")
        _assert_source_unchanged(store, sources_before, "completed decision repair")
        forbidden = tuple(
            str(store.sources[target.decision_id][field])
            for target in targets
            for field in ("title", "rationale", "impact")
        ) + (str(root),) + tuple(str(path) for path in paths)
        _assert_private(repaired, forbidden, "completed decision repair summary")

        writes_before = vault.write_calls
        audited = reconcile_pending_decision_projections(store, vault, limit=2)
        if (
            audited.completed != 2
            or vault.write_calls != writes_before
            or any(
                store.jobs[target.decision_id]["audit_count"] != 1
                for target in targets
            )
        ):
            raise SystemExit(f"healthy completed decision audit rewrote evidence: {audited}")
        _assert_source_unchanged(store, sources_before, "healthy completed audit")


def test_pending_failure_does_not_starve_completed_audit() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-projection-audit-fairness-") as temp:
        root = Path(temp)
        store = FakeDecisionStore()
        vault = FakeDecisionVault(root / "vault-private-root")
        pending = store.add_decision(
            title="PRIVATE-PERSISTENT-PENDING-TITLE",
            rationale="PRIVATE-PERSISTENT-PENDING-RATIONALE",
            impact="PRIVATE-PERSISTENT-PENDING-IMPACT",
        )
        completed = store.add_decision(
            title="PRIVATE-AUDIT-FAIRNESS-TITLE",
            rationale="PRIVATE-AUDIT-FAIRNESS-RATIONALE",
            impact="PRIVATE-AUDIT-FAIRNESS-IMPACT",
        )
        initial = reconcile_decision_projection(store, vault, completed)
        if initial.status != "completed":
            raise SystemExit(f"audit fairness fixture did not complete: {initial}")
        completed_path = vault.path_for(
            completed.decision_id,
            str(store.sources[completed.decision_id]["title"]),
        )
        completed_path.write_text(
            completed_path.read_text(encoding="utf-8") + "CORRUPTED-BY-FAIRNESS-SMOKE\n",
            encoding="utf-8",
        )
        vault.fail_writes = 1
        summary = reconcile_pending_decision_projections(store, vault, limit=1)
        completed_job = store.get_decision_projection_job(completed.decision_id)
        pending_job = store.get_decision_projection_job(pending.decision_id)
        if (
            summary.attempted != 2
            or summary.completed != 1
            or summary.pending != 1
            or pending_job["state"] != "pending"
            or completed_job["state"] != "completed"
            or "CORRUPTED-BY-FAIRNESS-SMOKE" in completed_path.read_text(encoding="utf-8")
        ):
            raise SystemExit(f"persistent pending job starved completed audit: {summary}")


def test_malformed_fairness_and_error_code() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-projection-malformed-") as temp:
        root = Path(temp)
        store = FakeDecisionStore()
        vault = FakeDecisionVault(root / "vault-private-root")
        malformed = store.add_decision(
            title="PRIVATE-MALFORMED-DECISION-TITLE",
            rationale="PRIVATE-MALFORMED-DECISION-RATIONALE",
            impact="PRIVATE-MALFORMED-DECISION-IMPACT",
        )
        healthy = store.add_decision(
            title="PRIVATE-HEALTHY-DECISION-TITLE",
            rationale="PRIVATE-HEALTHY-DECISION-RATIONALE",
            impact="PRIVATE-HEALTHY-DECISION-IMPACT",
        )
        raw_private_path = str(root / "PRIVATE-TITLE-IN-REAL-PATH.md")
        store.jobs[malformed.decision_id]["path_display"] = raw_private_path
        sources_before = copy.deepcopy(store.sources)

        first = reconcile_pending_decision_projections(store, vault, limit=1)
        malformed_job = store.jobs[malformed.decision_id]
        if (
            tuple(outcome.status for outcome in first.outcomes) != ("malformed",)
            or malformed_job["last_error_code"] != "malformed_job"
            or vault.write_calls != 0
        ):
            raise SystemExit(f"malformed decision job did not fail closed: {first}")
        _assert_private(
            first,
            (
                str(store.sources[malformed.decision_id]["title"]),
                str(store.sources[malformed.decision_id]["rationale"]),
                str(store.sources[malformed.decision_id]["impact"]),
                raw_private_path,
                str(root),
            ),
            "malformed decision projection outcome",
        )

        second = reconcile_pending_decision_projections(store, vault, limit=1)
        healthy_job = store.jobs[healthy.decision_id]
        if (
            tuple(outcome.status for outcome in second.outcomes) != ("completed",)
            or healthy_job["state"] != "completed"
            or second.pending != 1
            or vault.write_calls != 1
        ):
            raise SystemExit(f"malformed decision job starved healthy bounded work: {second}")
        _assert_source_unchanged(store, sources_before, "malformed fairness reconciliation")


def main() -> None:
    test_startup_publish_idempotency_privacy_and_exact_target()
    test_failure_retry_privacy_and_no_source_mutation()
    test_pending_current_supersession()
    test_completed_missing_and_corrupt_repair()
    test_pending_failure_does_not_starve_completed_audit()
    test_malformed_fairness_and_error_code()
    print("Decision projection reconciliation smoke test passed.")


if __name__ == "__main__":
    main()
