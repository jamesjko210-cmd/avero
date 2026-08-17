from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.preference_projection import (
    PREFERENCE_PROJECTION_PATH_DISPLAY,
    PreferenceProjectionOutcome,
    reconcile_pending_preference_projections,
    reconcile_preference_projection,
)
from jarvis_v2.memory.store import (
    MemoryStore,
    PreferenceProjectionTarget,
    preference_projection_source_digest,
)


STORE_METHODS = (
    "get_preference_projection_job",
    "ensure_current_preference_projection_job",
    "get_current_preference_projection_snapshot",
    "mark_preference_projection_error",
    "complete_preference_projection",
    "reopen_completed_preference_projection",
    "mark_preference_projection_audited",
    "count_pending_preference_projection_jobs",
    "ensure_missing_preference_projection_job",
)
VAULT_METHODS = (
    "write_preferences_with_evidence",
    "verify_preferences_projection_evidence",
)
STORE_IDENTITY = "ab" * 16


def _assert_runtime_contract() -> None:
    missing_store = [name for name in STORE_METHODS if not hasattr(MemoryStore, name)]
    missing_vault = [name for name in VAULT_METHODS if not hasattr(ObsidianVault, name)]
    if missing_store or missing_vault:
        missing = [f"MemoryStore.{name}" for name in missing_store]
        missing.extend(f"ObsidianVault.{name}" for name in missing_vault)
        raise SystemExit("missing preference projection APIs: " + ", ".join(missing))


def _row(index: int, *, secret_prefix: str = "PRIVATE-PREFERENCE") -> dict[str, object]:
    timestamp = f"2026-07-12T10:{index % 60:02d}:00Z"
    return {
        "id": index + 1,
        "category": f"category-{index % 7}",
        "key": f"key-{index:03d}",
        "value": f"{secret_prefix}-VALUE-{index:03d}",
        "status": "retired" if index % 5 == 0 else "active",
        "revision": index + 1,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def _target(generation: int, rows: tuple[dict[str, object], ...]) -> PreferenceProjectionTarget:
    return PreferenceProjectionTarget(
        generation,
        preference_projection_source_digest(generation, rows),
    )


class MockPreferenceStore:
    def __init__(
        self,
        generation: int,
        rows: tuple[dict[str, object], ...],
    ) -> None:
        self.snapshots: dict[
            PreferenceProjectionTarget, tuple[dict[str, object], ...]
        ] = {}
        self.job: dict[str, object] | None = None
        self.audit_count = 0
        self.reopen_count = 0
        self.ensure_current_count = 0
        self.ensure_missing_count = 0
        self.install(generation, rows)

    def install(
        self,
        generation: int,
        rows: tuple[dict[str, object], ...],
    ) -> PreferenceProjectionTarget:
        target = _target(generation, rows)
        self.snapshots[target] = rows
        self.job = {
            "singleton_id": 1,
            "generation": target.generation,
            "store_identity": STORE_IDENTITY,
            "state": "pending",
            "source_digest": target.source_digest,
            "path_display": None,
            "content_digest": None,
            "last_error_code": None,
        }
        return target

    def get_preference_projection_job(self):
        return self.job

    def ensure_current_preference_projection_job(self):
        self.ensure_current_count += 1
        return self.current_target()

    def ensure_missing_preference_projection_job(self):
        self.ensure_missing_count += 1
        return self.job is not None

    def current_target(self) -> PreferenceProjectionTarget | None:
        if self.job is None:
            return None
        return PreferenceProjectionTarget(
            int(self.job["generation"]),
            str(self.job["source_digest"]),
        )

    def get_current_preference_projection_snapshot(
        self, target: PreferenceProjectionTarget
    ) -> tuple[dict[str, object], ...] | None:
        if target != self.current_target():
            return None
        return self.snapshots.get(target)

    def mark_preference_projection_error(
        self,
        target: PreferenceProjectionTarget,
        error_code: str,
    ) -> bool:
        if self.job is None or self.job["state"] != "pending":
            return False
        if target != self.current_target():
            return False
        self.job["last_error_code"] = error_code
        return True

    def complete_preference_projection(
        self,
        target: PreferenceProjectionTarget,
        path_display: str,
        content_digest: str,
    ) -> bool:
        if self.job is None or self.job["state"] != "pending":
            return False
        if target != self.current_target():
            return False
        self.job.update(
            {
                "state": "completed",
                "path_display": path_display,
                "content_digest": content_digest,
                "last_error_code": None,
            }
        )
        return True

    def reopen_completed_preference_projection(
        self,
        target: PreferenceProjectionTarget,
        expected_content_digest: str,
    ) -> bool:
        if (
            self.job is None
            or self.job["state"] != "completed"
            or target != self.current_target()
            or self.job["content_digest"] != expected_content_digest
        ):
            return False
        self.job.update(
            {
                "state": "pending",
                "path_display": None,
                "content_digest": None,
                "last_error_code": "completed_projection_invalid",
            }
        )
        self.reopen_count += 1
        return True

    def mark_preference_projection_audited(
        self,
        target: PreferenceProjectionTarget,
        expected_content_digest: str,
    ) -> bool:
        if (
            self.job is None
            or self.job["state"] != "completed"
            or target != self.current_target()
            or self.job["content_digest"] != expected_content_digest
        ):
            return False
        self.audit_count += 1
        return True

    def count_pending_preference_projection_jobs(self) -> int:
        return int(self.job is not None and self.job["state"] == "pending")


class MockPreferenceVault:
    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path
        self.path = root_path / PREFERENCE_PROJECTION_PATH_DISPLAY
        self.fail_writes = 0
        self.failure_message = "injected preference projection write failure"
        self.write_generations: list[int] = []
        self.write_snapshots: list[tuple[dict[str, object], ...]] = []

    @staticmethod
    def _content(
        rows: tuple[dict[str, object], ...],
        store_identity: str,
        generation: int,
    ) -> str:
        payload = {
            "generation": generation,
            "rows": [dict(row) for row in rows],
            "store_identity": store_identity,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ) + "\n"

    def write_preferences_with_evidence(
        self,
        rows: tuple[dict[str, object], ...],
        *,
        store_identity: str,
        generation: int,
    ) -> tuple[Path, str]:
        if self.fail_writes:
            self.fail_writes -= 1
            raise OSError(self.failure_message)
        snapshot = tuple(dict(row) for row in rows)
        content = self._content(snapshot, store_identity, generation)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(content, encoding="utf-8")
        self.write_generations.append(generation)
        self.write_snapshots.append(snapshot)
        return self.path, hashlib.sha256(content.encode("utf-8")).hexdigest()

    def verify_preferences_projection_evidence(
        self,
        *,
        store_identity: str,
        generation: int,
        expected_content_digest: str,
    ) -> bool:
        try:
            content = self.path.read_text(encoding="utf-8")
            payload = json.loads(content)
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return False
        return (
            payload.get("store_identity") == store_identity
            and payload.get("generation") == generation
            and hashlib.sha256(content.encode("utf-8")).hexdigest()
            == expected_content_digest
        )


def _assert_content_free(
    values: tuple[object, ...],
    forbidden: tuple[str, ...],
    label: str,
) -> None:
    rendered = "\n".join(str(value or "") for value in values)
    leaked = [value for value in forbidden if value and value in rendered]
    if leaked:
        raise SystemExit(f"{label} leaked source content or a private path: {leaked}")


def _assert_outcome_shape(outcome: PreferenceProjectionOutcome, label: str) -> None:
    if set(vars(outcome)) != {"generation", "status"}:
        raise SystemExit(f"{label} outcome retained projection evidence: {outcome}")


def test_publish_idempotency_privacy_and_full_snapshot() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-projection-full-") as temp:
        root = Path(temp)
        secret = "PRIVATE-PREFERENCE-FULL-SNAPSHOT"
        rows = tuple(_row(index, secret_prefix=secret) for index in range(205))
        store = MockPreferenceStore(7, rows)
        vault = MockPreferenceVault(root / "vault")
        target = store.current_target()
        if target is None:
            raise SystemExit("preference projection fixture did not create a target")
        source_before = preference_projection_source_digest(target.generation, rows)

        outcome = reconcile_preference_projection(store, vault, target)
        if (
            outcome != PreferenceProjectionOutcome(7, "completed")
            or store.job is None
            or store.job["state"] != "completed"
            or store.job["path_display"] != PREFERENCE_PROJECTION_PATH_DISPLAY
            or store.job["content_digest"]
            != hashlib.sha256(vault.path.read_bytes()).hexdigest()
            or vault.write_generations != [7]
            or len(vault.write_snapshots[0]) != 205
            or vault.write_snapshots[0] != rows
            or not any(row["status"] == "retired" for row in vault.write_snapshots[0])
            or vault.write_snapshots[0][-1]["id"] != 205
        ):
            raise SystemExit(f"full preference snapshot publication diverged: {outcome}")
        if preference_projection_source_digest(target.generation, rows) != source_before:
            raise SystemExit("preference reconciliation mutated its source snapshot")

        audited = reconcile_preference_projection(store, vault, target)
        if (
            audited != PreferenceProjectionOutcome(7, "completed")
            or vault.write_generations != [7]
            or store.audit_count != 1
            or store.count_pending_preference_projection_jobs() != 0
        ):
            raise SystemExit(f"completed preference projection was not idempotent: {audited}")
        _assert_outcome_shape(outcome, "published preference")
        _assert_content_free(
            tuple(vars(outcome).values())
            + tuple(vars(audited).values())
            + tuple(store.job.values()),
            (secret, str(root)),
            "published preference job/outcomes",
        )


def test_failure_retention_retry_and_summary_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-projection-retry-") as temp:
        root = Path(temp)
        secret = "PRIVATE-PREFERENCE-RETRY"
        rows = (_row(0, secret_prefix=secret), _row(1, secret_prefix=secret))
        store = MockPreferenceStore(11, rows)
        vault = MockPreferenceVault(root / "vault")
        vault.fail_writes = 1
        vault.failure_message = f"{secret}:{root}"
        source_before = preference_projection_source_digest(11, rows)

        failed = reconcile_pending_preference_projections(store, vault)
        if (
            failed.attempted != 1
            or failed.completed != 0
            or failed.pending != 1
            or failed.outcomes != (PreferenceProjectionOutcome(11, "pending_error"),)
            or store.job is None
            or store.job["state"] != "pending"
            or store.job["last_error_code"] != "vault_publish_failed"
            or store.job["source_digest"] != source_before
        ):
            raise SystemExit(f"preference write failure lost retry custody: {failed}")
        repaired = reconcile_pending_preference_projections(store, vault)
        if (
            repaired.attempted != 1
            or repaired.completed != 1
            or repaired.pending != 0
            or repaired.outcomes != (PreferenceProjectionOutcome(11, "completed"),)
            or store.job["last_error_code"] is not None
            or preference_projection_source_digest(11, rows) != source_before
        ):
            raise SystemExit(f"preference projection retry did not converge: {repaired}")
        _assert_content_free(
            tuple(vars(failed).values())
            + tuple(vars(repaired).values())
            + tuple(store.job.values()),
            (secret, str(root)),
            "preference failure/retry summaries",
        )


def test_exact_target_validation_and_supersession() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-projection-supersession-") as temp:
        root = Path(temp)
        old_rows = (_row(0, secret_prefix="PRIVATE-STALE-PREFERENCE"),)
        latest_rows = (
            _row(0, secret_prefix="PRIVATE-LATEST-PREFERENCE"),
            _row(1, secret_prefix="PRIVATE-LATEST-PREFERENCE"),
        )
        stale_target = _target(20, old_rows)
        store = MockPreferenceStore(21, latest_rows)
        vault = MockPreferenceVault(root / "vault")

        superseded = reconcile_preference_projection(store, vault, stale_target)
        if (
            superseded != PreferenceProjectionOutcome(21, "superseded")
            or vault.write_generations != [21]
            or vault.write_snapshots != [latest_rows]
            or store.job is None
            or store.job["source_digest"]
            != preference_projection_source_digest(21, latest_rows)
        ):
            raise SystemExit(f"stale preference target did not converge forward: {superseded}")

        invalid_targets = (
            PreferenceProjectionTarget(-1, "0" * 64),
            PreferenceProjectionTarget(True, "0" * 64),
            PreferenceProjectionTarget(1, "A" * 64),
        )
        for invalid in invalid_targets:
            try:
                reconcile_preference_projection(store, vault, invalid)
            except ValueError:
                pass
            else:
                raise SystemExit(f"invalid preference projection target was accepted: {invalid}")
        _assert_outcome_shape(superseded, "superseded preference")
        _assert_content_free(
            tuple(vars(superseded).values()) + tuple(store.job.values()),
            ("PRIVATE-STALE-PREFERENCE", "PRIVATE-LATEST-PREFERENCE", str(root)),
            "superseded preference job/outcome",
        )


def test_completed_missing_and_corrupt_byte_repair() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-projection-audit-") as temp:
        root = Path(temp)
        secret = "PRIVATE-PREFERENCE-COMPLETED-AUDIT"
        rows = tuple(_row(index, secret_prefix=secret) for index in range(3))
        store = MockPreferenceStore(30, rows)
        vault = MockPreferenceVault(root / "vault")
        initial = reconcile_preference_projection(store, vault)
        if initial.status != "completed":
            raise SystemExit(f"completed audit fixture did not publish: {initial}")

        vault.path.unlink()
        missing_repair = reconcile_preference_projection(store, vault)
        if (
            missing_repair != PreferenceProjectionOutcome(30, "completed")
            or store.reopen_count != 1
            or len(vault.write_generations) != 2
            or not vault.path.exists()
        ):
            raise SystemExit(f"missing completed preference note was not repaired: {missing_repair}")

        vault.path.write_text(
            vault.path.read_text(encoding="utf-8") + "CORRUPTED-BY-SMOKE\n",
            encoding="utf-8",
        )
        corrupt_repair = reconcile_preference_projection(store, vault)
        if (
            corrupt_repair != PreferenceProjectionOutcome(30, "completed")
            or store.reopen_count != 2
            or len(vault.write_generations) != 3
            or store.job is None
            or store.job["state"] != "completed"
            or store.job["content_digest"]
            != hashlib.sha256(vault.path.read_bytes()).hexdigest()
            or b"CORRUPTED-BY-SMOKE" in vault.path.read_bytes()
        ):
            raise SystemExit(f"corrupt completed preference note was not repaired: {corrupt_repair}")
        _assert_content_free(
            tuple(vars(missing_repair).values())
            + tuple(vars(corrupt_repair).values())
            + tuple(store.job.values()),
            (secret, str(root)),
            "completed preference repair job/outcomes",
        )


def main() -> None:
    _assert_runtime_contract()
    test_publish_idempotency_privacy_and_full_snapshot()
    test_failure_retention_retry_and_summary_privacy()
    test_exact_target_validation_and_supersession()
    test_completed_missing_and_corrupt_byte_repair()
    print("Preference projection reconciliation smoke passed")


if __name__ == "__main__":
    main()
