from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.automations.scheduler import Scheduler, iso
from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def _job(runtime, *, next_run_at: str | None = None) -> tuple[int, object]:
    job_id = runtime.store.upsert_job(
        "State Snapshot",
        360,
        "state_snapshot",
        next_run_at or iso(datetime.now() - timedelta(minutes=1)),
    )
    row = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
    return job_id, row


def _claim(runtime, job_id: int, lease_token: str):
    claimed = runtime.store.claim_job(job_id, lease_token, 60.0)
    if claimed is None:
        raise SystemExit(f"State Snapshot job #{job_id} could not be claimed")
    return claimed


def _stored_job(runtime, job_id: int):
    return next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)


def _assert_occurrence_key(value: object) -> str:
    occurrence_key = str(value or "")
    if re.fullmatch(r"scheduler-occurrence:v1:[0-9a-f]{64}", occurrence_key) is None:
        raise SystemExit(f"scheduler occurrence key is malformed: {occurrence_key!r}")
    return occurrence_key


def _assert_latest_receipt_uncertain(runtime, label: str) -> None:
    with runtime.store.connect() as conn:
        receipt = conn.execute(
            "SELECT state, result, resolution FROM auto_mutation_receipts ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if receipt is None or tuple(receipt) != ("uncertain", "unknown", "manual_review"):
        raise SystemExit(f"{label} receipt was not held uncertain: {receipt}")


def _expected_occurrence_key(
    job_id: int,
    next_run_at: str,
    schedule_revision: int,
    schedule_identity_revision: int,
) -> str:
    material = json.dumps(
        {
            "job_id": job_id,
            "next_run_at": next_run_at,
            "schedule_identity_revision": schedule_identity_revision,
            "schedule_revision": schedule_revision,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "scheduler-occurrence:v1:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def test_active_occurrence_survives_retry_and_clears_on_success() -> None:
    with TemporaryDirectory(prefix="jarvis-state-snapshot-occurrence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        job_id, _ = _job(runtime)

        first = _claim(runtime, job_id, "state-snapshot-first-lease")
        occurrence_key = _assert_occurrence_key(first["active_occurrence_key"])
        retry_at = iso(datetime.now() + timedelta(minutes=5))
        if not runtime.store.mark_claimed_job_run(
            job_id,
            "state-snapshot-first-lease",
            iso(datetime.now()),
            retry_at,
            int(first["schedule_revision"]),
            complete_occurrence=False,
        ):
            raise SystemExit("failed State Snapshot retry was not rescheduled")

        retried_row = _stored_job(runtime, job_id)
        if retried_row["active_occurrence_key"] != occurrence_key:
            raise SystemExit("State Snapshot retry changed its persisted occurrence identity")
        if str(retried_row["next_run_at"]) != retry_at:
            raise SystemExit("State Snapshot retry did not retain its retry schedule")

        second = _claim(runtime, job_id, "state-snapshot-second-lease")
        if second["active_occurrence_key"] != occurrence_key:
            raise SystemExit("State Snapshot retry claim did not reuse its occurrence identity")
        if not runtime.store.mark_claimed_job_run(
            job_id,
            "state-snapshot-second-lease",
            iso(datetime.now()),
            iso(datetime.now() + timedelta(hours=6)),
            int(second["schedule_revision"]),
        ):
            raise SystemExit("successful State Snapshot occurrence was not finalized")
        if _stored_job(runtime, job_id)["active_occurrence_key"] is not None:
            raise SystemExit("successful State Snapshot occurrence identity was not cleared")


def test_completed_occurrence_coalesces_and_audit_is_privacy_safe() -> None:
    with TemporaryDirectory(prefix="jarvis-state-snapshot-coalesce-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        private_canary = "STATE_SNAPSHOT_PRIVATE_CANARY_4f0d"
        runtime.store.add_memory(
            MemoryRecord(
                "private",
                f"{private_canary} {root}",
                "must remain inside the local projection",
                "smoke",
            )
        )
        job_id, _ = _job(runtime)
        first = _claim(runtime, job_id, "state-snapshot-success-lease")
        occurrence_key = _assert_occurrence_key(first["active_occurrence_key"])

        original_write = runtime.vault.write_current_context_with_evidence
        with mock.patch.object(
            runtime.vault,
            "write_current_context_with_evidence",
            wraps=original_write,
        ) as current_context_write:
            first_output = scheduler._run_receipted_state_snapshot(
                first, "state-snapshot-success-lease"
            )
            if current_context_write.call_count != 1:
                raise SystemExit("completed State Snapshot did not publish Current Context once")

            projection_path = root / "Vault" / "Jarvis" / "Memory Tree" / "Current Context.md"
            projection_text = projection_path.read_text(encoding="utf-8")
            if private_canary not in projection_text:
                raise SystemExit("private canary was not actually seeded into Current Context")

            # Simulate losing scheduler finalization after the receipt committed.
            if not runtime.store.release_job_claim(job_id, "state-snapshot-success-lease"):
                raise SystemExit("could not simulate scheduler-finalization loss")
            replay = _claim(runtime, job_id, "state-snapshot-replay-lease")
            if replay["active_occurrence_key"] != occurrence_key:
                raise SystemExit("finalization-loss replay changed the occurrence identity")
            replay_output = scheduler._run_receipted_state_snapshot(
                replay, "state-snapshot-replay-lease"
            )
            if current_context_write.call_count != 1:
                raise SystemExit("completed occurrence replay rewrote Current Context")

        if "already completed" not in replay_output or "no projection was rewritten" not in replay_output:
            raise SystemExit(f"completed receipt did not coalesce safely: {replay_output}")
        tool_runs = [
            row
            for row in runtime.store.recent_tool_runs(limit=10)
            if str(row["tool_name"]) == "export_state_snapshot"
        ]
        if len(tool_runs) != 1:
            raise SystemExit(f"completed occurrence emitted {len(tool_runs)} audit runs instead of one")
        audit = tool_runs[0]
        metadata = json.loads(str(audit["metadata"]))
        scheduled_surface = json.dumps(
            {
                "first_output": first_output,
                "replay_output": replay_output,
                "tool_run_output": audit["output"],
                "tool_run_metadata": metadata,
            },
            sort_keys=True,
        )
        if private_canary in scheduled_surface or str(root) in scheduled_surface:
            raise SystemExit("scheduled State Snapshot output or audit leaked private projection data")
        if "path" in metadata:
            raise SystemExit(f"scheduled State Snapshot audit retained a raw path: {metadata}")
        if metadata.get("path_display") != "Memory Tree/Current Context.md":
            raise SystemExit(f"scheduled State Snapshot lost safe path evidence: {metadata}")
        for key in ("content_sha256", "source_revision"):
            if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key) or "")) is None:
                raise SystemExit(f"scheduled State Snapshot lost {key} evidence: {metadata}")


def test_completed_replay_finalization_preserves_concurrent_reconfiguration() -> None:
    with TemporaryDirectory(prefix="jarvis-state-snapshot-reconfigured-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        job_id, _ = _job(runtime)
        first = _claim(runtime, job_id, "state-snapshot-reconfigure-first")
        old_occurrence = _assert_occurrence_key(first["active_occurrence_key"])
        old_origin_revision = int(first["active_occurrence_schedule_revision"])

        completed_output = scheduler._run_receipted_state_snapshot(
            first, "state-snapshot-reconfigure-first"
        )
        if "occurrence completed" not in completed_output:
            raise SystemExit(f"State Snapshot receipt did not complete before finalization loss: {completed_output}")
        if not runtime.store.release_job_claim(job_id, "state-snapshot-reconfigure-first"):
            raise SystemExit("could not simulate State Snapshot scheduler-finalization loss")

        configured_next_run = iso(datetime.now() + timedelta(days=2))
        runtime.store.upsert_job("State Snapshot", 720, "state_snapshot", configured_next_run)
        reconfigured = _stored_job(runtime, job_id)
        configured_revision = int(reconfigured["schedule_revision"])
        if configured_revision <= old_origin_revision:
            raise SystemExit("concurrent State Snapshot reconfiguration did not advance the revision")
        if reconfigured["active_occurrence_key"] != old_occurrence:
            raise SystemExit("concurrent reconfiguration discarded the unfinished occurrence identity")

        replay = _claim(runtime, job_id, "state-snapshot-reconfigure-replay")
        if replay["active_occurrence_key"] != old_occurrence:
            raise SystemExit("reconfigured replay did not coalesce the completed old occurrence")
        replay_output = scheduler._run_receipted_state_snapshot(
            replay, "state-snapshot-reconfigure-replay"
        )
        if "already completed" not in replay_output or "no projection was rewritten" not in replay_output:
            raise SystemExit(f"reconfigured completed receipt did not coalesce: {replay_output}")

        scheduler_next_run = iso(datetime.now() + timedelta(hours=6))
        finalized = scheduler._finalize_claim_with_delivery(
            replay,
            "state-snapshot-reconfigure-replay",
            datetime.now(),
            scheduler_next_run,
            None,
        )
        if not finalized:
            raise SystemExit("reconfigured completed receipt replay was not scheduler-finalized")
        after_finalization = _stored_job(runtime, job_id)
        if str(after_finalization["next_run_at"]) != configured_next_run:
            raise SystemExit("old occurrence finalization overwrote the concurrently configured next run")
        if int(after_finalization["schedule_revision"]) != configured_revision:
            raise SystemExit("old occurrence finalization changed the concurrently configured revision")
        for field in (
            "active_occurrence_key",
            "active_occurrence_schedule_revision",
            "active_occurrence_next_run_at",
            "active_occurrence_schedule_identity_revision",
        ):
            if after_finalization[field] is not None:
                raise SystemExit(f"old occurrence finalization did not clear {field}")

        genuine = _claim(runtime, job_id, "state-snapshot-reconfigure-genuine")
        genuine_occurrence = _assert_occurrence_key(genuine["active_occurrence_key"])
        expected = _expected_occurrence_key(
            job_id,
            configured_next_run,
            configured_revision,
            int(reconfigured["schedule_identity_revision"]),
        )
        if genuine_occurrence == old_occurrence or genuine_occurrence != expected:
            raise SystemExit("next claim did not create a genuine occurrence from the new configuration")
        if (
            int(genuine["active_occurrence_schedule_revision"]) != configured_revision
            or str(genuine["active_occurrence_next_run_at"]) != configured_next_run
        ):
            raise SystemExit("next genuine occurrence did not retain its configured origin")


def test_uncertain_replay_is_held_and_next_occurrence_can_run() -> None:
    with TemporaryDirectory(prefix="jarvis-state-snapshot-uncertain-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        job_id, _ = _job(runtime)
        first = _claim(runtime, job_id, "state-snapshot-uncertain-lease")
        first_occurrence = _assert_occurrence_key(first["active_occurrence_key"])
        mission_failure = [None] * 7 + [
            lambda _args: ToolResult(
                "export_mission_control",
                False,
                "simulated Mission Control publication failure",
            )
        ]

        original_write = runtime.vault.write_current_context_with_evidence
        with mock.patch.object(
            runtime.vault,
            "write_current_context_with_evidence",
            wraps=original_write,
        ) as current_context_write:
            with mock.patch(
                "jarvis_v2.automations.scheduler.make_next_step_tools",
                return_value=mission_failure,
            ):
                uncertain_output = scheduler._run_receipted_state_snapshot(
                    first, "state-snapshot-uncertain-lease"
                )
            if "status uncertain" not in uncertain_output:
                raise SystemExit(f"partial State Snapshot was not marked uncertain: {uncertain_output}")
            if current_context_write.call_count != 1:
                raise SystemExit("partial State Snapshot did not fail after Current Context publication")

            with runtime.store.connect() as conn:
                uncertain_receipt = conn.execute(
                    "SELECT state, result, resolution FROM auto_mutation_receipts ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if uncertain_receipt is None or tuple(uncertain_receipt) != (
                "uncertain",
                "unknown",
                "manual_review",
            ):
                raise SystemExit(f"partial State Snapshot receipt was not uncertain: {uncertain_receipt}")

            if not runtime.store.release_job_claim(job_id, "state-snapshot-uncertain-lease"):
                raise SystemExit("could not release uncertain State Snapshot claim for replay")
            replay = _claim(runtime, job_id, "state-snapshot-uncertain-replay")
            if replay["active_occurrence_key"] != first_occurrence:
                raise SystemExit("uncertain State Snapshot replay changed occurrence identity")
            replay_output = scheduler._run_receipted_state_snapshot(
                replay, "state-snapshot-uncertain-replay"
            )
            if "held for review" not in replay_output or "no automatic rewrite" not in replay_output:
                raise SystemExit(f"uncertain State Snapshot replay was not held: {replay_output}")
            if current_context_write.call_count != 1:
                raise SystemExit("uncertain State Snapshot replay rewrote Current Context")

            next_run_at = iso(datetime.now() + timedelta(hours=6))
            if not runtime.store.mark_claimed_job_run(
                job_id,
                "state-snapshot-uncertain-replay",
                iso(datetime.now()),
                next_run_at,
                int(replay["schedule_revision"]),
            ):
                raise SystemExit("uncertain State Snapshot occurrence was not scheduler-finalized")
            genuine = _claim(runtime, job_id, "state-snapshot-genuine-next")
            genuine_occurrence = _assert_occurrence_key(genuine["active_occurrence_key"])
            if genuine_occurrence == first_occurrence:
                raise SystemExit("next genuine State Snapshot reused the uncertain occurrence identity")
            next_output = scheduler._run_receipted_state_snapshot(
                genuine, "state-snapshot-genuine-next"
            )
            if "occurrence completed" not in next_output:
                raise SystemExit(f"next genuine State Snapshot did not run: {next_output}")
            if current_context_write.call_count != 2:
                raise SystemExit("next genuine State Snapshot did not publish a fresh Current Context")


def test_uncertainty_record_failure_raises_and_preserves_retry_occurrence() -> None:
    cases = (
        ("false", {"return_value": False}),
        ("raise", {"side_effect": OSError("simulated uncertainty receipt write failure")}),
    )
    for label, uncertainty_behavior in cases:
        with TemporaryDirectory(prefix=f"jarvis-state-snapshot-uncertainty-{label}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
            job_id, _ = _job(runtime)
            lease_token = f"state-snapshot-uncertainty-{label}"
            first = _claim(runtime, job_id, lease_token)
            occurrence_key = _assert_occurrence_key(first["active_occurrence_key"])
            mission_failure = [None] * 7 + [
                lambda _args: ToolResult(
                    "export_mission_control",
                    False,
                    "simulated post-Current-Context Mission Control failure",
                )
            ]

            original_write = runtime.vault.write_current_context_with_evidence
            with mock.patch.object(
                runtime.vault,
                "write_current_context_with_evidence",
                wraps=original_write,
            ) as current_context_write:
                with mock.patch(
                    "jarvis_v2.automations.scheduler.make_next_step_tools",
                    return_value=mission_failure,
                ), mock.patch.object(
                    runtime.store,
                    "mark_auto_mutation_uncertain",
                    **uncertainty_behavior,
                ):
                    try:
                        scheduler._run_receipted_state_snapshot(first, lease_token)
                    except RuntimeError as exc:
                        if str(exc) != "state_snapshot_uncertainty_record_failed":
                            raise SystemExit(
                                f"{label} uncertainty persistence failure raised the wrong error: {exc}"
                            ) from exc
                    else:
                        raise SystemExit(
                            f"{label} uncertainty persistence failure returned a held result"
                        )
                if current_context_write.call_count != 1:
                    raise SystemExit(
                        f"{label} uncertainty persistence case did not fail after Current Context"
                    )

            persisted = _stored_job(runtime, job_id)
            if persisted["active_occurrence_key"] != occurrence_key:
                raise SystemExit(f"{label} uncertainty persistence failure lost the active occurrence")
            retry_at = iso(datetime.now() + timedelta(minutes=5))
            if not runtime.store.mark_claimed_job_run(
                job_id,
                lease_token,
                iso(datetime.now()),
                retry_at,
                int(first["active_occurrence_schedule_revision"]),
                complete_occurrence=False,
            ):
                raise SystemExit(f"{label} uncertainty persistence failure could not schedule retry")
            retry = _claim(runtime, job_id, f"state-snapshot-uncertainty-{label}-retry")
            if retry["active_occurrence_key"] != occurrence_key:
                raise SystemExit(f"{label} uncertainty persistence retry changed occurrence identity")


def test_current_context_publication_rejects_expired_lease() -> None:
    with TemporaryDirectory(prefix="jarvis-state-snapshot-current-context-lease-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(
            runtime.store,
            runtime.vault,
            runtime.config,
            job_lease_seconds=0.15,
        )
        job_id, _ = _job(runtime)
        lease_token = "state-snapshot-current-context-owner"
        claimed = _claim(runtime, job_id, lease_token)
        original_recent = runtime.store.recent_memories

        def delayed_recent(*args, **kwargs):
            time.sleep(0.25)
            return original_recent(*args, **kwargs)

        with mock.patch.object(
            runtime.store,
            "recent_memories",
            side_effect=delayed_recent,
        ), mock.patch.object(
            runtime.vault,
            "write_current_context_with_evidence",
            wraps=runtime.vault.write_current_context_with_evidence,
        ) as current_context_write, mock.patch.object(
            runtime.vault,
            "write_mission_control",
            wraps=runtime.vault.write_mission_control,
        ) as mission_control_write:
            output = scheduler._run_receipted_state_snapshot(claimed, lease_token)

        if "held for review" not in output:
            raise SystemExit(f"expired Current Context owner was not held: {output}")
        if current_context_write.call_count != 0 or mission_control_write.call_count != 0:
            raise SystemExit("expired Current Context owner crossed a snapshot publication boundary")
        _assert_latest_receipt_uncertain(runtime, "Current Context lease loss")


def test_mission_control_publication_rejects_replaced_lease() -> None:
    with TemporaryDirectory(prefix="jarvis-state-snapshot-mission-lease-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(
            runtime.store,
            runtime.vault,
            runtime.config,
            job_lease_seconds=0.5,
        )
        job_id, _ = _job(runtime)
        lease_token = "state-snapshot-mission-owner"
        claimed = _claim(runtime, job_id, lease_token)
        original_pending = runtime.store.list_pending_approvals
        pending_calls = 0

        def replace_during_mission(*args, **kwargs):
            nonlocal pending_calls
            pending_calls += 1
            rows = original_pending(*args, **kwargs)
            if pending_calls == 2:
                time.sleep(0.65)
                replacement = runtime.store.claim_job(
                    job_id,
                    "state-snapshot-mission-replacement",
                    60.0,
                )
                if replacement is None:
                    raise AssertionError("expired Mission Control owner was not replaceable")
            return rows

        with mock.patch.object(
            runtime.store,
            "list_pending_approvals",
            side_effect=replace_during_mission,
        ), mock.patch.object(
            runtime.vault,
            "write_current_context_with_evidence",
            wraps=runtime.vault.write_current_context_with_evidence,
        ) as current_context_write, mock.patch.object(
            runtime.vault,
            "write_mission_control",
            wraps=runtime.vault.write_mission_control,
        ) as mission_control_write:
            output = scheduler._run_receipted_state_snapshot(claimed, lease_token)

        if "held for review" not in output:
            raise SystemExit(f"replaced Mission Control owner was not held: {output}")
        if current_context_write.call_count != 1 or mission_control_write.call_count != 0:
            raise SystemExit("replaced Mission Control owner crossed the wrong publication boundary")
        _assert_latest_receipt_uncertain(runtime, "Mission Control lease loss")


def test_post_mission_lease_loss_blocks_receipt_completion() -> None:
    with TemporaryDirectory(prefix="jarvis-state-snapshot-completion-lease-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(
            runtime.store,
            runtime.vault,
            runtime.config,
            job_lease_seconds=0.5,
        )
        job_id, _ = _job(runtime)
        lease_token = "state-snapshot-completion-owner"
        claimed = _claim(runtime, job_id, lease_token)
        original_mission_write = runtime.vault.write_mission_control

        def replace_after_mission(body: str):
            path = original_mission_write(body)
            time.sleep(0.65)
            replacement = runtime.store.claim_job(
                job_id,
                "state-snapshot-completion-replacement",
                60.0,
            )
            if replacement is None:
                raise AssertionError("post-publication expired owner was not replaceable")
            return path

        with mock.patch.object(
            runtime.vault,
            "write_current_context_with_evidence",
            wraps=runtime.vault.write_current_context_with_evidence,
        ) as current_context_write, mock.patch.object(
            runtime.vault,
            "write_mission_control",
            side_effect=replace_after_mission,
        ) as mission_control_write:
            output = scheduler._run_receipted_state_snapshot(claimed, lease_token)

        if "held for review" not in output:
            raise SystemExit(f"post-Mission lease loss was not held: {output}")
        if current_context_write.call_count != 1 or mission_control_write.call_count != 1:
            raise SystemExit("post-Mission lease fixture did not publish both projections once")
        _assert_latest_receipt_uncertain(runtime, "post-Mission completion lease loss")


def test_corrupt_occurrence_identity_is_regenerated_self_consistently() -> None:
    with TemporaryDirectory(prefix="jarvis-state-snapshot-corrupt-occurrence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        job_id, _ = _job(runtime)
        current = _stored_job(runtime, job_id)
        next_run_at = str(current["next_run_at"])
        schedule_revision = int(current["schedule_revision"])
        with runtime.store.connect() as conn:
            conn.execute(
                """
                UPDATE scheduled_jobs
                SET active_occurrence_key = ?,
                    active_occurrence_schedule_revision = ?,
                    active_occurrence_next_run_at = ?
                WHERE id = ?
                """,
                ("scheduler-occurrence:v1:corrupt", "not-a-revision", "", job_id),
            )

        claimed = _claim(runtime, job_id, "state-snapshot-corrupt-regeneration")
        regenerated = _assert_occurrence_key(claimed["active_occurrence_key"])
        expected = _expected_occurrence_key(
            job_id,
            next_run_at,
            schedule_revision,
            int(current["schedule_identity_revision"]),
        )
        if regenerated != expected:
            raise SystemExit("corrupt State Snapshot occurrence was not regenerated from current schedule")
        if (
            claimed["active_occurrence_schedule_revision"] != schedule_revision
            or claimed["active_occurrence_next_run_at"] != next_run_at
        ):
            raise SystemExit("regenerated State Snapshot occurrence origin is not self-consistent")


def test_legacy_key_without_origin_cannot_consume_current_schedule() -> None:
    with TemporaryDirectory(prefix="jarvis-state-snapshot-legacy-origin-") as temp:
        runtime = make_temp_runtime(Path(temp))
        job_id, _ = _job(runtime)
        original = _stored_job(runtime, job_id)
        legacy_material = json.dumps(
            {
                "job_id": job_id,
                "next_run_at": str(original["next_run_at"]),
                "schedule_revision": int(original["schedule_revision"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        legacy_key = "scheduler-occurrence:v1:" + hashlib.sha256(
            legacy_material.encode("utf-8")
        ).hexdigest()
        with runtime.store.connect() as conn:
            conn.execute(
                """
                UPDATE scheduled_jobs
                SET active_occurrence_key = ?,
                    active_occurrence_schedule_revision = NULL,
                    active_occurrence_next_run_at = NULL
                WHERE id = ?
                """,
                (legacy_key, job_id),
            )
        configured_next_run = iso(datetime.now() + timedelta(days=3))
        runtime.store.upsert_job("State Snapshot", 720, "state_snapshot", configured_next_run)
        configured = _stored_job(runtime, job_id)
        configured_revision = int(configured["schedule_revision"])

        claimed = _claim(runtime, job_id, "state-snapshot-legacy-origin")
        if claimed["active_occurrence_key"] != legacy_key:
            raise SystemExit("legacy occurrence key was not preserved for receipt retirement")
        if claimed["active_occurrence_schedule_revision"] != -1:
            raise SystemExit("legacy occurrence without origin was not revision-quarantined")
        if not runtime.store.mark_claimed_job_run(
            job_id,
            "state-snapshot-legacy-origin",
            iso(datetime.now()),
            iso(datetime.now() + timedelta(hours=1)),
            int(claimed["active_occurrence_schedule_revision"]),
        ):
            raise SystemExit("legacy occurrence could not be retired")
        retired = _stored_job(runtime, job_id)
        if (
            str(retired["next_run_at"]) != configured_next_run
            or int(retired["schedule_revision"]) != configured_revision
        ):
            raise SystemExit("legacy occurrence retirement consumed the current schedule")


def test_infinite_numeric_state_row_is_hidden_without_crashing_snapshot() -> None:
    with TemporaryDirectory(prefix="jarvis-state-snapshot-infinite-") as temp:
        runtime = make_temp_runtime(Path(temp))
        state_tool = runtime.registry.get("export_state_snapshot").handler
        malformed_sessions = [
            {
                "session_id": "malformed-infinite-session",
                "messages": float("inf"),
                "last_at": "2099-01-01T00:00:00Z",
            }
        ]
        with mock.patch.object(runtime.store, "list_sessions", return_value=malformed_sessions):
            result = state_tool({})
        if not result.ok:
            raise SystemExit(f"infinite numeric state row crashed snapshot export: {result.output}")
        if result.metadata.get("unreadable_session_rows") != 1:
            raise SystemExit(f"infinite numeric state row was not safely hidden: {result.metadata}")
        projection = Path(str(result.metadata["path"])).read_text(encoding="utf-8")
        if "Hidden malformed local state rows: 1" not in projection:
            raise SystemExit("snapshot did not report its hidden malformed numeric row")


def main() -> None:
    test_active_occurrence_survives_retry_and_clears_on_success()
    test_completed_occurrence_coalesces_and_audit_is_privacy_safe()
    test_completed_replay_finalization_preserves_concurrent_reconfiguration()
    test_uncertain_replay_is_held_and_next_occurrence_can_run()
    test_uncertainty_record_failure_raises_and_preserves_retry_occurrence()
    test_current_context_publication_rejects_expired_lease()
    test_mission_control_publication_rejects_replaced_lease()
    test_post_mission_lease_loss_blocks_receipt_completion()
    test_corrupt_occurrence_identity_is_regenerated_self_consistently()
    test_legacy_key_without_origin_cannot_consume_current_schedule()
    test_infinite_numeric_state_row_is_hidden_without_crashing_snapshot()
    print("state snapshot scheduler receipts smoke passed")


if __name__ == "__main__":
    main()
