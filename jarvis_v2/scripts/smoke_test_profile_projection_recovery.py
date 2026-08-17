from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.obsidian import _profile_note_block
from jarvis_v2.memory.profile_projection import (
    reconcile_pending_profile_projections,
    reconcile_profile_projection_candidate,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime


HEADING = "Long-term investing"
BODY = "Prefer durable evidence, controlled downside, and patient decisions."
CATEGORY = "decision-making"


def _seed(root: Path):
    runtime = make_temp_runtime(root)
    result = runtime.registry.get("add_profile_note").handler(
        {"heading": HEADING, "body": BODY, "category": CATEGORY}
    )
    if not result.ok:
        raise AssertionError(f"profile recovery setup failed: {result.output}")
    memory_id = result.metadata.get("memory_id")
    if type(memory_id) is not int or memory_id < 1:
        raise AssertionError(f"profile recovery setup missed memory identity: {result.metadata}")
    with runtime.store.connect() as conn:
        source = conn.execute(
            "SELECT * FROM ingested_sources WHERE source_type = 'profile_note'"
        ).fetchone()
    if source is None:
        raise AssertionError("profile recovery setup missed source custody")
    return runtime, memory_id, dict(source)


def _remove_profile_block(runtime, source_key: str) -> None:
    path = runtime.vault.root_path / "Profile.md"
    content = path.read_text(encoding="utf-8")
    block = _profile_note_block(source_key, HEADING, BODY)
    if block not in content:
        raise AssertionError("profile recovery setup could not find the generated block")
    path.write_text(content.replace(block, "", 1), encoding="utf-8")


def _reopen_source(runtime, source_key: str) -> None:
    with runtime.store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        changed = conn.execute(
            "UPDATE ingested_sources SET mirror_state = 'pending', "
            "mirror_completed_at = NULL WHERE source_key = ?",
            (source_key,),
        )
        if changed.rowcount != 1:
            raise AssertionError("profile recovery setup could not reopen the source")


def test_pending_profile_source_recovers_without_original_command() -> None:
    with TemporaryDirectory() as tmp:
        runtime, memory_id, source = _seed(Path(tmp))
        source_key = str(source["source_key"])
        _remove_profile_block(runtime, source_key)
        _reopen_source(runtime, source_key)

        if runtime.store.count_pending_profile_projection_sources() != 1:
            raise AssertionError("profile recovery did not enumerate the pending source")
        summary = reconcile_pending_profile_projections(runtime.store, runtime.vault)
        if (
            summary.attempted != 1
            or summary.completed != 1
            or summary.pending != 0
            or summary.outcomes[0].memory_id != memory_id
            or summary.outcomes[0].status != "completed"
        ):
            raise AssertionError(f"profile recovery did not converge: {summary}")
        profile = (runtime.vault.root_path / "Profile.md").read_text(encoding="utf-8")
        if (
            profile.count(BODY) != 1
            or profile.count(_profile_note_block(source_key, HEADING, BODY)) != 1
        ):
            raise AssertionError("profile recovery did not restore exactly one owned block")
        snapshot = runtime.store.read_profile_knowledge_snapshot()
        if snapshot.invalid_count != 0 or [note.memory_id for note in snapshot.notes] != [memory_id]:
            raise AssertionError(f"recovered profile custody is not readable: {snapshot}")


def test_startup_recovers_profile_before_memory_summary_finishes() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        runtime, memory_id, source = _seed(root)
        source_key = str(source["source_key"])
        _remove_profile_block(runtime, source_key)
        _reopen_source(runtime, source_key)

        restarted = make_temp_runtime(root)
        status = restarted.memory_projection_recovery_status
        if (
            status.get("status") != "completed"
            or status.get("pending") != 0
            or type(status.get("attempted")) is not int
            or status.get("attempted", 0) < 1
        ):
            raise AssertionError(f"startup missed profile projection recovery: {status}")
        result = restarted.registry.get("read_profile").handler({})
        if (
            not result.ok
            or result.metadata.get("profile_custody_state") != "ok"
            or BODY not in result.output
            or restarted.store.count_pending_profile_projection_sources() != 0
        ):
            raise AssertionError(f"startup recovered incomplete profile evidence: {result}")
        if [note.memory_id for note in restarted.store.read_profile_knowledge_snapshot().notes] != [memory_id]:
            raise AssertionError("startup changed the durable profile memory identity")


def test_existing_memory_repair_command_recovers_profile_source() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _memory_id, source = _seed(Path(tmp))
        source_key = str(source["source_key"])
        _remove_profile_block(runtime, source_key)
        _reopen_source(runtime, source_key)

        result = runtime.registry.get("repair_memory_projections").handler({"limit": 20})
        if (
            not result.ok
            or result.metadata.get("pending") != 0
            or result.metadata.get("attempted", 0) < 1
            or result.metadata.get("writes_database") is not True
            or "profile:completed" not in result.metadata.get("outcome_statuses", [])
            or BODY not in (runtime.vault.root_path / "Profile.md").read_text(encoding="utf-8")
        ):
            raise AssertionError(f"memory repair command missed profile recovery: {result}")


def test_post_publication_failure_retries_without_duplicate_block() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _memory_id, source = _seed(Path(tmp))
        source_key = str(source["source_key"])
        _remove_profile_block(runtime, source_key)
        _reopen_source(runtime, source_key)
        original_lock = runtime.vault.canonical_profile_note_evidence_lock

        @contextmanager
        def unavailable_evidence(**_kwargs):
            yield False

        runtime.vault.canonical_profile_note_evidence_lock = unavailable_evidence
        try:
            failed = reconcile_pending_profile_projections(runtime.store, runtime.vault)
        finally:
            runtime.vault.canonical_profile_note_evidence_lock = original_lock
        if (
            failed.completed != 0
            or failed.pending != 1
            or failed.outcomes[0].status != "pending_error"
            or failed.outcomes[0].file_written is not True
        ):
            raise AssertionError(f"post-publication failure was not retained: {failed}")

        repaired = reconcile_pending_profile_projections(runtime.store, runtime.vault)
        profile = (runtime.vault.root_path / "Profile.md").read_text(encoding="utf-8")
        if (
            repaired.completed != 1
            or repaired.pending != 0
            or profile.count(_profile_note_block(source_key, HEADING, BODY)) != 1
        ):
            raise AssertionError(f"post-publication retry duplicated or lost custody: {repaired}")


def test_unresolved_auto_mutation_blocks_recovery_until_reviewed() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _memory_id, source = _seed(Path(tmp))
        source_key = str(source["source_key"])
        _remove_profile_block(runtime, source_key)
        _reopen_source(runtime, source_key)
        operation_args = {"heading": HEADING, "body": BODY, "category": CATEGORY}
        request_token = "profile-projection-uncertain-owner"
        prepared = runtime.store.prepare_auto_mutation_receipts(
            request_token,
            [("add_profile_note", operation_args)],
        )
        if prepared.status != "PREPARED" or len(prepared.receipts) != 1:
            raise AssertionError(f"profile uncertainty setup did not prepare: {prepared}")
        receipt_id = prepared.receipts[0].receipt_id
        if receipt_id is None:
            raise AssertionError("profile uncertainty setup missed receipt identity")
        claim = runtime.store.claim_auto_mutation_receipt(receipt_id, request_token)
        if claim.status != "CLAIMED" or claim.run_token is None:
            raise AssertionError(f"profile uncertainty setup did not claim: {claim}")
        if not runtime.store.mark_auto_mutation_uncertain(receipt_id, claim.run_token):
            raise AssertionError("profile uncertainty setup did not become uncertain")

        blocked = reconcile_pending_profile_projections(runtime.store, runtime.vault)
        if (
            blocked.completed != 0
            or blocked.pending != 1
            or blocked.outcomes[0].status != "blocked_uncertain_mutation"
            or (runtime.vault.root_path / "Profile.md").read_text(encoding="utf-8").count(BODY)
            != 0
        ):
            raise AssertionError(f"profile recovery bypassed uncertain mutation custody: {blocked}")

        receipt = runtime.store.get_auto_mutation_reconciliation_receipt(receipt_id)
        if receipt is None or type(receipt["uncertainty_digest"]) is not str:
            raise AssertionError("profile uncertainty proof is unavailable")
        resolved = runtime.store.resolve_auto_mutation_receipt(
            receipt_id,
            "confirmed_applied",
            expected_uncertainty_digest=str(receipt["uncertainty_digest"]),
            session_id="profile-recovery-smoke",
            reviewed_at="2026-07-14T00:00:00Z",
            output="Reviewed profile projection custody.",
            metadata={"content_exposed": False},
        )
        if resolved.status != "RESOLVED":
            raise AssertionError(f"profile uncertainty review did not resolve: {resolved}")
        repaired = reconcile_pending_profile_projections(runtime.store, runtime.vault)
        if repaired.completed != 1 or repaired.pending != 0:
            raise AssertionError(f"reviewed profile projection did not recover: {repaired}")


def test_multiple_exact_legacy_blocks_upgrade_together() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _memory_id, first = _seed(Path(tmp))
        second_heading = "Long-term research"
        second_body = "Preserve primary evidence and record uncertainty."
        second_category = "research"
        second_result = runtime.registry.get("add_profile_note").handler(
            {
                "heading": second_heading,
                "body": second_body,
                "category": second_category,
            }
        )
        if not second_result.ok:
            raise AssertionError(f"second legacy profile setup failed: {second_result.output}")
        with runtime.store.connect() as conn:
            rows = list(
                conn.execute(
                    "SELECT source_key FROM ingested_sources "
                    "WHERE source_type = 'profile_note' ORDER BY id"
                )
            )
        if len(rows) != 2:
            raise AssertionError("multiple legacy profile setup missed source rows")
        profile_path = runtime.vault.root_path / "Profile.md"
        raw = profile_path.read_text(encoding="utf-8")
        for row in rows:
            marker = f"jarvis-{row['source_key']}"
            start_marker = marker.replace(
                "jarvis-profile-note:v1:",
                "jarvis-profile-note-start:v1:",
                1,
            )
            raw = raw.replace(f"<!-- {start_marker} -->\n", "", 1)
            _reopen_source(runtime, str(row["source_key"]))
        profile_path.write_text(raw, encoding="utf-8")

        repaired = reconcile_pending_profile_projections(
            runtime.store,
            runtime.vault,
            limit=2,
        )
        upgraded = profile_path.read_text(encoding="utf-8")
        if (
            repaired.completed != 2
            or repaired.pending != 0
            or upgraded.count("<!-- jarvis-profile-note-start:v1:") != 2
            or upgraded.count("<!-- jarvis-profile-note:v1:") != 2
            or upgraded.count(BODY) != 1
            or upgraded.count(second_body) != 1
        ):
            raise AssertionError(f"multiple exact legacy blocks did not upgrade safely: {repaired}")


def test_stale_generation_cannot_reopen_newer_completion() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _memory_id, source = _seed(Path(tmp))
        source_key = str(source["source_key"])
        _remove_profile_block(runtime, source_key)
        _reopen_source(runtime, source_key)
        captured = runtime.store.list_pending_profile_projection_sources(limit=1)[0]

        current = reconcile_profile_projection_candidate(runtime.store, runtime.vault, captured)
        if current.status != "completed":
            raise AssertionError(f"profile generation setup did not complete: {current}")
        with runtime.store.connect() as conn:
            completed = conn.execute(
                "SELECT mirror_state, mirror_generation FROM ingested_sources "
                "WHERE source_key = ?",
                (source_key,),
            ).fetchone()
        if completed is None or completed["mirror_state"] != "completed":
            raise AssertionError("newer profile generation is not complete")
        completed_generation = int(completed["mirror_generation"])

        stale = reconcile_profile_projection_candidate(runtime.store, runtime.vault, captured)
        with runtime.store.connect() as conn:
            after = conn.execute(
                "SELECT mirror_state, mirror_generation FROM ingested_sources "
                "WHERE source_key = ?",
                (source_key,),
            ).fetchone()
        if (
            stale.status != "superseded"
            or after is None
            or after["mirror_state"] != "completed"
            or after["mirror_generation"] != completed_generation
        ):
            raise AssertionError(f"stale profile generation reopened newer custody: {stale} / {after}")


def test_malformed_old_rows_do_not_starve_valid_recovery() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        runtime = make_temp_runtime(root)
        with runtime.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for index in range(1, 21):
                conn.execute(
                    """
                    INSERT INTO ingested_sources(
                        source_key, source_type, title, memory_id,
                        memory_link_required, mirror_state, mirror_completed_at,
                        mirror_generation, created_at
                    ) VALUES (?, 'profile_note', ?, NULL, 1, 'pending', NULL, 0, ?)
                    """,
                    (
                        f"profile-note:v1:{index:064x}",
                        f"Malformed source {index}",
                        "2026-07-14T00:00:00Z",
                    ),
                )
        result = runtime.registry.get("add_profile_note").handler(
            {"heading": HEADING, "body": BODY, "category": CATEGORY}
        )
        if not result.ok:
            raise AssertionError(f"valid profile starvation setup failed: {result.output}")
        with runtime.store.connect() as conn:
            source = conn.execute(
                "SELECT * FROM ingested_sources WHERE memory_id = ?",
                (result.metadata["memory_id"],),
            ).fetchone()
        if source is None:
            raise AssertionError("valid profile starvation setup missed source")
        source_key = str(source["source_key"])
        _remove_profile_block(runtime, source_key)
        _reopen_source(runtime, source_key)

        summary = reconcile_pending_profile_projections(runtime.store, runtime.vault, limit=1)
        if (
            summary.attempted != 1
            or summary.completed != 1
            or summary.outcomes[0].status != "completed"
            or summary.pending != 20
        ):
            raise AssertionError(f"malformed low-id rows starved valid profile recovery: {summary}")


def test_stale_candidate_cannot_overwrite_newer_memory() -> None:
    with TemporaryDirectory() as tmp:
        runtime, memory_id, source = _seed(Path(tmp))
        source_key = str(source["source_key"])
        _reopen_source(runtime, source_key)
        rows = runtime.store.list_pending_profile_projection_sources(limit=1)
        if len(rows) != 1:
            raise AssertionError("profile recovery stale-candidate setup failed")
        row = rows[0]
        newer_body = "A concurrent owner wrote newer profile state."
        with runtime.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE memories SET body = ?, revision = revision + 1, updated_at = updated_at "
                "WHERE id = ?",
                (newer_body, memory_id),
            )
        outcome = reconcile_profile_projection_candidate(runtime.store, runtime.vault, row)
        with runtime.store.connect() as conn:
            current = conn.execute(
                "SELECT body, revision FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        if current is None or current["body"] != newer_body or current["revision"] != row["revision"] + 1:
            raise AssertionError("stale profile recovery overwrote newer source state")
        if outcome.status != "superseded" or runtime.store.count_pending_profile_projection_sources() != 1:
            raise AssertionError(f"stale profile recovery did not fail closed: {outcome}")


def main() -> None:
    test_pending_profile_source_recovers_without_original_command()
    test_startup_recovers_profile_before_memory_summary_finishes()
    test_existing_memory_repair_command_recovers_profile_source()
    test_post_publication_failure_retries_without_duplicate_block()
    test_unresolved_auto_mutation_blocks_recovery_until_reviewed()
    test_multiple_exact_legacy_blocks_upgrade_together()
    test_stale_generation_cannot_reopen_newer_completion()
    test_malformed_old_rows_do_not_starve_valid_recovery()
    test_stale_candidate_cannot_overwrite_newer_memory()
    print("Profile projection recovery smoke passed")


if __name__ == "__main__":
    main()
