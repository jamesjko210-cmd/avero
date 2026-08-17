"""Adversarial history-provenance smokes for conversation compaction.

All summarizers are deterministic in-process fakes. No model or network is used.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.automations.compaction import (
    KEEP_RECENT_HOURS,
    MIN_BATCH_MESSAGES,
    CompactionSummarizer,
    build_conversation_compaction,
)
from jarvis_v2.memory.store import MemoryRecord, MemoryStore, history_source_digest
from jarvis_v2.scripts.smoke_test_conversation_compaction import (
    LOCAL_SUMMARIZER_ROUTE,
    _activate_route,
    _declared,
    _seed_messages,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def _old_stamp() -> str:
    return (
        datetime.now(timezone.utc).replace(tzinfo=None)
        - timedelta(hours=KEEP_RECENT_HOURS + 24)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rotate_epoch_process(db_path, start, attempted, completed, results) -> None:
    store = MemoryStore(Path(db_path))
    if not start.wait(5):
        results.put(("start_timeout", 0.0, ""))
        return
    attempted.set()
    started = time.monotonic()
    try:
        epoch_id = store.start_history_policy_epoch(
            policy_fingerprint="c" * 64,
            provider="test-local",
            model_identifier="rotated-v2",
            destination_class="in_process_test",
            session_generation=777,
            explicit_consent_satisfied=True,
        )
    except Exception as exc:
        results.put((type(exc).__name__, time.monotonic() - started, ""))
    else:
        results.put(("rotated", time.monotonic() - started, epoch_id))
    finally:
        completed.set()


def _receipts(store) -> list[dict]:
    with store.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM history_disclosure_receipts ORDER BY prepared_at, receipt_id"
            )
        ]


def _batches(store) -> list[dict]:
    with store.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM conversation_compaction_batches ORDER BY id"
            )
        ]


def _insert_unprovenanced(store, *, role: str, content: str) -> int:
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, content, metadata, created_at) "
            "VALUES ('smoke', ?, ?, '{}', ?)",
            (role, content, _old_stamp()),
        )
        return int(cur.lastrowid)


def test_maximal_eligible_prefix_stops_at_ineligible_frontier() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-prefix-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(
            runtime.store,
            MIN_BATCH_MESSAGES + 5,
            hours_old=KEEP_RECENT_HOURS + 24,
        )
        frontier_id = _insert_unprovenanced(
            runtime.store,
            role="user",
            content="WITHHELD FRONTIER MUST NOT EGRESS",
        )
        _seed_messages(runtime.store, 8, hours_old=KEEP_RECENT_HOURS + 24)
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE messages SET content = 'LATER HISTORY MUST NOT EGRESS' WHERE id > ?",
                (frontier_id,),
            )
        payloads: list[str] = []
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=_declared(
                lambda transcript: payloads.append(transcript) or "- eligible prefix only"
            ),
        )
        if watermark != frontier_id - 1 or "Compacted 25 messages" not in output:
            raise SystemExit(f"eligible prefix did not stop at its frontier: {output!r} {watermark}")
        joined_payload = "\n".join(payloads)
        if "WITHHELD FRONTIER" in joined_payload or "LATER HISTORY" in joined_payload:
            raise SystemExit(f"frontier or later history crossed dispatch: {joined_payload!r}")
        batch = _batches(runtime.store)[0]
        if batch["eligible_source_count"] != 25 or batch["withheld_source_count"] != 0:
            raise SystemExit(f"batch provenance counts did not describe the exact prefix: {batch!r}")


def test_legacy_first_row_and_malformed_role_advance_without_disclosure() -> None:
    for role, label in (("user", "legacy"), ("system", "malformed-role")):
        with TemporaryDirectory(prefix=f"jarvis-compaction-{label}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            _activate_route(runtime.store, LOCAL_SUMMARIZER_ROUTE)
            _insert_unprovenanced(
                runtime.store,
                role=role,
                content=f"{label} first row must remain local",
            )
            _seed_messages(
                runtime.store,
                MIN_BATCH_MESSAGES + 3,
                hours_old=KEEP_RECENT_HOURS + 24,
            )
            calls: list[str] = []
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                runtime.config,
                summarize=_declared(lambda text: calls.append(text) or "- impossible"),
            )
            if (
                calls
                or watermark != 1
                or runtime.store.conversation_compaction_watermark() != 1
                or "without disclosure" not in output
            ):
                raise SystemExit(f"{label} frontier was not skipped privately: {output!r}")
            batches = _batches(runtime.store)
            if _receipts(runtime.store) or len(batches) != 1:
                raise SystemExit(f"{label} frontier wrote a disclosure receipt or lost its batch")
            batch = batches[0]
            if any(
                (
                    batch["eligible_source_count"] != 0,
                    batch["withheld_source_count"] != 1,
                    batch["source_provenance_digest"] is not None,
                    batch["policy_epoch_id"] is not None,
                )
            ):
                raise SystemExit(f"{label} skip batch retained source lineage: {batch!r}")


def test_undeclared_callback_and_receipt_prepare_failure_block_dispatch() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-callback-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, MIN_BATCH_MESSAGES, hours_old=KEEP_RECENT_HOURS + 24)
        calls: list[str] = []
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=lambda text: calls.append(text) or "- undeclared",
        )
        if "must declare" not in output or calls or watermark is not None:
            raise SystemExit(f"undeclared callback crossed the boundary: {output!r}")

        original_prepare = runtime.store.prepare_history_disclosure_receipt
        runtime.store.prepare_history_disclosure_receipt = (  # type: ignore[assignment]
            lambda **_kwargs: (_ for _ in ()).throw(OSError("synthetic prepare failure"))
        )
        try:
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                runtime.config,
                summarize=_declared(lambda text: calls.append(text) or "- impossible"),
            )
        finally:
            runtime.store.prepare_history_disclosure_receipt = original_prepare  # type: ignore[assignment]
        if "receipt could not be prepared" not in output or calls or watermark is not None:
            raise SystemExit(f"prepare failure crossed dispatch: {output!r}")
        if _receipts(runtime.store) or runtime.store.conversation_compaction_watermark() != 0:
            raise SystemExit("prepare failure wrote a receipt or advanced the watermark")

        original_fence = runtime.store.history_disclosure_egress_fence
        runtime.store.history_disclosure_egress_fence = (  # type: ignore[assignment]
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("synthetic authority fence failure")
            )
        )
        try:
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                runtime.config,
                summarize=_declared(lambda text: calls.append(text) or "- impossible"),
            )
        finally:
            runtime.store.history_disclosure_egress_fence = original_fence  # type: ignore[assignment]
        receipts = _receipts(runtime.store)
        if (
            "authority fence" not in output
            or calls
            or watermark is not None
            or len(receipts) != 1
            or receipts[0]["state"] != "blocked"
        ):
            raise SystemExit(f"fence failure crossed compaction dispatch: {output!r} {receipts!r}")


def test_epoch_switch_before_dispatch_blocks_prepared_receipt() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-epoch-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, MIN_BATCH_MESSAGES, hours_old=KEEP_RECENT_HOURS + 24)
        original_prepare = runtime.store.prepare_history_disclosure_receipt

        def prepare_then_rotate(**kwargs):
            receipt_id = original_prepare(**kwargs)
            runtime.store.start_history_policy_epoch(
                policy_fingerprint="9" * 64,
                provider="test-local",
                model_identifier="deterministic-v1",
                destination_class="in_process_test",
                session_generation=2,
                explicit_consent_satisfied=True,
            )
            return receipt_id

        runtime.store.prepare_history_disclosure_receipt = prepare_then_rotate  # type: ignore[assignment]
        calls: list[str] = []
        try:
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                runtime.config,
                summarize=_declared(lambda text: calls.append(text) or "- impossible"),
            )
        finally:
            runtime.store.prepare_history_disclosure_receipt = original_prepare  # type: ignore[assignment]
        receipts = _receipts(runtime.store)
        if calls or watermark is not None or "authority fence" not in output:
            raise SystemExit(f"epoch switch reached callback or watermark: {output!r}")
        if len(receipts) != 1 or receipts[0]["state"] != "blocked":
            raise SystemExit(f"pre-dispatch epoch race receipt was not blocked: {receipts!r}")


def test_attempted_failure_is_uncertain_and_keeps_watermark() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-uncertain-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, MIN_BATCH_MESSAGES, hours_old=KEEP_RECENT_HOURS + 24)
        calls = 0

        def fail(_text: str) -> str:
            nonlocal calls
            calls += 1
            raise ConnectionError("synthetic summarizer failure")

        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=_declared(fail),
        )
        receipts = _receipts(runtime.store)
        if calls != 1 or "model unavailable" not in output or watermark is not None:
            raise SystemExit(f"attempted failure result drifted: {output!r} {watermark}")
        if len(receipts) != 1 or receipts[0]["state"] != "uncertain":
            raise SystemExit(f"attempted failure receipt was not uncertain: {receipts!r}")
        if runtime.store.conversation_compaction_watermark() != 0 or _batches(runtime.store):
            raise SystemExit("attempted failure advanced the watermark or committed a batch")


def _run_compaction_rotation_at_final_validation(
    *,
    fence_enabled: bool,
) -> tuple[bool, str, int | None, list[dict], str]:
    with TemporaryDirectory(prefix="jarvis-compaction-epoch-final-validation-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, MIN_BATCH_MESSAGES, hours_old=KEEP_RECENT_HOURS + 24)
        original_validate = runtime.store.validate_conversation_compaction_source
        original_finalize = runtime.store.finalize_history_disclosure_receipt
        original_fence = runtime.store.history_disclosure_egress_fence
        ctx = mp.get_context("spawn")
        start = ctx.Event()
        attempted = ctx.Event()
        completed = ctx.Event()
        results = ctx.Queue()
        process = ctx.Process(
            target=_rotate_epoch_process,
            args=(str(runtime.store.db_path), start, attempted, completed, results),
        )
        process.start()
        validate_calls = 0
        final_validation_succeeded = False
        durable_finalization_completed = False
        rotation_completed_before_callback = False

        def validate_then_release_rotator(**kwargs) -> bool:
            nonlocal validate_calls, final_validation_succeeded
            valid = original_validate(**kwargs)
            validate_calls += 1
            if valid:
                final_validation_succeeded = True
                start.set()
                if not attempted.wait(5):
                    raise AssertionError("rotation did not attempt after compaction validation")
                if fence_enabled:
                    if completed.wait(0.2):
                        raise AssertionError(
                            "rotation completed after validation while compaction held the fence"
                        )
                elif not completed.wait(5):
                    raise AssertionError(
                        "fence-disabled compaction rotation did not complete before callback"
                    )
            return valid

        def finalize_before_release(receipt_id: str, state: str):
            nonlocal durable_finalization_completed
            if fence_enabled and completed.is_set():
                raise AssertionError("rotation completed before compaction receipt finalization")
            finalized = original_finalize(receipt_id, state)
            if state == "confirmed":
                durable_finalization_completed = True
            if fence_enabled and completed.is_set():
                raise AssertionError("rotation completed during compaction receipt finalization")
            return finalized

        def summarize_at_boundary(_text: str) -> str:
            nonlocal rotation_completed_before_callback
            if not final_validation_succeeded or not attempted.is_set():
                raise AssertionError("compaction callback ran before final source validation")
            rotation_completed_before_callback = completed.is_set()
            if fence_enabled and rotation_completed_before_callback:
                raise AssertionError("rotation completed before fenced compaction callback")
            if not fence_enabled and not rotation_completed_before_callback:
                raise AssertionError("fence-disabled negative control did not expose the race")
            return "- fenced durable summary"

        runtime.store.validate_conversation_compaction_source = (  # type: ignore[assignment]
            validate_then_release_rotator
        )
        runtime.store.finalize_history_disclosure_receipt = (  # type: ignore[assignment]
            finalize_before_release
        )
        if not fence_enabled:
            runtime.store.history_disclosure_egress_fence = (  # type: ignore[assignment]
                lambda **_kwargs: nullcontext()
            )
        output = ""
        watermark = None
        try:
            try:
                output, watermark = build_conversation_compaction(
                    runtime.store,
                    runtime.vault,
                    runtime.config,
                    summarize=_declared(summarize_at_boundary),
                )
            except Exception as exc:
                if fence_enabled:
                    raise
                output = type(exc).__name__
        finally:
            runtime.store.validate_conversation_compaction_source = (  # type: ignore[assignment]
                original_validate
            )
            runtime.store.finalize_history_disclosure_receipt = (  # type: ignore[assignment]
                original_finalize
            )
            runtime.store.history_disclosure_egress_fence = original_fence  # type: ignore[assignment]
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(2)
            raise SystemExit("compaction epoch rotation deadlocked after callback release")
        status, elapsed, rotated_epoch = results.get(timeout=2)
        if process.exitcode != 0 or status != "rotated":
            raise SystemExit(
                f"compaction rotation did not wait for callback: exit={process.exitcode} "
                f"result={(status, elapsed, rotated_epoch)!r}"
            )
        receipts = _receipts(runtime.store)
        if validate_calls != 1 or not final_validation_succeeded:
            raise SystemExit(f"compaction final validation boundary drifted: {validate_calls}")
        if fence_enabled and not durable_finalization_completed:
            raise SystemExit("compaction did not durably finalize before releasing authority")
        if runtime.store.active_history_policy_epoch_id() != rotated_epoch:
            raise SystemExit("compaction rotation did not activate after callback release")
        return rotation_completed_before_callback, output, watermark, receipts, rotated_epoch


def test_cross_process_rotation_waits_for_compaction_callback_and_finalization() -> None:
    raced, output, watermark, receipts, _rotated_epoch = (
        _run_compaction_rotation_at_final_validation(fence_enabled=True)
    )
    if (
        raced
        or watermark != MIN_BATCH_MESSAGES
        or "Compacted" not in output
        or len(receipts) != 1
        or receipts[0]["state"] != "confirmed"
    ):
        raise SystemExit(f"fenced compaction result drifted: {output!r} {receipts!r}")
    receipt_metadata = repr(receipts)
    if "durable provenance fact" in receipt_metadata or "message 0:" in receipt_metadata:
        raise SystemExit(f"compaction receipt retained source content: {receipts!r}")

    raced_without_fence, _output, _watermark, _receipts, _epoch = (
        _run_compaction_rotation_at_final_validation(fence_enabled=False)
    )
    if not raced_without_fence:
        raise SystemExit("removing the compaction fence did not make the race assertion fail")


def test_restart_epoch_frontier_skips_then_compacts_current_lineage() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-restart-frontier-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 3, hours_old=KEEP_RECENT_HOURS + 24)
        runtime.store.start_history_policy_epoch(
            policy_fingerprint="7" * 64,
            provider=LOCAL_SUMMARIZER_ROUTE.provider,
            model_identifier=LOCAL_SUMMARIZER_ROUTE.model_identifier,
            destination_class=LOCAL_SUMMARIZER_ROUTE.destination_class,
            session_generation=1,
            explicit_consent_satisfied=True,
        )
        _seed_messages(
            runtime.store,
            MIN_BATCH_MESSAGES,
            hours_old=KEEP_RECENT_HOURS + 24,
        )
        calls: list[str] = []
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=_declared(lambda text: calls.append(text) or "- current lineage"),
        )
        if calls or watermark != 3 or "without disclosure" not in output:
            raise SystemExit(f"restart prefix was not privately advanced: {output!r} {watermark}")
        skipped = _batches(runtime.store)[0]
        if any(
            (
                skipped["withheld_source_count"] != 3,
                skipped["eligible_source_count"] != 0,
                skipped["source_provenance_digest"] is not None,
                skipped["policy_epoch_id"] is not None,
            )
        ):
            raise SystemExit(f"restart skip retained or mixed stale lineage: {skipped!r}")
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            last_compacted_id=3,
            summarize=_declared(lambda text: calls.append(text) or "- current lineage"),
        )
        if not calls or watermark != 3 + MIN_BATCH_MESSAGES or "Compacted 20" not in output:
            raise SystemExit(f"current lineage remained stalled after restart skip: {output!r}")
        current = _batches(runtime.store)[1]
        if current["eligible_source_count"] != MIN_BATCH_MESSAGES or current["withheld_source_count"]:
            raise SystemExit(f"current lineage mixed with restart prefix: {current!r}")


def test_message_mutation_races_fail_closed_at_egress_and_commit() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-pre-egress-mutation-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, MIN_BATCH_MESSAGES, hours_old=KEEP_RECENT_HOURS + 24)
        original_prepare = runtime.store.prepare_history_disclosure_receipt
        calls: list[str] = []

        def prepare_then_mutate(**kwargs):
            receipt_id = original_prepare(**kwargs)
            with runtime.store.connect() as conn:
                conn.execute("UPDATE messages SET content = 'MUTATED BEFORE EGRESS' WHERE id = 1")
            return receipt_id

        runtime.store.prepare_history_disclosure_receipt = prepare_then_mutate  # type: ignore[assignment]
        try:
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                runtime.config,
                summarize=_declared(lambda text: calls.append(text) or "- impossible"),
            )
        finally:
            runtime.store.prepare_history_disclosure_receipt = original_prepare  # type: ignore[assignment]
        receipts = _receipts(runtime.store)
        if (
            calls
            or watermark is not None
            or "content changed before dispatch" not in output
            or len(receipts) != 1
            or receipts[0]["state"] != "blocked"
        ):
            raise SystemExit(f"pre-egress mutation crossed disclosure: {output!r} {receipts!r}")
        if runtime.store.conversation_compaction_watermark() != 0 or _batches(runtime.store):
            raise SystemExit("pre-egress mutation advanced or committed compaction")

    with TemporaryDirectory(prefix="jarvis-compaction-pre-commit-mutation-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, MIN_BATCH_MESSAGES, hours_old=KEEP_RECENT_HOURS + 24)

        def mutate_after_egress(_text: str) -> str:
            with runtime.store.connect() as conn:
                conn.execute("UPDATE messages SET content = 'MUTATED BEFORE COMMIT' WHERE id = 1")
            return "- stale summary must not commit"

        try:
            build_conversation_compaction(
                runtime.store,
                runtime.vault,
                runtime.config,
                summarize=_declared(mutate_after_egress),
            )
        except ValueError:
            pass
        else:
            raise SystemExit("pre-commit mutation did not reject stale source lineage")
        receipts = _receipts(runtime.store)
        if len(receipts) != 1 or receipts[0]["state"] != "uncertain":
            raise SystemExit(f"pre-commit mutation receipt was not uncertain: {receipts!r}")
        if runtime.store.conversation_compaction_watermark() != 0 or _batches(runtime.store):
            raise SystemExit("pre-commit mutation advanced or committed compaction")


def test_digest_and_receipt_provenance_are_exact_and_content_free() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-digest-provenance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, MIN_BATCH_MESSAGES, hours_old=KEEP_RECENT_HOURS + 24)
        with runtime.store.connect() as conn:
            sources = list(
                conn.execute(
                    "SELECT message.id, provenance.lineage_token, provenance.content_digest "
                    "FROM messages AS message JOIN message_provenance AS provenance "
                    "ON provenance.message_id = message.id ORDER BY message.id"
                )
            )
        expected_digest = history_source_digest(
            [
                (
                    int(row["id"]),
                    str(row["lineage_token"]),
                    str(row["content_digest"]),
                )
                for row in sources
            ]
        )
        callback_observations: list[tuple[str, str]] = []

        def summarize(transcript: str) -> str:
            receipts = _receipts(runtime.store)
            callback_observations.append((receipts[0]["state"], transcript))
            return "- durable provenance fact"

        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=_declared(summarize),
        )
        if watermark != MIN_BATCH_MESSAGES or "Compacted 20 messages" not in output:
            raise SystemExit(f"digest provenance fixture did not compact: {output!r} {watermark}")
        if callback_observations[0][0] != "prepared":
            raise SystemExit("history reached the callback before its receipt was prepared")
        batch = _batches(runtime.store)[0]
        receipts = _receipts(runtime.store)
        if batch["source_provenance_digest"] != expected_digest:
            raise SystemExit(f"batch source digest was not exact: {batch!r}")
        if receipts[0]["source_digest"] != expected_digest or receipts[0]["state"] != "confirmed":
            raise SystemExit(f"disclosure receipt did not confirm the exact prefix: {receipts!r}")
        memory_id = int(batch["memory_id"])
        provenance = runtime.store.read_memory_provenance(memory_id)
        if provenance is None or any(
            (
                provenance["lineage_state"] != "complete",
                provenance["source_count"] != MIN_BATCH_MESSAGES,
                provenance["source_digest"] != expected_digest,
                provenance["policy_epoch_id"] != batch["policy_epoch_id"],
                provenance["destination_class"] != LOCAL_SUMMARIZER_ROUTE.destination_class,
            )
        ):
            raise SystemExit(f"digest did not inherit exact source provenance: {provenance!r}")
        serialized_provenance = repr((batch, receipts, provenance))
        for forbidden in ("message 0:", "durable provenance fact", "project jarvis"):
            if forbidden in serialized_provenance:
                raise SystemExit(f"content leaked into provenance metadata: {forbidden!r}")


def test_legacy_digest_adoption_remains_unknown_and_local_only() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-legacy-provenance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, MIN_BATCH_MESSAGES, hours_old=KEEP_RECENT_HOURS + 24)
        with runtime.store.connect() as conn:
            rows = list(conn.execute("SELECT id, created_at FROM messages ORDER BY id"))
        span = f"{str(rows[0]['created_at'])[:10]} → {str(rows[-1]['created_at'])[:10]}"
        title = f"Conversation digest {span} (messages #1–#{MIN_BATCH_MESSAGES})"
        memory_id = runtime.store.add_memory(
            MemoryRecord(
                "conversation-digest",
                title,
                "- legacy digest body",
                "conversation_compaction",
                0.8,
            )
        )
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=_declared(
                lambda _text: (_ for _ in ()).throw(
                    AssertionError("legacy adoption must not dispatch")
                )
            ),
        )
        provenance = runtime.store.read_memory_provenance(memory_id)
        if watermark != MIN_BATCH_MESSAGES or "Recovered legacy" not in output:
            raise SystemExit(f"legacy digest was not adopted: {output!r} {watermark}")
        if provenance is None or any(
            (
                provenance["lineage_state"] != "legacy_unknown",
                provenance["destination_class"] != "local-only",
                provenance["remote_eligible"] is not False,
            )
        ):
            raise SystemExit(f"legacy adoption laundered digest provenance: {provenance!r}")
        if _receipts(runtime.store):
            raise SystemExit("legacy adoption prepared an unnecessary disclosure receipt")


def main() -> None:
    test_maximal_eligible_prefix_stops_at_ineligible_frontier()
    test_legacy_first_row_and_malformed_role_advance_without_disclosure()
    test_undeclared_callback_and_receipt_prepare_failure_block_dispatch()
    test_epoch_switch_before_dispatch_blocks_prepared_receipt()
    test_attempted_failure_is_uncertain_and_keeps_watermark()
    test_cross_process_rotation_waits_for_compaction_callback_and_finalization()
    test_restart_epoch_frontier_skips_then_compacts_current_lineage()
    test_message_mutation_races_fail_closed_at_egress_and_commit()
    test_digest_and_receipt_provenance_are_exact_and_content_free()
    test_legacy_digest_adoption_remains_unknown_and_local_only()
    print("Compaction history provenance smoke passed")


if __name__ == "__main__":
    main()
