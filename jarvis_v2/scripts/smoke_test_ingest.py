from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, BrokenBarrierError
from unittest import mock

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.automations.scheduler import Scheduler
from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime
from jarvis_v2.tools.ingest import (
    MAX_INBOX_SOURCE_CHARS,
    _ingest_handoff_metadata,
    _metadata_bool,
    make_ingest_tools,
)


def test_planner_routes_ingest_aliases() -> None:
    # Real gaps found live 2026-07-09: "ingest my inbox", "clear my inbox",
    # "show recent file digest", and "what files changed recently" all fell
    # through to chat while their sibling phrasings worked.
    p = RuleBasedPlanner()
    for q, expected_tool in (
        ("ingest inbox", "ingest_obsidian_inbox"),
        ("ingest my inbox", "ingest_obsidian_inbox"),
        ("clear inbox", "clear_obsidian_inbox"),
        ("clear my inbox", "clear_obsidian_inbox"),
        ("recent file digest", "recent_file_digest"),
        ("show recent file digest", "recent_file_digest"),
        ("what files changed recently", "recent_file_digest"),
    ):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != [expected_tool]:
            raise SystemExit(f"{expected_tool} route missed: {q!r} -> {[a.tool_name for a in actions]}")


def assert_vault_relative_receipt(
    result,
    *,
    prefix: str,
    label: str,
    allow_exact_path: bool = False,
) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    response = result.response if hasattr(result, "response") else result.output
    receipt_line = response.split("\n", 1)[0]
    path_text = str(metadata.get("path") or "")
    path_display = metadata.get("path_display")
    if allow_exact_path and (not path_text or not Path(path_text).exists()):
        raise SystemExit(f"{label} should preserve its approved exact saved path metadata: {metadata}")
    if not allow_exact_path and path_text:
        raise SystemExit(f"{label} should not expose exact saved path metadata: {metadata}")
    if path_text and path_text in receipt_line:
        raise SystemExit(f"{label} receipt should not expose the raw local note path.")
    if "/private/" in receipt_line or "/\x55sers/" in receipt_line or "/var/folders/" in receipt_line or "/tmp/" in receipt_line:
        raise SystemExit(f"{label} receipt should not expose local temp or user paths: {receipt_line!r}")
    if not isinstance(path_display, str) or not path_display.startswith(prefix):
        raise SystemExit(f"{label} missed vault-relative path_display metadata: {metadata}")
    if path_display not in receipt_line:
        raise SystemExit(f"{label} receipt should print the vault-relative note label: {receipt_line!r}")


def assert_ingest_memory_note_displays(result, *, label: str) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    response = result.response if hasattr(result, "response") else result.output
    path_displays = metadata.get("path_displays") or []
    if metadata.get("paths") is not None:
        raise SystemExit(f"{label} should not expose exact projection paths: {metadata}")
    if not path_displays:
        raise SystemExit(f"{label} should preserve projection display labels: {metadata}")
    for path_display in path_displays:
        if not isinstance(path_display, str) or not path_display.startswith("Memory Tree/"):
            raise SystemExit(f"{label} should expose vault-relative memory note display paths: {metadata}")
        if path_display not in response:
            raise SystemExit(f"{label} response missed display path {path_display!r}: {response}")
    if "/private/" in response or "/\x55sers/" in response or "/var/folders/" in response or "/tmp/" in response:
        raise SystemExit(f"{label} response leaked a local path: {response}")


def assert_recent_file_digest_body_safe(result, *, root: Path, label: str) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    response = result.response if hasattr(result, "response") else result.output
    if str(root) in response or "/var/folders/" in response or "/private/" in response or "/\x55sers/" in response:
        raise SystemExit(f"{label} response leaked a watched local path: {response}")
    if "Watched directories: 1 configured (paths hidden)" not in response:
        raise SystemExit(f"{label} response should hide watched directory paths: {response}")
    if "[watched 1] recent-note.md" not in response:
        raise SystemExit(f"{label} response should show a relative watched-file label: {response}")
    if metadata.get("watched_dirs") is not None or metadata.get("recent_file_paths") is not None:
        raise SystemExit(f"{label} should not expose exact watched/file path metadata: {metadata}")
    if metadata.get("watched_dir_count") != 1:
        raise SystemExit(f"{label} should expose only the watched-directory count: {metadata}")


def assert_no_local_paths(value, *, label: str) -> None:
    text = repr(value)
    if "/private/" in text or "/\x55sers/" in text or "/var/folders/" in text or "/tmp/" in text:
        raise SystemExit(f"{label} leaked a local path: {text[:1000]}")


def assert_no_future_authority(metadata: dict, *, label: str) -> None:
    for field in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if metadata.get(field) is not False:
            raise SystemExit(f"{label} should keep {field}=False: {metadata}")


def assert_ingest_exact_metadata_bool() -> None:
    for value in ("true", "false", "yes", "0", 1, 0, [], ["value"], None, object()):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"ingest metadata bool should fail closed for {value!r}")
    if _metadata_bool(True) is not True:
        raise SystemExit("ingest metadata bool should preserve exact True")
    if _metadata_bool(False) is not False:
        raise SystemExit("ingest metadata bool should preserve exact False")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("ingest metadata bool should honor the explicit default")


def assert_ingest_malformed_handoff_flags() -> None:
    handoff = {
        "source": "recent_file_digest",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["recent file digest"],
        "boundaries": {"read_only": True},
    }
    metadata = _ingest_handoff_metadata("recent_file_digest_handoff", handoff)
    if metadata.get("state_changed") is not False or metadata.get("recent_file_digest_state_changed") is not False:
        raise SystemExit(f"malformed ingest state_changed should fail closed: {metadata}")
    if metadata.get("content_in_handoff") is not False or metadata.get("recent_file_digest_content_in_handoff") is not False:
        raise SystemExit(f"malformed ingest content_in_handoff should fail closed: {metadata}")


def assert_risk_boundaries(boundaries: dict, *, label: str) -> None:
    for field in ("queues_approval", "controls_computer", "external_side_effect"):
        if boundaries.get(field):
            raise SystemExit(f"{label} should keep {field}=False: {boundaries}")
    assert_no_future_authority(boundaries, label=f"{label} boundaries")


def assert_ingest_contract(
    metadata: dict,
    handoff: dict,
    *,
    label: str,
    prefix: str,
    state_changed: bool,
    changed: list[str],
    content_in_handoff: bool,
) -> None:
    for key, expected in (
        ("handoff_ready", True),
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if key == "handoff_ready":
            if handoff.get(key) != expected or metadata.get(f"{prefix}_handoff_ready") != expected:
                raise SystemExit(f"{label} missed ingest contract {key}={expected}: metadata={metadata} handoff={handoff}")
            continue
        if metadata.get(key) != expected or handoff.get(key) != expected:
            raise SystemExit(f"{label} missed ingest contract {key}={expected}: metadata={metadata} handoff={handoff}")
        prefixed_key = f"{prefix}_{key}"
        if metadata.get(prefixed_key) != expected:
            raise SystemExit(f"{label} missed prefixed ingest contract {prefixed_key}={expected}: metadata={metadata}")
    next_commands = list(handoff.get("next_commands") or [])
    expected_next = next_commands[0] if next_commands else ""
    if (
        handoff.get("next_safe_command") != expected_next
        or metadata.get("next_safe_command") != expected_next
        or metadata.get(f"{prefix}_next_safe_command") != expected_next
    ):
        raise SystemExit(f"{label} missed next_safe_command parity: metadata={metadata} handoff={handoff}")
    if (
        handoff.get("next_safe_commands") != next_commands
        or metadata.get("next_safe_commands") != next_commands
        or metadata.get(f"{prefix}_next_safe_commands") != next_commands
    ):
        raise SystemExit(f"{label} missed next_safe_commands parity: metadata={metadata} handoff={handoff}")
    if (
        handoff.get("next_safe_command_count") != len(next_commands)
        or metadata.get("next_safe_command_count") != len(next_commands)
        or metadata.get(f"{prefix}_next_safe_command_count") != len(next_commands)
    ):
        raise SystemExit(f"{label} missed next_safe_command_count parity: metadata={metadata} handoff={handoff}")


def assert_inbox_ingest_handoff(result, *, label: str) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    assert_no_future_authority(metadata, label=label)
    if metadata.get("reads_personal_data") is not True or metadata.get("reads_private_data") is not True:
        raise SystemExit(f"{label} should mark the attempted inbox read as personal/private: {metadata}")
    handoff = metadata.get("inbox_ingest_handoff")
    if not metadata.get("inbox_ingest_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready inbox ingest handoff: {metadata}")
    if handoff.get("source") != "ingest_obsidian_inbox" or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has the wrong inbox handoff source/readiness: {handoff}")
    memory_writes = bool(metadata.get("writes_memory"))
    note_writes = bool(metadata.get("writes_notes"))
    database_writes = bool(metadata.get("writes_database"))
    changed = []
    if memory_writes:
        changed.append("inbox_memory_import")
    if note_writes:
        changed.append("obsidian_memory_mirror")
    if metadata.get("source_completions") and not memory_writes and not note_writes:
        changed.append("inbox_source_projection_completion")
    if database_writes and not memory_writes and not metadata.get("source_completions"):
        changed.append("inbox_projection_custody_update")
    assert_ingest_contract(
        metadata,
        handoff,
        label=label,
        prefix="inbox_ingest",
        state_changed=database_writes or note_writes,
        changed=changed,
        content_in_handoff=bool(metadata.get("memory_rows")),
    )
    for field in (
        "imported",
        "mirrored",
        "mirror_failures",
        "source_completions",
        "skipped",
        "path_skipped",
        "limit",
        "warnings",
    ):
        if handoff.get(field) != metadata.get(field):
            raise SystemExit(f"{label} handoff {field} diverged from metadata: {handoff} vs {metadata}")
    memory_rows = metadata.get("memory_rows") or []
    if handoff.get("memory_rows") != memory_rows:
        raise SystemExit(f"{label} handoff memory rows diverged from metadata: {handoff} vs {metadata}")
    if handoff.get("memory_ids") != [row["memory_id"] for row in memory_rows]:
        raise SystemExit(f"{label} handoff memory id list is stale: {handoff}")
    for row in memory_rows:
        if (
            row.get("projection_status")
            not in {"published", "repaired", "verified", "pending_error", "superseded", "superseded_or_busy"}
            or type(row.get("revision")) is not int
            or row.get("revision", 0) < 1
            or type(row.get("source_digest_present")) is not bool
            or type(row.get("content_digest_present")) is not bool
        ):
            raise SystemExit(f"{label} has malformed bounded projection evidence: {row}")
    if handoff.get("path_displays") != metadata.get("path_displays", []):
        raise SystemExit(f"{label} handoff path displays diverged from metadata: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("read_only") is not (not (database_writes or note_writes)):
        raise SystemExit(f"{label} inbox read_only boundary is wrong: {boundaries}")
    for field, expected in (
        ("writes_files", note_writes),
        ("writes_database", database_writes),
        ("writes_memory", memory_writes),
        ("writes_notes", note_writes),
    ):
        if boundaries.get(field) is not expected:
            raise SystemExit(f"{label} inbox {field} boundary is wrong: {boundaries}")
    assert_risk_boundaries(boundaries, label=label)
    assert_no_local_paths(handoff, label=label)


def test_inbox_ingest_survives_mirror_failure_idempotently() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-mirror-failure-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        inbox = root / "Vault" / "Jarvis" / "Inbox.md"
        chunk = "Remember this mirror failure must never duplicate durable memory."
        inbox.write_text(f"# Inbox\n\n- {chunk}\n", encoding="utf-8")

        write_calls = 0
        original_write_memory = runtime.vault.write_memory_projection_with_evidence

        def fail_first_write(record, *, memory_id, **kwargs):
            nonlocal write_calls
            write_calls += 1
            if write_calls == 1:
                committed = runtime.store.get_memory(memory_id)
                if committed is None or committed["body"] != chunk:
                    raise AssertionError("projection failure did not occur after the durable memory insert")
                raise OSError(f"mock vault failure at {root / 'private-note.md'}")
            return original_write_memory(record, memory_id=memory_id, **kwargs)

        runtime.vault.write_memory_projection_with_evidence = fail_first_write
        ingest_obsidian_inbox = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]

        first = ingest_obsidian_inbox({})
        if first.ok or first.metadata.get("imported") != 1 or first.metadata.get("mirror_failures") != 1:
            raise SystemExit(f"incomplete projection should fail with durable repair custody: {first}")
        if first.metadata.get("mirrored") != 0 or first.metadata.get("writes_files") or first.metadata.get("writes_notes"):
            raise SystemExit(f"mirror failure should not claim a note write: {first.metadata}")
        memory_rows = first.metadata.get("memory_rows") or []
        if (
            first.metadata.get("paths")
            or first.metadata.get("path_displays")
            or len(memory_rows) != 1
            or memory_rows[0].get("mirror_written") is not False
            or memory_rows[0].get("path_display") != ""
            or memory_rows[0].get("projection_status") != "pending_error"
            or memory_rows[0].get("source_digest_present") is not True
            or memory_rows[0].get("content_digest_present") is not False
        ):
            raise SystemExit(f"projection failure should expose bounded pending evidence: {first.metadata}")
        if not first.metadata.get("writes_database") or not first.metadata.get("writes_memory"):
            raise SystemExit(f"mirror failure should report the committed memory write: {first.metadata}")
        warnings = first.metadata.get("warnings") or []
        if (
            len(warnings) != 1
            or warnings[0].get("code") != "obsidian_memory_mirror_write_failed"
            or warnings[0].get("count") != 1
        ):
            raise SystemExit(f"mirror failure should expose one bounded warning: {first.metadata}")
        if "memory rows are saved" not in first.output.lower() or "pending repair" not in first.output.lower():
            raise SystemExit(f"mirror failure should emit a bounded actionable warning: {first.output!r}")
        assert_no_local_paths(first.output, label="Mirror failure output")
        assert_no_local_paths(first.metadata, label="Mirror failure metadata")
        assert_inbox_ingest_handoff(first, label="Mirror failure ingest")

        with runtime.store.connect() as conn:
            pending = conn.execute(
                "SELECT memory_id, mirror_state FROM ingested_sources WHERE source_type = 'obsidian-inbox'"
            ).fetchone()
        if pending is None or pending["memory_id"] is None or pending["mirror_state"] != "pending":
            raise SystemExit(f"mirror failure should persist a repairable pending marker: {pending}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            retries = list(pool.map(lambda _: ingest_obsidian_inbox({}), range(4)))
        matching = [row for row in runtime.store.list_memories(limit=1000) if row["body"] == chunk]
        if len(matching) != 1:
            raise SystemExit(f"mirror failure retry duplicated durable memory: {matching}")
        if any(result.metadata.get("imported") != 0 for result in retries):
            raise SystemExit(f"mirror retries should not insert another memory row: {retries}")
        if any(result.metadata.get("skipped") != 1 for result in retries):
            raise SystemExit(f"mirror retries should recognize the committed source: {retries}")
        if sum(result.metadata.get("mirrored", 0) for result in retries) != 1:
            raise SystemExit(f"concurrent retries should converge on one mirror write: {retries}")
        repaired = [result for result in retries if result.metadata.get("mirrored") == 1][0]
        if repaired.metadata.get("mirror_failures") != 0 or repaired.metadata.get("warnings"):
            raise SystemExit(f"successful mirror repair should clear the prior failure: {repaired.metadata}")
        if (
            not repaired.metadata.get("writes_files")
            or not repaired.metadata.get("writes_database")
            or repaired.metadata.get("writes_memory")
            or not repaired.metadata.get("writes_notes")
        ):
            raise SystemExit(f"mirror repair should report only mirror/state writes: {repaired.metadata}")
        repaired_rows = repaired.metadata.get("memory_rows") or []
        if (
            len(repaired_rows) != 1
            or repaired_rows[0].get("projection_status") != "published"
            or repaired_rows[0].get("mirror_repaired") is not False
        ):
            raise SystemExit(f"pending publish retry should identify its publication: {repaired.metadata}")
        if write_calls < 2:
            raise SystemExit(f"projection retry never reached the canonical writer: {write_calls}")
        with runtime.store.connect() as conn:
            completed = conn.execute(
                "SELECT mirror_state, mirror_completed_at FROM ingested_sources WHERE source_type = 'obsidian-inbox'"
            ).fetchone()
        if completed is None or completed["mirror_state"] != "completed" or not completed["mirror_completed_at"]:
            raise SystemExit(f"successful mirror repair should durably complete the source: {completed}")
        mirror_paths = list((root / "Vault" / "Jarvis" / "Memory Tree" / "Records").glob(f"{matching[0]['id']:06d} *.md"))
        if len(mirror_paths) != 1:
            raise SystemExit(f"concurrent mirror retries should converge to one note: {mirror_paths}")
        for index, retry in enumerate(retries):
            assert_no_local_paths(retry.output, label=f"Mirror failure retry {index} output")
            assert_inbox_ingest_handoff(retry, label=f"Mirror failure retry {index}")

        writes_before_completed_audit = write_calls
        completed_retry = ingest_obsidian_inbox({})
        if completed_retry.metadata.get("mirrored") or completed_retry.metadata.get("memory_rows"):
            raise SystemExit(f"completed mirrors must not be retried: {completed_retry.metadata}")
        if any(
            completed_retry.metadata.get(field)
            for field in ("writes_files", "writes_database", "writes_memory", "writes_notes")
        ):
            raise SystemExit(f"completed mirror retry should be read-only: {completed_retry.metadata}")
        if completed_retry.metadata.get("skipped") != 1:
            raise SystemExit(f"completed projection should remain a duplicate skip: {completed_retry.metadata}")
        if write_calls != writes_before_completed_audit:
            raise SystemExit("verified duplicate audit unexpectedly invoked the projection writer")
        assert_inbox_ingest_handoff(completed_retry, label="Completed mirror retry")

        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE ingested_sources SET mirror_state = 'pending', mirror_completed_at = NULL "
                "WHERE source_type = 'obsidian-inbox'"
            )
        verified_completion = ingest_obsidian_inbox({})
        verified_rows = verified_completion.metadata.get("memory_rows") or []
        if (
            not verified_completion.ok
            or verified_completion.metadata.get("imported") != 0
            or verified_completion.metadata.get("mirrored") != 0
            or verified_completion.metadata.get("source_completions") != 1
            or verified_completion.metadata.get("writes_files")
            or not verified_completion.metadata.get("writes_database")
            or len(verified_rows) != 1
            or verified_rows[0].get("projection_status") != "verified"
            or verified_rows[0].get("mirror_written") is not False
            or write_calls != writes_before_completed_audit
        ):
            raise SystemExit(f"verified projection did not complete only the pending source ledger: {verified_completion}")
        assert_inbox_ingest_handoff(verified_completion, label="Verified source completion")


def test_inbox_ingest_reports_publish_before_source_completion_failure() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-source-completion-failure-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        inbox = root / "Vault" / "Jarvis" / "Inbox.md"
        chunk = "Remember projection publication and source completion are separate custody stages."
        inbox.write_text(f"# Inbox\n\n- {chunk}\n", encoding="utf-8")
        ingest = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]
        with mock.patch.object(
            runtime.store,
            "complete_ingested_source_projection",
            side_effect=RuntimeError("mock source completion failure with private details"),
        ):
            failed = ingest({})

        rows = failed.metadata.get("memory_rows") or []
        displays = failed.metadata.get("path_displays") or []
        if (
            failed.ok
            or failed.metadata.get("imported") != 1
            or failed.metadata.get("mirrored") != 1
            or failed.metadata.get("mirror_failures") != 1
            or failed.metadata.get("writes_files") is not True
            or failed.metadata.get("writes_notes") is not True
            or failed.metadata.get("writes_database") is not True
            or failed.metadata.get("state_changed") is not True
            or len(rows) != 1
            or rows[0].get("mirror_written") is not True
            or rows[0].get("projection_status") != "pending_error"
            or len(displays) != 1
            or not (runtime.vault.root_path / displays[0]).exists()
        ):
            raise SystemExit(f"post-publication source failure hid completed effects: {failed}")
        if failed.metadata.get("paths") is not None or displays[0] not in failed.output:
            raise SystemExit(f"post-publication failure exposed the wrong path form: {failed}")
        assert_no_local_paths(failed.output, label="Post-publication source failure output")
        assert_inbox_ingest_handoff(failed, label="Post-publication source failure")

        repaired = ingest({})
        if (
            not repaired.ok
            or repaired.metadata.get("imported") != 0
            or repaired.metadata.get("mirrored") != 0
            or repaired.metadata.get("source_completions") != 1
            or repaired.metadata.get("writes_files")
            or repaired.metadata.get("writes_notes")
            or not repaired.metadata.get("writes_database")
        ):
            raise SystemExit(f"source completion retry did not reuse the published evidence: {repaired}")


def test_inbox_ingest_degraded_reconcile_receipts_remain_truthful() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-reconcile-exception-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        inbox = root / "Vault" / "Jarvis" / "Inbox.md"
        inbox.write_text("# Inbox\n\n- Remember unexpected reconciliation failures stay reviewable.\n", encoding="utf-8")
        ingest = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]
        with mock.patch(
            "jarvis_v2.tools.ingest.reconcile_memory_projection",
            side_effect=RuntimeError("mock reconcile failure with private details"),
        ):
            failed = ingest({})
        rows = failed.metadata.get("memory_rows") or []
        if (
            failed.ok
            or failed.metadata.get("imported") != 1
            or failed.metadata.get("mirror_failures") != 1
            or failed.metadata.get("writes_database") is not True
            or failed.metadata.get("writes_files")
            or len(rows) != 1
            or rows[0].get("mirror_written") is not False
            or rows[0].get("projection_status") != "pending_error"
        ):
            raise SystemExit(f"unexpected reconciliation failure lost its bounded receipt: {failed}")
        assert_no_local_paths(failed.metadata, label="Unexpected reconciliation failure metadata")
        assert_inbox_ingest_handoff(failed, label="Unexpected reconciliation failure")

    with TemporaryDirectory(prefix="jarvis-ingest-post-replace-error-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        inbox = root / "Vault" / "Jarvis" / "Inbox.md"
        inbox.write_text("# Inbox\n\n- Remember post-replace durability errors retain file evidence.\n", encoding="utf-8")
        original_write = runtime.vault.write_memory_projection_with_evidence

        def write_then_raise(*args, **kwargs):
            original_write(*args, **kwargs)
            raise OSError("mock durability error after canonical replace")

        runtime.vault.write_memory_projection_with_evidence = write_then_raise
        failed = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]({})
        rows = failed.metadata.get("memory_rows") or []
        displays = failed.metadata.get("path_displays") or []
        if (
            failed.ok
            or failed.metadata.get("imported") != 1
            or failed.metadata.get("mirrored") != 0
            or failed.metadata.get("mirror_failures") != 1
            or failed.metadata.get("writes_files") is not True
            or failed.metadata.get("writes_notes") is not True
            or failed.metadata.get("writes_database") is not True
            or len(rows) != 1
            or rows[0].get("mirror_written") is not True
            or rows[0].get("content_digest_present") is not True
            or len(displays) != 1
            or not (runtime.vault.root_path / displays[0]).exists()
        ):
            raise SystemExit(f"post-replace failure underreported durable bytes: {failed}")
        assert_inbox_ingest_handoff(failed, label="Post-replace durability failure")


def test_concurrent_inbox_ingest_is_transactionally_idempotent() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-concurrent-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        inbox = root / "Vault" / "Jarvis" / "Inbox.md"
        chunk = "Remember synchronized inbox callers must create one durable memory."
        inbox.write_text(f"# Inbox\n\n- {chunk}\n", encoding="utf-8")

        start_barrier = Barrier(2)
        original_add = runtime.store.add_memory_if_source_new_with_projection

        def synchronized_add(*args, **kwargs):
            try:
                start_barrier.wait(timeout=5.0)
            except BrokenBarrierError as exc:
                raise TimeoutError("inbox callers did not reach the transaction together") from exc
            return original_add(*args, **kwargs)

        runtime.store.add_memory_if_source_new_with_projection = synchronized_add  # type: ignore[method-assign]
        ingest_obsidian_inbox = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: ingest_obsidian_inbox({}), range(2)))

        if sorted(result.metadata.get("imported") for result in results) != [0, 1]:
            raise SystemExit(f"concurrent inbox callers should have one insert winner: {results}")
        if sorted(result.metadata.get("skipped") for result in results) != [0, 1]:
            raise SystemExit(f"concurrent inbox callers should have one duplicate skip: {results}")
        if sum(result.metadata.get("mirrored", 0) for result in results) != 1:
            raise SystemExit(f"only the committed inbox winner should write a mirror: {results}")
        if any(not result.ok for result in results):
            raise SystemExit(f"concurrent inbox projection reconciliation did not converge: {results}")
        matching = [row for row in runtime.store.list_memories(limit=1000) if row["body"] == chunk]
        if len(matching) != 1:
            raise SystemExit(f"concurrent inbox callers duplicated durable memory: {matching}")
        with runtime.store.connect() as conn:
            source_count = conn.execute(
                "SELECT COUNT(*) FROM ingested_sources WHERE source_type = 'obsidian-inbox'"
            ).fetchone()[0]
        if source_count != 1:
            raise SystemExit(f"concurrent inbox callers should commit one source marker, got {source_count}")


def test_inbox_memory_and_projection_job_reserve_atomically() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-atomic-projection-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        inbox = runtime.vault.root_path / "Inbox.md"
        inbox.write_text("# Inbox\n\n- Remember projection reservation must be atomic.\n", encoding="utf-8")
        with runtime.store.connect() as conn:
            conn.execute(
                "CREATE TRIGGER fail_inbox_projection_reservation "
                "BEFORE INSERT ON memory_projection_jobs "
                "BEGIN SELECT RAISE(ABORT, 'injected projection reservation failure'); END"
            )
        ingest = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]
        try:
            ingest({})
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("projection reservation failure did not abort inbox ingestion")
        with runtime.store.connect() as conn:
            counts = tuple(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("memories", "ingested_sources", "memory_projection_jobs")
            )
            conn.execute("DROP TRIGGER fail_inbox_projection_reservation")
        if counts != (0, 0, 0):
            raise SystemExit(f"atomic inbox reservation left partial state: {counts}")
        recovered = ingest({})
        if not recovered.ok or recovered.metadata.get("imported") != 1 or recovered.metadata.get("mirrored") != 1:
            raise SystemExit(f"inbox did not recover after atomic reservation rollback: {recovered}")


def test_scheduled_projection_lease_loss_before_and_after_file_write() -> None:
    for rejected_call, expected_file_writes in ((3, 0), (4, 1)):
        with TemporaryDirectory(prefix=f"jarvis-ingest-lease-{rejected_call}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            runtime.vault.root_path.joinpath("Inbox.md").write_text(
                "# Inbox\n\n- Remember scheduled projection leases must fence publication.\n",
                encoding="utf-8",
            )
            scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
            scheduler.schedule_job("Lease-fenced Inbox", 60, "inbox_ingest")
            job = next(row for row in runtime.store.list_jobs() if row["name"] == "Lease-fenced Inbox")
            claimed = scheduler._claim_job(job)
            if claimed is None:
                raise SystemExit("scheduled inbox lease fixture could not claim its job")
            claimed_row, lease_token = claimed
            authority_calls = 0
            file_writes = 0
            original_write = runtime.vault.write_memory_projection_with_evidence

            def authority() -> None:
                nonlocal authority_calls
                authority_calls += 1
                if authority_calls == rejected_call:
                    raise RuntimeError("injected scheduled inbox lease loss")

            def counted_write(*args, **kwargs):
                nonlocal file_writes
                file_writes += 1
                return original_write(*args, **kwargs)

            with mock.patch.object(
                runtime.vault,
                "write_memory_projection_with_evidence",
                side_effect=counted_write,
            ):
                try:
                    scheduler._run_job_type(
                        "inbox_ingest",
                        row=claimed_row,
                        lease_token=lease_token,
                        lease_authority=authority,
                    )
                except RuntimeError:
                    pass
                else:
                    raise SystemExit(f"scheduled authority loss at call {rejected_call} was ignored")
            with runtime.store.connect() as conn:
                source = conn.execute(
                    "SELECT memory_id, mirror_state FROM ingested_sources WHERE source_type = 'obsidian-inbox'"
                ).fetchone()
                projection = conn.execute(
                    "SELECT state FROM memory_projection_jobs"
                ).fetchone()
            if (
                authority_calls != rejected_call
                or file_writes != expected_file_writes
                or source is None
                or source["memory_id"] is None
                or source["mirror_state"] != "pending"
                or projection is None
                or projection["state"] != "pending"
                or len(runtime.store.list_memories(limit=20)) != 1
            ):
                raise SystemExit(
                    "scheduled projection lease loss crossed its durable boundary: "
                    f"call={rejected_call}, authority={authority_calls}, files={file_writes}, "
                    f"source={dict(source) if source else None}, projection={dict(projection) if projection else None}"
                )


def test_completed_prefix_does_not_starve_direct_ingest() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-prefix-direct-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        inbox = root / "Vault" / "Jarvis" / "Inbox.md"
        completed = [f"Remember completed inbox prefix entry {index:02d}." for index in range(50)]
        target = "Remember the actionable fifty-first inbox entry."
        ingest = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]

        inbox.write_text("# Inbox\n\n" + "".join(f"- {chunk}\n" for chunk in completed), encoding="utf-8")
        seeded = ingest({"limit": 50})
        if seeded.metadata.get("imported") != 50 or seeded.metadata.get("mirrored") != 50:
            raise SystemExit(f"direct prefix fixture did not complete fifty sources: {seeded.metadata}")

        inbox.write_text(
            "# Inbox\n\n" + "".join(f"- {chunk}\n" for chunk in [*completed, target]),
            encoding="utf-8",
        )
        result = ingest({"limit": 1})
        if result.metadata.get("imported") != 1 or result.metadata.get("mirrored") != 1:
            raise SystemExit(f"fifty completed sources starved the actionable fifty-first: {result.metadata}")
        if result.metadata.get("skipped") != 50:
            raise SystemExit(f"completed prefix should not consume the actionable limit: {result.metadata}")
        rows = [row for row in runtime.store.list_memories(limit=1000) if row["body"] == target]
        if len(rows) != 1:
            raise SystemExit(f"direct prefix ingest did not create exactly one target memory: {rows}")


def test_projection_tamper_oversize_and_legacy_adoption() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-tampered-projection-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        inbox = root / "Vault" / "Jarvis" / "Inbox.md"
        target = "Remember projection tampering must fail closed."
        inbox.write_text(f"# Inbox\n\n- {target}\n", encoding="utf-8")
        ingest = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]
        first = ingest({})
        destination = runtime.vault.root_path / first.metadata["path_displays"][0]
        tampered = destination.read_text(encoding="utf-8") + "\nmanual private edit\n"
        destination.write_text(tampered, encoding="utf-8")
        second = ingest({})
        rows = second.metadata.get("memory_rows") or []
        if (
            not first.ok
            or second.ok
            or second.metadata.get("imported") != 0
            or second.metadata.get("mirrored") != 0
            or second.metadata.get("mirror_failures") != 1
            or second.metadata.get("writes_database") is not True
            or second.metadata.get("state_changed") is not True
            or len(rows) != 1
            or rows[0].get("projection_status") != "pending_error"
        ):
            raise SystemExit(f"tampered projection did not remain pending and fail closed: {second}")
        if destination.read_text(encoding="utf-8") != tampered:
            raise SystemExit("inbox projection reconciliation overwrote an externally edited note")
        assert_no_local_paths(second.metadata, label="Tampered projection metadata")
        assert_inbox_ingest_handoff(second, label="Tampered projection ingest")

    with TemporaryDirectory(prefix="jarvis-ingest-oversized-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        inbox = root / "Vault" / "Jarvis" / "Inbox.md"
        inbox.write_text("# Inbox\n\n- " + ("x" * MAX_INBOX_SOURCE_CHARS), encoding="utf-8")
        refused = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]({})
        warning_codes = [item.get("code") for item in refused.metadata.get("warnings") or []]
        if refused.ok or refused.metadata.get("imported") != 0:
            raise SystemExit(f"oversized inbox was partially ingested: {refused}")
        if warning_codes != ["obsidian_inbox_source_too_large"]:
            raise SystemExit(f"oversized inbox refusal missed its stable warning: {refused.metadata}")
        if refused.metadata.get("reads_personal_data") is not True or refused.metadata.get("reads_private_data") is not True:
            raise SystemExit(f"oversized inbox refusal hid the attempted private read: {refused.metadata}")
        if runtime.store.list_memories(limit=10):
            raise SystemExit("oversized inbox refusal wrote durable memories")
        inbox.write_text("# Inbox\n", encoding="utf-8")
        empty = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]({})
        if not empty.ok or empty.metadata.get("reads_personal_data") is not True or empty.metadata.get("reads_private_data") is not True:
            raise SystemExit(f"empty inbox result hid its private read: {empty}")
        assert_inbox_ingest_handoff(empty, label="Empty inbox ingest")

    with TemporaryDirectory(prefix="jarvis-ingest-legacy-adoption-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        body = "Remember a linked pre-projection inbox source must be adopted."
        record = MemoryRecord("facts", body.rstrip("."), body, "obsidian-inbox", 0.9)
        memory_id = runtime.store.add_memory(record)
        source_key = "obsidian-inbox:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
        with runtime.store.connect() as conn:
            conn.execute(
                "INSERT INTO ingested_sources("
                "source_key, source_type, title, memory_id, memory_link_required, "
                "mirror_state, created_at) VALUES (?, 'obsidian-inbox', ?, ?, 1, 'pending', ?)",
                (source_key, record.title, memory_id, "2026-07-12T00:00:00+00:00"),
            )
        legacy = runtime.vault.write_memory(
            record,
            memory_id,
            store_identity=runtime.store.get_store_identity(),
        )
        (runtime.vault.root_path / "Inbox.md").write_text(
            f"# Inbox\n\n- {body}\n", encoding="utf-8"
        )
        adopted = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]({})
        if (
            not adopted.ok
            or adopted.metadata.get("imported") != 0
            or adopted.metadata.get("mirrored") != 1
            or adopted.metadata.get("skipped") != 1
            or len(runtime.store.list_memories(limit=20)) != 1
            or legacy.exists()
        ):
            raise SystemExit(f"legacy linked inbox source was not adopted exactly once: {adopted}")
        assert_inbox_ingest_handoff(adopted, label="Legacy source adoption")

    with TemporaryDirectory(prefix="jarvis-ingest-edited-legacy-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        body = "Remember manually edited legacy notes must be retained."
        record = MemoryRecord("facts", body.rstrip("."), body, "obsidian-inbox", 0.9)
        memory_id = runtime.store.add_memory(record)
        source_key = "obsidian-inbox:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
        with runtime.store.connect() as conn:
            conn.execute(
                "INSERT INTO ingested_sources("
                "source_key, source_type, title, memory_id, memory_link_required, "
                "mirror_state, created_at) VALUES (?, 'obsidian-inbox', ?, ?, 1, 'pending', ?)",
                (source_key, record.title, memory_id, "2026-07-12T00:00:00+00:00"),
            )
        legacy = runtime.vault.write_memory(
            record,
            memory_id,
            store_identity=runtime.store.get_store_identity(),
        )
        edited_legacy = legacy.read_text(encoding="utf-8") + "\nmanual addition\n"
        legacy.write_text(edited_legacy, encoding="utf-8")
        (runtime.vault.root_path / "Inbox.md").write_text(
            f"# Inbox\n\n- {body}\n", encoding="utf-8"
        )
        refused = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]({})
        rows = refused.metadata.get("memory_rows") or []
        if (
            refused.ok
            or refused.metadata.get("imported") != 0
            or refused.metadata.get("mirrored") != 0
            or refused.metadata.get("mirror_failures") != 1
            or refused.metadata.get("writes_files") is not True
            or refused.metadata.get("writes_notes") is not True
            or refused.metadata.get("writes_database") is not True
            or len(rows) != 1
            or rows[0].get("mirror_written") is not True
            or rows[0].get("projection_status") != "pending_error"
            or legacy.read_text(encoding="utf-8") != edited_legacy
            or len(refused.metadata.get("path_displays") or []) != 1
            or not (
                runtime.vault.root_path / refused.metadata["path_displays"][0]
            ).exists()
        ):
            raise SystemExit(f"edited legacy cleanup hid or destroyed partial effects: {refused}")
        assert_inbox_ingest_handoff(refused, label="Edited legacy cleanup refusal")


def test_scheduler_repairs_missing_projection_beyond_completed_prefix_once() -> None:
    with TemporaryDirectory(prefix="jarvis-ingest-prefix-scheduler-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        inbox = root / "Vault" / "Jarvis" / "Inbox.md"
        completed = [f"Remember scheduler prefix entry {index:02d} is complete." for index in range(50)]
        target = "Remember scheduler mirror repair beyond the completed prefix."
        chunks = [*completed, target]
        inbox.write_text("# Inbox\n\n" + "".join(f"- {chunk}\n" for chunk in chunks), encoding="utf-8")
        ingest = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[0]
        seeded = ingest({"limit": 51})
        if seeded.metadata.get("imported") != 51 or seeded.metadata.get("mirrored") != 51:
            raise SystemExit(f"scheduler prefix fixture did not complete all sources: {seeded.metadata}")

        source_key = "obsidian-inbox:" + hashlib.sha256(target.encode("utf-8")).hexdigest()
        with runtime.store.connect() as conn:
            projection = conn.execute(
                "SELECT job.canonical_path_display FROM ingested_sources AS source "
                "JOIN memory_projection_jobs AS job ON job.memory_id = source.memory_id "
                "WHERE source.source_key = ?",
                (source_key,),
            ).fetchone()
        if projection is None or not projection["canonical_path_display"]:
            raise SystemExit("scheduler prefix fixture has no completed projection evidence")
        missing_note = runtime.vault.root_path / projection["canonical_path_display"]
        missing_note.unlink()

        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        scheduler.schedule_job("Prefix Inbox Ingest", 60, "inbox_ingest")
        target_write_calls = 0
        original_write_memory = runtime.vault.write_memory_projection_with_evidence

        def counted_write(record, *, memory_id, **kwargs):
            nonlocal target_write_calls
            if record.body == target:
                target_write_calls += 1
            return original_write_memory(record, memory_id=memory_id, **kwargs)

        with mock.patch.object(runtime.vault, "write_memory_projection_with_evidence", side_effect=counted_write):
            first = scheduler.run_job_now("Prefix Inbox Ingest")
            second = scheduler.run_job_now("Prefix Inbox Ingest")
        if "wrote 1 Obsidian mirror(s)" not in first or "wrote 0 Obsidian mirror(s)" not in second:
            raise SystemExit(f"scheduler runs did not repair then verify the missing projection: {first!r} {second!r}")
        if target_write_calls != 1:
            raise SystemExit(f"scheduler should repair the beyond-prefix mirror exactly once: {target_write_calls}")
        with runtime.store.connect() as conn:
            repaired = conn.execute(
                "SELECT mirror_state, mirror_completed_at FROM ingested_sources WHERE source_key = ?",
                (source_key,),
            ).fetchone()
        if repaired is None or repaired["mirror_state"] != "completed" or not repaired["mirror_completed_at"]:
            raise SystemExit(f"scheduler did not durably complete the beyond-prefix mirror: {repaired}")


def test_recent_file_digest_survives_post_append_file_races_without_duplicates() -> None:
    for race_mode in ("delete", "permission"):
        with TemporaryDirectory(prefix=f"jarvis-digest-{race_mode}-race-") as temp:
            root = Path(temp)
            runtime = make_temp_runtime(root)
            watched = root / "Watched"
            watched.mkdir()
            recent = watched / "recent-note.md"
            recent.write_text("bounded synthetic digest content", encoding="utf-8")
            recent_file_digest = make_ingest_tools(runtime.store, runtime.vault, runtime.config)[2]

            append_calls = 0
            append_committed = False
            original_append_daily = runtime.vault.append_daily
            original_stat = Path.stat

            def raced_stat(path, *args, **kwargs):
                if race_mode == "permission" and append_committed and path == recent:
                    raise PermissionError("mock watched-file permission loss after append")
                return original_stat(path, *args, **kwargs)

            def append_then_race(heading, body):
                nonlocal append_calls, append_committed
                append_calls += 1
                note_path = original_append_daily(heading, body)
                append_committed = True
                if race_mode == "delete" and recent.exists():
                    recent.unlink()
                return note_path

            runtime.vault.append_daily = append_then_race
            Path.stat = raced_stat  # type: ignore[method-assign]
            try:
                try:
                    result = recent_file_digest({})
                except OSError:
                    result = recent_file_digest({})
            finally:
                Path.stat = original_stat  # type: ignore[method-assign]

            if not result.ok or result.metadata.get("count") != 1:
                raise SystemExit(f"{race_mode} race should return the committed digest receipt: {result}")
            recent_rows = result.metadata.get("recent_file_rows") or []
            if len(recent_rows) != 1 or recent_rows[0].get("size_bytes") != len(
                "bounded synthetic digest content"
            ):
                raise SystemExit(f"{race_mode} race lost the pre-append metadata snapshot: {result.metadata}")
            if append_calls != 1:
                raise SystemExit(f"{race_mode} race triggered a duplicate append retry: {append_calls}")
            daily_text = (
                runtime.vault.root_path / result.metadata["path_display"]
            ).read_text(encoding="utf-8")
            if daily_text.count("## Jarvis Recent File Digest") != 1:
                raise SystemExit(f"{race_mode} race duplicated the digest note:\n{daily_text}")
            assert_recent_file_digest_handoff(result, label=f"Digest {race_mode} race")


def assert_recent_file_digest_handoff(result, *, label: str, writes: bool = True) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    assert_no_future_authority(metadata, label=label)
    handoff = metadata.get("recent_file_digest_handoff")
    if not metadata.get("recent_file_digest_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready recent file digest handoff: {metadata}")
    if handoff.get("source") != "recent_file_digest" or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has the wrong digest handoff source/readiness: {handoff}")
    assert_ingest_contract(
        metadata,
        handoff,
        label=label,
        prefix="recent_file_digest",
        state_changed=writes,
        changed=["recent_file_digest_export"] if writes else [],
        content_in_handoff=bool(metadata.get("recent_file_rows")),
    )
    for field in ("count", "skipped", "hours", "limit", "path_display"):
        if handoff.get(field) != metadata.get(field, ""):
            raise SystemExit(f"{label} digest handoff {field} diverged from metadata: {handoff} vs {metadata}")
    if handoff.get("recent_files") != metadata.get("recent_file_rows", []):
        raise SystemExit(f"{label} digest recent rows diverged from metadata: {handoff} vs {metadata}")
    if handoff.get("watched_dir_count") != metadata.get("watched_dir_count", 0):
        raise SystemExit(f"{label} digest watched count diverged from metadata: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("read_only") is not (not writes):
        raise SystemExit(f"{label} digest read_only boundary is wrong: {boundaries}")
    if boundaries.get("reads_file_metadata") is not True:
        raise SystemExit(f"{label} digest should mark metadata reads: {boundaries}")
    if boundaries.get("writes_files") is not writes or boundaries.get("writes_notes") is not writes:
        raise SystemExit(f"{label} digest write boundaries are wrong: {boundaries}")
    if boundaries.get("writes_database") or boundaries.get("writes_memory"):
        raise SystemExit(f"{label} digest should not mark database/memory writes: {boundaries}")
    assert_risk_boundaries(boundaries, label=label)
    assert_no_local_paths(handoff, label=label)


def assert_clear_inbox_handoff(result, *, label: str) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    assert_no_future_authority(metadata, label=label)
    handoff = metadata.get("clear_inbox_handoff")
    if not metadata.get("clear_inbox_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready clear inbox handoff: {metadata}")
    if handoff.get("source") != "clear_obsidian_inbox" or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has the wrong clear handoff source/readiness: {handoff}")
    assert_ingest_contract(
        metadata,
        handoff,
        label=label,
        prefix="clear_inbox",
        state_changed=True,
        changed=["inbox_clear"],
        content_in_handoff=False,
    )
    if handoff.get("path_display") != metadata.get("path_display"):
        raise SystemExit(f"{label} clear handoff path display diverged from metadata: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("read_only") or not boundaries.get("writes_files") or not boundaries.get("writes_notes"):
        raise SystemExit(f"{label} clear handoff should mark local note writes: {boundaries}")
    if boundaries.get("writes_database") or boundaries.get("writes_memory"):
        raise SystemExit(f"{label} clear handoff should not mark database/memory writes: {boundaries}")
    assert_risk_boundaries(boundaries, label=label)
    assert_no_local_paths(handoff, label=label)


def main() -> None:
    test_planner_routes_ingest_aliases()
    assert_ingest_exact_metadata_bool()
    assert_ingest_malformed_handoff_flags()
    test_concurrent_inbox_ingest_is_transactionally_idempotent()
    test_inbox_memory_and_projection_job_reserve_atomically()
    test_inbox_ingest_survives_mirror_failure_idempotently()
    test_inbox_ingest_reports_publish_before_source_completion_failure()
    test_inbox_ingest_degraded_reconcile_receipts_remain_truthful()
    test_scheduled_projection_lease_loss_before_and_after_file_write()
    test_completed_prefix_does_not_starve_direct_ingest()
    test_projection_tamper_oversize_and_legacy_adoption()
    test_scheduler_repairs_missing_projection_beyond_completed_prefix_once()
    test_recent_file_digest_survives_post_append_file_races_without_duplicates()
    with TemporaryDirectory(prefix="jarvis-ingest-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        ingest_obsidian_inbox, clear_obsidian_inbox, recent_file_digest = make_ingest_tools(
            runtime.store,
            runtime.vault,
            runtime.config,
        )
        inbox = root / "Vault" / "Jarvis" / "Inbox.md"
        inbox.write_text(
            "# Inbox\n\n"
            "- the operator prefers Jarvis responses to be direct but warm.\n"
            "- Goal: make Jarvis a full personal assistant with memory, voice, and computer control.\n",
            encoding="utf-8",
        )
        watched = root / "Watched"
        watched.mkdir()
        (watched / "recent-note.md").write_text("recent file digest smoke test", encoding="utf-8")

        cases = [
            ("list tools ingest", False),
            ("ingest inbox", False),
            ("ingest inbox", False),
            ("search memory for direct warm", False),
            ("recent file digest", False),
            ("schedule inbox ingest", False),
            ("schedule file digest", False),
            ("run job Inbox Ingest now", False),
            ("run job Recent File Digest now", False),
            ("clear inbox", False),
            ("clear inbox", True),
        ]
        for case, approved in cases:
            result = handle_runtime_case(runtime, case, approved=approved)
            status = "ok" if result.verified else "blocked"
            approval = " approved" if approved else ""
            print(f"[{status}{approval}] {case}")
            print(result.response[:1200])
            print()
            if case == "recent file digest":
                assert_vault_relative_receipt(result, prefix="Daily/", label="Recent file digest")
                assert_recent_file_digest_body_safe(result, root=root, label="Runtime recent file digest")
                assert_recent_file_digest_handoff(result, label="Runtime recent file digest")
                assert_no_local_paths(result, label="Runtime recent file digest result")
            if case == "ingest inbox" and result.tool_results[0].metadata.get("imported"):
                assert_ingest_memory_note_displays(result, label="Runtime ingest inbox")
                assert_inbox_ingest_handoff(result, label="Runtime ingest inbox")
                assert_no_local_paths(result, label="Runtime ingest inbox result")
            if case == "clear inbox" and approved:
                assert_vault_relative_receipt(
                    result,
                    prefix="Inbox.md",
                    label="Cleared Obsidian Inbox",
                    allow_exact_path=True,
                )
                assert_clear_inbox_handoff(result, label="Approved clear inbox")

        inbox.write_text(
            "# Inbox\n\n"
            "- Jarvis should bound bad ingest limits safely.\n"
            "- Goal: keep ingest local and reviewable.\n",
            encoding="utf-8",
        )
        ingest_result = ingest_obsidian_inbox({"limit": "not-a-number"})
        if ingest_result.metadata.get("limit") != 50 or not ingest_result.metadata.get("writes_files"):
            raise SystemExit("ingest_obsidian_inbox should sanitize bad limits and mark writes when imported.")
        if ingest_result.metadata.get("queues_approval") or ingest_result.metadata.get("controls_computer"):
            raise SystemExit("ingest_obsidian_inbox should not queue approvals or control the computer.")
        assert_ingest_memory_note_displays(ingest_result, label="Direct ingest_obsidian_inbox")
        assert_inbox_ingest_handoff(ingest_result, label="Direct ingest_obsidian_inbox")

        duplicate_result = ingest_obsidian_inbox({"limit": 999999})
        if duplicate_result.metadata.get("limit") != 200:
            raise SystemExit("ingest_obsidian_inbox should clamp huge limits.")
        if duplicate_result.metadata.get("imported") != 0 or duplicate_result.metadata.get("writes_database"):
            raise SystemExit("duplicate ingest should not mark database writes.")
        assert_inbox_ingest_handoff(duplicate_result, label="Duplicate ingest_obsidian_inbox")

        before_path_only_count = len(runtime.store.list_memories(limit=1000))
        inbox.write_text(
            "# Inbox\n\n"
            "- remember: /\x55sers/example/Desktop/Claude code/private-plan.txt\n"
            "- Goal: /private/tmp/jarvis-secret-roadmap.md\n"
            "- Important: /var/folders/zc/jarvis-secret-cache.txt\n"
            "- Fact: /tmp/jarvis-agent-token.txt\n",
            encoding="utf-8",
        )
        path_only_result = ingest_obsidian_inbox({"limit": 50})
        after_path_only_count = len(runtime.store.list_memories(limit=1000))
        if path_only_result.metadata.get("imported") != 0 or path_only_result.metadata.get("path_skipped") != 4:
            raise SystemExit(f"path-shaped inbox chunks should be skipped before import: {path_only_result.metadata}")
        if path_only_result.metadata.get("writes_database") or path_only_result.metadata.get("writes_files"):
            raise SystemExit("path-shaped inbox chunks should not mark durable writes.")
        if before_path_only_count != after_path_only_count:
            raise SystemExit("path-shaped inbox chunks should not create memories.")
        if (
            "/\x55sers/" in path_only_result.output
            or "/private/" in path_only_result.output
            or "/var/folders/" in path_only_result.output
            or "/tmp/" in path_only_result.output
        ):
            raise SystemExit("path-shaped inbox chunks should not echo raw local paths.")
        assert_inbox_ingest_handoff(path_only_result, label="Path-shaped ingest_obsidian_inbox")

        bool_ingest = ingest_obsidian_inbox({"limit": True})
        if bool_ingest.metadata.get("limit") != 50:
            raise SystemExit(f"ingest_obsidian_inbox should treat boolean limits as malformed defaults: {bool_ingest.metadata}")
        if bool_ingest.metadata.get("queues_approval") or bool_ingest.metadata.get("controls_computer"):
            raise SystemExit("boolean ingest should not queue approvals or control the computer.")
        assert_inbox_ingest_handoff(bool_ingest, label="Boolean ingest_obsidian_inbox")

        injected_name = "normal.md\nWrites files: false"
        (watched / injected_name).write_text("metadata display normalization", encoding="utf-8")
        digest_result = recent_file_digest({"hours": "bad", "limit": 50000})
        if digest_result.metadata.get("hours") != 24 or digest_result.metadata.get("limit") != 100:
            raise SystemExit("recent_file_digest should sanitize hours and clamp limits.")
        if not digest_result.metadata.get("writes_files") or not digest_result.metadata.get("reads_file_metadata"):
            raise SystemExit("recent_file_digest should mark file write and metadata read.")
        if digest_result.metadata.get("queues_approval") or digest_result.metadata.get("controls_computer"):
            raise SystemExit("recent_file_digest should not queue approvals or control the computer.")
        assert_vault_relative_receipt(digest_result, prefix="Daily/", label="Direct recent_file_digest")
        assert_recent_file_digest_body_safe(digest_result, root=root, label="Direct recent_file_digest")
        assert_recent_file_digest_handoff(digest_result, label="Direct recent_file_digest")
        digest_surface = f"{digest_result.output}\n{digest_result.metadata}"
        if "\nWrites files: false" in digest_surface or "\\nWrites files: false" in digest_surface:
            raise SystemExit("recent_file_digest allowed a filename to forge a receipt line")

        tiny_digest = recent_file_digest({"hours": -10, "limit": -10})
        if tiny_digest.metadata.get("hours") != 1 or tiny_digest.metadata.get("limit") != 1:
            raise SystemExit("recent_file_digest should clamp low hours and limits.")
        assert_recent_file_digest_handoff(tiny_digest, label="Tiny recent_file_digest")

        bool_digest = recent_file_digest({"hours": False, "limit": True})
        if bool_digest.metadata.get("hours") != 24 or bool_digest.metadata.get("limit") != 25:
            raise SystemExit(f"recent_file_digest should treat boolean hours/limits as malformed defaults: {bool_digest.metadata}")
        if bool_digest.metadata.get("queues_approval") or bool_digest.metadata.get("controls_computer"):
            raise SystemExit("boolean recent_file_digest should not queue approvals or control the computer.")
        assert_recent_file_digest_handoff(bool_digest, label="Boolean recent_file_digest")

        no_watch_ingest_tools = make_ingest_tools(runtime.store, runtime.vault, None)
        no_watch_digest = no_watch_ingest_tools[2]({})
        if no_watch_digest.metadata.get("count") != 0 or no_watch_digest.metadata.get("writes_files"):
            raise SystemExit(f"recent_file_digest without watched dirs should stay read-only/no-write: {no_watch_digest.metadata}")
        assert_recent_file_digest_handoff(no_watch_digest, label="No-watch recent_file_digest", writes=False)

        clear_binding = runtime.vault.inbox_clear_binding(
            max_bytes=MAX_INBOX_SOURCE_CHARS
        )
        if clear_binding is None:
            raise SystemExit("direct clear smoke could not bind Inbox.md")
        clear_result = clear_obsidian_inbox({"target_binding": clear_binding})
        if not clear_result.metadata.get("writes_files"):
            raise SystemExit("clear_obsidian_inbox should mark file writes.")
        if clear_result.metadata.get("queues_approval") or clear_result.metadata.get("controls_computer"):
            raise SystemExit("clear_obsidian_inbox should not queue approvals or control the computer.")
        assert_vault_relative_receipt(
            clear_result,
            prefix="Inbox.md",
            label="Direct clear_obsidian_inbox",
            allow_exact_path=True,
        )
        assert_clear_inbox_handoff(clear_result, label="Direct clear_obsidian_inbox")


if __name__ == "__main__":
    main()
