from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime


HEADING = "Investment philosophy"
BODY = "Prefer durable evidence, explicit downside, and patient decisions."
CATEGORY = "decision-making"
MANUAL_MARKER = "MANUAL_PROFILE_CUSTODY_MARKER"
ATTACK_MARKER = "UNTRUSTED_PROFILE_CUSTODY_MARKER"
LOCAL_PATH_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")


def _write_manual_profile(runtime: Any) -> None:
    path = runtime.vault.root_path / "Profile.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Profile\n\n## Manual context\n\n{MANUAL_MARKER}\n",
        encoding="utf-8",
    )


def _add_note(runtime: Any):
    result = runtime.registry.get("add_profile_note").handler(
        {"heading": HEADING, "body": BODY, "category": CATEGORY}
    )
    if not result.ok:
        raise AssertionError(f"profile custody setup failed: {result.output}")
    return result


def _fixture(root: Path) -> tuple[Any, Any]:
    runtime = make_temp_runtime(root)
    _write_manual_profile(runtime)
    result = _add_note(runtime)
    return runtime, result


def _row(runtime: Any, query: str, params: tuple[object, ...] = ()) -> dict[str, Any]:
    with runtime.store.connect() as conn:
        row = conn.execute(query, params).fetchone()
    if row is None:
        raise AssertionError(f"profile custody evidence row is missing: {query}")
    return dict(row)


def _source_row(runtime: Any) -> dict[str, Any]:
    return _row(
        runtime,
        "SELECT * FROM ingested_sources WHERE source_type = 'profile_note'",
    )


def _memory_row(runtime: Any, memory_id: int) -> dict[str, Any]:
    return _row(runtime, "SELECT * FROM memories WHERE id = ?", (memory_id,))


def _job_row(runtime: Any, memory_id: int) -> dict[str, Any]:
    return _row(
        runtime,
        "SELECT * FROM memory_projection_jobs WHERE memory_id = ?",
        (memory_id,),
    )


def _profile_context(runtime: Any) -> tuple[str, str]:
    context, state = runtime.chat._read_profile_context()
    if any(fragment in context for fragment in LOCAL_PATH_FRAGMENTS):
        raise AssertionError(f"profile grounding leaked a local path: {context}")
    return context, state


def _assert_manual_only(runtime: Any, *, label: str) -> None:
    context, state = _profile_context(runtime)
    if (
        state != "unavailable"
        or MANUAL_MARKER not in context
        or BODY in context
        or ATTACK_MARKER in context
    ):
        raise AssertionError(f"{label} did not fail closed around generated profile data: {state} / {context}")
    result = runtime.registry.get("read_profile").handler({})
    if (
        not result.ok
        or result.metadata.get("profile_custody_state") != "unavailable"
        or MANUAL_MARKER not in result.output
        or BODY in result.output
        or ATTACK_MARKER in result.output
    ):
        raise AssertionError(f"{label} bypassed custody through read_profile: {result}")


def test_fresh_write_has_exact_end_to_end_custody() -> None:
    with TemporaryDirectory() as tmp:
        runtime, result = _fixture(Path(tmp))
        memory_id = result.metadata.get("memory_id")
        if type(memory_id) is not int or memory_id < 1:
            raise AssertionError(f"profile write missed a bounded memory id: {result.metadata}")
        source = _source_row(runtime)
        memory = _memory_row(runtime, memory_id)
        job = _job_row(runtime, memory_id)
        if (
            source.get("memory_id") != memory_id
            or source.get("memory_link_required") != 1
            or source.get("mirror_state") != "completed"
            or not source.get("mirror_completed_at")
            or memory.get("source") != "profile"
            or memory.get("revision") != job.get("memory_revision")
            or job.get("operation") != "publish"
            or job.get("state") != "completed"
            or not job.get("canonical_path_display")
            or not job.get("content_digest")
        ):
            raise AssertionError(f"fresh profile custody is incomplete: {source} / {memory} / {job}")
        context, state = _profile_context(runtime)
        if state != "ok" or MANUAL_MARKER not in context or BODY not in context:
            raise AssertionError(f"fresh trusted profile was not grounded: {state} / {context}")
        snapshot = runtime.store.read_profile_knowledge_snapshot()
        if (
            snapshot.source_count != 1
            or snapshot.invalid_count != 0
            or snapshot.truncated
            or len(snapshot.notes) != 1
            or snapshot.notes[0].memory_id != memory_id
        ):
            raise AssertionError(f"fresh profile snapshot is not exact: {snapshot}")


def test_missing_memory_and_pending_source_fail_closed() -> None:
    with TemporaryDirectory() as tmp:
        runtime, result = _fixture(Path(tmp))
        memory_id = int(result.metadata["memory_id"])
        target = runtime.store.resolve_memory_approval_target(memory_id)
        if target is None:
            raise AssertionError("profile memory deletion target is missing")
        delete_result = runtime.registry.get("delete_memory").handler(
            {
                "memory_id": memory_id,
                "target_revision": target.revision,
                "target_binding": target.binding,
            }
        )
        if (
            delete_result.ok
            or delete_result.metadata.get("target_status") != "protected"
            or delete_result.metadata.get("writes_memory") is not False
        ):
            raise AssertionError(f"profile deletion did not fail safely at the tool boundary: {delete_result}")
        protected = runtime.store.delete_memory_exact_with_projection(
            memory_id,
            target.revision,
            target.binding,
        )
        if protected.status != "protected" or runtime.store.delete_memory(memory_id):
            raise AssertionError(f"profile-owned memory was not protected from generic deletion: {protected}")
        trusted_context, trusted_state = _profile_context(runtime)
        if trusted_state != "ok" or BODY not in trusted_context:
            raise AssertionError("a refused generic deletion damaged trusted profile grounding")
        with runtime.store.connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        source = _source_row(runtime)
        if source.get("memory_id") is not None or source.get("memory_link_required") != 1:
            raise AssertionError(f"deleted profile memory kept a trusted source link: {source}")
        _assert_manual_only(runtime, label="missing memory")
        repaired = _add_note(runtime)
        repaired_memory_id = repaired.metadata.get("memory_id")
        if type(repaired_memory_id) is not int or repaired_memory_id == memory_id:
            raise AssertionError("deleted profile memory was not repaired with a new identity")
        context, state = _profile_context(runtime)
        if state != "ok" or BODY not in context or MANUAL_MARKER not in context:
            raise AssertionError(f"deleted profile memory repair did not restore trust: {state} / {context}")

    with TemporaryDirectory() as tmp:
        runtime, result = _fixture(Path(tmp))
        memory_id = int(result.metadata["memory_id"])
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE ingested_sources SET mirror_state = 'pending', mirror_completed_at = NULL "
                "WHERE source_type = 'profile_note'"
            )
        _assert_manual_only(runtime, label="pending source")
        repaired = _add_note(runtime)
        if repaired.metadata.get("memory_id") != memory_id:
            raise AssertionError("profile repair changed canonical memory identity")
        context, state = _profile_context(runtime)
        if state != "ok" or BODY not in context or MANUAL_MARKER not in context:
            raise AssertionError(f"same mutation did not restore profile custody: {state} / {context}")


def test_revision_and_projection_drift_fail_closed() -> None:
    with TemporaryDirectory() as tmp:
        runtime, result = _fixture(Path(tmp))
        memory_id = int(result.metadata["memory_id"])
        stale_job = _job_row(runtime, memory_id)
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE memories SET body = ?, revision = revision + 1 WHERE id = ?",
                (ATTACK_MARKER, memory_id),
            )
        _assert_manual_only(runtime, label="memory revision drift")
        _add_note(runtime)
        if runtime.store.mark_ingested_source_projection_pending(
            _source_row(runtime)["source_key"],
            memory_id,
            int(stale_job["memory_revision"]),
            str(stale_job["source_digest"]),
        ):
            raise AssertionError("stale profile writer reopened newer completed custody")
        source = _source_row(runtime)
        if source.get("mirror_state") != "completed":
            raise AssertionError(f"stale profile writer damaged newer custody: {source}")

    with TemporaryDirectory() as tmp:
        runtime, result = _fixture(Path(tmp))
        memory_id = int(result.metadata["memory_id"])
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE memory_projection_jobs SET state = 'pending', completed_at = NULL, "
                "canonical_path_display = NULL, content_digest = NULL, "
                "completion_status = NULL, last_error_code = NULL "
                "WHERE memory_id = ?",
                (memory_id,),
            )
        _assert_manual_only(runtime, label="pending memory projection")


def test_file_tampering_and_duplicate_markers_fail_closed() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _result = _fixture(Path(tmp))
        profile_path = runtime.vault.root_path / "Profile.md"
        profile_path.write_text(
            profile_path.read_text(encoding="utf-8").replace(BODY, ATTACK_MARKER),
            encoding="utf-8",
        )
        _assert_manual_only(runtime, label="tampered generated profile block")
        _add_note(runtime)
        context, state = _profile_context(runtime)
        if state != "ok" or BODY not in context or ATTACK_MARKER in context:
            raise AssertionError(f"profile block repair did not restore trust: {state} / {context}")

    with TemporaryDirectory() as tmp:
        runtime, result = _fixture(Path(tmp))
        memory_id = int(result.metadata["memory_id"])
        job = _job_row(runtime, memory_id)
        note_path = runtime.vault.root_path / str(job["canonical_path_display"])
        note_path.write_text(ATTACK_MARKER, encoding="utf-8")
        _assert_manual_only(runtime, label="tampered memory projection note")

    with TemporaryDirectory() as tmp:
        runtime, _result = _fixture(Path(tmp))
        profile_path = runtime.vault.root_path / "Profile.md"
        raw = profile_path.read_text(encoding="utf-8")
        start = raw.index("<!-- jarvis-profile-note-start:v1:")
        duplicate = raw[start:]
        profile_path.write_text(raw + "\n" + duplicate, encoding="utf-8")
        _assert_manual_only(runtime, label="duplicate generated profile block")


def test_profile_evidence_exit_revalidation_reopens_custody() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _result = _fixture(Path(tmp))
        original_complete = runtime.store.complete_ingested_source_projection

        def complete_then_tamper(*args, **kwargs):
            evidence = original_complete(*args, **kwargs)
            profile_path = runtime.vault.root_path / "Profile.md"
            profile_path.write_text(
                profile_path.read_text(encoding="utf-8").replace(BODY, ATTACK_MARKER),
                encoding="utf-8",
            )
            return evidence

        runtime.store.complete_ingested_source_projection = complete_then_tamper
        try:
            try:
                _add_note(runtime)
            except RuntimeError as exc:
                if str(exc) != "profile_note_projection_evidence_changed":
                    raise
            else:
                raise AssertionError("profile evidence changed at lock exit but the write succeeded")
        finally:
            runtime.store.complete_ingested_source_projection = original_complete
        source = _source_row(runtime)
        if source.get("mirror_state") != "pending" or source.get("mirror_completed_at") is not None:
            raise AssertionError(f"exit revalidation did not reopen profile custody: {source}")
        _assert_manual_only(runtime, label="profile evidence exit race")
        _add_note(runtime)
        context, state = _profile_context(runtime)
        if state != "ok" or BODY not in context or ATTACK_MARKER in context:
            raise AssertionError(f"exit-race recovery did not restore trust: {state} / {context}")


def test_malformed_marker_variants_never_become_manual_context() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _result = _fixture(Path(tmp))
        profile_path = runtime.vault.root_path / "Profile.md"
        profile_path.write_bytes(profile_path.read_bytes().replace(b"\n", b"\r\n"))
        context, state = _profile_context(runtime)
        if state != "ok" or MANUAL_MARKER not in context or BODY not in context:
            raise AssertionError(f"equivalent CRLF profile custody drifted: {state} / {context}")

    with TemporaryDirectory() as tmp:
        runtime, _result = _fixture(Path(tmp))
        profile_path = runtime.vault.root_path / "Profile.md"
        digest = "e" * 64
        profile_path.write_text(
            profile_path.read_text(encoding="utf-8")
            + f"\n<!-- jarvis-profile-note-start:v2:{digest} -->\n"
            + f"## Future marker\n\n{ATTACK_MARKER}\n"
            + f"<!-- jarvis-profile-note:v2:{digest} -->\n",
            encoding="utf-8",
        )
        context, state = _profile_context(runtime)
        if state != "unavailable" or ATTACK_MARKER in context or BODY not in context:
            raise AssertionError(f"future marker payload became manual context: {state} / {context}")

    with TemporaryDirectory() as tmp:
        runtime, _result = _fixture(Path(tmp))
        profile_path = runtime.vault.root_path / "Profile.md"
        raw = profile_path.read_text(encoding="utf-8")
        nested_digest = "d" * 64
        insertion = (
            f"<!-- jarvis-profile-note-start:v1:{nested_digest} -->\n"
            f"## Nested\n\n{ATTACK_MARKER}\n"
            f"<!-- jarvis-profile-note:v1:{nested_digest} -->\n"
        )
        end_at = raw.index("<!-- jarvis-profile-note:v1:")
        profile_path.write_text(raw[:end_at] + insertion + raw[end_at:], encoding="utf-8")
        _assert_manual_only(runtime, label="nested generated marker block")
        try:
            _add_note(runtime)
        except RuntimeError:
            pass
        else:
            raise AssertionError("profile repair accepted globally nested marker ownership")


def test_profile_grounding_reports_output_clipping() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _result = _fixture(Path(tmp))
        snapshot = runtime.store.read_profile_knowledge_snapshot()
        view = runtime.vault.read_profile_grounding(snapshot.notes, max_chars=10)
        if not view.truncated or view.verified_source_keys:
            raise AssertionError(f"clipped profile output claimed complete generated custody: {view}")

    with TemporaryDirectory() as tmp:
        runtime = make_temp_runtime(Path(tmp))
        path = runtime.vault.root_path / "Profile.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Profile\n\n" + MANUAL_MARKER + ("X" * 2200), encoding="utf-8")
        context, state = _profile_context(runtime)
        if state != "unavailable" or len(context) > 2000:
            raise AssertionError(f"long manual profile was silently treated as complete: {state} / {len(context)}")


def test_profile_grounding_redacts_common_local_path_forms() -> None:
    with TemporaryDirectory() as tmp:
        runtime = make_temp_runtime(Path(tmp))
        path = runtime.vault.root_path / "Profile.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Profile\n\n"
            "volume /Volumes/Private/strategy.md\n"
            "home ~/Documents/private.txt\n"
            "linux /home/the operator/private.txt\n"
            "vartmp /var/tmp/private.txt\n"
            "windows C:\\Users\\the operator\\private.txt\n",
            encoding="utf-8",
        )
        context, state = _profile_context(runtime)
        forbidden = ("/Volumes/", "~/", "/home/", "/var/tmp/", "C:\\Users\\")
        if state != "ok" or "<local-path>" not in context or any(item in context for item in forbidden):
            raise AssertionError(f"profile grounding leaked a common local path: {state} / {context}")


def test_unknown_markers_store_outage_and_memory_bypass_fail_closed() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _result = _fixture(Path(tmp))
        unknown_digest = "f" * 64
        profile_path = runtime.vault.root_path / "Profile.md"
        profile_path.write_text(
            profile_path.read_text(encoding="utf-8")
            + f"\n<!-- jarvis-profile-note-start:v1:{unknown_digest} -->\n"
            + f"## Untrusted\n\n{ATTACK_MARKER}\n"
            + f"<!-- jarvis-profile-note:v1:{unknown_digest} -->\n",
            encoding="utf-8",
        )
        context, state = _profile_context(runtime)
        if state != "unavailable" or ATTACK_MARKER in context or BODY not in context:
            raise AssertionError(f"unknown marker block was not isolated: {state} / {context}")

    with TemporaryDirectory() as tmp:
        runtime, _result = _fixture(Path(tmp))
        original = runtime.store.read_profile_knowledge_snapshot

        def fail_store():
            raise RuntimeError("offline custody smoke")

        runtime.store.read_profile_knowledge_snapshot = fail_store
        try:
            _assert_manual_only(runtime, label="profile store outage")
        finally:
            runtime.store.read_profile_knowledge_snapshot = original

        original_search = runtime.store.search_memories
        original_recent = runtime.store.recent_memories
        runtime.store.search_memories = lambda *_args, **_kwargs: [
            {
                "id": 999,
                "category": CATEGORY,
                "title": HEADING,
                "body": ATTACK_MARKER,
                "source": "profile",
            }
        ]
        runtime.store.recent_memories = lambda *_args, **_kwargs: []
        try:
            memory_context, memory_state, source_digests = runtime.chat._read_memory_context(
                "remember investment philosophy"
            )
        finally:
            runtime.store.search_memories = original_search
            runtime.store.recent_memories = original_recent
        if memory_context or memory_state != "empty" or source_digests:
            raise AssertionError(
                "source=profile memory bypassed verified profile grounding: "
                f"{memory_state} / {memory_context} / {source_digests}"
            )

    with TemporaryDirectory() as tmp:
        runtime, result = _fixture(Path(tmp))
        memory_id = int(result.metadata["memory_id"])
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE memories SET source = 'manual', body = ?, revision = revision + 1 "
                "WHERE id = ?",
                (ATTACK_MARKER, memory_id),
            )
            owned = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if owned is None:
            raise AssertionError("profile-owned custody fixture lost its linked memory")
        owned_row = dict(owned)
        original_search = runtime.store.search_memories
        original_recent = runtime.store.recent_memories
        runtime.store.search_memories = lambda *_args, **_kwargs: [owned_row]
        runtime.store.recent_memories = lambda *_args, **_kwargs: []
        try:
            memory_context, memory_state, source_digests = runtime.chat._read_memory_context(
                "investment philosophy"
            )
        finally:
            runtime.store.search_memories = original_search
            runtime.store.recent_memories = original_recent
        if memory_context or memory_state != "empty" or source_digests:
            raise AssertionError(
                "durably profile-owned row bypassed custody after mutable source drift: "
                f"{memory_state} / {memory_context} / {source_digests}"
            )


def test_memory_grounding_is_field_and_aggregate_bounded() -> None:
    with TemporaryDirectory() as tmp:
        runtime = make_temp_runtime(Path(tmp))
        memory_id = runtime.store.add_memory(
            MemoryRecord(
                category="manual",
                title="Bounded memory",
                body="bounded",
                source="manual",
                confidence=1.0,
            )
        )
        oversized = {
            "id": memory_id,
            "category": "C" * 10_000,
            "title": "T" * 100_000,
            "body": "B" * 100_000,
            "source": "manual",
        }
        original_search = runtime.store.search_memories
        original_recent = runtime.store.recent_memories
        runtime.store.search_memories = lambda *_args, **_kwargs: [oversized]
        runtime.store.recent_memories = lambda *_args, **_kwargs: []
        try:
            context, state, _digests = runtime.chat._read_memory_context("bounded memory")
        finally:
            runtime.store.search_memories = original_search
            runtime.store.recent_memories = original_recent
        if len(context) > 4000 or state != "unavailable":
            raise AssertionError(f"oversized memory grounding was not bounded: {state} / {len(context)}")


def test_profile_read_uses_writer_compatible_evidence_lock_order() -> None:
    with TemporaryDirectory() as tmp:
        runtime, _result = _fixture(Path(tmp))
        order: list[str] = []
        original_profile_lock = runtime.vault.canonical_profile_note_evidence_lock
        original_memory_lock = runtime.vault.canonical_memory_projection_evidence_lock

        @contextmanager
        def recording_profile_lock(*args, **kwargs):
            order.append("profile")
            with original_profile_lock(*args, **kwargs) as current:
                yield current

        @contextmanager
        def recording_memory_lock(*args, **kwargs):
            order.append("memory")
            with original_memory_lock(*args, **kwargs) as current:
                yield current

        runtime.vault.canonical_profile_note_evidence_lock = recording_profile_lock
        runtime.vault.canonical_memory_projection_evidence_lock = recording_memory_lock
        try:
            result = runtime.registry.get("read_profile").handler({})
        finally:
            runtime.vault.canonical_profile_note_evidence_lock = original_profile_lock
            runtime.vault.canonical_memory_projection_evidence_lock = original_memory_lock
        if (
            not result.ok
            or result.metadata.get("profile_custody_state") != "ok"
            or BODY not in result.output
        ):
            raise AssertionError(f"instrumented profile read lost verified context: {result}")
        if order[:2] != ["profile", "memory"]:
            raise AssertionError(f"profile read evidence lock order can deadlock writers: {order}")


def test_chat_profile_read_revalidates_memory_projection_on_exit() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-chat-revalidation-") as tmp:
        runtime, result = _fixture(Path(tmp))
        memory_id = int(result.metadata["memory_id"])
        job = _job_row(runtime, memory_id)
        note_path = runtime.vault.root_path / str(job["canonical_path_display"])
        original_read = runtime.vault.read_profile_grounding
        tampered = False

        def read_then_tamper(*args, **kwargs):
            nonlocal tampered
            view = original_read(*args, **kwargs)
            if not tampered:
                tampered = True
                note_path.write_text(ATTACK_MARKER, encoding="utf-8")
            return view

        runtime.vault.read_profile_grounding = read_then_tamper
        try:
            context, state = runtime.chat._read_profile_context()
        finally:
            runtime.vault.read_profile_grounding = original_read
        if state != "unavailable" or BODY in context or ATTACK_MARKER in context:
            raise AssertionError(
                f"chat profile read disclosed content after projection drift: {state} / {context}"
            )


def main() -> None:
    test_fresh_write_has_exact_end_to_end_custody()
    test_missing_memory_and_pending_source_fail_closed()
    test_revision_and_projection_drift_fail_closed()
    test_file_tampering_and_duplicate_markers_fail_closed()
    test_profile_evidence_exit_revalidation_reopens_custody()
    test_malformed_marker_variants_never_become_manual_context()
    test_profile_grounding_reports_output_clipping()
    test_profile_grounding_redacts_common_local_path_forms()
    test_unknown_markers_store_outage_and_memory_bypass_fail_closed()
    test_memory_grounding_is_field_and_aggregate_bounded()
    test_profile_read_uses_writer_compatible_evidence_lock_order()
    test_chat_profile_read_revalidates_memory_projection_on_exit()
    print("profile custody smoke passed")


if __name__ == "__main__":
    main()
