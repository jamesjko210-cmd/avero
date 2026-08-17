from __future__ import annotations

import re
import unicodedata
from contextlib import ExitStack, nullcontext
from threading import Lock
from typing import Any

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault, ProfileEvidenceRevalidationError
from jarvis_v2.memory.store import MemoryRecord, MemoryStore, profile_note_source_key


MAX_PROFILE_READ_CHARS = 20000
MAX_PROFILE_WRITE_CHARS = 50000
MAX_PROFILE_HEADING_CHARS = 120
MAX_PROFILE_CATEGORY_CHARS = 64
MAX_PROFILE_GENERATION_ATTEMPTS = 4
PROFILE_GENERATION_STALE_ERROR = "ingested source projection generation is stale"
LOCAL_PATH_RE = re.compile(
    r"(?:(?<![:A-Za-z0-9])/(?:System/Volumes/Data/Users|Users|root|private|"
    r"var/(?:folders|tmp)|tmp|Volumes|home)/[^\n\r;]*"
    r"|~[/\\][^\n\r;]*|[A-Za-z]:[/\\][^\n\r;]*)",
    re.IGNORECASE,
)
PROFILE_MARKER_NAMESPACE_RE = re.compile(
    r"jarvis-profile-note\b",
    re.IGNORECASE,
)


def _safe_vault_path_display(path: Any, vault: ObsidianVault) -> str:
    try:
        return str(path.relative_to(vault.root_path))
    except (AttributeError, ValueError):
        return _short_raw(path, 160)


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_PROFILE_READ_CHARS) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _raw_int_metadata(value: Any, *, key: str, sanitized: int) -> dict[str, Any]:
    if value is None:
        return {key: sanitized}
    if isinstance(value, bool):
        return {key: sanitized, f"raw_{key}": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {key: sanitized, f"raw_{key}": _short_raw(value, 80)}
    return {key: sanitized}


def _short(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_raw(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _profile_user_text(text: str) -> str:
    lines = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        lines.append(raw_line)
    return "\n".join(lines).strip()


def _canonical_profile_label(value: Any, default: str, limit: int) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    collapsed = " ".join(normalized.split())
    return _short(collapsed, limit) or default


def _normalized_profile_note_args(args: dict[str, Any]) -> tuple[Any, Any, str, str, str]:
    raw_heading = args.get("heading") or "Profile Note"
    raw_category = args.get("category") or "identity"
    heading = _canonical_profile_label(raw_heading, "Profile Note", MAX_PROFILE_HEADING_CHARS)
    body = unicodedata.normalize("NFKC", str(args.get("body") or ""))
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    folded_category = unicodedata.normalize("NFKC", str(raw_category or "")).casefold()
    category = _canonical_profile_label(
        folded_category,
        "identity",
        MAX_PROFILE_CATEGORY_CHARS,
    )
    return raw_heading, raw_category, heading, body, category


def _profile_note_source_key(heading: str, body: str, category: str) -> str:
    return profile_note_source_key(heading, body, category)


def add_profile_note_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, Any]:
    _raw_heading, _raw_category, heading, body, category = _normalized_profile_note_args(args)
    return {
        "heading": heading,
        "body": body,
        "category": category,
    }


def add_profile_note_auto_mutation_preflight(args: dict[str, Any]) -> str | None:
    _raw_heading, _raw_category, heading, body, category = _normalized_profile_note_args(args)
    if LOCAL_PATH_RE.search(heading):
        return "invalid_heading"
    if LOCAL_PATH_RE.search(category):
        return "invalid_category"
    if LOCAL_PATH_RE.search(body):
        return "invalid_body"
    if any(PROFILE_MARKER_NAMESPACE_RE.search(value) for value in (heading, category, body)):
        return "reserved_marker_namespace"
    if not body:
        return "missing_body"
    if len(body) > MAX_PROFILE_WRITE_CHARS:
        return "oversized_body"
    return None


def _empty_profile_message() -> str:
    return (
        "I don't have curated profile notes yet.\n\n"
        "Next safe commands:\n"
        "- add profile note <heading>: <body>\n"
        "- search memory for profile\n"
        "- export state"
    )


def _profile_limit_message(max_chars: int) -> str:
    return (
        "Profile notes exist, but the current read limit is too small to show them.\n\n"
        f"Current read limit: {max_chars} char(s).\n"
        "Next safe commands:\n"
        "- read profile\n"
        "- read profile max_chars 4000\n"
        "- search memory for profile"
    )


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "reads_private_data": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_database": False,
        "controls_computer": False,
        "queues_approval": False,
        "external_side_effect": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _profile_write_metadata(**extra: Any) -> dict[str, Any]:
    metadata = _safe_metadata(**extra)
    metadata.update(
        {
            "writes_files": True,
            "writes_memory": True,
            "writes_notes": True,
            "writes_database": True,
        }
    )
    return metadata


def _profile_contract(
    *,
    state_changed: bool = False,
    changed: list[str] | None = None,
    content_in_handoff: bool = False,
) -> dict[str, Any]:
    return {
        "ready_for_operator": True,
        "state_changed": state_changed,
        "changed": changed or [],
        "content_in_handoff": content_in_handoff,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _normalize_profile_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        next_safe_commands = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        next_safe_commands = [str(value) for value in raw_next if str(value or "").strip()]
    handoff["handoff_ready"] = True
    handoff["next_safe_command"] = next_safe_commands[0] if next_safe_commands else ""
    handoff["next_safe_commands"] = list(next_safe_commands)
    handoff["next_safe_command_count"] = len(next_safe_commands)
    return handoff


def _profile_handoff_metadata(
    handoff_key: str,
    handoff: dict[str, Any],
    *,
    writes: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    _normalize_profile_handoff(handoff)
    prefix = handoff_key.removesuffix("_handoff")
    metadata = _profile_write_metadata(**extra) if writes else _safe_metadata(**extra)
    metadata.update(
        _profile_contract(
            state_changed=_metadata_bool(handoff.get("state_changed")),
            changed=list(handoff.get("changed") or []),
            content_in_handoff=_metadata_bool(handoff.get("content_in_handoff")),
        )
    )
    metadata[handoff_key] = handoff
    metadata[f"{handoff_key}_ready"] = True
    metadata[f"{prefix}_handoff_ready"] = True
    metadata[f"{prefix}_ready_for_operator"] = True
    metadata[f"{prefix}_state_changed"] = _metadata_bool(handoff.get("state_changed"))
    metadata[f"{prefix}_changed"] = list(handoff.get("changed") or [])
    metadata[f"{prefix}_content_in_handoff"] = _metadata_bool(handoff.get("content_in_handoff"))
    metadata[f"{prefix}_next_safe_command"] = handoff["next_safe_command"]
    metadata[f"{prefix}_next_safe_commands"] = list(handoff["next_safe_commands"])
    metadata[f"{prefix}_next_safe_command_count"] = handoff["next_safe_command_count"]
    metadata[f"{prefix}_authorizes_execution"] = False
    metadata[f"{prefix}_authorizes_completion_claim"] = False
    metadata[f"{prefix}_approval_granted"] = False
    metadata["next_safe_command"] = handoff["next_safe_command"]
    metadata["next_safe_commands"] = list(handoff["next_safe_commands"])
    metadata["next_safe_command_count"] = handoff["next_safe_command_count"]
    return metadata


def _profile_boundaries(*, writes: bool) -> dict[str, bool]:
    return {
        "read_only": not writes,
        "reads_profile": not writes,
        "writes_files": writes,
        "writes_memory": writes,
        "writes_notes": writes,
        "writes_database": writes,
        "queues_approval": False,
        "controls_computer": False,
        "external_side_effect": False,
        "requires_approval": False,
        "calls_model": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _profile_write_boundaries(
    *,
    writes_files: bool,
    writes_memory: bool,
    writes_notes: bool,
    writes_database: bool,
) -> dict[str, bool]:
    return {
        "read_only": not any((writes_files, writes_memory, writes_notes, writes_database)),
        "reads_profile": False,
        "writes_files": writes_files,
        "writes_memory": writes_memory,
        "writes_notes": writes_notes,
        "writes_database": writes_database,
        "queues_approval": False,
        "controls_computer": False,
        "external_side_effect": False,
        "requires_approval": False,
        "calls_model": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _profile_refusal_boundaries() -> dict[str, bool]:
    return {
        "read_only": True,
        "reads_profile": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_database": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "creates_profile_note": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _profile_refusal_handoff(
    *,
    reason: str,
    raw_heading: Any = None,
    raw_category: Any = None,
    raw_body: Any = None,
    body_chars: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    handoff: dict[str, Any] = {
        "source": "add_profile_note",
        **_profile_contract(
            content_in_handoff=any(value is not None for value in (raw_heading, raw_category, raw_body))
            or body_chars is not None
            or limit is not None,
        ),
        "mutation": "profile_note_create",
        "reason": reason,
        "refused": True,
        "changed": [],
        "retry_command": "add profile note <heading>: <body>",
        "next_commands": [
            "add profile note <heading>: <body>",
            "read profile",
            "search memory for profile",
        ],
        "boundaries": _profile_refusal_boundaries(),
    }
    if raw_heading is not None:
        handoff["raw_heading"] = _short_raw(raw_heading, 80)
    if raw_category is not None:
        handoff["raw_category"] = _short_raw(raw_category, 80)
    if raw_body is not None:
        handoff["raw_body"] = _short_raw(raw_body, 80)
    if body_chars is not None:
        handoff["body_chars"] = body_chars
    if limit is not None:
        handoff["limit"] = limit
    return handoff


def _profile_refusal_metadata(
    *,
    reason: str,
    raw_heading: Any = None,
    raw_category: Any = None,
    raw_body: Any = None,
    body_chars: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    handoff = _profile_refusal_handoff(
        reason=reason,
        raw_heading=raw_heading,
        raw_category=raw_category,
        raw_body=raw_body,
        body_chars=body_chars,
        limit=limit,
    )
    metadata = _profile_handoff_metadata(
        "profile_refusal_handoff",
        handoff,
        reason=reason,
    )
    if raw_heading is not None:
        metadata["raw_heading"] = _short_raw(raw_heading, 80)
    if raw_category is not None:
        metadata["raw_category"] = _short_raw(raw_category, 80)
    if raw_body is not None:
        metadata["raw_body"] = _short_raw(raw_body, 80)
    if body_chars is not None:
        metadata["body_chars"] = body_chars
    if limit is not None:
        metadata["limit"] = limit
    return metadata


def _profile_write_handoff(
    *,
    heading: str,
    body: str,
    category: str,
    memory_id: int,
    path_display: str,
    state_changed: bool = True,
    changed: list[str] | None = None,
    writes_files: bool = True,
    writes_memory: bool = True,
    writes_notes: bool = True,
    writes_database: bool = True,
) -> dict[str, Any]:
    return {
        "source": "add_profile_note",
        **_profile_contract(
            state_changed=state_changed,
            changed=changed if changed is not None else ["profile_note", "memory"],
            content_in_handoff=True,
        ),
        "memory_id": memory_id,
        "category": _short_raw(category, MAX_PROFILE_CATEGORY_CHARS),
        "heading": _short_raw(heading, MAX_PROFILE_HEADING_CHARS),
        "body_preview": _short_raw(body, 220),
        "body_chars": len(body),
        "path_display": _short_raw(path_display, 160),
        "next_commands": [
            "read profile",
            f"search memory for {_short_raw(heading, 80)}",
            "export state",
        ],
        "boundaries": _profile_write_boundaries(
            writes_files=writes_files,
            writes_memory=writes_memory,
            writes_notes=writes_notes,
            writes_database=writes_database,
        ),
    }


def _profile_read_handoff(*, text: str, max_chars: int) -> dict[str, Any]:
    return {
        "source": "read_profile",
        **_profile_contract(content_in_handoff=bool(text.strip())),
        "path_display": "Profile.md",
        "chars": len(text),
        "max_chars": max_chars,
        "empty": not bool(text.strip()),
        "truncated": len(text) >= max_chars and bool(text.strip()),
        "preview": _short_raw(text, 220),
        "next_commands": [
            "add profile note <heading>: <body>",
            "search memory for profile",
            "export state",
        ],
        "boundaries": _profile_boundaries(writes=False),
    }


def make_profile_tools(store: MemoryStore, vault: ObsidianVault):
    profile_generation_retry_lock = Lock()

    def verified_profile_text(max_chars: int) -> tuple[str, bool]:
        degraded = False
        try:
            with store.hold_profile_knowledge_snapshot() as snapshot:
                degraded = bool(snapshot.invalid_count or snapshot.truncated)
                verified_notes = []
                with ExitStack() as evidence_locks:
                    if snapshot.notes:
                        first = snapshot.notes[0]
                        profile_evidence_current = evidence_locks.enter_context(
                            vault.canonical_profile_note_evidence_lock(
                                source_key=first.source_key,
                                heading=first.heading,
                                body=first.body,
                            )
                        )
                        if profile_evidence_current is True:
                            current_memory_ids: set[int] = set()
                            for note in sorted(snapshot.notes, key=lambda item: item.memory_id):
                                evidence_current = evidence_locks.enter_context(
                                    vault.canonical_memory_projection_evidence_lock(
                                        memory_id=note.memory_id,
                                        store_identity=snapshot.store_identity,
                                        expected_relative_path=note.canonical_path_display,
                                        expected_content_digest=note.content_digest,
                                    )
                                )
                                if evidence_current is True:
                                    current_memory_ids.add(note.memory_id)
                                else:
                                    degraded = True
                            verified_notes = [
                                note for note in snapshot.notes if note.memory_id in current_memory_ids
                            ]
                        else:
                            degraded = True
                    view = vault.read_profile_grounding(
                        tuple(verified_notes),
                        max_chars=max_chars,
                    )
                    revalidated_view = vault.read_profile_grounding(
                        tuple(verified_notes),
                        max_chars=max_chars,
                    )
                    if revalidated_view != view:
                        raise ProfileEvidenceRevalidationError(
                            "Profile grounding changed during the final custody read."
                        )
                    view = revalidated_view
        except Exception:
            try:
                manual_view = vault.read_profile_grounding((), max_chars=max_chars)
            except Exception:
                return "", True
            return manual_view.text, True
        degraded = bool(
            degraded
            or view.invalid
            or view.truncated
            or len(view.verified_source_keys) != len(verified_notes)
        )
        return view.text, degraded

    def add_profile_note(args: dict[str, Any]) -> ToolResult:
        raw_heading, raw_category, heading, body, category = _normalized_profile_note_args(args)
        if LOCAL_PATH_RE.search(heading):
            return ToolResult(
                "add_profile_note",
                False,
                "Profile note heading should describe the note, not a local file path.",
                _profile_refusal_metadata(reason="invalid_heading", raw_heading=raw_heading),
            )
        if LOCAL_PATH_RE.search(category):
            return ToolResult(
                "add_profile_note",
                False,
                "Profile note category should describe a memory category, not a local file path.",
                _profile_refusal_metadata(reason="invalid_category", raw_category=raw_category),
            )
        if LOCAL_PATH_RE.search(body):
            return ToolResult(
                "add_profile_note",
                False,
                "Profile note body should not contain local file paths.",
                _profile_refusal_metadata(reason="invalid_body", raw_body=body),
            )
        if any(PROFILE_MARKER_NAMESPACE_RE.search(value) for value in (heading, category, body)):
            return ToolResult(
                "add_profile_note",
                False,
                "Profile note content uses a reserved Jarvis marker namespace.",
                _profile_refusal_metadata(
                    reason="reserved_marker_namespace",
                    raw_heading=raw_heading,
                    raw_category=raw_category,
                    raw_body=body,
                ),
            )
        if not body:
            return ToolResult("add_profile_note", False, "Profile note body is required.", _profile_refusal_metadata(reason="missing_body", raw_heading=raw_heading, raw_category=raw_category, raw_body=body))
        if len(body) > MAX_PROFILE_WRITE_CHARS:
            return ToolResult(
                "add_profile_note",
                False,
                f"Profile note body is too large ({len(body)} chars). Limit is {MAX_PROFILE_WRITE_CHARS}.",
                _profile_refusal_metadata(reason="oversized_body", raw_heading=raw_heading, raw_category=raw_category, body_chars=len(body), limit=MAX_PROFILE_WRITE_CHARS),
            )
        source_key = _profile_note_source_key(heading, body, category)
        record = MemoryRecord(
            category=category,
            title=heading,
            body=body,
            source="profile",
            confidence=1.0,
        )
        created = False
        memory_repaired = False
        appended = False
        mirror_completed = False
        projection_changed = False
        writes_files = False
        projection = None
        projection_target = None
        path = None
        evidence = None
        memory_id = 0
        for attempt_index in range(MAX_PROFILE_GENERATION_ATTEMPTS):
            retry_guard = (
                profile_generation_retry_lock
                if attempt_index
                else nullcontext()
            )
            try:
                with retry_guard:
                    created_now, memory_repaired_now, projection_target = (
                        store.ensure_profile_note_memory_with_projection(record, source_key)
                    )
                    created = created or created_now
                    memory_repaired = memory_repaired or memory_repaired_now
                    memory_id = projection_target.memory_id
                    projection = None
                    for _ in range(4):
                        projection = reconcile_memory_projection(
                            store,
                            vault,
                            memory_id,
                            expected_operation=projection_target.operation,
                            expected_revision=projection_target.revision,
                            expected_source_digest=projection_target.source_digest,
                        )
                        if projection.status == "completed":
                            break
                    assert projection is not None
                    projection_changed_now = projection.completion_status in {
                        "published",
                        "repaired",
                    }
                    projection_changed = projection_changed or projection_changed_now
                    writes_files = writes_files or projection.file_written
                    if (
                        projection.status != "completed"
                        or not projection.path_display
                        or not projection.content_digest
                    ):
                        store.mark_ingested_source_projection_pending(
                            source_key,
                            memory_id,
                            projection_target.revision,
                            projection_target.source_digest,
                            expected_generation=projection_target.attempt_generation,
                        )
                        raise RuntimeError("profile_note_memory_projection_pending")
                    path, appended_now = vault.append_profile_once(source_key, heading, body)
                    appended = appended or appended_now
                    writes_files = writes_files or appended_now
                    try:
                        with (
                            store.profile_memory_ownership_egress_fence(()),
                            vault.canonical_profile_note_evidence_lock(
                                source_key=source_key,
                                heading=heading,
                                body=body,
                            ) as profile_evidence_current,
                        ):
                            if not profile_evidence_current:
                                store.mark_ingested_source_projection_pending(
                                    source_key,
                                    memory_id,
                                    projection_target.revision,
                                    projection_target.source_digest,
                                    expected_generation=projection_target.attempt_generation,
                                )
                                raise RuntimeError("profile_note_projection_evidence_mismatch")
                            with vault.canonical_memory_projection_evidence_lock(
                                memory_id=memory_id,
                                store_identity=store.get_store_identity(),
                                expected_relative_path=projection.path_display,
                                expected_content_digest=projection.content_digest,
                            ) as memory_evidence_current:
                                if not memory_evidence_current:
                                    store.mark_ingested_source_projection_pending(
                                        source_key,
                                        memory_id,
                                        projection_target.revision,
                                        projection_target.source_digest,
                                        expected_generation=projection_target.attempt_generation,
                                    )
                                    raise RuntimeError("profile_note_memory_evidence_mismatch")
                                evidence = store.complete_ingested_source_projection(
                                    source_key,
                                    memory_id,
                                    projection_target.revision,
                                    projection_target.source_digest,
                                    expected_generation=projection_target.attempt_generation,
                                )
                    except ProfileEvidenceRevalidationError as exc:
                        store.mark_ingested_source_projection_pending(
                            source_key,
                            memory_id,
                            projection_target.revision,
                            projection_target.source_digest,
                            expected_generation=projection_target.attempt_generation,
                        )
                        raise RuntimeError("profile_note_projection_evidence_changed") from exc
                    mirror_completed = mirror_completed or evidence["completed_now"] is True
                break
            except RuntimeError as exc:
                if (
                    str(exc) != PROFILE_GENERATION_STALE_ERROR
                    or attempt_index + 1 >= MAX_PROFILE_GENERATION_ATTEMPTS
                ):
                    raise
                continue
        assert projection is not None
        assert projection_target is not None
        assert path is not None
        assert evidence is not None
        path_display = _safe_vault_path_display(path, vault)
        memory_changed = created or memory_repaired
        state_changed = memory_changed or appended or mirror_completed or projection_changed
        writes_database = memory_changed or mirror_completed or projection_changed
        changed = [
            identity
            for identity, changed_now in (
                ("profile_note", appended),
                ("memory", memory_changed),
            )
            if changed_now
        ]
        if (mirror_completed or projection_changed) and not (memory_changed or appended):
            changed.append("profile_mirror")
        handoff = _profile_write_handoff(
            heading=heading,
            body=body,
            category=category,
            memory_id=memory_id,
            path_display=path_display,
            state_changed=state_changed,
            changed=changed,
            writes_files=writes_files,
            writes_memory=memory_changed,
            writes_notes=writes_files,
            writes_database=writes_database,
        )
        if memory_changed and appended:
            output = f"Stored profile note '{heading}' and updated memory index #{memory_id}.\nSaved note: {path_display}"
        elif memory_changed:
            output = f"Restored the memory index for profile note '{heading}' as memory #{memory_id}.\nSaved note: {path_display}"
        elif appended:
            output = f"Restored profile note '{heading}' and verified memory index #{memory_id}.\nSaved note: {path_display}"
        elif mirror_completed:
            output = f"Recovered profile note mirror for memory #{memory_id}.\nSaved note: {path_display}"
        elif projection_changed:
            output = f"Recovered profile note custody for memory #{memory_id}.\nSaved note: {path_display}"
        else:
            output = f"Profile note '{heading}' is already saved as memory #{memory_id}.\nSaved note: {path_display}"
        metadata = _profile_handoff_metadata(
            "profile_write_handoff",
            handoff,
            writes=state_changed,
            path_display=path_display,
            memory_id=memory_id,
            heading_chars=len(heading),
            body_chars=len(body),
            category=category,
            source_new=created,
            memory_created=created,
            memory_repaired=memory_repaired,
            profile_appended=appended,
            profile_mirror_completed=mirror_completed,
            converged=not state_changed,
        )
        metadata.update(
            {
                "writes_files": writes_files,
                "writes_memory": memory_changed,
                "writes_notes": writes_files,
                "writes_database": writes_database,
                "memory_projection_status": projection.status,
                "memory_projection_completion": projection.completion_status,
            }
        )
        return ToolResult(
            "add_profile_note",
            True,
            output,
            metadata,
        )

    def read_profile(args: dict[str, Any]) -> ToolResult:
        max_chars = _bounded_int(args.get("max_chars"), 4000)
        text, custody_degraded = verified_profile_text(max_chars)
        user_text = _profile_user_text(text)
        full_user_text = user_text
        if not user_text and max_chars < MAX_PROFILE_READ_CHARS:
            full_text, full_degraded = verified_profile_text(MAX_PROFILE_READ_CHARS)
            full_user_text = _profile_user_text(full_text)
            custody_degraded = custody_degraded or full_degraded
        if not user_text:
            handoff = _profile_read_handoff(text="", max_chars=max_chars)
            has_curated_notes = bool(full_user_text)
            if custody_degraded and not has_curated_notes:
                output = (
                    "I couldn't fully verify curated profile notes, so I omitted generated content.\n\n"
                    "Next safe commands:\n"
                    "- read profile\n"
                    "- add profile note <heading>: <body>\n"
                    "- export state"
                )
                empty_reason = "profile_custody_unavailable"
            else:
                output = (
                    _profile_limit_message(max_chars)
                    if has_curated_notes
                    else _empty_profile_message()
                )
                empty_reason = (
                    "read_limit_before_profile_notes"
                    if has_curated_notes
                    else "no_curated_profile_notes"
                )
            return ToolResult(
                "read_profile",
                True,
                output,
                _profile_handoff_metadata(
                    "profile_read_handoff",
                    handoff,
                    chars=0,
                    profile_file_chars=len(text),
                    profile_has_curated_notes=has_curated_notes,
                    profile_custody_state=("unavailable" if custody_degraded else "ok"),
                    empty_reason=empty_reason,
                    **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
                ),
            )
        handoff = _profile_read_handoff(text=text, max_chars=max_chars)
        output = text
        if custody_degraded:
            output = (
                "Profile custody note: some generated profile content could not be verified and was omitted.\n\n"
                + text
            )
        return ToolResult(
            "read_profile",
            True,
            output,
            _profile_handoff_metadata(
                "profile_read_handoff",
                handoff,
                chars=len(text),
                profile_custody_state=("unavailable" if custody_degraded else "ok"),
                **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
            ),
        )

    return add_profile_note, read_profile
