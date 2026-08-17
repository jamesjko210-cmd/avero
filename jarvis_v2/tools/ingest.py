from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from jarvis_v2.agent.types import ApprovalArgumentResolution, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import InboxSnapshotError, ObsidianVault
from jarvis_v2.memory.store import (
    MemoryRecord,
    MemoryStore,
    ScheduledJobLeaseAuthorityLost,
)


MAX_INBOX_INGEST_LIMIT = 200
MAX_INBOX_SOURCE_CHARS = 256 * 1024
MAX_DIGEST_LIMIT = 100
MAX_DIGEST_HOURS = 24 * 30
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
_INBOX_SOURCE_BINDING_KEY = "_auto_mutation_inbox_snapshot_binding"
_INBOX_BINDING_MAX_BYTES = (MAX_INBOX_SOURCE_CHARS * 4) + 4


class _EffectAuthorityLost(RuntimeError):
    pass


def _require_effect_authority(effect_authority: Callable[[], None] | None) -> None:
    if effect_authority is None:
        return
    try:
        effect_authority()
    except Exception as exc:
        raise _EffectAuthorityLost("scheduled effect authority lost") from exc


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "reads_file_metadata": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "speaks": False,
        "completes_tasks": False,
    }
    metadata.update(extra)
    return metadata


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _ingest_contract(
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


def _ingest_handoff_metadata(
    handoff_key: str,
    handoff: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    next_safe_commands = list(handoff.get("next_commands") or [])
    next_safe_command = next_safe_commands[0] if next_safe_commands else ""
    handoff["handoff_ready"] = True
    handoff["next_safe_command"] = next_safe_command
    handoff["next_safe_commands"] = list(next_safe_commands)
    handoff["next_safe_command_count"] = len(next_safe_commands)
    prefix = handoff_key.removesuffix("_handoff")
    metadata = _safe_metadata(**extra)
    metadata.update(
        _ingest_contract(
            state_changed=_metadata_bool(handoff.get("state_changed")),
            changed=list(handoff.get("changed") or []),
            content_in_handoff=_metadata_bool(handoff.get("content_in_handoff")),
        )
    )
    metadata[f"{handoff_key}_ready"] = True
    metadata[f"{prefix}_handoff_ready"] = True
    metadata[f"{prefix}_ready_for_operator"] = True
    metadata[f"{prefix}_state_changed"] = _metadata_bool(handoff.get("state_changed"))
    metadata[f"{prefix}_changed"] = list(handoff.get("changed") or [])
    metadata[f"{prefix}_content_in_handoff"] = _metadata_bool(handoff.get("content_in_handoff"))
    metadata[f"{prefix}_next_safe_command"] = next_safe_command
    metadata[f"{prefix}_next_safe_commands"] = list(next_safe_commands)
    metadata[f"{prefix}_next_safe_command_count"] = len(next_safe_commands)
    metadata[f"{prefix}_authorizes_execution"] = False
    metadata[f"{prefix}_authorizes_completion_claim"] = False
    metadata[f"{prefix}_approval_granted"] = False
    metadata["next_safe_command"] = next_safe_command
    metadata["next_safe_commands"] = list(next_safe_commands)
    metadata["next_safe_command_count"] = len(next_safe_commands)
    metadata[handoff_key] = handoff
    return metadata


def _safe_vault_path_display(path: str | Path | None, vault: ObsidianVault) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        try:
            return str(candidate.resolve().relative_to(vault.root_path.resolve()))
        except (OSError, ValueError):
            return LOCAL_PATH_RE.sub("<local-path>", str(candidate))


def _short_metadata(value: Any, limit: int = 180) -> str:
    text = LOCAL_PATH_RE.sub("<local-path>", str(value or "").strip())
    text = "".join(
        " " if unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"} else char
        for char in text
    )
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _ingest_boundaries(*, writes: bool, reads_file_metadata: bool = False) -> dict[str, bool]:
    return {
        "read_only": not writes,
        "reads_file_metadata": reads_file_metadata,
        "writes_files": writes,
        "writes_database": writes,
        "writes_memory": writes,
        "writes_notes": writes,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "external_side_effect": False,
    }


def _inbox_ingest_handoff(
    *,
    imported: int,
    mirrored: int,
    mirror_failures: int,
    source_completions: int,
    skipped: int,
    path_skipped: int,
    limit: int,
    memory_rows: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    database_writes_override: bool | None = None,
    note_writes_override: bool | None = None,
) -> dict[str, Any]:
    database_writes = (
        imported > 0 or mirrored > 0 or source_completions > 0
        if database_writes_override is None
        else database_writes_override
    )
    memory_writes = imported > 0
    note_writes = mirrored > 0 if note_writes_override is None else note_writes_override
    changed = []
    if memory_writes:
        changed.append("inbox_memory_import")
    if note_writes:
        changed.append("obsidian_memory_mirror")
    if source_completions > 0 and not memory_writes and not note_writes:
        changed.append("inbox_source_projection_completion")
    if database_writes and not memory_writes and not source_completions:
        changed.append("inbox_projection_custody_update")
    return {
        "source": "ingest_obsidian_inbox",
        "ready_for_operator": True,
        "imported": imported,
        "mirrored": mirrored,
        "mirror_failures": mirror_failures,
        "source_completions": source_completions,
        "skipped": skipped,
        "path_skipped": path_skipped,
        "limit": limit,
        "memory_rows": memory_rows,
        "memory_ids": [row["memory_id"] for row in memory_rows],
        "path_displays": [row["path_display"] for row in memory_rows if row["path_display"]],
        "warnings": warnings,
        "next_commands": [
            "what do you remember",
            "search memory for inbox",
            "daily brief",
            "export state",
        ],
        "boundaries": {
            **_ingest_boundaries(writes=database_writes or note_writes),
            "writes_files": note_writes,
            "writes_database": database_writes,
            "writes_memory": memory_writes,
            "writes_notes": note_writes,
        },
        **_ingest_contract(
            state_changed=database_writes or note_writes,
            changed=changed,
            content_in_handoff=bool(memory_rows),
        ),
    }


def _recent_file_digest_handoff(
    *,
    count: int,
    skipped: int,
    hours: int,
    limit: int,
    path_display: str,
    recent_rows: list[dict[str, Any]],
    watched_count: int,
    writes: bool = True,
) -> dict[str, Any]:
    return {
        "source": "recent_file_digest",
        "ready_for_operator": True,
        "count": count,
        "skipped": skipped,
        "hours": hours,
        "limit": limit,
        "path_display": _short_metadata(path_display, 180),
        "watched_dir_count": watched_count,
        "recent_files": recent_rows,
        "next_commands": [
            "read jarvis note " + _short_metadata(path_display or "Daily", 180),
            "daily brief",
            "export state",
        ],
        "boundaries": {
            **_ingest_boundaries(writes=writes, reads_file_metadata=True),
            "writes_database": False,
            "writes_memory": False,
        },
        **_ingest_contract(
            state_changed=writes,
            changed=["recent_file_digest_export"] if writes else [],
            content_in_handoff=bool(recent_rows),
        ),
    }


def _clear_inbox_handoff(*, path_display: str) -> dict[str, Any]:
    return {
        "source": "clear_obsidian_inbox",
        "ready_for_operator": True,
        "path_display": _short_metadata(path_display, 180),
        "next_commands": [
            "read jarvis note Inbox",
            "ingest inbox",
            "export state",
        ],
        "boundaries": {
            **_ingest_boundaries(writes=True),
            "writes_database": False,
            "writes_memory": False,
        },
        **_ingest_contract(
            state_changed=True,
            changed=["inbox_clear"],
            content_in_handoff=False,
        ),
    }


def _clear_inbox_refusal(
    *,
    reason: str,
    output: str,
    handler_invoked: bool = False,
    reads_private_data: bool = False,
) -> ToolResult:
    return ToolResult(
        "clear_obsidian_inbox",
        False,
        output,
        _safe_metadata(
            reason=reason,
            mutation="inbox_clear",
            destructive=True,
            requires_approval=True,
            requires_confirmation=False,
            executed_handler=handler_invoked,
            handler_invoked=handler_invoked,
            approval_granted=False,
            approval_argument_resolution_status=reason,
            target_binding_status="unbound",
            reads_private_data=reads_private_data,
            authorizes_execution=False,
            authorizes_completion_claim=False,
        ),
    )


def make_clear_inbox_approval_resolver(vault: ObsidianVault):
    def resolve_clear_inbox_approval(
        _: dict[str, Any],
    ) -> ApprovalArgumentResolution | ToolResult:
        try:
            target_binding = vault.inbox_clear_binding(
                max_bytes=MAX_INBOX_SOURCE_CHARS,
            )
        except OverflowError:
            return _clear_inbox_refusal(
                reason="inbox_too_large",
                output=(
                    "The inbox is too large to bind safely for clearing, so no approval was "
                    "queued and nothing changed."
                ),
                reads_private_data=True,
            )
        except InboxSnapshotError:
            return _clear_inbox_refusal(
                reason="inbox_unreadable",
                output=(
                    "The inbox could not be read safely for approval, so no approval was "
                    "queued and nothing changed."
                ),
                reads_private_data=True,
            )
        if target_binding is None:
            return _clear_inbox_refusal(
                reason="inbox_missing",
                output="The inbox does not exist, so no approval was queued and nothing changed.",
                reads_private_data=True,
            )
        return ApprovalArgumentResolution(
            args={"target_binding": target_binding},
            metadata={
                "approval_argument_resolution_status": "resolved",
                "target_binding_status": "reviewed",
                "inbox_target_bound": True,
                "reads_private_data": True,
            },
        )

    return resolve_clear_inbox_approval


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _chunks_from_markdown(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        if len(line) >= 8:
            lines.append(line)

    chunks: list[str] = []
    paragraph: list[str] = []
    for line in lines:
        if len(line) <= 240:
            chunks.append(line)
            continue
        paragraph.append(line)
        if sum(len(item) for item in paragraph) > 600:
            chunks.append(" ".join(paragraph))
            paragraph = []
    if paragraph:
        chunks.append(" ".join(paragraph))
    return chunks


def _bounded_inbox_snapshot(vault: ObsidianVault) -> tuple[list[str], str, bool]:
    """Return chunks and binding derived from the exact same strict source bytes."""
    try:
        snapshot = vault.inbox_snapshot(max_bytes=_INBOX_BINDING_MAX_BYTES)
    except OverflowError:
        return [], "oversize", True
    if snapshot is None:
        return [], "missing", False
    text, binding = snapshot
    if len(text) > MAX_INBOX_SOURCE_CHARS:
        return [], binding, True
    return _chunks_from_markdown(text), binding, False


def _inbox_snapshot_binding(vault: ObsidianVault) -> str:
    """Return an opaque exact Inbox snapshot binding without retaining content."""
    try:
        binding = vault.inbox_clear_binding(max_bytes=_INBOX_BINDING_MAX_BYTES)
    except OverflowError:
        return "oversize"
    return binding or "missing"


def make_inbox_ingest_auto_mutation_operation_key(
    vault: ObsidianVault,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def operation_key(args: dict[str, Any]) -> dict[str, Any]:
        limit = _bounded_int(args.get("limit"), 50, 1, MAX_INBOX_INGEST_LIMIT)
        return {
            "source": "obsidian-inbox",
            "limit": limit,
            "snapshot_binding": _inbox_snapshot_binding(vault),
        }

    return operation_key


def inbox_ingest_auto_mutation_execution_args(
    args: dict[str, Any],
    operation_args: dict[str, Any],
) -> dict[str, Any]:
    if operation_args.get("source") != "obsidian-inbox":
        raise ValueError("invalid inbox source identity")
    binding = operation_args.get("snapshot_binding")
    if type(binding) is not str or (
        binding not in {"missing", "oversize"}
        and re.fullmatch(r"[0-9a-f]{64}", binding) is None
    ):
        raise ValueError("invalid inbox snapshot binding")
    bound = dict(args)
    bound[_INBOX_SOURCE_BINDING_KEY] = binding
    return bound


def _category_for(text: str) -> str:
    low = text.lower()
    if any(word in low for word in ("goal", "project", "build", "ship", "finish")):
        return "goals"
    if any(word in low for word in ("prefer", "like", "dislike", "style", "voice")):
        return "preferences"
    if any(word in low for word in ("remember", "fact", "important")):
        return "facts"
    return "inbox"


def _format_file_row(row: dict[str, Any]) -> str:
    size_kb = int(row["size_bytes"]) / 1024
    return f"- {row['path_display']} | {size_kb:.1f} KB | modified {row['modified']}"


def _watched_file_display(path: Path, watched_roots: list[Path]) -> str:
    resolved = path.resolve()
    for index, root in enumerate(watched_roots, start=1):
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return _short_metadata(f"[watched {index}] {relative}", limit=240)
    return _short_metadata(path, limit=240)


def _is_internal_path(path: Path, config: JarvisConfig | None, vault: ObsidianVault) -> bool:
    if path.suffix in {".sqlite", ".db"} or path.name.endswith(("-wal", "-shm")):
        return True
    try:
        path.relative_to(vault.root_path)
        return True
    except ValueError:
        pass
    return False


def ingest_inbox_notes(
    store: MemoryStore,
    vault: ObsidianVault,
    limit: int = 50,
    *,
    effect_authority: Callable[[], None] | None = None,
    scheduled_job_claim: tuple[int, str] | None = None,
    expected_snapshot_binding: str | None = None,
) -> ToolResult:
    limit = _bounded_int(limit, 50, 1, MAX_INBOX_INGEST_LIMIT)
    chunks, source_binding, source_too_large = _bounded_inbox_snapshot(vault)
    if expected_snapshot_binding is not None:
        if type(expected_snapshot_binding) is not str or (
            expected_snapshot_binding not in {"missing", "oversize"}
            and re.fullmatch(r"[0-9a-f]{64}", expected_snapshot_binding) is None
        ):
            return ToolResult(
                "ingest_obsidian_inbox",
                False,
                "The inbox ingestion source binding was invalid, so nothing was ingested.",
                _safe_metadata(
                    reason="inbox_source_binding_invalid",
                    auto_mutation_effects_started=False,
                    reads_personal_data=True,
                    reads_private_data=True,
                ),
            )
        if not hmac.compare_digest(expected_snapshot_binding, source_binding):
            return ToolResult(
                "ingest_obsidian_inbox",
                False,
                "The Inbox changed before ingestion could start, so nothing was ingested. Retry to review the current Inbox.",
                _safe_metadata(
                    reason="inbox_source_changed",
                    auto_mutation_effects_started=False,
                    reads_personal_data=True,
                    reads_private_data=True,
                ),
            )
    if source_too_large:
        warnings = [
            {
                "code": "obsidian_inbox_source_too_large",
                "count": 1,
                "message": (
                    "Inbox ingestion was refused before any writes because the source exceeds "
                    "the bounded scan size. Archive or split the inbox and retry."
                ),
            }
        ]
        handoff = _inbox_ingest_handoff(
            imported=0,
            mirrored=0,
            mirror_failures=0,
            source_completions=0,
            skipped=0,
            path_skipped=0,
            limit=limit,
            memory_rows=[],
            warnings=warnings,
        )
        return ToolResult(
            "ingest_obsidian_inbox",
            False,
            "Obsidian Inbox is too large for a complete bounded scan; no memories were ingested.",
            _ingest_handoff_metadata(
                "inbox_ingest_handoff",
                handoff,
                imported=0,
                mirrored=0,
                mirror_failures=0,
                source_completions=0,
                skipped=0,
                path_skipped=0,
                limit=limit,
                memory_rows=[],
                warnings=warnings,
                writes_files=False,
                writes_database=False,
                writes_memory=False,
                writes_notes=False,
                reads_personal_data=True,
                reads_private_data=True,
                reason="inbox_source_too_large",
                auto_mutation_effects_started=False,
            ),
        )
    if not chunks:
        handoff = _inbox_ingest_handoff(
            imported=0,
            mirrored=0,
            mirror_failures=0,
            source_completions=0,
            skipped=0,
            path_skipped=0,
            limit=limit,
            memory_rows=[],
            warnings=[],
        )
        return ToolResult(
            "ingest_obsidian_inbox",
            True,
            "Obsidian Inbox has no ingestible notes.",
            _ingest_handoff_metadata(
                "inbox_ingest_handoff",
                handoff,
                imported=0,
                mirrored=0,
                mirror_failures=0,
                source_completions=0,
                skipped=0,
                limit=limit,
                path_skipped=0,
                memory_rows=[],
                warnings=[],
                reads_personal_data=True,
                reads_private_data=True,
            ),
        )

    imported = 0
    mirrored = 0
    mirror_failures = 0
    source_completions = 0
    skipped = 0
    path_skipped = 0
    path_displays: list[str] = []
    memory_rows: list[dict[str, Any]] = []
    database_writes = False
    note_writes = False
    actionable = 0
    source_changed_after_effects = False
    projection_effect_outcome_uncertain = False
    for chunk in chunks:
        if LOCAL_PATH_RE.search(chunk):
            skipped += 1
            path_skipped += 1
            continue
        source_key = "obsidian-inbox:" + hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        title = chunk[:70].rstrip(".")
        record = MemoryRecord(
            category=_category_for(chunk),
            title=title,
            body=chunk,
            source="obsidian-inbox",
            confidence=0.9,
        )
        if expected_snapshot_binding is not None and not hmac.compare_digest(
            expected_snapshot_binding,
            _inbox_snapshot_binding(vault),
        ):
            if database_writes or note_writes:
                source_changed_after_effects = True
                break
            return ToolResult(
                "ingest_obsidian_inbox",
                False,
                "The Inbox changed before ingestion could start, so nothing was ingested. Retry to review the current Inbox.",
                _safe_metadata(
                    reason="inbox_source_changed",
                    auto_mutation_effects_started=False,
                    reads_personal_data=True,
                    reads_private_data=True,
                ),
            )
        _require_effect_authority(effect_authority)
        try:
            if scheduled_job_claim is None:
                inserted, target = store.add_memory_if_source_new_with_projection(
                    record,
                    source_key,
                    "obsidian-inbox",
                )
            else:
                job_id, lease_token = scheduled_job_claim
                inserted, target = store.add_memory_if_source_new_for_job_claim_with_projection(
                    record,
                    source_key,
                    "obsidian-inbox",
                    job_id=job_id,
                    lease_token=lease_token,
                )
        except ScheduledJobLeaseAuthorityLost as exc:
            raise _EffectAuthorityLost("scheduled effect authority lost") from exc
        if not inserted:
            skipped += 1
        else:
            imported += 1
            database_writes = True

        path_display = ""
        mirrored_this_chunk = False
        projection_completed = False
        projection_status = "pending_error"
        content_digest_present = False
        source_completed_now = False
        projection_file_written = False
        mirror_repaired = False
        try:
            outcome = reconcile_memory_projection(
                store,
                vault,
                target.memory_id,
                expected_operation=target.operation,
                expected_revision=target.revision,
                expected_source_digest=target.source_digest,
                effect_authority=(
                    (lambda: _require_effect_authority(effect_authority))
                    if effect_authority is not None
                    else None
                ),
            )
        except _EffectAuthorityLost:
            raise
        except Exception:
            # The reconciler performs filesystem effects.  A write-then-raise cannot
            # prove absence, so receipt custody must remain conservative.
            projection_effect_outcome_uncertain = True
            database_writes = (
                store.mark_ingested_source_projection_pending(
                    source_key,
                    target.memory_id,
                    target.revision,
                    target.source_digest,
                )
                or database_writes
            )
        else:
            projection_status = _short_metadata(outcome.completion_status or outcome.status, 32)
            content_digest_present = bool(outcome.content_digest)
            projection_file_written = outcome.file_written
            mirror_repaired = outcome.completion_status == "repaired"
            if outcome.status == "pending_error" or outcome.completion_status in {"published", "repaired"}:
                database_writes = True
            if outcome.file_written:
                note_writes = True
                if outcome.path_display:
                    path_display = _safe_vault_path_display(
                        vault.root_path / str(outcome.path_display),
                        vault,
                    )
            if outcome.completion_status in {"published", "repaired"}:
                mirrored_this_chunk = True
                mirrored += 1
                note_writes = True
                path_display = _safe_vault_path_display(
                    vault.root_path / str(outcome.path_display),
                    vault,
                )
            if outcome.completion_status in {"published", "repaired", "verified"} and (
                outcome.status == "completed" or outcome.completion_status == "verified"
            ):
                try:
                    with vault.canonical_memory_projection_evidence_lock(
                        memory_id=target.memory_id,
                        store_identity=store.get_store_identity(),
                        expected_relative_path=str(outcome.path_display),
                        expected_content_digest=str(outcome.content_digest),
                    ) as evidence_is_current:
                        if not evidence_is_current:
                            raise RuntimeError("memory projection evidence changed before source completion")
                        if scheduled_job_claim is None:
                            _require_effect_authority(effect_authority)
                            evidence = store.complete_ingested_source_projection(
                                source_key,
                                target.memory_id,
                                target.revision,
                                target.source_digest,
                            )
                        else:
                            job_id, lease_token = scheduled_job_claim
                            evidence = store.complete_ingested_source_projection_for_job_claim(
                                source_key,
                                target.memory_id,
                                target.revision,
                                target.source_digest,
                                job_id=job_id,
                                lease_token=lease_token,
                            )
                except ScheduledJobLeaseAuthorityLost as exc:
                    raise _EffectAuthorityLost("scheduled effect authority lost") from exc
                except _EffectAuthorityLost:
                    raise
                except Exception:
                    database_writes = (
                        store.mark_ingested_source_projection_pending(
                            source_key,
                            target.memory_id,
                            target.revision,
                            target.source_digest,
                        )
                        or database_writes
                    )
                    projection_status = "pending_error"
                else:
                    projection_completed = True
                    source_completed_now = evidence["completed_now"] is True
                    if source_completed_now:
                        source_completions += 1
                    path_display = _safe_vault_path_display(
                        vault.root_path / str(evidence["canonical_path_display"]),
                        vault,
                    )
                    content_digest_present = bool(evidence["content_digest"])
                    database_writes = database_writes or source_completed_now
            else:
                database_writes = (
                    store.mark_ingested_source_projection_pending(
                        source_key,
                        target.memory_id,
                        target.revision,
                        target.source_digest,
                    )
                    or database_writes
                )

        if not projection_completed:
            mirror_failures += 1
        if inserted or mirrored_this_chunk or source_completed_now or not projection_completed:
            if path_display and path_display not in path_displays:
                path_displays.append(path_display)
            actionable += 1
            memory_rows.append(
                {
                    "memory_id": target.memory_id,
                    "category": record.category,
                    "title": _short_metadata(record.title, 120),
                    "path_display": path_display,
                    "mirror_written": mirrored_this_chunk or projection_file_written,
                    "mirror_repaired": mirror_repaired,
                    "projection_status": projection_status,
                    "revision": target.revision,
                    "source_digest_present": bool(target.source_digest),
                    "content_digest_present": content_digest_present,
                }
            )
            if actionable >= limit:
                break

    if expected_snapshot_binding is not None and not hmac.compare_digest(
        expected_snapshot_binding,
        _inbox_snapshot_binding(vault),
    ):
        if database_writes or note_writes:
            source_changed_after_effects = True
        else:
            return ToolResult(
                "ingest_obsidian_inbox",
                False,
                "The Inbox changed before ingestion could start, so nothing was ingested. Retry to review the current Inbox.",
                _safe_metadata(
                    reason="inbox_source_changed",
                    auto_mutation_effects_started=False,
                    reads_personal_data=True,
                    reads_private_data=True,
                ),
            )

    warnings = []
    if mirror_failures:
        warnings.append(
            {
                "code": "obsidian_memory_mirror_write_failed",
                "count": mirror_failures,
                "message": (
                    "Durable memory was saved, but its projection remains pending repair. "
                    "Check vault write access or note integrity, then retry inbox ingestion."
                ),
            }
        )
    if source_changed_after_effects:
        warnings.append(
            {
                "code": "obsidian_inbox_source_changed_during_ingest",
                "count": 1,
                "message": "Inbox changed after ingestion began; review the current Inbox before another run.",
            }
        )
    output = (
        f"Ingested {imported} inbox memory row(s); wrote {mirrored} Obsidian mirror(s); "
        f"skipped {skipped} duplicate source(s)."
    )
    if path_displays:
        output += "\n" + "\n".join(f"- {path_display}" for path_display in path_displays[:8])
    if mirror_failures:
        output += (
            f"\nWarning: {mirror_failures} durable memory projection(s) remain pending repair. "
            "The memory rows are saved; check vault write access or note integrity, then retry."
        )
    handoff = _inbox_ingest_handoff(
        imported=imported,
        mirrored=mirrored,
        mirror_failures=mirror_failures,
        source_completions=source_completions,
        skipped=skipped,
        path_skipped=path_skipped,
        limit=limit,
        memory_rows=memory_rows,
        warnings=warnings,
        database_writes_override=database_writes,
        note_writes_override=note_writes,
    )
    return ToolResult(
        "ingest_obsidian_inbox",
        mirror_failures == 0 and not source_changed_after_effects,
        output,
        _ingest_handoff_metadata(
            "inbox_ingest_handoff",
            handoff,
            imported=imported,
            mirrored=mirrored,
            mirror_failures=mirror_failures,
            source_completions=source_completions,
            skipped=skipped,
            path_skipped=path_skipped,
            limit=limit,
            path_displays=path_displays,
            memory_rows=memory_rows,
            warnings=warnings,
            writes_files=note_writes,
            writes_database=database_writes,
            writes_memory=imported > 0,
            writes_notes=note_writes,
            reads_personal_data=True,
            reads_private_data=True,
            reason="inbox_source_changed" if source_changed_after_effects else None,
            auto_mutation_effects_started=(
                database_writes or note_writes or projection_effect_outcome_uncertain
            ),
        ),
    )


def build_recent_file_digest(
    vault: ObsidianVault,
    config: JarvisConfig | None = None,
    directory: str = "",
    hours: int = 24,
    limit: int = 25,
    *,
    effect_authority: Callable[[], None] | None = None,
    scheduled_note_key: str = "",
    scheduled_note_date: str = "",
) -> ToolResult:
    watched = list(config.watched_dirs if config else ())
    if directory.strip():
        watched = [Path(directory).expanduser()]
    hours = _bounded_int(hours, 24, 1, MAX_DIGEST_HOURS)
    limit = _bounded_int(limit, 25, 1, MAX_DIGEST_LIMIT)
    if not watched:
        handoff = _recent_file_digest_handoff(
            count=0,
            skipped=0,
            hours=hours,
            limit=limit,
            path_display="",
            recent_rows=[],
            watched_count=0,
            writes=False,
        )
        return ToolResult(
            "recent_file_digest",
            True,
            "No watched directories configured. Set JARVIS_WATCHED_DIRS with colon-separated paths.",
            _ingest_handoff_metadata(
                "recent_file_digest_handoff",
                handoff,
                count=0,
                skipped=0,
                hours=hours,
                limit=limit,
                recent_file_rows=[],
                reads_file_metadata=True,
            ),
        )

    cutoff = datetime.now() - timedelta(hours=hours)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for root in watched:
        if not root.exists():
            skipped += 1
            continue
        for path in root.rglob("*"):
            if len(rows) >= limit:
                break
            try:
                if not path.is_file() or any(part.startswith(".") for part in path.parts):
                    continue
                if _is_internal_path(path, config, vault):
                    continue
                stat = path.stat()
                modified_at = datetime.fromtimestamp(stat.st_mtime)
                if modified_at >= cutoff:
                    rows.append(
                        {
                            "path": path,
                            "path_display": _watched_file_display(path, watched),
                            "size_bytes": stat.st_size,
                            "modified": modified_at.strftime("%Y-%m-%d %H:%M"),
                            "modified_timestamp": stat.st_mtime,
                        }
                    )
            except OSError:
                skipped += 1
        if len(rows) >= limit:
            break

    lines = [f"Watched directories: {len(watched)} configured (paths hidden)", f"Window: last {hours} hours", ""]
    sorted_rows = sorted(rows, key=lambda item: item["modified_timestamp"], reverse=True)
    if rows:
        lines.append("## Recent Files")
        lines.extend(_format_file_row(row) for row in sorted_rows)
    else:
        lines.append("No recent files found.")
    if skipped:
        lines.append(f"\nSkipped {skipped} inaccessible or missing item(s).")
    body = "\n".join(lines)
    recent_rows = [
        {
            "path_display": row["path_display"],
            "size_bytes": row["size_bytes"],
            "modified": row["modified"],
        }
        for row in sorted_rows
    ]
    if scheduled_note_key:
        note_written = False
        note_path = ""
        path_display = ""
        output = body
    else:
        _require_effect_authority(effect_authority)
        note_path = vault.append_daily("Jarvis Recent File Digest", body)
        note_written = True
        path_display = _safe_vault_path_display(note_path, vault)
        output = f"Recent file digest written: {path_display}\n\n{body}"
    handoff = _recent_file_digest_handoff(
        count=len(rows),
        skipped=skipped,
        hours=hours,
        limit=limit,
        path_display=path_display,
        recent_rows=recent_rows,
        watched_count=len(watched),
        writes=note_written,
    )
    return ToolResult(
        "recent_file_digest",
        True,
        output,
        _ingest_handoff_metadata(
            "recent_file_digest_handoff",
            handoff,
            count=len(rows),
            skipped=skipped,
            hours=hours,
            limit=limit,
            path_display=path_display,
            watched_dir_count=len(watched),
            recent_file_rows=recent_rows,
            writes_files=note_written,
            reads_file_metadata=True,
            writes_notes=note_written,
            scheduled_note_replayed=False,
        ),
    )


def make_ingest_tools(store: MemoryStore, vault: ObsidianVault, config: JarvisConfig | None = None):
    def ingest_obsidian_inbox(args: dict[str, Any]) -> ToolResult:
        return ingest_inbox_notes(
            store,
            vault,
            _bounded_int(args.get("limit"), 50, 1, MAX_INBOX_INGEST_LIMIT),
            expected_snapshot_binding=args.get(_INBOX_SOURCE_BINDING_KEY),
        )

    def clear_obsidian_inbox(args: dict[str, Any]) -> ToolResult:
        target_binding = args.get("target_binding")
        if type(target_binding) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", target_binding
        ):
            return _clear_inbox_refusal(
                reason="invalid_target_binding",
                output=(
                    "The approved inbox target was not bound safely, so nothing was cleared. "
                    "Issue a fresh clear-inbox request."
                ),
                handler_invoked=True,
            )
        try:
            path, cleared = vault.reset_inbox_if_unchanged(
                target_binding,
                max_bytes=MAX_INBOX_SOURCE_CHARS,
            )
        except OverflowError:
            return _clear_inbox_refusal(
                reason="inbox_too_large",
                output=(
                    "The inbox is now too large to clear safely. Nothing changed; review it "
                    "and issue a fresh request."
                ),
                handler_invoked=True,
                reads_private_data=True,
            )
        except InboxSnapshotError:
            return _clear_inbox_refusal(
                reason="inbox_unreadable",
                output=(
                    "The inbox could not be revalidated safely. Nothing changed; review it "
                    "and issue a fresh request."
                ),
                handler_invoked=True,
                reads_private_data=True,
            )
        if not cleared:
            return ToolResult(
                "clear_obsidian_inbox",
                False,
                (
                    "The inbox changed after approval. Nothing was cleared; review the current "
                    "inbox and issue a fresh request if still intended."
                ),
                _safe_metadata(
                    reason="stale_inbox_binding",
                    mutation="inbox_clear",
                    destructive=True,
                    requires_approval=True,
                    target_binding_status="changed",
                    approval_rerun_blocked=True,
                    approval_rerun_block_reason="inbox_target_changed",
                    outcome_known=True,
                    side_effect_possible=False,
                    writes_files=False,
                    writes_notes=False,
                    reads_private_data=True,
                ),
            )
        path_display = _safe_vault_path_display(path, vault)
        handoff = _clear_inbox_handoff(path_display=path_display)
        return ToolResult(
            "clear_obsidian_inbox",
            True,
            f"Cleared Obsidian Inbox: {path_display}",
            _ingest_handoff_metadata(
                "clear_inbox_handoff",
                handoff,
                path=str(path),
                path_display=path_display,
                writes_files=True,
                executes_side_effect=True,
                destructive=True,
                requires_approval=True,
                reads_private_data=True,
                writes_notes=True,
            ),
        )

    def recent_file_digest(args: dict[str, Any]) -> ToolResult:
        return build_recent_file_digest(
            vault,
            config,
            directory=str(args.get("directory") or ""),
            hours=_bounded_int(args.get("hours"), 24, 1, MAX_DIGEST_HOURS),
            limit=_bounded_int(args.get("limit"), 25, 1, MAX_DIGEST_LIMIT),
        )

    return ingest_obsidian_inbox, clear_obsidian_inbox, recent_file_digest
