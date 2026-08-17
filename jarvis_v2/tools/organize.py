from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.goal_projection import reconcile_goal_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    DecisionRecord,
    GoalRecord,
    MemoryRecord,
    MemoryStore,
    OrganizedNoteBatchReservation,
    OrganizedNoteEntryRecord,
    TaskRecord,
)


MAX_ORGANIZE_TEXT_CHARS = 50000
MAX_ORGANIZE_LINES = 200
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")


@dataclass(frozen=True)
class ParsedOrganizeNote:
    entries: tuple[OrganizedNoteEntryRecord, ...]
    skipped: tuple[str, ...]
    overflow: int
    lines_processed: int


def _redact_local_paths(text: str) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", text)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "reads_private_data": False,
        "controls_computer": False,
        "queues_approval": False,
        "requires_approval": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
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


def _organize_contract(
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


def _organize_handoff_metadata(
    handoff_key: str,
    handoff: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    metadata = _safe_metadata(**extra)
    metadata.update(
        _organize_contract(
            state_changed=_metadata_bool(handoff.get("state_changed")),
            changed=list(handoff.get("changed") or []),
            content_in_handoff=_metadata_bool(handoff.get("content_in_handoff")),
        )
    )
    metadata[f"{handoff_key}_ready"] = True
    metadata[handoff_key] = handoff
    return metadata


def _short_metadata(value: Any, limit: int = 160) -> str:
    text = _redact_local_paths(str(value or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _organize_refusal_boundaries() -> dict[str, bool]:
    return {
        "read_only": True,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_tasks": False,
        "writes_goals": False,
        "writes_decisions": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
    }


def _organize_refusal_handoff(
    *,
    reason: str,
    text: str = "",
    skipped: list[str] | None = None,
    overflow: int = 0,
    lines_processed: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    skipped_rows = skipped or []
    handoff: dict[str, Any] = {
        "source": "organize_note",
        "ready_for_operator": True,
        "mutation": "brain_dump_organize",
        "reason": reason,
        "refused": True,
        "created": {
            "task_ids": [],
            "goal_ids": [],
            "memory_ids": [],
            "decision_ids": [],
            "task_count": 0,
            "goal_count": 0,
            "memory_count": 0,
            "decision_count": 0,
        },
        "created_total": 0,
        "changed": [],
        "text_preview": _short_metadata(text, 220),
        "text_chars": len(text),
        "skipped_count": len(skipped_rows),
        "skipped_preview": [_short_metadata(line, 180) for line in skipped_rows[:5]],
        "overflow": overflow,
        "lines_processed": lines_processed,
        "retry_command": "organize brain dump: task: <next action>",
        "next_commands": ["help organize", "tasks", "goals", "what do you remember"],
        "boundaries": _organize_refusal_boundaries(),
        **_organize_contract(
            content_in_handoff=bool(text or skipped_rows),
        ),
    }
    if limit is not None:
        handoff["limit"] = limit
    return handoff


def _organize_refusal_metadata(
    *,
    reason: str,
    text: str = "",
    skipped: list[str] | None = None,
    overflow: int = 0,
    lines_processed: int = 0,
    limit: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    handoff = _organize_refusal_handoff(
        reason=reason,
        text=text,
        skipped=skipped,
        overflow=overflow,
        lines_processed=lines_processed,
        limit=limit,
    )
    metadata = _organize_handoff_metadata(
        "organize_refusal_handoff",
        handoff,
        reason=reason,
        **extra,
    )
    if limit is not None:
        metadata["limit"] = limit
    metadata["skipped"] = len(skipped or [])
    metadata["overflow"] = overflow
    metadata["lines_processed"] = lines_processed
    return metadata


def _organize_handoff(
    *,
    created_tasks: list[int],
    created_goals: list[int],
    created_memories: list[int],
    created_decisions: list[int],
    skipped: list[str],
    overflow: int,
    lines_processed: int,
    state_changed: bool,
    writes_files: bool,
    writes_database: bool,
    writes_memory: bool,
    writes_notes: bool,
    created_now: bool,
) -> dict[str, Any]:
    affected = {
        "task_ids": created_tasks,
        "goal_ids": created_goals,
        "memory_ids": created_memories,
        "decision_ids": created_decisions,
        "task_count": len(created_tasks),
        "goal_count": len(created_goals),
        "memory_count": len(created_memories),
        "decision_count": len(created_decisions),
    }
    created = (
        affected
        if created_now
        else {
            "task_ids": [],
            "goal_ids": [],
            "memory_ids": [],
            "decision_ids": [],
            "task_count": 0,
            "goal_count": 0,
            "memory_count": 0,
            "decision_count": 0,
        }
    )
    next_commands = []
    if created_tasks:
        next_commands.extend(["tasks", f"show task {created_tasks[0]}"])
    if created_goals:
        next_commands.extend(["goals", f"goal status {created_goals[0]}"])
    if created_memories:
        next_commands.extend(["what do you remember", f"memory {created_memories[0]}"])
    if created_decisions:
        next_commands.extend(["decisions", f"show decision {created_decisions[0]}"])
    next_commands.extend(["daily brief", "export state"])
    deduped_commands: list[str] = []
    for command in next_commands:
        if command not in deduped_commands:
            deduped_commands.append(command)
    return {
        "source": "organize_note",
        "ready_for_operator": True,
        "created": created,
        "affected": affected,
        "created_total": sum(created[key] for key in ("task_count", "goal_count", "memory_count", "decision_count")),
        "affected_total": sum(affected[key] for key in ("task_count", "goal_count", "memory_count", "decision_count")),
        "skipped_count": len(skipped),
        "skipped_preview": [_short_metadata(line, 180) for line in skipped[:5]],
        "overflow": overflow,
        "lines_processed": lines_processed,
        "next_commands": deduped_commands,
        "boundaries": {
            "read_only": not state_changed,
            "writes_files": writes_files,
            "writes_database": writes_database,
            "writes_memory": writes_memory,
            "writes_notes": writes_notes,
            "writes_tasks": created_now and bool(created_tasks),
            "writes_goals": created_now and bool(created_goals),
            "writes_decisions": created_now and bool(created_decisions),
            "queues_approval": False,
            "controls_computer": False,
            "external_side_effect": False,
        },
        **_organize_contract(
            state_changed=state_changed,
            changed=[
                key
                for key, ids in (
                    ("tasks", created_tasks),
                    ("goals", created_goals),
                    ("memories", created_memories),
                    ("decisions", created_decisions),
                )
                if ids and state_changed
            ],
            content_in_handoff=True,
        ),
    }


def _clean_line(line: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)]|\[[ xX]\])\s*", "", line).strip()
    return re.sub(r"[ \t]+", " ", cleaned)


def _split_note(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1 and ";" in text:
        lines = [line.strip() for line in text.split(";") if line.strip()]
    return [_clean_line(line) for line in lines]


def _split_fields(body: str, max_parts: int) -> list[str]:
    parts = [part.strip() for part in body.split("|")]
    while len(parts) < max_parts:
        parts.append("")
    return parts[:max_parts]


def _normalized_organize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _parse_organize_note(text: str) -> ParsedOrganizeNote:
    entries: list[OrganizedNoteEntryRecord] = []
    skipped: list[str] = []
    lines = _split_note(text)
    overflow = max(0, len(lines) - MAX_ORGANIZE_LINES)
    for line in lines[:MAX_ORGANIZE_LINES]:
        match = re.match(
            r"^(?P<kind>task|todo|remember|memory|note|goal|project|decision)\s*:\s*(?P<body>.+)$",
            line,
            re.IGNORECASE,
        )
        if not match:
            skipped.append(_redact_local_paths(line))
            continue
        raw_kind = match.group("kind").lower()
        body = match.group("body").strip()
        if (
            not body
            or LOCAL_PATH_RE.search(body)
            or any(ord(character) < 32 or ord(character) == 127 for character in body)
        ):
            skipped.append(_redact_local_paths(line))
            continue
        if raw_kind in {"task", "todo"}:
            priority = (
                "high"
                if re.search(r"\b(priority\s+high|high priority|urgent)\b", body, re.IGNORECASE)
                else "normal"
            )
            due = ""
            due_match = re.search(
                r"\bdue\s+(.+?)(?:\s+priority\s+(?:low|normal|high)|$)",
                body,
                re.IGNORECASE,
            )
            if due_match:
                due = due_match.group(1).strip()
                body = (body[: due_match.start()] + body[due_match.end() :]).strip()
            body = re.sub(
                r"\b(?:priority\s+high|high priority|urgent)\b",
                "",
                body,
                flags=re.IGNORECASE,
            ).strip()
            if not body:
                skipped.append(_redact_local_paths(line))
                continue
            entries.append(
                OrganizedNoteEntryRecord(
                    "task",
                    task=TaskRecord(body=body, due=due, priority=priority, source="brain-dump"),
                )
            )
            continue
        if raw_kind in {"remember", "memory", "note"}:
            title = body[:60].rstrip(".")
            entries.append(
                OrganizedNoteEntryRecord(
                    "memory",
                    memory=MemoryRecord("facts", title, body, "brain-dump"),
                )
            )
            continue
        if raw_kind in {"goal", "project"}:
            title, purpose, horizon = _split_fields(body, 3)
            entries.append(
                OrganizedNoteEntryRecord(
                    "goal",
                    goal=GoalRecord(title=title, purpose=purpose, horizon=horizon),
                )
            )
            continue
        title, rationale, impact = _split_fields(body, 3)
        memory_body = f"{title}\n\nRationale: {rationale}\n\nImpact: {impact}".strip()
        entries.append(
            OrganizedNoteEntryRecord(
                "decision",
                decision=DecisionRecord(title=title, rationale=rationale, impact=impact),
                decision_memory=MemoryRecord("decisions", title, memory_body, "brain-dump"),
            )
        )
    return ParsedOrganizeNote(
        tuple(entries),
        tuple(skipped),
        overflow,
        min(len(lines), MAX_ORGANIZE_LINES),
    )


def _organized_entry_operation_payload(entry: OrganizedNoteEntryRecord) -> dict[str, Any]:
    if entry.kind == "task":
        return {
            "kind": "task",
            "body": entry.task.body,
            "due": entry.task.due,
            "priority": entry.task.priority,
            "source": entry.task.source,
            "status": entry.task.status,
        }
    if entry.kind == "goal":
        return {
            "kind": "goal",
            "title": entry.goal.title,
            "purpose": entry.goal.purpose,
            "horizon": entry.goal.horizon,
            "status": entry.goal.status,
        }
    if entry.kind == "memory":
        return {
            "kind": "memory",
            "category": entry.memory.category,
            "title": entry.memory.title,
            "body": entry.memory.body,
            "source": entry.memory.source,
            "confidence": entry.memory.confidence,
        }
    return {
        "kind": "decision",
        "title": entry.decision.title,
        "rationale": entry.decision.rationale,
        "impact": entry.decision.impact,
        "status": entry.decision.status,
        "memory": {
            "category": entry.decision_memory.category,
            "title": entry.decision_memory.title,
            "body": entry.decision_memory.body,
            "source": entry.decision_memory.source,
            "confidence": entry.decision_memory.confidence,
        },
    }


def organize_note_auto_mutation_operation_key(args: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_organize_note(_normalized_organize_text(args.get("text")))
    return {
        "version": 1,
        "entries": [_organized_entry_operation_payload(entry) for entry in parsed.entries],
    }


def organize_note_auto_mutation_preflight(args: dict[str, Any]) -> str | None:
    text = _normalized_organize_text(args.get("text"))
    if not text:
        return "missing_text"
    if len(text) > MAX_ORGANIZE_TEXT_CHARS:
        return "text_too_large"
    parsed = _parse_organize_note(text)
    if parsed.overflow:
        return "too_many_lines"
    if not parsed.entries:
        return "no_recognizable_lines"
    return None


def organize_note_auto_mutation_preflight_result(
    args: dict[str, Any],
    reason: str,
) -> ToolResult:
    text = _normalized_organize_text(args.get("text"))
    parsed = _parse_organize_note(text) if text and len(text) <= MAX_ORGANIZE_TEXT_CHARS else None
    metadata = _organize_refusal_metadata(
        reason=reason,
        text=text,
        skipped=list(parsed.skipped) if parsed is not None else [],
        overflow=parsed.overflow if parsed is not None else 0,
        lines_processed=(
            0 if reason == "too_many_lines" else parsed.lines_processed if parsed is not None else 0
        ),
        limit=(
            MAX_ORGANIZE_TEXT_CHARS
            if reason == "text_too_large"
            else MAX_ORGANIZE_LINES if reason == "too_many_lines" else None
        ),
    )
    metadata.update(
        {
            "failure_kind": "auto_mutation_semantic_preflight_rejected",
            "requires_confirmation": False,
            "executed_handler": False,
            "handler_invoked": False,
            "planned_arg_keys": sorted(str(key)[:80] for key in args),
            "authorizes_retry": False,
        }
    )
    outputs = {
        "missing_text": "Note text is required.",
        "text_too_large": (
            f"Note text is too large ({len(text)} chars). Limit is {MAX_ORGANIZE_TEXT_CHARS}."
        ),
        "too_many_lines": (
            f"Brain dump has {parsed.overflow if parsed is not None else 0} line(s) beyond the safe "
            f"{MAX_ORGANIZE_LINES}-line limit. Split it into smaller requests; nothing was organized."
        ),
        "no_recognizable_lines": (
            "No recognizable brain-dump lines found. Use prefixes like task:, remember:, goal:, or decision:."
        ),
    }
    return ToolResult(
        "organize_note",
        False,
        outputs.get(reason, "The brain dump failed deterministic validation; nothing was organized."),
        metadata,
    )


def _organized_note_source_key(parsed: ParsedOrganizeNote) -> str:
    material = json.dumps(
        {
            "version": 1,
            "entries": [_organized_entry_operation_payload(entry) for entry in parsed.entries],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "organize-note:v1:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _organized_ids(
    reservation: OrganizedNoteBatchReservation,
) -> tuple[list[int], list[int], list[int], list[int]]:
    memory_ids = [entry.memory_id for entry in reservation.entries if entry.memory_id is not None]
    memory_ids.extend(
        entry.decision_memory_id
        for entry in reservation.entries
        if entry.decision_memory_id is not None
    )
    return (
        [entry.task_id for entry in reservation.entries if entry.task_id is not None],
        [entry.goal_id for entry in reservation.entries if entry.goal_id is not None],
        memory_ids,
        [entry.decision_id for entry in reservation.entries if entry.decision_id is not None],
    )


def _organized_goal_source_is_original(
    goal: Any,
    steps: Any,
    entry: OrganizedNoteEntryRecord,
    created_at: str,
) -> bool:
    return bool(
        goal["revision"] == 1
        and goal["title"] == entry.goal.title
        and goal["purpose"] == entry.goal.purpose
        and goal["horizon"] == entry.goal.horizon
        and goal["status"] == entry.goal.status
        and goal["created_at"] == created_at
        and goal["updated_at"] == created_at
        and not tuple(steps)
    )


def _completed_goal_projection_matches(
    job: Any,
    target: Any,
    content_digest: str,
) -> bool:
    return bool(
        job is not None
        and job["state"] == "completed"
        and job["goal_revision"] == target.revision
        and job["source_digest"] == target.source_digest
        and job["content_digest"] == content_digest
    )


def make_organize_tools(store: MemoryStore, vault: ObsidianVault):
    def organize_note(args: dict[str, Any]) -> ToolResult:
        text = _normalized_organize_text(args.get("text"))
        if not text:
            return ToolResult(
                "organize_note",
                False,
                "Note text is required.",
                _organize_refusal_metadata(reason="missing_text", text=text),
            )
        if len(text) > MAX_ORGANIZE_TEXT_CHARS:
            return ToolResult(
                "organize_note",
                False,
                f"Note text is too large ({len(text)} chars). Limit is {MAX_ORGANIZE_TEXT_CHARS}.",
                _organize_refusal_metadata(reason="text_too_large", text=text, limit=MAX_ORGANIZE_TEXT_CHARS),
            )
        parsed = _parse_organize_note(text)
        skipped = list(parsed.skipped)
        if parsed.overflow:
            return ToolResult(
                "organize_note",
                False,
                (
                    f"Brain dump has {parsed.overflow} line(s) beyond the safe {MAX_ORGANIZE_LINES}-line limit. "
                    "Split it into smaller requests; nothing was organized."
                ),
                _organize_refusal_metadata(
                    reason="too_many_lines",
                    text=text,
                    skipped=skipped,
                    overflow=parsed.overflow,
                    lines_processed=0,
                    limit=MAX_ORGANIZE_LINES,
                ),
            )
        if not parsed.entries:
            return ToolResult(
                "organize_note",
                False,
                "No recognizable brain-dump lines found. Use prefixes like task:, remember:, goal:, or decision:.",
                _organize_refusal_metadata(
                    reason="no_recognizable_lines",
                    text=text,
                    skipped=skipped,
                    lines_processed=parsed.lines_processed,
                ),
            )

        store_identity = store.get_store_identity()
        if (
            type(store_identity) is not str
            or re.fullmatch(r"[0-9a-f]{32}", store_identity) is None
        ):
            raise RuntimeError("Organizer projection store identity is unavailable.")
        source_key = _organized_note_source_key(parsed)
        reservation = store.reserve_organized_note_batch(source_key, parsed.entries)
        created_tasks, created_goals, created_memories, created_decisions = _organized_ids(reservation)
        goal_prior_content_digests: dict[int, tuple[str, ...]] = {}
        if reservation.state == "pending":
            for index, (entry, state) in enumerate(
                zip(parsed.entries, reservation.entries)
            ):
                if entry.kind != "goal" or state.goal_id is None:
                    continue
                target = store.ensure_current_goal_projection_job(state.goal_id)
                snapshot = (
                    store.get_current_goal_projection_snapshot(target)
                    if target is not None
                    else None
                )
                if snapshot is None:
                    raise RuntimeError(
                        "organized goal projection source is unavailable"
                    )
                goal, steps = snapshot
                if not _organized_goal_source_is_original(
                    goal,
                    steps,
                    entry,
                    reservation.created_at,
                ):
                    continue
                candidate = vault.goal_projection_candidate_evidence(
                    goal,
                    steps,
                    store_identity=store_identity,
                )
                if (
                    type(candidate) is not tuple
                    or len(candidate) != 2
                    or re.fullmatch(r"[0-9a-f]{64}", str(candidate[0])) is None
                    or candidate[1] != target.source_digest
                ):
                    raise RuntimeError(
                        "organized goal projection candidate is invalid"
                    )
                prior_evidence = vault.inspect_goal_projection_prior_evidence(
                    goal,
                    steps,
                    store_identity=store_identity,
                )
                if (
                    type(prior_evidence) is not tuple
                    or len(prior_evidence) != 2
                    or prior_evidence[0] not in {"absent", "legacy", "owned"}
                    or (
                        prior_evidence[0] == "owned"
                        and re.fullmatch(
                            r"[0-9a-f]{64}", str(prior_evidence[1])
                        )
                        is None
                    )
                    or (
                        prior_evidence[0] != "owned"
                        and prior_evidence[1] != ""
                    )
                ):
                    raise RuntimeError(
                        "organized goal projection destination is unverified"
                    )
                existing_content_digest = (
                    prior_evidence[1]
                    if prior_evidence[0] == "owned"
                    else None
                )
                current_job = store.get_goal_projection_job(state.goal_id)
                if _completed_goal_projection_matches(
                    current_job,
                    target,
                    candidate[0],
                ):
                    if existing_content_digest != candidate[0]:
                        refreshed_evidence = (
                            vault.inspect_goal_projection_prior_evidence(
                                goal,
                                steps,
                                store_identity=store_identity,
                            )
                        )
                        existing_content_digest = (
                            refreshed_evidence[1]
                            if refreshed_evidence[0] == "owned"
                            else None
                        )
                    if existing_content_digest != candidate[0]:
                        raise RuntimeError(
                            "organized goal projection completed evidence is unavailable"
                        )
                    continue
                prepared = store.prepare_goal_projection_after_observed_content(
                    target,
                    existing_content_digest,
                    candidate[0],
                )
                if not prepared:
                    current_job = store.get_goal_projection_job(state.goal_id)
                    if _completed_goal_projection_matches(
                        current_job,
                        target,
                        candidate[0],
                    ):
                        refreshed_evidence = (
                            vault.inspect_goal_projection_prior_evidence(
                                goal,
                                steps,
                                store_identity=store_identity,
                            )
                        )
                        if refreshed_evidence == ("owned", candidate[0]):
                            continue
                    allowed = store.goal_projection_prior_content_digests(target)
                    if allowed is not None and candidate[0] in allowed:
                        goal_prior_content_digests[index] = allowed
                        continue
                    raise RuntimeError(
                        "organized goal projection preparation was lost"
                    )
                allowed = store.goal_projection_prior_content_digests(target)
                if allowed is None or candidate[0] not in allowed:
                    raise RuntimeError(
                        "organized goal projection evidence is unavailable"
                    )
                goal_prior_content_digests[index] = allowed
        projection_changed = False
        projection_evidence: list[dict[str, Any]] = []
        final_verification: list[tuple[Any, str]] = []
        task_projection_lock = (
            vault.open_task_projection_publication() if created_tasks else nullcontext()
        )
        with task_projection_lock:
            with store.organized_note_batch_publication(
                source_key,
                parsed.entries,
                projection_evidence,
            ) as publication:
                already_completed = not publication.publish_required
                publication_roles = set(publication.publication_roles)
                if not publication.publish_required:
                    for path_display, expected_sha256 in publication.verification_evidence:
                        vault.verify_organized_projection(
                            vault.root_path / path_display,
                            expected_sha256,
                        )
                stamp = publication.created_at
                for index, (entry, state) in enumerate(zip(parsed.entries, reservation.entries)):
                    if not publication.publish_required:
                        continue
                    if entry.kind == "memory" and (index, "memory") in publication_roles:
                        path, changed, content_sha256 = vault.write_organized_memory(
                            entry.memory,
                            state.memory_id,
                            stamp,
                        )
                        projection_evidence.append(
                            {
                                "entry_index": index,
                                "role": "memory",
                                "path_display": path.relative_to(vault.root_path).as_posix(),
                                "content_sha256": content_sha256,
                            }
                        )
                        final_verification.append((path, content_sha256))
                        projection_changed = projection_changed or changed
                    elif entry.kind == "goal" and (index, "goal") in publication_roles:
                        goal = {
                            "id": state.goal_id,
                            "revision": 1,
                            "title": entry.goal.title,
                            "purpose": entry.goal.purpose,
                            "horizon": entry.goal.horizon,
                            "status": entry.goal.status,
                            "created_at": stamp,
                            "updated_at": stamp,
                        }
                        path, changed, content_sha256 = vault.write_organized_goal(
                            goal,
                            [],
                            store_identity=store_identity,
                            expected_prior_content_digest=(
                                goal_prior_content_digests.get(index)
                            ),
                        )
                        projection_evidence.append(
                            {
                                "entry_index": index,
                                "role": "goal",
                                "path_display": path.relative_to(vault.root_path).as_posix(),
                                "content_sha256": content_sha256,
                            }
                        )
                        final_verification.append((path, content_sha256))
                        projection_changed = projection_changed or changed
                    elif entry.kind == "decision":
                        if (index, "decision") in publication_roles:
                            decision = {
                                "id": state.decision_id,
                                "title": entry.decision.title,
                                "rationale": entry.decision.rationale,
                                "impact": entry.decision.impact,
                                "status": entry.decision.status,
                                "created_at": stamp,
                                "updated_at": stamp,
                            }
                            path, changed, content_sha256 = vault.write_organized_decision(decision)
                            projection_evidence.append(
                                {
                                    "entry_index": index,
                                    "role": "decision",
                                    "path_display": path.relative_to(vault.root_path).as_posix(),
                                    "content_sha256": content_sha256,
                                }
                            )
                            final_verification.append((path, content_sha256))
                            projection_changed = projection_changed or changed
                        if (index, "decision_memory") in publication_roles:
                            path, changed, content_sha256 = vault.write_organized_memory(
                                entry.decision_memory,
                                state.decision_memory_id,
                                stamp,
                            )
                            projection_evidence.append(
                                {
                                    "entry_index": index,
                                    "role": "decision_memory",
                                    "path_display": path.relative_to(vault.root_path).as_posix(),
                                    "content_sha256": content_sha256,
                                }
                            )
                            final_verification.append((path, content_sha256))
                            projection_changed = projection_changed or changed
                if publication.publish_required and created_tasks:
                    path, _tasks, content_sha256, _source_revision = (
                        vault.write_tasks_with_evidence_under_publication_lock(
                        publication.open_tasks,
                        store_identity=store_identity,
                        )
                    )
                    final_verification.append((path, content_sha256))
                    projection_changed = True
                for path, expected_sha256 in final_verification:
                    vault.verify_organized_projection(path, expected_sha256)
        batch_completed_now = publication.publish_required
        for goal_id in created_goals:
            target = store.ensure_current_goal_projection_job(goal_id)
            if target is None:
                raise RuntimeError("Organized goal projection target is unavailable.")
            outcome = reconcile_goal_projection(store, vault, goal_id, target)
            if outcome.status not in {"completed", "superseded"}:
                raise RuntimeError("Organized goal projection could not be reconciled safely.")

        writes_files = projection_changed
        writes_database = reservation.created or batch_completed_now
        writes_memory = reservation.created and bool(created_memories or created_decisions)
        writes_notes = writes_files
        state_changed = any((writes_files, writes_database, writes_memory, writes_notes))

        lines = ["Organized brain dump:" if state_changed else "Brain dump was already organized:"]
        if created_tasks:
            lines.append(f"- tasks: {', '.join('#' + str(item) for item in created_tasks)}")
        if created_goals:
            lines.append(f"- goals: {', '.join('#' + str(item) for item in created_goals)}")
        if created_memories:
            lines.append(f"- memories: {', '.join('#' + str(item) for item in created_memories)}")
        if created_decisions:
            lines.append(f"- decisions: {', '.join('#' + str(item) for item in created_decisions)}")
        if skipped:
            lines.append("Skipped lines:")
            lines.extend(f"- {line}" for line in skipped[:5])
        lines_processed = parsed.lines_processed
        handoff = _organize_handoff(
            created_tasks=created_tasks,
            created_goals=created_goals,
            created_memories=created_memories,
            created_decisions=created_decisions,
            skipped=skipped,
            overflow=0,
            lines_processed=lines_processed,
            state_changed=state_changed,
            writes_files=writes_files,
            writes_database=writes_database,
            writes_memory=writes_memory,
            writes_notes=writes_notes,
            created_now=reservation.created,
        )
        return ToolResult(
            "organize_note",
            True,
            "\n".join(lines),
            _organize_handoff_metadata(
                "organize_note_handoff",
                handoff,
                tasks=len(created_tasks) if reservation.created else 0,
                goals=len(created_goals) if reservation.created else 0,
                memories=len(created_memories) if reservation.created else 0,
                decisions=len(created_decisions) if reservation.created else 0,
                affected_tasks=len(created_tasks),
                affected_goals=len(created_goals),
                affected_memories=len(created_memories),
                affected_decisions=len(created_decisions),
                skipped=len(skipped),
                overflow=0,
                lines_processed=lines_processed,
                task_ids=created_tasks,
                goal_ids=created_goals,
                memory_ids=created_memories,
                decision_ids=created_decisions,
                writes_files=writes_files,
                writes_database=writes_database,
                writes_memory=writes_memory,
                writes_notes=writes_notes,
                organized_batch_created=reservation.created,
                organized_batch_recovered=not reservation.created and batch_completed_now,
                organized_batch_completed=True,
                converged=already_completed and not reservation.created,
            ),
        )

    return organize_note
