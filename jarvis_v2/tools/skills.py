from __future__ import annotations

import hashlib
import re
from typing import Any

from jarvis_v2.agent.failure_guidance import declare_failure_guidance
from jarvis_v2.agent.types import ApprovalArgumentResolution, ToolResult
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.skill_projection import reconcile_skill_projection
from jarvis_v2.memory.store import MemoryStore, SkillRecord, skill_projection_source_digest


MAX_SKILL_LIMIT = 200
MAX_SKILL_BODY_CHARS = 50000
MAX_SKILL_NAME_CHARS = 160
MAX_SKILL_TRIGGER_CHARS = 500
MAX_SKILL_TAGS_CHARS = 300
MAX_SKILL_QUERY_CHARS = 500
MAX_LINKED_SOURCE_CHARS = 4000
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
SKILL_REFUSAL_RECOVERY_ACTION = (
    "Run `list skills`, correct the reported skill issue, then retry through the normal policy."
)


def _safe_vault_path_display(path: Any, vault: ObsidianVault) -> str:
    try:
        return str(path.relative_to(vault.root_path))
    except (AttributeError, ValueError):
        return _short_metadata(path, 160)


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_SKILL_LIMIT) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _raw_limit_metadata(value: Any, *, limit: int) -> dict[str, Any]:
    if value is None:
        return {"limit": limit}
    if isinstance(value, bool):
        return {"limit": limit, "raw_limit": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {"limit": limit, "raw_limit": _short_metadata(value, 80)}
    return {"limit": limit}


def _short(value: Any, limit: int) -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_metadata(value: Any, limit: int) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit))


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "reads_private_data": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_skills": False,
        "controls_computer": False,
        "queues_approval": False,
        "external_side_effect": False,
        "requires_approval": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


_MISSING_ROW_VALUE = object()


def _row_value(row: Any, key: str, default: Any = _MISSING_ROW_VALUE) -> Any:
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        keys = row.keys()
        if key not in keys:
            return default
        return row[key]
    except Exception:
        return default


def _row_short(row: Any, key: str, limit: int, default: str = "") -> str:
    value = _row_value(row, key)
    if value is _MISSING_ROW_VALUE or value is None:
        return default
    return _short(value, limit)


def _row_positive_int(row: Any, key: str = "id") -> int | None:
    value = _row_value(row, key)
    if isinstance(value, bool) or value is _MISSING_ROW_VALUE or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _skill_display_lines(rows: list[Any], *, body_limit: int) -> tuple[list[str], int]:
    lines: list[str] = []
    unreadable_rows = 0
    for row in rows:
        name = _row_short(row, "name", MAX_SKILL_NAME_CHARS)
        if not name:
            unreadable_rows += 1
            continue
        trigger = _row_short(row, "trigger", MAX_SKILL_TRIGGER_CHARS)
        body = _row_short(row, "body", body_limit)
        review_status = _row_short(row, "review_status", 12, "active")
        label = " [DRAFT]" if review_status == "draft" else ""
        lines.append(f"- {name}{label}: {trigger or body}")
    return lines, unreadable_rows


def _skill_row_status(row: Any) -> tuple[int | None, str, str, str, str, bool]:
    skill_id = _row_positive_int(row)
    name = _row_short(row, "name", MAX_SKILL_NAME_CHARS)
    trigger = _row_short(row, "trigger", MAX_SKILL_TRIGGER_CHARS)
    tags = _row_short(row, "tags", MAX_SKILL_TAGS_CHARS)
    body = _row_short(row, "body", MAX_SKILL_BODY_CHARS)
    readable = skill_id is not None and bool(name)
    return skill_id, name, trigger, tags, body, readable


def _text_sha256(value: str) -> str:
    text = " ".join(value.strip().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _skill_write_metadata(**extra: Any) -> dict[str, Any]:
    metadata = _safe_metadata()
    metadata.update({"writes_files": True, "writes_memory": True, "writes_notes": True, "writes_skills": True})
    metadata.update(extra)
    return metadata


def _skill_refusal_boundaries(*, read_only: bool, requires_approval: bool) -> dict[str, bool]:
    return {
        "read_only": read_only,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_skills": False,
        "writes_database": False,
        "queues_approval": False,
        "requires_approval": requires_approval,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "completes_tasks": False,
        "creates_skill": False,
        "deletes_skill": False,
        "reads_private_data": False,
    }


def _skill_refusal_handoff(
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool,
    requires_approval: bool = False,
    skill_id: int | None = None,
    raw_name: str | None = None,
    raw_query: str | None = None,
    query_chars: int | None = None,
    chars: int | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    handoff: dict[str, Any] = {
        "source": source,
        "ready_for_operator": True,
        "refused": True,
        "mutation": mutation,
        "reason": reason,
        "skill_id": skill_id,
        "changed": [],
        "next_commands": {
            "retry": f"retry {source} with valid skill input",
            "list": "list skills",
            "search": "search skills <query>",
        },
        "boundaries": _skill_refusal_boundaries(read_only=read_only, requires_approval=requires_approval),
    }
    if raw_name is not None:
        handoff["raw_name"] = raw_name
    if raw_query is not None:
        handoff["raw_query"] = raw_query
    if query_chars is not None:
        handoff["query_chars"] = query_chars
    if chars is not None:
        handoff["chars"] = chars
    if max_chars is not None:
        handoff["max_chars"] = max_chars
    return handoff


def _skill_refusal_metadata(
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool,
    requires_approval: bool = False,
    skill_id: int | None = None,
    raw_name: Any = None,
    raw_query: Any = None,
    query_chars: int | None = None,
    chars: int | None = None,
    max_chars: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    safe_raw_name = _short_metadata(raw_name, 80) if raw_name is not None else None
    safe_raw_query = _short_metadata(raw_query, 80) if raw_query is not None else None
    metadata = _safe_metadata(
        reason=reason,
        skill_id=skill_id,
        requires_approval=requires_approval,
        requires_confirmation=False,
        executed_handler=False,
        handler_invoked=False,
        skill_refusal_handoff_ready=True,
        **extra,
    )
    if safe_raw_name is not None:
        metadata["raw_name"] = safe_raw_name
    if safe_raw_query is not None:
        metadata["raw_query"] = safe_raw_query
    if query_chars is not None:
        metadata["query_chars"] = query_chars
    if chars is not None:
        metadata["chars"] = chars
    if max_chars is not None:
        metadata["max_chars"] = max_chars
    metadata["skill_refusal_handoff"] = _skill_refusal_handoff(
        source=source,
        mutation=mutation,
        reason=reason,
        read_only=read_only,
        requires_approval=requires_approval,
        skill_id=skill_id,
        raw_name=safe_raw_name,
        raw_query=safe_raw_query,
        query_chars=query_chars,
        chars=chars,
        max_chars=max_chars,
    )
    return metadata


def _skill_refusal_result(
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
) -> ToolResult:
    public_output = f"{output} {SKILL_REFUSAL_RECOVERY_ACTION}"
    return ToolResult(
        tool_name,
        False,
        public_output,
        declare_failure_guidance(
            metadata,
            output=public_output,
            action=SKILL_REFUSAL_RECOVERY_ACTION,
            commands=("list skills",),
        ),
    )


LINKED_SKILL_TEMPLATES: list[dict[str, str]] = [
    {
        "name": "Karpathy Goal-Driven Execution",
        "trigger": "when writing, reviewing, or refactoring code in Jarvis",
        "body": (
            "1. Surface assumptions before coding, and ask when the request has multiple plausible meanings.\n"
            "2. Choose the simplest implementation that satisfies the current requirement; avoid speculative abstractions.\n"
            "3. Keep edits surgical: touch only the files and lines needed for the task, and clean up only orphans introduced by the change.\n"
            "4. Turn the task into verifiable success criteria before implementation.\n"
            "5. Run the focused verification loop and report remaining uncertainty instead of claiming success early."
        ),
        "tags": "karpathy,coding,verification,simplicity",
        "source": "multica-ai/andrej-karpathy-skills",
        "cue": "andrej-karpathy-skills",
    },
    {
        "name": "Hermes Learning Loop",
        "trigger": "after a complex Jarvis task or repeated workflow",
        "body": (
            "1. Review the recent conversation, tool runs, failures, approvals, and user corrections.\n"
            "2. Extract only reusable behavior, not one-off private details.\n"
            "3. Draft a concise procedure with trigger, steps, safety boundaries, and verification.\n"
            "4. Save it as a Jarvis skill only after it is stable enough to reuse.\n"
            "5. Re-check the skill after future use and revise it when the operator gives feedback."
        ),
        "tags": "hermes,learning,skills",
        "source": "NousResearch/hermes-agent",
        "cue": "hermes-agent",
    },
    {
        "name": "Hermes Conversation Recall",
        "trigger": "when the operator asks what happened before or asks Jarvis to remember prior work",
        "body": (
            "1. Search recent conversations, saved memories, tasks, goals, decisions, and Obsidian notes.\n"
            "2. Separate confirmed facts from guesses and stale context.\n"
            "3. Return a short continuity brief with what was done, current blockers, and safe next actions.\n"
            "4. Do not invent missing context; say when memory is thin.\n"
            "5. Keep risky follow-up actions behind the normal approval queue."
        ),
        "tags": "hermes,conversation,recall,continuity",
        "source": "NousResearch/hermes-agent",
        "cue": "hermes-agent",
    },
    {
        "name": "Hermes Automation Audit",
        "trigger": "before creating or changing scheduled Jarvis work",
        "body": (
            "1. Identify the exact recurring task, schedule, stop condition, and expected output.\n"
            "2. Confirm whether the automation is read-only, local-safe, personal-data, or high-risk.\n"
            "3. Require approval for personal data, external side effects, computer control, shell/code, or destructive actions.\n"
            "4. Add a visible summary so the operator can see what will run later.\n"
            "5. Prefer updating an existing automation over creating a duplicate."
        ),
        "tags": "hermes,automation,scheduler,safety",
        "source": "NousResearch/hermes-agent",
        "cue": "hermes-agent",
    },
    {
        "name": "OpenHuman Memory Tree Ingest",
        "trigger": "when importing documents, notes, emails, chats, or files into Jarvis memory",
        "body": (
            "1. Treat raw data as private until the operator explicitly imports it.\n"
            "2. Chunk source material into small Markdown-sized notes with source, date, and topic.\n"
            "3. Store summaries in Obsidian-compatible locations and index durable facts into SQLite.\n"
            "4. Prefer hierarchical summaries: source note -> topic summary -> current context.\n"
            "5. Keep raw files transient by default and avoid saving secrets or bystander content."
        ),
        "tags": "openhuman,memory-tree,obsidian,ingest",
        "source": "tinyhumansai/openhuman",
        "cue": "openhuman",
    },
    {
        "name": "OpenHuman Connector Boundary",
        "trigger": "before adding calendar, email, docs, messages, search, or OAuth connectors",
        "body": (
            "1. Classify the connector action as read-only, personal-data, external side effect, or high-risk.\n"
            "2. Preview exactly what data would be read or what outside-world effect would happen.\n"
            "3. Avoid account connection, OAuth, personal-data reads, and sending/changing anything during preview.\n"
            "4. Require explicit per-action approval for private reads and all side effects.\n"
            "5. Log the result and show the operator how to revoke or disable the connector later."
        ),
        "tags": "openhuman,connectors,personal-data,approval",
        "source": "tinyhumansai/openhuman",
        "cue": "openhuman",
    },
    {
        "name": "OpenHuman Voice Memo Safety",
        "trigger": "when handling voice memos, transcripts, ASR, or microphone capture",
        "body": (
            "1. Prefer user-supplied audio files or push-to-talk before any wake-word/listener flow.\n"
            "2. Preview transcript text before routing it into planner actions.\n"
            "3. Do not save raw audio or transcripts unless the operator asks after review.\n"
            "4. Treat microphone and transcripts as personal data with visible start/stop state and hard duration limits.\n"
            "5. Keep shell, computer control, personal connectors, reminders, and external effects behind approvals even when spoken."
        ),
        "tags": "openhuman,voice,transcription,safety",
        "source": "tinyhumansai/openhuman",
        "cue": "openhuman",
    },
]


def _linked_skill_review_rows(candidates: list[dict[str, str]]) -> list[dict[str, Any]]:
    candidate_names = ", ".join(candidate["name"] for candidate in candidates) or "none"
    return [
        {
            "item": "source_scope_review",
            "status": "required",
            "evidence": f"candidate skills: {candidate_names}",
            "authorizes_install": False,
            "authorizes_tool_execution": False,
            "authorizes_approval": False,
            "authorizes_risky_action": False,
            "reusable_for_next_link": False,
        },
        {
            "item": "assumption_surface",
            "status": "required",
            "evidence": "external skills must state assumptions, ambiguity, and tradeoffs before behavior changes",
            "authorizes_install": False,
            "authorizes_tool_execution": False,
            "authorizes_approval": False,
            "authorizes_risky_action": False,
            "reusable_for_next_link": False,
        },
        {
            "item": "surgical_change_boundary",
            "status": "required",
            "evidence": "external skill guidance cannot justify unrelated refactors or broad code churn",
            "authorizes_install": False,
            "authorizes_tool_execution": False,
            "authorizes_approval": False,
            "authorizes_risky_action": False,
            "reusable_for_next_link": False,
        },
        {
            "item": "verification_loop",
            "status": "required",
            "evidence": "each imported skill needs focused smoke coverage or an explicit rehearsal path before use",
            "authorizes_install": False,
            "authorizes_tool_execution": False,
            "authorizes_approval": False,
            "authorizes_risky_action": False,
            "reusable_for_next_link": False,
        },
        {
            "item": "approval_boundary",
            "status": "hard_stop",
            "evidence": "skills guide behavior only; shell/code, computer control, personal data, destructive work, and side effects remain gated",
            "authorizes_install": False,
            "authorizes_tool_execution": False,
            "authorizes_approval": False,
            "authorizes_risky_action": False,
            "reusable_for_next_link": False,
        },
    ]


def _linked_skill_review_token_sha256(
    *,
    urls: list[str],
    candidates: list[dict[str, str]],
    review_rows: list[dict[str, Any]],
) -> str:
    return _text_sha256(
        "\n".join(
            [
                "linked_skill_review_token_v1",
                repr(sorted(urls)),
                repr([(candidate["name"], candidate["source"], candidate["trigger"]) for candidate in candidates]),
                repr(
                    [
                        (
                            row.get("item"),
                            row.get("status"),
                            row.get("authorizes_install"),
                            row.get("authorizes_tool_execution"),
                            row.get("authorizes_approval"),
                            row.get("authorizes_risky_action"),
                            row.get("reusable_for_next_link"),
                        )
                        for row in review_rows
                    ]
                ),
            ]
        )
    )


def _extract_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s<>)\\\"]+", text)
    cleaned = []
    seen = set()
    for url in urls:
        normalized = url.rstrip(".,;:")
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def _linked_skill_candidates(text: str) -> tuple[list[str], list[dict[str, str]]]:
    low = text.lower()
    urls = _extract_urls(text)
    candidates = []
    for template in LINKED_SKILL_TEMPLATES:
        cue = template["cue"]
        source = template["source"].lower()
        if not text.strip() or cue in low or source in low or any(cue in url.lower() or source in url.lower() for url in urls):
            candidates.append(template)
    if not candidates and urls:
        candidates.extend(
            [
                {
                    "name": "Linked Document Skill Extraction",
                    "trigger": "when the operator shares an agent/documentation link with reusable behavior",
                    "body": (
                        "1. Extract the link, source title, and the behavior it suggests.\n"
                        "2. Convert only reusable workflows into Jarvis skills.\n"
                        "3. Preserve safety boundaries for personal data, shell/code, computer control, external effects, and destructive actions.\n"
                        "4. Add smoke tests or a rehearsal command before using the skill autonomously.\n"
                        "5. Save the skill locally so it can be searched and revised later."
                    ),
                    "tags": "links,skills,extraction,safety",
                    "source": "generic linked document",
                    "cue": "generic",
                }
            ]
        )
    return urls, candidates


def _skill_terms(text: str) -> list[str]:
    terms = [word.strip(".,?!:;()[]{}<>\"'").lower() for word in text.split()]
    return [word for word in terms if len(word) >= 4]


def make_skill_tools(store: MemoryStore, vault: ObsidianVault):
    def save_skill(args: dict[str, Any]) -> ToolResult:
        raw_name = args.get("name")
        name = _short(raw_name, MAX_SKILL_NAME_CHARS)
        trigger = _short(args.get("trigger"), MAX_SKILL_TRIGGER_CHARS)
        body = str(args.get("body") or "").strip()
        tags = _short(args.get("tags"), MAX_SKILL_TAGS_CHARS)
        if not name:
            return _skill_refusal_result(
                "save_skill",
                "Skill name is required.",
                _skill_refusal_metadata(
                    source="save_skill",
                    mutation="skill_create",
                    reason="missing_name",
                    read_only=False,
                    skill_id=None,
                    raw_name=raw_name,
                ),
            )
        if LOCAL_PATH_RE.search(name):
            return _skill_refusal_result(
                "save_skill",
                "Skill name should describe reusable behavior, not a local file path.",
                _skill_refusal_metadata(
                    source="save_skill",
                    mutation="skill_create",
                    reason="invalid_name",
                    read_only=False,
                    skill_id=None,
                    raw_name=raw_name,
                ),
            )
        if not body:
            return _skill_refusal_result(
                "save_skill",
                "Skill procedure is required.",
                _skill_refusal_metadata(
                    source="save_skill",
                    mutation="skill_create",
                    reason="missing_body",
                    read_only=False,
                    skill_id=None,
                    raw_name=raw_name,
                ),
            )
        if len(body) > MAX_SKILL_BODY_CHARS:
            return _skill_refusal_result(
                "save_skill",
                f"Refusing to save {len(body)} chars as a skill; limit is {MAX_SKILL_BODY_CHARS}.",
                _skill_refusal_metadata(
                    source="save_skill",
                    mutation="skill_create",
                    reason="body_too_large",
                    read_only=False,
                    skill_id=None,
                    raw_name=raw_name,
                    chars=len(body),
                    max_chars=MAX_SKILL_BODY_CHARS,
                ),
            )
        try:
            existing = store.get_skill_by_identity(name)
        except ValueError:
            return _skill_refusal_result(
                "save_skill",
                "That skill name matches multiple saved records. Resolve the duplicate names before saving.",
                _skill_refusal_metadata(
                    source="save_skill",
                    mutation="skill_create",
                    reason="ambiguous_skill_identity",
                    read_only=False,
                    skill_id=None,
                    raw_name=raw_name,
                ),
            )
        promoted_from_draft = (
            existing is not None
            and _row_short(existing, "review_status", 12, "active") == "draft"
        )
        try:
            if promoted_from_draft:
                existing_id = _row_positive_int(existing)
                existing_revision = _row_positive_int(existing, "revision")
                if existing_id is None or existing_revision is None:
                    raise ValueError("skill draft identity is unreadable")
                target = store.prepare_skill_draft_promotion_with_projection_target(
                    SkillRecord(name=name, trigger=trigger, body=body, tags=tags),
                    expected_skill_id=existing_id,
                    expected_revision=existing_revision,
                )
            else:
                target = store.save_skill_with_projection_target(
                    SkillRecord(name=name, trigger=trigger, body=body, tags=tags)
                )
        except ValueError:
            return _skill_refusal_result(
                "save_skill",
                "The skill changed while it was being saved. Review the current skill and retry.",
                _skill_refusal_metadata(
                    source="save_skill",
                    mutation="skill_create",
                    reason="skill_state_changed",
                    read_only=False,
                    skill_id=_row_positive_int(existing),
                    raw_name=raw_name,
                ),
            )
        skill_id = target.skill_id
        projection = reconcile_skill_projection(
            store,
            vault,
            skill_id,
            expected_operation=target.operation,
            expected_revision=target.revision,
            expected_source_digest=target.source_digest,
        )
        if promoted_from_draft:
            current = store.get_skill_by_id(skill_id)
            current_status = _row_short(current, "review_status", 12, "unknown")
            current_revision = _row_positive_int(current, "revision")
            if current_status != "draft" or current_revision != target.revision:
                return ToolResult(
                    "save_skill",
                    False,
                    "The reviewed promotion was superseded by another skill update. Review the current skill before retrying.",
                    _skill_write_metadata(
                        skill_id=skill_id,
                        name_chars=len(name),
                        body_chars=len(body),
                        writes_files=False,
                        writes_notes=False,
                        reason="skill_promotion_superseded",
                        database_mutation_succeeded=True,
                        review_status=current_status,
                        behaviorally_active=current_status == "active",
                        promoted_from_draft=False,
                        skill_projection_operation=projection.operation,
                        skill_projection_status=projection.status,
                        skill_projection_pending=projection.status != "completed",
                    ),
                )
        if projection.status != "completed":
            review_status = "draft" if promoted_from_draft else "active"
            return ToolResult(
                "save_skill",
                False,
                (
                    f"Prepared skill '{name}' in memory, but its Obsidian note projection is still "
                    "pending. "
                    + (
                        "The draft remains behaviorally inactive; retry the reviewed save after projection repair."
                        if promoted_from_draft
                        else "The save is incomplete until projection reconciliation succeeds."
                    )
                ),
                _skill_write_metadata(
                    skill_id=skill_id,
                    name_chars=len(name),
                    body_chars=len(body),
                    writes_files=False,
                    writes_notes=False,
                    reason=(
                        "skill_promotion_projection_pending"
                        if promoted_from_draft
                        else "skill_projection_pending"
                    ),
                    database_mutation_succeeded=True,
                    review_status=review_status,
                    behaviorally_active=review_status == "active",
                    promoted_from_draft=promoted_from_draft,
                    skill_projection_operation=projection.operation,
                    skill_projection_status=projection.status,
                    skill_projection_pending=True,
                ),
            )
        if promoted_from_draft and not store.activate_skill_draft_revision(
            skill_id,
            target.revision,
            target.source_digest,
        ):
            current = store.get_skill_by_id(skill_id)
            current_status = _row_short(current, "review_status", 12, "unknown")
            return ToolResult(
                "save_skill",
                False,
                "The reviewed promotion was superseded by another skill update. Review the current skill before retrying.",
                _skill_write_metadata(
                    skill_id=skill_id,
                    name_chars=len(name),
                    body_chars=len(body),
                    writes_files=True,
                    writes_notes=True,
                    reason="skill_promotion_superseded",
                    database_mutation_succeeded=True,
                    review_status=current_status,
                    behaviorally_active=current_status == "active",
                    promoted_from_draft=False,
                    skill_projection_operation=projection.operation,
                    skill_projection_status=projection.status,
                    skill_projection_pending=False,
                ),
            )
        path = vault.root_path / projection.path_display
        path_display = _safe_vault_path_display(path, vault)
        return ToolResult(
            "save_skill",
            True,
            f"Saved skill '{name}'.\nSaved note: {path_display}",
            _skill_write_metadata(
                skill_id=skill_id,
                path=str(path),
                path_display=path_display,
                name_chars=len(name),
                body_chars=len(body),
                review_status="active",
                behaviorally_active=True,
                promoted_from_draft=promoted_from_draft,
            ),
        )

    def list_skills(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 25)
        rows = store.list_skills(limit)
        if not rows:
            return ToolResult(
                "list_skills",
                True,
                "No skills saved yet. Count: 0.",
                _safe_metadata(count=0, total_skills=0, **_raw_limit_metadata(args.get("limit"), limit=limit)),
            )
        lines, unreadable_rows = _skill_display_lines(rows, body_limit=90)
        if unreadable_rows:
            lines.append(
                f"- {unreadable_rows} skill row(s) could not be read safely; run `list skills` again after memory repair."
            )
        return ToolResult(
            "list_skills",
            True,
            f"Saved skills:\nCount: {len(rows)}\n" + "\n".join(lines),
            _safe_metadata(
                count=len(rows),
                total_skills=len(rows),
                readable_skill_rows=len(rows) - unreadable_rows,
                unreadable_skill_rows=unreadable_rows,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def search_skills(args: dict[str, Any]) -> ToolResult:
        query = _short(args.get("query"), MAX_SKILL_QUERY_CHARS)
        limit = _bounded_int(args.get("limit"), 20)
        if not query:
            return _skill_refusal_result(
                "search_skills",
                "Skill search query is empty.",
                _skill_refusal_metadata(
                    source="search_skills",
                    mutation="skill_search",
                    reason="missing_query",
                    read_only=True,
                    query_chars=0,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        if LOCAL_PATH_RE.search(query):
            return _skill_refusal_result(
                "search_skills",
                "Skill search query should describe behavior, not a local file path.",
                _skill_refusal_metadata(
                    source="search_skills",
                    mutation="skill_search",
                    reason="invalid_query",
                    read_only=True,
                    raw_query=args.get("query"),
                    query_chars=len(query),
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                ),
            )
        rows = store.search_skills(query, limit=limit)
        if not rows:
            return ToolResult(
                "search_skills",
                True,
                f"No skills found for '{query}'.",
                _safe_metadata(count=0, query_chars=len(query), **_raw_limit_metadata(args.get("limit"), limit=limit)),
            )
        lines, unreadable_rows = _skill_display_lines(rows, body_limit=120)
        if unreadable_rows:
            lines.append(
                f"- {unreadable_rows} skill row(s) could not be read safely; run `search skills {query}` again after memory repair."
            )
        return ToolResult(
            "search_skills",
            True,
            "\n".join(lines),
            _safe_metadata(
                count=len(rows),
                readable_skill_rows=len(rows) - unreadable_rows,
                unreadable_skill_rows=unreadable_rows,
                query_chars=len(query),
                **_raw_limit_metadata(args.get("limit"), limit=limit),
            ),
        )

    def get_skill(args: dict[str, Any]) -> ToolResult:
        raw_name = args.get("name")
        name = _short(raw_name, MAX_SKILL_NAME_CHARS)
        if not name:
            return _skill_refusal_result(
                "get_skill",
                "Skill name is empty.",
                _skill_refusal_metadata(
                    source="get_skill",
                    mutation="skill_read",
                    reason="missing_name",
                    read_only=True,
                    skill_id=None,
                    raw_name=raw_name,
                ),
            )
        if LOCAL_PATH_RE.search(name):
            return _skill_refusal_result(
                "get_skill",
                "Skill name should describe reusable behavior, not a local file path.",
                _skill_refusal_metadata(
                    source="get_skill",
                    mutation="skill_read",
                    reason="invalid_name",
                    read_only=True,
                    skill_id=None,
                    raw_name=raw_name,
                ),
            )
        row = store.get_skill(name)
        if not row:
            return ToolResult(
                "get_skill",
                True,
                f"No skill found for '{name}'.",
                _safe_metadata(skill_id=None),
            )
        skill_id, row_name, trigger, tags, body, readable = _skill_row_status(row)
        if not readable:
            return ToolResult(
                "get_skill",
                True,
                f"Skill row for '{name}' could not be read safely. Run `list skills` to review memory state.",
                _safe_metadata(skill_id=None, skill_status="unreadable", readable_skill_row=False, unreadable_skill_row=True),
            )
        review_status = _row_short(row, "review_status", 12, "active")
        status_line = "Review status: DRAFT (not available to behavioral routing)\n" if review_status == "draft" else ""
        output = (
            f"# {row_name}\n\n"
            f"{status_line}"
            f"Trigger: {trigger}\n"
            f"Tags: {tags}\n\n"
            f"{body}"
        )
        return ToolResult(
            "get_skill",
            True,
            output,
            _safe_metadata(
                skill_id=skill_id,
                skill_status="found",
                review_status=review_status,
                behaviorally_active=review_status == "active",
                readable_skill_row=True,
                unreadable_skill_row=False,
            ),
        )

    def resolve_delete_skill_approval(
        args: dict[str, Any],
    ) -> ApprovalArgumentResolution | ToolResult:
        raw_name = args.get("name")
        name = _short(raw_name, MAX_SKILL_NAME_CHARS)
        if not name:
            return _skill_refusal_result(
                "delete_skill",
                "Skill name is empty.",
                _skill_refusal_metadata(
                    source="delete_skill",
                    mutation="skill_delete",
                    reason="missing_name",
                    read_only=False,
                    requires_approval=True,
                    raw_name=raw_name,
                    destructive=True,
                ),
            )
        if LOCAL_PATH_RE.search(name):
            return _skill_refusal_result(
                "delete_skill",
                "Skill name should describe reusable behavior, not a local file path.",
                _skill_refusal_metadata(
                    source="delete_skill",
                    mutation="skill_delete",
                    reason="invalid_name",
                    read_only=False,
                    requires_approval=True,
                    raw_name=raw_name,
                    destructive=True,
                ),
            )
        resolution = store.resolve_skill_delete_target(name)
        if resolution.status == "not_found":
            return _skill_refusal_result(
                "delete_skill",
                f"No skill found for '{name}', so no approval was queued.",
                _skill_refusal_metadata(
                    source="delete_skill",
                    mutation="skill_delete",
                    reason="not_found",
                    read_only=False,
                    requires_approval=True,
                    raw_name=raw_name,
                    destructive=True,
                    approval_argument_resolution_status="not_found",
                ),
            )
        if resolution.status != "resolved":
            choices = [
                _short_metadata(candidate.name, MAX_SKILL_NAME_CHARS)
                for candidate in resolution.candidates
            ]
            names = ", ".join(choices)
            return _skill_refusal_result(
                "delete_skill",
                (
                    f"Multiple skills match '{name}' ({names}). Use the exact skill name; "
                    "no approval was queued."
                ),
                _skill_refusal_metadata(
                    source="delete_skill",
                    mutation="skill_delete",
                    reason="ambiguous_target",
                    read_only=False,
                    requires_approval=True,
                    raw_name=raw_name,
                    destructive=True,
                    approval_argument_resolution_status="ambiguous",
                    approval_argument_candidates=choices,
                    approval_argument_candidate_count=len(choices),
                ),
            )
        if (
            resolution.skill_id is None
            or resolution.revision is None
            or not resolution.name
        ):
            return _skill_refusal_result(
                "delete_skill",
                "The skill target could not be bound safely, so no approval was queued.",
                _skill_refusal_metadata(
                    source="delete_skill",
                    mutation="skill_delete",
                    reason="unreadable_target",
                    read_only=False,
                    requires_approval=True,
                    raw_name=raw_name,
                    destructive=True,
                    approval_argument_resolution_status="unreadable_target",
                ),
            )
        return ApprovalArgumentResolution(
            args={
                "name": name,
                "target_skill_id": resolution.skill_id,
                "target_skill_revision": resolution.revision,
                "target_skill_name": resolution.name,
            },
            metadata={
                "approval_argument_resolution_status": "resolved",
                "target_skill_id": resolution.skill_id,
                "target_skill_revision": resolution.revision,
                "target_skill_name": _short_metadata(
                    resolution.name, MAX_SKILL_NAME_CHARS
                ),
            },
        )

    def delete_skill(args: dict[str, Any]) -> ToolResult:
        raw_name = args.get("name")
        name = _short(raw_name, MAX_SKILL_NAME_CHARS)
        if not name:
            return _skill_refusal_result(
                "delete_skill",
                "Skill name is empty.",
                _skill_refusal_metadata(
                    source="delete_skill",
                    mutation="skill_delete",
                    reason="missing_name",
                    read_only=False,
                    requires_approval=True,
                    raw_name=raw_name,
                    destructive=True,
                ),
            )
        if LOCAL_PATH_RE.search(name):
            return _skill_refusal_result(
                "delete_skill",
                "Skill name should describe reusable behavior, not a local file path.",
                _skill_refusal_metadata(
                    source="delete_skill",
                    mutation="skill_delete",
                    reason="invalid_name",
                    read_only=False,
                    requires_approval=True,
                    raw_name=raw_name,
                    destructive=True,
                ),
            )
        target_skill_id = args.get("target_skill_id")
        target_skill_revision = args.get("target_skill_revision")
        target_skill_name = args.get("target_skill_name")
        if (
            type(target_skill_id) is not int
            or target_skill_id < 1
            or type(target_skill_revision) is not int
            or target_skill_revision < 1
            or type(target_skill_name) is not str
            or not target_skill_name
        ):
            return _skill_refusal_result(
                "delete_skill",
                (
                    "This delete request is not bound to an immutable skill version. Nothing was "
                    "deleted; issue a fresh delete request for a new approval."
                ),
                _skill_refusal_metadata(
                    source="delete_skill",
                    mutation="skill_delete",
                    reason="stale_skill_binding",
                    read_only=False,
                    requires_approval=True,
                    raw_name=raw_name,
                    destructive=True,
                    approval_rerun_blocked=True,
                    approval_rerun_block_reason="skill_target_not_bound",
                ),
            )
        deleted = store.delete_skill_exact(
            target_skill_id,
            target_skill_revision,
            target_skill_name,
        )
        deleted_skill = deleted.deleted_skill if deleted.status == "deleted" else None
        if deleted_skill is not None:
            canonical_name = str(deleted_skill["name"])
            delete_source_digest = skill_projection_source_digest(
                target_skill_id,
                target_skill_revision,
                str(deleted_skill["name"]),
                str(deleted_skill["trigger"]),
                str(deleted_skill["body"]),
                str(deleted_skill["tags"]),
                "delete",
            )
            projection = reconcile_skill_projection(
                store,
                vault,
                target_skill_id,
                expected_operation="delete",
                expected_revision=target_skill_revision,
                expected_source_digest=delete_source_digest,
            )
            if projection.status != "completed":
                return ToolResult(
                    "delete_skill",
                    False,
                    (
                        f"Deleted skill '{canonical_name}' from memory, but its Obsidian note "
                        "projection is still pending. The delete is incomplete until projection "
                        "reconciliation succeeds."
                    ),
                    _safe_metadata(
                        writes_skills=True,
                        writes_files=False,
                        writes_notes=False,
                        destructive=True,
                        requires_approval=True,
                        skill_id=target_skill_id,
                        deleted_revision=target_skill_revision,
                        target_binding_status="matched",
                        reason="skill_projection_pending",
                        database_mutation_succeeded=True,
                        skill_projection_operation=projection.operation,
                        skill_projection_status=projection.status,
                        skill_projection_pending=True,
                    ),
                )
            if projection.completion_status == "ownership_mismatch":
                return ToolResult(
                    "delete_skill",
                    False,
                    (
                        f"Deleted skill '{canonical_name}' from memory, but preserved an "
                        "unverified note at its expected Obsidian path. No foreign file was removed."
                    ),
                    _safe_metadata(
                        writes_skills=True,
                        writes_files=False,
                        writes_notes=False,
                        destructive=True,
                        requires_approval=True,
                        skill_id=target_skill_id,
                        deleted_revision=target_skill_revision,
                        target_binding_status="matched",
                        reason="skill_projection_ownership_mismatch",
                        database_mutation_succeeded=True,
                        skill_projection_operation=projection.operation,
                        skill_projection_status=projection.status,
                        skill_projection_completion_status=projection.completion_status,
                        skill_projection_pending=False,
                    ),
                )
            deleted_note = bool(projection.path_display)
            return ToolResult(
                "delete_skill",
                True,
                f"Deleted skill '{canonical_name}'.",
                _safe_metadata(
                    writes_skills=True,
                    writes_files=bool(deleted_note),
                    writes_notes=bool(deleted_note),
                    destructive=True,
                    requires_approval=True,
                    skill_id=target_skill_id,
                    deleted_revision=target_skill_revision,
                    target_binding_status="matched",
                ),
            )
        if deleted.status in {"stale", "not_found"}:
            return _skill_refusal_result(
                "delete_skill",
                (
                    "The approved skill target changed or no longer exists. Nothing was deleted; "
                    "inspect the skill list and issue a fresh delete request if still intended."
                ),
                _skill_refusal_metadata(
                    source="delete_skill",
                    mutation="skill_delete",
                    reason="stale_skill_binding",
                    read_only=False,
                    requires_approval=True,
                    skill_id=target_skill_id,
                    raw_name=raw_name,
                    destructive=True,
                    target_binding_status=deleted.status,
                    approval_rerun_blocked=True,
                    approval_rerun_block_reason="skill_target_changed",
                ),
            )
        return _skill_refusal_result(
            "delete_skill",
            f"No skill found for '{name}'.",
            _skill_refusal_metadata(
                source="delete_skill",
                mutation="skill_delete",
                reason="not_found",
                read_only=False,
                requires_approval=True,
                raw_name=raw_name,
                destructive=True,
            ),
        )

    def extract_linked_skills(args: dict[str, Any]) -> ToolResult:
        document = _short(args.get("document") or args.get("links") or args.get("source"), MAX_LINKED_SOURCE_CHARS)
        urls, candidates = _linked_skill_candidates(document)
        if not candidates:
            return ToolResult(
                "extract_linked_skills",
                True,
                "No link-inspired skill candidates found. Include links or mention Hermes/OpenHuman.",
                _safe_metadata(links=urls, candidates=0, source_chars=len(document)),
            )
        review_rows = _linked_skill_review_rows(candidates)
        review_token = _linked_skill_review_token_sha256(urls=urls, candidates=candidates, review_rows=review_rows)
        lines = [
            "Linked skill extraction:",
            "This is read-only. It extracts reusable Jarvis skill candidates from linked agent documents without fetching pages, saving skills, changing memory, or queuing approvals.",
            "",
            "Links found:",
        ]
        if urls:
            lines.extend(f"- {url}" for url in urls)
        else:
            lines.append("- none supplied; using known Hermes/OpenHuman cues from the request")
        lines.extend(["", "Skill candidates:"])
        for index, candidate in enumerate(candidates, start=1):
            lines.extend(
                [
                    f"{index}. {candidate['name']} ({candidate['source']})",
                    f"   Trigger: {candidate['trigger']}",
                    f"   Tags: {candidate['tags']}",
                ]
            )
        lines.extend(
            [
                "",
                "Linked skill review boundary:",
                f"- review token sha256: {review_token}",
                "- review token authorizes install: no",
                "- review token authorizes tool execution: no",
                "- review token authorizes risky action: no",
                "- next linked skill intake requires a fresh review token: yes",
                f"- boundary row count: {len(review_rows)}",
                *[f"- {row['item']}: {row['status']}; {row['evidence']}" for row in review_rows],
                "",
                "Install path:",
                "- Run `install linked skills from <links or source text>` to save these as searchable Jarvis skills.",
                "- Installed skills stay local in SQLite and the Obsidian skill vault.",
            ]
        )
        return ToolResult(
            "extract_linked_skills",
            True,
            "\n".join(lines),
            _safe_metadata(
                links=urls,
                candidates=len(candidates),
                source_chars=len(document),
                candidate_names=[candidate["name"] for candidate in candidates],
                linked_skill_review_rows=review_rows,
                linked_skill_review_row_count=len(review_rows),
                linked_skill_review_token_sha256=review_token,
                linked_skill_review_token_present=True,
                linked_skill_review_token_authorizes_install=False,
                linked_skill_review_token_authorizes_tool_execution=False,
                linked_skill_review_token_authorizes_approval=False,
                linked_skill_review_token_authorizes_risky_action=False,
                linked_skill_review_token_reusable_for_next_link=False,
                next_linked_skill_intake_requires_fresh_review_token=True,
            ),
        )

    def install_linked_skills(args: dict[str, Any]) -> ToolResult:
        document = _short(args.get("document") or args.get("links") or args.get("source"), MAX_LINKED_SOURCE_CHARS)
        limit = _bounded_int(args.get("limit"), 6, high=12)
        urls, candidates = _linked_skill_candidates(document)
        selected = candidates[:limit]
        if not selected:
            return ToolResult("install_linked_skills", False, "No link-inspired skill candidates found to install.", _safe_metadata(reason="no_candidates", links=urls, **_raw_limit_metadata(args.get("limit"), limit=limit)))
        review_rows = _linked_skill_review_rows(selected)
        review_token = _linked_skill_review_token_sha256(urls=urls, candidates=selected, review_rows=review_rows)
        installed = []
        pending = []
        blocked = []
        for candidate in selected:
            body = f"Source inspiration: {candidate['source']}\n\n{candidate['body']}"
            try:
                existing_candidate = store.get_skill_by_identity(candidate["name"])
            except ValueError:
                blocked.append(
                    {
                        "name": candidate["name"],
                        "skill_id": None,
                        "source": candidate["source"],
                        "skill_projection_operation": "publish",
                        "skill_projection_status": "ambiguous_identity",
                    }
                )
                continue
            if (
                existing_candidate is not None
                and _row_short(existing_candidate, "review_status", 12, "active") == "draft"
            ):
                blocked.append(
                    {
                        "name": candidate["name"],
                        "skill_id": _row_positive_int(existing_candidate),
                        "source": candidate["source"],
                        "skill_projection_operation": "publish",
                        "skill_projection_status": "draft_requires_explicit_promotion",
                    }
                )
                continue
            try:
                target = store.save_skill_with_projection_target(
                    SkillRecord(
                        name=candidate["name"],
                        trigger=candidate["trigger"],
                        body=body,
                        tags=candidate["tags"],
                    ),
                    allow_draft_activation=False,
                )
            except ValueError:
                blocked.append(
                    {
                        "name": candidate["name"],
                        "skill_id": None,
                        "source": candidate["source"],
                        "skill_projection_operation": "publish",
                        "skill_projection_status": "draft_or_identity_conflict",
                    }
                )
                continue
            skill_id = target.skill_id
            projection = reconcile_skill_projection(
                store,
                vault,
                skill_id,
                expected_operation=target.operation,
                expected_revision=target.revision,
                expected_source_digest=target.source_digest,
            )
            if projection.status != "completed":
                pending.append(
                    {
                        "name": candidate["name"],
                        "skill_id": skill_id,
                        "source": candidate["source"],
                        "skill_projection_operation": projection.operation,
                        "skill_projection_status": projection.status,
                    }
                )
                continue
            path = vault.root_path / projection.path_display
            path_display = _safe_vault_path_display(path, vault)
            installed.append(
                {
                    "name": candidate["name"],
                    "skill_id": skill_id,
                    "path": str(path),
                    "path_display": path_display,
                    "source": candidate["source"],
                }
            )
        if pending or blocked:
            lines = [f"Installed {len(installed)} of {len(selected)} linked skill candidate(s)."]
            if pending:
                lines.extend(
                    [
                        (
                            f"{len(pending)} Obsidian note projection(s) are still pending. "
                            "The install is incomplete until projection reconciliation succeeds."
                        ),
                        "",
                        "Pending notes:",
                        *[f"- {item['name']} ({item['source']})" for item in pending],
                    ]
                )
            if blocked:
                lines.extend(
                    [
                        "",
                        "Preserved for review:",
                        *[
                            f"- {item['name']} ({item['skill_projection_status']})"
                            for item in blocked
                        ],
                        "Promote an existing draft only through an explicit reviewed `save skill` command.",
                    ]
                )
            if installed:
                lines.extend(
                    [
                        "",
                        "Saved notes:",
                        *[f"- {item['path_display']}" for item in installed],
                    ]
                )
            return ToolResult(
                "install_linked_skills",
                False,
                "\n".join(lines),
                _skill_write_metadata(
                    links=urls,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                    installed=len(installed),
                    skills=installed,
                    linked_skill_review_rows=review_rows,
                    linked_skill_review_row_count=len(review_rows),
                    linked_skill_review_token_sha256=review_token,
                    linked_skill_review_token_present=True,
                    linked_skill_review_token_authorizes_install=False,
                    linked_skill_review_token_authorizes_tool_execution=False,
                    linked_skill_review_token_authorizes_approval=False,
                    linked_skill_review_token_authorizes_risky_action=False,
                    linked_skill_review_token_reusable_for_next_link=False,
                    next_linked_skill_intake_requires_fresh_review_token=True,
                    writes_files=bool(installed),
                    writes_memory=bool(installed or pending),
                    writes_notes=bool(installed),
                    writes_skills=bool(installed or pending),
                    reason=(
                        "skill_projection_pending"
                        if pending
                        else "linked_skill_review_required"
                    ),
                    database_mutation_succeeded=bool(installed or pending),
                    skill_projection_pending=bool(pending),
                    projected=len(installed),
                    pending_skills=pending,
                    pending_skill_count=len(pending),
                    blocked_skills=blocked,
                    blocked_skill_count=len(blocked),
                ),
            )
        lines = [
            "Installed linked skills:",
            *[f"- {item['name']} ({item['source']})" for item in installed],
            "",
            "Saved notes:",
            *[f"- {item['path_display']}" for item in installed],
            "",
            "Safety boundary:",
            "- This only saved local Jarvis skills; it did not fetch links, connect accounts, read personal data, execute tools, control the computer, or queue approvals.",
            "- These skills guide future behavior, but risky actions still go through ToolRegistry, PermissionPolicy, approvals, and audit logging.",
            "- The linked skill review token is proof-only; it does not authorize install reuse, tool execution, approvals, or risky actions.",
            f"- linked skill review token sha256: {review_token}",
        ]
        return ToolResult(
            "install_linked_skills",
            True,
            "\n".join(lines),
            _skill_write_metadata(
                links=urls,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
                installed=len(installed),
                skills=installed,
                linked_skill_review_rows=review_rows,
                linked_skill_review_row_count=len(review_rows),
                linked_skill_review_token_sha256=review_token,
                linked_skill_review_token_present=True,
                linked_skill_review_token_authorizes_install=False,
                linked_skill_review_token_authorizes_tool_execution=False,
                linked_skill_review_token_authorizes_approval=False,
                linked_skill_review_token_authorizes_risky_action=False,
                linked_skill_review_token_reusable_for_next_link=False,
                next_linked_skill_intake_requires_fresh_review_token=True,
            ),
        )

    def skill_match_preview(args: dict[str, Any]) -> ToolResult:
        request = _short(args.get("request") or args.get("prompt"), MAX_SKILL_QUERY_CHARS)
        if not request:
            return ToolResult(
                "skill_match_preview",
                False,
                "Request is required. Try `skill match preview: import these notes into memory`.",
                _safe_metadata(reason="missing_request"),
            )
        if LOCAL_PATH_RE.search(request):
            return ToolResult(
                "skill_match_preview",
                False,
                "Skill match request should describe behavior, not a local file path.",
                _safe_metadata(reason="invalid_request", raw_request=_short_metadata(args.get("request") or args.get("prompt"), 80)),
            )
        terms = _skill_terms(request)
        rows = []
        seen_ids = set()
        if terms:
            query = " ".join(terms[:8])
            for row in store.search_active_skills(query, limit=8):
                row_id = _row_positive_int(row)
                if row_id is not None and row_id not in seen_ids:
                    seen_ids.add(row_id)
                    rows.append(row)
            for term in terms[:8]:
                for row in store.search_active_skills(term, limit=4):
                    row_id = _row_positive_int(row)
                    if row_id is None or row_id in seen_ids:
                        continue
                    seen_ids.add(row_id)
                    rows.append(row)
                    if len(rows) >= 8:
                        break
                if len(rows) >= 8:
                    break
        if not rows:
            rows = store.list_active_skills(limit=5)

        lines = [
            "Jarvis skill match preview:",
            "This is read-only. It searches saved skills for a request without executing tools, writing memory, changing skills, or queuing approvals.",
            "",
            "Request:",
            f"- {request}",
            "",
            "Candidate skills:",
        ]
        if rows:
            unreadable_rows = 0
            for index, row in enumerate(rows[:8], start=1):
                name = _row_short(row, "name", MAX_SKILL_NAME_CHARS)
                if not name:
                    unreadable_rows += 1
                    continue
                body = _row_short(row, "body", 220).replace("\n", " ")
                lines.extend(
                    [
                        f"{index}. {name}",
                        f"   Trigger: {_row_short(row, 'trigger', MAX_SKILL_TRIGGER_CHARS) or '(none)'}",
                        f"   Tags: {_row_short(row, 'tags', MAX_SKILL_TAGS_CHARS) or '(none)'}",
                        f"   Procedure preview: {body[:220]}",
                    ]
                )
            if unreadable_rows:
                lines.append(f"- {unreadable_rows} skill row(s) could not be read safely during preview.")
        else:
            lines.append("- No saved skills yet. Use `save skill ...` or `install linked skills from ...` first.")
        lines.extend(
            [
                "",
                "Use boundary:",
                "- A matching skill can guide the next answer or plan, but it does not bypass planner routing, ToolRegistry, PermissionPolicy, approval queue, or audit logging.",
                "- If the request touches personal data, shell/code, computer control, reminders, external services, or destructive changes, Jarvis still asks first.",
            ]
        )
        return ToolResult(
            "skill_match_preview",
            True,
            "\n".join(lines),
            _safe_metadata(
                request=request,
                terms=terms[:8],
                matches=len(rows),
                match_names=[_row_short(row, "name", MAX_SKILL_NAME_CHARS) for row in rows[:8] if _row_short(row, "name", MAX_SKILL_NAME_CHARS)],
            ),
        )

    return (
        save_skill,
        list_skills,
        search_skills,
        get_skill,
        delete_skill,
        extract_linked_skills,
        install_linked_skills,
        skill_match_preview,
        resolve_delete_skill_approval,
    )
