from __future__ import annotations

import re
import sqlite3
from collections import Counter
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.store import MemoryStore


WORD_RE = re.compile(r"[a-z0-9][a-z0-9_'-]*", re.IGNORECASE)
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
SECRET_VALUE_RE = re.compile(
    r"(?:"
    r"\bsk_(?:live|test)_[A-Za-z0-9_-]+"
    r"|\bgh[pousr]_[A-Za-z0-9_-]+"
    r"|\bxox[baprs]-[A-Za-z0-9-]+"
    r"|\bAIza[A-Za-z0-9_-]{16,}"
    r"|\b\d{6,}:[A-Za-z0-9_-]{20,}"
    r")",
    re.IGNORECASE,
)
READ_ONLY_METADATA = {
    "calls_model": False,
    "writes_memory": False,
    "writes_files": False,
    "writes_notes": False,
    "executes_tools": False,
    "executes_side_effect": False,
    "external_calls": False,
    "controls_computer": False,
    "queues_approval": False,
    "approves_request": False,
    "dismisses_request": False,
    "reads_private_data": False,
    "reads_personal_data": False,
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
    "speaks": False,
    "completes_tasks": False,
}


def _short_raw(value: Any, *, limit: int = 80) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_metadata(value: Any, *, limit: int = 80) -> str:
    text = LOCAL_PATH_RE.sub("<local-path>", _short_raw(value, limit=limit))
    return SECRET_VALUE_RE.sub("<private-value>", text)


def _safe_display(value: Any, *, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if limit is not None and len(text) > limit:
        text = text[: max(0, limit - 3)].rstrip() + "..."
    return LOCAL_PATH_RE.sub("<local-path>", text)


def _first_present(args: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in args and args[name] is not None:
            return args[name]
    return None


def _brain_boundaries() -> dict[str, bool]:
    return {
        "read_only": True,
        "calls_model": False,
        "writes_memory": False,
        "writes_files": False,
        "writes_notes": False,
        "executes_tools": False,
        "executes_side_effect": False,
        "external_calls": False,
        "controls_computer": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "speaks": False,
        "completes_tasks": False,
    }


def _brain_handoff(
    *,
    source: str,
    status: str,
    reason: str | None = None,
    query: str | None = None,
    question: str | None = None,
    memory_id: int | None = None,
    count: int | None = None,
    citations: list[str] | None = None,
    gaps: list[str] | None = None,
    graph_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    handoff: dict[str, Any] = {
        "source": source,
        "status": status,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "next_commands": [
            "brain search <query>",
            "brain think <question>",
            "brain neighbors <memory_id>",
            "brain graph",
        ],
        "boundaries": _brain_boundaries(),
    }
    if reason:
        handoff["reason"] = reason
        handoff["refused"] = status == "refused"
    else:
        handoff["refused"] = False
    if query is not None:
        handoff["query"] = _safe_display(query, limit=160)
    if question is not None:
        handoff["question"] = _safe_display(question, limit=200)
    if memory_id is not None:
        handoff["memory_id"] = memory_id
    if count is not None:
        handoff["count"] = count
    if citations is not None:
        handoff["citations"] = [_safe_display(citation, limit=220) for citation in citations]
    if gaps is not None:
        handoff["gaps"] = [_safe_display(gap, limit=220) for gap in gaps]
    if graph_counts is not None:
        handoff["graph_counts"] = dict(graph_counts)
    return handoff


def _brain_metadata(*, source: str, status: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    query = extra.get("query")
    question = extra.get("question")
    memory_id = extra.get("memory_id") if isinstance(extra.get("memory_id"), int) else None
    count = extra.get("count") if isinstance(extra.get("count"), int) else None
    citations = extra.get("citations") if isinstance(extra.get("citations"), list) else None
    gaps = extra.get("gaps") if isinstance(extra.get("gaps"), list) else None
    graph_counts = {
        key: int(extra[key])
        for key in ("memory_nodes", "people_nodes", "category_nodes", "edges")
        if isinstance(extra.get(key), int)
    } or None
    handoff = _brain_handoff(
        source=source,
        status=status,
        reason=reason,
        query=query if isinstance(query, str) else None,
        question=question if isinstance(question, str) else None,
        memory_id=memory_id,
        count=count,
        citations=citations,
        gaps=gaps,
        graph_counts=graph_counts,
    )
    return {
        **READ_ONLY_METADATA,
        **extra,
        "brain_handoff": handoff,
        "brain_handoff_ready": True,
        "brain_ready_for_operator": handoff["ready_for_operator"],
        "brain_state_changed": handoff["state_changed"],
        "brain_changed": handoff["changed"],
        "brain_content_in_handoff": handoff["content_in_handoff"],
        "brain_next_commands": handoff["next_commands"],
        "brain_next_command_count": len(handoff["next_commands"]),
        "brain_boundaries": handoff["boundaries"],
        "brain_status": status,
        "refusal_reason": reason,
    }


def _brain_input_failure(
    tool_name: str,
    message: str,
    metadata: dict[str, Any],
) -> ToolResult:
    output = f"{message} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
    return ToolResult(
        tool_name,
        False,
        output,
        declare_retryable_local_read_failure(
            metadata,
            output=output,
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        ),
    )


def make_brain_tools(store: MemoryStore):
    def _limit(args: dict[str, Any], default: int = 8, maximum: int = 25) -> int:
        if isinstance(args.get("limit", default), bool):
            return default
        try:
            value = int(args.get("limit", default))
        except (TypeError, ValueError):
            value = default
        return max(1, min(maximum, value))

    def _terms(text: str) -> list[str]:
        return [
            term.lower()
            for term in WORD_RE.findall(text)
            if len(term) > 2 and term.lower() not in {"the", "and", "for", "that", "with", "what", "should"}
        ]

    def _excerpt(row: Any, size: int = 220) -> str:
        return _safe_display(row["body"], limit=size)

    def _citation(row: Any) -> str:
        return f"[M{row['id']}] {_safe_display(row['category'], limit=80)}/{_safe_display(row['title'], limit=160)}"

    def _fallback_search(query: str, limit: int) -> list[Any]:
        terms = _terms(query)
        if not terms:
            return store.recent_memories(limit)
        scored: list[tuple[int, Any]] = []
        for row in store.list_memories(limit=300):
            haystack = f"{row['category']} {row['title']} {row['body']}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                scored.append((score, row))
        scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        return [row for _, row in scored[:limit]]

    def _search_rows(query: str, limit: int) -> list[Any]:
        try:
            rows = store.search_memories(query, limit)
        except sqlite3.Error:
            rows = []
        if rows:
            return rows
        return _fallback_search(query, limit)

    def brain_search(args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip()
        if not query:
            return _brain_input_failure(
                "brain_search",
                "Brain search query is empty.",
                _brain_metadata(source="brain_search", status="refused", reason="missing_query", query="", count=0, citations=[]),
            )
        if LOCAL_PATH_RE.search(query):
            return _brain_input_failure(
                "brain_search",
                "Brain search query should describe memory, not a local file path.",
                _brain_metadata(source="brain_search", status="refused", reason="local_path_query", query="<local-path>", raw_query=_short_metadata(query), count=0, citations=[]),
            )
        limit = _limit(args)
        rows = _search_rows(query, limit)
        if not rows:
            return ToolResult(
                "brain_search",
                True,
                f"Jarvis GBrain-style search (read-only): no local memories matched '{_safe_display(query, limit=160)}'.",
                _brain_metadata(source="brain_search", status="empty", query=query, count=0, citations=[]),
            )
        lines = [f"Jarvis GBrain-style search (read-only): {len(rows)} local memory result(s) for '{_safe_display(query, limit=160)}'."]
        for index, row in enumerate(rows, start=1):
            lines.append(f"{index}. {_citation(row)}")
            excerpt = _excerpt(row)
            if excerpt:
                lines.append(f"   {excerpt}")
        citations = [_citation(row) for row in rows]
        return ToolResult(
            "brain_search",
            True,
            "\n".join(lines),
            _brain_metadata(source="brain_search", status="ok", query=query, count=len(rows), citations=citations),
        )

    def brain_think(args: dict[str, Any]) -> ToolResult:
        question = str(args.get("question") or args.get("query") or "").strip()
        if not question:
            return _brain_input_failure(
                "brain_think",
                "Brain think question is empty.",
                _brain_metadata(source="brain_think", status="refused", reason="missing_question", question="", count=0, citations=[], gaps=[]),
            )
        if LOCAL_PATH_RE.search(question):
            return _brain_input_failure(
                "brain_think",
                "Brain think question should describe memory, not a local file path.",
                _brain_metadata(source="brain_think", status="refused", reason="local_path_question", question="<local-path>", raw_question=_short_metadata(question), count=0, citations=[], gaps=[]),
            )
        limit = _limit(args)
        rows = _search_rows(question, limit)
        citations = [_citation(row) for row in rows]
        lines = [
            "Jarvis GBrain-style think (read-only):",
            f"Question: {_safe_display(question, limit=240)}",
            "",
            "Working answer:",
        ]
        if rows:
            for row in rows[:5]:
                excerpt = _excerpt(row, 180)
                lines.append(f"- From {_citation(row)}: {excerpt or 'memory body is empty'}")
        else:
            lines.append("- I do not have enough local memory evidence to answer this from Jarvis memory yet.")
        gaps = [
            "No model call, web search, email/calendar access, shell execution, Obsidian import, or computer control was used.",
            "This synthesis only uses local Jarvis memory rows already in the SQLite store.",
        ]
        if len(rows) < 3:
            gaps.append(f"Only {len(rows)} matching memory row(s) were found; ingest or save more notes for stronger answers.")
        if not rows:
            gaps.append("Ask Jarvis to remember relevant facts or ingest a known notes folder before relying on this answer.")
        lines.extend(["", "Citations:"])
        lines.extend([f"- {citation}" for citation in citations] or ["- none"])
        lines.extend(["", "Gap analysis:"])
        lines.extend(f"- {gap}" for gap in gaps)
        return ToolResult(
            "brain_think",
            True,
            "\n".join(lines),
            _brain_metadata(source="brain_think", status="ok" if rows else "empty", question=question, count=len(rows), citations=citations, gaps=gaps),
        )

    def brain_graph(args: dict[str, Any]) -> ToolResult:
        limit = _limit(args, default=12)
        memories = store.recent_memories(limit)
        people = store.list_people(limit)
        stats = store.memory_stats()
        category_counts = Counter({row["category"]: int(row["count"]) for row in stats})
        lines = [
            "Jarvis GBrain-style graph preview (read-only):",
            "Nodes:",
        ]
        if category_counts:
            for category, count in category_counts.most_common(8):
                lines.append(f"- category:{category} ({count} memories)")
        else:
            lines.append("- no memory categories yet")
        for person in people[:8]:
            relation = f" | {person['relation']}" if person["relation"] else ""
            lines.append(f"- person:{person['name']}{relation}")
        lines.append("")
        lines.append("Candidate edges:")
        edges: list[str] = []
        for memory in memories:
            edges.append(f"- memory M{memory['id']} -> category:{memory['category']}")
            memory_text = f"{memory['title']} {memory['body']}".lower()
            for person in people:
                if person["name"].lower() in memory_text:
                    edges.append(f"- memory M{memory['id']} -> person:{person['name']}")
            if len(edges) >= limit:
                break
        lines.extend(edges or ["- none yet"])
        lines.extend(
            [
                "",
                "Boundary: this is a preview only. It does not create graph tables, write notes, call external services, call a model, or control the computer.",
            ]
        )
        return ToolResult(
            "brain_graph",
            True,
            "\n".join(lines),
            _brain_metadata(
                source="brain_graph",
                status="ok",
                memory_nodes=len(memories),
                people_nodes=len(people),
                category_nodes=len(category_counts),
                edges=len(edges),
            ),
        )

    def brain_neighbors(args: dict[str, Any]) -> ToolResult:
        raw_memory_id = _first_present(args, ("memory_id", "id"))
        if isinstance(raw_memory_id, bool):
            return _brain_input_failure(
                "brain_neighbors",
                "memory_id must be a number.",
                _brain_metadata(source="brain_neighbors", status="refused", reason="bad_memory_id", memory_id=None, raw_memory_id=str(raw_memory_id), count=0, citations=[]),
            )
        try:
            memory_id = int(raw_memory_id)
        except (TypeError, ValueError):
            return _brain_input_failure(
                "brain_neighbors",
                "memory_id must be a number.",
                _brain_metadata(source="brain_neighbors", status="refused", reason="bad_memory_id", memory_id=None, raw_memory_id=_short_metadata(raw_memory_id), count=0, citations=[]),
            )
        if memory_id <= 0:
            return _brain_input_failure(
                "brain_neighbors",
                "memory_id must be a positive number.",
                _brain_metadata(source="brain_neighbors", status="refused", reason="bad_memory_id", memory_id=None, raw_memory_id=_short_metadata(raw_memory_id), count=0, citations=[]),
            )
        source = store.get_memory(memory_id)
        if not source:
            return ToolResult(
                "brain_neighbors",
                True,
                f"Jarvis GBrain-style neighbors (read-only): no memory found with id #{memory_id}.",
                _brain_metadata(source="brain_neighbors", status="empty", memory_id=memory_id, count=0, citations=[]),
            )
        limit = _limit(args, default=6, maximum=12)
        source_terms = set(_terms(f"{source['title']} {source['body']}"))
        scored: list[tuple[int, Any]] = []
        for row in store.list_memories(limit=300):
            if int(row["id"]) == memory_id:
                continue
            score = 0
            if row["category"] == source["category"]:
                score += 3
            row_terms = set(_terms(f"{row['title']} {row['body']}"))
            score += len(source_terms & row_terms)
            if score:
                scored.append((score, row))
        scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        rows = [row for _, row in scored[:limit]]
        lines = [
            "Jarvis GBrain-style neighbors (read-only):",
            f"Source: {_citation(source)}",
            "",
            "Related memories:",
        ]
        if rows:
            for index, row in enumerate(rows, start=1):
                lines.append(f"{index}. {_citation(row)}")
                excerpt = _excerpt(row, 160)
                if excerpt:
                    lines.append(f"   {excerpt}")
        else:
            lines.append("- No related local memories found yet.")
        lines.extend(
            [
                "",
                "Boundary: this only compares local memory category and overlapping terms. It does not call models, write memory, read files, or control the computer.",
            ]
        )
        citations = [_citation(row) for row in rows]
        return ToolResult(
            "brain_neighbors",
            True,
            "\n".join(lines),
            _brain_metadata(source="brain_neighbors", status="ok" if rows else "empty", memory_id=memory_id, count=len(rows), citations=citations),
        )

    return brain_search, brain_think, brain_graph, brain_neighbors
