"""Live news headlines for Jarvis V2 via Google News RSS (free, no API key)."""

from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    EXTERNAL_INFORMATION_RECOVERY_ACTION,
    declare_retryable_external_information_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools._http import http_get, http_get_json, friendly_http_error


MAX_QUERY_CHARS = 120
DEFAULT_NEWS_HL = "en-US"
DEFAULT_NEWS_GL = "US"
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": True,
        "calls_external_services": True,
        "executes_tools": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    base.update(extra)
    base["calls_external_services"] = base.get("calls_external_service") is True
    return base


def _news_boundaries(*, calls_external_service: bool) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": calls_external_service,
        "calls_external_services": calls_external_service,
        "executes_tools": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _headline_row(item: ET.Element, index: int) -> dict[str, Any]:
    title = _clean_raw(item.findtext("title"), 140)
    return {
        "index": index,
        "title": title,
        "title_chars": len(title),
    }


def _news_handoff(
    *,
    query: str,
    limit: int,
    rows: list[dict[str, Any]] | None,
    status: str,
    reason: str = "",
    calls_external_service: bool,
) -> dict[str, Any]:
    hl, gl = _news_locale()
    rows = rows or []
    cleaned_query = _clean_raw(query, MAX_QUERY_CHARS)
    next_safe_command = f"news about {query}" if query else "headlines"
    boundaries = _news_boundaries(calls_external_service=calls_external_service)
    handoff = {
        "source": "get_news",
        "news_handoff_ready": True,
        "handoff_ready": True,
        "ready_for_operator": True,
        "query": cleaned_query,
        "query_chars": len(cleaned_query),
        "limit": limit,
        "status": status,
        "reason": reason,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "headline_count": len(rows),
        "headlines": rows,
        "headline_titles": [row["title"] for row in rows],
        "locale": {"hl": hl, "gl": gl},
        "content_in_metadata": False,
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "retry_safe": status != "invalid_input",
        "boundaries": boundaries,
    }
    return {
        "news_handoff_ready": True,
        "news_ready_for_operator": handoff["ready_for_operator"],
        "news_state_changed": handoff["state_changed"],
        "news_changed": handoff["changed"],
        "news_content_in_handoff": handoff["content_in_handoff"],
        "news_content_in_metadata": handoff["content_in_metadata"],
        "news_next_safe_command": handoff["next_safe_command"],
        "news_next_safe_commands": handoff["next_safe_commands"],
        "news_next_safe_command_count": handoff["next_safe_command_count"],
        "news_authorizes_execution": handoff["authorizes_execution"],
        "news_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "news_approval_granted": handoff["approval_granted"],
        "news_boundaries": boundaries,
        "news_handoff": handoff,
    }


def _clean(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].strip()


def _clean_raw(value: Any, limit: int) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _clean(value, limit))


def _raw_int_metadata(value: Any, *, key: str, sanitized: int) -> dict[str, Any]:
    if value is None:
        return {key: sanitized}
    if isinstance(value, bool):
        return {key: sanitized, f"raw_{key}": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {key: sanitized, f"raw_{key}": _clean_raw(value, 80)}
    return {key: sanitized}


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(number, high))


def _news_locale() -> tuple[str, str]:
    raw = os.getenv("JARVIS_NEWS_LOCALE", "en-US:US").strip()
    if LOCAL_PATH_RE.search(raw):
        return DEFAULT_NEWS_HL, DEFAULT_NEWS_GL
    hl, _, gl = raw.partition(":")
    hl = _clean_locale_part(hl, DEFAULT_NEWS_HL, pattern=r"[A-Za-z]{2}(?:-[A-Za-z]{2})?")
    gl = _clean_locale_part(gl, DEFAULT_NEWS_GL, pattern=r"[A-Za-z]{2}").upper()
    return hl, gl


def _clean_locale_part(value: Any, default: str, *, pattern: str) -> str:
    text = str(value or "").strip()
    if not text or not re.fullmatch(pattern, text):
        return default
    pieces = text.split("-", 1)
    if len(pieces) == 2:
        return f"{pieces[0].lower()}-{pieces[1].upper()}"
    return text.lower()


def _news_url(query: str) -> str:
    hl, gl = _news_locale()
    ceid = f"{gl}:{hl.split('-')[0]}"
    base_q = {"hl": hl, "gl": gl, "ceid": ceid}
    if query:
        params = urllib.parse.urlencode({"q": query, **base_q})
        return "https://news.google.com/rss/search?" + params
    return "https://news.google.com/rss?" + urllib.parse.urlencode(base_q)


def _fetch(query: str) -> bytes:
    return http_get(_news_url(query), headers={"User-Agent": "Mozilla/5.0"})


def _news_recovery_message(prefix: str) -> str:
    return f"{prefix} Check network access to Google News, run setup check, then retry in a moment."


def _news_error_message(e: Exception) -> str:
    friendly = friendly_http_error(e, subject="news", service="The news service")
    lowered = friendly.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return _news_recovery_message("The news request timed out.")
    if friendly.startswith("Error:") or "having trouble" in lowered or "try again" in lowered:
        return _news_recovery_message("The news service is having trouble right now.")
    return friendly


def make_news_tools(config: JarvisConfig):
    def get_news(args: dict[str, Any]) -> ToolResult:
        raw_query = args.get("query") or args.get("topic") or args.get("about")
        query = _clean(raw_query, MAX_QUERY_CHARS)
        limit = _bounded_int(args.get("limit"), 5, 1, 10)
        limit_metadata = _raw_int_metadata(args.get("limit"), key="limit", sanitized=limit)
        if raw_query not in (None, "") and LOCAL_PATH_RE.search(str(raw_query)):
            return ToolResult(
                "get_news",
                False,
                "Please give me a real news topic, not a local file path.",
                _safe_metadata(
                    calls_external_service=False,
                    reason="invalid_query",
                    query=_clean_raw(raw_query, MAX_QUERY_CHARS),
                    local_path_query=True,
                    **limit_metadata,
                    **_news_handoff(
                        query=_clean_raw(raw_query, MAX_QUERY_CHARS),
                        limit=limit,
                        rows=[],
                        status="invalid_input",
                        reason="invalid_query",
                        calls_external_service=False,
                    ),
                ),
            )
        if raw_query not in (None, "") and query and not any(ch.isalnum() for ch in query):
            return ToolResult(
                "get_news",
                False,
                "Please give me a real news topic.",
                _safe_metadata(
                    calls_external_service=False,
                    reason="invalid_query",
                    query=query,
                    **limit_metadata,
                    **_news_handoff(
                        query=query,
                        limit=limit,
                        rows=[],
                        status="invalid_input",
                        reason="invalid_query",
                        calls_external_service=False,
                    ),
                ),
            )
        try:
            root = ET.fromstring(_fetch(query))
            items = root.findall(".//item")[:limit]
            if not items:
                return ToolResult(
                    "get_news",
                    True,
                    "No headlines found.",
                    _safe_metadata(
                        count=0,
                        query=query,
                        **limit_metadata,
                        **_news_handoff(
                            query=query,
                            limit=limit,
                            rows=[],
                            status="empty",
                            reason="no_headlines",
                            calls_external_service=True,
                        ),
                    ),
                )
            header = f"Top headlines about '{query}':" if query else "Top headlines:"
            lines = [header]
            rows = [_headline_row(item, index) for index, item in enumerate(items, start=1)]
            for row in rows:
                title = row["title"]
                # Google News titles are "Headline - Source"; keep as-is, it's readable.
                if title:
                    lines.append(f"  • {title}")
            return ToolResult(
                "get_news",
                True,
                "\n".join(lines),
                _safe_metadata(
                    count=len(rows),
                    query=query,
                    **limit_metadata,
                    **_news_handoff(
                        query=query,
                        limit=limit,
                        rows=rows,
                        status="ok",
                        calls_external_service=True,
                    ),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_news_error_message(e)} "
                f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_news",
                False,
                failure_output,
                declare_retryable_external_information_failure(
                    _safe_metadata(
                        query=query,
                        exception_type=type(e).__name__,
                        reason="fetch_error",
                        **limit_metadata,
                        **_news_handoff(
                            query=query,
                            limit=limit,
                            rows=[],
                            status="unavailable",
                            reason="fetch_error",
                            calls_external_service=True,
                        ),
                    ),
                    output=failure_output,
                    action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    from jarvis_v2.tools.registry import Tool, _tool_argument_contract
    return [
        Tool(
            "get_news",
            "Get current news headlines (free, Google News). Args: query (optional topic), limit (default 5).",
            RiskLevel.LOCAL_SAFE,
            get_news,
            "personal",
            argument_contract=_tool_argument_contract(
                optional_strings=("query", "topic", "about"),
                optional_integer_ranges=(("limit", 1, 10),),
            ),
        ),
    ]
