"""Live Wikipedia summaries for Jarvis V2 (REST API, free, no key)."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    EXTERNAL_INFORMATION_RECOVERY_ACTION,
    declare_retryable_external_information_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools._http import friendly_http_error
from jarvis_v2.tools.research_connector import (
    corpus_is_relevant as _corpus_is_relevant,
    significant_words as _significant_words,
)


MAX_QUERY_CHARS = 120
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)


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


def _wiki_boundaries(*, calls_external_service: bool, calls_model: bool = False, executes_tools: bool = False) -> dict[str, bool]:
    return {
        "calls_model": calls_model,
        "calls_external_service": calls_external_service,
        "calls_external_services": calls_external_service,
        "executes_tools": executes_tools,
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


def _query_preview(value: Any) -> str:
    text = LOCAL_PATH_RE.sub("<local-path>", str(value or "").strip())
    return text[:MAX_QUERY_CHARS]


def _wiki_handoff(
    *,
    raw_query: str,
    query: str,
    status: str,
    reason: str = "",
    title: str = "",
    source_url_present: bool = False,
    exception_type: str = "",
    calls_external_service: bool,
    source: str = "wikipedia",
    calls_model: bool = False,
    executes_tools: bool = False,
) -> dict[str, Any]:
    next_safe_command = "wikipedia <topic>"
    boundaries = _wiki_boundaries(calls_external_service=calls_external_service, calls_model=calls_model, executes_tools=executes_tools)
    handoff = {
        "source": source,
        "wiki_summary_handoff_ready": True,
        "handoff_ready": True,
        "ready_for_operator": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "status": status,
        "reason": reason,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "query_preview": _query_preview(raw_query or query),
        "query_length": len(query),
        "raw_query_length": len(raw_query),
        "max_query_chars": MAX_QUERY_CHARS,
        "query_truncated": len(str(raw_query or "").strip().strip("?.!\"'")) > MAX_QUERY_CHARS,
        "title": title,
        "source_url_present": source_url_present,
        "summary_in_metadata": False,
        "exception_type": exception_type,
        "retry_safe": status in {"empty", "unavailable"},
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "wiki_summary_handoff_ready": True,
        "wiki_summary_ready_for_operator": handoff["ready_for_operator"],
        "wiki_summary_state_changed": handoff["state_changed"],
        "wiki_summary_changed": handoff["changed"],
        "wiki_summary_content_in_handoff": handoff["content_in_handoff"],
        "wiki_summary_summary_in_metadata": handoff["summary_in_metadata"],
        "wiki_summary_next_safe_command": handoff["next_safe_command"],
        "wiki_summary_next_safe_commands": handoff["next_safe_commands"],
        "wiki_summary_next_safe_command_count": handoff["next_safe_command_count"],
        "wiki_summary_authorizes_execution": handoff["authorizes_execution"],
        "wiki_summary_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "wiki_summary_approval_granted": handoff["approval_granted"],
        "wiki_summary_boundaries": boundaries,
        "wiki_summary_handoff": handoff,
    }


def _get(url: str) -> bytes:
    from jarvis_v2.tools._http import http_get
    return http_get(url, headers={"User-Agent": "JarvisV2/1.0 (personal assistant)"})


def _wiki_recovery_message(prefix: str) -> str:
    return f"{prefix} Check network access to Wikipedia, run setup check, then retry in a moment."


def _wiki_error_message(e: Exception, *, query: str) -> str:
    friendly = friendly_http_error(e, subject=f"a Wikipedia article on '{query}'", service="Wikipedia")
    lowered = friendly.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return _wiki_recovery_message("The Wikipedia summary request timed out.")
    if friendly.startswith("I couldn't find"):
        return friendly
    if friendly.startswith("Error:") or "having trouble" in lowered or "try again" in lowered:
        return _wiki_recovery_message("Wikipedia summaries are having trouble right now.")
    return friendly


def _web_fallback_tool_result(
    *,
    config: JarvisConfig,
    raw_query: str,
    query: str,
    metadata: dict[str, Any],
    not_found_message: str,
    not_found_reason: str,
    title: str = "",
) -> ToolResult:
    """Shared by all three wiki_summary miss cases (no title, mismatched
    title, empty extract): try the real web, and only show the Wikipedia-style
    "not found" message if the web search itself comes up empty too. `title`
    is the Wikipedia title that was found-but-unusable (empty extract case);
    empty for the no-title/mismatched-title cases."""
    fallback = _web_fallback(config, query)
    if fallback is None:
        # A web search was attempted (calls_external_service=True), but it
        # came up empty or irrelevant before ever reaching page-fetch or
        # model synthesis -- calls_model/executes_tools stay False since
        # neither actually ran.
        return ToolResult(
            "wiki_summary",
            True,
            not_found_message,
            _safe_metadata(
                **metadata,
                count=0,
                **_wiki_handoff(
                    raw_query=raw_query,
                    query=query,
                    status="empty",
                    reason=not_found_reason,
                    title=title,
                    calls_external_service=True,
                ),
            ),
        )
    body, extra = fallback
    return ToolResult(
        "wiki_summary",
        True,
        body,
        _safe_metadata(
            **metadata,
            count=extra["result_count"],
            page_read_count=extra["page_read_count"],
            synthesized=extra["synthesized"],
            **_wiki_handoff(
                raw_query=raw_query,
                query=query,
                status="ok",
                reason="web_fallback",
                calls_external_service=True,
                calls_model=True,
                executes_tools=True,
                source="web_fallback",
            ),
        ),
    )


def _web_fallback(config: JarvisConfig, query: str) -> tuple[str, dict[str, Any]] | None:
    """Real web search + read + synthesize, for when Wikipedia has no article
    or its best title match doesn't actually relate to the query (opensearch
    is fuzzy title matching, not full-text relevance search -- see
    _title_matches_query). Reuses the same DuckDuckGo + page-read + local-model
    pipeline as the `research` tool. Returns None if the web search itself
    comes up empty or the results are clearly unrelated to the query, so the
    caller can fall back to its own "not found" message instead of claiming a
    fallback that found nothing useful."""
    from jarvis_v2.tools.research_connector import ddg_search, _fetch_page_text, _synthesize

    results = ddg_search(query, max_results=5)
    if not results:
        return None
    corpus_parts = []
    page_read_count = 0
    for r in results[:3]:
        text = _fetch_page_text(r["url"]) or r.get("snippet", "")
        if text:
            page_read_count += 1
            corpus_parts.append(f"[{r['title']}]\n{text}")
    corpus = "\n\n".join(corpus_parts) or "\n".join(r.get("snippet", "") for r in results)
    if not _corpus_is_relevant(query, corpus):
        return None
    answer = _synthesize(config, query, corpus)
    sources = "\n".join(f"  • {r['title']} — {r['url']}" for r in results[:3])
    if answer:
        body = f"{answer}\n\nSources:\n{sources}"
    else:
        snips = "\n".join(f"  • {r['title']}: {r.get('snippet', '')[:160]}" for r in results[:3])
        body = f"Here's what I found on '{query}':\n{snips}\n\nSources:\n{sources}"
    return body, {
        "result_count": len(results),
        "page_read_count": page_read_count,
        "synthesized": bool(answer),
    }


def _resolve_title(query: str) -> str | None:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "opensearch", "search": query, "limit": 1, "format": "json"}
    )
    data = json.loads(_get(url).decode("utf-8"))
    titles = data[1] if isinstance(data, list) and len(data) > 1 else []
    return titles[0] if titles else None


def _title_matches_query(query: str, title: str) -> bool:
    """Opensearch is prefix/fuzzy title matching, not full-text relevance search
    -- great for single-topic nouns ("Ada Lovelace", "black holes"), but it
    silently returns an unrelated title for relational/factual questions like
    "Anthropic CEO" (matched "Claude (AI)" once, with zero shared words).
    Require at least one real word (or plural/stem variant) in common before
    trusting the match; otherwise treat it as a miss and fall back to the web."""
    query_words = _significant_words(query)
    title_words = _significant_words(title)
    if not query_words or not title_words:
        return True  # nothing meaningful to compare; don't block on it
    for qw in query_words:
        for tw in title_words:
            if qw == tw or qw.startswith(tw) or tw.startswith(qw):
                return True
    return False


_ROLE_WORDS = (
    # Business
    "ceo", "cfo", "cto", "coo", "cio", "cmo", "founder", "co-founder", "cofounder",
    "president", "vice president", "vp", "chairman", "chairwoman", "chairperson", "chair",
    "director", "manager", "general manager", "owner", "leader", "head", "chief",
    "chief executive", "chief financial officer", "chief technology officer",
    "spokesperson", "spokesman", "spokeswoman", "executive",
    # Government / politics
    "prime minister", "premier", "governor", "mayor", "secretary", "senator",
    "minister", "chancellor", "ambassador", "commissioner", "mp", "congressman",
    "congresswoman", "monarch", "king", "queen", "emperor", "empress", "sultan", "pope",
    "head of state", "head of government",
    # Academic / institutional
    "dean", "principal", "superintendent", "administrator",
    # Military
    "general", "admiral", "commander",
    # Sports / entertainment
    "head coach", "coach", "captain", "conductor",
    # Creative attribution
    "creator", "inventor", "author", "composer", "architect", "designer",
)

_RELATIONAL_VERB_PATTERNS = (
    r"who\s+(?:founded|created|invented|leads|runs|owns|wrote|composed|directed|built|started|designed|discovered)\b",
    r"\bcurrent(?:ly)?\s+(?:the\s+)?\w+\s+of\b",
)


def _is_relational_role_query(query: str) -> bool:
    """"Who is the CEO/founder/president of X" needs a current, targeted fact.
    A Wikipedia article's opening summary paragraph is a static overview of the
    ENTITY (e.g. the company or product), not an answer to a sub-question about
    it -- title-matching correctly enough (e.g. "Anthropic CEO" -> "Anthropic
    Claude", a real, related title) doesn't help, since that article's summary
    was never going to name the CEO. Route this query shape straight to real
    web search instead of trying Wikipedia first.

    Several of these role words are ALSO legitimate standalone Wikipedia
    topics on their own ("Secretary bird", "Captain America", "conductor"
    the physics concept, "head injury") -- a bare word-boundary match would
    wrongly divert those to the (slower, still-correct) web path. Require
    that something besides the matched role phrase remain: if "secretary"
    or "captain" is the ONLY significant content word, it's a standalone
    definitional query ("what is a king") and should stay on the fast
    Wikipedia path; if an entity name is also present ("Anthropic CEO",
    "king of Spain", "Secretary of State"), it's genuinely relational."""
    low = query.lower()
    for role in _ROLE_WORDS:
        match = re.search(rf"\b{re.escape(role)}\b", low)
        if not match:
            continue
        remainder = low[: match.start()] + low[match.end() :]
        if _significant_words(remainder):
            return True
    return any(re.search(pattern, low) for pattern in _RELATIONAL_VERB_PATTERNS)


def _fetch_summary(title: str) -> dict:
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))
    return json.loads(_get(url).decode("utf-8"))


def _looks_like_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _clean_query(value: str) -> str:
    text = " ".join(str(value or "").strip().strip("?.!\"'").split())
    if len(text.split()) > 1:
        text = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", text, flags=re.IGNORECASE).strip()
    return text


def make_wikipedia_tools(config: JarvisConfig):
    def wiki_summary(args: dict[str, Any]) -> ToolResult:
        raw_query = str(args.get("query") or args.get("text") or "")
        cleaned_query = _clean_query(raw_query)
        query = cleaned_query[:MAX_QUERY_CHARS]
        metadata = _safe_metadata(
            query_length=len(query),
            raw_query_length=len(raw_query),
            max_query_chars=MAX_QUERY_CHARS,
            query_truncated=len(cleaned_query) > MAX_QUERY_CHARS,
        )
        if not query:
            return ToolResult(
                "wiki_summary",
                False,
                "What should I look up on Wikipedia?",
                _safe_metadata(
                    **{
                        **metadata,
                        "calls_external_service": False,
                        **_wiki_handoff(
                            raw_query=raw_query,
                            query=query,
                            status="refused",
                            reason="missing_query",
                            calls_external_service=False,
                        ),
                    }
                ),
            )
        if len(cleaned_query) > MAX_QUERY_CHARS or _looks_like_local_path(raw_query) or not any(ch.isalnum() for ch in query):
            return ToolResult(
                "wiki_summary",
                False,
                "Please give me a short Wikipedia topic to look up.",
                _safe_metadata(
                    **{
                        **metadata,
                        "calls_external_service": False,
                        "reason": "invalid_query",
                        "local_path_query": _looks_like_local_path(raw_query),
                        **_wiki_handoff(
                            raw_query=raw_query,
                            query=query,
                            status="refused",
                            reason="invalid_query",
                            calls_external_service=False,
                        ),
                    }
                ),
            )
        try:
            if _is_relational_role_query(query):
                return _web_fallback_tool_result(
                    config=config,
                    raw_query=raw_query,
                    query=query,
                    metadata=metadata,
                    not_found_message=f"No Wikipedia article found for '{query}'.",
                    not_found_reason="relational_role_query",
                )
            title = _resolve_title(query)
            if not title:
                return _web_fallback_tool_result(
                    config=config,
                    raw_query=raw_query,
                    query=query,
                    metadata=metadata,
                    not_found_message=f"No Wikipedia article found for '{query}'.",
                    not_found_reason="not_found",
                )
            if not _title_matches_query(query, title):
                # Opensearch resolved SOME title, but it shares no real word
                # with the query -- e.g. "Anthropic CEO" once matched
                # "Claude (AI)". Trusting this would confidently answer the
                # wrong question, so treat it like no match was found.
                return _web_fallback_tool_result(
                    config=config,
                    raw_query=raw_query,
                    query=query,
                    metadata=metadata,
                    not_found_message=f"No Wikipedia article found for '{query}'.",
                    not_found_reason="title_mismatch",
                )
            data = _fetch_summary(title)
            extract = (data.get("extract") or "").strip()
            if not extract:
                return _web_fallback_tool_result(
                    config=config,
                    raw_query=raw_query,
                    query=query,
                    metadata={**metadata, "title": title},
                    not_found_message=f"No summary available for '{title}'.",
                    not_found_reason="no_summary",
                    title=title,
                )
            url = ((data.get("content_urls") or {}).get("desktop") or {}).get("page", "")
            body = f"{data.get('title', title)}: {extract}"
            if url:
                body += f"\n{url}"
            resolved_title = data.get("title", title)
            return ToolResult(
                "wiki_summary",
                True,
                body,
                _safe_metadata(
                    **metadata,
                    title=resolved_title,
                    **_wiki_handoff(
                        raw_query=raw_query,
                        query=query,
                        status="ok",
                        title=resolved_title,
                        source_url_present=bool(url),
                        calls_external_service=True,
                    ),
                ),
            )
        except Exception as e:
            failure_output = _wiki_error_message(e, query=query)
            failure_reason = (
                "not_found"
                if failure_output.startswith("I couldn't find")
                else "fetch_error"
            )
            failure_metadata = _safe_metadata(
                **metadata,
                exception_type=type(e).__name__,
                **_wiki_handoff(
                    raw_query=raw_query,
                    query=query,
                    status="unavailable",
                    reason=failure_reason,
                    exception_type=type(e).__name__,
                    calls_external_service=True,
                ),
            )
            if failure_reason == "fetch_error":
                failure_output = (
                    f"{failure_output} "
                    f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
                )
                failure_metadata = (
                    declare_retryable_external_information_failure(
                        failure_metadata,
                        output=failure_output,
                        action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                        commands=("setup check",),
                    )
                )
            return ToolResult(
                "wiki_summary",
                False,
                failure_output,
                failure_metadata,
            )

    from jarvis_v2.tools.registry import Tool, _tool_argument_contract
    return [
        Tool(
            "wiki_summary",
            "Get a Wikipedia summary of a person, place, or topic (free). Args: query or text.",
            RiskLevel.LOCAL_SAFE,
            wiki_summary,
            "personal",
            argument_contract=_tool_argument_contract(optional_strings=("query", "text")),
        ),
    ]
