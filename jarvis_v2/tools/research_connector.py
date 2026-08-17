"""Browser research for Jarvis V2: web search + read + synthesize, with sources.

Uses DuckDuckGo Lite (not bot-blocked like the HTML endpoint) for results and the
existing browser page fetcher to read them. Synthesis uses the local model when
available, falling back to extracted snippets so it never fabricates.
"""

from __future__ import annotations

import html as _html
import ipaddress
import re
import urllib.parse
import urllib.request
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    EXTERNAL_INFORMATION_RECOVERY_ACTION,
    declare_retryable_external_information_failure,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.model_provider import normalized_model_provider, provider_output_token_limit
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig


MAX_QUERY_CHARS = 200
MAX_PAGE_CHARS = 2500
MAX_RESEARCH_OUTPUT_TOKENS = 700
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
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


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": True,
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
    return base


def _research_boundaries(
    *,
    calls_external_service: bool,
    calls_model: bool = False,
    executes_tools: bool = False,
) -> dict[str, bool]:
    return {
        "calls_model": calls_model,
        "calls_external_service": calls_external_service,
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


def _query_preview(value: str) -> str:
    public = LOCAL_PATH_RE.sub("<local-path>", value or "")
    return SECRET_VALUE_RE.sub("<private-value>", public).strip()[:MAX_QUERY_CHARS]


def _result_rows(results: list[dict], *, limit: int = 3) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result in results[:limit]:
        rows.append({"title": str(result.get("title") or "")[:160], "url": str(result.get("url") or "")[:300]})
    return rows


def _lookup_handoff(
    *,
    tool_name: str,
    raw_query: str,
    query: str,
    status: str,
    reason: str = "",
    results: list[dict] | None = None,
    synthesized: bool = False,
    page_read_count: int = 0,
    exception_type: str = "",
    calls_external_service: bool,
    calls_model: bool = False,
    executes_tools: bool = False,
    model_provider: str = "",
    external_model_call_attempted: bool = False,
) -> dict[str, Any]:
    rows = _result_rows(results or [])
    boundaries = _research_boundaries(
        calls_external_service=calls_external_service,
        calls_model=calls_model,
        executes_tools=executes_tools,
    )
    next_safe_command = "research <topic>" if tool_name == "research" else "look up <topic>"
    handoff = {
        "source": tool_name,
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "query_preview": _query_preview(raw_query or query),
        "query_length": len(query),
        "raw_query_length": len(raw_query),
        "max_query_chars": MAX_QUERY_CHARS,
        "query_truncated": len((raw_query or "").strip()) > MAX_QUERY_CHARS,
        "result_count": len(results or []),
        "result_rows": rows,
        "result_row_count": len(rows),
        "page_read_count": page_read_count,
        "synthesized": synthesized,
        "model_provider": model_provider,
        "external_model_call_attempted": external_model_call_attempted,
        "shares_research_query_with_external_model": external_model_call_attempted,
        "shares_research_sources_with_external_model": external_model_call_attempted,
        "model_source_corpus_in_metadata": False,
        "model_response_content_in_metadata": False,
        "answer_in_metadata": False,
        "snippet_in_metadata": False,
        "page_text_in_metadata": False,
        "exception_type": exception_type,
        "retry_safe": status in {"empty", "unavailable"},
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        f"{tool_name}_handoff_ready": True,
        f"{tool_name}_ready_for_operator": handoff["ready_for_operator"],
        f"{tool_name}_state_changed": handoff["state_changed"],
        f"{tool_name}_changed": handoff["changed"],
        f"{tool_name}_content_in_handoff": handoff["content_in_handoff"],
        f"{tool_name}_next_safe_command": handoff["next_safe_command"],
        f"{tool_name}_next_safe_commands": handoff["next_safe_commands"],
        f"{tool_name}_next_safe_command_count": handoff["next_safe_command_count"],
        f"{tool_name}_authorizes_execution": handoff["authorizes_execution"],
        f"{tool_name}_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        f"{tool_name}_approval_granted": handoff["approval_granted"],
        f"{tool_name}_boundaries": boundaries,
        "ready_for_operator": handoff["ready_for_operator"],
        "state_changed": handoff["state_changed"],
        "changed": handoff["changed"],
        "content_in_handoff": handoff["content_in_handoff"],
        "authorizes_execution": handoff["authorizes_execution"],
        "authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "approval_granted": handoff["approval_granted"],
        f"{tool_name}_handoff": handoff,
    }


def _strip_tags(value: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()


def _looks_like_local_path(value: str) -> bool:
    return bool(LOCAL_PATH_RE.search(value or ""))


def _looks_like_private_query(value: str) -> bool:
    return _looks_like_local_path(value) or bool(SECRET_VALUE_RE.search(value or ""))


def _safe_result_url(value: object) -> str:
    """Keep displayed/fetched search results on public-looking HTTP(S) URLs.

    The browser tool performs the authoritative DNS, redirect, and response-size
    checks before a page is fetched.  This first-pass fence prevents a search
    result from exposing credentials or handing a non-web/private-literal URL to
    that tool at all.
    """

    url = str(value or "").strip()
    if len(url) > 300:
        return ""
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        if parsed.username is not None or parsed.password is not None:
            return ""
        hostname = parsed.hostname.rstrip(".").lower()
        if hostname == "localhost" or hostname.endswith(".localhost"):
            return ""
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            return ""
    except (TypeError, ValueError):
        return ""
    return url


def _local_query_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
    action: str,
    tool_name: str,
    raw_query: str,
    query: str,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    return declare_retryable_local_read_failure(
        _safe_metadata(
            **metadata,
            calls_external_service=False,
            reason=reason,
            **extra,
            **_lookup_handoff(
                tool_name=tool_name,
                raw_query=raw_query,
                query=query,
                status="refused",
                reason=reason,
                calls_external_service=False,
            ),
        ),
        output=output,
        action=action,
    )


def _external_lookup_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
    tool_name: str,
    raw_query: str,
    query: str,
    exception: Exception,
) -> dict[str, Any]:
    return declare_retryable_external_information_failure(
        _safe_metadata(
            **metadata,
            exception_type=type(exception).__name__,
            **_lookup_handoff(
                tool_name=tool_name,
                raw_query=raw_query,
                query=query,
                status="unavailable",
                reason="fetch_error",
                exception_type=type(exception).__name__,
                calls_external_service=True,
            ),
        ),
        output=output,
        action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
        commands=("setup check",),
    )


def _query_and_metadata(args: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    raw_query = str(args.get("query") or args.get("text") or "")
    cleaned_query = raw_query.strip()
    query = cleaned_query[:MAX_QUERY_CHARS]
    metadata = {
        "query_length": len(query),
        "raw_query_length": len(raw_query),
        "max_query_chars": MAX_QUERY_CHARS,
        "query_truncated": len(cleaned_query) > MAX_QUERY_CHARS,
    }
    return query, cleaned_query, metadata


def _get(url: str) -> str:
    from jarvis_v2.tools._http import http_get
    return http_get(url).decode("utf-8", errors="replace")


def _search_recovery_message(prefix: str) -> str:
    return f"{prefix} Check network access to DuckDuckGo Lite, run `setup check`, then retry in a moment."


def _search_error_message(e: Exception, *, subject: str, prefix: str) -> str:
    from jarvis_v2.tools._http import friendly_http_error

    friendly = friendly_http_error(e, subject=subject, service="DuckDuckGo Lite search")
    lowered = friendly.lower()
    if friendly.startswith("I couldn't find"):
        return friendly
    if "timed out" in lowered or "timeout" in lowered:
        return _search_recovery_message("The search request timed out.")
    if friendly.startswith("Error:") or "having trouble" in lowered or "try again" in lowered:
        return _search_recovery_message(prefix)
    return friendly


def ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """Return [{title, url, snippet}] from DuckDuckGo Lite."""
    html = _get("https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query}))
    # Attribute order varies (href can come before or after class), so match the
    # whole anchor that contains "result-link" and pull href out separately.
    anchors = re.findall(r"<a\s+([^>]*?result-link[^>]*?)>(.*?)</a>", html, re.S)
    snippets = re.findall(r"result-snippet['\"][^>]*>(.*?)</td>", html, re.S)
    results: list[dict] = []
    for i, (attrs, title) in enumerate(anchors[:max_results]):
        href_m = re.search(r"href=\"([^\"]+)\"", attrs)
        href = href_m.group(1) if href_m else ""
        m = re.search(r"uddg=([^&]+)", href)
        url = urllib.parse.unquote(m.group(1)) if m else ("https:" + href if href.startswith("//") else href)
        url = _safe_result_url(url)
        if not url:
            continue
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        results.append({"title": _strip_tags(title), "url": url, "snippet": snippet})
    return results


def _fetch_page_text(url: str) -> str:
    from jarvis_v2.tools import browser
    try:
        res = browser.fetch_page({"url": url})
        return res.output[:MAX_PAGE_CHARS] if res.ok else ""
    except Exception:
        return ""


def _synthesize(config: JarvisConfig, query: str, corpus: str) -> str:
    try:
        from jarvis_v2.agent.model_provider import generate_model_text

        prompt = (
            "Using ONLY the sources below, answer the question concisely (3-5 sentences). "
            "If the sources don't cover it, say so. Do not invent facts.\n\n"
            f"Question: {query}\n\nSources:\n{corpus[:4000]}"
        )
        return generate_model_text(
            provider=config.model_provider,
            model=config.chat_model,
            messages=[{"role": "user", "content": prompt}],
            timeout_seconds=max(40.0, config.chat_timeout_seconds),
            max_output_tokens=provider_output_token_limit(
                config.model_provider,
                local_output_tokens=MAX_RESEARCH_OUTPUT_TOKENS,
                openai_output_tokens=config.openai_max_output_tokens,
            ),
            temperature=0.2,
            reasoning_effort=config.chat_reasoning_effort,
            keep_alive="30m",
        )
    except Exception:
        return ""


_RESEARCH_STOPWORDS = {
    "the", "a", "an", "of", "and", "in", "on", "at", "to", "for",
    "who", "what", "was", "are", "is", "were",
}


def significant_words(text: str) -> set[str]:
    """Lowercased, de-stopworded, len>=3 word set. Shared by the corpus
    relevance check below and reused by wikipedia_connector.py's web
    fallback and relational-role-query detection, so both surfaces judge
    relevance the same way instead of drifting apart."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in _RESEARCH_STOPWORDS}


def corpus_is_relevant(query: str, corpus: str) -> bool:
    """A DDG search for a truly nonsense query can still surface some
    coincidentally-ranked hit (found 2026-07-06: a fixed test string
    verbatim-matched bot-spam text on an unrelated blog); synthesizing an
    answer from it produces a confusing "I couldn't find that" disclaimer
    sitting next to an irrelevant source citation. Far more reliable than
    keyword-matching the model's own (highly variable) refusal phrasing:
    just check whether the fetched text actually contains any of the
    query's significant words before bothering to synthesize."""
    query_words = significant_words(query)
    if not query_words:
        return True
    return bool(query_words & significant_words(corpus))


def make_research_tools(config: JarvisConfig):
    def web_lookup(args: dict[str, Any]) -> ToolResult:
        query, cleaned_query, metadata = _query_and_metadata(args)
        if not query:
            action = "Provide a public-information search query, then retry."
            output = f"What should I look up? {action}"
            return ToolResult(
                "web_lookup",
                False,
                output,
                _local_query_failure_metadata(
                    metadata,
                    output=output,
                    action=action,
                    tool_name="web_lookup",
                    raw_query=cleaned_query,
                    query=query,
                    reason="missing_query",
                ),
            )
        if _looks_like_private_query(cleaned_query):
            action = "Replace private-looking input with a public-information search query, then retry."
            output = f"Please give me a topic or search query, not a local file path or private value. {action}"
            return ToolResult(
                "web_lookup",
                False,
                output,
                _local_query_failure_metadata(
                    metadata,
                    output=output,
                    action=action,
                    tool_name="web_lookup",
                    raw_query=cleaned_query,
                    query=query,
                    reason="invalid_query",
                    local_path_query=_looks_like_local_path(cleaned_query),
                    private_query=True,
                ),
            )
        if len(cleaned_query) > MAX_QUERY_CHARS or not any(ch.isalnum() for ch in query):
            action = f"Use {MAX_QUERY_CHARS} or fewer characters with letters or numbers, then retry."
            output = f"Please give me a short search query. {action}"
            return ToolResult(
                "web_lookup",
                False,
                output,
                _local_query_failure_metadata(
                    metadata,
                    output=output,
                    action=action,
                    tool_name="web_lookup",
                    raw_query=cleaned_query,
                    query=query,
                    reason="invalid_query",
                ),
            )
        try:
            results = ddg_search(query, max_results=5)
            if not results:
                return ToolResult(
                    "web_lookup",
                    True,
                    f"No results found for '{query}'.",
                    _safe_metadata(
                        **metadata,
                        count=0,
                        **_lookup_handoff(
                            tool_name="web_lookup",
                            raw_query=cleaned_query,
                            query=query,
                            status="empty",
                            reason="no_results",
                            calls_external_service=True,
                        ),
                    ),
                )
            lines = [f"Top results for '{query}':"]
            for r in results:
                lines.append(f"  • {r['title']}\n    {r['url']}")
                if r.get("snippet"):
                    lines.append(f"    {r['snippet'][:160]}")
            return ToolResult(
                "web_lookup",
                True,
                "\n".join(lines),
                _safe_metadata(
                    **metadata,
                    count=len(results),
                    **_lookup_handoff(
                        tool_name="web_lookup",
                        raw_query=cleaned_query,
                        query=query,
                        status="ok",
                        results=results,
                        calls_external_service=True,
                    ),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_search_error_message(e, subject='those results', prefix='Search is having trouble right now.')} "
                f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "web_lookup",
                False,
                failure_output,
                _external_lookup_failure_metadata(
                    metadata,
                    output=failure_output,
                    tool_name="web_lookup",
                    raw_query=cleaned_query,
                    query=query,
                    exception=e,
                ),
            )

    def research(args: dict[str, Any]) -> ToolResult:
        query, cleaned_query, metadata = _query_and_metadata(args)
        if not query:
            action = "Provide a public-information research query, then retry."
            output = f"What should I research? {action}"
            return ToolResult(
                "research",
                False,
                output,
                _local_query_failure_metadata(
                    metadata,
                    output=output,
                    action=action,
                    tool_name="research",
                    raw_query=cleaned_query,
                    query=query,
                    reason="missing_query",
                ),
            )
        if _looks_like_private_query(cleaned_query):
            action = "Replace private-looking input with a public-information research query, then retry."
            output = f"Please give me a topic or research query, not a local file path or private value. {action}"
            return ToolResult(
                "research",
                False,
                output,
                _local_query_failure_metadata(
                    metadata,
                    output=output,
                    action=action,
                    tool_name="research",
                    raw_query=cleaned_query,
                    query=query,
                    reason="invalid_query",
                    local_path_query=_looks_like_local_path(cleaned_query),
                    private_query=True,
                ),
            )
        if len(cleaned_query) > MAX_QUERY_CHARS or not any(ch.isalnum() for ch in query):
            action = f"Use {MAX_QUERY_CHARS} or fewer characters with letters or numbers, then retry."
            output = f"Please give me a short research query. {action}"
            return ToolResult(
                "research",
                False,
                output,
                _local_query_failure_metadata(
                    metadata,
                    output=output,
                    action=action,
                    tool_name="research",
                    raw_query=cleaned_query,
                    query=query,
                    reason="invalid_query",
                ),
            )
        try:
            results = ddg_search(query, max_results=5)
            if not results:
                return ToolResult(
                    "research",
                    True,
                    f"No results found for '{query}'.",
                    _safe_metadata(
                        **metadata,
                        count=0,
                        **_lookup_handoff(
                            tool_name="research",
                            raw_query=cleaned_query,
                            query=query,
                            status="empty",
                            reason="no_results",
                            calls_external_service=True,
                        ),
                    ),
                )
            corpus_parts = []
            page_read_count = 0
            for r in results[:3]:
                page_text = _fetch_page_text(r["url"])
                text = page_text or r.get("snippet", "")
                if page_text:
                    page_read_count += 1
                if text:
                    corpus_parts.append(f"[{r['title']}]\n{text}")
            corpus = "\n\n".join(corpus_parts) or "\n".join(r.get("snippet", "") for r in results)

            if not corpus_is_relevant(query, corpus):
                # A DDG hit came back, but it doesn't actually contain any of
                # the query's own significant words -- synthesizing an answer
                # from it would just produce a confusing "couldn't find that"
                # disclaimer next to an irrelevant source. Treat it like no
                # results, rather than showing that non-answer as if it were one.
                return ToolResult(
                    "research",
                    True,
                    f"No relevant results found for '{query}'.",
                    _safe_metadata(
                        **metadata,
                        count=0,
                        **_lookup_handoff(
                            tool_name="research",
                            raw_query=cleaned_query,
                            query=query,
                            status="empty",
                            reason="irrelevant_results",
                            calls_external_service=True,
                        ),
                    ),
                )

            answer = _synthesize(config, query, corpus)
            model_provider = normalized_model_provider(config.model_provider)
            external_model_call_attempted = model_provider == "openai"
            sources = "\n".join(f"  • {r['title']} — {r['url']}" for r in results[:3])
            if answer:
                body = f"{answer}\n\nSources:\n{sources}"
            else:
                # No model: give the snippets + sources rather than nothing.
                snips = "\n".join(f"  • {r['title']}: {r.get('snippet', '')[:160]}" for r in results[:3])
                body = f"Here's what I found on '{query}':\n{snips}\n\nSources:\n{sources}"
            return ToolResult(
                "research",
                True,
                body,
                _safe_metadata(
                    **metadata,
                    count=len(results),
                    synthesized=bool(answer),
                    calls_model=True,
                    model_provider=model_provider,
                    model_call_attempted=True,
                    external_model_call_attempted=external_model_call_attempted,
                    shares_research_query_with_external_model=external_model_call_attempted,
                    shares_research_sources_with_external_model=external_model_call_attempted,
                    model_source_corpus_in_metadata=False,
                    model_response_content_in_metadata=False,
                    executes_tools=True,
                    **_lookup_handoff(
                        tool_name="research",
                        raw_query=cleaned_query,
                        query=query,
                        status="ok",
                        results=results,
                        synthesized=bool(answer),
                        page_read_count=page_read_count,
                        calls_external_service=True,
                        calls_model=True,
                        executes_tools=True,
                        model_provider=model_provider,
                        external_model_call_attempted=external_model_call_attempted,
                    ),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_search_error_message(e, subject='that research', prefix='Research is having trouble right now.')} "
                f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "research",
                False,
                failure_output,
                _external_lookup_failure_metadata(
                    metadata,
                    output=failure_output,
                    tool_name="research",
                    raw_query=cleaned_query,
                    query=query,
                    exception=e,
                ),
            )

    from jarvis_v2.tools.registry import Tool, _tool_argument_contract
    return [
        Tool(
            "web_lookup",
            "Quick web search: top results with links and snippets. Args: query.",
            RiskLevel.READ_ONLY,
            web_lookup,
            "browser",
            argument_contract=_tool_argument_contract(optional_strings=("query", "text")),
        ),
        Tool(
            "research",
            "Research a topic: searches, reads the top pages, and synthesizes an answer with sources. Args: query.",
            RiskLevel.LOCAL_SAFE,
            research,
            "browser",
            argument_contract=_tool_argument_contract(optional_strings=("query", "text")),
        ),
    ]
