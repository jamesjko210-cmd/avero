"""Smoke tests for the Wikipedia connector (mocked fetch, no network)."""

from __future__ import annotations

from typing import Any

from jarvis_v2.agent.failure_guidance import EXTERNAL_INFORMATION_RECOVERY_ACTION
from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import wikipedia_connector as wc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry


NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


def _tools():
    return {t.name: t for t in wc.make_wikipedia_tools(load_config())}


def _assert_wiki_recovery_message(output: str, label: str) -> None:
    lowered = output.lower()
    for fragment in [
        "network access to wikipedia",
        "setup check",
        "retry in a moment",
    ]:
        if fragment not in lowered:
            raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")
    for forbidden in ["http error", "offline", "traceback", "/users/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in lowered:
            raise SystemExit(f"{label} leaked raw backend text {forbidden!r}: {output}")


def _assert_external_information_guidance(result, label: str) -> None:
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    if EXTERNAL_INFORMATION_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} hid the canonical recovery action: {result.output}")
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} recovery field {key} drifted: {result.metadata}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": EXTERNAL_INFORMATION_RECOVERY_ACTION,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")


def _assert_wiki_handoff(
    metadata: dict[str, Any],
    label: str,
    *,
    status: str,
    reason: str = "",
    calls_external_service: bool = True,
    retry_safe: bool = False,
) -> dict[str, Any]:
    handoff = metadata.get("wiki_summary_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing wiki_summary_handoff: {metadata}")
    if handoff.get("source") != "wikipedia" or handoff.get("status") != status:
        raise SystemExit(f"{label} has wrong source/status: {handoff}")
    if metadata.get("wiki_summary_handoff_ready") is not True or handoff.get("wiki_summary_handoff_ready") is not True:
        raise SystemExit(f"{label} should expose flat and nested readiness flags: {metadata} vs {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should expose nested handoff_ready: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should be operator-ready: {handoff}")
    if metadata.get("wiki_summary_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} ready alias parity failed: {metadata} vs {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
        alias = f"wiki_summary_{key}"
        if metadata.get(alias) is not expected_value or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} has wrong reason: {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} should not claim changed state: {handoff}")
    if metadata.get("wiki_summary_state_changed") != handoff.get("state_changed") or metadata.get("wiki_summary_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} state alias parity failed: {metadata} vs {handoff}")
    if handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep content out of handoff: {handoff}")
    if metadata.get("wiki_summary_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} content alias parity failed: {metadata} vs {handoff}")
    for key in ("query_length", "raw_query_length", "max_query_chars", "query_truncated", "exception_type"):
        if handoff.get(key) != metadata.get(key, ""):
            raise SystemExit(f"{label} should mirror {key}: {handoff} vs {metadata}")
    if handoff.get("summary_in_metadata") is not False:
        raise SystemExit(f"{label} should exclude summary text from metadata: {handoff}")
    if metadata.get("wiki_summary_summary_in_metadata") != handoff.get("summary_in_metadata"):
        raise SystemExit(f"{label} summary-in-metadata alias parity failed: {metadata} vs {handoff}")
    if handoff.get("retry_safe") is not retry_safe:
        raise SystemExit(f"{label} has wrong retry safety: {handoff}")
    if handoff.get("next_safe_command") != "wikipedia <topic>":
        raise SystemExit(f"{label} has wrong next safe command: {handoff}")
    expected_commands = ["wikipedia <topic>"]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} has wrong next safe command list: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} has wrong next safe command count: {handoff}")
    if metadata.get("wiki_summary_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("wiki_summary_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("wiki_summary_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} next-command-count alias parity failed: {metadata} vs {handoff}")
    boundaries = handoff.get("boundaries")
    expected = {
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
        **NO_AUTHORITY_FLAGS,
    }
    if boundaries != expected:
        raise SystemExit(f"{label} has wrong boundaries: {boundaries}")
    if metadata.get("wiki_summary_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary alias parity failed: {metadata} vs {handoff}")
    if metadata.get("calls_external_service") is not calls_external_service:
        raise SystemExit(f"{label} flat external-call flag should match handoff: {metadata}")
    if metadata.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} flat plural external-call flag should match handoff: {metadata}")
    combined = f"{metadata}\n{handoff}".lower()
    for fragment in ("/users/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in combined:
            raise SystemExit(f"{label} leaked local path fragment {fragment!r}: {metadata} {handoff}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["wiki_summary"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("wiki_summary should be LOCAL_SAFE")


def test_wiki_summary_argument_contract_fences_malformed_input() -> None:
    resolved_queries: list[str] = []
    original_resolve = wc._resolve_title
    original_fetch_summary = wc._fetch_summary

    def mocked_resolve(query: str) -> str:
        resolved_queries.append(query)
        return "Ada Lovelace"

    try:
        wc._resolve_title = mocked_resolve  # type: ignore[assignment]
        private_sentinel = "private-wikipedia-contract-sentinel"
        tool = _tools()["wiki_summary"]
        contract = tool.argument_contract
        actual_shape = (
            tuple(
                (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                for field in contract.fields
            )
            if contract is not None
            else ()
        )
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown is not False
            or actual_shape != (("query", ("string",), False), ("text", ("string",), False))
        ):
            raise SystemExit(f"wiki_summary should have an exact strict query/text contract: {contract}")

        registry = ToolRegistry()
        registry.register(tool)
        executor = Executor(registry, PermissionPolicy())
        malformed_args: list[object] = [
            None,
            ["not", "an", "object"],
            {"query": {"private": private_sentinel}},
            {"query": True},
            {"query": [private_sentinel]},
            {"text": 7},
            {"query": "Ada Lovelace", "unexpected": private_sentinel},
        ]
        rejected_surfaces: list[object] = []
        for args in malformed_args:
            result = executor.execute(PlannedAction("wiki_summary", args, "wikipedia contract smoke"))  # type: ignore[arg-type]
            if (
                result.ok
                or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
                or result.metadata.get("requires_confirmation") is not False
            ):
                raise SystemExit(f"malformed Wikipedia input passed the executor: {result}")
            rejected_surfaces.append({"output": result.output, "metadata": result.metadata})
        if resolved_queries:
            raise SystemExit(f"malformed Wikipedia input reached the resolver: {resolved_queries}")
        if private_sentinel in str(rejected_surfaces):
            raise SystemExit("Wikipedia argument rejection leaked a supplied private value")

        wc._fetch_summary = lambda title: {  # type: ignore[assignment]
            "title": title,
            "extract": "Ada Lovelace was an English mathematician.",
            "content_urls": {},
        }
        for args in ({"query": "Ada Lovelace"}, {"text": "Ada Lovelace"}):
            result = executor.execute(PlannedAction("wiki_summary", args, "valid Wikipedia contract smoke"))
            if not result.ok or resolved_queries[-1] != "Ada Lovelace":
                raise SystemExit(f"valid Wikipedia input did not preserve the query alias: {args} {result}")
        empty = executor.execute(PlannedAction("wiki_summary", {}, "empty Wikipedia contract smoke"))
        if (
            empty.ok
            or empty.metadata.get("handler_invoked") is not True
            or "what should i look up on wikipedia" not in empty.output.lower()
        ):
            raise SystemExit(f"empty Wikipedia compatibility changed: {empty}")
    finally:
        wc._resolve_title = original_resolve  # type: ignore[assignment]
        wc._fetch_summary = original_fetch_summary  # type: ignore[assignment]


def test_summarizes_topic() -> None:
    wc._resolve_title = lambda query: "Eiffel Tower"  # type: ignore
    wc._fetch_summary = lambda title: {  # type: ignore
        "title": "Eiffel Tower",
        "extract": "The Eiffel Tower is a tower in Paris, France.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Eiffel_Tower"}},
    }
    out = _tools()["wiki_summary"].handler({"query": "eiffel tower"})
    if not out.ok or "Eiffel Tower" not in out.output or "Paris" not in out.output:
        raise SystemExit(f"wiki output wrong: {out.output}")
    if "en.wikipedia.org" not in out.output:
        raise SystemExit("wiki summary should include the source link")
    handoff = _assert_wiki_handoff(out.metadata, "wiki success", status="ok")
    if handoff.get("title") != "Eiffel Tower" or handoff.get("source_url_present") is not True:
        raise SystemExit(f"wiki success should preserve title and source-url presence: {handoff}")
    if "Paris" in str(handoff):
        raise SystemExit(f"wiki success should not copy summary content into handoff: {handoff}")


def test_strips_polite_suffix_from_query() -> None:
    resolved = []

    def fake_resolve(query):
        resolved.append(query)
        return "Ada Lovelace"

    wc._resolve_title = fake_resolve  # type: ignore
    wc._fetch_summary = lambda title: {  # type: ignore
        "title": "Ada Lovelace",
        "extract": "Ada Lovelace was an English mathematician.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Ada_Lovelace"}},
    }
    out = _tools()["wiki_summary"].handler({"query": "Ada Lovelace please"})
    if not out.ok or "Ada Lovelace" not in out.output:
        raise SystemExit(f"polite wiki query should succeed: {out.output} {out.metadata}")
    if resolved != ["Ada Lovelace"]:
        raise SystemExit(f"polite wiki query should resolve cleaned topic: {resolved}")
    if out.metadata.get("raw_query_length") != len("Ada Lovelace please") or out.metadata.get("query_length") != len("Ada Lovelace"):
        raise SystemExit(f"polite wiki query should preserve raw and cleaned lengths: {out.metadata}")
    handoff = _assert_wiki_handoff(out.metadata, "polite wiki query", status="ok")
    if handoff.get("query_preview") != "Ada Lovelace please" or handoff.get("query_length") != len("Ada Lovelace"):
        raise SystemExit(f"polite wiki handoff should preserve raw preview and cleaned length: {handoff}")


def test_no_article_is_clean() -> None:
    # wiki_summary now falls back to real web search when Wikipedia has no
    # article (2026-07-07, the operator: "I want him to use the whole web instead").
    # Mock that fallback empty too, so this stays a deterministic, network-free
    # assertion instead of depending on live DuckDuckGo results for a made-up
    # query string.
    from jarvis_v2.tools import research_connector as rc

    wc._resolve_title = lambda query: None  # type: ignore
    original_ddg_search = rc.ddg_search
    rc.ddg_search = lambda query, max_results=5: []  # type: ignore
    try:
        out = _tools()["wiki_summary"].handler({"query": "asdkjfhqwe"})
    finally:
        rc.ddg_search = original_ddg_search
    if not out.ok or "No Wikipedia article found" not in out.output:
        raise SystemExit(f"missing article should be a clean message: {out.output}")
    _assert_wiki_handoff(
        out.metadata,
        "wiki no article",
        status="empty",
        reason="not_found",
        retry_safe=True,
    )


def test_no_summary_is_clean() -> None:
    # Empty extract now also triggers the web-search fallback; mock it empty
    # too so this stays deterministic (see test_no_article_is_clean).
    from jarvis_v2.tools import research_connector as rc

    wc._resolve_title = lambda query: "Empty Article"  # type: ignore
    wc._fetch_summary = lambda title: {"title": "Empty Article", "extract": ""}  # type: ignore
    original_ddg_search = rc.ddg_search
    rc.ddg_search = lambda query, max_results=5: []  # type: ignore
    try:
        out = _tools()["wiki_summary"].handler({"query": "empty article"})
    finally:
        rc.ddg_search = original_ddg_search
    if not out.ok or "No summary available" not in out.output:
        raise SystemExit(f"missing summary should be a clean message: {out.output}")
    handoff = _assert_wiki_handoff(
        out.metadata,
        "wiki no summary",
        status="empty",
        reason="no_summary",
        retry_safe=True,
    )
    if handoff.get("title") != "Empty Article":
        raise SystemExit(f"wiki no-summary should preserve resolved title: {handoff}")


def test_fetch_failure_preserves_bounded_exception_type() -> None:
    def offline(query):
        raise RuntimeError("offline")

    wc._resolve_title = offline  # type: ignore
    out = _tools()["wiki_summary"].handler({"query": "black holes"})
    if out.ok or not out.output.strip() or "offline" in out.output:
        raise SystemExit(f"wiki failure should stay friendly: {out.output}")
    if out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"wiki failure should preserve bounded exception type: {out.metadata}")
    _assert_wiki_recovery_message(out.output, "wiki generic fetch error")
    _assert_external_information_guidance(out, "wiki generic fetch error")
    _assert_wiki_handoff(
        out.metadata,
        "wiki failure",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )

    def server_error(query):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    wc._resolve_title = server_error  # type: ignore
    server_out = _tools()["wiki_summary"].handler({"query": "black holes"})
    if server_out.ok or server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"HTTP 500 should preserve bounded exception metadata: {server_out.output} {server_out.metadata}")
    _assert_wiki_recovery_message(server_out.output, "wiki HTTP 500")
    _assert_wiki_handoff(
        server_out.metadata,
        "wiki HTTP 500 fetch error",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )

    def not_found_error(query):
        raise HttpError(404, "HTTP Error 404: Not Found")

    wc._resolve_title = not_found_error  # type: ignore
    not_found_out = _tools()["wiki_summary"].handler({"query": "missing topic"})
    if (
        not_found_out.ok
        or "couldn't find" not in not_found_out.output
        or not_found_out.metadata.get("recovery_guidance") is not None
    ):
        raise SystemExit(
            "wiki HTTP 404 should stay a precise lookup miss without generic "
            f"transport recovery: {not_found_out.output} {not_found_out.metadata}"
        )
    for fragment in ("setup check", "network access"):
        if fragment in not_found_out.output.lower():
            raise SystemExit(
                f"wiki HTTP 404 should not include {fragment!r}: "
                f"{not_found_out.output}"
            )
    _assert_wiki_handoff(
        not_found_out.metadata,
        "wiki HTTP 404 lookup miss",
        status="unavailable",
        reason="not_found",
        retry_safe=True,
    )

    def timeout_error(query):
        raise TimeoutError("request timed out near /\x55sers/example/private")

    wc._resolve_title = timeout_error  # type: ignore
    timeout_out = _tools()["wiki_summary"].handler({"query": "black holes"})
    if timeout_out.ok or timeout_out.metadata.get("exception_type") != "TimeoutError":
        raise SystemExit(f"timeout should preserve bounded exception metadata: {timeout_out.output} {timeout_out.metadata}")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"timeout recovery should preserve timeout cause: {timeout_out.output}")
    _assert_wiki_recovery_message(timeout_out.output, "wiki timeout")
    _assert_wiki_handoff(
        timeout_out.metadata,
        "wiki timeout fetch error",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )


def test_requires_query() -> None:
    out = _tools()["wiki_summary"].handler({"query": ""})
    if out.ok or "wikipedia" not in out.output.lower():
        raise SystemExit(f"empty query should ask: {out.output}")
    if out.metadata.get("raw_query_length") != 0 or out.metadata.get("max_query_chars") != wc.MAX_QUERY_CHARS:
        raise SystemExit(f"empty query should preserve bounded metadata: {out.metadata}")
    _assert_wiki_handoff(
        out.metadata,
        "wiki missing query",
        status="refused",
        reason="missing_query",
        calls_external_service=False,
    )


def test_invalid_query_never_resolves() -> None:
    calls = []

    def fail_resolve(query):
        calls.append(query)
        raise AssertionError("invalid query should not resolve")

    wc._resolve_title = fail_resolve  # type: ignore
    for bad in [
        "---",
        "x" * (wc.MAX_QUERY_CHARS + 1),
        "/\x55sers/example/private/wiki-query",
        "/var/folders/zc/wiki-query",
        "/tmp/wiki-query",
    ]:
        out = _tools()["wiki_summary"].handler({"query": bad})
        if out.ok or "short wikipedia topic" not in out.output.lower():
            raise SystemExit(f"invalid query should be rejected locally: {bad!r} -> {out.output}")
        if out.metadata.get("reason") != "invalid_query":
            raise SystemExit(f"invalid query should include reason metadata: {out.metadata}")
        handoff = _assert_wiki_handoff(
            out.metadata,
            f"invalid query {bad!r}",
            status="refused",
            reason="invalid_query",
            calls_external_service=False,
        )
        if bad.startswith("/") and "<local-path>" not in handoff.get("query_preview", ""):
            raise SystemExit(f"invalid path query should be redacted in handoff: {handoff}")
    if calls:
        raise SystemExit(f"invalid query unexpectedly resolved: {calls}")


def test_planner_routes_wiki_not_specific_queries() -> None:
    p = RuleBasedPlanner()
    for q in [
        "who is Albert Einstein",
        "who is Albert Einstein please",
        "tell me about the Eiffel Tower",
        "tell me about Ada Lovelace please",
        "wikipedia black holes",
        "wikipedia Ada Lovelace please",
        "wiki Ada Lovelace",
        "wiki about Ada Lovelace",
        "wikipedia for Ada Lovelace",
        "wiki summary Ada Lovelace",
        "wikipedia summary Ada Lovelace",
        "summarize wiki Ada Lovelace",
        "summarize wikipedia Ada Lovelace",
        "summary of wikipedia Ada Lovelace",
        "search wikipedia for Ada Lovelace",
        "look up Ada Lovelace on wikipedia",
        "look up Ada Lovelace wiki",
        "Ada Lovelace wiki",
        "Ada Lovelace wikipedia",
        "Ada Lovelace wiki please",
        "what is photosynthesis",
        "what are black holes",
    ]:
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["wiki_summary"]:
            raise SystemExit(f"wiki route missed: {q!r}")
        query = actions[0].args.get("query", "")
        if query.lower().endswith("please"):
            raise SystemExit(f"planner should strip polite wiki suffix: {q!r} -> {actions[0].args}")
        if q.startswith("Ada Lovelace") and query != "Ada Lovelace":
            raise SystemExit(f"planner should strip suffix wiki marker: {q!r} -> {actions[0].args}")
        if "Ada Lovelace" in q and query != "Ada Lovelace":
            raise SystemExit(f"planner should pass clean wiki query: {q!r} -> {actions[0].args}")
    # specific queries must not be hijacked
    if [a.tool_name for a in p.plan("what's the weather").actions] != ["get_weather"]:
        raise SystemExit("weather wrongly routed to wiki")
    if [a.tool_name for a in p.plan("what time is it").actions] != ["current_time"]:
        raise SystemExit("time wrongly routed to wiki")
    if [a.tool_name for a in p.plan("what is bitcoin worth").actions] != ["get_crypto_price"]:
        raise SystemExit("crypto price wrongly routed to wiki")
    for q in [
        "what is on my schedule today",
        "what's on my agenda tomorrow",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["list_events"]:
            raise SystemExit(f"personal schedule query wrongly routed to wiki: {q!r}")
    for q in ["summarize Ada Lovelace", "summary of Ada Lovelace"]:
        if [a.tool_name for a in p.plan(q).actions] == ["wiki_summary"]:
            raise SystemExit(f"generic summary query should not be forced to wiki: {q!r}")


def _mock_research(*, ddg_results=None, page_texts=None, synth_answer=""):
    """Context manager-ish helper: patches research_connector's ddg_search /
    _fetch_page_text / _synthesize (the same pipeline wiki_summary's web
    fallback reuses) and returns the originals for restoration."""
    from jarvis_v2.tools import research_connector as rc

    originals = (rc.ddg_search, rc._fetch_page_text, rc._synthesize)
    rc.ddg_search = lambda query, max_results=5: ddg_results or []  # type: ignore
    rc._fetch_page_text = lambda url: (page_texts or {}).get(url, "")  # type: ignore
    rc._synthesize = lambda config, query, corpus: synth_answer  # type: ignore
    return originals


def _restore_research(originals) -> None:
    from jarvis_v2.tools import research_connector as rc

    rc.ddg_search, rc._fetch_page_text, rc._synthesize = originals


def test_title_matches_query_word_overlap() -> None:
    # Real bug found 2026-07-06: opensearch("Anthropic CEO") once resolved to
    # "Claude (AI)" -- zero shared words, confidently wrong. Word overlap must
    # catch that miss while still trusting genuine encyclopedic matches.
    good_pairs = [
        ("Ada Lovelace", "Ada Lovelace"),
        ("black holes", "Black hole"),
        ("what is the eiffel tower", "Eiffel Tower"),
        ("photosynthesis", "Photosynthesis"),
    ]
    for query, title in good_pairs:
        if not wc._title_matches_query(query, title):
            raise SystemExit(f"should trust a genuine match: {query!r} -> {title!r}")
    if wc._title_matches_query("Anthropic CEO", "Claude (AI)"):
        raise SystemExit("should reject a title sharing zero words with the query")


def test_relational_role_query_detection() -> None:
    should_trigger = [
        "who is the CEO of Anthropic",
        "who is Anthropic CEO",
        "who is the president of France",
        "who is the founder of Tesla",
        "who is the current prime minister of Japan",
        "king of Spain",
        "who founded Microsoft",
        "who leads Google",
        "who runs OpenAI",
        "who owns Twitter",
        "Secretary of State",
        "the CEO of OpenAI",
    ]
    for q in should_trigger:
        if not wc._is_relational_role_query(q):
            raise SystemExit(f"should detect a relational role query: {q!r}")
    should_not_trigger = [
        "Ada Lovelace",
        "black holes",
        "photosynthesis",
        "the eiffel tower",
        "a king",
        "conductor",
        "what is a chief",
    ]
    for q in should_not_trigger:
        if wc._is_relational_role_query(q):
            raise SystemExit(f"should NOT flag a plain standalone-role/definitional query: {q!r}")
    # Known, accepted tradeoff: a role word immediately followed by what looks
    # like an entity name can't be cheaply distinguished from a role word that
    # is itself PART of a compound proper-noun title ("Secretary bird",
    # "Captain America", "Chief Joseph"). These over-trigger the web fallback
    # instead of using the faster Wikipedia path -- confirmed live (see
    # CODEX_TASKS 2026-07-07) that the web fallback still answers them
    # completely correctly, just slower. Documenting the current behavior
    # here rather than silently letting it drift.
    for q in ("Secretary bird", "Captain America", "Chief Joseph"):
        if not wc._is_relational_role_query(q):
            raise SystemExit(
                f"expected the known compound-name-vs-relational tradeoff to still over-trigger for {q!r} "
                "-- if this now returns False, the heuristic changed; update this test's expectation "
                "deliberately rather than assume it's a bug"
            )


def test_relational_role_query_skips_wikipedia_entirely() -> None:
    def _boom(query):
        raise AssertionError("relational role queries must not call Wikipedia opensearch at all")

    original_resolve = wc._resolve_title
    wc._resolve_title = _boom  # type: ignore
    originals = _mock_research(
        ddg_results=[{"title": "Dario Amodei - Wikipedia", "url": "https://en.wikipedia.org/wiki/Dario_Amodei", "snippet": "CEO of Anthropic"}],
        page_texts={"https://en.wikipedia.org/wiki/Dario_Amodei": "Dario Amodei is the CEO and co-founder of Anthropic."},
        synth_answer="Dario Amodei is the CEO of Anthropic.",
    )
    try:
        out = _tools()["wiki_summary"].handler({"query": "who is Anthropic CEO"})
    finally:
        wc._resolve_title = original_resolve
        _restore_research(originals)
    if not out.ok or "Dario Amodei" not in out.output:
        raise SystemExit(f"relational role query should use the real web answer: {out.output}")
    if "Sources:" not in out.output:
        raise SystemExit(f"web fallback answer should cite sources: {out.output}")
    handoff = out.metadata.get("wiki_summary_handoff") or {}
    if handoff.get("source") != "web_fallback":
        raise SystemExit(f"handoff should record the web fallback source: {handoff}")


def test_mismatched_title_falls_back_to_web() -> None:
    wc._resolve_title = lambda query: "Completely Unrelated Article"  # type: ignore
    originals = _mock_research(
        ddg_results=[{"title": "Real Answer Page", "url": "https://example.com/a", "snippet": "the real answer"}],
        page_texts={"https://example.com/a": "banana phone conspiracy theories explained in detail"},
        synth_answer="Here is the real answer about the banana phone conspiracy.",
    )
    try:
        out = _tools()["wiki_summary"].handler({"query": "banana phone conspiracy"})
    finally:
        _restore_research(originals)
    if not out.ok or "banana phone conspiracy" not in out.output.lower():
        raise SystemExit(f"mismatched wiki title should fall back to the real web answer: {out.output}")
    if "Completely Unrelated Article" in out.output:
        raise SystemExit(f"the mismatched wiki title must not leak into the answer: {out.output}")


def test_corpus_relevance_check() -> None:
    if not wc._corpus_is_relevant("black holes", "an article about black holes and gravity"):
        raise SystemExit("relevant corpus should pass")
    if wc._corpus_is_relevant("asdkjfhqwe", "an unrelated article about German composers"):
        raise SystemExit("irrelevant corpus should be rejected")


def test_web_fallback_none_when_ddg_empty() -> None:
    wc._resolve_title = lambda query: None  # type: ignore
    originals = _mock_research(ddg_results=[])
    try:
        out = _tools()["wiki_summary"].handler({"query": "totally nonexistent query xyz"})
    finally:
        _restore_research(originals)
    if not out.ok or "No Wikipedia article found" not in out.output:
        raise SystemExit(f"empty web search should still show the clean not-found message: {out.output}")
    handoff = out.metadata.get("wiki_summary_handoff") or {}
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("calls_model") or boundaries.get("executes_tools"):
        raise SystemExit(f"no model/synthesis call should have happened on empty search: {boundaries}")


def main() -> None:
    test_is_local_safe()
    test_wiki_summary_argument_contract_fences_malformed_input()
    test_summarizes_topic()
    test_strips_polite_suffix_from_query()
    test_no_article_is_clean()
    test_no_summary_is_clean()
    test_fetch_failure_preserves_bounded_exception_type()
    test_requires_query()
    test_invalid_query_never_resolves()
    test_planner_routes_wiki_not_specific_queries()
    test_title_matches_query_word_overlap()
    test_relational_role_query_detection()
    test_relational_role_query_skips_wikipedia_entirely()
    test_mismatched_title_falls_back_to_web()
    test_corpus_relevance_check()
    test_web_fallback_none_when_ddg_empty()
    print("Wikipedia connector smoke passed")


if __name__ == "__main__":
    main()
