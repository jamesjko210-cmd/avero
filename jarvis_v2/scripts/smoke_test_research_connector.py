"""Smoke tests for the research/web-lookup connector (mocked fetch, no network)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import research_connector as rc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry


_LITE_HTML = """
<table>
<a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FUlaanbaatar&rut=x" class='result-link'>Ulaanbaatar - Wikipedia</a>
<td class='result-snippet'>Ulaanbaatar is the capital of Mongolia.</td>
<a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.britannica.com%2Fplace%2FUlaanbaatar&rut=y" class='result-link'>Ulaanbaatar | Britannica</a>
<td class='result-snippet'>The capital and largest city of Mongolia.</td>
</table>
"""


def _tools():
    return {t.name: t for t in rc.make_research_tools(load_config())}


def _assert_search_recovery_message(output: str, label: str) -> None:
    lower = output.lower()
    for expected in ["network access to duckduckgo lite", "setup check", "retry in a moment"]:
        if expected not in lower:
            raise SystemExit(f"{label} should name actionable search recovery step {expected!r}: {output}")
    for forbidden in [
        "raw search backend exploded",
        "raw research backend exploded",
        "http error",
        "offline",
        "traceback",
        "/users/",
        "/private/",
        "/var/folders/",
        "/tmp/",
    ]:
        if forbidden in lower:
            raise SystemExit(f"{label} leaked raw/backend text {forbidden!r}: {output}")


def _assert_lookup_handoff(
    metadata: dict[str, Any],
    tool_name: str,
    label: str,
    *,
    status: str,
    reason: str = "",
    calls_external_service: bool = True,
    calls_model: bool = False,
    executes_tools: bool = False,
    retry_safe: bool = False,
) -> dict[str, Any]:
    if metadata.get(f"{tool_name}_handoff_ready") is not True:
        raise SystemExit(f"{label} missing flat handoff readiness: {metadata}")
    for key, expected_value in [
        ("ready_for_operator", True),
        ("state_changed", False),
        ("content_in_handoff", False),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ]:
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
    if metadata.get("changed") != []:
        raise SystemExit(f"{label} flat changed should be empty: {metadata}")
    handoff = metadata.get(f"{tool_name}_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing {tool_name}_handoff: {metadata}")
    if handoff.get("source") != tool_name or handoff.get("status") != status:
        raise SystemExit(f"{label} has wrong source/status: {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} nested handoff readiness missing: {handoff}")
    if metadata.get(f"{tool_name}_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} ready alias parity failed: {metadata} vs {handoff}")
    if metadata.get(f"{tool_name}_state_changed") != handoff.get("state_changed"):
        raise SystemExit(f"{label} state alias parity failed: {metadata} vs {handoff}")
    if metadata.get(f"{tool_name}_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} changed alias parity failed: {metadata} vs {handoff}")
    if metadata.get(f"{tool_name}_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} content alias parity failed: {metadata} vs {handoff}")
    for key in [
        "ready_for_operator",
        "state_changed",
        "changed",
        "content_in_handoff",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} should mirror {key}: {handoff} vs {metadata}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} has wrong reason: {handoff}")
    for key in ("query_length", "raw_query_length", "max_query_chars", "query_truncated", "exception_type"):
        if handoff.get(key) != metadata.get(key, ""):
            raise SystemExit(f"{label} should mirror {key}: {handoff} vs {metadata}")
    if "count" in metadata and handoff.get("result_count") != metadata.get("count"):
        raise SystemExit(f"{label} should mirror result count: {handoff} vs {metadata}")
    if handoff.get("answer_in_metadata") is not False:
        raise SystemExit(f"{label} should exclude synthesized answers from metadata: {handoff}")
    if handoff.get("snippet_in_metadata") is not False:
        raise SystemExit(f"{label} should exclude snippets from metadata: {handoff}")
    if handoff.get("page_text_in_metadata") is not False:
        raise SystemExit(f"{label} should exclude page text from metadata: {handoff}")
    if handoff.get("retry_safe") is not retry_safe:
        raise SystemExit(f"{label} has wrong retry safety: {handoff}")
    expected_next = "research <topic>" if tool_name == "research" else "look up <topic>"
    if handoff.get("next_safe_command") != expected_next:
        raise SystemExit(f"{label} has wrong next safe command: {handoff}")
    expected_commands = [expected_next]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} safe command list wrong: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} safe command count wrong: {handoff}")
    if metadata.get(f"{tool_name}_next_safe_command") != expected_next:
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get(f"{tool_name}_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get(f"{tool_name}_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} next-command count alias parity failed: {metadata} vs {handoff}")
    boundaries = handoff.get("boundaries")
    expected = {
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
    if boundaries != expected:
        raise SystemExit(f"{label} has wrong boundaries: {boundaries}")
    if metadata.get(f"{tool_name}_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary alias parity failed: {metadata} vs {handoff}")
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        alias = f"{tool_name}_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    for key, expected_value in [
        ("calls_external_service", calls_external_service),
        ("calls_model", calls_model),
        ("executes_tools", executes_tools),
    ]:
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should match handoff: {metadata}")
    return handoff


def test_risk_levels() -> None:
    tools = _tools()
    if tools["web_lookup"].risk != RiskLevel.READ_ONLY:
        raise SystemExit("web_lookup should be READ_ONLY")
    if tools["research"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("research should be LOCAL_SAFE")


def test_research_query_argument_contracts_fence_malformed_input() -> None:
    search_calls: list[str] = []
    original_get = rc._get

    def mocked_get(url: str) -> str:
        search_calls.append(url)
        return _LITE_HTML

    try:
        rc._get = mocked_get  # type: ignore[assignment]
        private_sentinel = "private-lookup-contract-sentinel"
        malformed_args: list[object] = [
            None,
            ["not", "an", "object"],
            {"query": {"private": private_sentinel}},
            {"query": True},
            {"query": [private_sentinel]},
            {"text": 7},
            {"query": "valid", "unexpected": private_sentinel},
        ]
        rejected_surfaces: list[object] = []
        for tool_name in ("web_lookup", "research"):
            tool = _tools()[tool_name]
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
                raise SystemExit(f"{tool_name} should have an exact strict query/text contract: {contract}")

            registry = ToolRegistry()
            registry.register(tool)
            executor = Executor(registry, PermissionPolicy())
            for args in malformed_args:
                result = executor.execute(PlannedAction(tool_name, args, f"{tool_name} contract smoke"))  # type: ignore[arg-type]
                if (
                    result.ok
                    or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or result.metadata.get("handler_invoked") is not False
                    or result.metadata.get("executed_handler") is not False
                    or result.metadata.get("requires_confirmation") is not False
                ):
                    raise SystemExit(f"malformed {tool_name} input passed the executor: {result}")
                rejected_surfaces.append({"output": result.output, "metadata": result.metadata})
        if search_calls:
            raise SystemExit(f"malformed research-query input reached DuckDuckGo: {search_calls}")
        if private_sentinel in str(rejected_surfaces):
            raise SystemExit("research-query argument rejection leaked a supplied private value")

        lookup_registry = ToolRegistry()
        lookup_registry.register(_tools()["web_lookup"])
        executor = Executor(lookup_registry, PermissionPolicy())

        for args, expected_query in [
            ({"query": "capital of mongolia"}, "capital+of+mongolia"),
            ({"text": "capital of mongolia"}, "capital+of+mongolia"),
        ]:
            result = executor.execute(PlannedAction("web_lookup", args, "valid web lookup contract smoke"))
            if not result.ok:
                raise SystemExit(f"valid web_lookup input was rejected: {args} {result}")
            if expected_query not in search_calls[-1]:
                raise SystemExit(f"valid web_lookup input did not preserve the query alias: {args} {search_calls}")
        empty = executor.execute(PlannedAction("web_lookup", {}, "empty web lookup contract smoke"))
        if (
            empty.ok
            or empty.metadata.get("handler_invoked") is not True
            or "what should i look up" not in empty.output.lower()
        ):
            raise SystemExit(f"empty web_lookup compatibility changed: {empty}")

        research_registry = ToolRegistry()
        research_registry.register(_tools()["research"])
        research_empty = Executor(research_registry, PermissionPolicy()).execute(
            PlannedAction("research", {}, "empty research contract smoke")
        )
        if (
            research_empty.ok
            or research_empty.metadata.get("handler_invoked") is not True
            or "what should i research" not in research_empty.output.lower()
        ):
            raise SystemExit(f"empty research compatibility changed: {research_empty}")
    finally:
        rc._get = original_get  # type: ignore[assignment]


def test_ddg_parse_decodes_urls() -> None:
    rc._get = lambda url: _LITE_HTML  # type: ignore
    results = rc.ddg_search("capital of mongolia", 5)
    if len(results) != 2:
        raise SystemExit(f"expected 2 results: {results}")
    if results[0]["url"] != "https://en.wikipedia.org/wiki/Ulaanbaatar":
        raise SystemExit(f"url not decoded: {results[0]}")
    if results[0]["title"] != "Ulaanbaatar - Wikipedia" or "capital of Mongolia" not in results[0]["snippet"]:
        raise SystemExit(f"title/snippet wrong: {results[0]}")


def test_web_lookup_lists_results() -> None:
    rc._get = lambda url: _LITE_HTML  # type: ignore
    out = _tools()["web_lookup"].handler({"query": "capital of mongolia"})
    if not out.ok or "Ulaanbaatar - Wikipedia" not in out.output or "en.wikipedia.org" not in out.output:
        raise SystemExit(f"web_lookup wrong: {out.output}")
    if out.metadata.get("query_length") != len("capital of mongolia"):
        raise SystemExit(f"web_lookup should preserve bounded query metadata: {out.metadata}")
    handoff = _assert_lookup_handoff(out.metadata, "web_lookup", "web lookup success", status="ok")
    if handoff.get("result_count") != 2 or handoff.get("result_row_count") != 2:
        raise SystemExit(f"web lookup should preserve bounded result rows: {handoff}")
    if "capital of Mongolia" in str(handoff):
        raise SystemExit(f"web lookup handoff should not copy snippets: {handoff}")


def test_web_lookup_failure_is_clean() -> None:
    def boom(_url):
        raise RuntimeError("raw search backend exploded")

    rc._get = boom  # type: ignore
    out = _tools()["web_lookup"].handler({"query": "capital of mongolia"})
    if out.ok or "Search is having trouble" not in out.output:
        raise SystemExit(f"web_lookup failure should be friendly: {out.output}")
    if "raw search backend exploded" in out.output or "Search error" in out.output:
        raise SystemExit(f"web_lookup failure should not leak raw exceptions: {out.output}")
    if out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"web_lookup failure should preserve bounded diagnostic metadata: {out.metadata}")
    _assert_search_recovery_message(out.output, "web lookup generic fetch error")
    _assert_lookup_handoff(
        out.metadata,
        "web_lookup",
        "web lookup failure",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )
    def server_error(_url):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    rc._get = server_error  # type: ignore
    server_out = _tools()["web_lookup"].handler({"query": "capital of mongolia"})
    if server_out.ok or server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"web_lookup HTTP 500 should preserve bounded metadata: {server_out.output} {server_out.metadata}")
    _assert_search_recovery_message(server_out.output, "web lookup HTTP 500")
    _assert_lookup_handoff(
        server_out.metadata,
        "web_lookup",
        "web lookup HTTP 500",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )
    def timeout(_url):
        raise TimeoutError("request timed out near /\x55sers/example/private")

    rc._get = timeout  # type: ignore
    timeout_out = _tools()["web_lookup"].handler({"query": "capital of mongolia"})
    if timeout_out.ok or timeout_out.metadata.get("exception_type") != "TimeoutError":
        raise SystemExit(f"web_lookup timeout should preserve bounded metadata: {timeout_out.output} {timeout_out.metadata}")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"web_lookup timeout should keep timeout wording: {timeout_out.output}")
    _assert_search_recovery_message(timeout_out.output, "web lookup timeout")
    _assert_lookup_handoff(
        timeout_out.metadata,
        "web_lookup",
        "web lookup timeout",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )


def test_empty_queries_never_search() -> None:
    calls = []
    original_search = rc.ddg_search

    def fail_search(query, max_results=5):
        calls.append(query)
        raise AssertionError("empty query should not search")

    try:
        rc.ddg_search = fail_search  # type: ignore
        for tool_name, prompt in [("web_lookup", "look up"), ("research", "research")]:
            out = _tools()[tool_name].handler({"query": ""})
            if out.ok or prompt not in out.output.lower():
                raise SystemExit(f"{tool_name} should ask for a query: {out.output}")
            _assert_lookup_handoff(
                out.metadata,
                tool_name,
                f"{tool_name} empty query",
                status="refused",
                reason="missing_query",
                calls_external_service=False,
            )
        if calls:
            raise SystemExit(f"empty query unexpectedly searched: {calls}")
    finally:
        rc.ddg_search = original_search  # type: ignore


def test_invalid_queries_never_search() -> None:
    calls = []
    original_search = rc.ddg_search
    original_fetch = rc._fetch_page_text
    original_synthesize = rc._synthesize

    def fail_search(query, max_results=5):
        calls.append(("search", query))
        raise AssertionError("invalid query should not search")

    def fail_fetch(url):
        calls.append(("fetch", url))
        raise AssertionError("invalid query should not fetch pages")

    def fail_synthesize(config, query, corpus):
        calls.append(("synthesize", query))
        raise AssertionError("invalid query should not synthesize")

    try:
        rc.ddg_search = fail_search  # type: ignore
        rc._fetch_page_text = fail_fetch  # type: ignore
        rc._synthesize = fail_synthesize  # type: ignore
        cases = [
            ("web_lookup", "---", "short search query", False),
            ("web_lookup", "x" * (rc.MAX_QUERY_CHARS + 1), "short search query", False),
            ("web_lookup", "/\x55sers/example/private/research-query", "not a local file path", True),
            ("web_lookup", "/var/folders/zc/jarvis/research-query", "not a local file path", True),
            ("research", "---", "short research query", False),
            ("research", "x" * (rc.MAX_QUERY_CHARS + 1), "short research query", False),
            ("research", "/private/tmp/jarvis/research-query", "not a local file path", True),
            ("research", "/tmp/jarvis/research-query", "not a local file path", True),
        ]
        for tool_name, query, expected, is_local_path in cases:
            out = _tools()[tool_name].handler({"query": query})
            if out.ok or expected not in out.output.lower():
                raise SystemExit(f"{tool_name} should reject invalid query locally: {query!r} -> {out.output}")
            if out.metadata.get("reason") != "invalid_query":
                raise SystemExit(f"{tool_name} should include invalid query reason: {out.metadata}")
            if bool(out.metadata.get("local_path_query")) != is_local_path:
                raise SystemExit(f"{tool_name} should mark only local-path queries: {out.metadata}")
            if "/\x55sers/" in out.output or "/private/" in out.output:
                raise SystemExit(f"{tool_name} should not echo local paths: {out.output}")
            handoff = _assert_lookup_handoff(
                out.metadata,
                tool_name,
                f"{tool_name} invalid query {query!r}",
                status="refused",
                reason="invalid_query",
                calls_external_service=False,
            )
            if is_local_path and "<local-path>" not in handoff.get("query_preview", ""):
                raise SystemExit(f"{tool_name} should redact local path in handoff preview: {handoff}")
        if calls:
            raise SystemExit(f"invalid query unexpectedly left local validation: {calls}")
    finally:
        rc.ddg_search = original_search  # type: ignore
        rc._fetch_page_text = original_fetch  # type: ignore
        rc._synthesize = original_synthesize  # type: ignore


def test_research_synthesizes_with_sources() -> None:
    rc._get = lambda url: _LITE_HTML  # type: ignore
    rc._fetch_page_text = lambda url: "Ulaanbaatar is the capital and most populous city of Mongolia."  # type: ignore
    rc._synthesize = lambda config, query, corpus: "Ulaanbaatar is Mongolia's capital."  # type: ignore
    out = _tools()["research"].handler({"query": "capital of mongolia"})
    if not out.ok or "Ulaanbaatar is Mongolia's capital." not in out.output:
        raise SystemExit(f"research synthesis missing: {out.output}")
    if "Sources:" not in out.output or "wikipedia.org" not in out.output:
        raise SystemExit(f"research sources missing: {out.output}")
    if not out.metadata.get("synthesized"):
        raise SystemExit("research should mark synthesized when model answers")
    handoff = _assert_lookup_handoff(
        out.metadata,
        "research",
        "research synthesized",
        status="ok",
        calls_model=True,
        executes_tools=True,
    )
    if handoff.get("synthesized") is not True or handoff.get("page_read_count") != 2:
        raise SystemExit(f"research synthesized should preserve model/tool state: {handoff}")
    if "Mongolia's capital" in str(handoff) or "most populous city" in str(handoff):
        raise SystemExit(f"research handoff should not copy answer/page text: {handoff}")


def test_openai_research_discloses_query_and_source_sharing() -> None:
    config = replace(
        load_config(),
        model_provider="openai",
        chat_model="gpt-5.6-terra",
    )
    rc._get = lambda url: _LITE_HTML  # type: ignore
    rc._fetch_page_text = lambda url: "Ulaanbaatar is the capital and most populous city of Mongolia."  # type: ignore
    rc._synthesize = lambda _config, _query, _corpus: "Ulaanbaatar is Mongolia's capital."  # type: ignore
    tools = {tool.name: tool for tool in rc.make_research_tools(config)}
    out = tools["research"].handler({"query": "capital of mongolia"})
    if not out.ok or out.metadata.get("model_provider") != "openai":
        raise SystemExit(f"OpenAI research missed provider receipt: {out}")
    for key in (
        "model_call_attempted",
        "external_model_call_attempted",
        "shares_research_query_with_external_model",
        "shares_research_sources_with_external_model",
    ):
        if out.metadata.get(key) is not True:
            raise SystemExit(f"OpenAI research missed disclosure {key}: {out.metadata}")
    for key in ("model_source_corpus_in_metadata", "model_response_content_in_metadata"):
        if out.metadata.get(key) is not False:
            raise SystemExit(f"OpenAI research should keep {key}=False: {out.metadata}")
    handoff = out.metadata.get("research_handoff") or {}
    if handoff.get("model_provider") != "openai" or handoff.get("external_model_call_attempted") is not True:
        raise SystemExit(f"OpenAI research handoff missed remote model receipt: {handoff}")
    if "most populous city" in str(handoff) or "Mongolia's capital" in str(handoff):
        raise SystemExit(f"OpenAI research handoff copied source/answer content: {handoff}")


def test_research_falls_back_without_model() -> None:
    rc._get = lambda url: _LITE_HTML  # type: ignore
    rc._fetch_page_text = lambda url: ""  # type: ignore
    rc._synthesize = lambda config, query, corpus: ""  # model unavailable
    out = _tools()["research"].handler({"query": "capital of mongolia"})
    if not out.ok or "Ulaanbaatar" not in out.output or "Sources:" not in out.output:
        raise SystemExit(f"research fallback wrong: {out.output}")
    if out.metadata.get("synthesized"):
        raise SystemExit("fallback must not claim synthesized")
    handoff = _assert_lookup_handoff(
        out.metadata,
        "research",
        "research fallback",
        status="ok",
        calls_model=True,
        executes_tools=True,
    )
    if handoff.get("synthesized") is not False:
        raise SystemExit(f"research fallback should preserve unsynthesized state: {handoff}")


def test_research_failure_is_clean() -> None:
    def boom(_query, max_results=5):
        raise RuntimeError("raw research backend exploded")

    rc.ddg_search = boom  # type: ignore
    out = _tools()["research"].handler({"query": "capital of mongolia"})
    if out.ok or "Research is having trouble" not in out.output:
        raise SystemExit(f"research failure should be friendly: {out.output}")
    if "raw research backend exploded" in out.output or "Research error" in out.output:
        raise SystemExit(f"research failure should not leak raw exceptions: {out.output}")
    if out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"research failure should preserve bounded diagnostic metadata: {out.metadata}")
    _assert_search_recovery_message(out.output, "research generic fetch error")
    _assert_lookup_handoff(
        out.metadata,
        "research",
        "research failure",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )
    def server_error(_query, max_results=5):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    rc.ddg_search = server_error  # type: ignore
    server_out = _tools()["research"].handler({"query": "capital of mongolia"})
    if server_out.ok or server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"research HTTP 500 should preserve bounded metadata: {server_out.output} {server_out.metadata}")
    _assert_search_recovery_message(server_out.output, "research HTTP 500")
    _assert_lookup_handoff(
        server_out.metadata,
        "research",
        "research HTTP 500",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )
    def timeout(_query, max_results=5):
        raise TimeoutError("request timed out near /private/tmp/research")

    rc.ddg_search = timeout  # type: ignore
    timeout_out = _tools()["research"].handler({"query": "capital of mongolia"})
    if timeout_out.ok or timeout_out.metadata.get("exception_type") != "TimeoutError":
        raise SystemExit(f"research timeout should preserve bounded metadata: {timeout_out.output} {timeout_out.metadata}")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"research timeout should keep timeout wording: {timeout_out.output}")
    _assert_search_recovery_message(timeout_out.output, "research timeout")
    _assert_lookup_handoff(
        timeout_out.metadata,
        "research",
        "research timeout",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )


def test_no_results_are_clean() -> None:
    rc.ddg_search = lambda query, max_results=5: []  # type: ignore
    for tool_name in ["web_lookup", "research"]:
        out = _tools()[tool_name].handler({"query": "zzzz no results"})
        if not out.ok or "No results found" not in out.output:
            raise SystemExit(f"{tool_name} empty results should be clean: {out.output}")
        _assert_lookup_handoff(
            out.metadata,
            tool_name,
            f"{tool_name} no results",
            status="empty",
            reason="no_results",
            retry_safe=True,
        )


def test_irrelevant_results_are_clean() -> None:
    # Found 2026-07-06: a nonsense query can still surface a coincidentally-
    # ranked DDG hit (a real fixed test string once verbatim-matched bot-spam
    # text on an unrelated blog). Synthesizing an answer from a page that
    # shares zero words with the query produces a confusing "I couldn't find
    # that" disclaimer next to an irrelevant source -- research() must treat
    # this the same as finding nothing, not show that non-answer as an answer.
    rc.ddg_search = lambda query, max_results=5: [
        {"title": "Unrelated Blog Post", "url": "https://example.com/unrelated", "snippet": "something else entirely"}
    ]  # type: ignore
    rc._fetch_page_text = lambda url: "a biography of a german composer with no connection to the query"  # type: ignore
    rc._synthesize = lambda config, query, corpus: "I couldn't find anything about that in the sources."  # type: ignore
    out = _tools()["research"].handler({"query": "xqzplmvwrkt"})
    if not out.ok or "No relevant results found" not in out.output:
        raise SystemExit(f"irrelevant results should read as clean/empty, not a confusing non-answer: {out.output}")
    if "Unrelated Blog Post" in out.output or "couldn't find anything" in out.output:
        raise SystemExit(f"irrelevant source/disclaimer must not leak into the reply: {out.output}")
    _assert_lookup_handoff(
        out.metadata,
        "research",
        "research irrelevant results",
        status="empty",
        reason="irrelevant_results",
        retry_safe=True,
    )


def test_corpus_relevance_shared_with_wikipedia_connector() -> None:
    # Both connectors must judge relevance identically -- consolidated into
    # research_connector.corpus_is_relevant/significant_words 2026-07-07 so
    # they can't silently drift apart.
    from jarvis_v2.tools import wikipedia_connector as wc

    if wc._corpus_is_relevant is not rc.corpus_is_relevant:
        raise SystemExit("wikipedia_connector must reuse research_connector's corpus_is_relevant, not its own copy")
    if wc._significant_words is not rc.significant_words:
        raise SystemExit("wikipedia_connector must reuse research_connector's significant_words, not its own copy")
    if not rc.corpus_is_relevant("black holes", "an article about black holes and gravity"):
        raise SystemExit("relevant corpus should pass")
    if rc.corpus_is_relevant("xqzplmvwrkt", "an unrelated article about German composers"):
        raise SystemExit("irrelevant corpus should be rejected")


def test_planner_routes_research_and_lookup() -> None:
    p = RuleBasedPlanner()
    if [a.tool_name for a in p.plan("research the latest on NVIDIA earnings").actions] != ["research"]:
        raise SystemExit("research route missed")
    read_up_actions = p.plan("read up ada lovelace").actions
    if [a.tool_name for a in read_up_actions] != ["research"] or read_up_actions[0].args != {"query": "ada lovelace"}:
        raise SystemExit(f"read-up research route missed: {read_up_actions}")
    for q in [
        "research ada lovelace please",
        "research about ada lovelace please",
        "deep dive on ada lovelace please",
        "look into ada lovelace please",
        "read up on ada lovelace please",
    ]:
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["research"] or actions[0].args != {"query": "ada lovelace"}:
            raise SystemExit(f"polite research route should clean query: {q!r} -> {actions}")
    if [a.tool_name for a in p.plan("look up the capital of Mongolia").actions] != ["web_lookup"]:
        raise SystemExit("web_lookup route missed")
    for q, expected_query in {
        "lookup ada lovelace": "ada lovelace",
        "lookup ada lovelace please": "ada lovelace",
        "look up ada lovelace please": "ada lovelace",
        "look up the capital of Mongolia please": "the capital of Mongolia",
        "search ada lovelace": "ada lovelace",
        "search ada lovelace please": "ada lovelace",
        "search the web ada lovelace": "ada lovelace",
        "search the web for ada lovelace please": "ada lovelace",
        "find information about ada lovelace please": "ada lovelace",
    }.items():
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["web_lookup"] or actions[0].args != {"query": expected_query}:
            raise SystemExit(f"casual web lookup route missed: {q!r} -> {actions}")
    for q in ["web search ada lovelace please", "google ada lovelace please"]:
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["web_search"] or actions[0].args != {"query": "ada lovelace"}:
            raise SystemExit(f"polite web_search route should clean query: {q!r} -> {actions}")
    if [a.tool_name for a in p.plan("search files for invoice").actions] != ["find_files"]:
        raise SystemExit("casual web lookup route should not hijack file search")
    if [a.tool_name for a in p.plan("search notes for meeting").actions] != ["search_jarvis_notes"]:
        raise SystemExit("casual web lookup route should not hijack notes search")
    if p.plan("search my files for invoice").actions:
        raise SystemExit("casual web lookup route should not hijack unsupported personal file search")
    if [a.tool_name for a in p.plan("lookup contact fixture").actions] != ["find_contact"]:
        raise SystemExit("bare lookup route should not hijack contact lookup")
    for q in ["lookup contacts fixture", "lookup file invoice", "lookup calendar tomorrow", "lookup email receipt"]:
        actions = p.plan(q).actions
        if actions and actions[0].tool_name == "web_lookup":
            raise SystemExit(f"bare lookup route should not hijack reserved lookup phrase: {q!r} -> {actions}")


def main() -> None:
    test_risk_levels()
    test_research_query_argument_contracts_fence_malformed_input()
    test_ddg_parse_decodes_urls()
    test_web_lookup_lists_results()
    test_web_lookup_failure_is_clean()
    test_empty_queries_never_search()
    test_invalid_queries_never_search()
    test_research_synthesizes_with_sources()
    test_openai_research_discloses_query_and_source_sharing()
    test_research_falls_back_without_model()
    test_research_failure_is_clean()
    test_no_results_are_clean()
    test_irrelevant_results_are_clean()
    test_corpus_relevance_shared_with_wikipedia_connector()
    test_planner_routes_research_and_lookup()
    print("Research connector smoke passed")


if __name__ == "__main__":
    main()
