"""Smoke tests for the news connector (mocked RSS fetch, no network)."""

from __future__ import annotations

import os

from jarvis_v2.agent.failure_guidance import EXTERNAL_INFORMATION_RECOVERY_ACTION
from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import news_connector as nc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry


_RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>Markets rally on rate news - Reuters</title></item>
  <item><title>Local team wins championship - ESPN</title></item>
  <item><title>New phone announced - The Verge</title></item>
</channel></rss>"""
NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


def _tools():
    return {t.name: t for t in nc.make_news_tools(load_config())}


def assert_no_raw_local_path(value: object, label: str) -> None:
    text = str(value)
    if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"{label} leaked a raw local path: {value}")


def assert_news_recovery_message(output: str, label: str) -> None:
    lowered = output.lower()
    for fragment in [
        "network access to google news",
        "setup check",
        "retry in a moment",
    ]:
        if fragment not in lowered:
            raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")
    for forbidden in ["http error", "offline", "traceback", "/users/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in lowered:
            raise SystemExit(f"{label} leaked raw backend text {forbidden!r}: {output}")


def assert_external_information_guidance(result, label: str) -> None:
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


def assert_news_handoff(metadata: dict, label: str, *, status: str, calls_external_service: bool) -> dict:
    handoff = metadata.get("news_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed news handoff: {metadata}")
    if handoff.get("source") != "get_news":
        raise SystemExit(f"{label} handoff source wrong: {handoff}")
    if metadata.get("news_handoff_ready") is not True or handoff.get("news_handoff_ready") is not True:
        raise SystemExit(f"{label} should expose flat and nested handoff readiness: {metadata} vs {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should expose nested handoff_ready: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} handoff should be operator-ready: {handoff}")
    if metadata.get("news_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} ready alias parity failed: {metadata} vs {handoff}")
    if handoff.get("status") != status:
        raise SystemExit(f"{label} handoff status wrong: {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} handoff should not claim changed state: {handoff}")
    if metadata.get("news_state_changed") != handoff.get("state_changed") or metadata.get("news_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} state alias parity failed: {metadata} vs {handoff}")
    if handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} handoff should declare no output content payload: {handoff}")
    if metadata.get("news_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} content-in-handoff alias parity failed: {metadata} vs {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
        alias = f"news_{key}"
        if metadata.get(alias) is not expected_value or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if handoff.get("query") != metadata.get("query"):
        raise SystemExit(f"{label} handoff query parity failed: {metadata}")
    if handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} handoff limit parity failed: {metadata}")
    if handoff.get("headline_count") != metadata.get("count", 0):
        raise SystemExit(f"{label} handoff count parity failed: {metadata}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} handoff should declare bounded metadata content: {handoff}")
    if metadata.get("news_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} content-in-metadata alias parity failed: {metadata} vs {handoff}")
    if not isinstance(handoff.get("headline_titles"), list) or len(handoff.get("headline_titles")) != handoff.get("headline_count"):
        raise SystemExit(f"{label} handoff headline title parity failed: {handoff}")
    locale = handoff.get("locale")
    if not isinstance(locale, dict) or not locale.get("hl") or not locale.get("gl"):
        raise SystemExit(f"{label} handoff missed locale: {handoff}")
    if not isinstance(handoff.get("next_safe_command"), str) or not handoff["next_safe_command"]:
        raise SystemExit(f"{label} handoff missed next command: {handoff}")
    expected_commands = [handoff["next_safe_command"]]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} has wrong next safe command list: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} has wrong next safe command count: {handoff}")
    if metadata.get("news_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("news_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("news_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} next-command-count alias parity failed: {metadata} vs {handoff}")
    assert_no_raw_local_path(handoff, f"{label} handoff")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} handoff missed boundaries: {handoff}")
    if boundaries.get("calls_external_service") is not calls_external_service:
        raise SystemExit(f"{label} external-service boundary wrong: {handoff}")
    if boundaries.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} plural external-service boundary wrong: {handoff}")
    if metadata.get("news_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary alias parity failed: {metadata} vs {handoff}")
    if metadata.get("calls_external_service") is not calls_external_service:
        raise SystemExit(f"{label} flat external-service metadata wrong: {metadata}")
    if metadata.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} flat plural external-service metadata wrong: {metadata}")
    for key in [
        "calls_model",
        "executes_tools",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "requires_approval",
        "controls_computer",
    ]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} should not perform {key}: {metadata}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if boundaries.get(key) is not expected_value:
            raise SystemExit(f"{label} boundary {key} should be {expected_value}: {boundaries}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["get_news"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("get_news should be LOCAL_SAFE")


def test_news_argument_contract_fences_malformed_input() -> None:
    fetch_calls: list[str] = []
    original_fetch = nc._fetch

    def mocked_fetch(query: str) -> bytes:
        fetch_calls.append(query)
        return _RSS

    try:
        nc._fetch = mocked_fetch  # type: ignore[assignment]
        tool = _tools()["get_news"]
        contract = tool.argument_contract
        actual_shape = (
            tuple(
                (
                    field.name,
                    tuple(sorted(kind.value for kind in field.types)),
                    field.required,
                    field.minimum,
                    field.maximum,
                )
                for field in contract.fields
            )
            if contract is not None
            else ()
        )
        expected_shape = (
            ("query", ("string",), False, None, None),
            ("topic", ("string",), False, None, None),
            ("about", ("string",), False, None, None),
            ("limit", ("integer",), False, 1, 10),
        )
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown is not False
            or actual_shape != expected_shape
        ):
            raise SystemExit(f"get_news should have an exact strict argument contract: {contract}")

        registry = ToolRegistry()
        registry.register(tool)
        executor = Executor(registry, PermissionPolicy())
        private_sentinel = "private-news-contract-sentinel"
        malformed_args: list[object] = [
            {"query": {"value": private_sentinel}},
            {"topic": ["korea"]},
            {"about": True},
            {"limit": "5"},
            {"limit": False},
            {"limit": 0},
            {"limit": 11},
            {"unknown": private_sentinel},
            ["query", private_sentinel],
        ]
        for args in malformed_args:
            result = executor.execute(
                PlannedAction("get_news", args, "news contract smoke")  # type: ignore[arg-type]
            )
            if (
                result.ok
                or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
            ):
                raise SystemExit(f"malformed get_news arguments crossed the executor fence: {args!r} -> {result}")
            if private_sentinel in f"{result.output}\n{result.metadata}":
                raise SystemExit(f"get_news rejection leaked an argument value: {result}")
        if fetch_calls:
            raise SystemExit(f"malformed get_news arguments reached the network seam: {fetch_calls}")

        valid_args = (
            {},
            {"query": "korea"},
            {"topic": "technology"},
            {"about": "markets"},
            {"query": "ai", "limit": 1},
            {"limit": 10},
        )
        for args in valid_args:
            result = executor.execute(PlannedAction("get_news", args, "valid news contract smoke"))
            if not result.ok:
                raise SystemExit(f"valid get_news arguments were rejected: {args!r} -> {result}")
        if len(fetch_calls) != len(valid_args):
            raise SystemExit(f"valid get_news arguments did not each reach the mocked fetch seam: {fetch_calls}")
    finally:
        nc._fetch = original_fetch  # type: ignore[assignment]


def test_lists_headlines() -> None:
    nc._fetch = lambda query: _RSS  # type: ignore
    out = _tools()["get_news"].handler({"limit": 2})
    if not out.ok or "Markets rally" not in out.output or "Local team" not in out.output:
        raise SystemExit(f"headlines wrong: {out.output}")
    if "New phone" in out.output:
        raise SystemExit("limit not respected")
    handoff = assert_news_handoff(out.metadata, "news list", status="ok", calls_external_service=True)
    if handoff.get("headline_titles") != ["Markets rally on rate news - Reuters", "Local team wins championship - ESPN"]:
        raise SystemExit(f"news handoff should preserve bounded headline rows: {handoff}")


def test_query_passed_and_labeled() -> None:
    seen = {}

    def fake_fetch(query):
        seen["q"] = query
        return _RSS

    nc._fetch = fake_fetch  # type: ignore
    out = _tools()["get_news"].handler({"query": "korea"})
    if seen.get("q") != "korea" or "korea" not in out.output.lower():
        raise SystemExit(f"query not used: {seen} / {out.output}")
    if out.metadata.get("query") != "korea" or out.metadata.get("limit") != 5:
        raise SystemExit(f"news metadata should preserve cleaned query and limit: {out.metadata}")
    handoff = assert_news_handoff(out.metadata, "news query", status="ok", calls_external_service=True)
    if handoff.get("query") != "korea" or handoff.get("next_safe_command") != "news about korea":
        raise SystemExit(f"news query handoff missed query command: {handoff}")


def test_invalid_query_never_fetches() -> None:
    calls = []

    def fail_fetch(query):
        calls.append(query)
        raise AssertionError("invalid query should not fetch")

    nc._fetch = fail_fetch  # type: ignore
    cases = [
        ("---", "---", False),
        ("/\x55sers/example/private/news-query", "<local-path>", True),
        ("/var/folders/zc/jarvis/news-query", "<local-path>", True),
        ("/tmp/jarvis/news-query", "<local-path>", True),
    ]
    for query, metadata_query, is_local_path in cases:
        out = _tools()["get_news"].handler({"query": query, "limit": 2})
        if out.ok or "real news topic" not in out.output.lower():
            raise SystemExit(f"invalid news query should ask for a real topic: {query!r} -> {out.output}")
        if out.metadata.get("reason") != "invalid_query" or out.metadata.get("query") != metadata_query:
            raise SystemExit(f"invalid news query should preserve bounded metadata: {out.metadata}")
        handoff = assert_news_handoff(out.metadata, f"invalid news query {query!r}", status="invalid_input", calls_external_service=False)
        if handoff.get("reason") != "invalid_query" or handoff.get("retry_safe") is not False:
            raise SystemExit(f"invalid news handoff should preserve invalid-input state: {handoff}")
        if bool(out.metadata.get("local_path_query")) != is_local_path:
            raise SystemExit(f"only local-path news queries should be marked: {out.metadata}")
        if "/\x55sers/" in out.output or "/var/folders/" in out.output or "/tmp/" in out.output:
            raise SystemExit(f"invalid news query should not echo local paths: {out.output}")
    if calls:
        raise SystemExit(f"invalid news query unexpectedly fetched: {calls}")


def test_news_locale_reads_environment_at_call_time() -> None:
    old = os.environ.get("JARVIS_NEWS_LOCALE")
    try:
        os.environ["JARVIS_NEWS_LOCALE"] = "ko-KR:KR"
        url = nc._news_url("")
        if "hl=ko-KR" not in url or "gl=KR" not in url or "ceid=KR%3Ako" not in url:
            raise SystemExit(f"news locale did not read env at call time: {url}")
    finally:
        if old is None:
            os.environ.pop("JARVIS_NEWS_LOCALE", None)
        else:
            os.environ["JARVIS_NEWS_LOCALE"] = old


def test_news_locale_trims_parts() -> None:
    old = os.environ.get("JARVIS_NEWS_LOCALE")
    try:
        os.environ["JARVIS_NEWS_LOCALE"] = " ko-KR : KR "
        url = nc._news_url("")
        if "hl=ko-KR" not in url or "gl=KR" not in url or "ceid=KR%3Ako" not in url:
            raise SystemExit(f"news locale did not trim parts: {url}")
    finally:
        if old is None:
            os.environ.pop("JARVIS_NEWS_LOCALE", None)
        else:
            os.environ["JARVIS_NEWS_LOCALE"] = old


def test_invalid_news_locale_falls_back() -> None:
    old = os.environ.get("JARVIS_NEWS_LOCALE")
    try:
        os.environ["JARVIS_NEWS_LOCALE"] = "bad host:K!"
        url = nc._news_url("")
        if "hl=en-US" not in url or "gl=US" not in url or "ceid=US%3Aen" not in url:
            raise SystemExit(f"invalid news locale should fall back to defaults: {url}")
    finally:
        if old is None:
            os.environ.pop("JARVIS_NEWS_LOCALE", None)
        else:
            os.environ["JARVIS_NEWS_LOCALE"] = old


def test_path_shaped_news_locale_falls_back() -> None:
    old = os.environ.get("JARVIS_NEWS_LOCALE")
    try:
        for locale in (
            "/\x55sers/example/private/news-locale:KR",
            "/var/folders/zc/jarvis/news-locale:KR",
            "/tmp/jarvis/news-locale:KR",
        ):
            os.environ["JARVIS_NEWS_LOCALE"] = locale
            url = nc._news_url("")
            if "hl=en-US" not in url or "gl=US" not in url or "ceid=US%3Aen" not in url:
                raise SystemExit(f"path-shaped news locale should fall back to defaults: {locale!r} -> {url}")
            if "Users" in url or "private" in url or "var%2Ffolders" in url or "tmp" in url:
                raise SystemExit(f"path-shaped news locale leaked into URL: {url}")
    finally:
        if old is None:
            os.environ.pop("JARVIS_NEWS_LOCALE", None)
        else:
            os.environ["JARVIS_NEWS_LOCALE"] = old


def test_invalid_limit_falls_back() -> None:
    nc._fetch = lambda query: _RSS  # type: ignore
    out = _tools()["get_news"].handler({"limit": "not-a-number"})
    if not out.ok or out.metadata.get("count") != 3:
        raise SystemExit(f"invalid limit should fall back to default: {out.output} / {out.metadata}")
    if out.metadata.get("limit") != 5 or out.metadata.get("raw_limit") != "not-a-number":
        raise SystemExit(f"invalid limit should preserve sanitized raw metadata: {out.metadata}")
    long_out = _tools()["get_news"].handler({"limit": "l" * 120})
    if long_out.metadata.get("raw_limit") != ("l" * 80):
        raise SystemExit(f"long invalid limit should be bounded in metadata: {long_out.metadata}")
    false_out = _tools()["get_news"].handler({"limit": False})
    if not false_out.ok or false_out.metadata.get("limit") != 5 or false_out.metadata.get("raw_limit") != "False":
        raise SystemExit(f"boolean false limit should fall back to default with raw metadata: {false_out.metadata}")
    true_out = _tools()["get_news"].handler({"limit": True})
    if not true_out.ok or true_out.metadata.get("limit") != 5 or true_out.metadata.get("raw_limit") != "True":
        raise SystemExit(f"boolean true limit should fall back to default with raw metadata: {true_out.metadata}")
    path_out = _tools()["get_news"].handler({"limit": "/\x55sers/example/private/news-limit"})
    if not path_out.ok or path_out.metadata.get("limit") != 5 or path_out.metadata.get("raw_limit") != "<local-path>":
        raise SystemExit(f"path-shaped limit should be redacted in metadata: {path_out.metadata}")
    assert_news_handoff(path_out.metadata, "path-shaped news limit", status="ok", calls_external_service=True)
    tmp_path_out = _tools()["get_news"].handler({"limit": "/tmp/jarvis-news-limit"})
    if not tmp_path_out.ok or tmp_path_out.metadata.get("limit") != 5 or tmp_path_out.metadata.get("raw_limit") != "<local-path>":
        raise SystemExit(f"temp path-shaped limit should be redacted in metadata: {tmp_path_out.metadata}")


def test_handles_error() -> None:
    def boom(query):
        raise RuntimeError("offline")

    nc._fetch = boom  # type: ignore
    out = _tools()["get_news"].handler({"query": "  korea   tech  ", "limit": 2})
    if out.ok or not out.output.strip():
        raise SystemExit(f"error not handled: {out.output}")
    if out.metadata.get("query") != "korea tech" or out.metadata.get("limit") != 2:
        raise SystemExit(f"news error metadata should preserve cleaned query and limit: {out.metadata}")
    assert_news_recovery_message(out.output, "generic news fetch error")
    assert_external_information_guidance(out, "generic news fetch error")
    if out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"news error should be friendly with bounded exception metadata: {out.output} {out.metadata}")
    handoff = assert_news_handoff(out.metadata, "news fetch error", status="unavailable", calls_external_service=True)
    if handoff.get("reason") != "fetch_error" or handoff.get("retry_safe") is not True:
        raise SystemExit(f"news fetch error handoff should preserve retryable state: {handoff}")

    def server_error(query):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    nc._fetch = server_error  # type: ignore
    server_out = _tools()["get_news"].handler({"query": "korea", "limit": 2})
    if server_out.ok:
        raise SystemExit(f"HTTP 500 should be a service-trouble message: {server_out.output}")
    assert_news_recovery_message(server_out.output, "news HTTP 500")
    if server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"news HTTP error should keep bounded exception type metadata: {server_out.metadata}")

    def timeout_error(query):
        raise TimeoutError("request timed out near /\x55sers/example/private")

    nc._fetch = timeout_error  # type: ignore
    timeout_out = _tools()["get_news"].handler({"query": "korea", "limit": 2})
    if timeout_out.ok:
        raise SystemExit(f"timeout should be a retryable service message: {timeout_out.output}")
    assert_news_recovery_message(timeout_out.output, "news timeout")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"timeout message should preserve the safe cause: {timeout_out.output}")

    nc._fetch = boom  # type: ignore
    bad_limit = _tools()["get_news"].handler({"query": "korea", "limit": "bad"})
    if bad_limit.ok or bad_limit.metadata.get("raw_limit") != "bad" or bad_limit.metadata.get("limit") != 5:
        raise SystemExit(f"news error metadata should preserve sanitized raw bad limit: {bad_limit.metadata}")
    path_limit = _tools()["get_news"].handler({"query": "korea", "limit": "/private/tmp/jarvis-news-limit"})
    if path_limit.ok or path_limit.metadata.get("raw_limit") != "<local-path>" or path_limit.metadata.get("limit") != 5:
        raise SystemExit(f"news error metadata should redact path-shaped raw bad limit: {path_limit.metadata}")
    assert_news_handoff(path_limit.metadata, "news path limit error", status="unavailable", calls_external_service=True)


def test_empty_headlines_have_handoff() -> None:
    nc._fetch = lambda query: b"<rss><channel></channel></rss>"  # type: ignore
    out = _tools()["get_news"].handler({"query": "korea", "limit": 3})
    if not out.ok or "No headlines found" not in out.output:
        raise SystemExit(f"empty headlines should be a successful empty state: {out.output}")
    if out.metadata.get("count") != 0 or out.metadata.get("query") != "korea":
        raise SystemExit(f"empty headlines metadata should preserve count/query: {out.metadata}")
    handoff = assert_news_handoff(out.metadata, "empty news", status="empty", calls_external_service=True)
    if handoff.get("reason") != "no_headlines" or handoff.get("headline_titles") != []:
        raise SystemExit(f"empty news handoff should preserve no-headlines state: {handoff}")


def test_planner_routes_news() -> None:
    p = RuleBasedPlanner()
    for q in [
        "what is the news",
        "headlines",
        "what are the headlines",
        "what are the latest headlines",
        "news please",
        "headlines please",
        "show news",
        "show latest news",
        "top headlines please",
        "news about Korea",
        "headlines about Korea",
        "show news about Korea",
        "news tech",
        "headlines tech",
        "news for tech",
        "headlines for tech",
        "Korea news",
        "AI news today",
        "what is happening in Korea",
        "what is happening in the world",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["get_news"]:
            raise SystemExit(f"planner missed news route: {q!r}")
    for query, expected in {
        "news about Korea": "korea",
        "headlines about Korea": "korea",
        "show news about Korea": "korea",
        "news tech": "tech",
        "headlines tech": "tech",
        "news for tech": "tech",
        "headlines for tech": "tech",
        "latest news for AI": "ai",
        "Korea news": "korea",
        "AI news today": "ai",
        "what is happening in Korea": "korea",
        "headlines tech please": "tech",
    }.items():
        actual = p.plan(query).actions[0].args.get("query")
        if actual != expected:
            raise SystemExit(f"planner did not extract news query: {query!r} -> {actual!r}")
    for query in [
        "latest news",
        "top news",
        "today news",
        "news please",
        "headlines please",
        "show news",
        "show latest news",
        "top headlines please",
        "what are the headlines",
        "what are the latest headlines",
        "what is happening in the world",
        # Real bug found live 2026-07-08: "what's IN the news today" fell through
        # the "what's" prefix-strip (which didn't allow for "in") to a fallback
        # pattern that greedily captured "what's in the" itself as the topic,
        # producing irrelevant results for a plain "what's in the news" request.
        "what's in the news today",
        "what is in the news",
        "what's in the news",
    ]:
        args = p.plan(query).actions[0].args
        if args.get("query"):
            raise SystemExit(f"generic news route should not invent a topic: {query!r} -> {args}")
    wiki_actions = p.plan("what are black holes").actions
    if [a.tool_name for a in wiki_actions] != ["wiki_summary"]:
        raise SystemExit(f"planner should preserve generic what-are Wikipedia route: {wiki_actions}")


def main() -> None:
    test_is_local_safe()
    test_news_argument_contract_fences_malformed_input()
    test_lists_headlines()
    test_query_passed_and_labeled()
    test_invalid_query_never_fetches()
    test_news_locale_reads_environment_at_call_time()
    test_news_locale_trims_parts()
    test_invalid_news_locale_falls_back()
    test_path_shaped_news_locale_falls_back()
    test_invalid_limit_falls_back()
    test_handles_error()
    test_empty_headlines_have_handoff()
    test_planner_routes_news()
    print("News connector smoke passed")


if __name__ == "__main__":
    main()
