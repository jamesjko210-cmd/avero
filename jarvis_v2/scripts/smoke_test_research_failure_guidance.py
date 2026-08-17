"""Focused offline proof for research failure guidance and public-read fences."""

from __future__ import annotations

from unittest.mock import patch

from jarvis_v2.agent.failure_guidance import FAILURE_GUIDANCE_VERSION
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.config import load_config
from jarvis_v2.tools import research_connector as rc


PRIVATE_VALUES = (
    "/\x55sers/example/private/research.txt",
    "/private/tmp/research.txt",
    "sk_" + "live_researchPrivateToken123456",
    "123456:telegramPrivateTokenValue1234567890",
)


def _tools():
    return {tool.name: tool for tool in rc.make_research_tools(load_config())}


def _assert_public_failure(result: ToolResult, label: str, *, external: bool) -> None:
    if result.ok:
        raise SystemExit(f"{label} should fail: {result}")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("version") != FAILURE_GUIDANCE_VERSION:
        raise SystemExit(f"{label} lacks canonical guidance: {result.metadata}")
    action = guidance.get("action")
    if not isinstance(action, str) or action not in result.output:
        raise SystemExit(f"{label} guidance is not user-visible: {result}")
    for key, expected in {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
    }.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} has wrong {key}: {result.metadata}")
    if result.metadata.get("calls_external_service") is not external:
        raise SystemExit(f"{label} has wrong external-call truth: {result.metadata}")
    surface = result.output + repr(result.metadata)
    for private in PRIVATE_VALUES:
        if private in surface:
            raise SystemExit(f"{label} leaked private input: {surface}")


def test_validation_failures_are_local_and_private_safe() -> None:
    with patch.object(rc, "ddg_search", side_effect=AssertionError("validation reached search")):
        for tool_name in ("web_lookup", "research"):
            for value in ("", "---", "x" * (rc.MAX_QUERY_CHARS + 1), *PRIVATE_VALUES):
                result = _tools()[tool_name].handler({"query": value})
                _assert_public_failure(result, f"{tool_name} validation {value[:12]!r}", external=False)
                handoff = result.metadata.get(f"{tool_name}_handoff") or {}
                if handoff.get("status") != "refused" or handoff.get("state_changed") is not False:
                    raise SystemExit(f"{tool_name} validation handoff changed truth: {handoff}")
                if any(private in repr(handoff) for private in PRIVATE_VALUES):
                    raise SystemExit(f"{tool_name} validation handoff leaked private input: {handoff}")


def test_search_and_parse_failures_are_canonical_reads() -> None:
    for tool_name in ("web_lookup", "research"):
        with patch.object(
            rc,
            "ddg_search",
            side_effect=TimeoutError("private backend /\x55sers/example/private/research.txt"),
        ):
            result = _tools()[tool_name].handler({"query": "public research topic"})
        _assert_public_failure(result, f"{tool_name} search timeout", external=True)
        if result.metadata.get("exception_type") != "TimeoutError":
            raise SystemExit(f"{tool_name} lost bounded failure provenance: {result.metadata}")
        if result.metadata.get("recovery_commands") != ["setup check"]:
            raise SystemExit(f"{tool_name} lost setup recovery command: {result.metadata}")

        with patch.object(rc, "ddg_search", side_effect=ValueError("malformed private response")):
            result = _tools()[tool_name].handler({"query": "public research topic"})
        _assert_public_failure(result, f"{tool_name} parse failure", external=True)
        if result.metadata.get("exception_type") != "ValueError":
            raise SystemExit(f"{tool_name} parse failure lost exception class: {result.metadata}")


def test_result_urls_are_fenced_before_output_or_page_fetch() -> None:
    unsafe_targets = (
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://127.0.0.1/admin",
        "http://10.0.0.1/admin",
        "https://user:secret@example.com/private",
    )
    anchors = [
        (
            "https://public.example/research",
            "Public result",
            "public research evidence",
        ),
        *((target, f"Unsafe {index}", "private") for index, target in enumerate(unsafe_targets)),
    ]
    html = "".join(
        f'<a class="result-link" href="//duckduckgo.com/l/?uddg={rc.urllib.parse.quote(url, safe="")}">{title}</a>'
        f'<td class="result-snippet">{snippet}</td>'
        for url, title, snippet in anchors
    )
    with patch.object(rc, "_get", return_value=html):
        parsed = rc.ddg_search("public research", max_results=10)
    if [row["url"] for row in parsed] != ["https://public.example/research"]:
        raise SystemExit(f"unsafe result URLs escaped the parser fence: {parsed}")

    fetches: list[dict[str, str]] = []

    def fake_fetch(args: dict[str, str]) -> ToolResult:
        fetches.append(args)
        return ToolResult("fetch_page", True, "public research evidence", {})

    with patch.object(rc, "ddg_search", return_value=parsed), patch(
        "jarvis_v2.tools.browser.fetch_page", side_effect=fake_fetch
    ), patch.object(rc, "_synthesize", return_value="public answer"):
        result = _tools()["research"].handler({"query": "public research"})
    if not result.ok or fetches != [{"url": "https://public.example/research"}]:
        raise SystemExit(f"research did not fetch only the fenced public URL: {fetches} {result}")
    if any(target in result.output for target in unsafe_targets):
        raise SystemExit(f"research output exposed an unsafe result URL: {result.output}")


def test_success_provenance_does_not_claim_history_or_state_writes() -> None:
    rows = [
        {
            "title": "Public source",
            "url": "https://public.example/source",
            "snippet": "public research topic evidence",
        }
    ]
    with patch.object(rc, "ddg_search", return_value=rows), patch.object(
        rc, "_fetch_page_text", return_value="public research topic evidence"
    ), patch.object(rc, "_synthesize", return_value="A source-bounded answer."):
        result = _tools()["research"].handler({"query": "public research topic"})
    if not result.ok or "Sources:" not in result.output or rows[0]["url"] not in result.output:
        raise SystemExit(f"research success lost visible source provenance: {result}")
    for key in ("writes_files", "writes_database", "writes_memory", "writes_notes"):
        if result.metadata.get(key) is not False:
            raise SystemExit(f"research falsely claimed a history/provenance write: {result.metadata}")
    handoff = result.metadata.get("research_handoff") or {}
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"research falsely claimed changed state: {handoff}")
    if handoff.get("result_rows") != [{"title": "Public source", "url": rows[0]["url"]}]:
        raise SystemExit(f"research lost bounded source provenance: {handoff}")
    for key in ("answer_in_metadata", "snippet_in_metadata", "page_text_in_metadata"):
        if handoff.get(key) is not False:
            raise SystemExit(f"research copied private content into metadata: {handoff}")

    with patch.object(rc, "ddg_search", return_value=rows), patch(
        "jarvis_v2.tools.browser.fetch_page",
        return_value=ToolResult("fetch_page", False, "bounded public-read failure", {}),
    ), patch.object(rc, "_synthesize", return_value=""):
        fallback = _tools()["research"].handler({"query": "public research topic"})
    fallback_handoff = fallback.metadata.get("research_handoff") or {}
    if not fallback.ok or fallback_handoff.get("page_read_count") != 0:
        raise SystemExit(f"snippet fallback falsely claimed a successful page read: {fallback}")
    if fallback.metadata.get("writes_database") is not False or fallback_handoff.get("state_changed") is not False:
        raise SystemExit(f"failed page read falsely claimed history/state changes: {fallback}")


def main() -> None:
    test_validation_failures_are_local_and_private_safe()
    test_search_and_parse_failures_are_canonical_reads()
    test_result_urls_are_fenced_before_output_or_page_fetch()
    test_success_provenance_does_not_claim_history_or_state_writes()
    print("Research failure guidance smoke passed")


if __name__ == "__main__":
    main()
