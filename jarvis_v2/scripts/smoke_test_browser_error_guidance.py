from __future__ import annotations

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.tools import browser


PRIVATE_MARKERS = (
    "/\x55sers/owner/private",
    "/private/var/folders/private-browser-state",
    "owner-secret-token",
)


def _assert_guidance(result: ToolResult, *, label: str) -> None:
    if result.ok is not False:
        raise SystemExit(f"{label} unexpectedly succeeded")
    declaration = result.metadata.get("recovery_guidance")
    if not isinstance(declaration, dict) or declaration.get("version") != 1:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result.metadata}")
    action = declaration.get("action")
    if not isinstance(action, str) or action not in result.output:
        raise SystemExit(f"{label} recovery action is not user-visible: {result}")
    expected_truth = {
        "outcome_known": True,
        "outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, value in expected_truth.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} recovery truth drifted for {key}: {result.metadata}")
    public = f"{result.output}\n{declaration}".lower()
    if any(marker.lower() in public for marker in PRIVATE_MARKERS):
        raise SystemExit(f"{label} leaked private detail: {public}")


def main() -> None:
    factory_fetch, factory_links, factory_search, *_rest = browser.make_browser_tools()
    implementations = (
        ("global fetch", browser.fetch_page),
        ("factory fetch", factory_fetch),
        ("global links", browser.extract_links),
        ("factory links", factory_links),
    )
    fixtures = (
        ("missing", {}),
        ("local path", {"url": PRIVATE_MARKERS[0]}),
        ("embedded credentials", {"url": "https://owner-secret-token@example.com/page"}),
    )
    for implementation_label, implementation in implementations:
        for fixture_label, args in fixtures:
            _assert_guidance(
                implementation(args),
                label=f"{implementation_label} {fixture_label}",
            )

    original_fetch = browser._fetch

    def fail_fetch(_url: str):
        raise RuntimeError(PRIVATE_MARKERS[1])

    browser._fetch = fail_fetch
    try:
        for implementation_label, implementation in implementations:
            _assert_guidance(
                implementation({"url": "https://example.test/public"}),
                label=f"{implementation_label} network failure",
            )
    finally:
        browser._fetch = original_fetch

    for implementation_label, implementation in (
        ("global search", browser.web_search),
        ("factory search", factory_search),
    ):
        _assert_guidance(
            implementation({}),
            label=f"{implementation_label} missing query",
        )
        _assert_guidance(
            implementation({"query": PRIVATE_MARKERS[0]}),
            label=f"{implementation_label} local-path query",
        )

    for fixture_label, args in (
        ("missing URL", {}),
        ("local-path URL", {"url": PRIVATE_MARKERS[0]}),
        ("credential URL", {"url": "https://owner-secret-token@example.com/page"}),
        ("disabled OS launch", {"url": "https://example.test/public"}),
    ):
        result = browser.open_url(args)
        _assert_guidance(result, label=f"open URL {fixture_label}")
        if result.metadata.get("controls_computer") is not False:
            raise SystemExit(f"open URL {fixture_label} gained computer control")
        if fixture_label == "disabled OS launch" and result.metadata.get(
            "recovery_commands"
        ) != ["fetch page https://example.test/public"]:
            raise SystemExit(f"open URL safe fetch handoff drifted: {result.metadata}")

    print("Browser error-guidance smoke passed: 24 public-read failure paths")


if __name__ == "__main__":
    main()
