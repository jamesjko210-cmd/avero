from __future__ import annotations

import subprocess
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.memory import obsidian
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import browser


SAFE_FALSE_FLAGS = [
    "calls_model",
    "executes_tools",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "reads_private_data",
    "reads_personal_data",
    "writes_memory",
    "external_side_effect",
    "queues_approval",
    "requires_approval",
    "controls_computer",
    "speaks",
    "completes_tasks",
]


HTML = """
<html>
  <head><title>Jarvis Browser Smoke</title></head>
  <body>
    <h1>Jarvis Browser Smoke</h1>
    <p>Jarvis should fetch readable text while keeping browser actions safe and bounded.</p>
    <a href="/docs">Documentation</a>
    <a href="https://example.com/next">Next step</a>
  </body>
</html>
"""


class FakeBrowserHeaders:
    def get_content_charset(self) -> str:
        return "utf-8"


class FakeBrowserResponse:
    def __init__(self, payload: bytes, url: str = "https://example.test/"):
        self.payload = payload
        self.url = url
        self.headers = FakeBrowserHeaders()
        self.read_size = -1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        self.read_size = size
        return self.payload if size < 0 else self.payload[:size]

    def geturl(self) -> str:
        return self.url


def assert_vault_relative_receipt(result, root: Path, prefix: str, label: str) -> None:
    metadata = result.metadata
    path_text = str(metadata.get("path") or "")
    path_display = metadata.get("path_display")
    receipt_line = result.output.split("\n", 1)[0]
    if not path_text or not Path(path_text).exists():
        raise SystemExit(f"{label} should preserve exact saved path metadata: {metadata}")
    if path_text in receipt_line:
        raise SystemExit(f"{label} should not print the raw local note path.")
    if str(root) in receipt_line or "/private/" in receipt_line or "/\x55sers/" in receipt_line:
        raise SystemExit(f"{label} receipt should not expose local temp or user paths.")
    if not isinstance(path_display, str) or not path_display.startswith(prefix):
        raise SystemExit(f"{label} missed vault-relative display metadata: {metadata}")
    if path_display not in receipt_line:
        raise SystemExit(f"{label} should print the vault-relative saved-note label.")


def assert_no_local_path_leak(result, label: str) -> None:
    combined = result.output + " " + str(result.metadata)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in combined:
            raise SystemExit(f"{label} leaked a local path: {result.output} {result.metadata}")


def assert_no_browser_authority(metadata: dict, label: str) -> None:
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False: {metadata}")


def assert_browser_state_recovery(result, label: str, retry_command: str) -> None:
    if result.ok or result.metadata.get("reason") != "browser_state_unavailable":
        raise SystemExit(f"{label} should fail closed with browser-state metadata: {result}")
    for expected in ["setup check", "JARVIS_DATA_DIR", "JARVIS_DB_PATH", retry_command]:
        if expected not in result.output:
            raise SystemExit(f"{label} missed recovery guidance {expected!r}: {result.output}")
    if result.metadata.get("next_command") != "setup check":
        raise SystemExit(f"{label} should name setup check first: {result.metadata}")
    if result.metadata.get("recovery_commands") != ["setup check", retry_command]:
        raise SystemExit(f"{label} missed ordered recovery commands: {result.metadata}")
    if result.metadata.get("retry_requires_storage_repair") is not True:
        raise SystemExit(f"{label} should require storage repair: {result.metadata}")
    if result.metadata.get("authorizes_retry") is not False:
        raise SystemExit(f"{label} should not authorize retry: {result.metadata}")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("action") not in result.output:
        raise SystemExit(f"{label} missed canonical user-visible recovery guidance: {result.metadata}")
    if result.metadata.get("writes_database") or result.metadata.get("writes_files") or result.metadata.get("writes_notes"):
        raise SystemExit(f"{label} failure should not claim successful writes: {result.metadata}")
    assert_no_browser_authority(result.metadata, label)
    assert_no_local_path_leak(result, label)


def test_planner_routes_open_url() -> None:
    # Real gap found live 2026-07-09: "open url google.com" (and "open website ..."
    # / "open link ...") fell through the http(s)/www-only open_url pattern into a
    # generic local-file read match, which then tried to read a file literally named
    # "url google.com" because the "." in "google.com" satisfied that route's
    # local-file heuristic. Fixed by teaching open_url to recognize an explicit
    # url/website/site/link lead-in word ahead of a bare domain, and by teaching
    # the local-file heuristic to check for an embedded http(s)/www URL anywhere
    # in the phrase, not just at the very start.
    cases = {
        "open url google.com": "google.com",
        "open website google.com": "google.com",
        "open the website example.com": "example.com",
        "open link https://example.com": "https://example.com",
        "open https://example.com": "https://example.com",
        "open www.example.com": "www.example.com",
    }
    for q, expected_url in cases.items():
        actions = RuleBasedPlanner().plan(q).actions
        if [a.tool_name for a in actions] != ["open_url"] or actions[0].args.get("url") != expected_url:
            raise SystemExit(f"open_url route missed: {q!r} -> {actions}")
    # Local file reads must still win when there is no explicit URL signal.
    for q, expected_path in {
        "open notes.txt": "notes.txt",
        "open file notes.txt": "notes.txt",
        "read notes.txt": "notes.txt",
        "open ~/Desktop/notes.txt": "~/Desktop/notes.txt",
    }.items():
        actions = RuleBasedPlanner().plan(q).actions
        if [a.tool_name for a in actions] != ["read_text_file"] or actions[0].args.get("path") != expected_path:
            raise SystemExit(f"local file route regressed: {q!r} -> {actions}")


def test_browser_fetch_bounds_response_before_html_parsing() -> None:
    exact = FakeBrowserResponse(b"12345678", "https://example.test/exact")
    with patch.object(browser, "MAX_BROWSER_RESPONSE_BYTES", 8), patch.object(
        browser, "urlopen", return_value=exact
    ) as opener, patch.object(
        browser.socket, "getaddrinfo", return_value=[(browser.socket.AF_INET, browser.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    ):
        if browser._fetch("https://example.test/exact") != ("12345678", "https://example.test/exact"):
            raise SystemExit("Browser rejected a response exactly at its byte ceiling")
    if exact.read_size != 9 or opener.call_count != 1:
        raise SystemExit("Browser fetch did not use one bounded max+1-byte read")

    oversized = FakeBrowserResponse(b"123456789", "https://example.test/oversized")
    with patch.object(browser, "MAX_BROWSER_RESPONSE_BYTES", 8), patch.object(
        browser, "urlopen", return_value=oversized
    ) as opener, patch.object(
        browser.socket, "getaddrinfo", return_value=[(browser.socket.AF_INET, browser.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    ):
        try:
            browser._fetch("https://example.test/oversized")
        except ValueError as exc:
            if "too large" not in str(exc):
                raise SystemExit(f"Oversized browser response used the wrong diagnostic: {exc}")
        else:
            raise SystemExit("Browser accepted a response above its byte ceiling")
    if oversized.read_size != 9 or opener.call_count != 1:
        raise SystemExit("Oversized browser response should fail after one bounded read")


def test_browser_connection_uses_validated_address_without_dns_rebinding() -> None:
    dns_answers = ["93.184.216.34", "10.20.30.40"]
    dns_calls = []

    def rebinding_getaddrinfo(host, port, *args, **kwargs):
        dns_calls.append((host, port))
        address = dns_answers[min(len(dns_calls) - 1, len(dns_answers) - 1)]
        return [(browser.socket.AF_INET, browser.socket.SOCK_STREAM, 6, "", (address, port))]

    class FakeConnectedSocket:
        def __init__(self):
            self.connected_to = []

        def settimeout(self, _timeout):
            pass

        def bind(self, _source_address):
            pass

        def connect(self, sockaddr):
            self.connected_to.append(sockaddr)

        def setsockopt(self, *_args):
            pass

        def sendall(self, _payload):
            pass

        def makefile(self, *_args, **_kwargs):
            return BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")

        def close(self):
            pass

    fake_socket = FakeConnectedSocket()
    with patch.object(browser.socket, "getaddrinfo", side_effect=rebinding_getaddrinfo), patch.object(
        browser.socket, "socket", return_value=fake_socket
    ):
        fetched = browser._fetch("http://rebind.example/resource")

    if fetched != ("OK", "http://rebind.example/resource"):
        raise SystemExit(f"Pinned browser fetch returned the wrong payload or URL: {fetched}")
    if dns_calls != [("rebind.example", 80)]:
        raise SystemExit(f"Browser connection performed a second DNS lookup: {dns_calls}")
    if fake_socket.connected_to != [("93.184.216.34", 80)]:
        raise SystemExit(f"Browser connection did not use the validated public address: {fake_socket.connected_to}")


def test_browser_network_boundary() -> None:
    resolutions = {
        "start.example": "93.184.216.34",
        "public.example": "93.184.216.34",
        "private.example": "10.20.30.40",
    }

    def fake_getaddrinfo(host, port, *args, **kwargs):
        address = resolutions.get(host)
        if address is None:
            raise browser.socket.gaierror("unknown mocked host")
        return [(browser.socket.AF_INET, browser.socket.SOCK_STREAM, 6, "", (address, port))]

    invalid_urls = [
        "file:///etc/passwd",
        "ftp://public.example/file",
        "https:///missing-host",
        "https://user:secret@example.com/",
        "http://localhost/",
        "http://api.localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://169.254.1.1/",
        "http://224.0.0.1/",
        "http://0.0.0.0/",
        "http://240.0.0.1/",
        "http://[::1]/",
        "https://bad_host/",
    ]
    with patch.object(browser, "urlopen", side_effect=AssertionError("invalid URL reached network")):
        for url in invalid_urls:
            result = browser.fetch_page({"url": url})
            if result.ok or result.metadata.get("reason") != "invalid_url":
                raise SystemExit(f"Browser should reject unsafe URL {url!r}: {result}")
            if result.metadata.get("external_network") or url in result.output or url in str(result.metadata):
                raise SystemExit(f"Unsafe URL refusal should be inert and redacted: {result}")

    with patch.object(browser.socket, "getaddrinfo", side_effect=fake_getaddrinfo), patch.object(
        browser, "urlopen", side_effect=AssertionError("private DNS result reached HTTP")
    ):
        private_result = browser.fetch_page({"url": "https://private.example/admin"})
    if private_result.ok or private_result.metadata.get("exception_type") != "UnsafeBrowserURL":
        raise SystemExit(f"Browser should fail closed on private DNS answers: {private_result}")

    request = browser.Request("https://public.example/start")
    handler = browser._SafeRedirectHandler()
    with patch.object(browser.socket, "getaddrinfo", side_effect=fake_getaddrinfo):
        public_redirect = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://public.example/redirected",
        )
        for target in ("file:///etc/passwd", "http://127.0.0.1/admin", "https://private.example/admin"):
            try:
                handler.redirect_request(request, None, 302, "Found", {}, target)
            except browser.UnsafeBrowserURL:
                pass
            else:
                raise SystemExit(f"Browser redirect handler accepted unsafe target: {target}")
    with patch.object(browser.socket, "getaddrinfo", side_effect=AssertionError("redirect performed second DNS lookup")):
        redirect_addresses = browser._request_addresses(public_redirect)
    if [address.sockaddr for address in redirect_addresses] != [("93.184.216.34", 443)]:
        raise SystemExit(f"Browser redirect did not retain its validated public address: {redirect_addresses}")

    long_title = "T" * (browser.MAX_BROWSER_TITLE_CHARS + 500)
    payload = (
        f"<html><title>{long_title}</title><body>Public page text."
        '<a href="child">Relative child</a></body></html>'
    ).encode()
    response = FakeBrowserResponse(payload, "https://public.example/final/page")
    opened_urls = []

    def fake_urlopen(req, *, timeout):
        opened_urls.append((req.full_url, timeout))
        return response

    class CaptureStore:
        def __init__(self):
            self.saved = []

        def save_browser_page(self, url, title, text):
            self.saved.append((url, title, text))

    store = CaptureStore()
    fetch_page, extract_links, *_ = browser.make_browser_tools(store=store)  # type: ignore[arg-type]
    with patch.object(browser.socket, "getaddrinfo", side_effect=fake_getaddrinfo), patch.object(
        browser, "urlopen", side_effect=fake_urlopen
    ):
        allowed = fetch_page({"url": "start.example/origin"})
        links = extract_links({"url": "start.example/origin"})
    if not allowed.ok or opened_urls[0][0] != "https://start.example/origin":
        raise SystemExit(f"Browser should preserve bare-domain normalization for public URLs: {allowed}")
    if allowed.metadata.get("url") != "https://public.example/final/page":
        raise SystemExit(f"Browser should expose the final validated response URL: {allowed.metadata}")
    if len(allowed.metadata.get("title", "")) > browser.MAX_BROWSER_TITLE_CHARS:
        raise SystemExit(f"Browser title metadata exceeded its bound: {allowed.metadata}")
    if len(allowed.output.split("\n", 1)[0].removeprefix("# ")) > browser.MAX_BROWSER_TITLE_CHARS:
        raise SystemExit("Browser title output exceeded its bound")
    if not store.saved or store.saved[-1][0] != "https://public.example/final/page":
        raise SystemExit(f"Browser history should store the final URL: {store.saved}")
    if any(len(saved[1]) > browser.MAX_BROWSER_TITLE_CHARS for saved in store.saved):
        raise SystemExit(f"Browser history title exceeded its bound: {store.saved}")
    if "https://public.example/final/child" not in links.output:
        raise SystemExit(f"Relative links should resolve against the final URL: {links.output}")


def test_open_url_fails_closed_without_process_or_network() -> None:
    with patch.object(subprocess, "run", side_effect=AssertionError("open_url launched a process")) as launcher, patch.object(
        browser.socket,
        "getaddrinfo",
        side_effect=AssertionError("disabled open_url should not resolve a host"),
    ) as resolver:
        result = browser.open_url({"url": "https://public.example/path"})

    expected_command = "fetch page https://public.example/path"
    if result.ok or result.metadata.get("reason") != "os_browser_launch_disabled":
        raise SystemExit(f"open_url should fail closed when address pinning cannot be preserved: {result}")
    if expected_command not in result.output or result.metadata.get("next_command") != expected_command:
        raise SystemExit(f"open_url should direct recovery through pinned fetch: {result}")
    if result.metadata.get("recovery_commands") != [expected_command] or result.metadata.get("pinned_fetch_available") is not True:
        raise SystemExit(f"open_url missed pinned-fetch recovery metadata: {result.metadata}")
    if result.metadata.get("external_network") or result.metadata.get("controls_computer"):
        raise SystemExit(f"disabled open_url should report no launch or network activity: {result.metadata}")
    if result.metadata.get("authorizes_retry") is not False:
        raise SystemExit(f"disabled open_url should not authorize an unsafe retry: {result.metadata}")
    if launcher.call_count or resolver.call_count:
        raise SystemExit("disabled open_url unexpectedly launched a process or resolved a host")
    assert_no_browser_authority(result.metadata, "disabled open_url")


def test_browser_history_writers_are_local_safe_and_truthful() -> None:
    with TemporaryDirectory(prefix="jarvis-browser-history-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = {
            "fetch_page": {"url": "https://fetch.example.test/page"},
            "extract_links": {"url": "https://links.example.test/page"},
            "web_search": {"query": "jarvis mocked browser history"},
        }
        with patch.object(browser, "_fetch", side_effect=lambda url, timeout=12: (HTML, url)):
            for name, args in cases.items():
                tool = runtime.registry.get(name)
                if tool.risk.name != "LOCAL_SAFE":
                    raise SystemExit(f"{name} should be LOCAL_SAFE because it writes browser history: {tool.risk.name}")
                result = tool.handler(args)
                if not result.ok or result.tool_name != name:
                    raise SystemExit(f"{name} should return a successful name-bound receipt: {result}")
                if result.metadata.get("history_saved") is not True:
                    raise SystemExit(f"{name} should confirm its browser history write: {result.metadata}")
                if result.metadata.get("writes_database") is not True:
                    raise SystemExit(f"{name} should truthfully mark its database write: {result.metadata}")
                if result.metadata.get("writes_files"):
                    raise SystemExit(f"{name} should not claim a file write: {result.metadata}")

        rows = runtime.store.recent_browser_pages(limit=10)
        if len(rows) != len(cases):
            raise SystemExit(f"browser history writers should persist one page each: {len(rows)} row(s)")


def main() -> None:
    test_planner_routes_open_url()
    test_browser_fetch_bounds_response_before_html_parsing()
    test_browser_connection_uses_validated_address_without_dns_rebinding()
    test_browser_network_boundary()
    test_open_url_fails_closed_without_process_or_network()
    test_browser_history_writers_are_local_safe_and_truthful()
    with TemporaryDirectory(prefix="jarvis-browser-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        original_fetch = browser._fetch
        try:
            browser._fetch = lambda _url, timeout=12: (HTML, _url)  # type: ignore[assignment]
            fetch_page, extract_links, web_search, recent_pages, summarize_page, save_page_note = browser.make_browser_tools(
                runtime.store,
                runtime.vault,
            )

            fetched = fetch_page({"url": "example.com", "max_chars": "bad"})
            if not fetched.ok or fetched.metadata.get("max_chars") != 4000:
                raise SystemExit("fetch_page should normalize URLs and sanitize bad max_chars.")
            if not fetched.metadata.get("writes_database") or fetched.metadata.get("writes_files"):
                raise SystemExit("fetch_page should mark browser history writes but no file writes.")
            assert_no_browser_authority(fetched.metadata, "fetch_page")
            for key in SAFE_FALSE_FLAGS:
                if fetched.metadata.get(key):
                    raise SystemExit(f"fetch_page unsafe metadata {key}: {fetched.metadata}")

            clipped = fetch_page({"url": "https://example.com", "max_chars": 999999})
            if clipped.metadata.get("max_chars") != browser.MAX_BROWSER_TEXT_CHARS:
                raise SystemExit("fetch_page should clamp huge max_chars.")
            bool_clipped = fetch_page({"url": "https://example.com", "max_chars": False})
            if not bool_clipped.ok or bool_clipped.metadata.get("max_chars") != 4000:
                raise SystemExit(f"fetch_page should treat boolean max_chars as malformed defaults: {bool_clipped.metadata}")

            long_url = fetch_page({"url": "example.com/" + ("x" * 3000)})
            if not long_url.ok or len(long_url.metadata.get("url", "")) > browser.MAX_BROWSER_URL_CHARS + len("https://"):
                raise SystemExit("fetch_page should bound oversized URLs.")

            links = extract_links({"url": "https://example.com"})
            if not links.ok or links.metadata.get("count") != 2 or links.metadata.get("max_links") != browser.MAX_BROWSER_LINKS:
                raise SystemExit("extract_links should parse links and include bounds metadata.")
            assert_no_browser_authority(links.metadata, "extract_links")
            for key in SAFE_FALSE_FLAGS:
                if links.metadata.get(key):
                    raise SystemExit(f"extract_links unsafe metadata {key}: {links.metadata}")

            def fail_fetch(_url, timeout=12):
                raise RuntimeError("network denied near /\x55sers/example/private/browser")

            browser._fetch = fail_fetch  # type: ignore[assignment]
            failed_fetch = fetch_page({"url": "https://example.com"})
            if failed_fetch.ok or "Could not fetch page." not in failed_fetch.output:
                raise SystemExit(f"fetch_page failure should return friendly output: {failed_fetch.output}")
            for fragment in ["setup check", "confirm network access", "retry the lookup"]:
                if fragment not in failed_fetch.output:
                    raise SystemExit(f"fetch_page failure missed recovery guidance {fragment!r}: {failed_fetch.output}")
            if "/\x55sers/operator" in failed_fetch.output or "network denied" in failed_fetch.output:
                raise SystemExit(f"fetch_page failure leaked raw exception text: {failed_fetch.output}")
            if failed_fetch.metadata.get("exception_type") != "RuntimeError" or not failed_fetch.metadata.get("external_network"):
                raise SystemExit(f"fetch_page failure missed diagnostic metadata: {failed_fetch.metadata}")
            assert_no_browser_authority(failed_fetch.metadata, "fetch_page failure")

            failed_links = extract_links({"url": "https://example.com"})
            if failed_links.ok or "Could not extract links." not in failed_links.output:
                raise SystemExit(f"extract_links failure should return friendly output: {failed_links.output}")
            for fragment in ["setup check", "confirm network access", "retry the lookup"]:
                if fragment not in failed_links.output:
                    raise SystemExit(f"extract_links failure missed recovery guidance {fragment!r}: {failed_links.output}")
            if "/\x55sers/operator" in failed_links.output or "network denied" in failed_links.output:
                raise SystemExit(f"extract_links failure leaked raw exception text: {failed_links.output}")
            if failed_links.metadata.get("exception_type") != "RuntimeError" or not failed_links.metadata.get("external_network"):
                raise SystemExit(f"extract_links failure missed diagnostic metadata: {failed_links.metadata}")
            assert_no_browser_authority(failed_links.metadata, "extract_links failure")

            direct_failed_fetch = browser.fetch_page({"url": "https://example.com"})
            if direct_failed_fetch.ok or "Could not fetch page." not in direct_failed_fetch.output:
                raise SystemExit(f"direct fetch_page failure should return friendly output: {direct_failed_fetch.output}")
            if "setup check" not in direct_failed_fetch.output or "retry" not in direct_failed_fetch.output:
                raise SystemExit(f"direct fetch_page failure missed recovery guidance: {direct_failed_fetch.output}")
            if "/\x55sers/operator" in direct_failed_fetch.output or "network denied" in direct_failed_fetch.output:
                raise SystemExit(f"direct fetch_page failure leaked raw exception text: {direct_failed_fetch.output}")
            if direct_failed_fetch.metadata.get("exception_type") != "RuntimeError":
                raise SystemExit(f"direct fetch_page failure missed diagnostic metadata: {direct_failed_fetch.metadata}")
            assert_no_browser_authority(direct_failed_fetch.metadata, "direct fetch_page failure")

            direct_failed_links = browser.extract_links({"url": "https://example.com"})
            if direct_failed_links.ok or "Could not extract links." not in direct_failed_links.output:
                raise SystemExit(f"direct extract_links failure should return friendly output: {direct_failed_links.output}")
            if "setup check" not in direct_failed_links.output or "retry" not in direct_failed_links.output:
                raise SystemExit(f"direct extract_links failure missed recovery guidance: {direct_failed_links.output}")
            if "/\x55sers/operator" in direct_failed_links.output or "network denied" in direct_failed_links.output:
                raise SystemExit(f"direct extract_links failure leaked raw exception text: {direct_failed_links.output}")
            if direct_failed_links.metadata.get("exception_type") != "RuntimeError":
                raise SystemExit(f"direct extract_links failure missed diagnostic metadata: {direct_failed_links.metadata}")
            assert_no_browser_authority(direct_failed_links.metadata, "direct extract_links failure")

            def raise_unexpected_fetch(_url, timeout=12):
                raise AssertionError("path-shaped browser inputs should be rejected before network fetch")

            browser._fetch = raise_unexpected_fetch  # type: ignore[assignment]
            path_fetch = fetch_page({"url": "file:///\x55sers/example/private/browser.html"})
            temp_path_fetch = fetch_page({"url": "file:///var/folders/zc/jarvis/browser.html"})
            path_links = extract_links({"url": "/private/tmp/jarvis-browser.html"})
            temp_path_links = extract_links({"url": "/tmp/jarvis-browser.html"})
            path_search = web_search({"query": "/\x55sers/example/private/browser-search"})
            temp_path_search = web_search({"query": "/var/folders/zc/jarvis/browser-search"})
            direct_path_fetch = browser.fetch_page({"url": "file:///\x55sers/example/private/browser.html"})
            direct_temp_path_fetch = browser.fetch_page({"url": "file:///var/folders/zc/jarvis/browser.html"})
            direct_path_links = browser.extract_links({"url": "/private/tmp/jarvis-browser.html"})
            direct_temp_path_links = browser.extract_links({"url": "/tmp/jarvis-browser.html"})
            direct_path_search = browser.web_search({"query": "/private/tmp/jarvis-browser-search"})
            direct_temp_path_search = browser.web_search({"query": "/var/folders/zc/jarvis-browser-search"})
            for label, path_result in {
                "fetch_page": path_fetch,
                "temp fetch_page": temp_path_fetch,
                "extract_links": path_links,
                "temp extract_links": temp_path_links,
                "web_search": path_search,
                "temp web_search": temp_path_search,
                "direct fetch_page": direct_path_fetch,
                "direct temp fetch_page": direct_temp_path_fetch,
                "direct extract_links": direct_path_links,
                "direct temp extract_links": direct_temp_path_links,
                "direct web_search": direct_path_search,
                "direct temp web_search": direct_temp_path_search,
            }.items():
                if path_result.ok:
                    raise SystemExit(f"{label} should reject local-path-shaped browser input.")
                if path_result.metadata.get("reason") not in {"invalid_url", "invalid_query"}:
                    raise SystemExit(f"{label} path refusal missed reason metadata: {path_result.metadata}")
                if path_result.metadata.get("url") not in (None, "<local-path>") or path_result.metadata.get("query") not in (None, "<local-path>"):
                    raise SystemExit(f"{label} path refusal missed redacted metadata: {path_result.metadata}")
                if path_result.metadata.get("external_network") or path_result.metadata.get("writes_database") or path_result.metadata.get("writes_files"):
                    raise SystemExit(f"{label} path refusal should be local and inert: {path_result.metadata}")
                assert_no_browser_authority(path_result.metadata, label)
                assert_no_local_path_leak(path_result, label)

            browser._fetch = lambda _url, timeout=12: (HTML, _url)  # type: ignore[assignment]

            search = web_search({"query": "jarvis browser smoke " + ("x" * 1000), "max_chars": -10})
            if not search.ok or search.metadata.get("max_chars") != 1:
                raise SystemExit("web_search should clamp low max_chars through fetch_page.")
            if len(search.metadata.get("url", "")) > browser.MAX_SEARCH_QUERY_CHARS + 80:
                raise SystemExit("web_search should bound oversized queries before building the URL.")

            pages = recent_pages({"limit": "bad"})
            if not pages.ok or pages.metadata.get("limit") != 10 or pages.metadata.get("external_network"):
                raise SystemExit("recent_browser_pages should sanitize limits and stay local.")
            assert_no_browser_authority(pages.metadata, "recent_browser_pages")
            for key in SAFE_FALSE_FLAGS:
                if pages.metadata.get(key):
                    raise SystemExit(f"recent_browser_pages unsafe metadata {key}: {pages.metadata}")
            bool_pages = recent_pages({"limit": True})
            if not bool_pages.ok or bool_pages.metadata.get("limit") != 10 or bool_pages.metadata.get("external_network"):
                raise SystemExit(f"recent_browser_pages should treat boolean limits as malformed defaults: {bool_pages.metadata}")
            assert_no_browser_authority(bool_pages.metadata, "recent_browser_pages boolean limit")
            for key in SAFE_FALSE_FLAGS:
                if bool_pages.metadata.get(key):
                    raise SystemExit(f"recent_browser_pages boolean limit unsafe metadata {key}: {bool_pages.metadata}")

            summary = summarize_page({})
            if not summary.ok or summary.metadata.get("writes_files") or summary.metadata.get("external_network"):
                raise SystemExit("summarize_page should summarize stored history without writing or network.")
            assert_no_browser_authority(summary.metadata, "summarize_page")
            for key in SAFE_FALSE_FLAGS:
                if summary.metadata.get(key):
                    raise SystemExit(f"summarize_page unsafe metadata {key}: {summary.metadata}")

            with patch("jarvis_v2.memory.obsidian._replace_text", wraps=obsidian._replace_text) as atomic_replace, patch.object(
                Path,
                "write_text",
                side_effect=AssertionError("save_page_note performed a second path-based write"),
            ):
                saved = save_page_note({"page_id": "latest"})
            if not saved.ok or not saved.metadata.get("writes_files") or not saved.metadata.get("writes_notes"):
                raise SystemExit("save_page_note should save a local note and mark file writes.")
            if atomic_replace.call_count != 1:
                raise SystemExit(f"save_page_note should perform exactly one atomic vault write, got {atomic_replace.call_count}.")
            assert_vault_relative_receipt(saved, root, "Sources/", "save_page_note")
            assert_no_browser_authority(saved.metadata, "save_page_note")
            for key in SAFE_FALSE_FLAGS:
                if saved.metadata.get(key):
                    raise SystemExit(f"save_page_note unsafe metadata {key}: {saved.metadata}")

            _, _, _, unavailable_recent, _, unavailable_save = browser.make_browser_tools(None, None)
            missing_history = unavailable_recent({})
            assert_browser_state_recovery(missing_history, "missing browser history dependency", "recent browser pages")
            missing_vault = unavailable_save({})
            assert_browser_state_recovery(missing_vault, "missing browser vault dependency", "save latest page note")
            if "JARVIS_OBSIDIAN_VAULT" not in missing_vault.output:
                raise SystemExit(f"Missing browser vault dependency should name its setting: {missing_vault.output}")

            store_type = type(runtime.store)
            original_history_save = store_type.save_browser_page

            def raising_history_save(_store, _url, _title, _text):
                raise OSError("history write denied near /\x55sers/example/private/browser.sqlite")

            store_type.save_browser_page = raising_history_save
            try:
                degraded_fetch = fetch_page({"url": "https://example.com"})
                degraded_links = extract_links({"url": "https://example.com"})
                browser._fetch = lambda _url, timeout=12: ("<html><title>No links</title><body>Enough readable text for a no-links history failure.</body></html>", _url)  # type: ignore[assignment]
                degraded_no_links = extract_links({"url": "https://example.com/no-links"})
            finally:
                browser._fetch = lambda _url, timeout=12: (HTML, _url)  # type: ignore[assignment]
                store_type.save_browser_page = original_history_save
            for degraded, label in [
                (degraded_fetch, "fetch with failed history write"),
                (degraded_links, "links with failed history write"),
                (degraded_no_links, "no-links result with failed history write"),
            ]:
                if not degraded.ok or "History warning" not in degraded.output or "setup check" not in degraded.output:
                    raise SystemExit(f"{label} should preserve fetched content with recovery warning: {degraded.output}")
                if degraded.metadata.get("history_saved") is not False or degraded.metadata.get("writes_database") is not False:
                    raise SystemExit(f"{label} should report unsaved history accurately: {degraded.metadata}")
                if degraded.metadata.get("history_exception_type") != "OSError":
                    raise SystemExit(f"{label} missed bounded history error type: {degraded.metadata}")
                if degraded.metadata.get("history_recovery_commands") != ["setup check"]:
                    raise SystemExit(f"{label} missed history recovery metadata: {degraded.metadata}")
                if degraded.metadata.get("history_retry_requires_storage_repair") is not True:
                    raise SystemExit(f"{label} should require storage repair before history retry: {degraded.metadata}")
                if degraded.metadata.get("authorizes_retry") is not False:
                    raise SystemExit(f"{label} should not authorize retry: {degraded.metadata}")
                assert_no_browser_authority(degraded.metadata, label)
                assert_no_local_path_leak(degraded, label)
                if "history write denied" in degraded.output:
                    raise SystemExit(f"{label} leaked backend error text: {degraded.output}")

            original_recent_pages = store_type.recent_browser_pages

            def raising_recent_pages(_store, _limit):
                raise OSError("history read denied near /private/tmp/browser.sqlite")

            store_type.recent_browser_pages = raising_recent_pages
            try:
                failed_recent = recent_pages({})
            finally:
                store_type.recent_browser_pages = original_recent_pages
            assert_browser_state_recovery(failed_recent, "browser history read failure", "recent browser pages")
            if failed_recent.metadata.get("exception_type") != "OSError":
                raise SystemExit(f"Browser history read failure missed bounded error type: {failed_recent.metadata}")

            original_get_page = store_type.get_browser_page

            def raising_get_page(_store, _page_id=None):
                raise PermissionError("history lookup denied near /var/folders/zc/browser.sqlite")

            store_type.get_browser_page = raising_get_page
            try:
                failed_summary = summarize_page({})
                failed_save_load = save_page_note({"page_id": "latest"})
            finally:
                store_type.get_browser_page = original_get_page
            assert_browser_state_recovery(
                failed_summary,
                "browser summary history failure",
                "summarize latest browser page",
            )
            assert_browser_state_recovery(
                failed_save_load,
                "browser note history failure",
                "save latest page note",
            )
            if "JARVIS_OBSIDIAN_VAULT" not in failed_save_load.output:
                raise SystemExit(f"Browser note history failure should name the vault setting: {failed_save_load.output}")

            vault_type = type(runtime.vault)
            original_write_source = vault_type.write_source

            def raising_write_source(_vault, _title, _body):
                raise PermissionError("vault write denied near /\x55sers/example/private/Vault")

            vault_type.write_source = raising_write_source
            try:
                failed_note_write = save_page_note({"page_id": "latest"})
            finally:
                vault_type.write_source = original_write_source
            assert_browser_state_recovery(
                failed_note_write,
                "browser note vault failure",
                "save latest page note",
            )
            if failed_note_write.metadata.get("exception_type") != "PermissionError":
                raise SystemExit(f"Browser note vault failure missed bounded error type: {failed_note_write.metadata}")
            if "JARVIS_OBSIDIAN_VAULT" not in failed_note_write.output:
                raise SystemExit(f"Browser note vault failure should name the vault setting: {failed_note_write.output}")

            bad_page = summarize_page({"page_id": "not-a-number"})
            if bad_page.ok or "page_id must be a number" not in bad_page.output:
                raise SystemExit("summarize_page should reject bad page ids.")
            if bad_page.metadata.get("raw_page_id") != "not-a-number":
                raise SystemExit("summarize_page should preserve bounded raw page id metadata.")
            assert_no_browser_authority(bad_page.metadata, "bad summarize_page")
            for key in SAFE_FALSE_FLAGS:
                if bad_page.metadata.get(key):
                    raise SystemExit(f"bad summarize_page unsafe metadata {key}: {bad_page.metadata}")

            long_bad_page = summarize_page({"page_id": "p" * 200})
            if long_bad_page.metadata.get("raw_page_id") != ("p" * 77 + "..."):
                raise SystemExit("summarize_page should bound raw page id metadata.")
            path_bad_page = summarize_page({"page_id": "/\x55sers/example/private/browser-page-id"})
            if path_bad_page.ok or path_bad_page.metadata.get("raw_page_id") != "<local-path>":
                raise SystemExit(f"summarize_page should redact path-shaped bad page ids: {path_bad_page.metadata}")
            temp_path_bad_page = summarize_page({"page_id": "/var/folders/zc/jarvis/browser-page-id"})
            if temp_path_bad_page.ok or temp_path_bad_page.metadata.get("raw_page_id") != "<local-path>":
                raise SystemExit(f"summarize_page should redact temp path-shaped bad page ids: {temp_path_bad_page.metadata}")
            assert_no_local_path_leak(temp_path_bad_page, "summarize_page temp page id")

            bad_save = save_page_note({"page_id": "not-a-number"})
            if bad_save.ok or "page_id must be a number" not in bad_save.output:
                raise SystemExit("save_page_note should reject bad page ids.")
            if bad_save.metadata.get("raw_page_id") != "not-a-number" or bad_save.metadata.get("writes_files"):
                raise SystemExit("save_page_note should preserve bounded raw page id metadata without writes.")
            path_bad_save = save_page_note({"page_id": "/private/tmp/jarvis-browser-page-id"})
            if path_bad_save.ok or path_bad_save.metadata.get("raw_page_id") != "<local-path>" or path_bad_save.metadata.get("writes_files"):
                raise SystemExit(f"save_page_note should redact path-shaped bad page ids without writes: {path_bad_save.metadata}")
            temp_path_bad_save = save_page_note({"page_id": "/tmp/jarvis-browser-page-id"})
            if temp_path_bad_save.ok or temp_path_bad_save.metadata.get("raw_page_id") != "<local-path>" or temp_path_bad_save.metadata.get("writes_files"):
                raise SystemExit(f"save_page_note should redact temp path-shaped bad page ids without writes: {temp_path_bad_save.metadata}")
            assert_no_local_path_leak(temp_path_bad_save, "save_page_note temp page id")

            with patch.object(subprocess, "run", side_effect=AssertionError("open_url launched a process")) as launcher:
                path_open = browser.open_url({"url": "file:///\x55sers/example/private/browser.html"})
                temp_path_open = browser.open_url({"url": "file:///tmp/jarvis-browser.html"})
                custom_scheme_opens = [
                    browser.open_url({"url": "javascript:alert(1)"}),
                    browser.open_url({"url": "ssh://public.example/"}),
                ]
            if launcher.call_count:
                raise SystemExit("Rejected open_url input unexpectedly launched a process")
            for label, open_result in {"open_url": path_open, "temp open_url": temp_path_open}.items():
                if open_result.ok or open_result.metadata.get("reason") != "invalid_url":
                    raise SystemExit(f"{label} should reject path-shaped URLs: {open_result.metadata}")
                if open_result.metadata.get("url") != "<local-path>" or open_result.metadata.get("controls_computer"):
                    raise SystemExit(f"{label} path refusal missed redacted inert metadata: {open_result.metadata}")
                assert_no_browser_authority(open_result.metadata, label)
                assert_no_local_path_leak(open_result, label)
            for open_result in custom_scheme_opens:
                if open_result.ok or open_result.metadata.get("reason") != "invalid_url":
                    raise SystemExit(f"open_url should reject non-HTTP schemes: {open_result}")
                if open_result.metadata.get("controls_computer") or open_result.metadata.get("external_network"):
                    raise SystemExit(f"Custom-scheme refusal should be local and inert: {open_result.metadata}")

        finally:
            browser._fetch = original_fetch  # type: ignore[assignment]


if __name__ == "__main__":
    main()
