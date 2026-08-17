from __future__ import annotations

import html
import ipaddress
import re
import socket
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import HTTPHandler, HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from jarvis_v2.agent.failure_guidance import (
    declare_failure_guidance,
    declare_retryable_external_information_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore


USER_AGENT = "JarvisV2/0.1 (+local personal assistant)"
MAX_BROWSER_TEXT_CHARS = 20000
MAX_BROWSER_LINKS = 80
MAX_BROWSER_HISTORY_LIMIT = 100
MAX_BROWSER_URL_CHARS = 2048
MAX_BROWSER_RESPONSE_BYTES = 5_000_000
MAX_BROWSER_TITLE_CHARS = 300
MAX_SEARCH_QUERY_CHARS = 500
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
HOST_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _short_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_metadata(value: Any, limit: int) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short_text(value, limit))


def _has_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "calls_model": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "external_network": False,
        "external_side_effect": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "speaks": False,
        "completes_tasks": False,
    }
    metadata.update(extra)
    return metadata


def _safe_vault_path_display(path: str | Path | None, vault: ObsidianVault) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(vault.root_path))
    except ValueError:
        return _short_metadata(candidate, 160)


def _browser_error(action: str, exc: Exception | None = None) -> str:
    return (
        f"Could not {action}. Run `setup check`, confirm network access, then retry the lookup."
    )


def _browser_public_read_failure(
    tool_name: str,
    output: str,
    *,
    action: str,
    reason: str,
    external_network: bool = False,
    commands: tuple[str, ...] = (),
    **metadata: Any,
) -> ToolResult:
    return ToolResult(
        tool_name,
        False,
        output,
        declare_retryable_external_information_failure(
            _safe_metadata(
                reason=reason,
                external_network=external_network,
                **metadata,
            ),
            output=output,
            action=action,
            commands=commands or (("setup check",) if "`setup check`" in output else ()),
        ),
    )


def _browser_state_error(
    tool_name: str,
    action: str,
    retry_command: str,
    *,
    exc: Exception | None = None,
    needs_vault: bool = False,
) -> ToolResult:
    settings = "`JARVIS_DATA_DIR` and `JARVIS_DB_PATH`"
    if needs_vault:
        settings += " plus `JARVIS_OBSIDIAN_VAULT`"
    detail = f" ({type(exc).__name__})" if exc is not None else ""
    recovery_action = (
        f"Run `setup check`, verify {settings} point to writable local locations, "
        f"then retry `{retry_command}`."
    )
    output = f"Could not {action}. {recovery_action}{detail}"
    return ToolResult(
        tool_name,
        False,
        output,
        declare_failure_guidance(
            _safe_metadata(
                reason="browser_state_unavailable",
                exception_type=type(exc).__name__ if exc is not None else "",
                retry_requires_storage_repair=True,
                authorizes_retry=False,
            ),
            output=output,
            action=recovery_action,
            commands=("setup check", retry_command),
        ),
    )


def _browser_history_warning(action: str, exc: Exception) -> str:
    return (
        f"History warning: {action} succeeded, but Jarvis could not save it to browser history. "
        "Run `setup check`, verify `JARVIS_DATA_DIR` and `JARVIS_DB_PATH` are writable, then retry "
        "the request if history is required."
    )


def _as_tool_result(tool_name: str, result: ToolResult) -> ToolResult:
    """Preserve delegated output while reporting the tool the caller invoked."""
    return ToolResult(tool_name, result.ok, result.output, dict(result.metadata))


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str
    links: list[tuple[str, str]]


def _normalize_url(url: str) -> str:
    url = _short_text(url, MAX_BROWSER_URL_CHARS)
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme:
        return "https://" + url
    return url


class UnsafeBrowserURL(ValueError):
    pass


@dataclass(frozen=True)
class _ResolvedAddress:
    family: int
    socktype: int
    proto: int
    sockaddr: tuple[Any, ...]


def _is_unsafe_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        not address.is_global
        or address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    )


def _validate_public_http_url(url: str, *, resolve: bool = True) -> tuple[_ResolvedAddress, ...]:
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise UnsafeBrowserURL("invalid URL") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise UnsafeBrowserURL("unsupported URL scheme")
    if not host or not parsed.netloc:
        raise UnsafeBrowserURL("missing URL host")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeBrowserURL("embedded URL credentials are not allowed")

    host = host.lower().rstrip(".")
    if not host or host == "localhost" or host.endswith(".localhost"):
        raise UnsafeBrowserURL("local URL host is not allowed")

    if "%" in host:
        raise UnsafeBrowserURL("invalid URL host")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise UnsafeBrowserURL("invalid URL host") from exc
        labels = ascii_host.split(".")
        if len(ascii_host) > 253 or any(not HOST_LABEL_RE.fullmatch(label) for label in labels):
            raise UnsafeBrowserURL("invalid URL host")
    else:
        if _is_unsafe_ip(literal):
            raise UnsafeBrowserURL("non-public IP address is not allowed")
        if not resolve:
            return ()
        resolved_port = port or (443 if parsed.scheme.lower() == "https" else 80)
        if literal.version == 6:
            return (
                _ResolvedAddress(
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    (str(literal), resolved_port, 0, 0),
                ),
            )
        return (
            _ResolvedAddress(
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                (str(literal), resolved_port),
            ),
        )

    if not resolve:
        return ()
    try:
        addresses = socket.getaddrinfo(
            ascii_host,
            port or (443 if parsed.scheme.lower() == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise UnsafeBrowserURL("URL host could not be resolved safely") from exc
    if not addresses:
        raise UnsafeBrowserURL("URL host did not resolve")
    validated = []
    for family, socktype, proto, _canonname, sockaddr in addresses:
        if family not in {socket.AF_INET, socket.AF_INET6} or socktype != socket.SOCK_STREAM:
            raise UnsafeBrowserURL("URL host returned an unsupported address")
        try:
            resolved = ipaddress.ip_address(sockaddr[0].split("%", 1)[0])
        except (IndexError, ValueError) as exc:
            raise UnsafeBrowserURL("URL host returned an invalid address") from exc
        if _is_unsafe_ip(resolved):
            raise UnsafeBrowserURL("URL host resolved to a non-public address")
        validated.append(_ResolvedAddress(family, socktype, proto, tuple(sockaddr)))
    return tuple(validated)


def _normalize_and_validate_url(url: str, *, resolve: bool = False) -> str:
    normalized = _normalize_url(url)
    if normalized:
        _validate_public_http_url(normalized, resolve=resolve)
    return normalized


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        addresses = _validate_public_http_url(redirected.full_url)
        setattr(redirected, "_jarvis_validated_addresses", addresses)
        return redirected


def _request_addresses(request: Request) -> tuple[_ResolvedAddress, ...]:
    addresses = getattr(request, "_jarvis_validated_addresses", None)
    if addresses is None:
        addresses = _validate_public_http_url(request.full_url)
        setattr(request, "_jarvis_validated_addresses", addresses)
    return addresses


def _connect_to_validated_address(connection) -> None:
    last_error: OSError | None = None
    for address in connection._jarvis_validated_addresses:
        sock = None
        try:
            sock = socket.socket(address.family, address.socktype, address.proto)
            if connection.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(connection.timeout)
            if connection.source_address:
                sock.bind(connection.source_address)
            sock.connect(address.sockaddr)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            connection.sock = sock
            return
        except OSError as exc:
            last_error = exc
            if sock is not None:
                sock.close()
    if last_error is not None:
        raise last_error
    raise OSError("URL host did not provide a validated address")


class _PinnedHTTPConnection(HTTPConnection):
    def __init__(self, *args, validated_addresses: tuple[_ResolvedAddress, ...], **kwargs):
        super().__init__(*args, **kwargs)
        self._jarvis_validated_addresses = validated_addresses

    def connect(self):
        _connect_to_validated_address(self)
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(HTTPSConnection):
    def __init__(self, *args, validated_addresses: tuple[_ResolvedAddress, ...], **kwargs):
        super().__init__(*args, **kwargs)
        self._jarvis_validated_addresses = validated_addresses

    def connect(self):
        _connect_to_validated_address(self)
        if self._tunnel_host:
            self._tunnel()
        server_hostname = self._tunnel_host or self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


class _PinnedHTTPHandler(HTTPHandler):
    def http_open(self, req):
        return self.do_open(_PinnedHTTPConnection, req, validated_addresses=_request_addresses(req))


class _PinnedHTTPSHandler(HTTPSHandler):
    def https_open(self, req):
        return self.do_open(
            _PinnedHTTPSConnection,
            req,
            context=self._context,
            validated_addresses=_request_addresses(req),
        )


_SAFE_URL_OPENER = build_opener(
    ProxyHandler({}),
    _PinnedHTTPHandler(),
    _PinnedHTTPSHandler(),
    _SafeRedirectHandler(),
)


def urlopen(request: Request, *, timeout: int):
    return _SAFE_URL_OPENER.open(request, timeout=timeout)


def _fetch(url: str, timeout: int = 12) -> tuple[str, str]:
    addresses = _validate_public_http_url(url)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    setattr(request, "_jarvis_validated_addresses", addresses)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_BROWSER_RESPONSE_BYTES + 1)
        if len(raw) > MAX_BROWSER_RESPONSE_BYTES:
            raise ValueError("browser response too large")
        final_url = response.geturl()
        _validate_public_http_url(final_url, resolve=False)
        charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace"), final_url


def _parse_page(url: str, content: str) -> Page:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", content, flags=re.I | re.S)
    title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip()) if title_match else url
    title = _short_text(title, MAX_BROWSER_TITLE_CHARS)

    no_script = re.sub(r"<script[^>]*>.*?</script>", " ", content, flags=re.I | re.S)
    no_style = re.sub(r"<style[^>]*>.*?</style>", " ", no_script, flags=re.I | re.S)

    links = []
    for match in re.finditer(r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", no_style, flags=re.I | re.S):
        href = urljoin(url, html.unescape(match.group(1)))
        label = html.unescape(re.sub(r"<[^>]+>", " ", match.group(2)))
        label = re.sub(r"\s+", " ", label).strip()
        if href and label:
            links.append((label[:120], href))

    text = re.sub(r"<[^>]+>", " ", no_style)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return Page(url=url, title=title, text=text, links=links[:80])


def _summarize_text(title: str, text: str, max_bullets: int = 6) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for sentence in sentences:
        clean = sentence.strip()
        if len(clean) < 40:
            continue
        score = min(len(clean), 240)
        for marker in ["important", "key", "new", "how", "why", "because", "announced", "released"]:
            if marker in clean.lower():
                score += 60
        scored.append((score, clean))
    scored.sort(reverse=True)
    bullets = [sentence[:260] for _, sentence in scored[:max_bullets]]
    if not bullets:
        bullets = [text[:500] or "No readable text found."]
    return "# " + title + "\n\n" + "\n".join(f"- {bullet}" for bullet in bullets)


def make_browser_tools(store: MemoryStore | None = None, vault: ObsidianVault | None = None):
    def fetch_page(args: dict[str, Any]) -> ToolResult:
        raw_url = str(args.get("url") or "")
        if _has_local_path(raw_url):
            output = "URL cannot be a local file path. Provide a public HTTP or HTTPS URL, then retry."
            return _browser_public_read_failure(
                "fetch_page",
                output,
                action="Provide a public HTTP or HTTPS URL, then retry.",
                reason="invalid_url",
                url="<local-path>",
            )
        try:
            url = _normalize_and_validate_url(raw_url)
        except UnsafeBrowserURL:
            output = (
                "URL must be a public HTTP or HTTPS URL without embedded credentials. "
                "Remove credentials or private addresses, then retry."
            )
            return _browser_public_read_failure(
                "fetch_page",
                output,
                action="Remove credentials or private addresses, then retry.",
                reason="invalid_url",
            )
        max_chars = _bounded_int(args.get("max_chars"), 4000, 1, MAX_BROWSER_TEXT_CHARS)
        if not url:
            output = "URL is required. Provide a public HTTP or HTTPS URL, then retry."
            return _browser_public_read_failure(
                "fetch_page",
                output,
                action="Provide a public HTTP or HTTPS URL, then retry.",
                reason="missing_url",
            )
        try:
            content, final_url = _fetch(url)
            page = _parse_page(final_url, content)
        except Exception as exc:
            output = _browser_error("fetch page", exc)
            return _browser_public_read_failure(
                "fetch_page",
                output,
                action="Run `setup check`, confirm network access, then retry the lookup.",
                reason="fetch_failed",
                external_network=True,
                url=url,
                exception_type=type(exc).__name__,
            )
        history_saved = False
        history_exception_type = ""
        history_warning = ""
        if store is not None:
            try:
                store.save_browser_page(page.url, page.title, page.text[:20000])
                history_saved = True
            except Exception as exc:
                history_exception_type = type(exc).__name__
                history_warning = _browser_history_warning("Page fetch", exc)
        text = page.text[:max_chars]
        truncated = len(page.text) > max_chars
        if truncated:
            text += f"\n\n... truncated {len(page.text) - max_chars} chars"
        if history_warning:
            text += "\n\n" + history_warning
        return ToolResult(
            "fetch_page",
            True,
            f"# {page.title}\n{page.url}\n\n{text}",
            _safe_metadata(
                url=page.url,
                title=page.title,
                max_chars=max_chars,
                truncated=truncated,
                writes_database=history_saved,
                history_saved=history_saved,
                history_exception_type=history_exception_type,
                history_recovery_commands=[] if history_saved or store is None else ["setup check"],
                history_retry_requires_storage_repair=bool(history_warning),
                authorizes_retry=False,
                external_network=True,
            ),
        )

    def extract_links(args: dict[str, Any]) -> ToolResult:
        raw_url = str(args.get("url") or "")
        if _has_local_path(raw_url):
            output = "URL cannot be a local file path. Provide a public HTTP or HTTPS URL, then retry."
            return _browser_public_read_failure(
                "extract_links",
                output,
                action="Provide a public HTTP or HTTPS URL, then retry.",
                reason="invalid_url",
                url="<local-path>",
            )
        try:
            url = _normalize_and_validate_url(raw_url)
        except UnsafeBrowserURL:
            output = (
                "URL must be a public HTTP or HTTPS URL without embedded credentials. "
                "Remove credentials or private addresses, then retry."
            )
            return _browser_public_read_failure(
                "extract_links",
                output,
                action="Remove credentials or private addresses, then retry.",
                reason="invalid_url",
            )
        if not url:
            output = "URL is required. Provide a public HTTP or HTTPS URL, then retry."
            return _browser_public_read_failure(
                "extract_links",
                output,
                action="Provide a public HTTP or HTTPS URL, then retry.",
                reason="missing_url",
            )
        try:
            content, final_url = _fetch(url)
            page = _parse_page(final_url, content)
        except Exception as exc:
            output = _browser_error("extract links", exc)
            return _browser_public_read_failure(
                "extract_links",
                output,
                action="Run `setup check`, confirm network access, then retry the lookup.",
                reason="fetch_failed",
                external_network=True,
                url=url,
                exception_type=type(exc).__name__,
            )
        history_saved = False
        history_exception_type = ""
        history_warning = ""
        if store is not None:
            try:
                store.save_browser_page(page.url, page.title, page.text[:20000])
                history_saved = True
            except Exception as exc:
                history_exception_type = type(exc).__name__
                history_warning = _browser_history_warning("Link extraction", exc)
        if not page.links:
            output = f"No links found on {page.url}."
            if history_warning:
                output += "\n\n" + history_warning
            return ToolResult(
                "extract_links",
                True,
                output,
                _safe_metadata(
                    count=0,
                    url=page.url,
                    writes_database=history_saved,
                    history_saved=history_saved,
                    history_exception_type=history_exception_type,
                    history_recovery_commands=[] if history_saved or store is None else ["setup check"],
                    history_retry_requires_storage_repair=bool(history_warning),
                    authorizes_retry=False,
                    external_network=True,
                ),
            )
        lines = [f"- {label}: {href}" for label, href in page.links]
        if history_warning:
            lines.extend(["", history_warning])
        return ToolResult(
            "extract_links",
            True,
            "\n".join(lines),
            _safe_metadata(
                count=len(page.links),
                max_links=MAX_BROWSER_LINKS,
                url=page.url,
                writes_database=history_saved,
                history_saved=history_saved,
                history_exception_type=history_exception_type,
                history_recovery_commands=[] if history_saved or store is None else ["setup check"],
                history_retry_requires_storage_repair=bool(history_warning),
                authorizes_retry=False,
                external_network=True,
            ),
        )

    def web_search(args: dict[str, Any]) -> ToolResult:
        query = _short_text(args.get("query"), MAX_SEARCH_QUERY_CHARS)
        if not query:
            output = "Search query is empty. Provide a public-information search query, then retry."
            return _browser_public_read_failure(
                "web_search",
                output,
                action="Provide a public-information search query, then retry.",
                reason="missing_query",
            )
        if _has_local_path(query):
            output = (
                "Search query cannot be a local file path. Replace it with a public-information "
                "search query, then retry."
            )
            return _browser_public_read_failure(
                "web_search",
                output,
                action="Replace it with a public-information search query, then retry.",
                reason="invalid_query",
                query="<local-path>",
            )
        engine = str(args.get("engine") or "duckduckgo").lower()
        if engine == "google":
            url = "https://www.google.com/search?q=" + quote_plus(query)
        else:
            url = "https://duckduckgo.com/html/?q=" + quote_plus(query)
        fetched = fetch_page(
            {"url": url, "max_chars": _bounded_int(args.get("max_chars"), 5000, 1, MAX_BROWSER_TEXT_CHARS)}
        )
        return _as_tool_result("web_search", fetched)

    def recent_pages(args: dict[str, Any]) -> ToolResult:
        if store is None:
            return _browser_state_error(
                "recent_browser_pages",
                "load browser history",
                "recent browser pages",
            )
        limit = _bounded_int(args.get("limit"), 10, 1, MAX_BROWSER_HISTORY_LIMIT)
        try:
            rows = store.recent_browser_pages(limit)
        except Exception as exc:
            return _browser_state_error(
                "recent_browser_pages",
                "load browser history",
                "recent browser pages",
                exc=exc,
            )
        if not rows:
            return ToolResult(
                "recent_browser_pages",
                True,
                "No fetched browser pages yet.",
                _safe_metadata(count=0, limit=limit),
            )
        lines = [f"- #{row['id']} {row['title']} | {row['url']} | {row['fetched_at']}" for row in rows]
        return ToolResult(
            "recent_browser_pages",
            True,
            "\n".join(lines),
            _safe_metadata(count=len(rows), limit=limit),
        )

    def summarize_page(args: dict[str, Any]) -> ToolResult:
        page_id = args.get("page_id")
        row = None
        if store is not None:
            try:
                row = store.get_browser_page(int(page_id)) if page_id is not None else store.get_browser_page()
            except (TypeError, ValueError):
                return ToolResult("summarize_page", False, "page_id must be a number.", _safe_metadata(raw_page_id=_short_metadata(page_id, 80)))
            except Exception as exc:
                return _browser_state_error(
                    "summarize_page",
                    "load the browser page for summary",
                    "summarize latest browser page",
                    exc=exc,
                )
        if row is None:
            url = str(args.get("url") or "")
            if not url:
                return ToolResult("summarize_page", False, "No browser page history yet. Provide url or fetch a page first.", _safe_metadata())
            fetched = fetch_page({"url": url, "max_chars": 20000})
            if not fetched.ok:
                return fetched
            if store is not None:
                try:
                    row = store.get_browser_page()
                except Exception as exc:
                    return _browser_state_error(
                        "summarize_page",
                        "load the fetched page for summary",
                        "summarize latest browser page",
                        exc=exc,
                    )
        if row is None:
            return ToolResult(
                "summarize_page",
                False,
                "Could not load a page for summary. Run `recent browser pages`; if the page is absent, "
                "fetch its URL, then retry `summarize latest browser page`.",
                _safe_metadata(
                    reason="page_not_found",
                    next_command="recent browser pages",
                    recovery_commands=["recent browser pages", "summarize latest browser page"],
                    authorizes_retry=False,
                ),
            )
        summary = _summarize_text(row["title"], row["text"])
        return ToolResult(
            "summarize_page",
            True,
            summary,
            _safe_metadata(page_id=row["id"], url=row["url"]),
        )

    def save_page_note(args: dict[str, Any]) -> ToolResult:
        if store is None or vault is None:
            return _browser_state_error(
                "save_page_note",
                "access browser history or the Jarvis note vault",
                "save latest page note",
                needs_vault=True,
            )
        raw_page_id = args.get("page_id")
        try:
            page_id = int(raw_page_id) if raw_page_id not in (None, "", "latest") else None
        except (TypeError, ValueError):
            return ToolResult("save_page_note", False, "page_id must be a number.", _safe_metadata(raw_page_id=_short_metadata(raw_page_id, 80)))
        try:
            row = store.get_browser_page(page_id)
        except Exception as exc:
            return _browser_state_error(
                "save_page_note",
                "load the browser page for note saving",
                "save latest page note",
                exc=exc,
                needs_vault=True,
            )
        if row is None:
            return ToolResult(
                "save_page_note",
                False,
                "No browser page found. Run `recent browser pages`; fetch the page first if needed, "
                "then retry `save latest page note`.",
                _safe_metadata(
                    reason="page_not_found",
                    next_command="recent browser pages",
                    recovery_commands=["recent browser pages", "save latest page note"],
                    authorizes_retry=False,
                ),
            )
        summary = _summarize_text(row["title"], row["text"])
        body = f"Source: {row['url']}\nFetched: {row['fetched_at']}\n\n{summary}"
        try:
            path = vault.write_source(row["title"], body)
        except Exception as exc:
            return _browser_state_error(
                "save_page_note",
                "write the browser page note",
                "save latest page note",
                exc=exc,
                needs_vault=True,
            )
        path_display = _safe_vault_path_display(path, vault)
        return ToolResult(
            "save_page_note",
            True,
            f"Saved page note: {path_display}",
            _safe_metadata(
                path=str(path),
                path_display=path_display,
                page_id=row["id"],
                writes_files=True,
                writes_notes=True,
            ),
        )

    return fetch_page, extract_links, web_search, recent_pages, summarize_page, save_page_note


def fetch_page(args: dict[str, Any]) -> ToolResult:
    raw_url = str(args.get("url") or "")
    if _has_local_path(raw_url):
        output = "URL cannot be a local file path. Provide a public HTTP or HTTPS URL, then retry."
        return _browser_public_read_failure(
            "fetch_page",
            output,
            action="Provide a public HTTP or HTTPS URL, then retry.",
            reason="invalid_url",
            url="<local-path>",
        )
    try:
        url = _normalize_and_validate_url(raw_url)
    except UnsafeBrowserURL:
        output = (
            "URL must be a public HTTP or HTTPS URL without embedded credentials. "
            "Remove credentials or private addresses, then retry."
        )
        return _browser_public_read_failure(
            "fetch_page",
            output,
            action="Remove credentials or private addresses, then retry.",
            reason="invalid_url",
        )
    max_chars = _bounded_int(args.get("max_chars"), 4000, 1, MAX_BROWSER_TEXT_CHARS)
    if not url:
        output = "URL is required. Provide a public HTTP or HTTPS URL, then retry."
        return _browser_public_read_failure(
            "fetch_page",
            output,
            action="Provide a public HTTP or HTTPS URL, then retry.",
            reason="missing_url",
        )
    try:
        content, final_url = _fetch(url)
        page = _parse_page(final_url, content)
    except Exception as exc:
        output = _browser_error("fetch page", exc)
        return _browser_public_read_failure(
            "fetch_page",
            output,
            action="Run `setup check`, confirm network access, then retry the lookup.",
            reason="fetch_failed",
            external_network=True,
            url=url,
            exception_type=type(exc).__name__,
        )
    text = page.text[:max_chars]
    truncated = len(page.text) > max_chars
    if truncated:
        text += f"\n\n... truncated {len(page.text) - max_chars} chars"
    return ToolResult(
        "fetch_page",
        True,
        f"# {page.title}\n{page.url}\n\n{text}",
        _safe_metadata(
            url=page.url,
            title=page.title,
            max_chars=max_chars,
            truncated=truncated,
            external_network=True,
        ),
    )


def extract_links(args: dict[str, Any]) -> ToolResult:
    raw_url = str(args.get("url") or "")
    if _has_local_path(raw_url):
        output = "URL cannot be a local file path. Provide a public HTTP or HTTPS URL, then retry."
        return _browser_public_read_failure(
            "extract_links",
            output,
            action="Provide a public HTTP or HTTPS URL, then retry.",
            reason="invalid_url",
            url="<local-path>",
        )
    try:
        url = _normalize_and_validate_url(raw_url)
    except UnsafeBrowserURL:
        output = (
            "URL must be a public HTTP or HTTPS URL without embedded credentials. "
            "Remove credentials or private addresses, then retry."
        )
        return _browser_public_read_failure(
            "extract_links",
            output,
            action="Remove credentials or private addresses, then retry.",
            reason="invalid_url",
        )
    if not url:
        output = "URL is required. Provide a public HTTP or HTTPS URL, then retry."
        return _browser_public_read_failure(
            "extract_links",
            output,
            action="Provide a public HTTP or HTTPS URL, then retry.",
            reason="missing_url",
        )
    try:
        content, final_url = _fetch(url)
        page = _parse_page(final_url, content)
    except Exception as exc:
        output = _browser_error("extract links", exc)
        return _browser_public_read_failure(
            "extract_links",
            output,
            action="Run `setup check`, confirm network access, then retry the lookup.",
            reason="fetch_failed",
            external_network=True,
            url=url,
            exception_type=type(exc).__name__,
        )
    if not page.links:
        return ToolResult(
            "extract_links",
            True,
            f"No links found on {page.url}.",
            _safe_metadata(count=0, url=page.url, external_network=True),
        )
    lines = [f"- {label}: {href}" for label, href in page.links]
    return ToolResult(
        "extract_links",
        True,
        "\n".join(lines),
        _safe_metadata(
            count=len(page.links),
            max_links=MAX_BROWSER_LINKS,
            url=page.url,
            external_network=True,
        ),
    )


def open_url(args: dict[str, Any]) -> ToolResult:
    raw_url = str(args.get("url") or "")
    if _has_local_path(raw_url):
        output = "URL cannot be a local file path. Provide a public HTTP or HTTPS URL, then retry."
        return _browser_public_read_failure(
            "open_url",
            output,
            action="Provide a public HTTP or HTTPS URL, then retry.",
            reason="invalid_url",
            url="<local-path>",
            controls_computer=False,
        )
    try:
        url = _normalize_and_validate_url(raw_url)
    except UnsafeBrowserURL:
        output = (
            "URL must be a public HTTP or HTTPS URL without embedded credentials. "
            "Remove credentials or private addresses, then retry."
        )
        return _browser_public_read_failure(
            "open_url",
            output,
            action="Remove credentials or private addresses, then retry.",
            reason="invalid_url",
            controls_computer=False,
        )
    if not url:
        output = "URL is required. Provide a public HTTP or HTTPS URL, then retry."
        return _browser_public_read_failure(
            "open_url",
            output,
            action="Provide a public HTTP or HTTPS URL, then retry.",
            reason="missing_url",
            controls_computer=False,
        )
    recovery_command = f"fetch page {url}"
    output = (
        "Opening URLs in the OS browser is disabled because the browser can re-resolve hosts or follow "
        f"redirects outside Jarvis containment. Use `{recovery_command}` to retrieve the page through "
        "Jarvis' pinned public-address fetch path."
    )
    return _browser_public_read_failure(
        "open_url",
        output,
        action=f"Use `{recovery_command}` to retrieve the page through Jarvis' pinned public-address fetch path.",
        reason="os_browser_launch_disabled",
        commands=(recovery_command,),
        url=url,
        pinned_fetch_available=True,
        controls_computer=False,
    )


def web_search(args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not query:
        output = "Search query is empty. Provide a public-information search query, then retry."
        return _browser_public_read_failure(
            "web_search",
            output,
            action="Provide a public-information search query, then retry.",
            reason="missing_query",
        )
    if _has_local_path(query):
        output = (
            "Search query cannot be a local file path. Replace it with a public-information "
            "search query, then retry."
        )
        return _browser_public_read_failure(
            "web_search",
            output,
            action="Replace it with a public-information search query, then retry.",
            reason="invalid_query",
            query="<local-path>",
        )
    # Structured results via DuckDuckGo Lite (the HTML endpoint is bot-blocked).
    try:
        from jarvis_v2.tools.research_connector import ddg_search
        results = ddg_search(query, max_results=_bounded_int(args.get("max_results"), 5, 1, 10))
        if results:
            lines = [f"Results for '{query}':"]
            for r in results:
                lines.append(f"- {r['title']} ({r['url']})")
                if r.get("snippet"):
                    lines.append(f"  {r['snippet'][:200]}")
            return ToolResult(
                "web_search",
                True,
                "\n".join(lines),
                _safe_metadata(count=len(results), external_network=True),
            )
    except Exception:
        pass
    # Fallback: fetch the lite results page as text.
    url = "https://lite.duckduckgo.com/lite/?q=" + quote_plus(query)
    fetched = fetch_page(
        {"url": url, "max_chars": _bounded_int(args.get("max_chars"), 5000, 1, MAX_BROWSER_TEXT_CHARS)}
    )
    return _as_tool_result("web_search", fetched)
