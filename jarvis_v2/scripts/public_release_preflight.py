"""Read-only privacy preflight for an already-prepared public release tree.

This module never creates an export, initializes a repository, copies runtime
state, or publishes anything.  It only walks the explicitly supplied directory
without following symlinks and reports content-free findings.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import stat
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


POLICY_VERSION = "3"
EXTRA_DENY_ENV = "JARVIS_PUBLIC_RELEASE_DENY_LITERALS_JSON"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_SCAN_FILES = 2_000
MAX_SCAN_ENTRIES = 4_096
MAX_SCAN_TOTAL_BYTES = 64 * 1024 * 1024
MAX_DIRECTORY_ENTRIES = 2_048
MAX_EXTRA_DENY_LITERALS = 64
MAX_EXTRA_DENY_LITERAL_BYTES = 512
MAX_VARIANT_SCAN_INPUT_BYTES = 3 * 1024 * 1024
MAX_VARIANT_SCAN_FILE_COUNT = 2_000
MAX_VARIANT_SCAN_TOTAL_INPUT_BYTES = 64 * 1024 * 1024
MAX_NORMALIZED_DENY_LITERAL_CHARS = 20 * MAX_EXTRA_DENY_LITERAL_BYTES
MAX_NORMALIZED_VARIANT_CHARS = 20 * MAX_VARIANT_SCAN_INPUT_BYTES
_READ_CHUNK_BYTES = 64 * 1024
_OPEN_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_OPEN_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)

PRIVATE_DOCUMENT_BASENAMES = frozenset(
    {
        "CODEX_TASKS.md",
        "FINISH_PLAN_AUGUST.md",
        "LIVE_TEST_RUNBOOK.md",
        "V3_BACKLOG_PRIVATE.md",
    }
)
FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {
        ".aws",
        ".azure",
        ".docker",
        ".gcloud",
        ".git",
        ".gnupg",
        ".jarvis_v2_runtime",
        ".jarvis_v3_runtime",
        ".jarvis_v3_durable",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".ssh",
        ".kube",
        "Jarvis Vault",
        "__pycache__",
        "htmlcov",
    }
)
SENSITIVE_FILE_BASENAMES = frozenset(
    {
        ".git-credentials",
        ".envrc",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "Cookies",
        "Login Data",
        "credentials",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
FORBIDDEN_FILE_SUFFIXES = (
    ".bak",
    ".backup",
    ".cer",
    ".crt",
    ".db",
    ".key",
    ".log",
    ".p12",
    ".pem",
    ".pid",
    ".pyc",
    ".sqlite",
    ".sqlite3",
)
OPAQUE_ARCHIVE_SUFFIXES = (
    ".7z",
    ".bz2",
    ".dmg",
    ".gz",
    ".jar",
    ".pkg",
    ".rar",
    ".tar",
    ".tgz",
    ".whl",
    ".xz",
    ".zip",
)
COMPILED_ARTIFACT_SUFFIXES = (
    ".a",
    ".app",
    ".class",
    ".dylib",
    ".exe",
    ".o",
    ".so",
)
SERVICE_PLIST_NAME = re.compile(r"^com\.jarvis(?:-v[0-9]+)?\..+\.plist$")
TOKENISH_FILENAME = re.compile(
    r"(?:credential|secret|token).*(?:\.json|\.txt|\.pem|\.key)$",
    re.IGNORECASE,
)

# Split the macOS home prefix so this scanner does not flag its own source.
ABSOLUTE_USERS_PREFIX = b"/" + b"Users" + b"/"
EMAIL_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9._%+-])"
    rb"[A-Za-z0-9._%+-]{1,64}@"
    rb"[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,63}"
)
RESERVED_EMAIL_DOMAINS = frozenset(
    {b"example.com", b"example.net", b"example.org"}
)
CREDENTIAL_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    # OpenAI and Anthropic keys. Anthropic's provider-specific prefix remains
    # explicit even though legacy ``sk-`` coverage also catches current keys.
    re.compile(rb"(?<![A-Za-z0-9_-])sk-ant-(?:api[0-9]{2}-)?[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?<![A-Za-z0-9])sk-(?:proj-|live-)?[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{20,}"),
    # GitLab publishes these fixed prefixes for its token families. Custom
    # administrator-selected PAT prefixes cannot be distinguished safely.
    re.compile(
        rb"(?<![A-Za-z0-9_-])"
        rb"(?:glpat|gloas|gldt|glrt|glrtr|glcbt|glptt|glft|glimt|"
        rb"glagent|glwt|glsoat|glffct)-[A-Za-z0-9_-]{16,}"
        rb"(?![A-Za-z0-9_-])"
    ),
    # npm's older prefixed form and current hexadecimal tokens only when
    # attached to an npm-specific configuration key. Bare hex is ambiguous.
    re.compile(rb"(?<![A-Za-z0-9_])npm_[A-Za-z0-9]{36}(?![A-Za-z0-9])"),
    re.compile(
        rb"(?im)(?<![A-Za-z0-9_])(?:NPM_TOKEN|_authToken)[\"']?"
        rb"[ \t]*[=:][ \t]*[\"']?[A-Fa-f0-9]{32,64}(?![A-Fa-f0-9])"
    ),
    # PyPI documents an 85-or-more URL-safe base64 Macaroon after ``pypi-``.
    re.compile(rb"(?<![A-Za-z0-9_-])pypi-[A-Za-z0-9_-]{85,}(?![A-Za-z0-9_-])"),
    # Hugging Face documents the provider-specific ``hf_`` prefix. Requiring
    # a substantial opaque body avoids flagging prose, variable names, and
    # placeholders such as ``hf_...``.
    re.compile(rb"(?<![A-Za-z0-9_])hf_[A-Za-z0-9_]{30,}(?![A-Za-z0-9_])"),
    re.compile(rb"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{30,}"),
    re.compile(rb"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),
    # AWS documents AKIA for long-lived and ASIA for STS-issued access IDs.
    re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    # Secret/session token bodies have no safely unique prefix, so recognize
    # them only beside AWS's exact environment/shared-credentials keys.
    re.compile(
        rb"(?im)(?<![A-Za-z0-9_])(?:AWS_SECRET_ACCESS_KEY|aws_secret_access_key)"
        rb"[\"']?[ \t]*[=:][ \t]*[\"']?[A-Za-z0-9/+=]{40}"
        rb"(?![A-Za-z0-9/+=])"
    ),
    re.compile(
        rb"(?im)(?<![A-Za-z0-9_])(?:AWS_SESSION_TOKEN|aws_session_token)"
        rb"[\"']?[ \t]*[=:][ \t]*[\"']?[A-Za-z0-9/+=]{80,}"
        rb"(?![A-Za-z0-9/+=])"
    ),
    re.compile(rb"(?<![0-9])[0-9]{8,20}:[A-Za-z0-9_-]{30,}"),
)


class PreflightConfigError(ValueError):
    """Raised for invalid invocation or hidden deny-list configuration."""


class _StableTreeError(OSError):
    """Raised when one filesystem snapshot cannot be proven stable."""


@dataclass(frozen=True)
class _PrivateDenyMatcher:
    exact: tuple[bytes, ...]
    normalized_casefold: tuple[str, ...]


def _stat_token(value: os.stat_result) -> tuple[int, ...]:
    """Return the mutation-sensitive identity used by anchored tree reads."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return _stat_token(left) == _stat_token(right)


def _bounded_directory_names(directory_fd: int) -> tuple[tuple[str, ...], bool]:
    """Enumerate at most one more than the fixed per-directory entry cap."""

    names: list[str] = []
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            names.append(entry.name)
            if len(names) > MAX_DIRECTORY_ENTRIES:
                return (), True
    return tuple(sorted(names)), False


def _open_release_root(root: str | Path) -> tuple[Path, int, os.stat_result]:
    candidate = Path(root)
    try:
        fd = os.open(candidate, _OPEN_DIRECTORY_FLAGS)
    except OSError as exc:
        raise PreflightConfigError("release tree must be a real directory") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise PreflightConfigError("release tree must be a real directory")
        try:
            named = os.stat(candidate, follow_symlinks=False)
        except OSError as exc:
            raise PreflightConfigError("release tree must be a real directory") from exc
        if not stat.S_ISDIR(named.st_mode) or not _same_entry(opened, named):
            raise PreflightConfigError("release tree must be a stable real directory")
        return candidate, fd, opened
    except Exception:
        os.close(fd)
        raise


def _root_identity_matches_fd(path: Path, root_fd: int, opened: os.stat_result) -> bool:
    try:
        current_fd = os.fstat(root_fd)
        current_path = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    identity = (opened.st_dev, opened.st_ino)
    return (
        stat.S_ISDIR(current_fd.st_mode)
        and stat.S_ISDIR(current_path.st_mode)
        and identity == (current_fd.st_dev, current_fd.st_ino)
        and identity == (current_path.st_dev, current_path.st_ino)
    )


def _open_directory_at(parent_fd: int, name: str, expected: os.stat_result) -> int:
    try:
        fd = os.open(name, _OPEN_DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise _StableTreeError("directory entry changed during release scan") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode) or not _same_entry(expected, opened):
            raise _StableTreeError("directory entry changed during release scan")
        return fd
    except Exception:
        os.close(fd)
        raise


def _read_regular_at(
    parent_fd: int,
    name: str,
    *,
    max_bytes: int,
    require_single_link: bool = False,
) -> tuple[bytes, os.stat_result]:
    """Read one anchored regular file, bounded to ``max_bytes + 1`` bytes."""

    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise _StableTreeError("file entry changed during release scan") from exc
    if not stat.S_ISREG(before.st_mode) or (require_single_link and before.st_nlink != 1):
        raise _StableTreeError("release entry is not a private regular file")
    try:
        fd = os.open(name, _OPEN_FILE_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise _StableTreeError("file entry changed during release scan") from exc
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (require_single_link and opened.st_nlink != 1)
            or not _same_entry(before, opened)
        ):
            raise _StableTreeError("file entry changed during release scan")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(fd)
        try:
            named_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise _StableTreeError("file entry changed during release scan") from exc
        if not _same_entry(opened, after) or not _same_entry(after, named_after):
            raise _StableTreeError("file entry changed during release scan")
        if len(content) > max_bytes:
            raise OverflowError("release entry exceeds the bounded read limit")
        if len(content) != after.st_size:
            raise _StableTreeError("file entry changed during release scan")
        return content, after
    finally:
        os.close(fd)


def _revalidate_regular_at(
    root_fd: int,
    parts: tuple[str, ...],
    expected: os.stat_result,
) -> bool:
    """Re-open an anchored parent chain and revalidate one prior file token."""

    if not parts:
        return False
    directory_fd: int | None = None
    try:
        directory_fd = os.dup(root_fd)
        for part in parts[:-1]:
            entry = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISDIR(entry.st_mode):
                return False
            child_fd = _open_directory_at(directory_fd, part, entry)
            os.close(directory_fd)
            directory_fd = child_fd
        current = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
        return stat.S_ISREG(current.st_mode) and _same_entry(expected, current)
    except (OSError, _StableTreeError):
        return False
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


@dataclass(frozen=True, order=True)
class ReleaseFinding:
    code: str
    path: str


@dataclass(frozen=True)
class ReleasePreflightReport:
    policy_version: str
    ok: bool
    files_scanned: int
    bytes_scanned: int
    findings: tuple[ReleaseFinding, ...]
    extra_deny_literal_count: int = 0

    def json_payload(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "ok": self.ok,
            "files_scanned": self.files_scanned,
            "bytes_scanned": self.bytes_scanned,
            "findings": [asdict(item) for item in self.findings],
            "extra_private_deny_literals_supplied": self.extra_deny_literal_count > 0,
            "extra_private_deny_literal_count": self.extra_deny_literal_count,
            "read_only": True,
            "creates_export": False,
            "initializes_git": False,
            "publishes": False,
        }

    def summary_json_payload(self) -> dict[str, object]:
        counts = Counter(item.code for item in self.findings)
        return {
            "policy_version": self.policy_version,
            "ok": self.ok,
            "files_scanned": self.files_scanned,
            "bytes_scanned": self.bytes_scanned,
            "finding_count": len(self.findings),
            "finding_counts_by_code": dict(sorted(counts.items())),
            "extra_private_deny_literals_supplied": self.extra_deny_literal_count > 0,
            "extra_private_deny_literal_count": self.extra_deny_literal_count,
            "paths_included": False,
            "read_only": True,
            "creates_export": False,
            "initializes_git": False,
            "publishes": False,
        }


def _nfkc_casefold(value: str) -> str:
    """Return the bounded comparison form used only for private-deny matching."""

    return unicodedata.normalize(
        "NFKC",
        unicodedata.normalize("NFKC", value).casefold(),
    )


def _compile_private_deny_matcher(
    extra_deny_literals: Sequence[bytes],
) -> _PrivateDenyMatcher:
    if len(extra_deny_literals) > MAX_EXTRA_DENY_LITERALS:
        raise PreflightConfigError("invalid private deny-list configuration")
    exact_values: set[bytes] = set()
    normalized_values: set[str] = set()
    for literal in extra_deny_literals:
        if (
            type(literal) is not bytes
            or not literal
            or len(literal) > MAX_EXTRA_DENY_LITERAL_BYTES
            or b"\x00" in literal
        ):
            raise PreflightConfigError("invalid private deny-list configuration")
        exact_values.add(literal)
        try:
            decoded = literal.decode("utf-8")
        except UnicodeDecodeError:
            # Direct byte-oriented callers retain exact matching for malformed
            # data. Runtime JSON configuration always supplies valid UTF-8.
            continue
        normalized = _nfkc_casefold(decoded)
        if len(normalized) > MAX_NORMALIZED_DENY_LITERAL_CHARS:
            raise PreflightConfigError("invalid private deny-list configuration")
        if normalized:
            normalized_values.add(normalized)
    return _PrivateDenyMatcher(
        exact=tuple(sorted(exact_values)),
        normalized_casefold=tuple(sorted(normalized_values)),
    )


def _variant_text_match(
    value: str,
    matcher: _PrivateDenyMatcher,
    *,
    input_bytes: int,
) -> tuple[bool, bool]:
    """Return ``(matched, bounded)`` without retaining candidate text.

    Exact bytes are checked separately. Variant scanning is restricted to a
    fixed UTF-8 input size and a fixed post-normalization size. An over-limit
    valid-text file fails closed rather than silently skipping private review.
    """

    if not matcher.normalized_casefold:
        return False, True
    if input_bytes > MAX_VARIANT_SCAN_INPUT_BYTES:
        return False, False
    normalized = _nfkc_casefold(value)
    if len(normalized) > MAX_NORMALIZED_VARIANT_CHARS:
        return False, False
    return any(literal in normalized for literal in matcher.normalized_casefold), True


def extra_deny_literals_from_env(environ: dict[str, str] | None = None) -> tuple[bytes, ...]:
    source = os.environ if environ is None else environ
    raw = source.get(EXTRA_DENY_ENV, "")
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise PreflightConfigError("invalid private deny-list configuration") from exc
    if not isinstance(payload, list) or len(payload) > MAX_EXTRA_DENY_LITERALS:
        raise PreflightConfigError("invalid private deny-list configuration")

    values: list[bytes] = []
    for value in payload:
        if not isinstance(value, str):
            raise PreflightConfigError("invalid private deny-list configuration")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PreflightConfigError("invalid private deny-list configuration") from exc
        if not encoded or len(encoded) > MAX_EXTRA_DENY_LITERAL_BYTES or b"\x00" in encoded:
            raise PreflightConfigError("invalid private deny-list configuration")
        values.append(encoded)
    result = tuple(sorted(set(values)))
    _compile_private_deny_matcher(result)
    return result


def _safe_display_path(
    relative: str,
    extra_deny_literals: Sequence[bytes],
    matcher: _PrivateDenyMatcher | None = None,
) -> str:
    selected_matcher = matcher or _compile_private_deny_matcher(extra_deny_literals)
    display_bytes = relative.encode("utf-8")
    variant_match, variant_bounded = _variant_text_match(
        relative,
        selected_matcher,
        input_bytes=len(display_bytes),
    )
    if variant_match or not variant_bounded:
        return "<private>"
    for literal in extra_deny_literals:
        display_bytes = display_bytes.replace(literal, b"<private>")
    display_bytes = EMAIL_PATTERN.sub(b"<private-email>", display_bytes)
    for pattern in CREDENTIAL_PATTERNS:
        display_bytes = pattern.sub(b"<private-token>", display_bytes)
    display = display_bytes.decode(
        "utf-8", errors="replace"
    )
    return display or "."


def _path_findings(relative: str, *, is_directory: bool) -> set[str]:
    path = Path(relative)
    name = path.name
    lowered = name.lower()
    findings: set[str] = set()

    if name in PRIVATE_DOCUMENT_BASENAMES:
        findings.add("private_document")
    if is_directory and name in FORBIDDEN_DIRECTORY_NAMES:
        findings.add("runtime_or_repository_directory")
    if is_directory and lowered.endswith(".app"):
        findings.add("compiled_artifact")
    if not is_directory:
        if name in SENSITIVE_FILE_BASENAMES:
            findings.add("credential_or_token_file")
        if name == ".DS_Store":
            findings.add("filesystem_metadata")
        if SERVICE_PLIST_NAME.fullmatch(name):
            findings.add("service_plist")
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            findings.add("populated_environment_file")
        if lowered.endswith(FORBIDDEN_FILE_SUFFIXES) or ".db-" in lowered or ".sqlite-" in lowered:
            findings.add("runtime_artifact")
        if lowered.endswith(OPAQUE_ARCHIVE_SUFFIXES):
            findings.add("opaque_archive")
        if lowered.endswith(COMPILED_ARTIFACT_SUFFIXES):
            findings.add("compiled_artifact")
        if TOKENISH_FILENAME.search(name):
            findings.add("credential_or_token_file")
    return findings


def _sensitive_path_findings(
    relative: str,
    extra_deny_literals: Sequence[bytes],
    matcher: _PrivateDenyMatcher | None = None,
) -> set[str]:
    selected_matcher = matcher or _compile_private_deny_matcher(extra_deny_literals)
    encoded = relative.encode("utf-8")
    findings: set[str] = set()
    if any(pattern.search(encoded) for pattern in CREDENTIAL_PATTERNS):
        findings.add("likely_credential")
    for match in EMAIL_PATTERN.finditer(encoded):
        domain = match.group(0).rsplit(b"@", 1)[-1].lower()
        if domain not in RESERVED_EMAIL_DOMAINS and not domain.endswith((b".test", b".invalid")):
            findings.add("likely_email")
            break
    if any(literal in encoded for literal in extra_deny_literals):
        findings.add("extra_private_literal")
    variant_match, variant_bounded = _variant_text_match(
        relative,
        selected_matcher,
        input_bytes=len(encoded),
    )
    if variant_match:
        findings.add("extra_private_literal")
    if not variant_bounded:
        findings.add("private_variant_scan_limit")
    return findings


def _content_findings(
    content: bytes,
    extra_deny_literals: Sequence[bytes],
    matcher: _PrivateDenyMatcher | None = None,
    variant_scan_allowed: bool = True,
) -> set[str]:
    selected_matcher = matcher or _compile_private_deny_matcher(extra_deny_literals)
    findings: set[str] = set()
    decoded_content: str | None = None
    if b"\x00" in content:
        findings.add("binary_content")
    else:
        try:
            decoded_content = content.decode("utf-8")
        except UnicodeDecodeError:
            findings.add("binary_content")
    if ABSOLUTE_USERS_PREFIX in content:
        findings.add("absolute_users_path")
    if any(pattern.search(content) for pattern in CREDENTIAL_PATTERNS):
        findings.add("likely_credential")
    for match in EMAIL_PATTERN.finditer(content):
        domain = match.group(0).rsplit(b"@", 1)[-1].lower()
        if domain not in RESERVED_EMAIL_DOMAINS and not domain.endswith((b".test", b".invalid")):
            findings.add("likely_email")
            break
    if any(literal in content for literal in extra_deny_literals):
        findings.add("extra_private_literal")
    if decoded_content is not None:
        if variant_scan_allowed:
            variant_match, variant_bounded = _variant_text_match(
                decoded_content,
                selected_matcher,
                input_bytes=len(content),
            )
        else:
            variant_match = False
            variant_bounded = not selected_matcher.normalized_casefold
        if variant_match:
            findings.add("extra_private_literal")
        if not variant_bounded:
            findings.add("private_variant_scan_limit")
    return findings


def scan_release_tree(
    root: str | Path,
    *,
    extra_deny_literals: Sequence[bytes] = (),
    max_file_bytes: int = MAX_FILE_BYTES,
) -> ReleasePreflightReport:
    if max_file_bytes < 1:
        raise PreflightConfigError("invalid scan size limit")

    candidate, root_fd, root_opened = _open_release_root(root)
    try:
        report = _scan_release_tree_fd(
            root_fd,
            extra_deny_literals=extra_deny_literals,
            max_file_bytes=max_file_bytes,
        )
        if not _root_identity_matches_fd(candidate, root_fd, root_opened):
            raise PreflightConfigError("release tree changed during scan")
        return report
    finally:
        os.close(root_fd)


def _scan_release_tree_fd(
    root_fd: int,
    *,
    extra_deny_literals: Sequence[bytes] = (),
    max_file_bytes: int = MAX_FILE_BYTES,
) -> ReleasePreflightReport:
    """Scan an already-opened directory without resolving child pathnames."""

    if max_file_bytes < 1:
        raise PreflightConfigError("invalid scan size limit")
    deny_matcher = _compile_private_deny_matcher(extra_deny_literals)

    findings: set[ReleaseFinding] = set()
    files_scanned = 0
    bytes_scanned = 0
    files_attempted = 0
    bytes_attempted = 0
    entries_seen = 0
    scan_work_limit_hit = False
    variant_files_scanned = 0
    variant_bytes_scanned = 0
    scanned_tokens: list[tuple[tuple[str, ...], os.stat_result, str]] = []

    def walk(directory_fd: int, relative_parent: str) -> None:
        nonlocal files_scanned, bytes_scanned, files_attempted, bytes_attempted
        nonlocal entries_seen, scan_work_limit_hit
        nonlocal variant_files_scanned, variant_bytes_scanned
        if scan_work_limit_hit:
            return
        try:
            before_directory = os.fstat(directory_fd)
            names, directory_limit_hit = _bounded_directory_names(directory_fd)
        except OSError:
            findings.add(ReleaseFinding("unreadable_entry", "<unreadable-path>"))
            return
        if directory_limit_hit:
            findings.add(ReleaseFinding("scan_work_limit", "<bounded-scan>"))
            scan_work_limit_hit = True
            return
        for name in names:
            entries_seen += 1
            if entries_seen > MAX_SCAN_ENTRIES:
                findings.add(ReleaseFinding("scan_work_limit", "<bounded-scan>"))
                scan_work_limit_hit = True
                break
            relative = f"{relative_parent}/{name}" if relative_parent else name
            display = _safe_display_path(relative, extra_deny_literals, deny_matcher)
            try:
                entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                findings.add(ReleaseFinding("unreadable_entry", display))
                continue
            if stat.S_ISLNK(entry_stat.st_mode):
                findings.add(ReleaseFinding("symlink_entry", display))
                continue
            if stat.S_ISDIR(entry_stat.st_mode):
                codes = _path_findings(relative, is_directory=True) | _sensitive_path_findings(
                    relative, extra_deny_literals, deny_matcher
                )
                for code in codes:
                    findings.add(ReleaseFinding(code, display))
                if codes:
                    continue
                child_fd: int | None = None
                try:
                    child_fd = _open_directory_at(directory_fd, name, entry_stat)
                    walk(child_fd, relative)
                    child_after = os.fstat(child_fd)
                    named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if not _same_entry(entry_stat, child_after) or not _same_entry(
                        child_after, named_after
                    ):
                        raise _StableTreeError("directory entry changed during release scan")
                except (OSError, _StableTreeError):
                    findings.add(ReleaseFinding("unreadable_entry", display))
                finally:
                    if child_fd is not None:
                        os.close(child_fd)
                if scan_work_limit_hit:
                    break
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                findings.add(ReleaseFinding("non_regular_entry", display))
                continue

            if (
                files_attempted >= MAX_SCAN_FILES
                or entry_stat.st_size < 0
                or bytes_attempted + min(entry_stat.st_size, max_file_bytes + 1)
                > MAX_SCAN_TOTAL_BYTES
            ):
                findings.add(ReleaseFinding("scan_work_limit", "<bounded-scan>"))
                scan_work_limit_hit = True
                break

            files_attempted += 1
            bytes_attempted += min(entry_stat.st_size, max_file_bytes + 1)

            path_codes = _path_findings(relative, is_directory=False) | _sensitive_path_findings(
                relative, extra_deny_literals, deny_matcher
            )
            for code in path_codes:
                findings.add(ReleaseFinding(code, display))
            try:
                content, stable_stat = _read_regular_at(
                    directory_fd,
                    name,
                    max_bytes=max_file_bytes,
                )
            except OverflowError:
                findings.add(ReleaseFinding("oversized_file", display))
                continue
            except (OSError, _StableTreeError):
                findings.add(ReleaseFinding("unreadable_entry", display))
                continue
            files_scanned += 1
            bytes_scanned += len(content)
            scanned_tokens.append((tuple(relative.split("/")), stable_stat, display))
            variant_scan_allowed = True
            if deny_matcher.normalized_casefold:
                variant_scan_allowed = (
                    variant_files_scanned < MAX_VARIANT_SCAN_FILE_COUNT
                    and variant_bytes_scanned + len(content)
                    <= MAX_VARIANT_SCAN_TOTAL_INPUT_BYTES
                )
                if variant_scan_allowed:
                    variant_files_scanned += 1
                    variant_bytes_scanned += len(content)
            for code in _content_findings(
                content,
                extra_deny_literals,
                deny_matcher,
                variant_scan_allowed=variant_scan_allowed,
            ):
                findings.add(ReleaseFinding(code, display))
        try:
            after_directory = os.fstat(directory_fd)
        except OSError:
            findings.add(ReleaseFinding("unreadable_entry", "<unreadable-path>"))
            return
        if not _same_entry(before_directory, after_directory):
            display = _safe_display_path(relative_parent, extra_deny_literals, deny_matcher)
            findings.add(ReleaseFinding("unreadable_entry", display))

    root_before = os.fstat(root_fd)
    walk(root_fd, "")
    root_after = os.fstat(root_fd)
    if not _same_entry(root_before, root_after):
        findings.add(ReleaseFinding("unreadable_entry", "."))
    for parts, expected, display in scanned_tokens:
        if not _revalidate_regular_at(root_fd, parts, expected):
            findings.add(ReleaseFinding("unreadable_entry", display))
    ordered = tuple(sorted(findings))
    return ReleasePreflightReport(
        policy_version=POLICY_VERSION,
        ok=not ordered,
        files_scanned=files_scanned,
        bytes_scanned=bytes_scanned,
        findings=ordered,
        extra_deny_literal_count=len(extra_deny_literals),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only privacy scan of an already-prepared public release tree."
    )
    parser.add_argument("--root", required=True, help="candidate release-tree directory")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="emit a path-level JSON report")
    output.add_argument(
        "--summary-json",
        action="store_true",
        help="emit only aggregate finding counts, without candidate paths",
    )
    return parser


def _render_text(report: ReleasePreflightReport) -> str:
    status = "PASS" if report.ok else "FAIL"
    lines = [
        f"{status} public-release-preflight policy={report.policy_version} "
        f"files={report.files_scanned} bytes={report.bytes_scanned} "
        f"findings={len(report.findings)}",
        "Read-only scan: no export, repository, copy, or publication was created.",
    ]
    lines.extend(f"- {item.code}: {item.path}" for item in report.findings)
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        extra = extra_deny_literals_from_env()
        report = scan_release_tree(args.root, extra_deny_literals=extra)
    except PreflightConfigError:
        # Do not echo the root or hidden deny-list payload.
        print("ERROR public-release-preflight configuration is invalid")
        return 2
    if args.summary_json:
        print(json.dumps(report.summary_json_payload(), sort_keys=True, separators=(",", ":")))
    elif args.json:
        print(json.dumps(report.json_payload(), sort_keys=True, separators=(",", ":")))
    else:
        print(_render_text(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
