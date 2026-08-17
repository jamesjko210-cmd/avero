from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import threading
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, Iterable

from .store import (
    MAX_DECISION_OUTCOME_SUMMARY_CHARS,
    MAX_DECISION_PROJECTION_OUTCOMES,
    MemoryRecord,
    MemoryStore,
    decision_projection_source_digest,
    goal_projection_source_digest,
    preference_projection_source_digest,
)


FOLDERS = [
    "Daily",
    "Sessions",
    "People",
    "Projects",
    "Decisions",
    "Tasks",
    "Skills",
    "Automations",
    "Reflections",
    "Sources",
    "Memory Tree",
]

LOCAL_PATH_RE = re.compile(
    r"(?:(?<![:A-Za-z0-9])/(?:System/Volumes/Data/Users|Users|private|var/(?:folders|tmp)|"
    r"tmp|Volumes|home|root)(?:/[^\n\r]*)?(?=$|[\s,;:)}\]])"
    r"|~[/\\][^\n\r]*|(?<![:A-Za-z0-9/])[A-Za-z]:[/\\][^\n\r]*)",
    re.IGNORECASE,
)
_HTTP_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
MAX_SKILL_PROJECTION_CHARS = 100_000
MAX_MEMORY_PROJECTION_CHARS = 1_000_000
MAX_GOAL_PROJECTION_CHARS = 1_000_000
MAX_ORGANIZED_PROJECTION_BYTES = 4_000_000
MAX_ATOMIC_EXCHANGE_JOURNAL_BYTES = 64_000
MAX_ATOMIC_EXCHANGE_FILE_BYTES = 16_000_000
MAX_SCHEDULED_NOTE_FILE_BYTES = 8_000_000
MAX_SCHEDULED_NOTE_BLOCK_BYTES = 400_000
_UNSET_EXPECTED_CONTENT = object()
_ADVISORY_LOCK_OPEN_FDS: set[int] = set()
_ADVISORY_LOCK_FDS_GUARD = threading.Lock()


def _prepare_advisory_lock_fds_for_fork() -> None:
    _ADVISORY_LOCK_FDS_GUARD.acquire()


def _release_advisory_lock_fds_after_fork_parent() -> None:
    _ADVISORY_LOCK_FDS_GUARD.release()


def _reset_advisory_lock_fds_after_fork() -> None:
    global _ADVISORY_LOCK_OPEN_FDS
    global _ADVISORY_LOCK_FDS_GUARD
    for fd in tuple(_ADVISORY_LOCK_OPEN_FDS):
        try:
            os.close(fd)
        except OSError:
            pass
    _ADVISORY_LOCK_OPEN_FDS = set()
    _ADVISORY_LOCK_FDS_GUARD = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_prepare_advisory_lock_fds_for_fork,
        after_in_parent=_release_advisory_lock_fds_after_fork_parent,
        after_in_child=_reset_advisory_lock_fds_after_fork,
    )


def _release_advisory_lock_fd(fd: int, *, owner_pid: int) -> None:
    with _ADVISORY_LOCK_FDS_GUARD:
        if os.getpid() == owner_pid:
            os.close(fd)
        _ADVISORY_LOCK_OPEN_FDS.discard(fd)

_SCHEDULED_NOTE_MARKER_RE = re.compile(
    r"^<!-- jarvis-scheduled-note-(?P<edge>start|end):v1:"
    r"(?P<source>[0-9a-f]{64}):(?P<digest>[0-9a-f]{64}) -->[ \t]*\r?$",
    re.MULTILINE,
)


def _validated_daily_target_date(target_date: object) -> str:
    if type(target_date) is not str or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}", target_date
    ) is None:
        raise ValueError("Daily target date must use YYYY-MM-DD.")
    try:
        parsed_date = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Daily target date must be a real calendar date.") from exc
    if parsed_date.strftime("%Y-%m-%d") != target_date:
        raise ValueError("Daily target date must use canonical YYYY-MM-DD.")
    return target_date


_SCHEDULED_NOTE_CONTROL_LINE_RE = re.compile(
    r"^.*jarvis-scheduled-note-(?:start|end):.*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SkillProjectionDeleteResult:
    status: str
    path_display: str = ""


@dataclass(frozen=True)
class MemoryProjectionDeleteResult:
    status: str
    path_display: str = ""


@dataclass(frozen=True)
class PersonProjectionDeleteResult:
    status: str


@dataclass(frozen=True)
class ProfileGroundingView:
    text: str
    verified_source_keys: tuple[str, ...]
    invalid: bool
    truncated: bool


class ProfileEvidenceRevalidationError(RuntimeError):
    pass


class InboxSnapshotError(RuntimeError):
    pass


def safe_text(value: object) -> str:
    text = str(value or "")
    redacted: list[str] = []
    cursor = 0
    for match in _HTTP_URL_RE.finditer(text):
        redacted.append(LOCAL_PATH_RE.sub("<local-path>", text[cursor : match.start()]))
        redacted.append(match.group(0))
        cursor = match.end()
    redacted.append(LOCAL_PATH_RE.sub("<local-path>", text[cursor:]))
    return "".join(redacted)


def safe_name(value: str) -> str:
    value = value.strip() or "Untitled"
    value = re.sub(r"[\\/:*?\"<>|#^[\\]]+", "-", value)
    value = re.sub(r"\s+", " ", value)
    return value[:90].strip(" .-") or "Untitled"


_PROFILE_NOTE_START_RE = re.compile(
    r"<!-- jarvis-profile-note-start:v1:([0-9a-f]{64}) -->[ \t]*(?:\n|$)"
)
_PROFILE_NOTE_END_RE = re.compile(
    r"<!-- jarvis-profile-note:v1:([0-9a-f]{64}) -->[ \t]*(?:\n|$)"
)
_PROFILE_NOTE_MARKER_RE = re.compile(r"jarvis-profile-note", re.IGNORECASE)
_PROFILE_NOTE_MARKER_SHAPE_RE = re.compile(
    r"[ \t]*<!--[ \t]*jarvis-profile-note(?P<start>-start)?"
    r":[^>\r\n]*-->[ \t]*(?:\r\n|\n|\r|$)",
    re.IGNORECASE,
)
_PROFILE_NOTE_MARKER_PREFIXES = (
    "<!-- jarvis-profile-note:v1:",
    "<!-- jarvis-profile-note-start:v1:",
)


def _profile_note_block(source_key: str, heading: str, body: str) -> str:
    if re.fullmatch(r"profile-note:v1:[0-9a-f]{64}", source_key) is None:
        raise ValueError("Profile note source key is invalid.")
    marker = f"jarvis-{source_key}"
    start_marker = marker.replace(
        "jarvis-profile-note:v1:", "jarvis-profile-note-start:v1:", 1
    )
    return (
        f"\n<!-- {start_marker} -->\n"
        f"## {safe_text(heading)}\n\n{safe_text(body).strip()}\n"
        f"<!-- {marker} -->\n"
    )


def _split_profile_owned_blocks(
    content: str,
) -> tuple[str, dict[str, tuple[str, ...]], bool]:
    manual_lines: list[str] = []
    blocks: dict[str, list[str]] = {}
    open_depth = 0
    outer_key: str | None = None
    outer_lines: list[str] = []
    outer_canonical = False
    malformed = False

    def marker_kind(line: str) -> tuple[str, str | None]:
        start = _PROFILE_NOTE_START_RE.fullmatch(line)
        if start is not None:
            return "start", "profile-note:v1:" + start.group(1)
        end = _PROFILE_NOTE_END_RE.fullmatch(line)
        if end is not None:
            return "end", "profile-note:v1:" + end.group(1)

        classified = unicodedata.normalize("NFKC", line)
        start = _PROFILE_NOTE_START_RE.fullmatch(classified)
        if start is not None:
            return "suspicious_start", None
        end = _PROFILE_NOTE_END_RE.fullmatch(classified)
        if end is not None:
            return "suspicious_end", None
        if not classified.endswith(("\n", "\r")):
            candidate = classified.lstrip(" \t").lower()
            if candidate.startswith("<!--") and any(
                prefix.startswith(candidate) for prefix in _PROFILE_NOTE_MARKER_PREFIXES
            ):
                return "suspicious_start", None
        if _PROFILE_NOTE_MARKER_RE.search(classified) is None:
            return "text", None
        shaped = _PROFILE_NOTE_MARKER_SHAPE_RE.fullmatch(classified)
        if shaped is not None:
            return ("suspicious_start" if shaped.group("start") else "suspicious_end"), None
        if "jarvis-profile-note-start" in classified.lower():
            return "suspicious_start", None
        return "ambiguous", None

    def discard_ambiguous_manual_prefix() -> None:
        manual_lines.clear()

    for line in content.splitlines(keepends=True):
        kind, key = marker_kind(line)
        if open_depth == 0:
            if kind in {"start", "suspicious_start"}:
                open_depth = 1
                outer_key = key
                outer_lines = [line] if kind == "start" else []
                outer_canonical = kind == "start"
                malformed = malformed or kind != "start"
                continue
            if kind in {"end", "suspicious_end"}:
                malformed = True
                discard_ambiguous_manual_prefix()
                continue
            if kind == "ambiguous":
                malformed = True
                discard_ambiguous_manual_prefix()
                open_depth = 1
                outer_key = None
                outer_lines = []
                outer_canonical = False
                continue
            manual_lines.append(line)
            continue

        if kind in {"start", "suspicious_start"}:
            malformed = True
            outer_canonical = False
            outer_lines = []
            open_depth += 1
            continue
        if kind == "ambiguous":
            malformed = True
            outer_canonical = False
            outer_lines = []
            continue
        if kind in {"end", "suspicious_end"}:
            if open_depth > 1:
                malformed = True
                outer_canonical = False
                outer_lines = []
                open_depth -= 1
                continue
            if kind == "end" and outer_canonical and key == outer_key:
                outer_lines.append(line)
                assert outer_key is not None
                blocks.setdefault(outer_key, []).append("".join(outer_lines))
            else:
                malformed = True
            open_depth = 0
            outer_key = None
            outer_lines = []
            outer_canonical = False
            continue
        if outer_canonical:
            outer_lines.append(line)

    if open_depth:
        malformed = True
    return (
        "".join(manual_lines),
        {key: tuple(values) for key, values in blocks.items()},
        malformed,
    )


def _decision_outcomes(decision, outcomes=None) -> tuple[object, ...]:
    if outcomes is None:
        try:
            outcomes = decision["outcomes"]
        except (KeyError, IndexError, TypeError):
            outcomes = ()
    if isinstance(outcomes, (str, bytes, bytearray, dict)):
        raise ValueError("Decision outcomes must be an iterable of rows.")
    try:
        frozen_items: list[object] = []
        for outcome in outcomes:
            if len(frozen_items) >= MAX_DECISION_PROJECTION_OUTCOMES:
                raise ValueError("Decision outcome snapshot is too large.")
            frozen_items.append(outcome)
        frozen = tuple(frozen_items)
        decision_id = decision["id"]
        seen_ids: set[int] = set()
        for outcome in frozen:
            if (
                type(outcome["id"]) is not int
                or outcome["id"] < 1
                or outcome["id"] in seen_ids
                or type(outcome["decision_id"]) is not int
                or outcome["decision_id"] != decision_id
                or type(outcome["decision_revision"]) is not int
                or outcome["decision_revision"] < 1
                or type(outcome["summary"]) is not str
                or not outcome["summary"].strip()
                or len(outcome["summary"]) > MAX_DECISION_OUTCOME_SUMMARY_CHARS
                or any(
                    unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
                    for character in outcome["summary"]
                )
                or type(outcome["created_at"]) is not str
                or outcome["provenance"] != "user_reported"
                or type(outcome["provenance"]) is not str
            ):
                raise ValueError("Decision outcome row is malformed.")
            seen_ids.add(outcome["id"])
        ordered = tuple(
            sorted(
                frozen,
                key=lambda outcome: (outcome["created_at"], outcome["id"]),
            )
        )
        if ordered != frozen:
            raise ValueError("Decision outcomes are not ordered oldest-first.")
        return ordered
    except Exception as exc:
        raise ValueError("Decision outcomes are malformed.") from exc


def _append_decision_outcomes(lines: list[str], outcomes: tuple[object, ...]) -> None:
    lines.extend(["## Outcomes", ""])
    if not outcomes:
        lines.extend(["No outcomes captured.", ""])
        return
    for outcome in outcomes:
        summary_lines = safe_text(outcome["summary"]).splitlines() or [""]
        lines.extend(
            [
                f"### Outcome #{outcome['id']}",
                "",
                f"- Decision revision: {outcome['decision_revision']}",
                f"- Recorded: {safe_text(outcome['created_at'])}",
                "- Provenance: user-reported",
                "- Verification: Not independently verified.",
                "",
                "User-reported summary:",
                "",
            ]
        )
        lines.extend(f"> {line}" if line else ">" for line in summary_lines)
        lines.append("")


def _legacy_decision_content(
    decision,
    *,
    status: str,
    updated_at: str,
    frontmatter_updated: str,
    outcomes: tuple[object, ...] | None,
) -> str:
    lines = [
        "---",
        f"id: {decision['id']}",
        f"status: {safe_text(status)}",
        f"updated: {frontmatter_updated}",
        "---",
        "",
        f"# {safe_text(decision['title'])}",
        "",
        "## Decision",
        "",
        safe_text(decision["title"]),
        "",
        "## Rationale",
        "",
        safe_text(decision["rationale"]) or "No rationale captured.",
        "",
        "## Impact",
        "",
        safe_text(decision["impact"]) or "No impact captured.",
        "",
    ]
    if outcomes is not None:
        _append_decision_outcomes(lines, outcomes)
    lines.extend(
        [
            "## Metadata",
            "",
            f"- Created: {decision['created_at']}",
            f"- Updated: {updated_at}",
        ]
    )
    return "\n".join(lines) + "\n"


def _bounded_filename_component(value: str, *, max_bytes: int = 180) -> str:
    component = safe_name(value)
    while len(component.encode("utf-8")) > max_bytes:
        component = component[:-1].rstrip(" .-")
    return component or "Untitled"


def _digest_safe_payload(value: object, *, depth: int = 0) -> object:
    if depth >= 8:
        return {"unsupported_type": "depth_limit"}
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"unsupported_type": "non_finite_number"}
    if type(value) is dict:
        safe: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 1000:
                safe["<truncated>"] = True
                break
            safe_key = key if type(key) is str else "<non_string_key>"
            safe[safe_key[:240]] = _digest_safe_payload(item, depth=depth + 1)
        return safe
    if type(value) in {list, tuple}:
        return [_digest_safe_payload(item, depth=depth + 1) for item in value[:1000]]
    return {"unsupported_type": type(value).__name__[:80]}


_CONTAINMENT_ERROR = "Obsidian path must stay within the Jarvis vault's resolved root and may not be a symlink."
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_contained_parent(
    root: Path,
    path: Path,
    *,
    create_parents: bool = True,
) -> tuple[int, str]:
    requested_root = root.expanduser()
    if create_parents:
        requested_root.mkdir(parents=True, exist_ok=True)
    resolved_root = requested_root.resolve(strict=True)
    candidate = path.expanduser()
    if not candidate.is_absolute():
        relative = candidate
    else:
        absolute_candidate = candidate.absolute()
        relative = None
        for base in (requested_root.absolute(), resolved_root):
            try:
                relative = absolute_candidate.relative_to(base)
                break
            except ValueError:
                continue
        if relative is None:
            raise ValueError(_CONTAINMENT_ERROR)
    if relative == Path(".") or ".." in relative.parts:
        raise ValueError(_CONTAINMENT_ERROR)

    parent_fd = os.open(resolved_root, _DIRECTORY_FLAGS)
    try:
        for part in relative.parent.parts:
            if part in ("", "."):
                continue
            created = False
            if create_parents:
                try:
                    os.mkdir(part, dir_fd=parent_fd)
                    created = True
                except FileExistsError:
                    pass
            if created:
                os.fsync(parent_fd)
            component_stat = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(component_stat.st_mode):
                raise ValueError(_CONTAINMENT_ERROR)
            try:
                child_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(_CONTAINMENT_ERROR) from exc
                raise
            os.close(parent_fd)
            parent_fd = child_fd
        try:
            destination_stat = os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            destination_stat = None
        if destination_stat is not None and stat.S_ISLNK(destination_stat.st_mode):
            raise ValueError(_CONTAINMENT_ERROR)
        return parent_fd, relative.name
    except BaseException:
        os.close(parent_fd)
        raise


def _ensure_directory(root: Path, path: Path) -> None:
    parent_fd, _ = _open_contained_parent(root, path / ".jarvis-directory-check")
    os.close(parent_fd)


def _contained_exists(root: Path, path: Path) -> bool:
    parent_fd, name = _open_contained_parent(root, path)
    try:
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(parent_fd)


def _replace_text(
    root: Path,
    path: Path,
    content: str,
    *,
    expected_content: object = _UNSET_EXPECTED_CONTENT,
    expected_bytes: object = _UNSET_EXPECTED_CONTENT,
    max_exchange_bytes: int = MAX_ATOMIC_EXCHANGE_FILE_BYTES,
    parent_fd: int | None = None,
    name: str | None = None,
) -> None:
    if (
        expected_content is not _UNSET_EXPECTED_CONTENT
        and expected_bytes is not _UNSET_EXPECTED_CONTENT
    ):
        raise TypeError("expected Obsidian content and bytes are mutually exclusive")
    if expected_bytes is not _UNSET_EXPECTED_CONTENT and type(expected_bytes) is not bytes:
        raise TypeError("expected Obsidian bytes must be bytes or unset")
    if type(max_exchange_bytes) is not int or max_exchange_bytes < 1:
        raise ValueError("Obsidian exchange read limit must be positive")
    if parent_fd is None and name is None:
        parent_fd, name = _open_contained_parent(root, path)
    elif type(parent_fd) is int and type(name) is str and name and "/" not in name:
        parent_fd = os.dup(parent_fd)
    else:
        raise ValueError("pre-opened Obsidian parent and name must be supplied together")
    assert parent_fd is not None and name is not None
    candidate_bytes = content.encode("utf-8")
    temp_name: str | None = None
    journal_name: str | None = None
    temp_fd: int | None = None
    exchange_active = False
    try:
        for _ in range(100):
            candidate = f".{name}.{secrets.token_hex(8)}.tmp"
            try:
                temp_fd = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            temp_name = candidate
            break
        if temp_fd is None or temp_name is None:
            raise FileExistsError(f"Could not allocate a temporary note for {name}.")
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            temp_fd = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if (
            expected_content is _UNSET_EXPECTED_CONTENT
            and expected_bytes is _UNSET_EXPECTED_CONTENT
        ):
            os.replace(temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temp_name = None
        elif expected_content is None and expected_bytes is _UNSET_EXPECTED_CONTENT:
            try:
                _rename_noreplace(
                    temp_name,
                    name,
                    source_dir_fd=parent_fd,
                    destination_dir_fd=parent_fd,
                )
            except FileExistsError as exc:
                raise RuntimeError("Obsidian destination changed during atomic publication.") from exc
            temp_name = None
        elif isinstance(expected_content, str) or type(expected_bytes) is bytes:
            compare_exact_bytes = type(expected_bytes) is bytes
            expected_payload = (
                expected_content.encode("utf-8")
                if isinstance(expected_content, str)
                else expected_bytes
            )
            assert type(expected_payload) is bytes
            expected_value: bytes | str = (
                expected_payload if compare_exact_bytes else expected_content
            )
            candidate_value: bytes | str = (
                candidate_bytes if compare_exact_bytes else content
            )
            exchange_read_limit = max(
                max_exchange_bytes,
                len(expected_payload),
                len(candidate_bytes),
                1,
            )

            def read_exchange_file(file_name: str) -> bytes | str:
                raw = _read_bytes_at(
                    parent_fd,
                    file_name,
                    max_bytes=exchange_read_limit,
                )
                if raw is None:
                    raise FileNotFoundError(file_name)
                if compare_exact_bytes:
                    return raw
                return raw.decode("utf-8", errors="replace").replace(
                    "\r\n", "\n"
                ).replace("\r", "\n")

            journal_name = temp_name.removesuffix(".tmp") + ".exchange.json"
            journal_fd = os.open(
                journal_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            with os.fdopen(journal_fd, "w", encoding="utf-8") as journal:
                json.dump(
                    {
                        "version": 1,
                        "temp_name": temp_name,
                        "expected_sha256": hashlib.sha256(expected_payload).hexdigest(),
                        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                    },
                    journal,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                journal.flush()
                os.fsync(journal.fileno())
            os.fsync(parent_fd)
            exchange_active = True
            _rename_exchange(
                temp_name,
                name,
                source_dir_fd=parent_fd,
                destination_dir_fd=parent_fd,
            )
            displaced_content = read_exchange_file(temp_name)
            if displaced_content != expected_value:
                current_content = read_exchange_file(name)
                if current_content == candidate_value:
                    _rename_exchange(
                        temp_name,
                        name,
                        source_dir_fd=parent_fd,
                        destination_dir_fd=parent_fd,
                    )
                    rolled_back_content = read_exchange_file(temp_name)
                    if rolled_back_content != candidate_value:
                        conflict_name = f".{name}.jarvis-conflict-{secrets.token_hex(8)}"
                        os.rename(temp_name, conflict_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                        temp_name = None
                        if journal_name is not None:
                            os.rename(
                                journal_name,
                                conflict_name + ".exchange.json",
                                src_dir_fd=parent_fd,
                                dst_dir_fd=parent_fd,
                            )
                            journal_name = None
                        exchange_active = False
                        os.fsync(parent_fd)
                        raise RuntimeError("Obsidian changed again during atomic rollback.")
                    os.fsync(parent_fd)
                    exchange_active = False
                else:
                    conflict_name = f".{name}.jarvis-conflict-{secrets.token_hex(8)}"
                    os.rename(temp_name, conflict_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                    temp_name = None
                    if journal_name is not None:
                        os.rename(
                            journal_name,
                            conflict_name + ".exchange.json",
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                        )
                        journal_name = None
                    exchange_active = False
                    os.fsync(parent_fd)
                raise RuntimeError("Obsidian destination changed during atomic publication.")
            os.unlink(temp_name, dir_fd=parent_fd)
            temp_name = None
            if journal_name is not None:
                os.unlink(journal_name, dir_fd=parent_fd)
                journal_name = None
            exchange_active = False
        else:
            raise TypeError("expected Obsidian content must be text, bytes, None, or unset")
        os.fsync(parent_fd)
    except BaseException:
        cleanup_changed = False
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_name is not None and not exchange_active:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
                cleanup_changed = True
            except OSError:
                pass
        if journal_name is not None and not exchange_active:
            try:
                os.unlink(journal_name, dir_fd=parent_fd)
                cleanup_changed = True
            except OSError:
                pass
        if cleanup_changed:
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_fd)


@contextmanager
def _exclusive_sidecar_lock(
    root: Path, path: Path, *, timeout_seconds: float = 5.0
) -> Generator[None, None, None]:
    owner_pid = os.getpid()
    parent_fd, name = _open_contained_parent(root, path)
    fd: int | None = None
    try:
        with _ADVISORY_LOCK_FDS_GUARD:
            fd = os.open(
                name,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            _ADVISORY_LOCK_OPEN_FDS.add(fd)
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Jarvis projection lock timed out.")
                time.sleep(0.05)
        try:
            yield
        finally:
            if os.getpid() == owner_pid:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        if fd is not None:
            _release_advisory_lock_fd(fd, owner_pid=owner_pid)
        os.close(parent_fd)


def _write_all(fd: int, content: str) -> None:
    remaining = memoryview(content.encode("utf-8"))
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("Obsidian append made no progress.")
        remaining = remaining[written:]


def _append_text_under_inode_lock(
    root: Path,
    path: Path,
    content: str,
    *,
    initial_content: str = "",
    existing_prefix: str = "",
    create_only: bool = False,
) -> bool:
    owner_pid = os.getpid()
    parent_fd, name = _open_contained_parent(root, path)
    fd: int | None = None
    created = False
    locked = False
    try:
        base_flags = (
            os.O_WRONLY
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        with _ADVISORY_LOCK_FDS_GUARD:
            if create_only:
                fd = os.open(
                    name,
                    base_flags | os.O_CREAT | os.O_EXCL,
                    0o666,
                    dir_fd=parent_fd,
                )
                created = True
            else:
                for attempt in range(3):
                    try:
                        fd = os.open(
                            name,
                            base_flags | os.O_CREAT | os.O_EXCL,
                            0o666,
                            dir_fd=parent_fd,
                        )
                        created = True
                        break
                    except FileExistsError:
                        try:
                            fd = os.open(name, base_flags, dir_fd=parent_fd)
                            break
                        except FileNotFoundError:
                            if attempt == 2:
                                raise
            if fd is not None:
                _ADVISORY_LOCK_OPEN_FDS.add(fd)
        if fd is None:
            raise OSError(f"Could not open Obsidian note for append: {name}")
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        try:
            size = os.fstat(fd).st_size
            if initial_content and size == 0:
                _write_all(fd, initial_content)
            elif existing_prefix and size > 0:
                _write_all(fd, existing_prefix)
            _write_all(fd, content)
            os.fsync(fd)
            os.fsync(parent_fd)
        finally:
            if locked and os.getpid() == owner_pid:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        if fd is not None:
            _release_advisory_lock_fd(fd, owner_pid=owner_pid)
        os.close(parent_fd)
    return created


def _append_text(
    root: Path,
    path: Path,
    content: str,
    *,
    initial_content: str = "",
    existing_prefix: str = "",
    create_only: bool = False,
) -> bool:
    owner_pid = os.getpid()
    lock_path = path.with_name(f".{path.name}.jarvis.lock")
    with _exclusive_sidecar_lock(root, lock_path):
        return _append_text_under_inode_lock(
            root,
            path,
            content,
            initial_content=initial_content,
            existing_prefix=existing_prefix,
            create_only=create_only,
        )


def _append_text_once_under_inode_lock(
    root: Path,
    path: Path,
    *,
    marker: str,
    content: str,
    initial_content: str = "",
    match_content: str | None = None,
) -> bool:
    owner_pid = os.getpid()
    marker_bytes = marker.encode("utf-8")
    if not marker_bytes or len(marker_bytes) > 512:
        raise ValueError("Obsidian idempotency marker must be 1-512 UTF-8 bytes.")
    match_bytes = match_content.encode("utf-8") if match_content is not None else marker_bytes
    if not match_bytes or len(match_bytes) > 200_000:
        raise ValueError("Obsidian idempotency match content is invalid.")
    parent_fd, name = _open_contained_parent(root, path)
    fd: int | None = None
    locked = False
    try:
        base_flags = (
            os.O_RDWR
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        with _ADVISORY_LOCK_FDS_GUARD:
            for attempt in range(3):
                try:
                    fd = os.open(
                        name,
                        base_flags | os.O_CREAT | os.O_EXCL,
                        0o666,
                        dir_fd=parent_fd,
                    )
                    break
                except FileExistsError:
                    try:
                        fd = os.open(name, base_flags, dir_fd=parent_fd)
                        break
                    except FileNotFoundError:
                        if attempt == 2:
                            raise
            if fd is not None:
                _ADVISORY_LOCK_OPEN_FDS.add(fd)
        if fd is None:
            raise OSError(f"Could not open Obsidian note for idempotent append: {name}")
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            overlap = b""
            while True:
                chunk = os.read(fd, 64 * 1024)
                if not chunk:
                    break
                scanned = overlap + chunk
                if match_bytes in scanned:
                    return False
                overlap = scanned[-(len(match_bytes) - 1) :] if len(match_bytes) > 1 else b""
            size = os.fstat(fd).st_size
            if initial_content and size == 0:
                _write_all(fd, initial_content)
            _write_all(fd, content)
            os.fsync(fd)
            os.fsync(parent_fd)
            return True
        finally:
            if locked and os.getpid() == owner_pid:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        if fd is not None:
            _release_advisory_lock_fd(fd, owner_pid=owner_pid)
        os.close(parent_fd)


def _append_text_once(
    root: Path,
    path: Path,
    *,
    marker: str,
    content: str,
    initial_content: str = "",
    match_content: str | None = None,
) -> bool:
    lock_path = path.with_name(f".{path.name}.jarvis.lock")
    with _exclusive_sidecar_lock(root, lock_path):
        return _append_text_once_under_inode_lock(
            root,
            path,
            marker=marker,
            content=content,
            initial_content=initial_content,
            match_content=match_content,
        )


def _recover_atomic_exchange_journals(
    root: Path,
    path: Path,
    *,
    max_file_bytes: int = MAX_ATOMIC_EXCHANGE_FILE_BYTES,
) -> None:
    if type(max_file_bytes) is not int or max_file_bytes < 1:
        raise ValueError("Obsidian exchange recovery limit must be positive.")
    parent_fd, name = _open_contained_parent(root, path)
    try:
        journal_pattern = re.compile(
            rf"^\.{re.escape(name)}\.([0-9a-f]{{16}})\.exchange\.json$"
        )

        def read_named(candidate: str, *, max_bytes: int) -> bytes | None:
            return _read_bytes_at(parent_fd, candidate, max_bytes=max_bytes)

        for journal_name in sorted(os.listdir(parent_fd)):
            match = journal_pattern.fullmatch(journal_name)
            if match is None:
                continue
            raw_journal = read_named(
                journal_name,
                max_bytes=MAX_ATOMIC_EXCHANGE_JOURNAL_BYTES,
            )
            try:
                journal = json.loads(
                    (raw_journal or b"").decode("utf-8", errors="strict")
                )
            except (TypeError, UnicodeError, ValueError):
                journal = None
            token = match.group(1)
            expected_temp = f".{name}.{token}.tmp"
            if (
                not isinstance(journal, dict)
                or journal.get("version") != 1
                or journal.get("temp_name") != expected_temp
                or re.fullmatch(r"[0-9a-f]{64}", str(journal.get("expected_sha256") or "")) is None
                or re.fullmatch(r"[0-9a-f]{64}", str(journal.get("candidate_sha256") or "")) is None
            ):
                raise RuntimeError("Obsidian atomic exchange journal is malformed.")
            expected_digest = str(journal["expected_sha256"])
            candidate_digest = str(journal["candidate_sha256"])
            destination_content = read_named(name, max_bytes=max_file_bytes)
            temp_content = read_named(expected_temp, max_bytes=max_file_bytes)
            destination_digest = (
                hashlib.sha256(destination_content).hexdigest()
                if destination_content is not None
                else None
            )
            temp_digest = (
                hashlib.sha256(temp_content).hexdigest()
                if temp_content is not None
                else None
            )
            if temp_digest == candidate_digest and destination_digest != candidate_digest:
                os.unlink(expected_temp, dir_fd=parent_fd)
                os.unlink(journal_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                continue
            if destination_digest == candidate_digest and temp_digest == expected_digest:
                os.unlink(expected_temp, dir_fd=parent_fd)
                os.unlink(journal_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                continue
            if destination_digest == candidate_digest and temp_content is None:
                os.unlink(journal_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                continue
            if destination_digest == expected_digest and temp_content is None:
                os.unlink(journal_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                continue
            if destination_digest == candidate_digest and temp_content is not None:
                _rename_exchange(
                    expected_temp,
                    name,
                    source_dir_fd=parent_fd,
                    destination_dir_fd=parent_fd,
                )
                restored_candidate = read_named(
                    expected_temp,
                    max_bytes=max_file_bytes,
                )
                if (
                    restored_candidate is not None
                    and hashlib.sha256(restored_candidate).hexdigest()
                    == candidate_digest
                ):
                    os.unlink(expected_temp, dir_fd=parent_fd)
                    os.unlink(journal_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                    continue
            conflict_name = f".{name}.jarvis-conflict-{secrets.token_hex(8)}"
            if temp_content is not None:
                os.rename(
                    expected_temp,
                    conflict_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
            os.rename(
                journal_name,
                conflict_name + ".exchange.json",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
            raise RuntimeError("Obsidian atomic exchange recovery requires manual review.")
    finally:
        os.close(parent_fd)


def _atomic_append_text_once(
    root: Path,
    path: Path,
    *,
    content: str,
    initial_content: str = "",
    match_content: str | None = None,
    legacy_match_content: str | None = None,
    owned_start_marker: str | None = None,
    owned_marker: str | None = None,
) -> bool:
    """Append one complete block by atomic replacement under a sidecar lock."""
    if not content or len(content.encode("utf-8")) > 200_000:
        raise ValueError("Obsidian atomic append content is invalid.")
    expected = match_content
    if expected is not None and (not expected or len(expected.encode("utf-8")) > 200_000):
        raise ValueError("Obsidian atomic append match content is invalid.")
    for marker_value in (owned_start_marker, owned_marker):
        if marker_value is not None and (
            not marker_value or len(marker_value.encode("utf-8")) > 512 or "\n" in marker_value
        ):
            raise ValueError("Obsidian atomic append owned marker is invalid.")
    if (owned_start_marker is None) != (owned_marker is None):
        raise ValueError("Obsidian atomic append owned markers must be paired.")
    owned_source_key: str | None = None
    if owned_start_marker is not None and owned_marker is not None:
        owned_match = re.fullmatch(
            r"jarvis-(profile-note:v1:[0-9a-f]{64})",
            owned_marker,
        )
        if owned_match is None or owned_start_marker != owned_marker.replace(
            "jarvis-profile-note:v1:",
            "jarvis-profile-note-start:v1:",
            1,
        ):
            raise ValueError("Profile projection ownership markers are invalid.")
        owned_source_key = owned_match.group(1)
        generated = content.lstrip("\n")
        generated_manual, generated_blocks, generated_malformed = _split_profile_owned_blocks(
            generated
        )
        if (
            generated_malformed
            or generated_manual
            or generated_blocks != {owned_source_key: (generated,)}
        ):
            raise ValueError("Profile projection content conflicts with ownership markers.")
    lock_path = path.with_name(f".{path.name}.jarvis.lock")
    with _exclusive_sidecar_lock(root, lock_path):
        _recover_atomic_exchange_journals(root, path)
        parent_fd, name = _open_contained_parent(root, path)
        try:
            stale_pattern = re.compile(rf"^\.{re.escape(name)}\.[0-9a-f]{{16}}\.tmp$")
            for candidate in os.listdir(parent_fd):
                if stale_pattern.fullmatch(candidate) is None:
                    continue
                try:
                    candidate_stat = os.stat(candidate, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(candidate_stat.st_mode):
                    conflict_name = f".{name}.jarvis-conflict-{secrets.token_hex(8)}"
                    os.rename(candidate, conflict_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                    os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        existing = _read_text(root, path)
        base = existing if existing is not None else initial_content
        start_matches: list[re.Match[str]] = []
        end_matches: list[re.Match[str]] = []
        if owned_source_key is not None:
            _manual, owned_blocks, malformed = _split_profile_owned_blocks(base)
            if malformed:
                legacy_count = (
                    base.count(legacy_match_content) if legacy_match_content is not None else 0
                )
                marker_line_count = sum(
                    _PROFILE_NOTE_MARKER_RE.search(
                        unicodedata.normalize("NFKC", line)
                    )
                    is not None
                    for line in base.splitlines(keepends=True)
                )
                if legacy_count != 1 or marker_line_count != 1:
                    raise RuntimeError(
                        "Profile projection ownership marker structure is globally malformed or ambiguous."
                    )
                assert legacy_match_content is not None
                base = base.replace(legacy_match_content, "", 1)
                _manual, owned_blocks, malformed = _split_profile_owned_blocks(base)
            if malformed or any(len(items) != 1 for items in owned_blocks.values()):
                raise RuntimeError(
                    "Profile projection ownership marker structure is globally malformed or ambiguous."
                )
            expected_block = expected.lstrip("\n") if expected is not None else None
            if expected_block is not None and owned_blocks.get(owned_source_key) == (
                expected_block,
            ):
                return False
            start_matches = list(
                re.finditer(
                    rf"^<!-- {re.escape(owned_start_marker)} -->[ \t]*(?:\n|$)",
                    base,
                    flags=re.MULTILINE,
                )
            )
            end_matches = list(
                re.finditer(
                    rf"^<!-- {re.escape(owned_marker)} -->[ \t]*(?:\n|$)",
                    base,
                    flags=re.MULTILINE,
                )
            )
        if start_matches or end_matches:
            if len(start_matches) != len(end_matches):
                raise RuntimeError("Profile projection ownership markers are unbalanced.")
            owned_ranges = []
            end_index = 0
            for start_match in start_matches:
                while end_index < len(end_matches) and end_matches[end_index].start() < start_match.end():
                    end_index += 1
                if end_index >= len(end_matches):
                    raise RuntimeError("Profile projection ownership markers are misordered.")
                end_match = end_matches[end_index]
                if any(
                    nested.start() < end_match.start()
                    for nested in start_matches
                    if nested.start() > start_match.start()
                ):
                    raise RuntimeError("Profile projection ownership markers are nested.")
                owned_ranges.append((start_match.start(), end_match.end()))
                end_index += 1
            if end_index != len(end_matches):
                raise RuntimeError("Profile projection ownership markers are misordered.")
            for start, end in reversed(owned_ranges):
                base = base[:start] + base[end:]
        _replace_text(root, path, base + content, expected_content=existing)
        return True


def _atomic_append_text(
    root: Path,
    path: Path,
    content: str,
    *,
    initial_content: str = "",
    existing_prefix: str = "",
) -> bool:
    """Append a complete block through one atomic, compare-and-swap publication."""
    lock_path = path.with_name(f".{path.name}.jarvis.lock")
    with _exclusive_sidecar_lock(root, lock_path):
        _recover_atomic_exchange_journals(root, path)
        existing = _read_text(root, path)
        created = existing is None
        if existing is None or existing == "":
            candidate = initial_content + content
        else:
            candidate = existing + existing_prefix + content
        _replace_text(root, path, candidate, expected_content=existing)
    return created


def _escape_scheduled_note_markers(value: str) -> str:
    """Keep report content from impersonating Jarvis-owned control lines."""
    return re.sub(
        r"jarvis-scheduled-note-(?=(?:start|end):)",
        "jarvis&#45;scheduled&#45;note-",
        value,
    )


def _scheduled_note_blocks(
    text: str,
) -> tuple[dict[str, tuple[str, str]], bool]:
    """Return source -> (digest, owned payload), rejecting ambiguous markers."""
    blocks: dict[str, tuple[str, str]] = {}
    opened: re.Match[str] | None = None
    malformed = False
    matches = list(_SCHEDULED_NOTE_MARKER_RE.finditer(text))
    control_lines = list(_SCHEDULED_NOTE_CONTROL_LINE_RE.finditer(text))
    if {match.span() for match in matches} != {match.span() for match in control_lines}:
        return {}, True
    for match in matches:
        edge = match.group("edge")
        if edge == "start":
            if opened is not None:
                malformed = True
                break
            opened = match
            continue
        if (
            opened is None
            or opened.group("source") != match.group("source")
            or opened.group("digest") != match.group("digest")
        ):
            malformed = True
            break
        source = match.group("source")
        if source in blocks:
            malformed = True
            break
        blocks[source] = (
            match.group("digest"),
            text[opened.end() : match.start()],
        )
        opened = None
    if opened is not None:
        malformed = True
    return blocks, malformed


def _publish_scheduled_note_block(
    root: Path,
    path: Path,
    *,
    source_key: str,
    heading: str,
    body: str,
    initial_content: str,
    effect_authority: Callable[[], None] | None = None,
) -> tuple[bool, str]:
    """Publish one occurrence-owned block and replay its exact persisted body."""
    if type(source_key) is not str or re.fullmatch(r"[0-9a-f]{64}", source_key) is None:
        raise ValueError("Scheduled note source identity is invalid.")
    safe_heading = _escape_scheduled_note_markers(safe_text(heading).strip())
    safe_body = _escape_scheduled_note_markers(safe_text(body).strip())
    if not safe_heading:
        raise ValueError("Scheduled note heading is empty.")
    owned_payload = f"\n## {safe_heading}\n\n{safe_body}\n"
    payload_bytes = owned_payload.encode("utf-8")
    if len(payload_bytes) > MAX_SCHEDULED_NOTE_BLOCK_BYTES:
        raise ValueError("Scheduled note block exceeds the bounded publication size.")
    content_digest = hashlib.sha256(payload_bytes).hexdigest()
    start_marker = (
        f"<!-- jarvis-scheduled-note-start:v1:{source_key}:{content_digest} -->"
    )
    end_marker = (
        f"<!-- jarvis-scheduled-note-end:v1:{source_key}:{content_digest} -->"
    )
    block = f"\n{start_marker}{owned_payload}{end_marker}\n"
    lock_path = path.with_name(f".{path.name}.jarvis.lock")
    with _exclusive_sidecar_lock(root, lock_path):
        _recover_atomic_exchange_journals(root, path)
        existing_bytes = _read_bytes_bounded(
            root,
            path,
            max_bytes=MAX_SCHEDULED_NOTE_FILE_BYTES,
        )
        try:
            decoded = existing_bytes.decode("utf-8") if existing_bytes is not None else ""
            existing = decoded if decoded else initial_content
        except UnicodeDecodeError as exc:
            raise ValueError("Scheduled note destination is not valid UTF-8.") from exc
        blocks, malformed = _scheduled_note_blocks(existing)
        if malformed:
            raise RuntimeError("Scheduled note ownership markers are malformed or ambiguous.")
        prior = blocks.get(source_key)
        if prior is not None:
            prior_digest, prior_payload = prior
            if not secrets.compare_digest(
                hashlib.sha256(prior_payload.encode("utf-8")).hexdigest(),
                prior_digest,
            ):
                raise RuntimeError("Scheduled note occurrence evidence no longer matches its content.")
            expected_prefix = f"\n## {safe_heading}\n\n"
            if not prior_payload.startswith(expected_prefix) or not prior_payload.endswith("\n"):
                raise RuntimeError("Scheduled note occurrence heading no longer matches its owner.")
            persisted_body = prior_payload[len(expected_prefix) : -1]
            if effect_authority is not None and not secrets.compare_digest(
                persisted_body,
                safe_body,
            ):
                raise RuntimeError(
                    "Scheduled note occurrence evidence conflicts with the claimed payload."
                )
            if effect_authority is not None:
                effect_authority()
            return False, persisted_body

        candidate = existing + block
        if len(candidate.encode("utf-8")) > MAX_SCHEDULED_NOTE_FILE_BYTES:
            raise OverflowError("Scheduled note destination exceeds the bounded publication size.")
        if effect_authority is not None:
            effect_authority()
        if existing_bytes is None:
            _replace_text(root, path, candidate, expected_content=None)
        else:
            _replace_text(root, path, candidate, expected_bytes=existing_bytes)
        readback = _read_bytes_bounded(
            root,
            path,
            max_bytes=MAX_SCHEDULED_NOTE_FILE_BYTES,
        )
        if readback != candidate.encode("utf-8"):
            raise OSError("Scheduled note publication readback did not match.")
        return True, safe_body


def _scheduled_note_block_evidence(
    root: Path,
    path: Path,
    *,
    source_key: str,
    heading: str,
) -> str | None:
    """Replay a valid occurrence-owned body without creating or changing a note."""
    if type(source_key) is not str or re.fullmatch(r"[0-9a-f]{64}", source_key) is None:
        raise ValueError("Scheduled note source identity is invalid.")
    safe_heading = _escape_scheduled_note_markers(safe_text(heading).strip())
    if not safe_heading:
        raise ValueError("Scheduled note heading is empty.")
    lock_path = path.with_name(f".{path.name}.jarvis.lock")
    with _exclusive_sidecar_lock(root, lock_path):
        _recover_atomic_exchange_journals(root, path)
        existing_bytes = _read_bytes_bounded(
            root,
            path,
            max_bytes=MAX_SCHEDULED_NOTE_FILE_BYTES,
        )
        if not existing_bytes:
            return None
        try:
            existing = existing_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Scheduled note destination is not valid UTF-8.") from exc
        blocks, malformed = _scheduled_note_blocks(existing)
        if malformed:
            raise RuntimeError("Scheduled note ownership markers are malformed or ambiguous.")
        prior = blocks.get(source_key)
        if prior is None:
            return None
        prior_digest, prior_payload = prior
        if not secrets.compare_digest(
            hashlib.sha256(prior_payload.encode("utf-8")).hexdigest(),
            prior_digest,
        ):
            raise RuntimeError("Scheduled note occurrence evidence no longer matches its content.")
        expected_prefix = f"\n## {safe_heading}\n\n"
        if not prior_payload.startswith(expected_prefix) or not prior_payload.endswith("\n"):
            raise RuntimeError("Scheduled note occurrence heading no longer matches its owner.")
        return prior_payload[len(expected_prefix) : -1]


def _read_text(root: Path, path: Path, *, max_chars: int | None = None) -> str | None:
    parent_fd, name = _open_contained_parent(root, path)
    fd: int | None = None
    try:
        try:
            fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return None
        target_stat = os.fstat(fd)
        if not stat.S_ISREG(target_stat.st_mode):
            raise ValueError(_CONTAINMENT_ERROR)
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as handle:
            fd = None
            return handle.read() if max_chars is None else handle.read(max_chars)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def _read_bytes_at(parent_fd: int, name: str, *, max_bytes: int) -> bytes | None:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Obsidian byte read limit must be a positive integer.")
    fd: int | None = None
    try:
        try:
            target_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(target_stat.st_mode):
            raise ValueError(_CONTAINMENT_ERROR)
        if not stat.S_ISREG(target_stat.st_mode):
            raise IsADirectoryError("Obsidian note is not a regular file.")
        try:
            fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError(_CONTAINMENT_ERROR) from exc
            raise
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise IsADirectoryError("Obsidian note is not a regular file.")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > max_bytes:
            raise OverflowError("Obsidian note exceeds the bounded byte read limit.")
        return content
    finally:
        if fd is not None:
            os.close(fd)


def _read_bytes_bounded(root: Path, path: Path, *, max_bytes: int) -> bytes | None:
    try:
        parent_fd, name = _open_contained_parent(
            root,
            path,
            create_parents=False,
        )
    except FileNotFoundError:
        return None
    try:
        return _read_bytes_at(parent_fd, name, max_bytes=max_bytes)
    finally:
        os.close(parent_fd)


def _inbox_clear_target_binding(
    resolved_root: Path,
    root_stat: os.stat_result,
    content: bytes,
) -> str:
    root_identity = (
        f"{resolved_root}\0{root_stat.st_dev}\0{root_stat.st_ino}".encode(
            "utf-8", errors="strict"
        )
    )
    return hashlib.sha256(
        b"jarvis-clear-inbox-v2\0"
        + len(root_identity).to_bytes(8, "big")
        + root_identity
        + b"Inbox.md\0"
        + content
    ).hexdigest()


def _frontmatter_goal_id(header: str) -> int | None:
    opening = re.match(r"\A---[ \t]*\r?\n", header)
    if opening is None:
        return None
    closing = re.search(r"^---[ \t]*\r?$", header[opening.end() :], flags=re.MULTILINE)
    if closing is None:
        return None
    frontmatter = header[opening.end() : opening.end() + closing.start()]
    goal_id = re.search(r"^id:[ \t]*(\d+)[ \t]*$", frontmatter, flags=re.MULTILINE)
    return int(goal_id.group(1)) if goal_id is not None else None


def _is_owned_goal_projection(
    header: str,
    *,
    goal_id: int,
    store_identity: str,
) -> bool:
    fields = _goal_projection_frontmatter_fields(header, legacy=False)
    if fields is None:
        return False
    revision = _goal_json_string(fields["goal_revision"])
    source_digest = _goal_json_string(fields["source_digest"])
    return (
        fields["jarvis_projection"] == "goal"
        and fields["store_identity"] == store_identity
        and fields["goal_id"] == str(goal_id)
        and fields["id"] == str(goal_id)
        and revision is not None
        and re.fullmatch(r"[1-9]\d*", revision) is not None
        and source_digest is not None
        and re.fullmatch(r"[0-9a-f]{64}", source_digest) is not None
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields["updated"]) is not None
    )


def _is_owned_legacy_goal_projection(
    header: str,
    *,
    goal_id: int,
    store_identity: str,
) -> bool:
    fields = _goal_projection_frontmatter_fields(header, legacy=True)
    return bool(
        fields is not None
        and fields["jarvis_projection"] == "goal"
        and fields["store_identity"] == store_identity
        and fields["goal_id"] == str(goal_id)
        and fields["id"] == str(goal_id)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields["updated"]) is not None
    )


def _goal_projection_frontmatter_fields(
    header: str,
    *,
    legacy: bool,
) -> dict[str, str] | None:
    opening = re.match(r"\A---[ \t]*\r?\n", header)
    if opening is None:
        return None
    closing = re.search(r"^---[ \t]*\r?$", header[opening.end() :], flags=re.MULTILINE)
    if closing is None:
        return None
    frontmatter = header[opening.end() : opening.end() + closing.start()]
    allowed = {
        "jarvis_projection",
        "store_identity",
        "goal_id",
        "id",
        "status",
        "horizon",
        "updated",
    }
    if not legacy:
        allowed.update({"goal_revision", "source_digest"})
    fields: dict[str, str] = {}
    for line in frontmatter.splitlines():
        key, separator, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if (
            separator != ":"
            or key not in allowed
            or key in fields
            or not value
            or re.fullmatch(r"[a-z_]+", key) is None
        ):
            return None
        fields[key] = value
    if set(fields) != allowed:
        return None
    for key in ("status", "horizon"):
        if _goal_json_string(fields[key]) is None:
            return None
    return fields


def _goal_json_string(value: str) -> str | None:
    if not value.startswith('"'):
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return decoded if type(decoded) is str else None


def _goal_projection_freshness(header: str) -> tuple[int, str] | None:
    revision_text = _unique_frontmatter_text_value(header, "goal_revision")
    source_text = _unique_frontmatter_text_value(header, "source_digest")
    if revision_text is None or source_text is None:
        return None
    revision = _goal_json_string(revision_text)
    source_digest = _goal_json_string(source_text)
    if (
        revision is None
        or re.fullmatch(r"[1-9]\d*", revision) is None
        or source_digest is None
        or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None
    ):
        return None
    return int(revision), source_digest


def _read_goal_projection_snapshot(
    root: Path,
    path: Path,
) -> tuple[bytes, str] | None:
    raw = _read_bytes_bounded(
        root,
        path,
        max_bytes=MAX_GOAL_PROJECTION_CHARS * 4,
    )
    if raw is None:
        return None
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError("Goal projection is not valid UTF-8.") from exc
    if len(text) > MAX_GOAL_PROJECTION_CHARS:
        raise ValueError("Goal projection is too large to verify safely.")
    return raw, text


def _read_goal_projection(root: Path, path: Path) -> str | None:
    snapshot = _read_goal_projection_snapshot(root, path)
    return None if snapshot is None else snapshot[1]


def _goal_filename_component(value: object) -> str:
    text = "".join(
        character
        for character in safe_text(value)
        if ord(character) >= 32 and ord(character) != 127
    )
    return _bounded_filename_component(text)


def _goal_frontmatter_value(value: object) -> str:
    return json.dumps(safe_text(value))


def _render_goal_projection(
    goal,
    steps: tuple,
    *,
    store_identity: str,
    goal_id: int,
    frontmatter_updated: str,
    goal_revision: int | None,
    source_digest: str | None,
) -> str:
    lines = [
        "---",
        "jarvis_projection: goal",
        f"store_identity: {store_identity}",
        f"goal_id: {goal_id}",
        f"id: {goal_id}",
    ]
    if goal_revision is not None and source_digest is not None:
        lines.extend(
            [
                f"goal_revision: {_goal_frontmatter_value(str(goal_revision))}",
                f"source_digest: {_goal_frontmatter_value(source_digest)}",
            ]
        )
    lines.extend(
        [
            f"status: {_goal_frontmatter_value(goal['status'])}",
            f"horizon: {_goal_frontmatter_value(goal['horizon'])}",
            f"updated: {frontmatter_updated}",
            "---",
            "",
            f"# {safe_text(goal['title'])}",
            "",
            "## Purpose",
            "",
            safe_text(goal["purpose"]) or "No purpose captured yet.",
            "",
            "## Steps",
            "",
        ]
    )
    if steps:
        for step in steps:
            mark = "x" if step["status"] == "done" else " "
            lines.append(f"- [{mark}] #{step['id']} {safe_text(step['body'])}")
    else:
        lines.append("- [ ] No steps captured yet.")
    lines.extend(
        [
            "",
            "## Metadata",
            "",
            f"- Created: {goal['created_at']}",
            f"- Updated: {goal['updated_at']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _frontmatter_person_id(header: str) -> int | None:
    values = _frontmatter_text_values(header, "id")
    if len(values) != 1 or re.fullmatch(r"\d+", values[0]) is None:
        return None
    person_id = int(values[0])
    return person_id if person_id > 0 else None


def _frontmatter_text_value(header: str, key: str) -> str | None:
    opening = re.match(r"\A---[ \t]*\r?\n", header)
    if opening is None:
        return None
    closing = re.search(r"^---[ \t]*\r?$", header[opening.end() :], flags=re.MULTILINE)
    if closing is None:
        return None
    frontmatter = header[opening.end() : opening.end() + closing.start()]
    match = re.search(rf"^{re.escape(key)}:[ \t]*([^\r\n]+?)[ \t]*$", frontmatter, flags=re.MULTILINE)
    return match.group(1) if match is not None else None


def _unique_frontmatter_text_value(header: str, key: str) -> str | None:
    values = _frontmatter_text_values(header, key)
    return values[0] if len(values) == 1 else None


def _frontmatter_text_values(header: str, key: str) -> list[str]:
    opening = re.match(r"\A---[ \t]*\r?\n", header)
    if opening is None:
        return []
    closing = re.search(r"^---[ \t]*\r?$", header[opening.end() :], flags=re.MULTILINE)
    if closing is None:
        return []
    frontmatter = header[opening.end() : opening.end() + closing.start()]
    return re.findall(
        rf"^{re.escape(key)}:[ \t]*([^\r\n]+?)[ \t]*$",
        frontmatter,
        flags=re.MULTILINE,
    )


def _is_legacy_open_task_projection(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 5 or lines[0] != "# Open Tasks" or lines[1] != "":
        return False
    if re.fullmatch(r"Updated: \d{4}-\d{2}-\d{2} \d{2}:\d{2}", lines[2]) is None or lines[3] != "":
        return False
    return bool(lines[4:]) and all(
        line == "- [ ] No open tasks." or re.fullmatch(r"- \[ \] #\d+ .+", line) is not None
        for line in lines[4:]
    )


def _is_legacy_current_context_projection(text: str) -> bool:
    if text == "# Current Context\n\n":
        return True
    lines = text.splitlines()
    required_headings = {
        "## Active Goals",
        "## Recent Memories",
        "## Open Tasks",
        "## Active Decisions",
        "## People Context",
        "## Active Preferences",
        "## Scheduled Jobs",
        "## Background Rhythm",
        "## Recent Sessions",
        "## Recent Tool Runs",
        "## Pending Approvals",
    }
    return (
        len(lines) >= 4
        and lines[0] == "# Current Context"
        and lines[1] == ""
        and re.fullmatch(r"Updated: \d{4}-\d{2}-\d{2} \d{2}:\d{2}", lines[2]) is not None
        and required_headings.issubset(lines)
    )


def _is_legacy_approval_review_projection(text: str) -> bool:
    lines = text.splitlines()
    if not (
        len(lines) >= 7
        and lines[0] == "# Approval Review"
        and lines[1] == ""
        and re.fullmatch(r"Updated: \d{4}-\d{2}-\d{2} \d{2}:\d{2}", lines[2]) is not None
        and lines[3] == ""
        and lines[4] == "Approval review:"
    ):
        return False
    body = lines[4:]
    safety_line = (
        "- Keep using `autonomy plan: ...`, `privacy report`, and `safety status` before risky work."
    )
    if body == ["Approval review:", "- No pending approvals.", safety_line]:
        return True
    if (
        len(body) == 3
        and re.fullmatch(
            r"- No readable approvals\. \d+ unreadable approval row\(s\) hidden for safety\.",
            body[1],
        )
        is not None
        and body[2] == safety_line
    ):
        return True
    populated_intro = (
        "Use this before approving blocked actions. Run approval readiness before the last-look packet "
        "because approval means Jarvis may rerun the exact queued request once."
    )
    operator_line = (
        "Operator limit: the operator's explicit stop times, work windows, pause commands, and newer instructions "
        "override pending approvals, approved-looking packets, and priority goals."
    )
    expected_tail = [
        "Before approving:",
        "- Operator limit: the operator's explicit stop times, work windows, pause commands, and newer instructions "
        "override pending approvals, approved-looking packets, and priority goals.",
        "- Confirm the target app/file/account is correct.",
        "- Prefer read-only inspection first when uncertain.",
        "- Dismiss stale approvals instead of leaving them ambiguous.",
    ]
    return (
        len(body) > len(expected_tail) + 3
        and body[1] == populated_intro
        and body[2] == operator_line
        and body[-len(expected_tail) :] == expected_tail
        and any(re.fullmatch(r"Approval #\d+: .+", line) is not None for line in body)
    )


def _is_legacy_feedback_report_projection(text: str) -> bool:
    lines = text.splitlines()
    if not (
        len(lines) >= 7
        and lines[:4] == [
            "# Feedback Report",
            "",
            "Jarvis feedback report:",
            "Use this to improve behavior through reviewable memory, preferences, skills, or code changes. It never changes risky permissions by itself.",
        ]
        and lines[4] == ""
        and lines[5] == "Recent feedback:"
    ):
        return False
    if "Safe improvement paths:" in lines:
        required_tail = {
            "- Convert stable behavior feedback into `set preference ...`.",
            "- Turn repeated workflow feedback into a reviewed skill draft.",
            "- Use code changes only after a smoke test can prove the new behavior.",
            "- Keep shell, personal data, reminders, external actions, and computer control approval-gated.",
        }
        return required_tail.issubset(lines)
    return (
        "- No feedback captured yet." in lines
        and "Good feedback commands:" in lines
        and "- feedback: Jarvis should explain approvals more clearly" in lines
        and "- Jarvis feedback: responses should be shorter when I ask what changed" in lines
    )


def _is_legacy_feedback_actions_projection(text: str) -> bool:
    lines = text.splitlines()
    if not (
        len(lines) >= 6
        and lines[:4] == [
            "# Feedback Actions",
            "",
            "Jarvis feedback actions:",
            "These are reviewable suggestions only. Jarvis does not auto-change preferences, skills, code, permissions, or risky tools from feedback.",
        ]
        and lines[4] == ""
    ):
        return False
    if lines[5:7] == [
        "No feedback captured yet.",
        "Capture one with `feedback: ...`, then rerun `feedback actions`.",
    ]:
        return True
    return all(
        heading in lines
        for heading in (
            "Suggested preference updates:",
            "Skill or workflow candidates:",
            "Test/code candidates:",
            "Still approval-gated:",
            "- Shell, file writes outside safe tools, clipboard reads, personal data, reminders, external actions, and computer control.",
        )
    )


def _is_legacy_learning_review_projection(text: str) -> bool:
    lines = text.splitlines()
    return (
        len(lines) >= 20
        and lines[:4] == [
            "# Learning Review",
            "",
            "Jarvis learning review:",
            "This is a review-only loop. Jarvis can suggest memory, preference, skill, and test improvements, but it does not auto-edit behavior, permissions, code, or risky tools from this report.",
        ]
        and all(
            heading in lines
            for heading in (
                "Feedback signals:",
                "Memory health:",
                "Personalization base:",
                "Execution learning debt:",
                "Safe learning actions:",
                "Still manual or approval-gated:",
                "Memory curation boundary:",
            )
        )
        and "- the operator's explicit stop times, work windows, pause commands, and newer instructions override this learning loop." in lines
        and "- Memory deletion, edits, merges, code changes, shell/code, personal data, external actions, and computer control remain approval-gated." in lines
    )


def _is_legacy_chat_context_projection(text: str, *, body: str) -> bool:
    return text == safe_text(f"{body.rstrip()}\n")


def _legacy_skill_projection_content(
    name: str, trigger: str, body: str, tags: str
) -> str:
    return (
        "---\n"
        f"name: {safe_text(name)}\n"
        f"trigger: {safe_text(trigger)}\n"
        f"tags: {safe_text(tags)}\n"
        "---\n\n"
        f"# {safe_text(name)}\n\n"
        f"## Trigger\n\n{safe_text(trigger).strip()}\n\n"
        f"## Procedure\n\n{safe_text(body).strip()}\n"
    )


def _is_legacy_skill_projection(
    text: str, *, name: str, trigger: str, body: str, tags: str
) -> bool:
    return text == _legacy_skill_projection_content(name, trigger, body, tags)


def _skill_frontmatter_value(value: object) -> str:
    safe = safe_text(value)
    if "\r" in safe or "\n" in safe:
        return json.dumps(safe, ensure_ascii=False)
    return safe


def _skill_markdown_value(value: object) -> str:
    return re.sub(r"\r\n?|\r", "\n", safe_text(value))


def _read_skill_projection(root: Path, path: Path) -> str | None:
    text = _read_text(root, path, max_chars=MAX_SKILL_PROJECTION_CHARS + 1)
    if text is not None and len(text) > MAX_SKILL_PROJECTION_CHARS:
        raise ValueError("Skill projection is too large to verify safely.")
    return text


def _is_owned_skill_projection(
    text: str,
    *,
    skill_id: int,
    store_identity: str,
) -> bool:
    ownership = {
        "jarvis_projection": "skill",
        "store_identity": store_identity,
        "skill_id": str(skill_id),
    }
    for key, expected in ownership.items():
        matches = _frontmatter_text_values(text, key)
        if matches != [expected]:
            return False
    return True


def _read_memory_projection(root: Path, path: Path) -> str | None:
    text = _read_text(root, path, max_chars=MAX_MEMORY_PROJECTION_CHARS + 1)
    if text is not None and len(text) > MAX_MEMORY_PROJECTION_CHARS:
        raise ValueError("Memory projection is too large to verify safely.")
    return text


def _is_owned_memory_projection(
    text: str,
    *,
    memory_id: int,
    store_identity: str,
) -> bool:
    ownership = {
        "jarvis_projection": "memory",
        "store_identity": store_identity,
        "memory_id": str(memory_id),
    }
    return all(
        _frontmatter_text_values(text, key) == [expected]
        for key, expected in ownership.items()
    )


def _legacy_memory_projection_content(
    record: MemoryRecord,
    *,
    memory_id: int,
    store_identity: str,
    created_date: str,
) -> str:
    return (
        "---\n"
        "jarvis_projection: memory\n"
        f"store_identity: {store_identity}\n"
        f"memory_id: {memory_id}\n"
        f"category: {safe_text(record.category)}\n"
        f"source: {safe_text(record.source)}\n"
        f"confidence: {record.confidence}\n"
        f"created: {created_date}\n"
        "---\n\n"
        f"# {safe_text(record.title)}\n\n"
        f"{safe_text(record.body).strip()}\n"
    )


def _memory_projection_content(
    record: MemoryRecord,
    *,
    memory_id: int,
    store_identity: str,
    memory_revision: int,
    source_digest: str,
    created_at: str,
) -> str:
    if type(memory_id) is not int or memory_id < 1:
        raise ValueError("Memory projection id is invalid.")
    if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
        raise ValueError("Memory projection store identity is invalid.")
    if type(memory_revision) is not int or memory_revision < 1:
        raise ValueError("Memory projection revision is invalid.")
    if re.fullmatch(r"[0-9a-f]{64}", str(source_digest or "")) is None:
        raise ValueError("Memory projection source digest is invalid.")
    created_text = str(created_at or "")
    if (
        re.match(r"^\d{4}-\d{2}-\d{2}", created_text) is None
        or "\r" in created_text
        or "\n" in created_text
    ):
        raise ValueError("Memory projection creation time is invalid.")
    confidence = float(record.confidence)
    if not math.isfinite(confidence):
        raise ValueError("Memory projection confidence is invalid.")
    return (
        "---\n"
        "jarvis_projection: memory\n"
        f"store_identity: {store_identity}\n"
        f"memory_id: {memory_id}\n"
        f"memory_revision: {memory_revision}\n"
        f"source_digest: {source_digest}\n"
        f"category: {_skill_frontmatter_value(record.category)}\n"
        f"title: {_skill_frontmatter_value(record.title)}\n"
        f"source: {_skill_frontmatter_value(record.source)}\n"
        f"confidence: {confidence}\n"
        f"created_at: {created_text}\n"
        "---\n\n"
        f"# {_skill_markdown_value(record.title)}\n\n"
        f"{_skill_markdown_value(record.body).strip()}\n"
    )


def _unlink(root: Path, path: Path) -> bool:
    parent_fd, name = _open_contained_parent(root, path)
    try:
        try:
            os.unlink(name, dir_fd=parent_fd)
        except FileNotFoundError:
            return False
        os.fsync(parent_fd)
        return True
    finally:
        os.close(parent_fd)


def _rename_noreplace(
    source_name: str,
    destination_name: str,
    *,
    source_dir_fd: int,
    destination_dir_fd: int,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    renameatx_np = getattr(libc, "renameatx_np", None)
    if renameatx_np is not None:
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            source_dir_fd,
            source,
            destination_dir_fd,
            destination,
            0x00000004,
        )
    else:
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            source_dir_fd,
            source,
            destination_dir_fd,
            destination,
            0x00000001,
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _rename_exchange(
    source_name: str,
    destination_name: str,
    *,
    source_dir_fd: int,
    destination_dir_fd: int,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    renameatx_np = getattr(libc, "renameatx_np", None)
    if renameatx_np is not None:
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            source_dir_fd,
            source,
            destination_dir_fd,
            destination,
            0x00000002,
        )
    else:
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "atomic exchange rename is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            source_dir_fd,
            source,
            destination_dir_fd,
            destination,
            0x00000002,
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


class ObsidianVault:
    def __init__(self, vault_path: Path, root: str = "Jarvis"):
        self.vault_path = vault_path
        self.root = root.strip("/") or "Jarvis"

    @property
    def root_path(self) -> Path:
        return self.vault_path / self.root

    def init(self) -> None:
        self.root_path.mkdir(parents=True, exist_ok=True)
        for folder in FOLDERS:
            _ensure_directory(self.root_path, self.root_path / folder)
        self._ensure_file("Profile.md", "# Profile\n\n")
        self._ensure_file("Inbox.md", "# Inbox\n\n")
        for name in [
            "Identity",
            "Preferences",
            "Goals",
            "Current Context",
            "Relationships",
            "Active Projects",
        ]:
            self._ensure_file(f"Memory Tree/{name}.md", f"# {name}\n\n")

    def _ensure_file(self, relative_path: str, content: str) -> Path:
        path = self.root_path / relative_path
        if relative_path == "Profile.md":
            lock_path = path.with_name(f".{path.name}.jarvis.lock")
            with _exclusive_sidecar_lock(self.root_path, lock_path):
                if not _contained_exists(self.root_path, path):
                    _replace_text(self.root_path, path, content, expected_content=None)
            return path
        if not _contained_exists(self.root_path, path):
            _replace_text(self.root_path, path, content)
        return path

    def _write_owned_projection_with_evidence(
        self,
        relative_path: str,
        body: str,
        *,
        projection: str,
        store_identity: str,
        source_payload: object,
        accept_legacy_current_context: bool = False,
        accept_legacy_approval_review: bool = False,
        accept_legacy_feedback_report: bool = False,
        accept_legacy_feedback_actions: bool = False,
        accept_legacy_learning_review: bool = False,
        accept_legacy_chat_context: bool = False,
    ) -> tuple[Path, str, str]:
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Jarvis projection store identity is invalid.")
        path = self.root_path / relative_path
        lock_path = path.with_name(f".{path.name}.lock")
        content = (
            "---\n"
            f"jarvis_projection: {projection}\n"
            f"store_identity: {store_identity}\n"
            "---\n\n"
            f"{body.rstrip()}\n"
        )
        content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        source_revision = hashlib.sha256(
            json.dumps(
                _digest_safe_payload(source_payload),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            existing = _read_text(self.root_path, path, max_chars=262_145)
            if existing is not None and len(existing) > 262_144:
                raise FileExistsError("Jarvis projection destination is too large to verify safely.")
            if existing is not None:
                existing_projection = _unique_frontmatter_text_value(existing, "jarvis_projection")
                existing_owner = _unique_frontmatter_text_value(existing, "store_identity")
                owned = existing_projection == projection and existing_owner == store_identity
                legacy = False
                if not owned and (
                    accept_legacy_current_context
                    or accept_legacy_approval_review
                    or accept_legacy_feedback_report
                    or accept_legacy_feedback_actions
                    or accept_legacy_learning_review
                    or accept_legacy_chat_context
                ):
                    legacy_text = _read_text(self.root_path, path, max_chars=262_145) or ""
                    if len(legacy_text) <= 262_144:
                        legacy = (
                            (
                                accept_legacy_current_context
                                and _is_legacy_current_context_projection(legacy_text)
                            )
                            or (
                                accept_legacy_approval_review
                                and _is_legacy_approval_review_projection(legacy_text)
                            )
                            or (
                                accept_legacy_feedback_report
                                and _is_legacy_feedback_report_projection(legacy_text)
                            )
                            or (
                                accept_legacy_feedback_actions
                                and _is_legacy_feedback_actions_projection(legacy_text)
                            )
                            or (
                                accept_legacy_learning_review
                                and _is_legacy_learning_review_projection(legacy_text)
                            )
                            or (
                                accept_legacy_chat_context
                                and _is_legacy_chat_context_projection(legacy_text, body=body)
                            )
                        )
                if not (owned or legacy):
                    raise FileExistsError("Jarvis projection destination ownership could not be verified.")
            _replace_text(self.root_path, path, content, expected_content=existing)
            if _read_text(self.root_path, path) != content:
                raise OSError("Jarvis projection readback did not match the published snapshot.")
        return path, content_sha256, source_revision

    def write_current_context_with_evidence(
        self, body: str, *, store_identity: str, source_payload: object
    ) -> tuple[Path, str, str]:
        return self._write_owned_projection_with_evidence(
            "Memory Tree/Current Context.md",
            body,
            projection="current_context",
            store_identity=store_identity,
            source_payload=source_payload,
            accept_legacy_current_context=True,
        )

    def write_memory_tree_snapshot_with_evidence(
        self, body: str, *, store_identity: str, source_payload: object
    ) -> tuple[Path, str, str]:
        return self._write_owned_projection_with_evidence(
            "Memory Tree/Memory Tree Snapshot.md",
            body,
            projection="memory_tree_snapshot",
            store_identity=store_identity,
            source_payload=source_payload,
        )

    def write_approval_review_with_evidence(
        self, body: str, *, store_identity: str, source_payload: object
    ) -> tuple[Path, str, str]:
        return self._write_owned_projection_with_evidence(
            "Automations/Approval Review.md",
            body,
            projection="approval_review",
            store_identity=store_identity,
            source_payload=source_payload,
            accept_legacy_approval_review=True,
        )

    def write_feedback_report_with_evidence(
        self, body: str, *, store_identity: str, source_payload: object
    ) -> tuple[Path, str, str]:
        return self._write_owned_projection_with_evidence(
            "Automations/Feedback Report.md",
            body,
            projection="feedback_report",
            store_identity=store_identity,
            source_payload=source_payload,
            accept_legacy_feedback_report=True,
        )

    def write_feedback_actions_with_evidence(
        self, body: str, *, store_identity: str, source_payload: object
    ) -> tuple[Path, str, str]:
        return self._write_owned_projection_with_evidence(
            "Automations/Feedback Actions.md",
            body,
            projection="feedback_actions",
            store_identity=store_identity,
            source_payload=source_payload,
            accept_legacy_feedback_actions=True,
        )

    def write_learning_review_with_evidence(
        self, body: str, *, store_identity: str, source_payload: object
    ) -> tuple[Path, str, str]:
        return self._write_owned_projection_with_evidence(
            "Automations/Learning Review.md",
            body,
            projection="learning_review",
            store_identity=store_identity,
            source_payload=source_payload,
            accept_legacy_learning_review=True,
        )

    def write_chat_context_with_evidence(
        self, body: str, *, store_identity: str, source_payload: object
    ) -> tuple[Path, str, str]:
        stamp = datetime.now().strftime("%Y-%m-%d")
        return self._write_owned_projection_with_evidence(
            f"Reflections/{stamp} Chat Context.md",
            body,
            projection="chat_context",
            store_identity=store_identity,
            source_payload=source_payload,
            accept_legacy_chat_context=True,
        )

    def expected_memory_projection_evidence(
        self,
        record: MemoryRecord,
        *,
        memory_id: int,
        store_identity: str,
        memory_revision: int,
        source_digest: str,
        created_at: str,
    ) -> tuple[Path, str]:
        """Return canonical path/digest without publishing any bytes."""
        content = _memory_projection_content(
            record,
            memory_id=memory_id,
            store_identity=store_identity,
            memory_revision=memory_revision,
            source_digest=source_digest,
            created_at=created_at,
        )
        path = (
            self.root_path
            / "Memory Tree"
            / "Records"
            / f"{memory_id:06d} [{store_identity}].md"
        )
        return path, hashlib.sha256(content.encode("utf-8")).hexdigest()

    def write_memory_projection_with_evidence(
        self,
        record: MemoryRecord,
        *,
        memory_id: int,
        store_identity: str,
        memory_revision: int,
        source_digest: str,
        created_at: str,
        expected_prior_content_digest: str | None = None,
    ) -> tuple[Path, str]:
        if expected_prior_content_digest is not None and re.fullmatch(
            r"[0-9a-f]{64}", expected_prior_content_digest
        ) is None:
            raise ValueError("Memory projection prior content digest is invalid.")
        path = (
            self.root_path
            / "Memory Tree"
            / "Records"
            / f"{memory_id:06d} [{store_identity}].md"
        )
        content = _memory_projection_content(
            record,
            memory_id=memory_id,
            store_identity=store_identity,
            memory_revision=memory_revision,
            source_digest=source_digest,
            created_at=created_at,
        )
        content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        lock_path = self.root_path / "Memory Tree" / f".memory-{memory_id}.lock"
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            existing = _read_memory_projection(self.root_path, path)
            if existing == content:
                return path, content_digest
            if existing is not None:
                if not _is_owned_memory_projection(
                    existing,
                    memory_id=memory_id,
                    store_identity=store_identity,
                ):
                    raise FileExistsError(
                        "Memory projection destination ownership could not be verified."
                    )
                existing_revision = _unique_frontmatter_text_value(
                    existing, "memory_revision"
                )
                existing_digest = _unique_frontmatter_text_value(existing, "source_digest")
                if (
                    existing_revision is None
                    or re.fullmatch(r"\d+", existing_revision) is None
                    or int(existing_revision) < 1
                    or existing_digest is None
                    or re.fullmatch(r"[0-9a-f]{64}", existing_digest) is None
                ):
                    raise FileExistsError(
                        "Memory projection freshness could not be verified."
                    )
                prior_revision = int(existing_revision)
                if prior_revision > memory_revision:
                    raise RuntimeError("Stale memory projection snapshot was refused.")
                if prior_revision == memory_revision and existing_digest != source_digest:
                    raise RuntimeError("Conflicting memory projection revision was refused.")
                if prior_revision == memory_revision:
                    raise RuntimeError("Externally changed memory projection was refused.")
                if expected_prior_content_digest is None:
                    raise RuntimeError(
                        "Memory projection prior bytes were not durably bound."
                    )
                if (
                    hashlib.sha256(existing.encode("utf-8")).hexdigest()
                    != expected_prior_content_digest
                ):
                    raise RuntimeError("Externally changed memory projection was refused.")
            _replace_text(
                self.root_path,
                path,
                content,
                expected_content=existing,
            )
            if _read_memory_projection(self.root_path, path) != content:
                raise OSError("Memory projection readback did not match the published record.")
        return path, content_digest

    def verify_memory_projection_evidence(
        self,
        *,
        memory_id: int,
        store_identity: str,
        expected_relative_path: str,
        expected_content_digest: str,
    ) -> bool:
        """Verify the canonical store-owned memory note against retained evidence."""
        with self.canonical_memory_projection_evidence_lock(
            memory_id=memory_id,
            store_identity=store_identity,
            expected_relative_path=expected_relative_path,
            expected_content_digest=expected_content_digest,
        ) as verified:
            return verified

    @contextmanager
    def canonical_memory_projection_evidence_lock(
        self,
        *,
        memory_id: int,
        store_identity: str,
        expected_relative_path: str,
        expected_content_digest: str,
    ) -> Generator[bool, None, None]:
        """Hold the canonical note lock while its retained evidence is finalized."""
        if type(memory_id) is not int or memory_id < 1:
            raise ValueError("Memory projection id is invalid.")
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Memory projection store identity is invalid.")
        if type(expected_relative_path) is not str:
            raise ValueError("Memory projection relative path is invalid.")
        if re.fullmatch(r"[0-9a-f]{64}", expected_content_digest) is None:
            raise ValueError("Memory projection content digest is invalid.")

        path = (
            self.root_path
            / "Memory Tree"
            / "Records"
            / f"{memory_id:06d} [{store_identity}].md"
        )
        canonical_relative_path = str(path.relative_to(self.root_path))
        if expected_relative_path != canonical_relative_path:
            yield False
            return

        lock_path = self.root_path / "Memory Tree" / f".memory-{memory_id}.lock"

        def evidence_is_current() -> bool:
            content = _read_memory_projection(self.root_path, path)
            return bool(
                content is not None
                and _is_owned_memory_projection(
                    content,
                    memory_id=memory_id,
                    store_identity=store_identity,
                )
                and hashlib.sha256(content.encode("utf-8")).hexdigest()
                == expected_content_digest
            )

        with _exclusive_sidecar_lock(self.root_path, lock_path):
            entry_valid = evidence_is_current()
            try:
                yield entry_valid
            except BaseException:
                raise
            else:
                if entry_valid and not evidence_is_current():
                    raise ProfileEvidenceRevalidationError(
                        "Memory projection evidence changed while its lock was held."
                    )

    def _delete_owned_memory_candidate(
        self,
        path: Path,
        *,
        memory_id: int,
        store_identity: str,
        expected_content_digest: str | None = None,
        require_content_digest: bool = False,
    ) -> MemoryProjectionDeleteResult:
        if expected_content_digest is not None and re.fullmatch(
            r"[0-9a-f]{64}", expected_content_digest
        ) is None:
            raise ValueError("Memory projection delete digest is invalid.")
        quarantine = path.with_name(f".{path.name}.delete-{memory_id}.pending")

        def owned(text: str) -> bool:
            if not _is_owned_memory_projection(
                text,
                memory_id=memory_id,
                store_identity=store_identity,
            ):
                return False
            if not require_content_digest:
                return True
            return (
                expected_content_digest is not None
                and hashlib.sha256(text.encode("utf-8")).hexdigest()
                == expected_content_digest
            )

        def restore_quarantine() -> MemoryProjectionDeleteResult:
            parent_fd, source_name = _open_contained_parent(self.root_path, path)
            try:
                try:
                    os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    _rename_noreplace(
                        quarantine.name,
                        source_name,
                        source_dir_fd=parent_fd,
                        destination_dir_fd=parent_fd,
                    )
                    os.fsync(parent_fd)
                    return MemoryProjectionDeleteResult("ownership_mismatch")
                return MemoryProjectionDeleteResult("error")
            finally:
                os.close(parent_fd)

        quarantined = _read_memory_projection(self.root_path, quarantine)
        if quarantined is not None:
            if not owned(quarantined):
                return restore_quarantine()
            recreated = _read_memory_projection(self.root_path, path)
            if recreated is None:
                if _unlink(self.root_path, quarantine):
                    return MemoryProjectionDeleteResult(
                        "deleted", str(path.relative_to(self.root_path))
                    )
                return MemoryProjectionDeleteResult("error")
            if not owned(recreated):
                if _unlink(self.root_path, quarantine):
                    return MemoryProjectionDeleteResult("ownership_mismatch")
                return MemoryProjectionDeleteResult("error")
            if not _unlink(self.root_path, quarantine):
                return MemoryProjectionDeleteResult("error")

        source = _read_memory_projection(self.root_path, path)
        if source is None:
            return MemoryProjectionDeleteResult("absent")
        if not owned(source):
            return MemoryProjectionDeleteResult("ownership_mismatch")

        parent_fd, source_name = _open_contained_parent(self.root_path, path)
        try:
            try:
                _rename_noreplace(
                    source_name,
                    quarantine.name,
                    source_dir_fd=parent_fd,
                    destination_dir_fd=parent_fd,
                )
            except FileNotFoundError:
                return MemoryProjectionDeleteResult("absent")
            except FileExistsError:
                return MemoryProjectionDeleteResult("error")
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

        quarantined = _read_memory_projection(self.root_path, quarantine)
        if quarantined is None:
            return MemoryProjectionDeleteResult("error")
        if not owned(quarantined):
            return restore_quarantine()
        if not _unlink(self.root_path, quarantine):
            return MemoryProjectionDeleteResult("error")
        return MemoryProjectionDeleteResult(
            "deleted", str(path.relative_to(self.root_path))
        )

    def delete_memory_projection(
        self,
        *,
        memory_id: int,
        store_identity: str,
        legacy_category: str | None = None,
        legacy_title: str | None = None,
        expected_content_digest: str | None = None,
    ) -> MemoryProjectionDeleteResult:
        if type(memory_id) is not int or memory_id < 1:
            raise ValueError("Memory projection id is invalid.")
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Memory projection store identity is invalid.")
        if (legacy_category is None) != (legacy_title is None):
            raise ValueError("Legacy memory projection identity is incomplete.")
        if expected_content_digest is not None and re.fullmatch(
            r"[0-9a-f]{64}", expected_content_digest
        ) is None:
            raise ValueError("Memory projection delete digest is invalid.")

        paths = [
            self.root_path
            / "Memory Tree"
            / "Records"
            / f"{memory_id:06d} [{store_identity}].md"
        ]
        if legacy_category is not None and legacy_title is not None:
            paths.append(
                self.root_path
                / "Memory Tree"
                / safe_name(safe_text(legacy_category).title())
                / f"{memory_id:06d} {safe_name(safe_text(legacy_title))}.md"
            )
        lock_path = self.root_path / "Memory Tree" / f".memory-{memory_id}.lock"
        mismatch = False
        deleted_path = ""
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            for index, path in enumerate(dict.fromkeys(paths)):
                try:
                    result = self._delete_owned_memory_candidate(
                        path,
                        memory_id=memory_id,
                        store_identity=store_identity,
                        expected_content_digest=(
                            expected_content_digest if index == 0 else None
                        ),
                        require_content_digest=index == 0,
                    )
                except (OSError, ValueError):
                    return MemoryProjectionDeleteResult("error")
                if result.status == "deleted":
                    deleted_path = deleted_path or result.path_display
                elif result.status == "error":
                    return result
                else:
                    mismatch = mismatch or result.status == "ownership_mismatch"
        if deleted_path:
            return MemoryProjectionDeleteResult("deleted", deleted_path)
        return MemoryProjectionDeleteResult(
            "ownership_mismatch" if mismatch else "absent"
        )

    def delete_legacy_memory_projection(
        self,
        *,
        record: MemoryRecord,
        memory_id: int,
        store_identity: str,
        legacy_category: str | None,
        legacy_title: str | None,
    ) -> MemoryProjectionDeleteResult:
        """Remove only the former category/title mirror after canonical publication."""
        if type(memory_id) is not int or memory_id < 1:
            raise ValueError("Memory projection id is invalid.")
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Memory projection store identity is invalid.")
        if (legacy_category is None) != (legacy_title is None):
            raise ValueError("Legacy memory projection identity is incomplete.")
        if legacy_category is None or legacy_title is None:
            return MemoryProjectionDeleteResult("absent")
        path = (
            self.root_path
            / "Memory Tree"
            / safe_name(safe_text(legacy_category).title())
            / f"{memory_id:06d} {safe_name(safe_text(legacy_title))}.md"
        )
        lock_path = self.root_path / "Memory Tree" / f".memory-{memory_id}.lock"
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            try:
                existing = _read_memory_projection(self.root_path, path)
                if existing is None:
                    return MemoryProjectionDeleteResult("absent")
                if not _is_owned_memory_projection(
                    existing,
                    memory_id=memory_id,
                    store_identity=store_identity,
                ):
                    return MemoryProjectionDeleteResult("ownership_mismatch")
                created_date = _unique_frontmatter_text_value(existing, "created")
                if (
                    created_date is None
                    or re.fullmatch(r"\d{4}-\d{2}-\d{2}", created_date) is None
                ):
                    return MemoryProjectionDeleteResult("ownership_mismatch")
                expected = _legacy_memory_projection_content(
                    record,
                    memory_id=memory_id,
                    store_identity=store_identity,
                    created_date=created_date,
                )
                return self._delete_owned_memory_candidate(
                    path,
                    memory_id=memory_id,
                    store_identity=store_identity,
                    expected_content_digest=hashlib.sha256(
                        expected.encode("utf-8")
                    ).hexdigest(),
                    require_content_digest=True,
                )
            except (OSError, ValueError):
                return MemoryProjectionDeleteResult("error")

    def write_memory(self, record: MemoryRecord, memory_id: int, *, store_identity: str) -> Path:
        if type(memory_id) is not int or memory_id < 1:
            raise ValueError("Memory mirror id is invalid.")
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Memory mirror store identity is invalid.")
        category = safe_name(safe_text(record.category).title())
        folder = self.root_path / "Memory Tree" / category

        stamp = datetime.now().strftime("%Y-%m-%d")
        path = folder / f"{memory_id:06d} {safe_name(safe_text(record.title))}.md"
        legacy_content = (
            "---\n"
            f"category: {safe_text(record.category)}\n"
            f"source: {safe_text(record.source)}\n"
            f"confidence: {record.confidence}\n"
            f"created: {stamp}\n"
            "---\n\n"
            f"# {safe_text(record.title)}\n\n"
            f"{safe_text(record.body).strip()}\n"
        )
        content = _legacy_memory_projection_content(
            record,
            memory_id=memory_id,
            store_identity=store_identity,
            created_date=stamp,
        )
        lock_path = self.root_path / "Memory Tree" / f".memory-{memory_id}.lock"
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            existing = _read_text(self.root_path, path)
            if existing == content:
                return path
            owned = (
                existing is not None
                and _unique_frontmatter_text_value(existing, "jarvis_projection") == "memory"
                and _unique_frontmatter_text_value(existing, "store_identity") == store_identity
                and _unique_frontmatter_text_value(existing, "memory_id") == str(memory_id)
            )
            if owned:
                owned_stamp = _unique_frontmatter_text_value(existing, "created") or ""
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", owned_stamp) is None:
                    raise FileExistsError("Memory mirror creation date could not be verified.")
                content = content.replace(f"created: {stamp}\n", f"created: {owned_stamp}\n", 1)
            legacy = existing == legacy_content
            if existing is not None and not (owned or legacy):
                legacy_stamp = _unique_frontmatter_text_value(existing, "created") or ""
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", legacy_stamp):
                    historical_legacy = legacy_content.replace(
                        f"created: {stamp}\n",
                        f"created: {legacy_stamp}\n",
                        1,
                    )
                    if existing == historical_legacy:
                        legacy = True
                        content = content.replace(
                            f"created: {stamp}\n",
                            f"created: {legacy_stamp}\n",
                            1,
                        )
            if existing is not None and not (owned or legacy):
                raise FileExistsError("Memory mirror destination ownership could not be verified.")
            _replace_text(self.root_path, path, content, expected_content=existing)
            if _read_text(self.root_path, path) != content:
                raise OSError("Memory mirror readback did not match the published record.")
        return path

    def write_organized_memory(
        self,
        record: MemoryRecord,
        memory_id: int,
        created_at: str,
    ) -> tuple[Path, bool, str]:
        if type(memory_id) is not int or memory_id < 1:
            raise ValueError("Organized memory id is invalid.")
        stamp_match = re.match(r"^(\d{4}-\d{2}-\d{2})", str(created_at or ""))
        if stamp_match is None:
            raise ValueError("Organized memory creation time is invalid.")
        category = safe_name(safe_text(record.category).title())
        path = (
            self.root_path
            / "Memory Tree"
            / category
            / f"{memory_id:06d} {safe_name(safe_text(record.title))}.md"
        )
        content = (
            "---\n"
            f"category: {safe_text(record.category)}\n"
            f"source: {safe_text(record.source)}\n"
            f"confidence: {record.confidence}\n"
            f"created: {stamp_match.group(1)}\n"
            "---\n\n"
            f"# {safe_text(record.title)}\n\n"
            f"{safe_text(record.body).strip()}\n"
        )
        content_bytes = content.encode("utf-8")
        content_sha256 = hashlib.sha256(content_bytes).hexdigest()
        lock_path = path.with_name(f".{path.name}.jarvis.lock")
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            existing_bytes = _read_bytes_bounded(
                self.root_path,
                path,
                max_bytes=MAX_ORGANIZED_PROJECTION_BYTES,
            )
            if existing_bytes == content_bytes:
                return path, False, content_sha256
            if existing_bytes is not None:
                raise FileExistsError("Organized memory destination ownership could not be verified.")
            _replace_text(self.root_path, path, content, expected_content=None)
            if _read_bytes_bounded(
                self.root_path,
                path,
                max_bytes=MAX_ORGANIZED_PROJECTION_BYTES,
            ) != content_bytes:
                raise OSError("Organized memory mirror readback did not match the published record.")
        return path, True, content_sha256

    def append_daily(self, heading: str, body: str) -> Path:
        day = datetime.now().strftime("%Y-%m-%d")
        return self.append_daily_for_date(day, heading, body)

    @contextmanager
    def daily_append_fence(
        self,
        target_date: str,
    ) -> Generator[Callable[[str, str], Path], None, None]:
        """Hold one daily-note sidecar while a source snapshot is published."""
        day = _validated_daily_target_date(target_date)
        path = self.root_path / "Daily" / f"{day}.md"
        lock_path = path.with_name(f".{path.name}.jarvis.lock")
        attempted = False
        active = False
        publication_guard = threading.Lock()
        owner_pid = os.getpid()

        def append(heading: str, body: str) -> Path:
            nonlocal attempted
            if os.getpid() != owner_pid:
                raise RuntimeError("Daily note publication authority is not active.")
            with publication_guard:
                if not active:
                    raise RuntimeError("Daily note publication authority is not active.")
                if attempted:
                    raise RuntimeError("Daily note publication was already attempted.")
                attempted = True
                safe_heading = _escape_scheduled_note_markers(safe_text(heading))
                safe_body = _escape_scheduled_note_markers(safe_text(body).strip())
                _append_text_under_inode_lock(
                    self.root_path,
                    path,
                    f"\n## {safe_heading}\n\n{safe_body}\n",
                    initial_content=f"# {day}\n\n",
                )
                return path

        with _exclusive_sidecar_lock(self.root_path, lock_path):
            with publication_guard:
                active = True
            try:
                yield append
            finally:
                if os.getpid() == owner_pid:
                    with publication_guard:
                        active = False

    def append_daily_for_date(self, target_date: str, heading: str, body: str) -> Path:
        day = _validated_daily_target_date(target_date)
        path = self.root_path / "Daily" / f"{day}.md"
        safe_heading = _escape_scheduled_note_markers(safe_text(heading))
        safe_body = _escape_scheduled_note_markers(safe_text(body).strip())
        _append_text(
            self.root_path,
            path,
            f"\n## {safe_heading}\n\n{safe_body}\n",
            initial_content=f"# {day}\n\n",
        )
        return path

    def append_daily_once(self, marker: str, heading: str, body: str) -> tuple[Path, bool]:
        day = datetime.now().strftime("%Y-%m-%d")
        return self.append_daily_once_for_date(day, marker, heading, body)

    def append_daily_once_for_date(
        self,
        target_date: str,
        marker: str,
        heading: str,
        body: str,
    ) -> tuple[Path, bool]:
        """Append an idempotent daily entry to an explicitly reserved date."""
        target_date = _validated_daily_target_date(target_date)
        path = self.root_path / "Daily" / f"{target_date}.md"
        safe_marker = _escape_scheduled_note_markers(safe_text(marker).strip())
        safe_heading = _escape_scheduled_note_markers(safe_text(heading))
        safe_body = _escape_scheduled_note_markers(safe_text(body).strip())
        appended = _append_text_once(
            self.root_path,
            path,
            marker=safe_marker,
            content=(
                f"\n## {safe_heading}\n\n{safe_body}\n"
                f"<!-- {safe_marker} -->\n"
            ),
            initial_content=f"# {target_date}\n\n",
        )
        return path, appended

    def append_scheduled_daily_once_for_date(
        self,
        target_date: str,
        source_key: str,
        heading: str,
        body: str,
        *,
        effect_authority: Callable[[], None] | None = None,
    ) -> tuple[Path, bool, str]:
        """Publish one exact scheduled occurrence into its reserved daily note."""
        if type(target_date) is not str or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}", target_date
        ) is None:
            raise ValueError("Scheduled daily target date must use YYYY-MM-DD.")
        try:
            parsed_date = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("Scheduled daily target date must be a real calendar date.") from exc
        if parsed_date.strftime("%Y-%m-%d") != target_date:
            raise ValueError("Scheduled daily target date must be canonical.")
        path = self.root_path / "Daily" / f"{target_date}.md"
        appended, persisted_body = _publish_scheduled_note_block(
            self.root_path,
            path,
            source_key=source_key,
            heading=heading,
            body=body,
            initial_content=f"# {target_date}\n\n",
            effect_authority=effect_authority,
        )
        return path, appended, persisted_body

    def scheduled_note_payload(self, body: object) -> str:
        """Return the exact bounded body safe for outbox and marker publication."""
        payload = _escape_scheduled_note_markers(safe_text(body).strip())
        if not payload:
            raise ValueError("Scheduled note body is empty.")
        if len(payload.encode("utf-8")) > MAX_SCHEDULED_NOTE_BLOCK_BYTES:
            raise ValueError("Scheduled note body exceeds the bounded publication size.")
        return payload

    def scheduled_note_publication_evidence(
        self,
        target_kind: str,
        target_date: str,
        source_key: str,
        title: str,
    ) -> tuple[Path, str] | None:
        """Read exact durable evidence for a previously started publication."""
        if type(target_date) is not str or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}", target_date
        ) is None:
            raise ValueError("Scheduled note evidence date must use YYYY-MM-DD.")
        try:
            parsed_date = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("Scheduled note evidence date must be a real calendar date.") from exc
        if parsed_date.strftime("%Y-%m-%d") != target_date:
            raise ValueError("Scheduled note evidence date must be canonical.")
        if target_kind == "daily":
            path = self.root_path / "Daily" / f"{target_date}.md"
        elif target_kind == "reflection":
            safe_title = safe_name(safe_text(title))
            path = self.root_path / "Reflections" / f"{target_date} {safe_title}.md"
        else:
            raise ValueError("Scheduled note evidence target kind is invalid.")
        persisted = _scheduled_note_block_evidence(
            self.root_path,
            path,
            source_key=source_key,
            heading=title,
        )
        return None if persisted is None else (path, persisted)

    def append_scheduled_reflection_once_for_date(
        self,
        target_date: str,
        source_key: str,
        title: str,
        body: str,
        *,
        effect_authority: Callable[[], None] | None = None,
    ) -> tuple[Path, bool, str]:
        """Publish one exact scheduled reflection occurrence without overwrites."""
        if type(target_date) is not str or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}", target_date
        ) is None:
            raise ValueError("Scheduled reflection target date must use YYYY-MM-DD.")
        try:
            parsed_date = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("Scheduled reflection target date must be a real calendar date.") from exc
        if parsed_date.strftime("%Y-%m-%d") != target_date:
            raise ValueError("Scheduled reflection target date must be canonical.")
        safe_title = safe_name(safe_text(title))
        path = self.root_path / "Reflections" / f"{target_date} {safe_title}.md"
        appended, persisted_body = _publish_scheduled_note_block(
            self.root_path,
            path,
            source_key=source_key,
            heading=title,
            body=body,
            initial_content=f"# {safe_text(title).strip()}\n\n",
            effect_authority=effect_authority,
        )
        return path, appended, persisted_body

    def append_profile(self, heading: str, body: str) -> Path:
        path = self.root_path / "Profile.md"
        _atomic_append_text_once(
            self.root_path,
            path,
            content=f"\n## {safe_text(heading)}\n\n{safe_text(body).strip()}\n",
            initial_content="# Profile\n\n",
        )
        return path

    def upgrade_profile_legacy_blocks(
        self,
        notes: Iterable[tuple[str, str, str]],
    ) -> bool:
        """Atomically upgrade every exact store-backed trailing-marker profile block."""
        materialized = tuple(notes)
        if len(materialized) > 100:
            raise ValueError("Profile legacy upgrade set is too large.")
        replacements: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for item in materialized:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError("Profile legacy upgrade entries must be triples.")
            source_key, heading, body = item
            if (
                re.fullmatch(r"profile-note:v1:[0-9a-f]{64}", source_key) is None
                or type(heading) is not str
                or type(body) is not str
                or source_key in seen
            ):
                raise ValueError("Profile legacy upgrade entry is invalid.")
            seen.add(source_key)
            marker = f"jarvis-{source_key}"
            legacy = (
                f"\n## {safe_text(heading)}\n\n{safe_text(body).strip()}\n"
                f"<!-- {marker} -->\n"
            )
            replacements.append((source_key, legacy, _profile_note_block(source_key, heading, body)))
        if not replacements:
            return False

        path = self.root_path / "Profile.md"
        lock_path = path.with_name(f".{path.name}.jarvis.lock")
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            _recover_atomic_exchange_journals(self.root_path, path)
            existing = _read_text(self.root_path, path)
            if existing is None:
                return False
            _manual, _blocks, malformed = _split_profile_owned_blocks(existing)
            if not malformed:
                return False
            candidate = existing
            changed = False
            for _source_key, legacy, canonical in replacements:
                count = candidate.count(legacy)
                if count > 1:
                    raise RuntimeError(
                        "Profile legacy ownership marker structure is ambiguous."
                    )
                if count == 1:
                    candidate = candidate.replace(legacy, canonical, 1)
                    changed = True
            if not changed:
                return False
            _manual, blocks, malformed = _split_profile_owned_blocks(candidate)
            if malformed or any(len(items) != 1 for items in blocks.values()):
                raise RuntimeError(
                    "Profile legacy ownership marker structure remains malformed or ambiguous."
                )
            _replace_text(self.root_path, path, candidate, expected_content=existing)
            return True

    def append_profile_once(self, source_key: str, heading: str, body: str) -> tuple[Path, bool]:
        if re.fullmatch(r"profile-note:v1:[0-9a-f]{64}", source_key) is None:
            raise ValueError("Profile note source key is invalid.")
        path = self.root_path / "Profile.md"
        marker = f"jarvis-{source_key}"
        start_marker = marker.replace("jarvis-profile-note:v1:", "jarvis-profile-note-start:v1:", 1)
        legacy_content = (
            f"\n## {safe_text(heading)}\n\n{safe_text(body).strip()}\n"
            f"<!-- {marker} -->\n"
        )
        content = _profile_note_block(source_key, heading, body)
        appended = _atomic_append_text_once(
            self.root_path,
            path,
            content=content,
            initial_content="# Profile\n\n",
            match_content=content,
            legacy_match_content=legacy_content,
            owned_start_marker=start_marker,
            owned_marker=marker,
        )
        return path, appended

    @contextmanager
    def profile_grounding_evidence_lock(self) -> Generator[None, None, None]:
        """Hold the complete Profile.md snapshot stable through one disclosure."""
        path = self.root_path / "Profile.md"
        lock_path = path.with_name(f".{path.name}.jarvis.lock")

        def fingerprint() -> bytes:
            content = _read_text(self.root_path, path, max_chars=1_000_001)
            if content is None:
                content = ""
            return hashlib.sha256(content.encode("utf-8")).digest()

        with _exclusive_sidecar_lock(self.root_path, lock_path):
            entry_fingerprint = fingerprint()
            yield
            if not hmac.compare_digest(entry_fingerprint, fingerprint()):
                raise ProfileEvidenceRevalidationError(
                    "Profile grounding changed while its disclosure lock was held."
                )

    @contextmanager
    def canonical_profile_note_evidence_lock(
        self,
        *,
        source_key: str,
        heading: str,
        body: str,
    ) -> Generator[bool, None, None]:
        """Hold Profile.md stable while one exact generated block is finalized."""
        expected = _profile_note_block(source_key, heading, body).lstrip("\n")
        path = self.root_path / "Profile.md"
        lock_path = path.with_name(f".{path.name}.jarvis.lock")

        def evidence_is_current() -> bool:
            content = _read_text(self.root_path, path, max_chars=1_000_001)
            if content is None or len(content) > 1_000_000:
                return False
            _manual, blocks, malformed = _split_profile_owned_blocks(content)
            return bool(
                not malformed
                and blocks.get(source_key) == (expected,)
                and all(len(items) == 1 for items in blocks.values())
            )

        with _exclusive_sidecar_lock(self.root_path, lock_path):
            entry_valid = evidence_is_current()
            yield entry_valid
            if entry_valid and not evidence_is_current():
                raise ProfileEvidenceRevalidationError(
                    "Profile projection evidence changed while the evidence lock was held."
                )

    def read_profile_grounding(
        self,
        expected_notes: tuple[object, ...],
        *,
        max_chars: int = 1800,
    ) -> ProfileGroundingView:
        """Read manual profile text plus only exact store-validated generated blocks."""
        if type(max_chars) is not int or not 1 <= max_chars <= 20_000:
            raise ValueError("Profile grounding read limit is invalid.")
        expected: dict[str, tuple[str, str]] = {}
        invalid = False
        for note in expected_notes:
            try:
                source_key = note.source_key
                heading = note.heading
                body = note.body
            except AttributeError:
                invalid = True
                continue
            if (
                type(source_key) is not str
                or type(heading) is not str
                or type(body) is not str
                or source_key in expected
                or re.fullmatch(r"profile-note:v1:[0-9a-f]{64}", source_key) is None
            ):
                invalid = True
                continue
            expected[source_key] = (heading, body)

        path = self.root_path / "Profile.md"
        content = _read_text(self.root_path, path, max_chars=1_000_001)
        if content is None:
            content = ""
        truncated = len(content) > 1_000_000
        if truncated:
            content = content[:1_000_000]
            invalid = True
        manual, blocks, malformed = _split_profile_owned_blocks(content)
        invalid = invalid or malformed
        generated: list[tuple[str, str]] = []
        for source_key, (heading, body) in expected.items():
            expected_block = _profile_note_block(source_key, heading, body).lstrip("\n")
            if blocks.get(source_key) != (expected_block,):
                invalid = True
                continue
            generated.append(
                (source_key, f"## {safe_text(heading)}\n\n{safe_text(body).strip()}")
            )
        if set(blocks) != {source_key for source_key, _text in generated}:
            invalid = True

        visible_parts: list[tuple[str | None, str]] = []
        manual_text = safe_text(manual).strip()
        if manual_text:
            visible_parts.append((None, manual_text))
        visible_parts.extend(
            (source_key, text.strip())
            for source_key, text in generated
            if text.strip()
        )
        visible_chunks: list[str] = []
        generated_ends: list[tuple[str, int]] = []
        visible_length = 0
        for source_key, text in visible_parts:
            if visible_chunks:
                visible_length += 2
            visible_chunks.append(text)
            visible_length += len(text)
            if source_key is not None:
                generated_ends.append((source_key, visible_length))
        visible = "\n\n".join(visible_chunks)
        clipped_length = min(len(visible), max_chars)
        truncated = truncated or len(visible) > max_chars
        return ProfileGroundingView(
            text=safe_text(visible[:max_chars]),
            verified_source_keys=tuple(
                source_key
                for source_key, section_end in generated_ends
                if section_end <= clipped_length
            ),
            invalid=invalid,
            truncated=truncated,
        )

    def append_note(self, note_path: str | Path, body: str, *, heading: str) -> tuple[Path, bool]:
        path = Path(note_path).expanduser()
        if not path.is_absolute():
            path = self.root_path / path
        created = _atomic_append_text(
            self.root_path,
            path,
            body.rstrip() + "\n",
            initial_content=f"# {heading}\n\n",
            existing_prefix="\n\n",
        )
        return path, created

    def note_matches_exact_bytes(self, note_path: str | Path, expected: bytes) -> bool | None:
        path = Path(note_path).expanduser()
        if not path.is_absolute():
            path = self.root_path / path
        try:
            parent_fd, name = _open_contained_parent(
                self.root_path,
                path,
                create_parents=False,
            )
        except FileNotFoundError:
            return None
        fd: int | None = None
        try:
            try:
                target_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None
            if stat.S_ISLNK(target_stat.st_mode):
                raise ValueError(_CONTAINMENT_ERROR)
            if not stat.S_ISREG(target_stat.st_mode):
                raise IsADirectoryError("Obsidian note is not a regular file.")
            try:
                fd = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                return None
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ValueError(_CONTAINMENT_ERROR) from exc
                raise
            target_stat = os.fstat(fd)
            if not stat.S_ISREG(target_stat.st_mode):
                raise IsADirectoryError("Obsidian note is not a regular file.")
            if target_stat.st_size != len(expected):
                return False
            chunks: list[bytes] = []
            remaining = len(expected) + 1
            while remaining > 0:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks) == expected
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent_fd)

    def read_note_bounded(
        self,
        note_path: str | Path,
        *,
        max_chars: int,
    ) -> str | None:
        """Read one regular vault file without following links or exceeding the cap."""
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError("Vault note read limit must be a positive integer.")
        path = Path(note_path).expanduser()
        if not path.is_absolute():
            path = self.root_path / path
        try:
            parent_fd, name = _open_contained_parent(
                self.root_path,
                path,
                create_parents=False,
            )
        except FileNotFoundError:
            return None
        fd: int | None = None
        try:
            try:
                target_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None
            if stat.S_ISLNK(target_stat.st_mode):
                raise ValueError(_CONTAINMENT_ERROR)
            if not stat.S_ISREG(target_stat.st_mode):
                raise IsADirectoryError("Obsidian note is not a regular file.")
            try:
                fd = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ValueError(_CONTAINMENT_ERROR) from exc
                raise
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise IsADirectoryError("Obsidian note is not a regular file.")
            with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as handle:
                fd = None
                content = handle.read(max_chars + 1)
            if len(content) > max_chars:
                raise OverflowError("Obsidian note exceeds the bounded read limit.")
            return content
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent_fd)

    def create_note(self, note_path: str | Path, body: str, *, heading: str) -> Path:
        path = Path(note_path).expanduser()
        if not path.is_absolute():
            path = self.root_path / path
        try:
            _replace_text(
                self.root_path,
                path,
                f"# {heading}\n\n{body.rstrip()}\n",
                expected_content=None,
            )
        except RuntimeError as exc:
            if str(exc) != "Obsidian destination changed during atomic publication.":
                raise
            raise FileExistsError(path) from exc
        return path

    def read_profile(self, max_chars: int = 4000) -> str:
        path = self.root_path / "Profile.md"
        source_limit = min(max(int(max_chars), 1) * 16 + 8192, 1_000_000)
        content = _read_text(self.root_path, path, max_chars=source_limit)
        if content is None:
            return ""
        manual, _blocks, _malformed = _split_profile_owned_blocks(content)
        return safe_text(manual[:max_chars])

    def write_skill(
        self,
        name: str,
        trigger: str,
        body: str,
        tags: str = "",
        *,
        skill_id: int | None = None,
        store_identity: str | None = None,
        skill_revision: int | None = None,
        source_digest: str | None = None,
    ) -> Path:
        title = safe_name(safe_text(name))
        markdown_name = _skill_markdown_value(name)
        markdown_trigger = _skill_markdown_value(trigger)
        markdown_body = _skill_markdown_value(body)
        if skill_id is None or store_identity is None:
            path = self.root_path / "Skills" / f"{title}.md"
            content = (
                "---\n"
                f"name: {_skill_frontmatter_value(name)}\n"
                f"trigger: {_skill_frontmatter_value(trigger)}\n"
                f"tags: {_skill_frontmatter_value(tags)}\n"
                "---\n\n"
                f"# {markdown_name}\n\n"
                f"## Trigger\n\n{markdown_trigger.strip()}\n\n"
                f"## Procedure\n\n{markdown_body.strip()}\n"
            )
            _replace_text(self.root_path, path, content)
            return path
        if type(skill_id) is not int or skill_id <= 0 or not re.fullmatch(r"[0-9a-f]{32}", store_identity):
            raise ValueError("Skill projection identity is invalid.")
        if type(skill_revision) is not int or skill_revision <= 0:
            raise ValueError("Skill projection revision is invalid.")
        if type(source_digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", source_digest):
            raise ValueError("Skill projection source digest is invalid.")
        content = (
            "---\n"
            "jarvis_projection: skill\n"
            f"store_identity: {store_identity}\n"
            f"skill_id: {skill_id}\n"
            f"skill_revision: {skill_revision}\n"
            f"source_digest: {source_digest}\n"
            f"name: {_skill_frontmatter_value(name)}\n"
            f"trigger: {_skill_frontmatter_value(trigger)}\n"
            f"tags: {_skill_frontmatter_value(tags)}\n"
            "---\n\n"
            f"# {markdown_name}\n\n"
            f"## Trigger\n\n{markdown_trigger.strip()}\n\n"
            f"## Procedure\n\n{markdown_body.strip()}\n"
        )
        lock_path = self.root_path / "Skills" / ".skill-projections.lock"
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            legacy_path = self.root_path / "Skills" / f"{title}.md"
            old_id_path = self.root_path / "Skills" / f"{title} [{skill_id}].md"
            canonical_path = (
                self.root_path
                / "Skills"
                / f"{title} [{skill_id}-{store_identity}].md"
            )
            candidates = (canonical_path, old_id_path, legacy_path)
            path: Path | None = None
            legacy = _read_skill_projection(self.root_path, legacy_path)
            if legacy is not None and _is_legacy_skill_projection(
                legacy, name=name, trigger=trigger, body=body, tags=tags
            ):
                path = legacy_path
            else:
                for candidate in candidates:
                    candidate_text = _read_skill_projection(self.root_path, candidate)
                    if candidate_text is not None and _is_owned_skill_projection(
                        candidate_text,
                        skill_id=skill_id,
                        store_identity=store_identity,
                    ):
                        path = candidate
                        break
            path = path or canonical_path
            existing = _read_skill_projection(self.root_path, path)
            if existing is not None:
                owned = _is_owned_skill_projection(
                    existing,
                    skill_id=skill_id,
                    store_identity=store_identity,
                )
                if not owned and not (
                    path == legacy_path
                    and _is_legacy_skill_projection(
                        existing, name=name, trigger=trigger, body=body, tags=tags
                    )
                ):
                    raise FileExistsError("Skill mirror destination ownership could not be verified.")
            _replace_text(self.root_path, path, content)
            if _read_skill_projection(self.root_path, path) != content:
                raise OSError("Skill mirror readback did not match the published snapshot.")
        return path

    def write_skill_with_evidence(
        self,
        name: str,
        trigger: str,
        body: str,
        tags: str = "",
        *,
        skill_id: int,
        store_identity: str,
        skill_revision: int,
        source_digest: str,
    ) -> tuple[Path, str]:
        path = self.write_skill(
            name,
            trigger,
            body,
            tags,
            skill_id=skill_id,
            store_identity=store_identity,
            skill_revision=skill_revision,
            source_digest=source_digest,
        )
        content = _read_text(self.root_path, path)
        if content is None:
            raise OSError("Skill mirror disappeared before evidence could be recorded.")
        return path, hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _delete_owned_skill_candidate(
        self,
        path: Path,
        *,
        skill_id: int,
        store_identity: str,
    ) -> SkillProjectionDeleteResult:
        quarantine = path.with_name(f".{path.name}.delete-{skill_id}.pending")

        def owned(text: str) -> bool:
            return _is_owned_skill_projection(
                text,
                skill_id=skill_id,
                store_identity=store_identity,
            )

        def restore_quarantine() -> SkillProjectionDeleteResult:
            parent_fd, source_name = _open_contained_parent(self.root_path, path)
            try:
                quarantine_name = quarantine.name
                try:
                    os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    _rename_noreplace(
                        quarantine_name,
                        source_name,
                        source_dir_fd=parent_fd,
                        destination_dir_fd=parent_fd,
                    )
                    os.fsync(parent_fd)
                    return SkillProjectionDeleteResult("ownership_mismatch")
                return SkillProjectionDeleteResult("error")
            finally:
                os.close(parent_fd)

        quarantined = _read_skill_projection(self.root_path, quarantine)
        if quarantined is not None:
            if not owned(quarantined):
                return restore_quarantine()
            recreated = _read_skill_projection(self.root_path, path)
            if recreated is None:
                if _unlink(self.root_path, quarantine):
                    return SkillProjectionDeleteResult(
                        "deleted", str(path.relative_to(self.root_path))
                    )
                return SkillProjectionDeleteResult("error")
            if not owned(recreated):
                if _unlink(self.root_path, quarantine):
                    return SkillProjectionDeleteResult("ownership_mismatch")
                return SkillProjectionDeleteResult("error")
            if not _unlink(self.root_path, quarantine):
                return SkillProjectionDeleteResult("error")

        parent_fd, source_name = _open_contained_parent(self.root_path, path)
        try:
            try:
                _rename_noreplace(
                    source_name,
                    quarantine.name,
                    source_dir_fd=parent_fd,
                    destination_dir_fd=parent_fd,
                )
            except FileNotFoundError:
                return SkillProjectionDeleteResult("absent")
            except FileExistsError:
                return SkillProjectionDeleteResult("error")
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

        quarantined = _read_skill_projection(self.root_path, quarantine)
        if quarantined is None:
            return SkillProjectionDeleteResult("error")
        if not owned(quarantined):
            return restore_quarantine()
        if not _unlink(self.root_path, quarantine):
            return SkillProjectionDeleteResult("error")
        return SkillProjectionDeleteResult(
            "deleted", str(path.relative_to(self.root_path))
        )

    def delete_skill_projection(
        self,
        name: str,
        *,
        skill_id: int,
        store_identity: str,
    ) -> SkillProjectionDeleteResult:
        if type(skill_id) is not int or skill_id <= 0 or not re.fullmatch(
            r"[0-9a-f]{32}", store_identity
        ):
            raise ValueError("Skill projection identity is invalid.")
        title = safe_name(safe_text(name))
        lock_path = self.root_path / "Skills" / ".skill-projections.lock"
        mismatch = False
        deleted_path = ""
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            for path in (
                self.root_path
                / "Skills"
                / f"{title} [{skill_id}-{store_identity}].md",
                self.root_path / "Skills" / f"{title} [{skill_id}].md",
                self.root_path / "Skills" / f"{title}.md",
            ):
                try:
                    result = self._delete_owned_skill_candidate(
                        path,
                        skill_id=skill_id,
                        store_identity=store_identity,
                    )
                except (OSError, ValueError):
                    return SkillProjectionDeleteResult("error")
                if result.status == "deleted":
                    deleted_path = deleted_path or result.path_display
                    continue
                if result.status == "error":
                    return result
                mismatch = mismatch or result.status == "ownership_mismatch"
        if deleted_path:
            return SkillProjectionDeleteResult("deleted", deleted_path)
        return SkillProjectionDeleteResult("ownership_mismatch" if mismatch else "absent")

    def delete_skill(
        self,
        name: str,
        *,
        skill_id: int | None = None,
        store_identity: str | None = None,
    ) -> bool:
        title = safe_name(safe_text(name))
        if skill_id is None or store_identity is None:
            return _unlink(self.root_path, self.root_path / "Skills" / f"{title}.md")
        return self.delete_skill_projection(
            name,
            skill_id=skill_id,
            store_identity=store_identity,
        ).status == "deleted"

    def write_session(self, session_id: str, body: str) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d")
        path = self.root_path / "Sessions" / f"{stamp} {safe_name(safe_text(session_id))}.md"
        _replace_text(self.root_path, path, safe_text(body))
        return path

    def write_reflection(self, title: str, body: str) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d")
        path = self.root_path / "Reflections" / f"{stamp} {safe_name(safe_text(title))}.md"
        _replace_text(self.root_path, path, safe_text(body))
        return path

    def write_source(self, title: str, body: str) -> Path:
        path = self.root_path / "Sources" / f"{safe_name(safe_text(title))}.md"
        _replace_text(self.root_path, path, safe_text(body))
        return path

    def write_execution_case(self, case_id: int, request: str, body: str) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d %H%M")
        folder = self.root_path / "Automations" / "Execution Cases"
        path = folder / f"{stamp} Case {case_id:04d} {safe_name(safe_text(request))}.md"
        _replace_text(self.root_path, path, safe_text(body).strip() + "\n")
        return path

    def append_execution_case_event(self, note_path: str, body: str) -> Path:
        candidate = Path(note_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root_path / candidate
        _append_text(self.root_path, candidate, safe_text(body).strip() + "\n")
        return candidate.resolve(strict=False)

    def write_return_brief(self, body: str) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d %H%M")
        path = self.root_path / "Reflections" / f"{stamp} Return Brief.md"
        _replace_text(self.root_path, path, safe_text(body).strip() + "\n")
        return path

    def read_inbox(self) -> str:
        path = self.root_path / "Inbox.md"
        return _read_text(self.root_path, path) or ""

    def reset_inbox(self) -> Path:
        path = self.root_path / "Inbox.md"
        _replace_text(self.root_path, path, "# Inbox\n\n")
        return path

    def inbox_snapshot(self, *, max_bytes: int) -> tuple[str, str] | None:
        """Read one strict Inbox snapshot and its opaque binding from identical bytes."""
        root_fd: int | None = None
        try:
            resolved_root = self.root_path.expanduser().resolve(strict=True)
            root_fd = os.open(resolved_root, _DIRECTORY_FLAGS)
            root_stat = os.fstat(root_fd)
            content = _read_bytes_at(
                root_fd,
                "Inbox.md",
                max_bytes=max_bytes,
            )
            if content is None:
                return None
            return (
                content.decode("utf-8", errors="strict"),
                _inbox_clear_target_binding(
                    resolved_root,
                    root_stat,
                    content,
                ),
            )
        except OverflowError:
            raise
        except (IsADirectoryError, OSError, UnicodeError, ValueError) as exc:
            raise InboxSnapshotError("Inbox snapshot could not be bound safely.") from exc
        finally:
            if root_fd is not None:
                os.close(root_fd)

    def inbox_clear_binding(self, *, max_bytes: int) -> str | None:
        snapshot = self.inbox_snapshot(max_bytes=max_bytes)
        return snapshot[1] if snapshot is not None else None

    def reset_inbox_if_unchanged(
        self,
        expected_binding: str,
        *,
        max_bytes: int,
    ) -> tuple[Path, bool]:
        """Clear Inbox.md only when it still matches the reviewed snapshot."""
        if type(expected_binding) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", expected_binding
        ):
            raise ValueError("Inbox target binding is malformed.")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Inbox read limit must be a positive integer.")

        try:
            resolved_root = self.root_path.expanduser().resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise InboxSnapshotError("Inbox snapshot could not be resolved safely.") from exc
        path = resolved_root / "Inbox.md"
        root_fd: int | None = None
        try:
            root_fd = os.open(resolved_root, _DIRECTORY_FLAGS)
            root_stat = os.fstat(root_fd)
            try:
                current_bytes = _read_bytes_at(
                    root_fd,
                    "Inbox.md",
                    max_bytes=max_bytes,
                )
                if current_bytes is None:
                    return path, False
                current_bytes.decode("utf-8", errors="strict")
                current_binding = _inbox_clear_target_binding(
                    resolved_root,
                    root_stat,
                    current_bytes,
                )
            except OverflowError:
                raise
            except (IsADirectoryError, OSError, UnicodeError, ValueError) as exc:
                raise InboxSnapshotError(
                    "Inbox snapshot could not be read safely."
                ) from exc
            if not secrets.compare_digest(current_binding, expected_binding):
                return path, False
            try:
                _replace_text(
                    resolved_root,
                    path,
                    "# Inbox\n\n",
                    expected_bytes=current_bytes,
                    parent_fd=root_fd,
                    name="Inbox.md",
                )
            except RuntimeError as exc:
                if str(exc) == "Obsidian destination changed during atomic publication.":
                    return path, False
                raise
            return path, True
        finally:
            if root_fd is not None:
                os.close(root_fd)

    def _goal_projection_candidate(
        self,
        goal,
        steps,
        *,
        store_identity: str,
    ) -> tuple[Path, Path, tuple, int, int, str, str]:
        goal_id = goal["id"]
        revision = goal["revision"]
        if (
            type(goal_id) is not int
            or goal_id < 1
            or type(revision) is not int
            or revision < 1
            or re.fullmatch(r"[0-9a-f]{32}", store_identity) is None
        ):
            raise ValueError("Goal projection identity or revision is invalid.")
        updated_match = re.match(
            r"^(\d{4}-\d{2}-\d{2})",
            str(goal["updated_at"] or ""),
        )
        if updated_match is None:
            raise ValueError("Goal projection update time is invalid.")
        steps = tuple(sorted(tuple(steps), key=lambda step: step["id"]))
        source_digest = goal_projection_source_digest(
            goal_id,
            revision,
            goal["title"],
            goal["purpose"],
            goal["horizon"],
            goal["status"],
            goal["created_at"],
            goal["updated_at"],
            steps,
        )
        if type(source_digest) is not str or re.fullmatch(
            r"[0-9a-f]{64}", source_digest
        ) is None:
            raise ValueError("Goal projection source digest is invalid.")
        content = _render_goal_projection(
            goal,
            steps,
            store_identity=store_identity,
            goal_id=goal_id,
            frontmatter_updated=updated_match.group(1),
            goal_revision=revision,
            source_digest=source_digest,
        )
        if len(content) > MAX_GOAL_PROJECTION_CHARS:
            raise ValueError("Goal projection is too large to publish safely.")
        if not _is_owned_goal_projection(
            content,
            goal_id=goal_id,
            store_identity=store_identity,
        ):
            raise ValueError("Generated goal projection ownership is invalid.")
        title = _goal_filename_component(goal["title"])
        path = (
            self.root_path
            / "Projects"
            / f"{title} [{goal_id}-{store_identity}].md"
        )
        lock_path = (
            self.root_path
            / "Projects"
            / f".goal-{goal_id}-{store_identity}.lock"
        )
        return (
            path,
            lock_path,
            steps,
            goal_id,
            revision,
            source_digest,
            content,
        )

    def _write_goal_projection(
        self,
        goal,
        steps,
        *,
        store_identity: str,
        expected_prior_content_digest: str | tuple[str, ...] | None = None,
    ) -> tuple[Path, bool, str, str]:
        (
            path,
            lock_path,
            steps,
            goal_id,
            revision,
            source_digest,
            content,
        ) = self._goal_projection_candidate(
            goal,
            steps,
            store_identity=store_identity,
        )
        content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if expected_prior_content_digest is None:
            expected_prior_content_digests: tuple[str, ...] = ()
        elif type(expected_prior_content_digest) is str:
            expected_prior_content_digests = (expected_prior_content_digest,)
        elif type(expected_prior_content_digest) is tuple:
            expected_prior_content_digests = expected_prior_content_digest
        else:
            raise ValueError("Previous goal projection evidence is invalid.")
        if (
            len(expected_prior_content_digests) > 257
            or any(
                type(item) is not str or re.fullmatch(r"[0-9a-f]{64}", item) is None
                for item in expected_prior_content_digests
            )
            or len(set(expected_prior_content_digests))
            != len(expected_prior_content_digests)
        ):
            raise ValueError("Previous goal projection evidence is invalid.")
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            _recover_atomic_exchange_journals(
                self.root_path,
                path,
                max_file_bytes=MAX_GOAL_PROJECTION_CHARS * 4,
            )
            existing_snapshot = _read_goal_projection_snapshot(self.root_path, path)
            existing_bytes = (
                None if existing_snapshot is None else existing_snapshot[0]
            )
            existing = None if existing_snapshot is None else existing_snapshot[1]
            if existing is not None:
                if _is_owned_goal_projection(
                    existing,
                    goal_id=goal_id,
                    store_identity=store_identity,
                ):
                    freshness = _goal_projection_freshness(existing)
                    if freshness is None:
                        raise FileExistsError(
                            "Goal projection freshness could not be verified."
                        )
                    existing_revision, existing_source_digest = freshness
                    if existing_revision > revision:
                        raise RuntimeError("Stale goal projection snapshot was refused.")
                    if existing_revision < revision:
                        assert existing_bytes is not None
                        existing_content_digest = hashlib.sha256(existing_bytes).hexdigest()
                        if not any(
                            secrets.compare_digest(existing_content_digest, candidate)
                            for candidate in expected_prior_content_digests
                        ):
                            raise FileExistsError(
                                "Previous goal projection content changed outside Jarvis."
                            )
                    if (
                        existing_revision == revision
                        and existing_source_digest != source_digest
                    ):
                        raise RuntimeError(
                            "Conflicting goal projection revision was refused."
                        )
                    if (
                        existing_revision == revision
                        and existing_source_digest == source_digest
                        and existing != content
                    ):
                        raise FileExistsError(
                            "Owned goal projection content changed outside Jarvis."
                        )
                elif _is_owned_legacy_goal_projection(
                    existing,
                    goal_id=goal_id,
                    store_identity=store_identity,
                ):
                    legacy_fields = _goal_projection_frontmatter_fields(
                        existing,
                        legacy=True,
                    )
                    assert legacy_fields is not None
                    expected_legacy = _render_goal_projection(
                        goal,
                        steps,
                        store_identity=store_identity,
                        goal_id=goal_id,
                        frontmatter_updated=legacy_fields["updated"],
                        goal_revision=None,
                        source_digest=None,
                    )
                    if existing != expected_legacy:
                        raise FileExistsError(
                            "Legacy goal projection content could not be verified."
                        )
                else:
                    raise FileExistsError(
                        "Goal projection destination ownership could not be verified."
                    )
            if existing_bytes == content.encode("utf-8"):
                return path, False, content_digest, source_digest
            try:
                if existing_bytes is None:
                    _replace_text(
                        self.root_path,
                        path,
                        content,
                        expected_content=None,
                        max_exchange_bytes=MAX_GOAL_PROJECTION_CHARS * 4,
                    )
                else:
                    _replace_text(
                        self.root_path,
                        path,
                        content,
                        expected_bytes=existing_bytes,
                        max_exchange_bytes=MAX_GOAL_PROJECTION_CHARS * 4,
                    )
            except RuntimeError as exc:
                if str(exc) != "Obsidian destination changed during atomic publication.":
                    raise
                raise FileExistsError(
                    "Goal projection ownership changed during publication."
                ) from exc
            readback = _read_goal_projection_snapshot(self.root_path, path)
            if readback is None or readback[0] != content.encode("utf-8"):
                raise OSError(
                    "Goal projection readback did not match the published snapshot."
                )
        return path, True, content_digest, source_digest

    def goal_projection_candidate_evidence(
        self,
        goal,
        steps,
        *,
        store_identity: str,
    ) -> tuple[str, str]:
        """Return deterministic content and source digests without writing a file."""
        (
            _path,
            _lock_path,
            _steps,
            _goal_id,
            _revision,
            source_digest,
            content,
        ) = self._goal_projection_candidate(
            goal,
            steps,
            store_identity=store_identity,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest(), source_digest

    def inspect_goal_projection_prior_evidence(
        self,
        goal,
        steps,
        *,
        store_identity: str,
    ) -> tuple[str, str]:
        """Classify the current projection without changing it."""
        (
            path,
            lock_path,
            steps,
            goal_id,
            _revision,
            _source_digest,
            _content,
        ) = self._goal_projection_candidate(
            goal,
            steps,
            store_identity=store_identity,
        )
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            _recover_atomic_exchange_journals(
                self.root_path,
                path,
                max_file_bytes=MAX_GOAL_PROJECTION_CHARS * 4,
            )
            existing_snapshot = _read_goal_projection_snapshot(self.root_path, path)
            if existing_snapshot is None:
                return "absent", ""
            existing_bytes, existing = existing_snapshot
            if _is_owned_goal_projection(
                existing,
                goal_id=goal_id,
                store_identity=store_identity,
            ):
                return "owned", hashlib.sha256(existing_bytes).hexdigest()
            if _is_owned_legacy_goal_projection(
                existing,
                goal_id=goal_id,
                store_identity=store_identity,
            ):
                legacy_fields = _goal_projection_frontmatter_fields(
                    existing,
                    legacy=True,
                )
                assert legacy_fields is not None
                expected_legacy = _render_goal_projection(
                    goal,
                    steps,
                    store_identity=store_identity,
                    goal_id=goal_id,
                    frontmatter_updated=legacy_fields["updated"],
                    goal_revision=None,
                    source_digest=None,
                )
                if existing == expected_legacy:
                    return "legacy", ""
            return "unverified", ""

    def write_goal_with_evidence(
        self,
        goal,
        steps,
        *,
        store_identity: str,
        expected_prior_content_digest: str | tuple[str, ...] | None = None,
    ) -> tuple[Path, str, str]:
        path, _changed, content_digest, source_digest = self._write_goal_projection(
            goal,
            steps,
            store_identity=store_identity,
            expected_prior_content_digest=expected_prior_content_digest,
        )
        return path, content_digest, source_digest

    def write_goal(self, goal, steps, *, store_identity: str) -> Path:
        path, _content_digest, _source_digest = self.write_goal_with_evidence(
            goal,
            steps,
            store_identity=store_identity,
        )
        return path

    def verify_goal_projection_evidence(
        self,
        *,
        goal_id: int,
        title: str,
        store_identity: str,
        expected_content_digest: str,
        expected_revision: int,
        expected_source_digest: str,
    ) -> bool:
        """Verify retained completion evidence for the current ID-bound goal note."""
        if type(goal_id) is not int or goal_id < 1:
            raise ValueError("Goal projection id is invalid.")
        if type(title) is not str:
            raise ValueError("Goal projection title is invalid.")
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Goal projection store identity is invalid.")
        if re.fullmatch(r"[0-9a-f]{64}", expected_content_digest) is None:
            raise ValueError("Goal projection content digest is invalid.")
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("Goal projection expected revision is invalid.")
        if re.fullmatch(r"[0-9a-f]{64}", expected_source_digest) is None:
            raise ValueError("Goal projection expected source digest is invalid.")
        title_component = _goal_filename_component(title)
        path = (
            self.root_path
            / "Projects"
            / f"{title_component} [{goal_id}-{store_identity}].md"
        )
        lock_path = (
            self.root_path
            / "Projects"
            / f".goal-{goal_id}-{store_identity}.lock"
        )
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            _recover_atomic_exchange_journals(
                self.root_path,
                path,
                max_file_bytes=MAX_GOAL_PROJECTION_CHARS * 4,
            )
            snapshot = _read_goal_projection_snapshot(self.root_path, path)
            if snapshot is None:
                return False
            content_bytes, content = snapshot
            if not _is_owned_goal_projection(
                content,
                goal_id=goal_id,
                store_identity=store_identity,
            ):
                return False
            freshness = _goal_projection_freshness(content)
            return bool(
                freshness == (expected_revision, expected_source_digest)
                and hashlib.sha256(content_bytes).hexdigest()
                == expected_content_digest
            )

    def write_organized_goal(
        self,
        goal,
        steps,
        *,
        store_identity: str,
        expected_prior_content_digest: str | tuple[str, ...] | None = None,
    ) -> tuple[Path, bool, str]:
        path, changed, content_digest, _source_digest = self._write_goal_projection(
            goal,
            steps,
            store_identity=store_identity,
            expected_prior_content_digest=expected_prior_content_digest,
        )
        return path, changed, content_digest

    def write_tasks(self, tasks, *, store_identity: str) -> Path:
        path, _tasks, _content_sha256, _source_revision = self.write_tasks_with_evidence(
            tasks, store_identity=store_identity
        )
        return path

    def _write_tasks_with_evidence_unlocked(
        self, tasks, *, store_identity: str
    ) -> tuple[Path, list, str, str]:
        tasks = list(tasks)
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Open-task mirror store identity is invalid.")
        path = self.root_path / "Tasks" / "Open Tasks.md"
        lines = [
            "---",
            "jarvis_projection: open_tasks",
            f"store_identity: {store_identity}",
            "---",
            "",
            "# Open Tasks",
            "",
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        if tasks:
            for task in tasks:
                due = f" due {safe_text(task['due'])}" if task["due"] else ""
                priority = f" priority {safe_text(task['priority'])}" if task["priority"] != "normal" else ""
                lines.append(f"- [ ] #{task['id']} {safe_text(task['body'])}{due}{priority}")
        else:
            lines.append("- [ ] No open tasks.")
        content = "\n".join(lines) + "\n"
        existing = _read_text(self.root_path, path, max_chars=2048)
        if existing is not None:
            projection = _frontmatter_text_value(existing, "jarvis_projection")
            owner = _frontmatter_text_value(existing, "store_identity")
            if not (
                (projection == "open_tasks" and owner == store_identity)
                or _is_legacy_open_task_projection(existing)
            ):
                raise FileExistsError("Open-task mirror destination ownership could not be verified.")
        source_payload = {
            "store_identity": store_identity,
            "tasks": [{key: task[key] for key in task.keys()} for task in tasks],
        }
        source_revision = hashlib.sha256(
            json.dumps(source_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        _replace_text(self.root_path, path, content)
        if _read_text(self.root_path, path) != content:
            raise OSError("Open-task mirror readback did not match the published snapshot.")
        return path, tasks, content_sha256, source_revision

    def write_tasks_with_evidence(
        self, tasks, *, store_identity: str
    ) -> tuple[Path, list, str, str]:
        path = self.root_path / "Tasks" / "Open Tasks.md"
        lock_path = path.with_name(f".{path.name}.lock")
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            return self._write_tasks_with_evidence_unlocked(
                tasks, store_identity=store_identity
            )

    @contextmanager
    def open_task_projection_publication(self):
        path = self.root_path / "Tasks" / "Open Tasks.md"
        lock_path = path.with_name(f".{path.name}.lock")
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            yield

    def write_tasks_with_evidence_under_publication_lock(
        self, tasks, *, store_identity: str
    ) -> tuple[Path, list, str, str]:
        return self._write_tasks_with_evidence_unlocked(tasks, store_identity=store_identity)

    def sync_open_tasks(self, store: MemoryStore) -> Path:
        path, _tasks, _content_sha256, _source_revision = self.sync_open_tasks_with_evidence(store)
        return path

    def sync_open_tasks_with_evidence(
        self, store: MemoryStore
    ) -> tuple[Path, list, str, str]:
        path = self.root_path / "Tasks" / "Open Tasks.md"
        lock_path = path.with_name(f".{path.name}.lock")
        store_identity = store.get_store_identity()
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            with store.open_task_mirror_snapshot() as tasks:
                return self._write_tasks_with_evidence_unlocked(
                    tasks, store_identity=store_identity
                )

    def write_pending_approvals(self, approvals) -> Path:
        def risk_review(tool_name: str, user_input: str) -> tuple[str, str, str]:
            if tool_name == "run_shell_command":
                return (
                    "shell/code execution",
                    "It can inspect or modify the machine depending on the command.",
                    "Read the command, prefer a narrow command, and avoid shell control operators.",
                )
            if tool_name == "write_text_file":
                return "file write", "It can create or overwrite local files.", "Confirm the exact path and content before approving."
            if tool_name == "clear_obsidian_inbox":
                return (
                    "Obsidian content reset",
                    "It clears reviewed inbox text from Jarvis-owned notes.",
                    "Make sure the inbox was ingested or manually reviewed first.",
                )
            if tool_name == "get_clipboard":
                return (
                    "personal data read",
                    "Clipboard contents may contain passwords, tokens, private messages, or copied documents.",
                    "Paste only the needed text into chat instead when possible.",
                )
            if tool_name == "enable_computer_control":
                return (
                    "computer control",
                    "It allows later mouse and keyboard actions after explicit tool requests.",
                    "Use `computer control status` and an `autonomy plan: ...` first.",
                )
            if tool_name == "observe_act_verify":
                return (
                    "computer control",
                    "It observes the screen, performs one primitive action, then observes again for verification.",
                    "Use `computer readiness: ...` and inspect exact coordinates/text, expectation, and stop conditions first.",
                )
            if tool_name == "click":
                return (
                    "computer control",
                    "It can click the wrong app, button, account, or destructive control.",
                    "Approve only one exact coordinate/button after a fresh screen observation and expected result are clear.",
                )
            if tool_name == "type_text":
                return (
                    "computer control",
                    "It can type private or incorrect text into the wrong field or app.",
                    "Approve only exact text and target field after a fresh screen observation confirms focus.",
                )
            if tool_name == "move_mouse":
                return (
                    "computer control",
                    "It moves the pointer and can set up a later click in the wrong place.",
                    "Approve only exact coordinates after the target screen layout is known.",
                )
            if tool_name == "observe_screen":
                return (
                    "screen observation",
                    "Screenshots may expose private apps, files, accounts, or messages.",
                    "Close private windows or describe the target manually if possible.",
                )
            if tool_name == "verify_screen":
                return (
                    "screen observation",
                    "Screenshots may expose private apps, files, accounts, or messages.",
                    "Close private windows, confirm the expectation, and approve only the needed observation.",
                )
            if tool_name == "screenshot":
                return (
                    "screen capture",
                    "Screenshots may expose private apps, files, accounts, or messages.",
                    "Crop or describe only the relevant area when possible.",
                )
            if tool_name == "create_reminder":
                return (
                    "external/local side effect",
                    "It creates a real macOS Reminder outside Jarvis memory.",
                    "Use a Jarvis task first if a system reminder is not necessary.",
                )
            if tool_name == "apple_reminders":
                return (
                    "personal Apple Reminders read",
                    "It reads private reminder content from the shared macOS account.",
                    "Open Reminders yourself and keep it open before approving; Jarvis will not launch the app. "
                    "Narrow the read to one exact list and bounded limit when possible. If an approved attempt "
                    "fails, that one-shot approval is consumed; fix the app or permission state and submit a "
                    "fresh bounded request instead of replaying it.",
                )
            low = f"{tool_name} {user_input}".lower()
            if any(word in low for word in ("delete", "clear", "remove", "overwrite")):
                return (
                    "destructive change",
                    "It may remove or replace local state.",
                    "Confirm there is a backup or use a read-only inspection command first.",
                )
            if any(word in low for word in ("click", "type", "mouse", "keyboard")):
                return (
                    "computer control",
                    "It can change the visible app or type into the wrong place.",
                    "Observe the screen and approve one small action at a time.",
                )
            return (
                "approval-gated action",
                "It exceeds Jarvis' auto-run safety ceiling.",
                "Approve only if the request, target, and expected result are clear.",
            )

        path = self.root_path / "Automations" / "Pending Approvals.md"
        lines = [
            "# Pending Approvals",
            "",
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        if approvals:
            for approval in approvals:
                try:
                    planned_args = json.loads(approval["planned_args"] or "{}") if "planned_args" in approval.keys() else {}
                except json.JSONDecodeError:
                    planned_args = {}
                if not isinstance(planned_args, dict):
                    planned_args = {}
                category, concern, safer_check = risk_review(safe_text(approval["tool_name"]), safe_text(approval["user_input"]))
                planned_arg_keys = ", ".join(safe_text(key) for key in sorted(planned_args)) if planned_args else "none"
                lines.extend(
                    [
                        f"## Approval #{approval['id']} - {safe_text(approval['tool_name'])}",
                        "",
                        f"- Status: {safe_text(approval['status'])}",
                        f"- Category: {category}",
                        f"- Why gated: {concern}",
                        f"- Created: {safe_text(approval['created_at'])}",
                        f"- Updated: {safe_text(approval['updated_at'])}",
                        f"- User input: {safe_text(approval['user_input'])}",
                        f"- Planned arg keys: {planned_arg_keys}",
                        f"- Safer check: {safer_check}",
                        "- One-shot rule: approving reruns only this exact stored request once; it does not grant ongoing permission.",
                        "",
                    ]
                )
                if planned_args:
                    lines.extend(["Planned arguments:", ""])
                    for key in sorted(planned_args):
                        display_value = (
                            "bound to reviewed memory version"
                            if str(key).endswith("_binding") or str(key) == "target_binding"
                            else planned_args[key]
                        )
                        lines.append(f"- {safe_text(key)}: {safe_text(display_value)}")
                    lines.append("")
                lines.extend(
                    [
                        "Commands:",
                        "",
                        f"- Review queue: `pending approvals`",
                        f"- Readiness: `approval readiness {approval['id']}`",
                        f"- Last look: `approval packet {approval['id']}`",
                        f"- Run if trusted: `approve approval {approval['id']}`",
                        f"- Skip: `dismiss approval {approval['id']}`",
                        "",
                        "Reason:",
                        "",
                        safe_text(approval["reason"]),
                        "",
                    ]
                )
        else:
            lines.append("- No pending approvals.")
        _replace_text(self.root_path, path, "\n".join(lines) + "\n")
        return path

    def write_mission_control(self, body: str) -> Path:
        path = self.root_path / "Automations" / "Mission Control.md"
        lines = [
            "# Mission Control",
            "",
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            body.strip(),
            "",
        ]
        _replace_text(self.root_path, path, "\n".join(lines))
        return path

    def write_decision(self, decision, outcomes=None) -> Path:
        outcomes = _decision_outcomes(decision, outcomes)
        path = self.root_path / "Decisions" / f"{decision['id']:04d} {safe_name(safe_text(decision['title']))}.md"
        content = _legacy_decision_content(
            decision,
            status=str(decision["status"]),
            updated_at=str(decision["updated_at"]),
            frontmatter_updated=datetime.now().strftime("%Y-%m-%d"),
            outcomes=outcomes,
        )
        _replace_text(self.root_path, path, content)
        return path

    def write_decision_with_evidence(
        self, decision, *, store_identity: str
    ) -> tuple[Path, str]:
        decision_id = decision["id"]
        revision = decision["revision"]
        if type(decision_id) is not int or decision_id < 1:
            raise ValueError("Decision projection id is invalid.")
        if type(revision) is not int or revision < 1:
            raise ValueError("Decision projection revision is invalid.")
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Decision projection store identity is invalid.")
        outcomes = _decision_outcomes(decision)
        source_digest = decision_projection_source_digest(
            decision_id,
            revision,
            str(decision["title"]),
            str(decision["rationale"]),
            str(decision["impact"]),
            str(decision["status"]),
            str(decision["created_at"]),
            str(decision["updated_at"]),
            outcomes,
        )
        path = self.root_path / "Decisions" / (
            f"{decision_id:04d} {safe_name(safe_text(decision['title']))}.md"
        )
        lines = [
            "---",
            "jarvis_projection: decision",
            f"store_identity: {store_identity}",
            f"id: {decision_id}",
            f"source_revision: {revision}",
            f"source_digest: {source_digest}",
            f"status: {safe_text(decision['status'])}",
            f"updated: {str(decision['updated_at'])[:10]}",
            "---",
            "",
            f"# {safe_text(decision['title'])}",
            "",
            "## Decision",
            "",
            safe_text(decision["title"]),
            "",
            "## Rationale",
            "",
            safe_text(decision["rationale"]) or "No rationale captured.",
            "",
            "## Impact",
            "",
            safe_text(decision["impact"]) or "No impact captured.",
            "",
        ]
        _append_decision_outcomes(lines, outcomes)
        lines.extend(
            [
                "## Metadata",
                "",
                f"- Created: {decision['created_at']}",
                f"- Updated: {decision['updated_at']}",
            ]
        )
        content = "\n".join(lines) + "\n"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        lock_path = self.root_path / "Decisions" / f".decision-{decision_id}.lock"
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            existing = _read_text(self.root_path, path, max_chars=1_000_001)
            if existing == content:
                return path, digest
            if existing is not None:
                owned = (
                    _unique_frontmatter_text_value(existing, "jarvis_projection") == "decision"
                    and _unique_frontmatter_text_value(existing, "store_identity") == store_identity
                    and _frontmatter_person_id(existing) == decision_id
                )
                legacy_status = _unique_frontmatter_text_value(existing, "status")
                legacy_frontmatter_updated = _unique_frontmatter_text_value(
                    existing, "updated"
                )
                legacy_updated_match = re.search(
                    r"(?m)^- Updated: (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)$",
                    existing,
                )
                legacy = False
                if (
                    _unique_frontmatter_text_value(existing, "jarvis_projection") is None
                    and _unique_frontmatter_text_value(existing, "store_identity") is None
                    and _frontmatter_person_id(existing) == decision_id
                    and legacy_status in {"active", "superseded", "retired"}
                    and legacy_frontmatter_updated is not None
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", legacy_frontmatter_updated)
                    is not None
                    and legacy_updated_match is not None
                ):
                    legacy_updated_at = legacy_updated_match.group(1)
                    legacy = existing in {
                        _legacy_decision_content(
                            decision,
                            status=legacy_status,
                            updated_at=legacy_updated_at,
                            frontmatter_updated=legacy_frontmatter_updated,
                            outcomes=None,
                        ),
                        _legacy_decision_content(
                            decision,
                            status=legacy_status,
                            updated_at=legacy_updated_at,
                            frontmatter_updated=legacy_frontmatter_updated,
                            outcomes=(),
                        ),
                    }
                if not owned and not legacy:
                    raise FileExistsError("Decision projection ownership could not be verified.")
                if owned:
                    prior_revision = _unique_frontmatter_text_value(existing, "source_revision")
                    prior_source = _unique_frontmatter_text_value(existing, "source_digest")
                    if prior_revision is None or re.fullmatch(r"\d+", prior_revision) is None:
                        raise FileExistsError("Decision projection freshness could not be verified.")
                    if int(prior_revision) > revision:
                        raise RuntimeError("Stale decision projection snapshot was refused.")
                    if int(prior_revision) == revision and prior_source != source_digest:
                        raise RuntimeError("Conflicting decision projection revision was refused.")
            _replace_text(self.root_path, path, content, expected_content=existing)
            if _read_text(self.root_path, path, max_chars=1_000_001) != content:
                raise OSError("Decision projection readback did not match.")
        return path, digest

    def verify_decision_projection_evidence(
        self,
        *,
        decision_id: int,
        decision_title: str,
        store_identity: str,
        expected_content_digest: str,
    ) -> bool:
        path = self.root_path / "Decisions" / (
            f"{decision_id:04d} {safe_name(safe_text(decision_title))}.md"
        )
        lock_path = self.root_path / "Decisions" / f".decision-{decision_id}.lock"
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            content = _read_text(self.root_path, path, max_chars=1_000_001)
            return bool(
                content is not None
                and len(content) <= 1_000_000
                and _unique_frontmatter_text_value(content, "jarvis_projection") == "decision"
                and _unique_frontmatter_text_value(content, "store_identity") == store_identity
                and _frontmatter_person_id(content) == decision_id
                and hashlib.sha256(content.encode("utf-8")).hexdigest()
                == expected_content_digest
            )

    def write_organized_decision(self, decision) -> tuple[Path, bool, str]:
        decision_id = int(decision["id"])
        if decision_id < 1:
            raise ValueError("Organized decision id is invalid.")
        updated_match = re.match(r"^(\d{4}-\d{2}-\d{2})", str(decision["updated_at"] or ""))
        if updated_match is None:
            raise ValueError("Organized decision update time is invalid.")
        path = self.root_path / "Decisions" / f"{decision_id:04d} {safe_name(safe_text(decision['title']))}.md"
        lines = [
            "---",
            f"id: {decision_id}",
            f"status: {safe_text(decision['status'])}",
            f"updated: {updated_match.group(1)}",
            "---",
            "",
            f"# {safe_text(decision['title'])}",
            "",
            "## Decision",
            "",
            safe_text(decision["title"]),
            "",
            "## Rationale",
            "",
            safe_text(decision["rationale"]) or "No rationale captured.",
            "",
            "## Impact",
            "",
            safe_text(decision["impact"]) or "No impact captured.",
            "",
            "## Metadata",
            "",
            f"- Created: {decision['created_at']}",
            f"- Updated: {decision['updated_at']}",
        ]
        content = "\n".join(lines) + "\n"
        content_bytes = content.encode("utf-8")
        content_sha256 = hashlib.sha256(content_bytes).hexdigest()
        lock_path = path.with_name(f".{path.name}.jarvis.lock")
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            existing_bytes = _read_bytes_bounded(
                self.root_path,
                path,
                max_bytes=MAX_ORGANIZED_PROJECTION_BYTES,
            )
            if existing_bytes == content_bytes:
                return path, False, content_sha256
            if existing_bytes is not None:
                raise FileExistsError("Organized decision destination ownership could not be verified.")
            _replace_text(self.root_path, path, content, expected_content=None)
            if _read_bytes_bounded(
                self.root_path,
                path,
                max_bytes=MAX_ORGANIZED_PROJECTION_BYTES,
            ) != content_bytes:
                raise OSError("Organized decision mirror readback did not match the published record.")
        return path, True, content_sha256

    def verify_organized_projection(self, path: Path, expected_sha256: str) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256 or "")) is None:
            raise ValueError("Organized projection digest is invalid.")
        content = _read_bytes_bounded(
            self.root_path,
            path,
            max_bytes=MAX_ORGANIZED_PROJECTION_BYTES,
        )
        if content is None or hashlib.sha256(content).hexdigest() != expected_sha256:
            raise RuntimeError("Organized projection changed before batch completion.")

    def write_person_with_evidence(
        self,
        person,
        interactions,
        *,
        store_identity: str,
        interaction_high_watermark: int | None = None,
    ) -> tuple[Path, str]:
        interactions = tuple(interactions)
        person_id = person["id"]
        if type(person_id) is not int or person_id < 1:
            raise ValueError("Person mirror id is invalid.")
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Person mirror store identity is invalid.")
        title = _bounded_filename_component(safe_text(person["name"]))
        source_updated_at = safe_text(person["updated_at"])
        if not source_updated_at:
            raise ValueError("Person mirror source timestamp is invalid.")
        source_revision = person["revision"]
        if type(source_revision) is not int or source_revision < 0:
            raise ValueError("Person mirror source revision is invalid.")
        rendered_interaction_high_watermark = 0
        for item in interactions:
            interaction_id = item["id"]
            if type(interaction_id) is not int or interaction_id < 1:
                raise ValueError("Person mirror interaction id is invalid.")
            rendered_interaction_high_watermark = max(
                rendered_interaction_high_watermark,
                interaction_id,
            )
        if interaction_high_watermark is None:
            interaction_high_watermark = rendered_interaction_high_watermark
        elif (
            type(interaction_high_watermark) is not int
            or interaction_high_watermark < rendered_interaction_high_watermark
        ):
            raise ValueError("Person mirror interaction high watermark is invalid.")
        lines = [
            "---",
            "jarvis_projection: person",
            f"store_identity: {store_identity}",
            f"id: {person_id}",
            f"source_revision: {source_revision}",
            f"source_updated_at: {source_updated_at}",
            f"interaction_high_watermark: {interaction_high_watermark}",
            f"relation: {safe_text(person['relation'])}",
            f"last_contact_at: {safe_text(person['last_contact_at']) if person['last_contact_at'] else ''}",
            f"updated: {datetime.now().strftime('%Y-%m-%d')}",
            "---",
            "",
            f"# {safe_text(person['name'])}",
            "",
            "## Relation",
            "",
            safe_text(person["relation"]) or "No relation captured.",
            "",
            "## Notes",
            "",
            safe_text(person["notes"]) or "No notes captured.",
            "",
            "## Interactions",
            "",
        ]
        if interactions:
            for item in interactions:
                lines.append(f"- {safe_text(item['happened_at'])}: {safe_text(item['summary'])}")
        else:
            lines.append("- No interactions captured.")
        lines.extend(["", "## Metadata", "", f"- Created: {person['created_at']}", f"- Updated: {person['updated_at']}"])
        content = "\n".join(lines) + "\n"
        lock_path = self.root_path / "People" / f".person-{person_id}.lock"
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            path = self.root_path / "People" / f"{title} [{person_id}].md"
            existing = _read_text(self.root_path, path)
            if existing == content:
                return path, hashlib.sha256(existing.encode("utf-8")).hexdigest()
            if existing is not None:
                owned = (
                    _unique_frontmatter_text_value(existing, "jarvis_projection") == "person"
                    and _unique_frontmatter_text_value(existing, "store_identity") == store_identity
                    and _frontmatter_person_id(existing) == person_id
                )
                if not owned:
                    raise FileExistsError("Person mirror destination ownership could not be verified.")
                existing_revision_text = _unique_frontmatter_text_value(existing, "source_revision") or ""
                existing_high_watermark_text = (
                    _unique_frontmatter_text_value(existing, "interaction_high_watermark") or ""
                )
                if (
                    re.fullmatch(r"\d+", existing_revision_text) is None
                    or re.fullmatch(r"\d+", existing_high_watermark_text) is None
                ):
                    raise FileExistsError("Person mirror freshness could not be verified.")
                existing_freshness = (int(existing_revision_text), int(existing_high_watermark_text))
                incoming_freshness = (source_revision, interaction_high_watermark)
                if existing_freshness > incoming_freshness:
                    raise RuntimeError("Stale person mirror snapshot was refused.")
            _replace_text(self.root_path, path, content, expected_content=existing)
            readback = _read_text(self.root_path, path)
            if readback != content:
                raise OSError("Person mirror readback did not match the published snapshot.")
            content_digest = hashlib.sha256(readback.encode("utf-8")).hexdigest()
        return path, content_digest

    def verify_person_projection_evidence(
        self,
        *,
        person_id: int,
        person_name: str,
        store_identity: str,
        expected_content_digest: str,
    ) -> bool:
        """Verify the current ID-bound person note against retained completion evidence."""
        if type(person_id) is not int or person_id < 1:
            raise ValueError("Person mirror id is invalid.")
        if type(person_name) is not str:
            raise ValueError("Person mirror name is invalid.")
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Person mirror store identity is invalid.")
        if re.fullmatch(r"[0-9a-f]{64}", expected_content_digest) is None:
            raise ValueError("Person mirror content digest is invalid.")
        title = _bounded_filename_component(safe_text(person_name))
        path = self.root_path / "People" / f"{title} [{person_id}].md"
        lock_path = self.root_path / "People" / f".person-{person_id}.lock"
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            content = _read_text(self.root_path, path, max_chars=1_000_001)
            if content is None or len(content) > 1_000_000:
                return False
            owned = (
                _unique_frontmatter_text_value(content, "jarvis_projection") == "person"
                and _unique_frontmatter_text_value(content, "store_identity") == store_identity
                and _frontmatter_person_id(content) == person_id
            )
            return owned and hashlib.sha256(content.encode("utf-8")).hexdigest() == expected_content_digest

    def delete_person_projection(
        self,
        *,
        person_id: int,
        store_identity: str,
        expected_content_digest: str | None = None,
    ) -> PersonProjectionDeleteResult:
        """Delete every exact store-owned note for a deleted person without retaining its name."""
        if type(person_id) is not int or person_id < 1:
            raise ValueError("Person mirror id is invalid.")
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Person mirror store identity is invalid.")
        if expected_content_digest is not None and re.fullmatch(
            r"[0-9a-f]{64}", expected_content_digest
        ) is None:
            raise ValueError("Person mirror delete digest is invalid.")

        people_root = self.root_path / "People"
        lock_path = people_root / f".person-{person_id}.lock"
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            try:
                candidates = sorted(people_root.glob("*.md"))[:10_001]
            except OSError:
                return PersonProjectionDeleteResult("error")
            if len(candidates) > 10_000:
                return PersonProjectionDeleteResult("error")
            owned: list[tuple[Path, str]] = []
            for path in candidates:
                try:
                    content = _read_text(self.root_path, path, max_chars=1_000_001)
                except (OSError, ValueError):
                    return PersonProjectionDeleteResult("error")
                if content is None or len(content) > 1_000_000:
                    continue
                if (
                    _unique_frontmatter_text_value(content, "jarvis_projection") == "person"
                    and _unique_frontmatter_text_value(content, "store_identity") == store_identity
                    and _frontmatter_person_id(content) == person_id
                ):
                    owned.append((path, content))
            if not owned:
                return PersonProjectionDeleteResult("absent")
            if expected_content_digest is not None and not any(
                hashlib.sha256(content.encode("utf-8")).hexdigest()
                == expected_content_digest
                for _path, content in owned
            ):
                return PersonProjectionDeleteResult("ownership_mismatch")
            for path, expected_content in owned:
                current = _read_text(self.root_path, path, max_chars=1_000_001)
                if current != expected_content:
                    return PersonProjectionDeleteResult("error")
            for path, _content in owned:
                if not _unlink(self.root_path, path):
                    return PersonProjectionDeleteResult("error")
        return PersonProjectionDeleteResult("deleted")

    def write_person(self, person, interactions, *, store_identity: str) -> Path:
        path, _content_digest = self.write_person_with_evidence(
            person,
            interactions,
            store_identity=store_identity,
        )
        return path

    def write_preferences(self, preferences) -> Path:
        path = self.root_path / "Memory Tree" / "Preferences.md"
        lines = [
            "# Preferences",
            "",
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        current_category = None
        if preferences:
            for row in preferences:
                if row["category"] != current_category:
                    current_category = row["category"]
                    lines.extend(["", f"## {safe_text(current_category).title()}", ""])
                lines.append(f"- #{row['id']} {safe_text(row['key'])}: {safe_text(row['value'])} [{safe_text(row['status'])}]")
        else:
            lines.append("- No preferences captured yet.")
        _replace_text(self.root_path, path, "\n".join(lines) + "\n")
        return path

    def write_preferences_with_evidence(
        self,
        preferences,
        *,
        store_identity: str,
        generation: int,
    ) -> tuple[Path, str]:
        rows = tuple(preferences)
        if re.fullmatch(r"[0-9a-f]{32}", store_identity) is None:
            raise ValueError("Preference projection store identity is invalid.")
        if type(generation) is not int or generation < 0:
            raise ValueError("Preference projection generation is invalid.")
        source_digest = preference_projection_source_digest(generation, rows)
        lines = [
            "---",
            "jarvis_projection: preferences",
            f"store_identity: {store_identity}",
            f"generation: {generation}",
            f"source_digest: {source_digest}",
            "---",
            "",
            "# Preferences",
            "",
        ]
        current_category = None
        for row in rows:
            if row["category"] != current_category:
                current_category = row["category"]
                lines.extend(["", f"## {safe_text(current_category).title()}", ""])
            lines.append(
                f"- #{row['id']} {safe_text(row['key'])}: {safe_text(row['value'])} "
                f"[{safe_text(row['status'])}]"
            )
        if not rows:
            lines.append("- No preferences captured yet.")
        content = "\n".join(lines) + "\n"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        path = self.root_path / "Memory Tree" / "Preferences.md"
        lock_path = path.with_name(".Preferences.jarvis.lock")
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            existing = _read_text(self.root_path, path, max_chars=1_000_001)
            if existing == content:
                return path, digest
            if existing is not None:
                owned = (
                    _unique_frontmatter_text_value(existing, "jarvis_projection") == "preferences"
                    and _unique_frontmatter_text_value(existing, "store_identity") == store_identity
                )
                legacy = False
                if (
                    _unique_frontmatter_text_value(existing, "jarvis_projection") is None
                    and _unique_frontmatter_text_value(existing, "store_identity") is None
                ):
                    if existing.strip() == "# Preferences":
                        legacy = True
                    else:
                        existing_lines = existing.splitlines()
                        if (
                            len(existing_lines) >= 4
                            and existing_lines[0] == "# Preferences"
                            and existing_lines[1] == ""
                            and re.fullmatch(
                                r"Updated: \d{4}-\d{2}-\d{2} \d{2}:\d{2}",
                                existing_lines[2],
                            )
                            is not None
                            and existing_lines[3] == ""
                        ):
                            legacy_render = [
                                "# Preferences",
                                "",
                                existing_lines[2],
                                "",
                            ]
                            legacy_category = None
                            for row in rows:
                                if row["category"] != legacy_category:
                                    legacy_category = row["category"]
                                    legacy_render.extend(
                                        ["", f"## {safe_text(legacy_category).title()}", ""]
                                    )
                                legacy_render.append(
                                    f"- #{row['id']} {safe_text(row['key'])}: "
                                    f"{safe_text(row['value'])} [{safe_text(row['status'])}]"
                                )
                            if not rows:
                                legacy_render.append("- No preferences captured yet.")
                            legacy = existing == "\n".join(legacy_render) + "\n"
                if not owned and not legacy:
                    raise FileExistsError("Preference projection ownership could not be verified.")
                if owned:
                    prior_generation = _unique_frontmatter_text_value(existing, "generation")
                    prior_source = _unique_frontmatter_text_value(existing, "source_digest")
                    if prior_generation is None or re.fullmatch(r"\d+", prior_generation) is None:
                        raise FileExistsError("Preference projection freshness could not be verified.")
                    if int(prior_generation) > generation:
                        raise RuntimeError("Stale preference projection snapshot was refused.")
                    if int(prior_generation) == generation and prior_source != source_digest:
                        raise RuntimeError("Conflicting preference projection generation was refused.")
            _replace_text(self.root_path, path, content, expected_content=existing)
            if _read_text(self.root_path, path, max_chars=1_000_001) != content:
                raise OSError("Preference projection readback did not match.")
        return path, digest

    def verify_preferences_projection_evidence(
        self,
        *,
        store_identity: str,
        generation: int,
        expected_content_digest: str,
    ) -> bool:
        path = self.root_path / "Memory Tree" / "Preferences.md"
        lock_path = path.with_name(".Preferences.jarvis.lock")
        with _exclusive_sidecar_lock(self.root_path, lock_path):
            content = _read_text(self.root_path, path, max_chars=1_000_001)
            return bool(
                content is not None
                and len(content) <= 1_000_000
                and _unique_frontmatter_text_value(content, "jarvis_projection") == "preferences"
                and _unique_frontmatter_text_value(content, "store_identity") == store_identity
                and _unique_frontmatter_text_value(content, "generation") == str(generation)
                and hashlib.sha256(content.encode("utf-8")).hexdigest()
                == expected_content_digest
            )
