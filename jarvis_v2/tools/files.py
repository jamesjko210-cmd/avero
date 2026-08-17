from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_resource_not_found_failure,
    declare_retryable_local_read_failure,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.tools.notes import is_protected_profile_path_or_content


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".css",
    ".html",
    ".csv",
}
MAX_LIST_LIMIT = 200
MAX_FIND_LIMIT = 200
MAX_READ_CHARS = 20000
MAX_READ_BYTES = 1_000_000
MAX_WRITE_CHARS = 100000
MAX_PATH_CHARS = 1000
MAX_PATTERN_CHARS = 160
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
FILE_WRITE_INPUT_RECOVERY_ACTION = (
    "Correct the reported file-write input, then submit a new write through the normal policy."
)
FILE_WRITE_RETRY_RECOVERY_ACTION = (
    "Run `setup check`, correct the reported storage issue, then submit a new file write."
)
FILE_WRITE_VERIFY_RECOVERY_ACTION = (
    "Inspect the target file before issuing any new write; do not retry automatically."
)
PROFILE_READ_RECOVERY_ACTION = "Use `read profile` instead."


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _raw_int_metadata(value: Any, *, key: str, sanitized: int) -> dict[str, Any]:
    if value is None:
        return {key: sanitized}
    if isinstance(value, bool):
        return {key: sanitized, f"raw_{key}": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {key: sanitized, f"raw_{key}": _short_metadata(value, limit=80)}
    return {key: sanitized}


def _short(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_metadata(value: Any, *, limit: int) -> str:
    text = _short(value, limit=limit)
    return LOCAL_PATH_RE.sub("<local-path>", text)


def _file_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
    }
    metadata.update(extra)
    return metadata


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _file_contract(*, state_changed: bool = False, changed: list[str] | None = None, content_in_handoff: bool = False) -> dict[str, Any]:
    return {
        "ready_for_operator": True,
        "state_changed": state_changed,
        "changed": changed or [],
        "content_in_handoff": content_in_handoff,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _file_handoff_metadata(handoff_key: str, handoff: dict[str, Any], **extra: Any) -> dict[str, Any]:
    metadata = _file_metadata(
        **extra,
        **_file_contract(
            state_changed=_metadata_bool(handoff.get("state_changed")),
            changed=list(handoff.get("changed") or []),
            content_in_handoff=_metadata_bool(handoff.get("content_in_handoff")),
        ),
    )
    metadata[handoff_key] = handoff
    metadata[f"{handoff_key}_ready"] = True
    return metadata


def _safe_path_label(path: Path | str | None, *, root: Path | None = None) -> str:
    if path is None:
        return ""
    try:
        candidate = Path(path)
    except TypeError:
        return _short_metadata(getattr(path, "name", str(path)), limit=120)
    if root is not None:
        try:
            return str(candidate.resolve().relative_to(root.resolve()))
        except (OSError, ValueError):
            pass
    return _short_metadata(candidate.name or str(candidate), limit=120)


def _file_boundaries(
    *,
    read: bool = False,
    write: bool = False,
    reads_private_data: bool = False,
) -> dict[str, bool]:
    return {
        "read_only": not write,
        "reads_private_data": reads_private_data,
        "reads_personal_data": False,
        "writes_files": write,
        "writes_memory": False,
        "writes_notes": False,
        "executes_side_effect": write,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "controls_computer": False,
        "external_side_effect": False,
        "read_file_contents": read,
        "write_file_contents": write,
    }


def _list_files_handoff(
    *,
    directory: Path,
    rows: list[dict[str, Any]],
    total_entries: int,
    limit: int,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "source": "list_files",
        **_file_contract(),
        "directory_display": _safe_path_label(directory),
        "count": len(rows),
        "total_entries": total_entries,
        "limit": limit,
        "truncated": truncated,
        "entries": rows,
        "next_commands": [
            "find files",
            "read text file",
            "list files",
        ],
        "boundaries": _file_boundaries(read=False, write=False, reads_private_data=True),
    }


def _read_text_file_handoff(
    *,
    path: Path,
    size_bytes: int,
    max_chars: int,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "source": "read_text_file",
        **_file_contract(),
        "path_display": _safe_path_label(path),
        "suffix": path.suffix.lower(),
        "size_bytes": size_bytes,
        "max_chars": max_chars,
        "truncated": truncated,
        "content_in_handoff": False,
        "next_commands": [
            "find files",
            "list files",
        ],
        "boundaries": _file_boundaries(read=True, write=False, reads_private_data=True),
    }


def _write_text_file_handoff(*, path: Path, chars: int, overwrite: bool) -> dict[str, Any]:
    return {
        "source": "write_text_file",
        **_file_contract(state_changed=True, changed=["file_write"]),
        "path_display": _safe_path_label(path),
        "suffix": path.suffix.lower(),
        "chars": chars,
        "overwrite": overwrite,
        "content_in_handoff": False,
        "next_commands": [
            "read text file",
            "list files",
            "find files",
        ],
        "boundaries": _file_boundaries(read=False, write=True, reads_private_data=False),
    }


def _find_files_handoff(
    *,
    root: Path,
    pattern: str,
    rows: list[dict[str, Any]],
    limit: int,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "source": "find_files",
        **_file_contract(),
        "root_display": _safe_path_label(root),
        "pattern": _short_metadata(pattern, limit=MAX_PATTERN_CHARS),
        "count": len(rows),
        "limit": limit,
        "truncated": truncated,
        "matches": rows,
        "next_commands": [
            "read text file",
            "list files",
            "find files",
        ],
        "boundaries": _file_boundaries(read=False, write=False, reads_private_data=True),
    }


def _file_refusal_handoff(
    *,
    source: str,
    mutation: str,
    reason: str,
    path: Path | str | None = None,
    root: Path | str | None = None,
    pattern: str | None = None,
    chars: int | None = None,
    size_bytes: int | None = None,
    max_bytes: int | None = None,
    exception_type: str | None = None,
    reads_private_data: bool = False,
) -> dict[str, Any]:
    next_commands = (
        ["read profile"]
        if reason == "protected_profile"
        else ["find files", "list files", "read text file"]
    )
    handoff: dict[str, Any] = {
        "source": source,
        "mutation": mutation,
        "reason": reason,
        "refused": True,
        **_file_contract(),
        "next_commands": next_commands,
        "boundaries": _file_boundaries(read=False, write=False, reads_private_data=reads_private_data),
    }
    if path is not None:
        handoff["path_display"] = _safe_path_label(path)
    if root is not None:
        handoff["root_display"] = _safe_path_label(root)
    if pattern is not None:
        handoff["pattern"] = _short_metadata(pattern, limit=MAX_PATTERN_CHARS)
    if chars is not None:
        handoff["chars"] = chars
        handoff["content_in_handoff"] = False
    if size_bytes is not None:
        handoff["size_bytes"] = size_bytes
    if max_bytes is not None:
        handoff["max_bytes"] = max_bytes
    if exception_type:
        handoff["exception_type"] = exception_type
    return handoff


def _file_refusal_metadata(
    *,
    source: str,
    mutation: str,
    reason: str,
    path: Path | str | None = None,
    root: Path | str | None = None,
    pattern: str | None = None,
    chars: int | None = None,
    size_bytes: int | None = None,
    max_bytes: int | None = None,
    exception_type: str | None = None,
    reads_private_data: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    metadata = _file_metadata(
        file_refusal_handoff_ready=True,
        refusal_reason=reason,
        reads_private_data=reads_private_data,
        **_file_contract(),
        **extra,
    )
    if path is not None:
        metadata["path_display"] = _safe_path_label(path)
    if root is not None:
        metadata["root_display"] = _safe_path_label(root)
    if pattern is not None:
        metadata["pattern"] = _short_metadata(pattern, limit=MAX_PATTERN_CHARS)
    if chars is not None:
        metadata["chars"] = chars
        metadata["content_in_handoff"] = False
    if size_bytes is not None:
        metadata["size_bytes"] = size_bytes
    if max_bytes is not None:
        metadata["max_bytes"] = max_bytes
    if exception_type:
        metadata["exception_type"] = exception_type
    metadata["file_refusal_handoff"] = _file_refusal_handoff(
        source=source,
        mutation=mutation,
        reason=reason,
        path=path,
        root=root,
        pattern=pattern,
        chars=chars,
        size_bytes=size_bytes,
        max_bytes=max_bytes,
        exception_type=exception_type,
        reads_private_data=reads_private_data,
    )
    return metadata


def _file_error(action: str, exc: Exception) -> str:
    permission = "write" if action == "write" else "read"
    return (
        f"Could not {action} file. Confirm the path exists and grant the current user {permission} "
        "permission in Finder > Get Info > Sharing & Permissions, then retry. Run `setup check` if "
        f"this is a Jarvis storage path. ({type(exc).__name__})"
    )


def _retryable_file_refusal(
    *,
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
    action: str,
    commands: tuple[str, ...] = (),
) -> ToolResult:
    return ToolResult(
        tool_name,
        False,
        output,
        declare_retryable_local_read_failure(
            metadata,
            output=output,
            action=action,
            commands=commands,
        ),
    )


def _known_unpublished_write_failure(
    metadata: dict[str, Any], *, output: str
) -> dict[str, Any]:
    result = dict(metadata)
    result.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            # Atomic publication did not change the target, but parent-directory
            # creation or temporary-file cleanup may already have occurred.
            "side_effect_possible": True,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        result,
        output=output,
        action=FILE_WRITE_RETRY_RECOVERY_ACTION,
        commands=("setup check",),
    )


def _published_durability_failure(
    metadata: dict[str, Any], *, output: str
) -> dict[str, Any]:
    result = dict(metadata)
    result.update(
        {
            # Namespace publication is known to have completed. Only the
            # directory durability acknowledgement is uncertain.
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": True,
            "retry_safe": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        result,
        output=output,
        action=FILE_WRITE_VERIFY_RECOVERY_ACTION,
    )


class _TextFilePublicationError(Exception):
    def __init__(self, cause: Exception, *, committed: bool) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.committed = committed


def _durable_atomic_write_text(path: Path, content: str, *, overwrite: bool) -> None:
    """Publish complete text in one same-directory atomic namespace change."""
    fd = -1
    temp_path: Path | None = None
    committed = False
    failure: Exception | None = None
    try:
        fd, raw_temp_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp_path = Path(raw_temp_path)
        if overwrite:
            try:
                os.fchmod(fd, stat.S_IMODE(path.stat().st_mode))
            except FileNotFoundError:
                pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        if overwrite:
            os.replace(temp_path, path)
            committed = True
            temp_path = None
        else:
            os.link(temp_path, path)
            committed = True
            temp_path.unlink()
            temp_path = None

        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, UnicodeError) as exc:
        failure = exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError as exc:
                if failure is None:
                    failure = exc
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                if failure is None:
                    failure = exc
    if failure is not None:
        raise _TextFilePublicationError(failure, committed=committed) from failure


def expand_path(value: Any) -> Path:
    return Path(_short(value, limit=MAX_PATH_CHARS)).expanduser().resolve()


def format_size(size: int) -> str:
    amount = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if amount < 1024:
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} PB"


def _missing_text_file_result(path: Path, *, max_chars: int, raw_max_chars: Any = None) -> ToolResult:
    path_label = _safe_path_label(path)
    recovery_commands = ["list files in .", f"find files {path_label}", f"read {path_label}"]
    output = (
        f"File not found: {path_label}. Run `list files in .` or `find files {path_label}` to "
        f"locate the correct path, then retry `read {path_label}` through the normal approval flow. "
        f"{RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
    )
    metadata = declare_resource_not_found_failure(
        _file_refusal_metadata(
            source="read_text_file",
            mutation="file_read",
            reason="file_not_found",
            path=path,
            reads_private_data=True,
            next_command=recovery_commands[0],
            recovery_commands=recovery_commands,
            retry_requires_path_refresh=True,
            retry_requires_fresh_approval=True,
            recovery_commands_require_normal_policy=True,
            authorizes_retry=False,
            **_raw_int_metadata(raw_max_chars, key="max_chars", sanitized=max_chars),
        ),
        output=output,
        action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    )
    return ToolResult(
        "read_text_file",
        False,
        output,
        metadata,
    )


def list_files(args: dict[str, Any]) -> ToolResult:
    directory = expand_path(args.get("directory") or ".")
    limit = _bounded_int(args.get("limit"), 80, 1, MAX_LIST_LIMIT)
    if not directory.exists():
        recovery_commands = ["list files in ."]
        output = (
            f"Directory not found: {_safe_path_label(directory)}. Run `list files in .` to locate the "
            "correct directory, then retry with the corrected path through the normal approval flow. "
            f"{RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
        )
        return ToolResult(
            "list_files",
            False,
            output,
            declare_resource_not_found_failure(
                _file_refusal_metadata(
                    source="list_files",
                    mutation="directory_list",
                    reason="directory_not_found",
                    path=directory,
                    reads_private_data=True,
                    next_command=recovery_commands[0],
                    recovery_commands=recovery_commands,
                    retry_requires_path_refresh=True,
                    retry_requires_fresh_approval=True,
                    recovery_commands_require_normal_policy=True,
                    authorizes_retry=False,
                    **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                ),
                output=output,
                action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
            ),
        )
    if not directory.is_dir():
        output = f"Not a directory: {_safe_path_label(directory)}. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        return _retryable_file_refusal(
            tool_name="list_files",
            output=output,
            metadata=_file_refusal_metadata(
                source="list_files",
                mutation="directory_list",
                reason="not_directory",
                path=directory,
                reads_private_data=True,
                **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
            ),
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        )

    entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = []
    rows = []
    for path in entries[:limit]:
        if path.is_dir():
            lines.append(f"[dir]  {path.name}/")
            rows.append({"name": _short_metadata(path.name, limit=120), "kind": "directory"})
        else:
            try:
                size_bytes = path.stat().st_size
                size = format_size(size_bytes)
            except OSError:
                size_bytes = None
                size = "unknown"
            lines.append(f"[file] {path.name} ({size})")
            rows.append({"name": _short_metadata(path.name, limit=120), "kind": "file", "size_bytes": size_bytes})
    if len(entries) > limit:
        lines.append(f"... {len(entries) - limit} more")
    handoff = _list_files_handoff(
        directory=directory,
        rows=rows,
        total_entries=len(entries),
        limit=limit,
        truncated=len(entries) > limit,
    )
    return ToolResult(
        "list_files",
        True,
        f"Contents of {directory}:\n" + "\n".join(lines),
        _file_handoff_metadata(
            "list_files_handoff",
            handoff,
            directory=str(directory),
            count=min(len(entries), limit),
            total_entries=len(entries),
            truncated=len(entries) > limit,
            reads_private_data=True,
            **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
        ),
    )


def read_text_file(args: dict[str, Any]) -> ToolResult:
    raw_path = _short(args.get("path"), limit=MAX_PATH_CHARS)
    max_chars = _bounded_int(args.get("max_chars"), 5000, 1, MAX_READ_CHARS)
    if not raw_path:
        output = f"No file path provided. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        return _retryable_file_refusal(
            tool_name="read_text_file",
            output=output,
            metadata=_file_refusal_metadata(
                source="read_text_file",
                mutation="file_read",
                reason="missing_path",
                reads_private_data=True,
                **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
            ),
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        )
    path = expand_path(raw_path)
    if is_protected_profile_path_or_content(path):
        output = f"That content is reserved for profile access. {PROFILE_READ_RECOVERY_ACTION}"
        return _retryable_file_refusal(
            tool_name="read_text_file",
            output=output,
            metadata=_file_refusal_metadata(
                source="read_text_file",
                mutation="file_read",
                reason="protected_profile",
                reads_private_data=True,
                next_command="read profile",
                recovery_commands=["read profile"],
                **_raw_int_metadata(
                    args.get("max_chars"),
                    key="max_chars",
                    sanitized=max_chars,
                ),
            ),
            action=PROFILE_READ_RECOVERY_ACTION,
            commands=("read profile",),
        )
    if not path.exists():
        return _missing_text_file_result(
            path,
            max_chars=max_chars,
            raw_max_chars=args.get("max_chars"),
        )
    if not path.is_file():
        output = f"Not a file: {_safe_path_label(path)}. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        return _retryable_file_refusal(
            tool_name="read_text_file",
            output=output,
            metadata=_file_refusal_metadata(
                source="read_text_file",
                mutation="file_read",
                reason="not_file",
                path=path,
                reads_private_data=True,
                **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
            ),
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        )
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        output = (
            f"Refusing to read non-text file type: {path.suffix or '(none)'}. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return _retryable_file_refusal(
            tool_name="read_text_file",
            output=output,
            metadata=_file_refusal_metadata(
                source="read_text_file",
                mutation="file_read",
                reason="non_text_file",
                path=path,
                reads_private_data=True,
                **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
            ),
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        )
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        failure_output = (
            f"{_file_error('inspect', exc)} "
            f"{LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION}"
        )
        return ToolResult(
            "read_text_file",
            False,
            failure_output,
            declare_retryable_personal_read_failure(
                _file_refusal_metadata(
                    source="read_text_file",
                    mutation="file_read",
                    reason="inspect_failed",
                    path=path,
                    reads_private_data=True,
                    exception_type=type(exc).__name__,
                    **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
                ),
                output=failure_output,
                action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                commands=("setup check",),
            ),
        )
    if size_bytes > MAX_READ_BYTES:
        output = (
            "Refusing to read large file without a narrower excerpt: "
            f"{format_size(size_bytes)} exceeds {format_size(MAX_READ_BYTES)}. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return _retryable_file_refusal(
            tool_name="read_text_file",
            output=output,
            metadata=_file_refusal_metadata(
                source="read_text_file",
                mutation="file_read",
                reason="file_too_large",
                path=path,
                size_bytes=size_bytes,
                max_bytes=MAX_READ_BYTES,
                max_read_bytes=MAX_READ_BYTES,
                reads_private_data=True,
            ),
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        )
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        failure_output = (
            f"{_file_error('read', exc)} "
            f"{LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION}"
        )
        return ToolResult(
            "read_text_file",
            False,
            failure_output,
            declare_retryable_personal_read_failure(
                _file_refusal_metadata(
                    source="read_text_file",
                    mutation="file_read",
                    reason="read_failed",
                    path=path,
                    reads_private_data=True,
                    exception_type=type(exc).__name__,
                    **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
                ),
                output=failure_output,
                action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                commands=("setup check",),
            ),
        )

    if is_protected_profile_path_or_content(path, content):
        output = f"That content is reserved for profile access. {PROFILE_READ_RECOVERY_ACTION}"
        return _retryable_file_refusal(
            tool_name="read_text_file",
            output=output,
            metadata=_file_refusal_metadata(
                source="read_text_file",
                mutation="file_read",
                reason="protected_profile",
                reads_private_data=True,
                next_command="read profile",
                recovery_commands=["read profile"],
                **_raw_int_metadata(
                    args.get("max_chars"),
                    key="max_chars",
                    sanitized=max_chars,
                ),
            ),
            action=PROFILE_READ_RECOVERY_ACTION,
            commands=("read profile",),
        )

    truncated = len(content) > max_chars
    body = content[:max_chars]
    if truncated:
        body += f"\n\n... truncated {len(content) - max_chars} chars"
    handoff = _read_text_file_handoff(path=path, size_bytes=size_bytes, max_chars=max_chars, truncated=truncated)
    return ToolResult(
        "read_text_file",
        True,
        body,
        _file_handoff_metadata(
            "read_text_file_handoff",
            handoff,
            path=str(path),
            truncated=truncated,
            size_bytes=size_bytes,
            reads_private_data=True,
            **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
        ),
    )


def write_text_file(args: dict[str, Any]) -> ToolResult:
    raw_path = _short(args.get("path"), limit=MAX_PATH_CHARS)
    content = str(args.get("content") or "")
    overwrite = bool(args.get("overwrite", False))
    if not raw_path:
        output = f"No file path provided. {FILE_WRITE_INPUT_RECOVERY_ACTION}"
        return _retryable_file_refusal(
            tool_name="write_text_file",
            output=output,
            metadata=_file_refusal_metadata(source="write_text_file", mutation="file_write", reason="missing_path", chars=len(content)),
            action=FILE_WRITE_INPUT_RECOVERY_ACTION,
        )
    path = expand_path(raw_path)
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        output = (
            f"Refusing to write non-text file type: {path.suffix or '(none)'}. "
            f"{FILE_WRITE_INPUT_RECOVERY_ACTION}"
        )
        return _retryable_file_refusal(
            tool_name="write_text_file",
            output=output,
            metadata=_file_refusal_metadata(source="write_text_file", mutation="file_write", reason="non_text_file", path=path, chars=len(content)),
            action=FILE_WRITE_INPUT_RECOVERY_ACTION,
        )
    if len(content) > MAX_WRITE_CHARS:
        output = (
            f"Refusing to write oversized text file: {len(content)} chars exceeds {MAX_WRITE_CHARS}. "
            f"{FILE_WRITE_INPUT_RECOVERY_ACTION}"
        )
        return _retryable_file_refusal(
            tool_name="write_text_file",
            output=output,
            metadata=_file_refusal_metadata(
                source="write_text_file",
                mutation="file_write",
                reason="content_too_large",
                path=path,
                chars=len(content),
                max_chars=MAX_WRITE_CHARS,
            ),
            action=FILE_WRITE_INPUT_RECOVERY_ACTION,
        )
    if path.exists() and not overwrite:
        output = (
            f"File already exists: {_safe_path_label(path)}. Set overwrite=true to replace it. "
            f"{FILE_WRITE_INPUT_RECOVERY_ACTION}"
        )
        return _retryable_file_refusal(
            tool_name="write_text_file",
            output=output,
            metadata=_file_refusal_metadata(source="write_text_file", mutation="file_write", reason="already_exists", path=path, chars=len(content), exists=True),
            action=FILE_WRITE_INPUT_RECOVERY_ACTION,
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _durable_atomic_write_text(path, content, overwrite=overwrite)
    except _TextFilePublicationError as publication_error:
        exc = publication_error.cause
        if isinstance(exc, FileExistsError) and not overwrite:
            output = (
                f"File already exists: {_safe_path_label(path)}. Set overwrite=true to replace it. "
                f"{FILE_WRITE_INPUT_RECOVERY_ACTION}"
            )
            return _retryable_file_refusal(
                tool_name="write_text_file",
                output=output,
                metadata=_file_refusal_metadata(
                    source="write_text_file",
                    mutation="file_write",
                    reason="already_exists",
                    path=path,
                    chars=len(content),
                    exists=True,
                ),
                action=FILE_WRITE_INPUT_RECOVERY_ACTION,
            )
        if publication_error.committed:
            handoff = _write_text_file_handoff(path=path, chars=len(content), overwrite=overwrite)
            handoff.update(
                durability_confirmed=False,
                commit_outcome="published_durability_uncertain",
                exception_type=type(exc).__name__,
            )
            output = (
                "The complete file was published, but its durable directory sync failed. "
                f"Verify {_safe_path_label(path)} before retrying; it may already contain the new content. "
                f"({type(exc).__name__}) {FILE_WRITE_VERIFY_RECOVERY_ACTION}"
            )
            metadata = _file_handoff_metadata(
                "write_text_file_handoff",
                handoff,
                path_display=_safe_path_label(path),
                chars=len(content),
                writes_files=True,
                executes_side_effect=True,
                durability_confirmed=False,
                commit_outcome="published_durability_uncertain",
                exception_type=type(exc).__name__,
            )
            return ToolResult(
                "write_text_file",
                False,
                output,
                _published_durability_failure(metadata, output=output),
            )
        output = f"{_file_error('write', exc)} {FILE_WRITE_RETRY_RECOVERY_ACTION}"
        return ToolResult(
            "write_text_file",
            False,
            output,
            _known_unpublished_write_failure(_file_refusal_metadata(
                source="write_text_file",
                mutation="file_write",
                reason="write_failed",
                path=path,
                chars=len(content),
                exception_type=type(exc).__name__,
            ), output=output),
        )
    except OSError as exc:
        output = f"{_file_error('write', exc)} {FILE_WRITE_RETRY_RECOVERY_ACTION}"
        return ToolResult(
            "write_text_file",
            False,
            output,
            _known_unpublished_write_failure(_file_refusal_metadata(
                source="write_text_file",
                mutation="file_write",
                reason="write_failed",
                path=path,
                chars=len(content),
                exception_type=type(exc).__name__,
            ), output=output),
        )
    handoff = _write_text_file_handoff(path=path, chars=len(content), overwrite=overwrite)
    return ToolResult(
        "write_text_file",
        True,
        f"Wrote {len(content)} chars to {path}",
        _file_handoff_metadata(
            "write_text_file_handoff",
            handoff,
            path=str(path),
            chars=len(content),
            writes_files=True,
            executes_side_effect=True,
        ),
    )


def find_files(args: dict[str, Any]) -> ToolResult:
    root = expand_path(args.get("root") or ".")
    pattern = _short(args.get("pattern"), limit=MAX_PATTERN_CHARS).lower()
    limit = _bounded_int(args.get("limit"), 50, 1, MAX_FIND_LIMIT)
    if not root.exists() or not root.is_dir():
        output = (
            f"Search root is not a directory: {_safe_path_label(root)}. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return _retryable_file_refusal(
            tool_name="find_files",
            output=output,
            metadata=_file_refusal_metadata(
                source="find_files",
                mutation="file_find",
                reason="root_not_directory",
                root=root,
                pattern=pattern,
                reads_private_data=True,
                **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
            ),
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        )
    if not pattern:
        output = f"Search pattern is empty. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        return _retryable_file_refusal(
            tool_name="find_files",
            output=output,
            metadata=_file_refusal_metadata(
                source="find_files",
                mutation="file_find",
                reason="missing_pattern",
                root=root,
                pattern=pattern,
                raw_pattern=_short_metadata(args.get("pattern"), limit=80),
                reads_private_data=True,
                **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
            ),
            action=LOCAL_READ_INPUT_RECOVERY_ACTION,
        )

    matches = []
    skipped_dirs = {".git", "__pycache__", "node_modules", "Library"}
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skipped_dirs and not d.startswith(".")]
        for name in files:
            if pattern in name.lower():
                matches.append(str(Path(current_root) / name))
                if len(matches) >= limit:
                    break
        if len(matches) >= limit:
            break
    if not matches:
        handoff = _find_files_handoff(root=root, pattern=pattern, rows=[], limit=limit, truncated=False)
        return ToolResult(
            "find_files",
            True,
            f"No files found matching '{pattern}' under {root}.",
            _file_handoff_metadata(
                "find_files_handoff",
                handoff,
                count=0,
                root=str(root),
                pattern=pattern,
                truncated=False,
                reads_private_data=True,
                **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
            ),
        )
    rows = [{"path_display": _safe_path_label(match, root=root), "name": _safe_path_label(match)} for match in matches]
    handoff = _find_files_handoff(root=root, pattern=pattern, rows=rows, limit=limit, truncated=len(matches) >= limit)
    return ToolResult(
        "find_files",
        True,
        "\n".join(matches),
        _file_handoff_metadata(
            "find_files_handoff",
            handoff,
            count=len(matches),
            root=str(root),
            pattern=pattern,
            truncated=len(matches) >= limit,
            reads_private_data=True,
            **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
        ),
    )
