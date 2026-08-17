from __future__ import annotations

import re
import stat
import unicodedata
from pathlib import Path
from typing import Any, Callable

from jarvis_v2.agent.failure_guidance import (
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    declare_outcome_unknown_failure,
    declare_resource_not_found_failure,
    declare_retryable_local_read_failure,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.obsidian import ObsidianVault


MAX_NOTE_LIMIT = 200
MAX_NOTE_READ_CHARS = 20000
MAX_NOTE_CUSTODY_CHARS = 1_000_000
MAX_NOTE_WRITE_CHARS = 50000
MAX_NOTE_PATH_CHARS = 240
MAX_NOTE_QUERY_CHARS = 500
MAX_NOTE_FOLDER_CHARS = 500
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
PROFILE_NOTE_PATH = "profile.md"
PROFILE_NOTE_MARKER_RE = re.compile(
    r"jarvis-profile-note",
    re.IGNORECASE,
)
MANAGED_NOTE_PROJECTIONS = {
    "memory tree/current context.md": ("Memory Tree/Current Context.md", "export state"),
    "memory tree/memory tree snapshot.md": (
        "Memory Tree/Memory Tree Snapshot.md",
        "memory tree summary",
    ),
    "automations/approval review.md": (
        "Automations/Approval Review.md",
        "save approval review",
    ),
    "automations/feedback report.md": (
        "Automations/Feedback Report.md",
        "save feedback report",
    ),
    "automations/feedback actions.md": (
        "Automations/Feedback Actions.md",
        "save feedback actions",
    ),
    "automations/learning review.md": (
        "Automations/Learning Review.md",
        "save learning review",
    ),
    "tasks/open tasks.md": ("Tasks/Open Tasks.md", "export tasks"),
}
PROFILE_READ_RECOVERY_ACTION = "Use `read profile` instead."
NOTE_TOO_LARGE_RECOVERY_ACTION = (
    "Choose a smaller note, then retry through the normal policy."
)


def _bounded_int(value: Any, default: int, low: int = 1, high: int = MAX_NOTE_LIMIT) -> int:
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
        return {key: sanitized, f"raw_{key}": _short_metadata(value, 80)}
    return {key: sanitized}


def _short(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_metadata(value: Any, limit: int) -> str:
    text = _short(value, limit)
    return LOCAL_PATH_RE.sub("<local-path>", text)


def _path_metadata(value: Any) -> str:
    return _short_metadata(value, MAX_NOTE_PATH_CHARS)


def _frontmatter_fields(content: str) -> dict[str, str] | None:
    normalized = str(content or "").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return None
    end = normalized.find("\n---\n", 4)
    if end < 0:
        return None
    fields: dict[str, str] = {}
    for line in normalized[4:end].splitlines():
        key, separator, value = line.partition(":")
        key = key.strip().casefold()
        if not separator or not key or key in fields:
            return None
        fields[key] = value.strip()
    return fields


def is_protected_profile_path_or_content(
    path: str | Path,
    content: str | None = None,
    *,
    vault_root: str | Path | None = None,
) -> bool:
    """Recognize profile custody without consulting mutable database state."""
    candidate = Path(path)
    if vault_root is None:
        # Generic file reads have no vault object. The exact reserved filename is
        # the only deterministic path signal available on that surface.
        if unicodedata.normalize("NFKC", candidate.name).casefold() == PROFILE_NOTE_PATH:
            return True
    else:
        try:
            resolved_root = Path(vault_root).resolve()
            lexical_candidate = (
                candidate.absolute()
                if candidate.is_absolute()
                else resolved_root / candidate
            )
            try:
                relative = lexical_candidate.relative_to(resolved_root)
            except ValueError:
                relative = None
            if (
                relative is not None
                and unicodedata.normalize("NFKC", relative.as_posix()).casefold()
                == PROFILE_NOTE_PATH
            ):
                return True
            if lexical_candidate.resolve() == (resolved_root / "Profile.md").resolve():
                return True
        except OSError:
            return True

    if content is None:
        return False
    normalized_content = unicodedata.normalize("NFKC", str(content))
    if PROFILE_NOTE_MARKER_RE.search(normalized_content):
        return True

    fields = _frontmatter_fields(normalized_content)
    if fields is None:
        return False
    return bool(
        fields.get("jarvis_projection", "").casefold() == "memory"
        and fields.get("source", "").casefold() == "profile"
        and re.fullmatch(r"[0-9a-f]{32}", fields.get("store_identity", ""))
        and re.fullmatch(r"[1-9][0-9]*", fields.get("memory_id", ""))
    )


def _raw_note_write_path(args: dict[str, Any]) -> str:
    return str(args.get("path") or "").strip()


def _normalized_note_write_args(args: dict[str, Any]) -> tuple[str, str, str]:
    raw_path = _raw_note_write_path(args)
    return (
        _short(raw_path, MAX_NOTE_PATH_CHARS),
        str(args.get("body") or "").strip(),
        _short(args.get("mode") or "append", 16).lower(),
    )


def _canonical_note_target(path: Path, vault: ObsidianVault) -> str:
    relative = path.relative_to(vault.root_path.resolve()).as_posix()
    return unicodedata.normalize("NFKC", relative).casefold()


def _is_profile_note_target(
    path: Path,
    vault: ObsidianVault,
    *,
    raw_path: str,
) -> bool:
    if is_protected_profile_path_or_content(path, vault_root=vault.root_path):
        return True
    candidates = [path]
    try:
        candidates.append(
            _lexical_note_path(
                vault.root_path,
                unicodedata.normalize("NFKC", raw_path),
            )
        )
    except ValueError:
        pass
    return any(
        _canonical_note_target(candidate, vault) == PROFILE_NOTE_PATH
        for candidate in candidates
    )


def _managed_note_projection(
    path: Path,
    vault: ObsidianVault,
) -> tuple[str, str] | None:
    return MANAGED_NOTE_PROJECTIONS.get(_canonical_note_target(path, vault))


def _note_title(path: Path) -> str:
    return path.stem.replace("-", " ").strip() or "Jarvis Note"


def _canonical_create_note_bytes(path: Path, body: str) -> bytes:
    return f"# {_note_title(path)}\n\n{body.rstrip()}\n".encode("utf-8")


def make_write_jarvis_note_auto_mutation_operation_key(
    vault: ObsidianVault,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def operation_key(args: dict[str, Any]) -> dict[str, Any]:
        raw_path, _body, _mode = _normalized_note_write_args(args)
        path = _lexical_note_path(vault.root_path, raw_path)
        return {"target": _canonical_note_target(path, vault)}

    return operation_key


def make_write_jarvis_note_auto_mutation_preflight(
    vault: ObsidianVault,
) -> Callable[[dict[str, Any]], str | None]:
    def preflight(args: dict[str, Any]) -> str | None:
        raw_path_value = _raw_note_write_path(args)
        raw_path, body, mode = _normalized_note_write_args(args)
        if not raw_path:
            return "missing_path"
        if len(raw_path_value) > MAX_NOTE_PATH_CHARS:
            return "path_too_large"
        if not body:
            return "missing_body"
        if len(body) > MAX_NOTE_WRITE_CHARS:
            return "body_too_large"
        try:
            path = _safe_note_path(vault.root_path, raw_path)
        except ValueError:
            return "unsafe_path"
        if _is_profile_note_target(path, vault, raw_path=raw_path_value):
            return "protected_profile"
        if mode not in {"append", "create"}:
            return "bad_mode"
        if _managed_note_projection(path, vault) is not None:
            return "managed_projection"
        if mode == "append":
            try:
                vault.note_matches_exact_bytes(path, b"")
            except FileNotFoundError:
                return None
            except (OSError, ValueError):
                return "unsafe_path"
            return None
        if not path.exists():
            return None
        if not path.is_file():
            if not path.exists():
                return None
            return "already_exists"
        canonical = _canonical_create_note_bytes(path, body)
        try:
            matches = vault.note_matches_exact_bytes(path, canonical)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return "already_exists"
        if matches is None:
            return None
        if not matches:
            return "already_exists"
        return None

    return preflight


def make_write_jarvis_note_auto_mutation_preflight_result(
    vault: ObsidianVault,
) -> Callable[[dict[str, Any], str], ToolResult]:
    def refusal(args: dict[str, Any], reason: str) -> ToolResult:
        raw_path_value = _raw_note_write_path(args)
        raw_path, body, mode = _normalized_note_write_args(args)
        path_display = raw_path
        next_command = ""
        if reason == "managed_projection":
            try:
                managed = _managed_note_projection(
                    _safe_note_path(vault.root_path, raw_path),
                    vault,
                )
            except ValueError:
                managed = None
            if managed is not None:
                path_display, next_command = managed

        outputs = {
            "missing_path": "Note path is empty.",
            "path_too_large": (
                f"Note path is too long; limit is {MAX_NOTE_PATH_CHARS} characters."
            ),
            "missing_body": "Note body is empty.",
            "body_too_large": (
                f"Refusing to write {len(body)} chars to a Jarvis note; "
                f"limit is {MAX_NOTE_WRITE_CHARS}."
            ),
            "bad_mode": "Mode must be append or create.",
            "unsafe_path": "Note path must stay inside the Jarvis Obsidian folder.",
            "protected_profile": (
                "Profile.md is reserved for profile access. Use `read profile` instead."
            ),
            "managed_projection": (
                f"{path_display} is a Jarvis-managed projection; use {next_command} to refresh it."
                if next_command
                else "That note is a Jarvis-managed projection; use its owning command to refresh it."
            ),
            "already_exists": f"Jarvis note already exists: {_path_metadata(raw_path)}",
        }
        metadata = _note_refusal_metadata(
            source="write_jarvis_note",
            mutation="note_write",
            reason=reason,
            path_display=path_display,
            mode=mode,
            body_chars=(len(body) if reason in {"missing_body", "body_too_large"} else None),
            max_chars=(MAX_NOTE_WRITE_CHARS if reason == "body_too_large" else None),
            next_command=next_command or None,
            path_chars=(len(raw_path_value) if reason == "path_too_large" else None),
            max_path_chars=(MAX_NOTE_PATH_CHARS if reason == "path_too_large" else None),
        )
        metadata.update(
            {
                "failure_kind": "auto_mutation_semantic_preflight_rejected",
                "requires_confirmation": False,
                "executed_handler": False,
                "handler_invoked": False,
                "planned_arg_keys": sorted(str(key)[:80] for key in args),
                "authorizes_retry": False,
                "writes_database": False,
            }
        )
        return ToolResult(
            "write_jarvis_note",
            False,
            outputs.get(reason, "The note write failed deterministic validation; nothing was written."),
            metadata,
        )

    return refusal


def _safe_vault_path_display(path: str | Path | None, vault: ObsidianVault) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(vault.root_path.resolve()))
    except ValueError:
        return _short_metadata(candidate, 160)


def _folder_metadata(value: Any) -> str:
    return _short_metadata(value, MAX_NOTE_FOLDER_CHARS)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "reads_private_data": False,
        "writes_files": False,
        "writes_notes": False,
        "writes_memory": False,
        "controls_computer": False,
        "queues_approval": False,
        "external_side_effect": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _note_contract(*, state_changed: bool = False, changed: list[str] | None = None, content_in_handoff: bool = False) -> dict[str, Any]:
    return {
        "ready_for_operator": True,
        "state_changed": state_changed,
        "changed": changed or [],
        "content_in_handoff": content_in_handoff,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _note_handoff_metadata(handoff_key: str, handoff: dict[str, Any], **extra: Any) -> dict[str, Any]:
    _normalize_note_handoff(handoff)
    prefix = handoff_key.removesuffix("_handoff")
    metadata = _safe_metadata(
        **extra,
        **_note_contract(
            state_changed=_metadata_bool(handoff.get("state_changed")),
            changed=list(handoff.get("changed") or []),
            content_in_handoff=_metadata_bool(handoff.get("content_in_handoff")),
        ),
    )
    metadata[handoff_key] = handoff
    metadata[f"{handoff_key}_ready"] = True
    metadata[f"{prefix}_handoff_ready"] = True
    metadata[f"{prefix}_ready_for_operator"] = True
    metadata[f"{prefix}_state_changed"] = _metadata_bool(handoff.get("state_changed"))
    metadata[f"{prefix}_changed"] = list(handoff.get("changed") or [])
    metadata[f"{prefix}_content_in_handoff"] = _metadata_bool(handoff.get("content_in_handoff"))
    metadata[f"{prefix}_next_safe_command"] = handoff["next_safe_command"]
    metadata[f"{prefix}_next_safe_commands"] = list(handoff["next_safe_commands"])
    metadata[f"{prefix}_next_safe_command_count"] = handoff["next_safe_command_count"]
    metadata[f"{prefix}_authorizes_execution"] = False
    metadata[f"{prefix}_authorizes_completion_claim"] = False
    metadata[f"{prefix}_approval_granted"] = False
    metadata["next_safe_command"] = handoff["next_safe_command"]
    metadata["next_safe_commands"] = list(handoff["next_safe_commands"])
    metadata["next_safe_command_count"] = handoff["next_safe_command_count"]
    return metadata


def _normalize_note_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        next_safe_commands = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        next_safe_commands = [str(value) for value in raw_next if str(value or "").strip()]
    next_safe_command = next_safe_commands[0] if next_safe_commands else ""
    handoff["handoff_ready"] = True
    handoff["next_safe_command"] = next_safe_command
    handoff["next_safe_commands"] = list(next_safe_commands)
    handoff["next_safe_command_count"] = len(next_safe_commands)
    return handoff


def _note_boundary(*, reads_contents: bool = False, writes: bool = False) -> dict[str, bool]:
    return {
        "read_only": not writes,
        "reads_note_contents": reads_contents,
        "writes_files": writes,
        "writes_notes": writes,
        "writes_memory": False,
        "queues_approval": False,
        "controls_computer": False,
        "external_side_effect": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _note_refusal_boundaries() -> dict[str, bool]:
    boundaries = _note_boundary(reads_contents=False, writes=False)
    boundaries.update(
        {
            "requires_approval": False,
            "calls_model": False,
            "executes_tools": False,
            "creates_note": False,
            "mutates_note": False,
        }
    )
    return boundaries


def _note_refusal_handoff(
    *,
    source: str,
    mutation: str,
    reason: str,
    path_display: str = "",
    mode: str = "",
    body_chars: int | None = None,
    max_chars: int | None = None,
    exception_type: str | None = None,
) -> dict[str, Any]:
    display = _path_metadata(path_display)
    if reason == "protected_profile":
        return {
            "source": source,
            **_note_contract(),
            "mutation": mutation,
            "reason": reason,
            "refused": True,
            "path_display": "",
            "changed": [],
            "retry_command": "read profile",
            "next_commands": ["read profile"],
            "boundaries": _note_refusal_boundaries(),
        }
    retry_command = "list jarvis notes"
    if source == "read_jarvis_note":
        retry_command = f"read jarvis note {display or '<path>'}"
    elif source == "outline_jarvis_note":
        retry_command = f"outline jarvis note {display or '<path>'}"
    elif source == "write_jarvis_note":
        retry_command = f"append jarvis note {display or '<path>'}: <body>"
    handoff: dict[str, Any] = {
        "source": source,
        **_note_contract(),
        "mutation": mutation,
        "reason": reason,
        "refused": True,
        "path_display": display,
        "changed": [],
        "retry_command": retry_command,
        "next_commands": [retry_command, "list jarvis notes", "search jarvis notes <query>"],
        "boundaries": _note_refusal_boundaries(),
    }
    if mode:
        handoff["mode"] = mode
    if body_chars is not None:
        handoff["body_chars"] = body_chars
    if max_chars is not None:
        handoff["max_chars"] = max_chars
    if exception_type:
        handoff["exception_type"] = exception_type
    return handoff


def _note_refusal_metadata(
    *,
    source: str,
    mutation: str,
    reason: str,
    path_display: str = "",
    mode: str = "",
    body_chars: int | None = None,
    max_chars: int | None = None,
    exception_type: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    handoff = _note_refusal_handoff(
        source=source,
        mutation=mutation,
        reason=reason,
        path_display=path_display,
        mode=mode,
        body_chars=body_chars,
        max_chars=max_chars,
        exception_type=exception_type,
    )
    metadata = _safe_metadata(
        reason=reason,
        **extra,
    )
    metadata.update(_note_handoff_metadata("note_refusal_handoff", handoff))
    if mode:
        metadata["mode"] = mode
    if body_chars is not None:
        metadata["chars"] = body_chars
    if max_chars is not None:
        metadata["max_chars"] = max_chars
    if exception_type:
        metadata["exception_type"] = exception_type
    return metadata


def _note_row(path_display: str, *, size: int | None = None, snippet: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {"path_display": _short_metadata(path_display, MAX_NOTE_PATH_CHARS)}
    if size is not None:
        row["size_bytes"] = size
    if snippet:
        row["snippet"] = _short_metadata(snippet, 220)
    row["read_command"] = f"read jarvis note {row['path_display']}"
    row["outline_command"] = f"outline jarvis note {row['path_display']}"
    return row


def _note_list_handoff(*, rows: list[tuple[str, int, float]], folder: str, limit: int, total: int) -> dict[str, Any]:
    note_rows = [_note_row(rel, size=size) for rel, size, _ in rows[:limit]]
    first_path = note_rows[0]["path_display"] if note_rows else ""
    next_commands: list[str] = []
    if first_path:
        next_commands.extend([f"read jarvis note {first_path}", f"outline jarvis note {first_path}"])
    next_commands.append("search jarvis notes <query>")
    return {
        "source": "list_jarvis_notes",
        **_note_contract(),
        "folder": _folder_metadata(folder),
        "limit": limit,
        "count": len(note_rows),
        "total": total,
        "notes": note_rows,
        "first_note_path": first_path,
        "next_commands": next_commands,
        "boundaries": _note_boundary(reads_contents=False, writes=False),
    }


def _note_search_handoff(*, query: str, matches: list[dict[str, str]], limit: int) -> dict[str, Any]:
    first_path = matches[0]["path_display"] if matches else ""
    next_commands: list[str] = []
    if first_path:
        next_commands.extend([f"read jarvis note {first_path}", f"outline jarvis note {first_path}"])
    next_commands.append("list jarvis notes")
    return {
        "source": "search_jarvis_notes",
        **_note_contract(content_in_handoff=bool(matches)),
        "query_preview": _short_metadata(query, 160),
        "query_chars": len(query),
        "limit": limit,
        "count": len(matches),
        "matches": matches,
        "first_note_path": first_path,
        "next_commands": next_commands,
        "boundaries": _note_boundary(reads_contents=True, writes=False),
    }


def _note_read_handoff(*, source: str, path_display: str, chars: int, truncated: bool, max_chars: int) -> dict[str, Any]:
    display = _path_metadata(path_display)
    return {
        "source": source,
        **_note_contract(),
        "path_display": display,
        "chars": chars,
        "truncated": truncated,
        "max_chars": max_chars,
        "next_commands": [
            f"outline jarvis note {display}",
            f"append jarvis note {display}: <body>",
            "search jarvis notes <query>",
        ],
        "boundaries": _note_boundary(reads_contents=True, writes=False),
    }


def _note_outline_handoff(*, path_display: str, lines: int, words: int, headings: list[str], tasks: list[str], links: list[str]) -> dict[str, Any]:
    display = _path_metadata(path_display)
    return {
        "source": "outline_jarvis_note",
        **_note_contract(content_in_handoff=bool(headings or tasks or links)),
        "path_display": display,
        "lines": lines,
        "words": words,
        "heading_count": len(headings),
        "task_count": len(tasks),
        "link_count": len(links),
        "headings_preview": [_short_metadata(heading, 160) for heading in headings[:8]],
        "tasks_preview": [_short_metadata(task, 180) for task in tasks[:8]],
        "links_preview": [_short_metadata(link, 180) for link in links[:8]],
        "next_commands": [
            f"read jarvis note {display}",
            f"append jarvis note {display}: <body>",
            "list jarvis notes",
        ],
        "boundaries": _note_boundary(reads_contents=True, writes=False),
    }


def _note_write_handoff(
    *, path_display: str, created: bool, mode: str, body: str
) -> dict[str, Any]:
    display = _path_metadata(path_display)
    return {
        "source": "write_jarvis_note",
        **_note_contract(
            state_changed=True,
            changed=["note_write"],
            content_in_handoff=bool(body),
        ),
        "path_display": display,
        "created": created,
        "mode": mode,
        "body_chars": len(body),
        "body_preview": _short_metadata(body, 220),
        "next_commands": [
            f"read jarvis note {display}",
            f"outline jarvis note {display}",
            "list jarvis notes",
        ],
        "boundaries": _note_boundary(reads_contents=False, writes=True),
    }


def _note_io_failure(tool_name: str, action: str, *, path: str, exc: OSError, vault: ObsidianVault, **extra: Any) -> ToolResult:
    path_display = _safe_vault_path_display(path, vault)
    metadata = _note_refusal_metadata(
        source=tool_name,
        mutation="note_write" if action == "write" else "note_read",
        reason=f"{action}_failed",
        path_display=path_display,
        exception_type=type(exc).__name__,
        path=path_display,
        **extra,
    )
    verb = "read" if action == "read" else "write"
    output = (
        f"Could not {verb} Jarvis note. Check JARVIS_OBSIDIAN_VAULT and vault permissions, "
        "run `setup check`, then retry."
    )
    if action == "read":
        output = f"{output} {LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION}"
        metadata = declare_retryable_personal_read_failure(
            metadata,
            output=output,
            action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
            commands=("setup check",),
        )
    else:
        output = (
            "Could not write Jarvis note; outcome unknown. Check JARVIS_OBSIDIAN_VAULT "
            "and vault permissions, run `setup check`, inspect the target note, then retry "
            "only if nothing changed; do not retry automatically."
        )
        metadata = declare_outcome_unknown_failure(
            metadata,
            output=output,
            commands=("setup check",),
        )
    return ToolResult(
        tool_name,
        False,
        output,
        metadata,
    )


def _missing_note_result(raw_path: str) -> ToolResult:
    path_display = _path_metadata(raw_path)
    retry_command = f"read jarvis note {path_display}"
    recovery_commands = ["list jarvis notes", retry_command]
    output = (
        f"Jarvis note not found: {path_display}. Run `list jarvis notes` to confirm the note "
        f"path, then retry `{retry_command}`. {RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
    )
    metadata = declare_resource_not_found_failure(
        _note_refusal_metadata(
            source="read_jarvis_note",
            mutation="note_read",
            reason="not_found",
            path_display=raw_path,
            path=path_display,
            next_command=recovery_commands[0],
            recovery_commands=recovery_commands,
            retry_requires_note_refresh=True,
            recovery_commands_require_normal_policy=True,
            authorizes_retry=False,
        ),
        output=output,
        action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    )
    return ToolResult(
        "read_jarvis_note",
        False,
        output,
        metadata,
    )


def make_note_tools(vault: ObsidianVault):
    def list_jarvis_notes(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 40)
        folder = _short(args.get("folder"), MAX_NOTE_FOLDER_CHARS)
        try:
            start = _safe_folder_path(vault.root_path, folder) if folder else vault.root_path
        except ValueError as exc:
            output = f"{exc} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _safe_metadata(
                    reason="unsafe_folder",
                    folder=_folder_metadata(folder),
                    **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                ),
                output=output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            )
            return ToolResult("list_jarvis_notes", False, output, metadata)
        if start.exists() and not start.is_dir():
            output = f"Folder path is not a folder. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _safe_metadata(
                    reason="not_folder",
                    folder=_folder_metadata(folder),
                    **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                ),
                output=output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            )
            return ToolResult("list_jarvis_notes", False, output, metadata)
        rows: list[tuple[str, int, float]] = []
        resolved_root = vault.root_path.resolve()
        for path in _iter_markdown(start):
            try:
                target_stat = path.lstat()
                if not stat.S_ISREG(target_stat.st_mode):
                    continue
                rel = str(path.resolve().relative_to(resolved_root))
            except (OSError, ValueError):
                continue
            rows.append((rel, target_stat.st_size, target_stat.st_mtime))
        rows.sort(key=lambda row: row[2], reverse=True)
        if not rows:
            label = f" in {folder}" if folder else ""
            handoff = _note_list_handoff(rows=[], folder=folder or "", limit=limit, total=0)
            return ToolResult(
                "list_jarvis_notes",
                True,
                f"No Jarvis notes found{label}. Count: 0.",
                _note_handoff_metadata("note_list_handoff", handoff, count=0, total=0, folder=folder or "", **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit)),
            )
        lines = [f"Count: {min(len(rows), limit)} shown / {len(rows)} total"]
        lines.extend(f"- {rel} ({size} bytes)" for rel, size, _ in rows[:limit])
        if len(rows) > limit:
            lines.append(f"... {len(rows) - limit} more note(s)")
        handoff = _note_list_handoff(rows=rows, folder=folder or "", limit=limit, total=len(rows))
        return ToolResult(
            "list_jarvis_notes",
            True,
            "Jarvis notes:\n" + "\n".join(lines),
            _note_handoff_metadata("note_list_handoff", handoff, count=min(len(rows), limit), total=len(rows), folder=folder or "", **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit)),
        )

    def search_jarvis_notes(args: dict[str, Any]) -> ToolResult:
        query = _short(args.get("query"), MAX_NOTE_QUERY_CHARS)
        limit = _bounded_int(args.get("limit"), 12)
        if not query:
            output = f"Search query is empty. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _safe_metadata(
                    reason="missing_query",
                    query_chars=0,
                    **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit),
                ),
                output=output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            )
            return ToolResult(
                "search_jarvis_notes",
                False,
                output,
                metadata,
            )
        query_low = query.lower()
        match_lines: list[str] = []
        match_rows: list[dict[str, str]] = []
        resolved_root = vault.root_path.resolve()
        for path in _iter_markdown(vault.root_path):
            if is_protected_profile_path_or_content(
                path,
                vault_root=vault.root_path,
            ):
                continue
            try:
                text = vault.read_note_bounded(
                    path,
                    max_chars=MAX_NOTE_CUSTODY_CHARS,
                )
                rel = path.resolve().relative_to(resolved_root)
            except (IsADirectoryError, OSError, OverflowError, ValueError):
                continue
            if text is None:
                continue
            if is_protected_profile_path_or_content(
                path,
                text,
                vault_root=vault.root_path,
            ):
                continue
            haystack = f"{rel}\n{text}".lower()
            if query_low not in haystack:
                continue
            snippet = _snippet(text, query_low)
            rel_text = str(rel)
            match_lines.append(f"- {rel}: {snippet}")
            match_rows.append(_note_row(rel_text, snippet=snippet))
            if len(match_lines) >= limit:
                break
        if not match_lines:
            handoff = _note_search_handoff(query=query, matches=[], limit=limit)
            return ToolResult(
                "search_jarvis_notes",
                True,
                f"No Jarvis notes found for '{query}'.",
                _note_handoff_metadata("note_search_handoff", handoff, count=0, query_chars=len(query), **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit)),
            )
        handoff = _note_search_handoff(query=query, matches=match_rows, limit=limit)
        return ToolResult(
            "search_jarvis_notes",
            True,
            "Jarvis notes:\n" + "\n".join(match_lines),
            _note_handoff_metadata("note_search_handoff", handoff, count=len(match_lines), query_chars=len(query), **_raw_int_metadata(args.get("limit"), key="limit", sanitized=limit)),
        )

    def read_jarvis_note(args: dict[str, Any]) -> ToolResult:
        raw_path = _short(args.get("path"), MAX_NOTE_PATH_CHARS)
        if not raw_path:
            output = f"Note path is empty. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="read_jarvis_note",
                    mutation="note_read",
                    reason="missing_path",
                ),
                output=output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            )
            return ToolResult("read_jarvis_note", False, output, metadata)
        try:
            path = _safe_note_path(vault.root_path, raw_path)
        except ValueError as exc:
            output = f"{exc} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="read_jarvis_note",
                    mutation="note_read",
                    reason="unsafe_path",
                    path_display=raw_path,
                    path=_path_metadata(raw_path),
                ),
                output=output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            )
            return ToolResult("read_jarvis_note", False, output, metadata)
        if _is_profile_note_target(path, vault, raw_path=raw_path):
            output = f"That content is reserved for profile access. {PROFILE_READ_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="read_jarvis_note",
                    mutation="note_read",
                    reason="protected_profile",
                ),
                output=output,
                action=PROFILE_READ_RECOVERY_ACTION,
                commands=("read profile",),
            )
            return ToolResult(
                "read_jarvis_note",
                False,
                output,
                metadata,
            )
        try:
            text = vault.read_note_bounded(
                path,
                max_chars=MAX_NOTE_CUSTODY_CHARS,
            )
        except OverflowError:
            output = (
                "Jarvis note is too large to read safely; "
                f"limit is {MAX_NOTE_CUSTODY_CHARS} characters. "
                f"{NOTE_TOO_LARGE_RECOVERY_ACTION}"
            )
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="read_jarvis_note",
                    mutation="note_read",
                    reason="note_too_large",
                    path_display=raw_path,
                    max_chars=MAX_NOTE_CUSTODY_CHARS,
                ),
                output=output,
                action=NOTE_TOO_LARGE_RECOVERY_ACTION,
            )
            return ToolResult(
                "read_jarvis_note",
                False,
                output,
                metadata,
            )
        except IsADirectoryError:
            text = None
        except ValueError as exc:
            output = f"{exc} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="read_jarvis_note",
                    mutation="note_read",
                    reason="unsafe_path",
                    path_display=raw_path,
                    path=_path_metadata(raw_path),
                ),
                output=output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            )
            return ToolResult(
                "read_jarvis_note",
                False,
                output,
                metadata,
            )
        except OSError as exc:
            return _note_io_failure(
                "read_jarvis_note",
                "read",
                path=str(path),
                exc=exc,
                vault=vault,
            )
        if text is None:
            return _missing_note_result(raw_path)
        if is_protected_profile_path_or_content(
            path,
            text,
            vault_root=vault.root_path,
        ):
            output = f"That content is reserved for profile access. {PROFILE_READ_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="read_jarvis_note",
                    mutation="note_read",
                    reason="protected_profile",
                ),
                output=output,
                action=PROFILE_READ_RECOVERY_ACTION,
                commands=("read profile",),
            )
            return ToolResult(
                "read_jarvis_note",
                False,
                output,
                metadata,
            )
        max_chars = _bounded_int(args.get("max_chars"), 6000, high=MAX_NOTE_READ_CHARS)
        output = text[:max_chars]
        if len(text) > max_chars:
            output += f"\n\n... truncated {len(text) - max_chars} chars"
        path_display = _safe_vault_path_display(path, vault)
        handoff = _note_read_handoff(source="read_jarvis_note", path_display=path_display, chars=len(text), truncated=len(text) > max_chars, max_chars=max_chars)
        return ToolResult(
            "read_jarvis_note",
            True,
            output,
            _note_handoff_metadata("note_read_handoff", handoff, path=str(path), path_display=path_display, chars=len(text), truncated=len(text) > max_chars, **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars)),
        )

    def outline_jarvis_note(args: dict[str, Any]) -> ToolResult:
        raw_path = _short(args.get("path"), MAX_NOTE_PATH_CHARS)
        if not raw_path:
            output = f"Note path is empty. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="outline_jarvis_note",
                    mutation="note_outline",
                    reason="missing_path",
                ),
                output=output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            )
            return ToolResult("outline_jarvis_note", False, output, metadata)
        try:
            path = _safe_note_path(vault.root_path, raw_path)
        except ValueError as exc:
            output = f"{exc} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="outline_jarvis_note",
                    mutation="note_outline",
                    reason="unsafe_path",
                    path_display=raw_path,
                    path=_path_metadata(raw_path),
                ),
                output=output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            )
            return ToolResult("outline_jarvis_note", False, output, metadata)
        if _is_profile_note_target(path, vault, raw_path=raw_path):
            output = f"That content is reserved for profile access. {PROFILE_READ_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="outline_jarvis_note",
                    mutation="note_outline",
                    reason="protected_profile",
                ),
                output=output,
                action=PROFILE_READ_RECOVERY_ACTION,
                commands=("read profile",),
            )
            return ToolResult(
                "outline_jarvis_note",
                False,
                output,
                metadata,
            )
        try:
            text = vault.read_note_bounded(
                path,
                max_chars=MAX_NOTE_CUSTODY_CHARS,
            )
        except OverflowError:
            output = (
                "Jarvis note is too large to outline safely; "
                f"limit is {MAX_NOTE_CUSTODY_CHARS} characters. "
                f"{NOTE_TOO_LARGE_RECOVERY_ACTION}"
            )
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="outline_jarvis_note",
                    mutation="note_outline",
                    reason="note_too_large",
                    path_display=raw_path,
                    max_chars=MAX_NOTE_CUSTODY_CHARS,
                ),
                output=output,
                action=NOTE_TOO_LARGE_RECOVERY_ACTION,
            )
            return ToolResult(
                "outline_jarvis_note",
                False,
                output,
                metadata,
            )
        except IsADirectoryError:
            text = None
        except ValueError as exc:
            output = f"{exc} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="outline_jarvis_note",
                    mutation="note_outline",
                    reason="unsafe_path",
                    path_display=raw_path,
                    path=_path_metadata(raw_path),
                ),
                output=output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            )
            return ToolResult(
                "outline_jarvis_note",
                False,
                output,
                metadata,
            )
        except OSError as exc:
            return _note_io_failure(
                "outline_jarvis_note",
                "read",
                path=str(path),
                exc=exc,
                vault=vault,
            )
        if text is None:
            path_display = _path_metadata(raw_path)
            retry_command = f"outline jarvis note {path_display}"
            recovery_commands = ["list jarvis notes", retry_command]
            output = (
                f"Jarvis note not found: {path_display}. Run `list jarvis notes` to confirm the note "
                f"path, then retry `{retry_command}`. {RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
            )
            metadata = declare_resource_not_found_failure(
                _note_refusal_metadata(
                    source="outline_jarvis_note",
                    mutation="note_outline",
                    reason="not_found",
                    path_display=raw_path,
                    path=path_display,
                    next_command=recovery_commands[0],
                    recovery_commands=recovery_commands,
                    retry_requires_note_refresh=True,
                    recovery_commands_require_normal_policy=True,
                    authorizes_retry=False,
                ),
                output=output,
                action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
            )
            return ToolResult(
                "outline_jarvis_note",
                False,
                output,
                metadata,
            )
        if is_protected_profile_path_or_content(
            path,
            text,
            vault_root=vault.root_path,
        ):
            output = f"That content is reserved for profile access. {PROFILE_READ_RECOVERY_ACTION}"
            metadata = declare_retryable_local_read_failure(
                _note_refusal_metadata(
                    source="outline_jarvis_note",
                    mutation="note_outline",
                    reason="protected_profile",
                ),
                output=output,
                action=PROFILE_READ_RECOVERY_ACTION,
                commands=("read profile",),
            )
            return ToolResult(
                "outline_jarvis_note",
                False,
                output,
                metadata,
            )
        lines = text.splitlines()
        headings = [line.strip() for line in lines if line.lstrip().startswith("#")]
        tasks = [line.strip() for line in lines if line.lstrip().startswith(("- [ ]", "- [x]", "* [ ]", "* [x]"))]
        links = _markdown_links(text)
        path_display = _safe_vault_path_display(path, vault)
        summary = [
            f"Jarvis note outline: {path_display}",
            f"- lines: {len(lines)}",
            f"- words: {len(text.split())}",
            f"- headings: {len(headings)}",
            f"- tasks: {len(tasks)}",
            f"- markdown links: {len(links)}",
        ]
        if headings:
            summary.extend(["", "Headings:", *[f"- {heading}" for heading in headings[:12]]])
        if tasks:
            summary.extend(["", "Tasks:", *[f"- {task}" for task in tasks[:12]]])
        if links:
            summary.extend(["", "Links:", *[f"- {link}" for link in links[:12]]])
        handoff = _note_outline_handoff(path_display=path_display, lines=len(lines), words=len(text.split()), headings=headings, tasks=tasks, links=links)
        return ToolResult(
            "outline_jarvis_note",
            True,
            "\n".join(summary),
            _note_handoff_metadata("note_outline_handoff", handoff, path=str(path), path_display=path_display, lines=len(lines), words=len(text.split()), headings=len(headings), tasks=len(tasks), links=len(links)),
        )

    def write_jarvis_note(args: dict[str, Any]) -> ToolResult:
        raw_path_value = _raw_note_write_path(args)
        raw_path, body, mode = _normalized_note_write_args(args)
        if not raw_path:
            return ToolResult("write_jarvis_note", False, "Note path is empty.", _note_refusal_metadata(source="write_jarvis_note", mutation="note_write", reason="missing_path", mode=mode))
        if len(raw_path_value) > MAX_NOTE_PATH_CHARS:
            return ToolResult(
                "write_jarvis_note",
                False,
                f"Note path is too long; limit is {MAX_NOTE_PATH_CHARS} characters.",
                _note_refusal_metadata(
                    source="write_jarvis_note",
                    mutation="note_write",
                    reason="path_too_large",
                    path_display=raw_path,
                    path=_path_metadata(raw_path),
                    mode=mode,
                    path_chars=len(raw_path_value),
                    max_path_chars=MAX_NOTE_PATH_CHARS,
                ),
            )
        if not body:
            return ToolResult("write_jarvis_note", False, "Note body is empty.", _note_refusal_metadata(source="write_jarvis_note", mutation="note_write", reason="missing_body", path_display=raw_path, path=_path_metadata(raw_path), mode=mode, body_chars=0))
        if len(body) > MAX_NOTE_WRITE_CHARS:
            return ToolResult(
                "write_jarvis_note",
                False,
                f"Refusing to write {len(body)} chars to a Jarvis note; limit is {MAX_NOTE_WRITE_CHARS}.",
                _note_refusal_metadata(source="write_jarvis_note", mutation="note_write", reason="body_too_large", path_display=raw_path, path=_path_metadata(raw_path), mode=mode, body_chars=len(body), max_chars=MAX_NOTE_WRITE_CHARS),
            )
        try:
            path = _safe_note_path(vault.root_path, raw_path)
        except ValueError as exc:
            return ToolResult("write_jarvis_note", False, str(exc), _note_refusal_metadata(source="write_jarvis_note", mutation="note_write", reason="unsafe_path", path_display=raw_path, path=_path_metadata(raw_path), mode=mode))
        if _is_profile_note_target(path, vault, raw_path=raw_path_value):
            return ToolResult(
                "write_jarvis_note",
                False,
                "Profile.md is reserved for profile access. Use `read profile` instead.",
                _note_refusal_metadata(
                    source="write_jarvis_note",
                    mutation="note_write",
                    reason="protected_profile",
                    mode=mode,
                ),
            )
        if mode not in {"append", "create"}:
            return ToolResult("write_jarvis_note", False, "Mode must be append or create.", _note_refusal_metadata(source="write_jarvis_note", mutation="note_write", reason="bad_mode", path_display=raw_path, path=_path_metadata(raw_path), mode=mode))
        managed_projection = _managed_note_projection(path, vault)
        if managed_projection is not None:
            path_display, next_command = managed_projection
            return ToolResult(
                "write_jarvis_note",
                False,
                f"{path_display} is a Jarvis-managed projection; use {next_command} to refresh it.",
                _note_refusal_metadata(
                    source="write_jarvis_note",
                    mutation="note_write",
                    reason="managed_projection",
                    path_display=path_display,
                    path=path_display,
                    mode=mode,
                    next_command=next_command,
                ),
            )
        title = _note_title(path)
        try:
            if mode == "create":
                path = vault.create_note(path, body, heading=title)
                created = True
            else:
                path, created = vault.append_note(path, body, heading=title)
        except FileExistsError:
            try:
                canonical = _canonical_create_note_bytes(path, body)
                converged = vault.note_matches_exact_bytes(path, canonical) is True
            except (OSError, ValueError) as exc:
                return _note_io_failure("write_jarvis_note", "write", path=str(path), exc=exc, vault=vault, mode=mode)
            if mode == "create" and converged:
                return ToolResult(
                    "write_jarvis_note",
                    True,
                    "Jarvis note already matches the requested content.",
                    _safe_metadata(
                        **_note_contract(),
                        converged=True,
                        created=False,
                        mode=mode,
                        body_chars=len(body),
                    ),
                )
            return ToolResult("write_jarvis_note", False, f"Jarvis note already exists: {raw_path}", _note_refusal_metadata(source="write_jarvis_note", mutation="note_write", reason="already_exists", path_display=_safe_vault_path_display(path, vault), path=str(path), mode=mode))
        except ValueError as exc:
            return ToolResult("write_jarvis_note", False, str(exc), _note_refusal_metadata(source="write_jarvis_note", mutation="note_write", reason="unsafe_path", path_display=raw_path, path=_path_metadata(raw_path), mode=mode))
        except OSError as exc:
            return _note_io_failure("write_jarvis_note", "write", path=str(path), exc=exc, vault=vault, mode=mode)
        action = "Created" if created else "Updated"
        path_display = _safe_vault_path_display(path, vault)
        handoff = _note_write_handoff(
            path_display=path_display,
            created=created,
            mode=mode,
            body=body,
        )
        return ToolResult(
            "write_jarvis_note",
            True,
            f"{action} Jarvis note: {path_display}",
            _note_handoff_metadata(
                "note_write_handoff",
                handoff,
                path=str(path),
                path_display=path_display,
                created=created,
                mode=mode,
                body_chars=len(body),
                writes_notes=True,
                writes_files=True,
                approves_request=False,
                dismisses_request=False,
            ),
        )

    return list_jarvis_notes, search_jarvis_notes, read_jarvis_note, outline_jarvis_note, write_jarvis_note


def _iter_markdown(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*.md")):
        try:
            target_stat = path.lstat()
        except OSError:
            continue
        if stat.S_ISREG(target_stat.st_mode):
            yield path


def _lexical_note_path(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError("Use a path relative to the Jarvis Obsidian folder.")
    if ".." in candidate.parts:
        raise ValueError("Note path must stay inside the Jarvis Obsidian folder.")
    if candidate.suffix.lower() != ".md":
        candidate = candidate.with_suffix(".md")
    resolved_root = root.resolve()
    lexical = resolved_root / candidate
    return lexical


def _safe_note_path(root: Path, raw_path: str) -> Path:
    lexical = _lexical_note_path(root, raw_path)
    resolved_root = root.resolve()
    resolved = lexical.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("Note path must stay inside the Jarvis Obsidian folder.")
    return lexical


def _safe_folder_path(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError("Use a folder path relative to the Jarvis Obsidian folder.")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("Folder path must stay inside the Jarvis Obsidian folder.")
    return resolved


def _snippet(text: str, query_low: str, width: int = 180) -> str:
    flat = " ".join(text.split())
    index = flat.lower().find(query_low)
    if index < 0:
        return flat[:width]
    start = max(0, index - 50)
    end = min(len(flat), index + width - 50)
    prefix = "..." if start else ""
    suffix = "..." if end < len(flat) else ""
    return prefix + flat[start:end] + suffix


def _markdown_links(text: str) -> list[str]:
    links: list[str] = []
    for chunk in text.split("[")[1:]:
        if "](" not in chunk:
            continue
        label, rest = chunk.split("](", 1)
        url = rest.split(")", 1)[0].strip()
        if label.strip() and url:
            links.append(f"{label.strip()} -> {url}")
    return links
