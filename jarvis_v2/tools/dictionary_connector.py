"""Live dictionary definitions for Jarvis V2 (dictionaryapi.dev, free, no key)."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    EXTERNAL_INFORMATION_RECOVERY_ACTION,
    declare_retryable_external_information_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools._http import http_get, http_get_json, friendly_http_error


MAX_WORD_CHARS = 60
_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": True,
        "calls_external_services": True,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
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
    }
    base.update(extra)
    base["calls_external_services"] = base.get("calls_external_service") is True
    return base


def _display_word(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = " ".join(text.split())
    if len(text) > MAX_WORD_CHARS:
        return text[: MAX_WORD_CHARS - 1].rstrip() + "..."
    return text


def _dictionary_boundaries(*, calls_external_service: bool = False) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": calls_external_service,
        "calls_external_services": calls_external_service,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
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
    }


def _dictionary_handoff(
    *,
    raw_word: str,
    word: str = "",
    status: str,
    reason: str = "",
    count: int = 0,
    calls_external_service: bool = False,
    exception_type: str = "",
) -> dict[str, Any]:
    safe_command_word = word if word and _WORD_RE.fullmatch(word) and not _looks_like_local_path(word) else ""
    next_safe_command = f"define {safe_command_word}" if safe_command_word else "define <word>"
    boundaries = _dictionary_boundaries(calls_external_service=calls_external_service)
    handoff = {
        "source": "define",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "word": _display_word(word),
        "raw_word_preview": _display_word(raw_word),
        "raw_word_length": len(raw_word),
        "word_length": len(word),
        "word_truncated": len(raw_word.strip().strip("?.!\"'")) > MAX_WORD_CHARS,
        "local_path_word": bool(LOCAL_PATH_RE.search(raw_word or "")),
        "definition_count": count,
        "exception_type": exception_type,
        "content_in_handoff": False,
        "content_in_metadata": False,
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "dictionary_handoff_ready": True,
        "dictionary_ready_for_operator": handoff["ready_for_operator"],
        "dictionary_state_changed": handoff["state_changed"],
        "dictionary_changed": handoff["changed"],
        "dictionary_content_in_handoff": handoff["content_in_handoff"],
        "dictionary_content_in_metadata": handoff["content_in_metadata"],
        "dictionary_next_safe_command": handoff["next_safe_command"],
        "dictionary_next_safe_commands": handoff["next_safe_commands"],
        "dictionary_next_safe_command_count": handoff["next_safe_command_count"],
        "dictionary_authorizes_execution": handoff["authorizes_execution"],
        "dictionary_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "dictionary_approval_granted": handoff["approval_granted"],
        "dictionary_boundaries": boundaries,
        "dictionary_handoff": handoff,
    }


def _looks_like_local_path(value: str) -> bool:
    return bool(LOCAL_PATH_RE.search(value or ""))


def _clean_word(value: str) -> str:
    text = " ".join(str(value or "").strip().strip("?.!\"'").split())
    if len(text.split()) > 1:
        text = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", text, flags=re.IGNORECASE).strip()
    return text


def _fetch(word: str) -> list:
    url = "https://api.dictionaryapi.dev/api/v2/entries/en/" + urllib.parse.quote(word)
    return http_get_json(url, headers={"User-Agent": "Mozilla/5.0"})


def _dictionary_recovery_message(prefix: str) -> str:
    return f"{prefix} Check network access to dictionaryapi.dev, run setup check, then retry in a moment."


def _dictionary_error_message(e: Exception, *, word: str) -> str:
    friendly = friendly_http_error(e, subject=f"a definition for '{word}'", service="The dictionary")
    lowered = friendly.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return _dictionary_recovery_message("The dictionary request timed out.")
    if friendly.startswith("I couldn't find"):
        return friendly
    if friendly.startswith("Error:") or "having trouble" in lowered or "try again" in lowered:
        return _dictionary_recovery_message("The dictionary service is having trouble right now.")
    return friendly


def make_dictionary_tools(config: JarvisConfig):
    def define(args: dict[str, Any]) -> ToolResult:
        raw_word = str(args.get("word") or args.get("text") or "")
        cleaned_word = _clean_word(raw_word)
        word = cleaned_word[:MAX_WORD_CHARS]
        metadata = {
            "word_length": len(word),
            "raw_word_length": len(raw_word),
            "max_word_chars": MAX_WORD_CHARS,
            "word_truncated": len(cleaned_word) > MAX_WORD_CHARS,
        }
        if not word:
            return ToolResult(
                "define",
                False,
                "Which word should I define?",
                _safe_metadata(
                    **metadata,
                    calls_external_service=False,
                    reason="missing_word",
                    **_dictionary_handoff(raw_word=raw_word, word=word, status="refused", reason="missing_word"),
                ),
            )
        if _looks_like_local_path(cleaned_word):
            return ToolResult(
                "define",
                False,
                "Please give me one English word to define, not a local file path.",
                _safe_metadata(
                    **metadata,
                    calls_external_service=False,
                    reason="invalid_word",
                    local_path_word=True,
                    **_dictionary_handoff(raw_word=raw_word, word=word, status="refused", reason="invalid_word"),
                ),
            )
        if len(cleaned_word) > MAX_WORD_CHARS or not _WORD_RE.fullmatch(word):
            return ToolResult(
                "define",
                False,
                "Please give me one English word to define.",
                _safe_metadata(
                    **metadata,
                    calls_external_service=False,
                    reason="invalid_word",
                    **_dictionary_handoff(raw_word=raw_word, word=word, status="refused", reason="invalid_word"),
                ),
            )
        try:
            data = _fetch(word)
            if not isinstance(data, list) or not data:
                return ToolResult(
                    "define",
                    True,
                    f"No definition found for '{word}'.",
                    _safe_metadata(
                        **metadata,
                        count=0,
                        **_dictionary_handoff(raw_word=raw_word, word=word, status="empty", count=0, calls_external_service=True),
                    ),
                )
            entry = data[0]
            phonetic = entry.get("phonetic") or ""
            lines = [f"{entry.get('word', word)} {phonetic}".strip()]
            shown = 0
            for meaning in entry.get("meanings", []):
                pos = meaning.get("partOfSpeech", "")
                for d in meaning.get("definitions", [])[:1]:
                    definition = d.get("definition", "").strip()
                    if definition:
                        lines.append(f"  ({pos}) {definition}")
                        shown += 1
                if shown >= 3:
                    break
            return ToolResult(
                "define",
                True,
                "\n".join(lines),
                _safe_metadata(
                    **metadata,
                    word=word,
                    count=shown,
                    **_dictionary_handoff(raw_word=raw_word, word=word, status="ok", count=shown, calls_external_service=True),
                ),
            )
        except Exception as e:
            msg = str(e)
            if "404" in msg:
                return ToolResult(
                    "define",
                    True,
                    f"No definition found for '{word}'.",
                    _safe_metadata(
                        **metadata,
                        count=0,
                        **_dictionary_handoff(raw_word=raw_word, word=word, status="empty", reason="not_found", count=0, calls_external_service=True),
                    ),
                )
            failure_output = (
                f"{_dictionary_error_message(e, word=word)} "
                f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "define",
                False,
                failure_output,
                declare_retryable_external_information_failure(
                    _safe_metadata(
                        **metadata,
                        exception_type=type(e).__name__,
                        **_dictionary_handoff(
                            raw_word=raw_word,
                            word=word,
                            status="unavailable",
                            reason="fetch_error",
                            calls_external_service=True,
                            exception_type=type(e).__name__,
                        ),
                    ),
                    output=failure_output,
                    action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    from jarvis_v2.tools.registry import Tool, _tool_argument_contract

    return [
        Tool(
            "define",
            "Define an English word (free dictionary). Args: word or text.",
            RiskLevel.LOCAL_SAFE,
            define,
            "personal",
            argument_contract=_tool_argument_contract(optional_strings=("word", "text")),
        ),
    ]
