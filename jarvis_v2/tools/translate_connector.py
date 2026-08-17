"""Live translation for Jarvis V2 via MyMemory (free, no API key)."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    EXTERNAL_INFORMATION_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_retryable_external_information_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools._http import http_get, http_get_json, friendly_http_error


LANG_ALIASES = {
    "english": "en", "영어": "en", "en": "en",
    "korean": "ko", "한국어": "ko", "ko": "ko", "한국말": "ko",
    "japanese": "ja", "일본어": "ja", "ja": "ja",
    "chinese": "zh", "중국어": "zh", "zh": "zh", "mandarin": "zh",
    "spanish": "es", "es": "es", "french": "fr", "fr": "fr",
    "german": "de", "de": "de", "italian": "it", "it": "it",
}
MAX_TEXT_CHARS = 500
MAX_RAW_LANG_CHARS = 80
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
PRIVATE_VALUE_RE = re.compile(
    r"(?:"
    r"\bsk_(?:live|test)_[A-Za-z0-9_-]+"
    r"|\bgh[pousr]_[A-Za-z0-9_-]+"
    r"|\bxox[baprs]-[A-Za-z0-9-]+"
    r"|\bAIza[A-Za-z0-9_-]{16,}"
    r"|\b\d{6,}:[A-Za-z0-9_-]{20,}"
    r"|\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*\S+"
    r")",
    re.IGNORECASE,
)
TRANSLATION_INPUT_RECOVERY_ACTION = (
    "Correct the translation text or language, then submit a new translation request."
)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": True,
        "calls_external_services": True,
        "executes_tools": False,
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
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    base.update(extra)
    base["calls_external_services"] = base.get("calls_external_service") is True
    return base


def _translate_boundaries(*, calls_external_service: bool) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": calls_external_service,
        "calls_external_services": calls_external_service,
        "executes_tools": False,
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
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _translate_handoff(
    *,
    status: str,
    reason: str = "",
    raw_source: Any = None,
    raw_target: Any = None,
    source: str = "",
    target: str = "",
    text_chars: int = 0,
    text_truncated: bool = False,
    local_path_text: bool = False,
    translated_text_available: bool = False,
    exception_type: str = "",
    calls_external_service: bool,
) -> dict[str, Any]:
    boundaries = _translate_boundaries(calls_external_service=calls_external_service)
    next_safe_command = "translate <text> to <language>"
    handoff = {
        "source": "mymemory",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "source_lang": source,
        "target_lang": target,
        "raw_from": _short_raw(raw_source),
        "raw_to": _short_raw(raw_target),
        "text_chars": text_chars,
        "max_text_chars": MAX_TEXT_CHARS,
        "text_truncated": text_truncated,
        "local_path_text": local_path_text,
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "input_text_in_metadata": False,
        "translated_text_in_metadata": False,
        "translated_text_available": translated_text_available,
        "exception_type": exception_type,
        "retry_safe": status in {"empty", "unavailable"},
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "translate_handoff_ready": True,
        "translate_ready_for_operator": handoff["ready_for_operator"],
        "translate_state_changed": handoff["state_changed"],
        "translate_changed": handoff["changed"],
        "translate_content_in_handoff": handoff["content_in_handoff"],
        "translate_input_text_in_metadata": handoff["input_text_in_metadata"],
        "translate_translated_text_in_metadata": handoff["translated_text_in_metadata"],
        "translate_next_safe_command": handoff["next_safe_command"],
        "translate_next_safe_commands": handoff["next_safe_commands"],
        "translate_next_safe_command_count": handoff["next_safe_command_count"],
        "translate_authorizes_execution": handoff["authorizes_execution"],
        "translate_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "translate_approval_granted": handoff["approval_granted"],
        "translate_boundaries": boundaries,
        "translate_handoff": handoff,
    }


def _short_raw(value: Any, limit: int = MAX_RAW_LANG_CHARS) -> str:
    text = "" if value is None else str(value)
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = PRIVATE_VALUE_RE.sub("<private-value>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _looks_like_local_path(value: str) -> bool:
    return bool(LOCAL_PATH_RE.search(value or ""))


def _looks_like_private_text(value: str) -> bool:
    return _looks_like_local_path(value) or bool(PRIVATE_VALUE_RE.search(value or ""))


def _lang(token: str) -> str | None:
    return LANG_ALIASES.get((token or "").strip().lower())


def _detect_lang(text: str) -> str:
    if re.search(r"[가-힣]", text):
        return "ko"
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[一-鿿]", text):
        return "zh"
    return "en"


def _translate_recovery_message(prefix: str) -> str:
    return (
        f"{prefix} Check network access to MyMemory, run setup check, then retry in a moment. "
        f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
    )


def _translate_input_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
) -> dict[str, Any]:
    return declare_failure_guidance(
        metadata,
        output=output,
        action=TRANSLATION_INPUT_RECOVERY_ACTION,
    )


def _translate_external_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
) -> dict[str, Any]:
    return declare_retryable_external_information_failure(
        metadata,
        output=output,
        action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
        commands=("setup check",),
    )


def _translate_error_message(e: Exception) -> str:
    friendly = friendly_http_error(e, service="The translation service")
    lowered = friendly.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return _translate_recovery_message("The translation request timed out.")
    if friendly.startswith("I couldn't find"):
        return _translate_recovery_message("The translation service endpoint could not be reached.")
    if friendly.startswith("Error:") or "having trouble" in lowered or "try again" in lowered:
        return _translate_recovery_message("The translation service is having trouble right now.")
    return friendly


def _looks_like_bare_phrase(phrase: str) -> bool:
    phrase = phrase.strip().strip("\"'")
    if not phrase or "?" in phrase or len(phrase) > MAX_TEXT_CHARS:
        return False
    low = phrase.lower()
    if re.match(
        r"^(?:what|when|where|who|why|how|is|are|do|does|did|will|should|can|could|would|tell me|look ?up|search|research|find|open|show|list|weather|forecast|news|market|markets|stock|crypto|calendar|schedule|agenda|reminder|timer|alarm|email|message)\b",
        low,
    ):
        return False
    return bool(re.search(r"[A-Za-z가-힣]", phrase))


def parse_translation(text: str) -> tuple[str, str, str] | None:
    """Return (phrase, source_lang, target_lang) or None."""
    raw = text.strip()
    low = raw.lower()
    target = None
    phrase = None

    m = re.search(r"\b(?:translate|say)\s+(.+?)\s+(?:to|in)(?:to)?\s+([a-z가-힣]+)\b", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"\bhow do you say\s+(.+?)\s+in\s+([a-z가-힣]+)\b", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"\bhow to say\s+(.+?)\s+in\s+([a-z가-힣]+)\b", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"\bwhat(?:'s| is)\s+(.+?)\s+in\s+([a-z가-힣]+)\b", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"\bwhat does\s+(.+?)\s+mean\s+in\s+([a-z가-힣]+)\b", raw, re.IGNORECASE)
    if not m:
        bare_m = re.search(r"^(.{1,120}?)\s+in\s+([a-z가-힣]+)$", raw, re.IGNORECASE)
        if bare_m and _lang(bare_m.group(2)) and _looks_like_bare_phrase(bare_m.group(1)):
            m = bare_m
    if m:
        phrase = m.group(1).strip().strip("\"'")
        target = _lang(m.group(2))
    else:
        m = re.search(r"^([a-z가-힣]+)\s+for\s+(.+?)$", raw, re.IGNORECASE)
        if m:
            target = _lang(m.group(1))
            if target:
                phrase = m.group(2).strip().strip("\"'")
    if not phrase:
        m = re.search(r"\btranslate\s+(.+)$", raw, re.IGNORECASE)
        if m:
            phrase = m.group(1).strip().strip("\"'")

    if not phrase:
        return None
    source = _detect_lang(phrase)
    if target is None:
        target = "ko" if source == "en" else "en"
    if target == source:
        target = "en" if source != "en" else "ko"
    return phrase[:MAX_TEXT_CHARS], source, target


def _fetch(phrase: str, source: str, target: str) -> dict:
    params = urllib.parse.urlencode({"q": phrase, "langpair": f"{source}|{target}"})
    url = "https://api.mymemory.translated.net/get?" + params
    return http_get_json(url, headers={"User-Agent": "Mozilla/5.0"})


MAX_SHORT_PHRASE_TOKENS = 3
"""MyMemory is a crowd-sourced translation-memory lookup, not a model -- a short
common phrase occasionally matches an unrelated, longer example sentence someone
submitted (observed: "hello" -> "제 이름은 Azlan입니다."
/ "My name is Azlan.", tied at quality=74 with the correct "안녕하세요").
Character-length comparison is unreliable across languages (Korean stays compact
even for full sentences), so token/word count is used instead: a short source
phrase resolving to a candidate with meaningfully more tokens than the source is
a general, language-agnostic signal of exactly that mismatch."""


def _int_or_zero(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _token_count(text: str) -> int:
    return len(text.split()) or 1


def _pick_best_translation(data: dict, phrase: str, top_pick: str) -> str:
    """Prefer a comparable-quality alternate from MyMemory's own `matches` list
    when the top pick looks like a token-count-mismatched crowd submission for a
    short source phrase, OR when the top pick is empty outright (a real,
    observed MyMemory data-quality issue: `responseData.translatedText` can be
    "" while `matches[]` still holds a perfectly good candidate -- e.g. "thank
    you" -> ja once returned an empty top pick tied at quality=74 with a
    genuinely correct non-empty match). Never invents a translation -- only
    re-ranks candidates MyMemory itself already returned, and only when a
    clearly safer/non-empty alternate exists; otherwise falls back to the
    original top pick unchanged, so normal (non-mismatched) translations are
    never touched."""
    phrase_tokens = _token_count(phrase)
    top_pick_is_empty = not top_pick
    if not top_pick_is_empty:
        top_pick_tokens = _token_count(top_pick)
        if phrase_tokens > MAX_SHORT_PHRASE_TOKENS or top_pick_tokens <= phrase_tokens + 1:
            return top_pick
    matches = data.get("matches")
    if not isinstance(matches, list):
        return top_pick
    top_pick_quality = 0
    if not top_pick_is_empty:
        for entry in matches:
            if isinstance(entry, dict) and str(entry.get("translation") or "").strip() == top_pick:
                top_pick_quality = _int_or_zero(entry.get("quality"))
                break
    best_candidate = ""
    best_score = (-1, -1.0)
    for entry in matches:
        if not isinstance(entry, dict):
            continue
        candidate = str(entry.get("translation") or "").strip()
        if not candidate or candidate == top_pick:
            continue
        if not top_pick_is_empty and _token_count(candidate) > phrase_tokens + 1:
            continue
        quality = _int_or_zero(entry.get("quality"))
        if quality < top_pick_quality:
            continue
        score = (quality, _float_or_zero(entry.get("match")))
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate or top_pick


def make_translate_tools(config: JarvisConfig):
    def translate(args: dict[str, Any]) -> ToolResult:
        phrase = str(args.get("text") or args.get("phrase") or "").strip()
        raw_source = args.get("from")
        raw_target = args.get("to")
        source = _lang(str(raw_source or "")) or ""
        target = _lang(str(raw_target or "")) or ""
        lang_metadata = {
            "raw_from": _short_raw(raw_source),
            "raw_to": _short_raw(raw_target),
            "text_chars": min(len(phrase), MAX_TEXT_CHARS),
            "max_text_chars": MAX_TEXT_CHARS,
            "text_truncated": len(phrase) > MAX_TEXT_CHARS,
        }
        if raw_source not in (None, "") and not source:
            failure_output = (
                f"Unsupported source language. {TRANSLATION_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "translate",
                False,
                failure_output,
                _translate_input_failure_metadata(
                    _safe_metadata(
                        **{
                            **lang_metadata,
                            "calls_external_service": False,
                            "reason": "unsupported_language",
                            **_translate_handoff(
                                status="refused",
                                reason="unsupported_language",
                                raw_source=raw_source,
                                raw_target=raw_target,
                                text_chars=lang_metadata["text_chars"],
                                text_truncated=lang_metadata["text_truncated"],
                                calls_external_service=False,
                            ),
                        }
                    ),
                    output=failure_output,
                ),
            )
        if raw_target not in (None, "") and not target:
            failure_output = (
                f"Unsupported target language. {TRANSLATION_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "translate",
                False,
                failure_output,
                _translate_input_failure_metadata(
                    _safe_metadata(
                        **{
                            **lang_metadata,
                            "calls_external_service": False,
                            "reason": "unsupported_language",
                            **_translate_handoff(
                                status="refused",
                                reason="unsupported_language",
                                raw_source=raw_source,
                                raw_target=raw_target,
                                source=source,
                                text_chars=lang_metadata["text_chars"],
                                text_truncated=lang_metadata["text_truncated"],
                                calls_external_service=False,
                            ),
                        }
                    ),
                    output=failure_output,
                ),
            )
        if phrase and target:
            phrase = phrase[:MAX_TEXT_CHARS]
            source = source or _detect_lang(phrase)
        else:
            parsed = parse_translation(str(args.get("request") or args.get("text") or phrase))
            if parsed is None:
                failure_output = (
                    "Tell me what to translate, e.g. 'translate hello to Korean'. "
                    f"{TRANSLATION_INPUT_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "translate",
                    False,
                    failure_output,
                    _translate_input_failure_metadata(
                        _safe_metadata(
                            **{
                                **lang_metadata,
                                "calls_external_service": False,
                                "reason": "missing_text",
                                **_translate_handoff(
                                    status="refused",
                                    reason="missing_text",
                                    raw_source=raw_source,
                                    raw_target=raw_target,
                                    text_chars=lang_metadata["text_chars"],
                                    text_truncated=lang_metadata["text_truncated"],
                                    calls_external_service=False,
                                ),
                            }
                        ),
                        output=failure_output,
                    ),
                )
            phrase, source, target = parsed
            lang_metadata = {**lang_metadata, "text_chars": len(phrase), "text_truncated": len(phrase) >= MAX_TEXT_CHARS}
        if _looks_like_private_text(phrase):
            local_path_text = _looks_like_local_path(phrase)
            refusal_reason = "invalid_text" if local_path_text else "private_text"
            refusal_message = (
                "Please give me text to translate, not a local file path."
                if local_path_text
                else "Please give me public text to translate, not a private-looking value."
            )
            failure_output = (
                f"{refusal_message} {TRANSLATION_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "translate",
                False,
                failure_output,
                _translate_input_failure_metadata(
                    _safe_metadata(
                        **{
                            **lang_metadata,
                            "source": source,
                            "target": target,
                            "text_chars": len(phrase),
                            "reason": refusal_reason,
                            "local_path_text": local_path_text,
                            "private_looking_text": True,
                            "calls_external_service": False,
                            **_translate_handoff(
                                status="refused",
                                reason=refusal_reason,
                                raw_source=raw_source,
                                raw_target=raw_target,
                                source=source,
                                target=target,
                                text_chars=len(phrase),
                                text_truncated=lang_metadata["text_truncated"],
                                local_path_text=local_path_text,
                                calls_external_service=False,
                            ),
                        }
                    ),
                    output=failure_output,
                ),
            )
        try:
            data = _fetch(phrase, source, target)
            translated = ((data.get("responseData") or {}).get("translatedText") or "").strip()
            translated = _pick_best_translation(data, phrase, translated)
            if not translated:
                failure_output = _translate_recovery_message("No translation returned.")
                return ToolResult(
                    "translate",
                    False,
                    failure_output,
                    _translate_external_failure_metadata(
                        _safe_metadata(
                            **{
                                **lang_metadata,
                                "source": source,
                                "target": target,
                                "text_chars": len(phrase),
                                "reason": "no_translation",
                                **_translate_handoff(
                                    status="empty",
                                    reason="no_translation",
                                    raw_source=raw_source,
                                    raw_target=raw_target,
                                    source=source,
                                    target=target,
                                    text_chars=len(phrase),
                                    text_truncated=lang_metadata["text_truncated"],
                                    calls_external_service=True,
                                ),
                            }
                        ),
                        output=failure_output,
                    ),
                )
            return ToolResult(
                "translate", True,
                f"{phrase} → {translated}",
                _safe_metadata(
                    **{
                        **lang_metadata,
                        "source": source,
                        "target": target,
                        "text_chars": len(phrase),
                        **_translate_handoff(
                            status="ok",
                            raw_source=raw_source,
                            raw_target=raw_target,
                            source=source,
                            target=target,
                            text_chars=len(phrase),
                            text_truncated=lang_metadata["text_truncated"],
                            translated_text_available=True,
                            calls_external_service=True,
                        ),
                    }
                ),
            )
        except Exception as e:
            failure_output = _translate_error_message(e)
            return ToolResult(
                "translate",
                False,
                failure_output,
                _translate_external_failure_metadata(
                    _safe_metadata(
                        **{
                            **lang_metadata,
                            "source": source,
                            "target": target,
                            "text_chars": len(phrase),
                            "reason": "fetch_error",
                            "exception_type": type(e).__name__,
                            **_translate_handoff(
                                status="unavailable",
                                reason="fetch_error",
                                raw_source=raw_source,
                                raw_target=raw_target,
                                source=source,
                                target=target,
                                text_chars=len(phrase),
                                text_truncated=lang_metadata["text_truncated"],
                                exception_type=type(e).__name__,
                                calls_external_service=True,
                            ),
                        }
                    ),
                    output=failure_output,
                ),
            )

    from jarvis_v2.tools.registry import Tool, _tool_argument_contract
    return [
        Tool(
            "translate",
            "Translate text between languages (free). Args: text + to (e.g. 'korean'), or request like 'translate hello to Korean'.",
            RiskLevel.LOCAL_SAFE,
            translate,
            "personal",
            argument_contract=_tool_argument_contract(
                optional_strings=("text", "phrase", "request", "from", "to"),
            ),
        ),
    ]
