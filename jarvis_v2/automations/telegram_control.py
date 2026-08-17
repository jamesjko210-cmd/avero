"""Owner-locked Telegram command channel for Jarvis V2.

This is the remote phone-control channel: you message your bot from anywhere,
your Mac (which polls Telegram outbound) runs the command through the V2 runtime,
and the reply comes back to your phone. Nothing is exposed to the internet — the
Mac only makes outbound requests to Telegram.

Security model:
- only messages from JARVIS_OWNER_TELEGRAM (your numeric chat id) are obeyed
- every command runs through JarvisRuntime, so the normal safety harness
  (planner, registry, permission policy, approval queue, audit log) still gates
  every side effect
- a persisted update offset prevents reprocessing on restart
- a fresh install drains backlog so stale setup messages are never executed
"""

from __future__ import annotations

import json
import os
import pwd
import re
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

import certifi

from jarvis_v2.automations import telegram_offset_state


STATE_FILE: Path | None = None
API_BASE = "https://api.telegram.org/bot{token}/{method}"
LONG_POLL_SECONDS = 20
MAX_TRANSPORT_UPDATE_ID = (1 << 63) - 1
TELEGRAM_TEXT_LIMIT = 4096
MAX_REPLY_CHARS = 3900  # Telegram hard limit is 4096; keep chunk headroom.
# Telegram's "typing…" status auto-expires after ~5s, so refresh just under that
# to keep it continuous while Jarvis is thinking (the model can take up to ~20s).
TYPING_REFRESH_SECONDS = 4.0
LOCAL_PATH_RE = re.compile(r"(?<![\w./-])/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
LOCAL_PATH_ROOT_RE = re.compile(
    r"(?<![\w./-])/(?:users|private|var/folders|tmp)(?:/|$)",
    re.IGNORECASE,
)
_STOP = threading.Event()
_THREAD: threading.Thread | None = None
_TELEGRAM_SSL_CONTEXT: ssl.SSLContext | None = None
_TELEGRAM_SSL_CONTEXT_LOCK = threading.Lock()

_COCKPIT_LANE_ALIASES = {
    "message": "messaging",
    "messages": "messaging",
    "messaging": "messaging",
    "send": "messaging",
    "sending": "messaging",
    "메시지": "messaging",
    "메시징": "messaging",
    "문자": "messaging",
    "call": "calls",
    "calls": "calls",
    "calling": "calls",
    "phonecall": "calls",
    "통화": "calls",
    "전화": "calls",
    "brief": "morning_brief",
    "dailybrief": "morning_brief",
    "morningbrief": "morning_brief",
    "브리핑": "morning_brief",
    "모닝브리핑": "morning_brief",
    "아침브리핑": "morning_brief",
    "contact": "contacts",
    "contacts": "contacts",
    "people": "contacts",
    "person": "contacts",
    "연락처": "contacts",
    "사람": "contacts",
    "calendaremail": "calendar_email",
    "calendar": "calendar_email",
    "email": "calendar_email",
    "emails": "calendar_email",
    "캘린더": "calendar_email",
    "캘린더이메일": "calendar_email",
    "일정": "calendar_email",
    "이메일": "calendar_email",
    "personalproof": "personal_proofs",
    "personalproofs": "personal_proofs",
    "personal": "personal_proofs",
    "integrationproof": "personal_proofs",
    "integrationproofs": "personal_proofs",
    "integrations": "personal_proofs",
    "개인증명": "personal_proofs",
    "개인통합": "personal_proofs",
    "통합증명": "personal_proofs",
    "write": "writing",
    "writer": "writing",
    "writing": "writing",
    "type": "writing",
    "typing": "writing",
    "paste": "writing",
    "compose": "writing",
    "draft": "writing",
    "쓰기": "writing",
    "작성": "writing",
    "글쓰기": "writing",
    "타이핑": "writing",
    "붙여넣기": "writing",
    "초안": "writing",
    "market": "markets",
    "markets": "markets",
    "weather": "markets",
    "marketweather": "markets",
    "마켓": "markets",
    "시장": "markets",
    "날씨": "markets",
    "research": "research_web",
    "researchandweb": "research_web",
    "researchweb": "research_web",
    "web": "research_web",
    "websearch": "research_web",
    "lookup": "research_web",
    "search": "research_web",
    "검색": "research_web",
    "웹": "research_web",
    "웹검색": "research_web",
    "연구": "research_web",
    "리서치": "research_web",
    "조사": "research_web",
    "memory": "memory",
    "memories": "memory",
    "notes": "memory",
    "기억": "memory",
    "메모리": "memory",
    "노트": "memory",
    "learning": "learning_loop",
    "learningloop": "learning_loop",
    "feedback": "learning_loop",
    "failurelearning": "learning_loop",
    "학습": "learning_loop",
    "러닝": "learning_loop",
    "피드백": "learning_loop",
    "voice": "voice",
    "speech": "voice",
    "mic": "voice",
    "microphone": "voice",
    "음성": "voice",
    "마이크": "voice",
    "diagnostic": "diagnostics",
    "diagnostics": "diagnostics",
    "doctor": "diagnostics",
    "failure": "diagnostics",
    "failures": "diagnostics",
    "진단": "diagnostics",
    "고장": "diagnostics",
    "실패": "diagnostics",
    "approval": "approvals",
    "approvals": "approvals",
    "approvalgate": "approvals",
    "approvalgates": "approvals",
    "승인": "approvals",
    "guardrail": "build_guardrails",
    "guardrails": "build_guardrails",
    "buildguardrail": "build_guardrails",
    "buildguardrails": "build_guardrails",
    "가드레일": "build_guardrails",
    "빌드가드레일": "build_guardrails",
    "workfloweval": "operator_workflow_evals",
    "workflowevals": "operator_workflow_evals",
    "operatoreval": "operator_workflow_evals",
    "operatorevals": "operator_workflow_evals",
    "워크플로우평가": "operator_workflow_evals",
    "제임스평가": "operator_workflow_evals",
    "schedule": "scheduler",
    "scheduler": "scheduler",
    "scheduledjob": "scheduler",
    "scheduledjobs": "scheduler",
    "스케줄": "scheduler",
    "예약작업": "scheduler",
    "orchestration": "orchestration",
    "agent": "orchestration",
    "subagent": "orchestration",
    "subagents": "orchestration",
    "agents": "orchestration",
    "worker": "orchestration",
    "workers": "orchestration",
    "오케스트레이션": "orchestration",
    "에이전트": "orchestration",
    "서브에이전트": "orchestration",
    "워커": "orchestration",
    "작업자": "orchestration",
}

_LANE_FILLER_WORDS = {
    "show",
    "the",
    "jarvis",
    "cockpit",
    "capability",
    "capabilities",
    "lane",
    "status",
    "health",
    "view",
    "detail",
    "details",
    "report",
}

_ENGLISH_LANE_MARKERS = {"lane", "status", "health", "cockpit", "detail", "details", "report"}

_KOREAN_LANE_PREFIXES = ("자비스", "콕핏", "기능")
_KOREAN_LANE_SUFFIXES = ("보여줘", "알려줘", "상태", "콕핏", "레인", "상세", "보기")
_UNREADABLE_OUTBOUND_TEXT = (
    "Jarvis produced a reply I couldn't safely display. Check the local Jarvis logs and run `execution health report` before retrying; do not replay an approval-gated action until its outcome is verified."
)


def _has_local_path(value: str) -> bool:
    raw = _clean(value)
    return bool(raw and (Path(raw).is_absolute() or "/" in raw or "\\" in raw))


def _clean(value) -> str:
    return "" if value is None else str(value).strip()


def _clean_untrusted_text(value: Any, *, fallback: str = "") -> str:
    try:
        return _clean(value)
    except Exception:
        return fallback


def _result_response_text(
    result: object,
    *,
    default: str = "Done.",
    unreadable: str = _UNREADABLE_OUTBOUND_TEXT,
) -> str:
    try:
        value = getattr(result, "response", "")
    except Exception:
        return unreadable or default
    text = _clean_untrusted_text(value, fallback=unreadable)
    return text or default


def _clean_cockpit_display(value: Any, *, limit: int = 180) -> str:
    try:
        raw = _clean(value)
    except Exception:
        return "<unreadable>"
    if not raw:
        return ""
    if LOCAL_PATH_RE.search(raw) or LOCAL_PATH_ROOT_RE.search(raw):
        return "<redacted-local-path>"
    return raw[:limit]


def _clean_cockpit_direction(value: Any, *, limit: int = 180) -> str:
    return _clean_cockpit_display(value, limit=limit)


def _exact_true(value: Any) -> bool:
    return value is True


def _cockpit_coverage_summary_line(metadata: dict[str, Any], lanes: list[dict[str, Any]]) -> str:
    raw_summary = metadata.get("coverage_summary")
    if isinstance(raw_summary, dict):
        summary = _clean_cockpit_display(raw_summary.get("summary"))
        if summary:
            return summary

    if not lanes:
        return ""
    lane_count = len(lanes)
    tool_ready = sum(
        1
        for lane in lanes
        if _exact_true(lane.get("tool_coverage_ready"))
        or _clean_cockpit_display(lane.get("tool_coverage")).lower() == "registered"
    )
    smoke_ready = sum(
        1
        for lane in lanes
        if _exact_true(lane.get("smoke_coverage_ready"))
        or _clean_cockpit_display(lane.get("smoke_coverage")).lower() == "registered"
    )
    approval_required = sum(1 for lane in lanes if _exact_true(lane.get("approval_required")))
    auto_run_safe = lane_count - approval_required
    return (
        f"tools {tool_ready}/{lane_count}, "
        f"smokes {smoke_ready}/{lane_count}, "
        f"approval-gated {approval_required}, "
        f"auto-run safe {auto_run_safe}"
    )


def _safe_outbound_text(value: str) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _clean_untrusted_text(value, fallback=_UNREADABLE_OUTBOUND_TEXT))


def _normalized_lane_alias(value: str) -> str:
    text = _clean(value).lower().replace("&", " and ")
    tokens = [token for token in re.findall(r"[a-z0-9]+", text) if token not in _LANE_FILLER_WORDS]
    return "".join(tokens)


def _normalized_compact_lane_alias(value: str) -> str:
    candidate = _clean(value).lower()
    changed = True
    while changed and candidate:
        changed = False
        for prefix in _KOREAN_LANE_PREFIXES:
            if candidate.startswith(prefix) and len(candidate) > len(prefix):
                candidate = candidate[len(prefix):]
                changed = True
        for suffix in _KOREAN_LANE_SUFFIXES:
            if candidate.endswith(suffix) and len(candidate) > len(suffix):
                candidate = candidate[: -len(suffix)]
                changed = True
    return candidate


def _cockpit_lane_key_for_command(low: str, compact: str) -> str:
    english_key = ""
    tokens = set(re.findall(r"[a-z0-9]+", _clean(low).lower()))
    if tokens & _ENGLISH_LANE_MARKERS:
        english_key = _COCKPIT_LANE_ALIASES.get(_normalized_lane_alias(low)) or ""
    korean_key = ""
    compact_value = _clean(compact).lower()
    if compact_value in {"가드레일", "가드레일상태"}:
        return english_key
    if any(compact_value.startswith(prefix) for prefix in _KOREAN_LANE_PREFIXES) or any(
        marker in compact_value for marker in _KOREAN_LANE_SUFFIXES
    ):
        korean_key = _COCKPIT_LANE_ALIASES.get(_normalized_compact_lane_alias(compact_value)) or ""
    return english_key or korean_key


def _token_status() -> dict[str, Any]:
    raw = _clean(os.getenv("TELEGRAM_BOT_TOKEN", ""))
    configured = bool(raw)
    valid = configured and not _has_local_path(raw) and any(ch.isalnum() for ch in raw)
    return {
        "value": raw,
        "configured": configured,
        "valid": valid,
        "reason": "missing_token" if not configured else "invalid_token",
    }


def _owner_status() -> dict[str, Any]:
    raw = _clean(os.getenv("JARVIS_OWNER_TELEGRAM", ""))
    configured = bool(raw)
    valid = configured and not _has_local_path(raw) and any(ch.isalnum() for ch in raw)
    return {
        "value": raw,
        "configured": configured,
        "valid": valid,
        "reason": "missing_owner" if not configured else "invalid_owner",
    }


def _bot_token() -> str:
    status = _token_status()
    return status["value"] if status["valid"] else ""


def _owner_chat_id() -> str:
    status = _owner_status()
    return status["value"] if status["valid"] else ""


def _token_error() -> str:
    status = _token_status()
    if not status["configured"]:
        return "no token"
    return "invalid token"


def _telegram_ssl_context() -> ssl.SSLContext:
    """Return a cached, verified TLS context backed by certifi's CA bundle."""

    global _TELEGRAM_SSL_CONTEXT
    with _TELEGRAM_SSL_CONTEXT_LOCK:
        if _TELEGRAM_SSL_CONTEXT is None:
            context = ssl.create_default_context(cafile=certifi.where())
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            _TELEGRAM_SSL_CONTEXT = context
        return _TELEGRAM_SSL_CONTEXT


def _api_call(token: str, method: str, params: dict, timeout: float) -> dict:
    url = API_BASE.format(token=token, method=method)
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(
            req,
            timeout=timeout,
            context=_telegram_ssl_context(),
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        retry_after: int | None = None
        try:
            parsed = json.loads(exc.read(65536).decode("utf-8", errors="replace"))
            parameters = parsed.get("parameters") if isinstance(parsed, dict) else None
            raw_retry = parameters.get("retry_after") if isinstance(parameters, dict) else None
            if not isinstance(raw_retry, bool):
                retry_after = max(0, min(int(raw_retry), 86400)) if raw_retry is not None else None
        except (AttributeError, TypeError, ValueError, OverflowError, json.JSONDecodeError):
            retry_after = None
        result: dict[str, Any] = {
            "ok": False,
            "error": f"http_{int(exc.code)}",
            "http_status": int(exc.code),
        }
        if retry_after is not None:
            result["retry_after"] = retry_after
        return result
    except (urllib.error.URLError, TimeoutError, OSError):
        # No authoritative Telegram response means the request outcome is
        # unknown. Omit a boolean `ok` so delivery callers quarantine rather
        # than replaying a message that Telegram may already have accepted.
        return {"error": "network_outcome_unknown"}
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        # A malformed acknowledgement is also an unknown outcome, not a
        # definite rejection. Never expose the response body or retry blindly.
        return {"error": "malformed_response"}


MAX_VOICE_FILE_BYTES = 20 * 1024 * 1024  # Telegram bot-download cap is 20MB.
FILE_API_BASE = "https://api.telegram.org/file/bot{token}/{file_path}"


def _get_voice_file_id(message: dict[str, Any]) -> str:
    """Return a downloadable file_id for a voice/audio message, else ''."""
    media = message.get("voice") or message.get("audio") or {}
    if isinstance(media, dict):
        size = media.get("file_size")
        if isinstance(size, (int, float)) and size > MAX_VOICE_FILE_BYTES:
            return ""
        return _clean(media.get("file_id"))
    return ""


def _download_telegram_file(token: str, file_id: str, dest_dir: str) -> Path | None:
    """getFile + download a Telegram file to dest_dir. Returns the local path or None."""
    info = _api_call(token, "getFile", {"file_id": file_id}, timeout=20)
    if not info.get("ok"):
        return None
    file_path = str((info.get("result") or {}).get("file_path") or "").strip()
    if not file_path:
        return None
    url = FILE_API_BASE.format(token=token, file_path=urllib.parse.quote(file_path))
    suffix = Path(file_path).suffix or ".oga"
    dest = Path(dest_dir) / f"voice{suffix}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(
            req,
            timeout=60,
            context=_telegram_ssl_context(),
        ) as resp:
            dest.write_bytes(resp.read(MAX_VOICE_FILE_BYTES + 1))
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    if dest.stat().st_size > MAX_VOICE_FILE_BYTES:
        return None
    return dest


def _telegram_voice_setup_guidance() -> str:
    return (
        "Voice transcription isn't set up on the Mac. Install the `whisper` CLI "
        "(or set JARVIS_VOICE_WHISPER_MODEL_PATH) and try again."
    )


def _telegram_voice_recovery_guidance(stage: str) -> str:
    guidance = {
        "invalid_media": (
            "That voice note was too large or had no downloadable audio. "
            "Send a voice/audio file under 20MB, or type the command instead."
        ),
        "download": (
            "I couldn't download that voice note from Telegram. Check Telegram network access, "
            "then resend the voice note or type the command instead."
        ),
        "transcribe": (
            "I couldn't transcribe that voice note just now. Run `voice setup check`, "
            "then resend a clear voice note or type the command instead."
        ),
        "empty": (
            "I couldn't make out any words in that voice note. Resend a clearer voice note, "
            "or type the command instead."
        ),
    }
    return guidance.get(
        stage,
        "Voice-note processing is unavailable. Run `voice setup check`, then type the command instead.",
    )


def transcribe_voice_message(token: str, message: dict[str, Any]) -> tuple[str, str]:
    """Download + transcribe a Telegram voice/audio message.

    Returns (transcript, error). Exactly one is non-empty. Used so a voice memo is
    treated as an equivalent input channel to typed text (the transcript is then
    routed through the same approval-gated runtime).
    """
    import tempfile

    from jarvis_v2.tools import voice

    file_id = _get_voice_file_id(message)
    if not file_id:
        return "", _telegram_voice_recovery_guidance("invalid_media")
    transcriber = voice._audio_file_transcriber()
    if transcriber is None:
        return "", _telegram_voice_setup_guidance()
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-tg-voice-") as tmp:
            path = _download_telegram_file(token, file_id, tmp)
            if path is None:
                return "", _telegram_voice_recovery_guidance("download")
            transcript = (transcriber(path) or "").strip()
    except Exception:
        return "", _telegram_voice_recovery_guidance("transcribe")
    if not transcript:
        return "", _telegram_voice_recovery_guidance("empty")
    return transcript, ""


def _get_image_file_id(message: dict[str, Any]) -> str:
    """Return a downloadable file_id for a photo or image document, else ''."""
    photos = message.get("photo")
    if isinstance(photos, list) and photos:
        # Telegram sends multiple sizes; pick the largest within the download cap.
        sized = [p for p in photos if isinstance(p, dict) and p.get("file_id")]
        sized.sort(key=lambda p: (p.get("file_size") or 0, p.get("width") or 0))
        for p in reversed(sized):
            if (p.get("file_size") or 0) <= MAX_VOICE_FILE_BYTES:
                return _clean(p.get("file_id"))
        return ""
    doc = message.get("document")
    if isinstance(doc, dict) and str(doc.get("mime_type") or "").startswith("image/"):
        if (doc.get("file_size") or 0) > MAX_VOICE_FILE_BYTES:
            return ""
        return _clean(doc.get("file_id"))
    return ""


def _telegram_image_recovery_guidance(stage: str) -> str:
    guidance = {
        "invalid_media": (
            "I can only read text from a PNG/JPG photo or image file under 20MB. "
            "Resend a supported image, or type the text instead."
        ),
        "unavailable": (
            "On-device OCR isn't available on the Mac. Run `setup check` and `swift --version`, "
            "repair the macOS Swift/Vision setup, then resend the image."
        ),
        "download": (
            "I couldn't download that image from Telegram. Check Telegram network access, "
            "then resend a PNG/JPG image or type the text instead."
        ),
        "read": (
            "I couldn't read text from that image just now. Run `setup check`, confirm "
            "macOS Swift/Vision is available, then resend a clear PNG/JPG image or type the text instead."
        ),
        "empty": (
            "I didn't find any readable text in that image. Resend a clear, high-contrast "
            "PNG/JPG image, or type the text instead."
        ),
    }
    return guidance.get(
        stage,
        "Image OCR is unavailable. Run `setup check`, then type the text instead.",
    )


def extract_photo_text(token: str, message: dict[str, Any]) -> tuple[str, str]:
    """Download + OCR a Telegram photo/image-document. Returns (text, error).

    OCR output is content, not a command, so the caller surfaces the text back to
    the owner rather than routing it through the runtime.
    """
    import tempfile

    from jarvis_v2.tools import ocr

    file_id = _get_image_file_id(message)
    if not file_id:
        return "", _telegram_image_recovery_guidance("invalid_media")
    if not ocr.ocr_available():
        return "", _telegram_image_recovery_guidance("unavailable")
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-tg-img-") as tmp:
            path = _download_telegram_file(token, file_id, tmp)
            if path is None:
                return "", _telegram_image_recovery_guidance("download")
            text = ocr.extract_text_from_image(path)
    except Exception:
        return "", _telegram_image_recovery_guidance("read")
    if not text:
        return "", _telegram_image_recovery_guidance("empty")
    return text, ""


def get_updates(token: str, offset: int | None, timeout: int = LONG_POLL_SECONDS) -> list[dict]:
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    result = _api_call(token, "getUpdates", params, timeout=timeout + 10)
    if not result.get("ok"):
        return []
    return result.get("result", [])


def _text_chunks(text: str, *, limit: int = MAX_REPLY_CHARS) -> list[str]:
    body = _safe_outbound_text(text) or "Done."
    if len(body) <= limit:
        return [body]
    chunks: list[str] = []
    remaining = body
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < max(1, limit // 3):
            split_at = limit
        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:limit]
            split_at = limit
        chunks.append(chunk)
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks or ["Done."]


def _single_message_body(text: str) -> str:
    chunks = _text_chunks(text)
    if len(chunks) == 1:
        return chunks[0]
    return chunks[0][: MAX_REPLY_CHARS - 1] + "…"


def send_chat_action(chat_id: str, action: str = "typing") -> dict:
    token = _bot_token()
    if not token:
        return {"ok": False, "error": _token_error()}
    return _api_call(token, "sendChatAction", {"chat_id": chat_id, "action": action}, timeout=10)


def send_message(
    chat_id: str,
    text: str,
    reply_markup: dict[str, Any] | None = None,
    parse_mode: str | None = None,
) -> dict:
    token = _bot_token()
    if not token:
        return {"ok": False, "error": _token_error()}
    results: list[dict] = []
    for index, body in enumerate(_text_chunks(text)):
        params: dict[str, Any] = {"chat_id": chat_id, "text": body}
        if parse_mode:
            params["parse_mode"] = parse_mode
        if reply_markup and index == 0:
            params["reply_markup"] = json.dumps(reply_markup)
        results.append(_api_call(token, "sendMessage", params, timeout=15))
    if len(results) == 1:
        return results[0]
    return {"ok": all(result.get("ok") for result in results), "results": results}


def answer_callback_query(callback_query_id: str, text: str = "") -> dict:
    token = _bot_token()
    if not token:
        return {"ok": False, "error": _token_error()}
    params = {"callback_query_id": callback_query_id}
    if text:
        params["text"] = text[:200]
    return _api_call(token, "answerCallbackQuery", params, timeout=15)


def edit_message_text(
    chat_id: str,
    message_id: int,
    text: str,
    reply_markup: dict[str, Any] | None = None,
    parse_mode: str | None = None,
) -> dict:
    token = _bot_token()
    if not token:
        return {"ok": False, "error": _token_error()}
    body = _single_message_body(text)
    params: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": body}
    if parse_mode:
        params["parse_mode"] = parse_mode
    if reply_markup:
        params["reply_markup"] = json.dumps(reply_markup)
    return _api_call(token, "editMessageText", params, timeout=15)


def _approval_id_from_result(result: object) -> int | None:
    try:
        tool_results = getattr(result, "tool_results", []) or []
    except Exception:
        tool_results = []
    for tool_result in tool_results:
        try:
            metadata = getattr(tool_result, "metadata", None)
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue
        if not metadata.get("requires_confirmation"):
            continue
        approval_id = metadata.get("approval_id")
        try:
            parsed = int(approval_id)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed

    response = _result_response_text(result, default="", unreadable="")
    if response:
        # Some higher-level decision/dispatch paths return the normal approval
        # receipt as text instead of exposing ToolResult metadata. Only trust
        # the fallback when the reply clearly describes a queued safety receipt.
        for pattern in (
            r"Safety receipt:\s*queued as approval\s*#(\d+)",
            r"\bqueued as approval\s*#(\d+)",
        ):
            match = re.search(pattern, response, flags=re.IGNORECASE)
            if match:
                try:
                    parsed = int(match.group(1))
                except (TypeError, ValueError):
                    return None
                if parsed > 0:
                    return parsed
                return None
    try:
        metadata = getattr(result, "metadata", None)
    except Exception:
        metadata = None
    if not isinstance(metadata, dict):
        return None
    trace = metadata.get("runtime_trace")
    if not isinstance(trace, dict):
        return None
    for tool_result in trace.get("tool_results", []) or []:
        if not isinstance(tool_result, dict):
            continue
        tool_metadata = tool_result.get("metadata")
        if not isinstance(tool_metadata, dict):
            continue
        if not tool_metadata.get("requires_confirmation"):
            continue
        approval_id = tool_metadata.get("approval_id")
        try:
            parsed = int(approval_id)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _approval_keyboard(approval_id: int) -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Approve", "callback_data": f"approve:{approval_id}"},
                {"text": "❌ Deny", "callback_data": f"deny:{approval_id}"},
            ]
        ]
    }


def _runtime_error_reply(exc: Exception) -> str:
    return (
        "Jarvis hit an internal error while handling that Telegram command. "
        f"The outcome may be unknown; do not retry automatically. Run `execution health report` first. ({type(exc).__name__})"
    )


def _state_file() -> Path:
    if STATE_FILE is not None:
        return STATE_FILE
    raw = os.getenv("JARVIS_TELEGRAM_STATE", "").strip()
    try:
        account_home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except (KeyError, OSError, AttributeError, TypeError):
        raise telegram_offset_state.TelegramOffsetStateError("state_path_invalid") from None
    if raw == "~":
        return account_home
    if raw.startswith("~/"):
        return account_home / raw[2:]
    if raw.startswith("~"):
        raise telegram_offset_state.TelegramOffsetStateError("state_path_invalid")
    return Path(raw) if raw else account_home / ".jarvis_v3" / "telegram_control.json"


def _offset_identity(token: str | None = None, owner: str | None = None) -> tuple[str, str]:
    selected_token = token if token is not None else _bot_token()
    selected_owner = owner if owner is not None else _owner_chat_id()
    if not selected_token or not selected_owner:
        raise telegram_offset_state.TelegramOffsetStateError("state_identity_invalid")
    return selected_token, selected_owner


def _load_offset_snapshot(
    token: str | None = None,
    owner: str | None = None,
) -> telegram_offset_state.OffsetSnapshot:
    selected_token, selected_owner = _offset_identity(token, owner)
    return telegram_offset_state.load_offset(
        _state_file(),
        bot_token=selected_token,
        owner_id=selected_owner,
    )


def _load_offset(token: str | None = None, owner: str | None = None) -> int | None:
    """Load a bound offset; only a truly absent state file is fresh."""

    return _load_offset_snapshot(token, owner).offset


def _save_offset(
    offset: int,
    token: str | None = None,
    owner: str | None = None,
) -> None:
    """Monotonically publish an offset or raise a bounded persistence error."""

    selected_token, selected_owner = _offset_identity(token, owner)
    expected = telegram_offset_state.load_offset(
        _state_file(),
        bot_token=selected_token,
        owner_id=selected_owner,
    )
    _advance_offset(offset, expected, selected_token, selected_owner)


def _advance_offset(
    offset: int,
    expected: telegram_offset_state.OffsetSnapshot,
    token: str | None = None,
    owner: str | None = None,
) -> telegram_offset_state.OffsetSnapshot:
    """Advance only from the exact snapshot held before external work."""

    selected_token, selected_owner = _offset_identity(token, owner)
    return telegram_offset_state.advance_offset(
        _state_file(),
        bot_token=selected_token,
        owner_id=selected_owner,
        new_offset=offset,
        expected=expected,
    )


def _update_id(update: dict[str, Any]) -> int | None:
    if not isinstance(update, dict):
        return None
    value = update.get("update_id")
    if type(value) is not int or value < 0 or value > MAX_TRANSPORT_UPDATE_ID:
        return None
    return value


def _update_request_token(update_id: object) -> str | None:
    if type(update_id) is not int or update_id < 0 or update_id > MAX_TRANSPORT_UPDATE_ID:
        return None
    return f"telegram-update:v1:{update_id}"


def _telegram_message_id(result: object) -> int | None:
    if not isinstance(result, dict):
        return None
    payload = result.get("result")
    if not isinstance(payload, dict):
        return None
    value = payload.get("message_id")
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _reminder_delivery_disposition(result: object) -> tuple[str, str, int, int | None]:
    """Classify a Telegram receipt without retaining untrusted response text."""
    if isinstance(result, dict) and result.get("ok") is True:
        message_id = _telegram_message_id(result)
        if message_id is not None:
            return "accepted", "", 0, message_id
        return "uncertain", "missing_message_id", 0, None
    if not isinstance(result, dict) or result.get("ok") is not False:
        return "uncertain", "malformed_response", 0, None
    nested = result.get("results")
    if isinstance(nested, list) and any(
        isinstance(item, dict) and item.get("ok") is True for item in nested
    ):
        return "uncertain", "partial_acceptance", 0, None
    raw_error = str(result.get("error") or "telegram_rejected").strip().casefold()
    error_code = re.sub(r"[^a-z0-9_.:-]+", "_", raw_error)[:128] or "telegram_rejected"
    try:
        http_status = int(result.get("http_status"))
    except (TypeError, ValueError, OverflowError):
        http_status = 0
    try:
        retry_after = int(result.get("retry_after"))
    except (TypeError, ValueError, OverflowError):
        retry_after = 300
    retry_after = max(0, min(retry_after, 86_400))
    permanent_errors = {
        "missing_or_invalid_owner",
        "no token",
        "invalid token",
        "no_token",
        "invalid_token",
    }
    if raw_error in permanent_errors or http_status in {400, 401, 403, 404}:
        return "failed", error_code, 0, None
    return "rejected", error_code, retry_after, None


class TelegramCommandBridge:
    def __init__(
        self,
        *,
        runtime_factory: Callable[[], object] | None = None,
        send_func: Callable[[str, str, dict[str, Any] | None], dict] = send_message,
        fetch_func: Callable[[str, int | None, int], list[dict]] = get_updates,
        answer_callback_func: Callable[[str, str], dict] = answer_callback_query,
        edit_message_func: Callable[[str, int, str, dict[str, Any] | None], dict] = edit_message_text,
        chat_action_func: Callable[[str, str], dict] | None = send_chat_action,
    ):
        self.runtime_factory = runtime_factory
        self.send_func = send_func
        self.fetch_func = fetch_func
        self.answer_callback_func = answer_callback_func
        self.edit_message_func = edit_message_func
        self.chat_action_func = chat_action_func
        self._runtime = None

    def _runtime_instance(self):
        if self._runtime is None:
            if self.runtime_factory is None:
                from jarvis_v2.agent.runtime import JarvisRuntime
                self._runtime = JarvisRuntime()
            else:
                self._runtime = self.runtime_factory()
        return self._runtime

    def _deliver_due_reminders(self) -> None:
        """Claim due reminders and send them through the validated owner channel."""
        owner = _owner_chat_id()
        if not owner:
            return
        try:
            from jarvis_v2.automations.reminders import (
                MAX_DELIVERY_ATTEMPTS,
                _display_message,
                REMINDER_DELIVERY_BATCH_LIMIT,
                begin_delivery,
                claim_due_result,
                finish_delivery,
                record_delivery_health,
            )

            counts = {
                "claimed_count": 0,
                "started_count": 0,
                "durably_finalized_count": 0,
                "api_accepted_count": 0,
                "rejected_count": 0,
                "failed_count": 0,
                "uncertain_count": 0,
                "pre_send_blocked_count": 0,
                "finalization_failed_count": 0,
            }
            claim_result = claim_due_result(
                owner_chat_id=owner,
                limit=REMINDER_DELIVERY_BATCH_LIMIT,
            )
            if not claim_result.store_available:
                record_delivery_health(
                    status=(
                        "store_unavailable"
                        if claim_result.status == "store_unavailable"
                        else "outcome_unknown"
                    ),
                    reason=claim_result.reason or claim_result.status,
                    **counts,
                )
                return
            claimed_items = list(claim_result.items)
            counts["claimed_count"] = len(claimed_items)
            saturated = len(claimed_items) == REMINDER_DELIVERY_BATCH_LIMIT
            for item in claimed_items:
                reminder_id = str(item.get("id") or "")
                token = str(item.get("token") or "")
                if not begin_delivery(reminder_id, token):
                    counts["pre_send_blocked_count"] += 1
                    continue
                counts["started_count"] += 1
                message = _display_message(item.get("message")) or "⏰ Reminder!"
                try:
                    result = self.send_func(owner, _safe_outbound_text(f"⏰ {message}"), None)
                except Exception:
                    finalized = finish_delivery(
                        reminder_id,
                        token,
                        "uncertain",
                        error_code="send_exception",
                    )
                    if finalized:
                        counts["durably_finalized_count"] += 1
                        counts["uncertain_count"] += 1
                    else:
                        counts["finalization_failed_count"] += 1
                    continue
                disposition, error_code, retry_after, message_id = _reminder_delivery_disposition(result)
                if disposition == "accepted":
                    counts["api_accepted_count"] += 1
                finalized = finish_delivery(
                    reminder_id,
                    token,
                    disposition,
                    error_code=error_code,
                    retry_after=retry_after,
                    telegram_message_id=message_id,
                )
                if not finalized:
                    counts["finalization_failed_count"] += 1
                    continue
                counts["durably_finalized_count"] += 1
                durable_disposition = disposition
                if (
                    disposition == "rejected"
                    and int(item.get("attempt_count") or 0) + 1 >= MAX_DELIVERY_ATTEMPTS
                ):
                    durable_disposition = "failed"
                count_key = {
                    "rejected": "rejected_count",
                    "failed": "failed_count",
                    "uncertain": "uncertain_count",
                }.get(durable_disposition)
                if count_key:
                    counts[count_key] += 1

            if counts["finalization_failed_count"]:
                health_status = "outcome_unknown"
                health_reason = "finalization_failed"
            elif counts["pre_send_blocked_count"]:
                health_status = "degraded"
                health_reason = "pre_send_blocked"
            elif counts["uncertain_count"]:
                health_status = "degraded"
                health_reason = "uncertain_delivery"
            elif counts["failed_count"]:
                health_status = "degraded"
                health_reason = "permanent_failure"
            elif counts["rejected_count"]:
                health_status = "degraded"
                health_reason = "retry_scheduled"
            elif counts["claimed_count"]:
                health_status = "healthy_batch"
                health_reason = ""
            else:
                health_status = "healthy_idle"
                health_reason = ""
            record_delivery_health(
                status=health_status,
                reason=health_reason,
                saturated=saturated,
                **counts,
            )
        except Exception:
            pass

    def process_once(self, *, poll_timeout: int = LONG_POLL_SECONDS) -> int:
        token = _bot_token()
        owner = _owner_chat_id()
        if not token or not owner:
            return 0

        offset_snapshot = _load_offset_snapshot(token, owner)
        offset = offset_snapshot.offset
        self._deliver_due_reminders()
        try:
            updates = list(self.fetch_func(token, offset, poll_timeout))
        except Exception:
            return 0

        # Fresh install: drain backlog without executing so stale setup messages
        # (e.g. the "hi" you sent to capture your chat id) are never run.
        if offset is None:
            update_ids = [update_id for update in updates if (update_id := _update_id(update)) is not None]
            if update_ids:
                _advance_offset(max(update_ids) + 1, offset_snapshot, token, owner)
            return 0

        processed = 0
        max_update_id = offset - 1
        for update in updates:
            update_id = _update_id(update)
            if update_id is None:
                continue
            request_token = _update_request_token(update_id)
            if request_token is None:
                continue
            max_update_id = max(max_update_id, update_id)
            callback = update.get("callback_query")
            if isinstance(callback, dict):
                if self._process_callback(callback, owner):
                    processed += 1
                continue
            raw_message = update.get("message") or update.get("edited_message") or {}
            message = raw_message if isinstance(raw_message, dict) else {}
            raw_chat = message.get("chat") or {}
            chat = raw_chat if isinstance(raw_chat, dict) else {}
            chat_id = _clean_untrusted_text(chat.get("id"))
            text = _clean_untrusted_text(message.get("text"))
            # Owner lock: ignore anyone who is not the configured owner.
            if chat_id != owner:
                continue
            # Voice memo? Transcribe it and treat the transcript as a typed command,
            # so the normal approval-gated runtime still governs any side effect.
            if not text and (message.get("voice") or message.get("audio")):
                raw_transcript, raw_voice_error = transcribe_voice_message(token, message)
                voice_error = _clean_untrusted_text(
                    raw_voice_error,
                    fallback=_telegram_voice_recovery_guidance("transcribe"),
                )
                if voice_error:
                    try:
                        self.send_func(owner, _safe_outbound_text(voice_error), None)
                    except Exception:
                        pass
                    processed += 1
                    continue
                transcript = _clean_untrusted_text(raw_transcript)
                if not transcript:
                    try:
                        self.send_func(
                            owner,
                            _telegram_voice_recovery_guidance("empty"),
                            None,
                        )
                    except Exception:
                        pass
                    processed += 1
                    continue
                try:
                    self.send_func(owner, _safe_outbound_text(f"🎙 Heard: {transcript}"), None)
                except Exception:
                    pass
                text = transcript
            # Photo/image document? OCR it locally and surface the text. OCR output is
            # content, not a command, so it is shown back rather than routed/executed.
            if not text and (message.get("photo") or message.get("document")):
                raw_ocr_text, raw_ocr_error = extract_photo_text(token, message)
                ocr_error = _clean_untrusted_text(
                    raw_ocr_error,
                    fallback=_telegram_image_recovery_guidance("read"),
                )
                ocr_text = _clean_untrusted_text(raw_ocr_text)
                body = ocr_error or (
                    f"📄 Text from the image:\n{ocr_text}"
                    if ocr_text
                    else _telegram_image_recovery_guidance("empty")
                )
                caption = _clean_untrusted_text(message.get("caption"))
                reply_text = f"🖼 caption: {caption}\n{body}" if caption and ocr_text and not ocr_error else body
                try:
                    self.send_func(owner, _safe_outbound_text(reply_text), None)
                except Exception:
                    pass
                processed += 1
                continue
            if not text:
                continue
            reply, reply_markup = self._handle_command_with_typing(
                owner,
                text,
                request_token=request_token,
            )
            if reply:
                try:
                    self.send_func(owner, _safe_outbound_text(reply), reply_markup)
                except Exception:
                    pass
            processed += 1

        if max_update_id >= offset:
            _advance_offset(max_update_id + 1, offset_snapshot, token, owner)
        return processed

    def _process_callback(self, callback: dict[str, Any], owner: str) -> bool:
        if not isinstance(callback, dict):
            return False
        raw_from_user = callback.get("from") or {}
        from_user = raw_from_user if isinstance(raw_from_user, dict) else {}
        if _clean_untrusted_text(from_user.get("id")) != owner:
            return False
        callback_id = _clean_untrusted_text(callback.get("id"))
        data = _clean_untrusted_text(callback.get("data"))
        message = callback.get("message")
        if not isinstance(message, dict):
            return False
        chat = message.get("chat")
        if not isinstance(chat, dict):
            return False
        chat_id = _clean_untrusted_text(chat.get("id"))
        if not chat_id:
            return False
        if chat_id != owner:
            return False
        try:
            message_id = int(message.get("message_id"))
        except Exception:
            message_id = 0
        if not data:
            return False
        reply = self._handle_approval_callback(data)
        if not reply:
            return False
        try:
            if callback_id:
                self.answer_callback_func(callback_id, _safe_outbound_text(reply).splitlines()[0][:200])
        except Exception:
            pass
        try:
            if message_id:
                self.edit_message_func(chat_id, message_id, _safe_outbound_text(reply), None)
            else:
                self.send_func(owner, _safe_outbound_text(reply), None)
        except Exception:
            pass
        return True

    def _handle_approval_callback(self, data: str) -> str:
        action, sep, raw_id = data.partition(":")
        if sep != ":" or action not in {"approve", "deny"}:
            return "Unsupported approval action. Open the approval in Jarvis and use a fresh packet's current Approve or Deny button."
        if not raw_id.isdigit():
            return "Invalid approval action. Open the approval in Jarvis and use a fresh packet's current Approve or Deny button."
        approval_id = int(raw_id)
        if approval_id <= 0:
            return "Invalid approval action. Open the approval in Jarvis and use a fresh packet's current Approve or Deny button."
        runtime = self._runtime_instance()
        if action == "deny":
            result = runtime.handle(f"dismiss approval {approval_id}")
            return _result_response_text(result, default=f"Dismissed approval #{approval_id}.")
        runtime.handle(f"approval packet {approval_id}")
        result = runtime.handle(f"approve approval {approval_id}")
        return _result_response_text(result, default=f"Approved approval #{approval_id}.")

    def _handle_command_with_typing(
        self,
        owner: str,
        text: str,
        *,
        request_token: str,
    ) -> tuple[str, dict[str, Any] | None]:
        """Run the command while keeping Telegram's 'typing…' indicator alive, so
        the owner sees Jarvis working instead of wondering if it is offline."""
        stop = threading.Event()
        thread: threading.Thread | None = None
        if self.chat_action_func is not None:
            # First ping is synchronous so "typing…" is guaranteed to show before
            # the reply; the thread only handles later refreshes.
            try:
                self.chat_action_func(owner, "typing")
            except Exception:
                pass

            def _keepalive() -> None:
                while not stop.wait(TYPING_REFRESH_SECONDS):
                    try:
                        self.chat_action_func(owner, "typing")
                    except Exception:
                        pass

            thread = threading.Thread(target=_keepalive, name="JarvisTelegramTyping", daemon=True)
            thread.start()
        try:
            return self._handle_command(text, request_token=request_token)
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=1.0)

    def _capability_cockpit_result(self):
        runtime = self._runtime_instance()
        registry = getattr(runtime, "registry", None)
        if registry is not None:
            tool = registry.get("capability_cockpit")
            handler = getattr(tool, "handler", None)
            if callable(handler):
                return handler({})

        class _EmptyStore:
            def recent_tool_runs(self, *, limit: int = 400):  # noqa: ARG002 - cockpit fallback API
                return []

        from jarvis_v2.tools.cockpit import make_capability_cockpit_tool

        return make_capability_cockpit_tool(_EmptyStore(), lambda: [])({})

    def _format_cockpit_text(self) -> str:
        result = self._capability_cockpit_result()
        return str(getattr(result, "output", "") or "Cockpit unavailable.")

    def _format_live_test_matrix(self) -> str:
        """Compact operator-facing matrix for the frozen live channel proof pass."""
        return (
            "Pending live-proof matrix:\n"
            "- KakaoTalk send: approval card shown, Hangul preserved, delivery pass/fail.\n"
            "- Telegram send: approval card shown, target opens, delivery pass/fail.\n"
            "- Instagram DM: approval card shown, thread verified, delivery pass/fail.\n"
            "- iMessage send: approval card shown, recipient verified, delivery pass/fail.\n"
            "- Phone: exact phone target + mode shown on the approval card; approved run requests the call; recipient confirms ringing/connection.\n"
            "- FaceTime: exact resolved target + audio/video mode shown on the approval card; approved run requests the call; recipient confirms ringing/connection.\n"
            "- KakaoTalk call: exact visible recipient + audio/video mode shown on the approval card; recipient confirms ringing/connection.\n"
            "- Telegram call: exact visible recipient + audio/video mode shown on the approval card; recipient confirms ringing/connection.\n"
            "- Instagram call: exact visible recipient + audio/video mode shown on the approval card; recipient confirms ringing/connection.\n\n"
            "Call proof gate (all fields are required):\n"
            "- approval binding: the approval card's channel, target, and effective mode match the request; do not approve a mismatch.\n"
            "- execution evidence: record the bounded approval ID and tool-run ID after the approved rerun.\n"
            "- placement outcome: call_requested / known_not_started / outcome_unknown. `call_requested` alone is not live proof.\n"
            "- recipient confirmation: ringing / connected / not_received / not_checked. Only ringing or connected closes that channel's live proof.\n"
            "- outcome-unknown rule: inspect recent call state and ask the recipient; never retry automatically. A retry needs a fresh request and approval.\n\n"
            "Report format:\n"
            "- channel: KakaoTalk / Telegram / Instagram / iMessage / phone / FaceTime\n"
            "- result: pass / fail / blocked\n"
            "- approval card shown: yes / no\n"
            "- approved target + mode matched: yes / no / n/a\n"
            "- bounded approval ID + tool-run ID: numbers only, or n/a\n"
            "- call placement outcome: call_requested / known_not_started / outcome_unknown / n/a\n"
            "- recipient confirmation: ringing / connected / not_received / not_checked / n/a\n"
            "- last visible stage/error: short text only\n"
            "- do not include secrets, tokens, full handles, or message content.\n\n"
            "This is read-only: it does not approve, send, call, run live_check, edit frozen files, "
            "or claim live proof done."
        )

    def _format_conversation_research_proof_matrix(self) -> str:
        """Compact operator-facing matrix for Conversation & Research proof."""
        return (
            "Pending conversation-research proof matrix:\n"
            "- Research answer: `research <topic>` returns current sourced answers; nonsense queries fail cleanly.\n"
            "- Wiki/web fallback: `who is` / `what is` uses Wikipedia when right and full web when needed.\n"
            "- Ordinary chat: common chat uses the model path grounded in memory/profile, not template fallback.\n"
            "- Mixed conversation: 10+ turns include chat, research, calendar, tasks, and one utility/tool lookup.\n"
            "- Recovery check: if blocked, report whether model, planner, research connector, network, calendar, tasks, or live_check was unavailable.\n\n"
            "Report format:\n"
            "- area: research / wiki-fallback / chat-model / mixed-conversation / recovery\n"
            "- result: pass / fail / blocked\n"
            "- source count: number only, or n/a\n"
            "- chat p95 ms: number only, or n/a\n"
            "- model path: model / fallback / n/a\n"
            "- last visible stage/error: short text only\n"
            "- optional live_check row: research / mixed conversation / model routing status\n\n"
            "Privacy boundary: do not include transcript contents, event names, task text, email text, "
            "raw source excerpts, private query text, full prompts, source URLs with private terms, tokens, "
            "secrets, local paths, screenshots with private text, or full model output.\n\n"
            "This is read-only: it does not run research, fetch pages, call the model, run a conversation, "
            "access accounts, approve, send messages, run live_check, or claim conversation-research proof done."
        )

    def _format_mixed_conversation_proof_matrix(self) -> str:
        """Compact operator-facing matrix for the mixed-conversation latency proof."""
        return (
            "Pending mixed-conversation proof matrix:\n"
            "- 10+ turn run: include ordinary chat plus research, calendar, tasks, and one utility/tool lookup.\n"
            "- Coherence check: confirm Jarvis keeps context, does not fabricate, and answers naturally.\n"
            "- Latency check: confirm chat p95 is <= 8000ms and each tool-routed turn stays responsive.\n"
            "- Model path check: confirm ordinary chat uses the model path, not template fallback.\n"
            "- Recovery check: if blocked, report whether model, planner, calendar, network, or live_check was unavailable.\n\n"
            "Report format:\n"
            "- area: turn-mix / coherence / latency / model-path / recovery\n"
            "- result: pass / fail / blocked\n"
            "- turn count: number only\n"
            "- chat p95 ms: number only, or n/a\n"
            "- slowest stage/error: short text only\n"
            "- optional live_check row: mixed conversation\n\n"
            "Privacy boundary: do not include transcript contents, calendar event titles, task text, email text, "
            "contact names, local paths, tokens, secrets, screenshots with private text, or full model prompts.\n\n"
            "This is read-only: it does not run a conversation, call the model, change tuning knobs, execute tools, "
            "run live_check, access accounts, approve, send messages, or claim the mixed-conversation proof done."
        )

    def _format_voice_proof_matrix(self) -> str:
        """Compact operator-facing matrix for voice end-to-end live proof results."""
        return (
            "Pending voice-proof matrix:\n"
            "- Telegram voice note: phone voice note transcribes, echoes the transcript, and routes only after confirmation/approval.\n"
            "- Local push-to-talk: mic permission, record, transcribe, runtime reply, and cleanup are live-proven on the Mac.\n"
            "- Spoken reply: `talk.py --speak` produces the expected audible reply without changing command routing.\n"
            "- Korean voice: Korean or mixed Korean/English speech transcribes acceptably and preserves meaning before routing.\n"
            "- Recovery check: if blocked, report whether mic permission, ffmpeg, Whisper, Telegram file download, runtime, or speech output was unavailable.\n\n"
            "Report format:\n"
            "- area: telegram-voice / local-talk / speak / korean-transcription / recovery\n"
            "- result: pass / fail / blocked\n"
            "- language: en / ko / mixed / n/a\n"
            "- approval card shown: yes / no / n/a\n"
            "- last visible stage/error: short text only\n"
            "- optional live_check row: telegram voice / local talk / voice setup\n\n"
            "Privacy boundary: do not include transcript contents, contact names, message bodies, audio file ids, "
            "audio file paths, chat ids, tokens, local paths, screenshots with private text, or full Telegram message content.\n\n"
            "This is read-only: it does not access the microphone, record audio, download voice files, transcribe audio, "
            "speak aloud, route commands, approve, send messages, run live_check, or claim the voice proof done."
        )

    def _format_personal_proof_matrix(self) -> str:
        """Compact operator-facing matrix for personal-integration live proof results."""
        return (
            "Pending personal-proof matrix:\n"
            "- Calendar write cycle: create, update, and delete one test event through approval.\n"
            "- Email read/search: prove a real mailbox search without reporting message contents.\n"
            "- Email send: approve one send to a operator-controlled address without reporting subject/body.\n"
            "- Reminder delivery: prove Jarvis's local SQLite + owner Telegram timed reminder path.\n"
            "- Contacts lookup: prove required real-person lookup without reporting full handles.\n\n"
            "Report format:\n"
            "- area: calendar / email-read / email-send / reminder / contacts\n"
            "- result: pass / fail / blocked\n"
            "- approval card shown: yes / no / n/a\n"
            "- last visible stage/error: short text only\n"
            "- optional audit row id: #...\n\n"
            "Privacy boundary: do not include email contents, subject/body, full addresses, phone numbers, "
            "contact handles, reminder text beyond 'test reminder', tokens, secrets, or screenshots with "
            "private text.\n\n"
            "This is read-only: it does not access accounts, read email, create/update/delete calendar events, "
            "send email, schedule reminders, read Contacts, approve, run live_check, or claim personal proof done."
        )

    def _format_scheduler_proof_matrix(self) -> str:
        """Compact operator-facing matrix for scheduled-job live proof results."""
        return (
            "Pending scheduler-proof matrix:\n"
            "- Morning Brief delivery: confirm the real Telegram brief arrives on schedule.\n"
            "- Scheduler freshness: confirm enabled jobs have recent last_run/next_run state.\n"
            "- 7-day streak: confirm all 8 expected jobs record 7 consecutive complete non-failed days.\n"
            "- Silent-failure check: report any missed due time, stale last_run, failed event, or daemon downtime.\n"
            "- Recovery check: if blocked, report whether the scheduler daemon, Telegram control, or network was unavailable.\n\n"
            "Report format:\n"
            "- area: morning-brief / freshness / 7-day-streak / silent-failure / recovery\n"
            "- result: pass / fail / blocked\n"
            "- date range: YYYY-MM-DD..YYYY-MM-DD\n"
            "- enabled jobs: count only\n"
            "- last visible stage/error: short text only\n"
            "- optional audit row id or live_check row: #... / scheduled jobs / jobs 7-day proof\n\n"
            "Privacy boundary: do not include calendar details, email subjects, reminder text, contact names, "
            "full job payloads, tokens, secrets, local paths, screenshots with private text, or full Telegram message content.\n\n"
            "This is read-only: it does not run jobs, send Telegram messages, restart daemons, edit launchd, "
            "approve, run live_check, access accounts, or claim scheduler proof done."
        )

    def _format_daily_value_proof_matrix(self) -> str:
        """Compact operator-facing matrix for the daily-value live proof."""
        return (
            "Pending daily-value proof matrix:\n"
            "- Morning Brief usefulness: confirm the real owner-phone brief includes weather, calendar, reminders, and news status.\n"
            "- Delivery cadence: confirm the scheduled brief arrives once for the day and does not duplicate.\n"
            "- On-demand retry: confirm the owner-triggered brief path works once when the operator asks for it.\n"
            "- Brief status: confirm `brief status` shows freshness, last run, next run, and visible failure stage if blocked.\n"
            "- Recovery check: if a section is unavailable, confirm the brief/status text names the recovery path without exposing content.\n\n"
            "Report format:\n"
            "- area: usefulness / cadence / on-demand / status / recovery\n"
            "- result: pass / fail / blocked\n"
            "- date: YYYY-MM-DD\n"
            "- sent to owner: yes / no / n/a\n"
            "- unavailable sections: count only, or n/a\n"
            "- last visible stage/error: short text only\n"
            "- optional live_check row: daily brief / scheduled jobs / jobs 7-day proof\n\n"
            "Privacy boundary: do not include full brief text, calendar titles, email subjects, reminder text, "
            "contact names, locations, chat ids, tokens, secrets, local paths, screenshots with private text, "
            "or full Telegram message content.\n\n"
            "This is read-only: it does not compose or send a brief, run scheduled jobs, call APIs, access accounts, "
            "approve, run live_check, or claim daily-value proof done."
        )

    def _format_approval_proof_matrix(self) -> str:
        """Compact operator-facing matrix for owner-phone approval-flow live proof results."""
        return (
            "Pending approval-proof matrix:\n"
            "- Risky action prompt: confirm a real risky request shows Telegram approve/deny buttons first.\n"
            "- Approval last-look: confirm the approval packet clearly names the exact stored action.\n"
            "- Owner approve path: confirm owner ✅ executes only the matching pending approval once.\n"
            "- Owner deny path: confirm owner ❌ dismisses without execution.\n"
            "- Non-owner guard: confirm another Telegram user/chat cannot approve or dismiss.\n"
            "- Recovery check: if blocked, report whether the bot, daemon, approval queue, or target tool was unavailable.\n\n"
            "Report format:\n"
            "- area: prompt / last-look / approve / deny / non-owner / recovery\n"
            "- result: pass / fail / blocked\n"
            "- approval id: number only, or n/a\n"
            "- callback shown: approve / deny / both / none\n"
            "- last visible stage/error: short text only\n"
            "- optional audit row id or live_check row: #... / phone approvals\n\n"
            "Privacy boundary: do not include message contents, full planned args, contact handles, email subjects, "
            "calendar details, local paths, tokens, secrets, screenshots with private text, or full Telegram message content.\n\n"
            "This is read-only: it does not create approvals, approve, dismiss, execute tools, send messages, "
            "restart daemons, run live_check, access accounts, or claim approval proof done."
        )

    def _format_phone_control_proof_matrix(self) -> str:
        """Compact operator-facing matrix for phone-control shortcut live proof results."""
        return (
            "Pending phone-control proof matrix:\n"
            "- What broke: confirm recent failures and attention lanes are visible without exposing private run payloads.\n"
            "- Cockpit summary: confirm capability lanes, health, risk, and the trust checklist stay accurate.\n"
            "- Cockpit attention: confirm failed lanes and approval-held review are separated clearly.\n"
            "- Approvals lane: confirm pending approvals are visible with review commands but are not executed.\n"
            "- Owner-only guard: confirm non-owner chats cannot inspect owner-only phone control state.\n"
            "- Recovery check: if blocked, report whether Telegram control, capability cockpit, audit store, or approval queue was unavailable.\n\n"
            "Report format:\n"
            "- area: what-broke / cockpit-summary / attention / approvals / owner-guard / recovery\n"
            "- result: pass / fail / blocked\n"
            "- approval-held shown: yes / no / n/a\n"
            "- failure lane shown: yes / no / n/a\n"
            "- last visible stage/error: short text only\n"
            "- optional live_check row: phone control / channel health / phone approvals\n\n"
            "Privacy boundary: do not include message contents, planned args, contact handles, email subjects, "
            "calendar details, run payloads, chat ids, callback data, tokens, secrets, local paths, screenshots "
            "with private text, or full Telegram message content.\n\n"
            "This is read-only: it does not inspect private payloads, create approvals, approve, dismiss, execute tools, "
            "send messages, restart daemons, run live_check, access accounts, or claim phone-control proof done."
        )

    def _format_reboot_proof_matrix(self) -> str:
        """Compact operator-facing matrix for reboot and daemon recovery proof results."""
        return (
            "Pending reboot-proof matrix:\n"
            "- Post-reboot status: confirm Telegram control, scheduler, and status dashboard are reachable after Mac reboot.\n"
            "- LaunchAgent install: confirm each expected LaunchAgent is loaded/enabled without pasting private plist paths.\n"
            "- Scheduler freshness: confirm scheduled jobs resume and last_run/next_run do not go stale after reboot.\n"
            "- Telegram control: confirm owner-only phone commands still answer after reboot.\n"
            "- Network recovery: confirm daemons recover after network loss without silent failure.\n"
            "- Recovery check: if blocked, report whether launchd, Telegram control, scheduler, status server, or network was unavailable.\n\n"
            "Report format:\n"
            "- area: post-reboot / launchagent / scheduler / telegram-control / network-recovery / recovery\n"
            "- result: pass / fail / blocked\n"
            "- daemon: telegram-control / scheduler / status-server / all / n/a\n"
            "- reboot observed: yes / no / n/a\n"
            "- network-loss observed: yes / no / n/a\n"
            "- last visible stage/error: short text only\n"
            "- optional live_check row: daemon startup / scheduled jobs / phone control / channel health\n\n"
            "Privacy boundary: do not include plist paths, LaunchAgent file contents, logs with private text, "
            "tokens, secrets, chat ids, local paths, calendar/email/reminder details, screenshots with private text, "
            "or full Telegram message content.\n\n"
            "This is read-only: it does not run launchctl, install or edit LaunchAgents, restart daemons, stop daemons, "
            "change network state, send messages, run live_check, access accounts, or claim reboot proof done."
        )

    def _format_error_guidance_proof_matrix(self) -> str:
        """Compact operator-facing matrix for user-visible recovery guidance proof results."""
        return (
            "Pending error-guidance proof matrix:\n"
            "- Calendar/Gmail setup: auth, credential, and network failures name the exact setup or reauth step.\n"
            "- Reminders: local reminder store, owner Telegram config, and due-delivery failures name the fix.\n"
            "- Voice/Whisper: missing CLI/model, ffmpeg, permission, and transcription setup failures name the fix.\n"
            "- Model/Ollama: unavailable local model or fallback paths name the status/setup command to run next.\n"
            "- Generic connector recovery: service/network failures avoid raw tracebacks and name a bounded retry/setup path.\n"
            "- Recovery check: if blocked, report whether the message lacked a fix, leaked raw internals, or named the wrong fix.\n\n"
            "Report format:\n"
            "- area: calendar / gmail / reminders / voice / model / connector / recovery\n"
            "- result: pass / fail / blocked\n"
            "- fix named: yes / no / wrong\n"
            "- raw internals leaked: yes / no\n"
            "- last visible stage/error: short text only\n"
            "- optional live_check row: error guidance\n\n"
            "Privacy boundary: do not include email subjects, calendar event titles, reminder text, transcripts, "
            "contact names, tokens, secrets, local paths, full tracebacks, screenshots with private text, or full Telegram message content.\n\n"
            "This is read-only: it does not access accounts, trigger failing operations, call models, record audio, "
            "transcribe audio, run live_check, approve, send messages, restart daemons, or claim error-guidance proof done."
        )

    def _format_next_cockpit_action(self) -> str:
        """Summarize the cockpit's staged command queue without running it."""
        result = self._capability_cockpit_result()
        metadata = getattr(result, "metadata", {}) or {}
        raw_queue = metadata.get("next_command_queue")
        queue = raw_queue if isinstance(raw_queue, list) else []
        if not queue:
            command = _clean_cockpit_display(metadata.get("next_command"))
            if command:
                queue = [{"command": command}]
        if not queue:
            return (
                "Next cockpit action: nothing is queued right now.\n"
                "Run cockpit for the full read-only capability view."
            )

        lines = ["Next cockpit action:"]
        for index, raw_entry in enumerate(queue[:3], start=1):
            entry = raw_entry if isinstance(raw_entry, dict) else {}
            command = _clean_cockpit_display(entry.get("command"))
            if not command:
                continue
            lane = _clean_cockpit_display(entry.get("lane_title") or entry.get("lane") or entry.get("key"))
            kind = _clean_cockpit_display(entry.get("kind") or entry.get("next_command_kind") or "suggested")
            status = _clean_cockpit_display(entry.get("status"))
            details = [value for value in (lane, kind, status) if value]
            detail_text = f" ({'; '.join(details)})" if details else ""
            lines.append(f"{index}. {command}{detail_text}")
            failure = _clean_cockpit_display(entry.get("last_failure"))
            failure_kind = _clean_cockpit_display(entry.get("last_failure_kind"))
            if failure:
                suffix = f" ({failure_kind})" if failure_kind else ""
                lines.append(f"   last failure: {failure}{suffix}")
            approval_hold = _clean_cockpit_display(entry.get("last_approval_hold"))
            if approval_hold:
                commands = entry.get("approval_next_commands")
                if isinstance(commands, list):
                    clean_commands = [_clean_cockpit_display(item) for item in commands[:2]]
                    command_text = ", ".join(item for item in clean_commands if item)
                else:
                    command_text = ""
                next_text = f" | review: {command_text}" if command_text else ""
                lines.append(f"   approval hold: {approval_hold}{next_text}")
        if len(lines) == 1:
            lines.append("1. Run cockpit for the full read-only capability view.")
        lines.append("This is read-only: it does not approve, send, call, or execute the command.")
        return "\n".join(lines)

    def _format_cockpit_attention(self) -> str:
        """Summarize cockpit lanes that need attention without running commands."""
        result = self._capability_cockpit_result()
        metadata = getattr(result, "metadata", {}) or {}
        raw_lanes = metadata.get("capability_lanes")
        lanes = raw_lanes if isinstance(raw_lanes, list) else []
        attention: list[dict[str, Any]] = []
        approval_held: list[dict[str, Any]] = []
        for raw_lane in lanes:
            lane = raw_lane if isinstance(raw_lane, dict) else {}
            status = _clean_cockpit_display(lane.get("status")).lower()
            reasons = lane.get("attention_reasons")
            has_reasons = isinstance(reasons, list) and any(
                _clean_cockpit_display(item) for item in reasons
            )
            is_approval_held = status == "awaiting approval" or bool(
                _clean_cockpit_display(lane.get("last_approval_hold"))
            )
            if is_approval_held:
                approval_held.append(lane)
            elif status in {"attention", "missing"} or has_reasons:
                attention.append(lane)

        if not attention and not approval_held:
            return (
                "Cockpit attention: no attention lanes right now.\n"
                "Run cockpit for the full read-only capability view."
            )

        lines = ["Cockpit attention:"]
        if attention:
            lines.append("Needs attention:")
            for index, lane in enumerate(attention[:5], start=1):
                lines.extend(self._format_cockpit_attention_lane(index, lane))
        if approval_held:
            lines.append("Approval-held review:")
            lines.append(
                "Approval-held means Jarvis is waiting for owner review; it is not counted as a tool failure."
            )
            for index, lane in enumerate(approval_held[:5], start=1):
                lines.extend(self._format_cockpit_attention_lane(index, lane))
        lines.append("This is read-only: it does not approve, send, call, or execute the command.")
        return "\n".join(lines)

    def _format_cockpit_attention_lane(self, index: int, lane: dict[str, Any]) -> list[str]:
        """Format a cockpit attention lane for read-only owner-phone replies."""
        lines: list[str] = []
        title = _clean_cockpit_display(
            lane.get("title") or lane.get("lane_title") or lane.get("key") or "Capability"
        )
        status = _clean_cockpit_display(lane.get("status") or "attention")
        command = _clean_cockpit_display(lane.get("next_command") or lane.get("diagnostic") or "cockpit")
        lines.append(f"{index}. {title}: {status} — next: {command}")
        reasons = lane.get("attention_reasons")
        if isinstance(reasons, list):
            clean_reasons = [_clean_cockpit_display(reason) for reason in reasons]
            clean_reasons = [reason for reason in clean_reasons if reason]
            if clean_reasons:
                lines.append(f"   reason: {'; '.join(clean_reasons[:2])}")
        failure = _clean_cockpit_display(lane.get("last_failure"))
        failure_kind = _clean_cockpit_display(lane.get("last_failure_kind"))
        if failure:
            suffix = f" ({failure_kind})" if failure_kind else ""
            lines.append(f"   last failure: {failure}{suffix}")
        approval_hold = _clean_cockpit_display(lane.get("last_approval_hold"))
        if approval_hold:
            commands = lane.get("approval_next_commands")
            if isinstance(commands, list):
                clean_commands = [_clean_cockpit_display(item) for item in commands[:2]]
                command_text = ", ".join(item for item in clean_commands if item)
            else:
                command_text = ""
            next_text = f" | review: {command_text}" if command_text else ""
            lines.append(f"   approval hold: {approval_hold}{next_text}")
        return lines

    def _format_cockpit_trust_line(self, lane: dict[str, Any], *, max_checks: int = 5) -> str:
        summary = _clean_cockpit_display(lane.get("trust_summary"))
        raw_checks = lane.get("trust_checklist")
        checks = raw_checks if isinstance(raw_checks, list) else []
        fragments: list[str] = []
        for raw_check in checks[:max_checks]:
            check = raw_check if isinstance(raw_check, dict) else {}
            label = _clean_cockpit_display(check.get("label"))
            if not label:
                continue
            marker = "ok" if check.get("ready") is True else "missing"
            fragments.append(f"{label}: {marker}")
        detail = f" ({'; '.join(fragments)})" if fragments else ""
        if summary:
            return f"- trust: {summary}{detail}"
        if detail:
            return f"- trust:{detail}"
        return ""

    def _format_cockpit_lane(self, lane_key: str) -> str:
        """Show one cockpit lane without requiring the full cockpit dump."""
        result = self._capability_cockpit_result()
        metadata = getattr(result, "metadata", {}) or {}
        raw_lanes = metadata.get("capability_lanes")
        lanes = raw_lanes if isinstance(raw_lanes, list) else []
        target: dict[str, Any] | None = None
        for raw_lane in lanes:
            lane = raw_lane if isinstance(raw_lane, dict) else {}
            key = _clean_cockpit_display(lane.get("key"))
            title_alias = _normalized_lane_alias(_clean_cockpit_display(lane.get("title")))
            if key == lane_key or _COCKPIT_LANE_ALIASES.get(title_alias) == lane_key:
                target = lane
                break
        if target is None:
            return (
                "Cockpit lane: I couldn't find that lane in the current cockpit metadata.\n"
                "Run cockpit for the full read-only capability view."
            )

        title = _clean_cockpit_display(target.get("title") or target.get("key") or "Capability")
        status = _clean_cockpit_display(target.get("status") or "unknown")
        risk = _clean_cockpit_display(target.get("risk") or "unknown")
        approval = "yes" if bool(target.get("approval_required")) else "no"
        tool_coverage = _clean_cockpit_display(target.get("tool_coverage") or "unknown")
        smoke = _clean_cockpit_display(target.get("smoke_coverage") or "unknown")
        next_command = _clean_cockpit_display(
            target.get("next_command") or target.get("example_command") or "cockpit"
        )
        next_kind = _clean_cockpit_display(target.get("next_command_kind") or "suggested")
        example = _clean_cockpit_display(target.get("example_command"))

        lines = [
            f"Cockpit lane: {title}",
            f"- status: {status}",
            f"- risk: {risk}; approval required: {approval}",
            f"- tools: {tool_coverage}; smoke: {smoke}",
            f"- next: {next_command} ({next_kind})",
        ]
        trust_line = self._format_cockpit_trust_line(target)
        if trust_line:
            lines.append(trust_line)
        if example and example != next_command:
            lines.append(f"- example: {example}")
        reasons = target.get("attention_reasons")
        if isinstance(reasons, list):
            clean_reasons = [_clean_cockpit_display(reason) for reason in reasons]
            clean_reasons = [reason for reason in clean_reasons if reason]
            if clean_reasons:
                lines.append(f"- attention: {'; '.join(clean_reasons[:3])}")
        proof_points = target.get("proof_points")
        if isinstance(proof_points, list):
            clean_proofs = [_clean_cockpit_display(point) for point in proof_points]
            clean_proofs = [point for point in clean_proofs if point]
            if clean_proofs:
                lines.append(f"- proofs: {'; '.join(clean_proofs[:6])}")
        success = _clean_cockpit_display(target.get("last_success"))
        if success:
            lines.append(f"- last success: {success}")
        failure = _clean_cockpit_display(target.get("last_failure"))
        failure_kind = _clean_cockpit_display(target.get("last_failure_kind"))
        if failure:
            suffix = f" ({failure_kind})" if failure_kind else ""
            lines.append(f"- last failure: {failure}{suffix}")
        approval_hold = _clean_cockpit_display(target.get("last_approval_hold"))
        if approval_hold:
            commands = target.get("approval_next_commands")
            if isinstance(commands, list):
                clean_commands = [_clean_cockpit_display(item) for item in commands[:3]]
                command_text = ", ".join(item for item in clean_commands if item)
            else:
                command_text = ""
            review_text = f" | review: {command_text}" if command_text else ""
            lines.append(f"- approval hold: {approval_hold}{review_text}")
            lines.append(
                "- approval-held meaning: waiting for owner review; not counted as a tool failure"
            )
        notes = target.get("guardrail_notes")
        if isinstance(notes, list):
            clean_notes = [_clean_cockpit_display(note) for note in notes]
            clean_notes = [note for note in clean_notes if note]
            if clean_notes:
                lines.append(f"- guardrail: {'; '.join(clean_notes[:2])}")
        lines.append("This is read-only: it does not approve, send, call, or execute the command.")
        return "\n".join(lines)

    def _format_cockpit_lane_or_text(self, lane_key: str) -> str:
        """Show one lane when present, otherwise keep sparse cockpit fakes useful."""
        text = self._format_cockpit_lane(lane_key)
        if "I couldn't find that lane in the current cockpit metadata" in text:
            return self._format_cockpit_text()
        return text

    def _cockpit_proof_summary_entries(
        self,
        metadata: dict[str, Any],
        lanes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        raw_proof_summary = metadata.get("proof_summary")
        proof_summary = raw_proof_summary if isinstance(raw_proof_summary, list) else []
        if proof_summary:
            return proof_summary
        fallback_summary: list[dict[str, Any]] = []
        for lane in lanes:
            raw_points = lane.get("proof_points")
            points = raw_points if isinstance(raw_points, list) else []
            clean_points = [_clean_cockpit_display(point) for point in points]
            clean_points = [point for point in clean_points if point]
            if clean_points:
                fallback_summary.append(
                    {
                        "lane_title": _clean_cockpit_display(
                            lane.get("title") or lane.get("key") or "Capability"
                        ),
                        "proof_count": len(clean_points),
                        "sample_proofs": clean_points[:2],
                    }
                )
            if len(fallback_summary) >= 4:
                break
        return fallback_summary

    def _cockpit_hidden_proof_line(self, metadata: dict[str, Any]) -> str:
        hidden_count_text = _clean_cockpit_display(metadata.get("proof_summary_hidden_lane_count"), limit=12)
        if not hidden_count_text.isdigit():
            return ""
        hidden_count = int(hidden_count_text)
        if hidden_count <= 0:
            return ""
        raw_hidden_lanes = metadata.get("proof_summary_hidden_lanes")
        hidden_lanes = raw_hidden_lanes if isinstance(raw_hidden_lanes, list) else []
        hidden_titles: list[str] = []
        for raw_hidden in hidden_lanes[:4]:
            hidden = raw_hidden if isinstance(raw_hidden, dict) else {}
            title = _clean_cockpit_display(hidden.get("lane_title") or hidden.get("lane_key") or "Capability")
            if title:
                hidden_titles.append(title)
        line = f"- {hidden_count} more proof lane(s) available"
        if hidden_titles:
            line += f": {', '.join(hidden_titles)}"
        line += "; ask for `<lane> lane` to inspect one."
        return line

    def _format_cockpit_proofs(self) -> str:
        """Compact proof/evidence view for the owner phone control center."""
        result = self._capability_cockpit_result()
        metadata = getattr(result, "metadata", {}) or {}
        raw_lanes = metadata.get("capability_lanes")
        lanes = raw_lanes if isinstance(raw_lanes, list) else []
        proof_summary = self._cockpit_proof_summary_entries(metadata, lanes)
        if not proof_summary:
            return (
                "Cockpit proofs: no proof coverage is published right now.\n"
                "Run `trust tests` for the Operator Workflow Evals lane."
            )

        proof_lane_count = _clean_cockpit_display(metadata.get("proof_lane_count")) or str(len(proof_summary))
        proof_point_count = _clean_cockpit_display(metadata.get("proof_point_count"))
        if not proof_point_count:
            proof_point_total = 0
            for raw_entry in proof_summary:
                entry = raw_entry if isinstance(raw_entry, dict) else {}
                try:
                    proof_point_total += int(entry.get("proof_count") or 0)
                except (TypeError, ValueError):
                    continue
            proof_point_count = str(proof_point_total) if proof_point_total else ""

        coverage = f"{proof_lane_count} lane(s)"
        if proof_point_count:
            coverage += f", {proof_point_count} proof point(s)"
        lines = ["Cockpit proofs:", f"- coverage: {coverage}"]
        for raw_entry in proof_summary[:4]:
            entry = raw_entry if isinstance(raw_entry, dict) else {}
            title = _clean_cockpit_display(entry.get("lane_title") or entry.get("lane_key") or "Capability")
            count = _clean_cockpit_display(entry.get("proof_count"))
            raw_points = entry.get("sample_proofs")
            points = raw_points if isinstance(raw_points, list) else []
            heading = title
            if count:
                heading += f" ({count} proof(s))"
            lines.append(f"- {heading}")
            for point in points[:3]:
                clean_point = _clean_cockpit_display(point)
                if clean_point:
                    lines.append(f"  • {clean_point}")
        line = self._cockpit_hidden_proof_line(metadata)
        if line:
            lines.append(line)
        lines.append("This is read-only: it does not approve, send, call, or execute the command.")
        return "\n".join(lines)

    def _format_cockpit_summary(self) -> str:
        """Compact cockpit overview for the phone, without the full lane dump."""
        result = self._capability_cockpit_result()
        metadata = getattr(result, "metadata", {}) or {}
        raw_counts = metadata.get("status_counts")
        counts = raw_counts if isinstance(raw_counts, dict) else {}
        lane_count = _clean_cockpit_display(metadata.get("lane_count"))
        attention_count = _clean_cockpit_display(metadata.get("attention_lane_count"))
        raw_queue = metadata.get("next_command_queue")
        queue = raw_queue if isinstance(raw_queue, list) else []
        raw_lanes = metadata.get("capability_lanes")
        lanes = raw_lanes if isinstance(raw_lanes, list) else []
        proof_summary = self._cockpit_proof_summary_entries(metadata, lanes)
        approval_held_count = 0
        for raw_lane in lanes:
            lane = raw_lane if isinstance(raw_lane, dict) else {}
            status = _clean_cockpit_display(lane.get("status")).lower()
            if status == "awaiting approval" or _clean_cockpit_display(lane.get("last_approval_hold")):
                approval_held_count += 1
        if not approval_held_count:
            for key, value in counts.items():
                normalized_key = _clean_cockpit_display(key).lower().replace("_", " ").replace("-", " ")
                if normalized_key in {"awaiting approval", "approval held"}:
                    try:
                        approval_held_count = int(value)
                    except (TypeError, ValueError):
                        approval_held_count = 0
                    break

        lines = ["Cockpit summary:"]
        direction_summary = _clean_cockpit_direction(metadata.get("direction_summary"))
        raw_direction_principles = metadata.get("direction_principles")
        direction_principles = raw_direction_principles if isinstance(raw_direction_principles, list) else []
        if direction_summary:
            lines.append(f"- direction: {direction_summary}")
            for raw_principle in direction_principles[:3]:
                principle = _clean_cockpit_direction(raw_principle)
                if principle:
                    lines.append(f"  • {principle}")
        if lane_count:
            lane_text = f"{lane_count} lane(s)"
            if attention_count:
                lane_text += f", {attention_count} needing attention"
            if approval_held_count:
                lane_text += f", {approval_held_count} approval-held"
            lines.append(f"- lanes: {lane_text}")
        if counts:
            count_parts = []
            for key in sorted(counts, key=lambda item: _clean_cockpit_display(item)):
                value = counts.get(key)
                if isinstance(value, (int, float, str)):
                    clean_key = _clean_cockpit_display(key)
                    clean_value = _clean_cockpit_display(value)
                    if clean_key and clean_value:
                        count_parts.append(f"{clean_key}: {clean_value}")
            if count_parts:
                lines.append(f"- status: {', '.join(count_parts)}")
        coverage_line = _cockpit_coverage_summary_line(metadata, lanes)
        if coverage_line:
            lines.append(f"- coverage: {coverage_line}")
        trust_summary = metadata.get("trust_summary")
        if isinstance(trust_summary, dict):
            trust_line = _clean_cockpit_display(trust_summary.get("summary"))
            if trust_line:
                lines.append(f"- trust: {trust_line}")
        if proof_summary:
            proof_lane_count = _clean_cockpit_display(metadata.get("proof_lane_count")) or str(len(proof_summary))
            proof_point_count = _clean_cockpit_display(metadata.get("proof_point_count"))
            if not proof_point_count:
                proof_point_total = 0
                for entry in proof_summary:
                    item = entry if isinstance(entry, dict) else {}
                    try:
                        proof_point_total += int(item.get("proof_count") or 0)
                    except (TypeError, ValueError):
                        continue
                proof_point_count = str(proof_point_total) if proof_point_total else ""
            count_text = f"{proof_lane_count} lane(s)"
            if proof_point_count:
                count_text += f", {proof_point_count} proof point(s)"
            lines.append(f"- proofs: {count_text}")
            for raw_entry in proof_summary[:2]:
                entry = raw_entry if isinstance(raw_entry, dict) else {}
                title = _clean_cockpit_display(entry.get("lane_title") or entry.get("lane_key") or "Capability")
                raw_points = entry.get("sample_proofs")
                points = raw_points if isinstance(raw_points, list) else []
                clean_points = [_clean_cockpit_display(point) for point in points[:2]]
                sample_text = "; ".join(point for point in clean_points if point)
                if title and sample_text:
                    lines.append(f"  • {title}: {sample_text}")
            line = self._cockpit_hidden_proof_line(metadata)
            if line:
                lines.append(line)
        if approval_held_count:
            lines.append(
                "Approval-held means Jarvis is waiting for owner review; it is not counted as a tool failure."
            )
        if queue:
            lines.append("- next:")
            for entry in queue[:3]:
                item = entry if isinstance(entry, dict) else {}
                command = _clean_cockpit_display(item.get("command"))
                if not command:
                    continue
                lane = _clean_cockpit_display(item.get("lane_title") or item.get("lane_key"))
                kind = _clean_cockpit_display(item.get("kind") or "suggested")
                suffix = f" ({kind}, {lane})" if lane else f" ({kind})"
                lines.append(f"  • {command}{suffix}")
        else:
            next_command = _clean_cockpit_display(metadata.get("next_command"))
            if next_command:
                lines.append(f"- next: {next_command}")

        lane_lines = []
        for raw_lane in lanes[:6]:
            lane = raw_lane if isinstance(raw_lane, dict) else {}
            title = _clean_cockpit_display(lane.get("title") or lane.get("key"))
            status = _clean_cockpit_display(lane.get("status"))
            next_command = _clean_cockpit_display(lane.get("next_command"))
            if title and status:
                next_text = f" -> {next_command}" if next_command else ""
                details: list[str] = []
                failure = _clean_cockpit_display(lane.get("last_failure"))
                failure_kind = _clean_cockpit_display(lane.get("last_failure_kind"))
                if failure:
                    suffix = f" ({failure_kind})" if failure_kind else ""
                    details.append(f"last failure: {failure}{suffix}")
                approval_hold = _clean_cockpit_display(lane.get("last_approval_hold"))
                if approval_hold:
                    commands = lane.get("approval_next_commands")
                    if isinstance(commands, list):
                        clean_commands = [_clean_cockpit_display(item) for item in commands[:2]]
                        command_text = ", ".join(item for item in clean_commands if item)
                    else:
                        command_text = ""
                    review_text = f" | review: {command_text}" if command_text else ""
                    details.append(f"approval hold: {approval_hold}{review_text}")
                success = _clean_cockpit_display(lane.get("last_success"))
                if success and not failure and not approval_hold:
                    details.append(f"last success: {success}")
                detail_text = f" | {'; '.join(details)}" if details else ""
                lane_lines.append(f"  • {title}: {status}{next_text}{detail_text}")
        if lane_lines:
            lines.append("- lanes:")
            lines.extend(lane_lines)
        lines.append("Ask for `cockpit attention` or `<lane> lane` for details.")
        lines.append("This is read-only: it does not approve, send, call, or execute the command.")
        return "\n".join(lines)

    def _handle_command(
        self,
        command: str,
        *,
        request_token: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        low = command.lower().strip()
        compact = "".join(ch for ch in low if ch.isalnum())
        if low in {"/start", "start"}:
            return (
                "👋 Jarvis V2 online — running on your Mac.\n"
                "Send any natural command, a 🎙 voice note, or a 📷 photo to read.\n"
                "Type /help to see what I can do."
            ), None
        if low in {"/help", "help"}:
            return (
                "Send natural commands — here's a taste:\n\n"
                "🗓 Day\n"
                "• what's on my calendar today\n"
                "• am I free tomorrow afternoon?\n"
                "• what's on my apple reminders\n\n"
                "✉️ Messages & people\n"
                "• email alex@example.com saying ...\n"
                "• find contact 가상연락처일\n\n"
                "🔎 Info & utilities\n"
                "• weather in Tokyo\n"
                "• convert 100 USD to KRW\n"
                "• stock price of AAPL\n\n"
                "🎙 Voice & 📷 photos\n"
                "• send a voice note → I transcribe and run it\n"
                "• send a photo → I read the text (on-device OCR)\n\n"
                "🚦 Status\n"
                "• status — Jarvis runtime + worker readiness\n"
                "• voice — voice command intake + safety gates\n"
                "• voice setup — local voice dependency readiness\n"
                "• voice stop — read-only stop/rerecord intent packet\n"
                "• voice lane — voice capability health and setup proof\n"
                "• voice proof status — open voice acceptance proof gaps\n"
                "• voice proof matrix — voice live-proof report format; no transcript contents\n"
                "• capabilities — what Jarvis can do safely\n"
                "• can you send messages? — messaging/calling capability lane (approval-gated; live-proof pending)\n"
                "• can you check email? — productivity capability lane\n"
                "• can you check weather? — info capability lane\n"
                "• can you run commands? — risk matrix\n"
                "• agent research — Zoey/OpenClaw/Hermes guidance\n"
                "• handoff notice — latest Claude/Codex handoff context\n"
                "• what does CODEX_TASKS say? — current frozen/editable work queue\n"
                "• what changed after Codex? — latest read-only change report\n"
                "• completion status — completion-claim gate\n"
                "• why isn't Jarvis done? — completion blockers and proof debt\n"
                "• AGI progress — current build progress report\n"
                "• roadmap — next assistant-layer roadmap\n"
                "• AGI status — AGI gate readiness\n"
                "• chat latency status / chat p95 status — mixed-conversation latency proof + tuning knobs\n"
                "• goals — active durable goals\n"
                "• trust checklist — earned-trust cockpit checks\n"
                "• earned trust checklist — why lanes have earned trust\n"
                "• can I trust Jarvis? — compact cockpit trust summary\n"
                "• trust tests — operator workflow eval lane\n"
                "• proofs — compact proof coverage summary\n"
                "• what broke — recent failures at a glance\n"
                "• worker status — internal worker readiness\n"
                "• orchestration lane / worker lane — internal worker orchestration health\n"
                "• cockpit — capability lanes with health + risk\n"
                "• cockpit summary — compact capability overview\n"
                "• next action — first safe cockpit command\n"
                "• attention — cockpit lanes that need review\n"
                "• approvals lane — pending approval review health\n"
                "• messaging lane — one capability lane in detail\n"
                "• calls lane — call capability health (approval-gated; live-proof pending)\n"
                "• morning brief lane — Morning Brief capability health\n"
                "• contacts lane — contact lookup capability health\n"
                "• calendar email lane — calendar/email capability health\n"
                "• personal proofs lane — personal integration proof gaps\n"
                "• markets lane / weather lane — markets/weather capability health\n"
                "• research lane / web lane — research/web lookup capability health\n"
                "• memory lane — memory/notes capability health\n"
                "• learning lane — learning-loop/recovery proof health\n"
                "• diagnostics lane — local diagnostics/recovery health\n"
                "• operator evals lane / workflow evals lane — operator workflow eval proof lane\n"
                "• scheduler lane — scheduled-job capability health\n"
                "• guardrails — why some build paths are frozen\n"
                "• freeze status — live-proof freeze + frozen build paths\n"
                "• live proof status — compact Acceptance Harness readiness\n"
                "• acceptance harness status — one-screen acceptance proof/gap view\n"
                "• acceptance gaps — open August acceptance gaps; use acceptance next for prioritized proof guidance\n"
                "• definition of done — August acceptance checklist status\n"
                "• acceptance next proof — next safe proof lane to run with the operator\n"
                "• smoke status — latest aggregate smoke proof from the handoff log\n"
                "• reboot status — daemon startup and reboot proof readiness\n"
                "• conversation research proof matrix — conversation/research proof report format; no transcript contents\n"
                "• mixed conversation proof matrix — 10+ turn chat/research/tool proof report format; no transcript contents\n"
                "• what should I test? — pending live-proof matrix\n"
                "• live test result format — report channel, pass/fail, last visible stage/error; no secrets or message content\n"
                "• personal proof matrix — personal integration proof report format; no private account details\n"
                "• scheduler proof matrix — scheduled-job 7-day proof report format; no private payloads\n"
                "• daily value proof matrix — Morning Brief usefulness/delivery proof report format; no private brief contents\n"
                "• approval proof matrix — phone approval-flow proof report format; no private planned args\n"
                "• phone control proof matrix — what broke/cockpit/approvals proof report format; no private run payloads\n"
                "• reboot proof matrix — daemon/reboot/network recovery proof report format; no private logs\n"
                "• error guidance proof matrix — user-visible recovery-message proof format; no private error payloads\n"
                "• what can Codex touch? — current frozen/editable build paths\n"
                "• safety / what requires approval? — approval boundaries + risky tools\n"
                "• privacy — private-data boundaries\n"
                "• risk / which commands are risky? — tool risk matrix\n"
                "• readiness — safe-use readiness report\n"
                "• memory status — local memory health counters\n"
                "• learning status — feedback and learning-loop review\n"
                "• setup — dependency/env readiness\n"
                "• doctor — deeper local diagnostics\n"
                "• brief status — scheduled jobs incl. Morning Brief\n"
                "• channels — messaging/calling channel health\n"
                "• did it send? — delivery/channel diagnostics\n"
                "• approvals — anything waiting for your ✅\n\n"
                "한국어도 가능: 상태, 음성, 마이크 확인, 음성 중지, 음성 레인, 마이크 레인, 음성 증명 상태, 한국어 음성 증명, 음성 증명 매트릭스, 음성 증명 결과 형식, 기능 알려줘, 조이 조사 결과, 인수 상태, 완료 기준, 검수 체크리스트, 남은 기준, 뭐 남았어, 다음 증명, 다음 라이브 증명, 스모크 상태, 테스트 초록, 재부팅 상태, 데몬 상태, 재부팅 증명 매트릭스, 재부팅 증명 형식, 데몬 증명 매트릭스, 네트워크 복구 증명, 오류 안내 증명 매트릭스, 오류 안내 형식, 코덱스 작업 뭐야, 작업 큐 보여줘, 뭐 수정했어, 자비스 끝났어?, 완료 뭐 막혀, AGI 진행상황, 로드맵, AGI 상태, 대화 지연 상태, 채팅 느려, 대화 연구 증명 매트릭스, 연구 증명 매트릭스, 대화 증명 매트릭스, 대화 증명 결과 형식, 목표 상태, 자비스 믿어도 돼?, 신뢰 체크리스트, 믿어도 되는 이유, 신뢰 테스트, 증거 상태, 뭐가 고장났어, 워커 상태, 워커 레인, 에이전트 레인, 콕핏 요약, 다음 행동, 주의 상태, 승인 레인, 승인 증명 매트릭스, 승인 결과 형식, 폰컨트롤 증명 매트릭스, 폰 제어 증명 형식, 뭐가 고장났어 증명 형식, 콕핏 승인 증명 형식, 메시지 상태(승인 필요/라이브 검증 대기), 통화 레인, 전화 레인, 브리핑 레인, 연락처 레인, 캘린더 이메일 레인, 개인 증명 레인, 개인 증명 매트릭스, 개인 증명 결과 형식, 스케줄 레인, 스케줄 증명 매트릭스, 예약 작업 증명 형식, 일일 가치 증명 매트릭스, 아침 브리핑 증명 형식, 시장 레인, 날씨 레인, 검색 레인, 연구 레인, 기억 레인, 학습 레인, 진단 레인, 제임스 평가 레인, 워크플로우 평가 레인, 승인 상태, 동결 상태, 수정 금지 파일, 라이브 증명 상태, 라이브 테스트 상태, 라이브 테스트 뭐 해야 해, 라이브 결과 형식, 기억 상태, 학습 상태, 브리핑 상태, 채널 상태, 채널 헬스, 텔레그램 보내졌어?, 카톡 갔나요?, 전화 연결됐어?, 승인\n\n"
                "Risky actions return ✅/❌ approval buttons first.\n"
                "Not sure? Just ask “what can you do?”"
            ), None
        # Phone-control shortcuts. All read-only; they map to existing planner
        # phrases or a read-only tool, so no new authority is granted here.
        if low in {"status", "jarvis status", "/status"} or compact in {
            "상태",
            "상태알려줘",
            "상태보여줘",
            "상태확인",
            "자비스상태",
            "자비스상태알려줘",
        }:
            command = "jarvis status"
        elif low in {"voice", "voice status", "voice command cockpit", "/voice"} or compact in {
            "음성",
            "음성상태",
            "음성명령",
            "음성명령상태",
            "목소리",
            "보이스",
        }:
            command = "voice command cockpit"
        elif low in {"voice setup", "voice setup check", "mic check", "microphone check", "/voice-setup"} or compact in {
            "마이크",
            "마이크상태",
            "마이크확인",
            "음성설정",
            "음성설정확인",
            "보이스설정",
        }:
            command = "voice setup check"
        elif low in {"voice stop", "stop voice", "cancel voice input", "rerecord voice input", "/voice-stop"} or compact in {
            "음성중지",
            "음성멈춰",
            "음성취소",
            "녹음중지",
            "녹음멈춰",
            "다시녹음",
        }:
            command = "voice stop intent: stop listening"
        elif low in {
            "capabilities",
            "capability map",
            "what can you do",
            "what can you do?",
            "/capabilities",
        } or compact in {
            "기능",
            "기능목록",
            "기능보여줘",
            "기능알려줘",
            "명령어",
            "명령어목록",
            "명령어보여줘",
            "명령어알려줘",
            "뭐할수있어",
            "뭐할수있니",
            "뭘할수있어",
            "무엇을할수있어",
            "무슨기능있어",
            "사용가능한기능",
            "자비스기능",
            "자비스기능알려줘",
            "자비스뭐할수있어",
        }:
            command = "capability map"
        elif low in {
            "can jarvis send messages",
            "can jarvis send messages?",
            "can you call my contacts",
            "can you call my contacts?",
            "can you call people",
            "can you call people?",
            "can you message people",
            "can you message people?",
            "can you send imessage",
            "can you send imessage?",
            "can you send kakao",
            "can you send kakao?",
            "can you send kakao messages",
            "can you send kakao messages?",
            "can you send kakaotalk",
            "can you send kakaotalk?",
            "can you send messages",
            "can you send messages?",
            "can you send telegram",
            "can you send telegram?",
            "can you send telegram messages",
            "can you send telegram messages?",
            "can you text people",
            "can you text people?",
            "do you support kakao",
            "do you support kakao?",
            "do you support telegram",
            "do you support telegram?",
        } or compact in {
            "메세지보낼수있어",
            "메시지보낼수있어",
            "문자보낼수있어",
            "전화걸수있어",
            "카톡보낼수있어",
            "텔레그램보낼수있어",
        }:
            command = "capability map messages"
        elif low in {
            "can jarvis check email",
            "can jarvis check email?",
            "can you brief me every morning",
            "can you brief me every morning?",
            "can you check email",
            "can you check email?",
            "can you give morning brief",
            "can you give morning brief?",
            "can you read email",
            "can you read email?",
            "can you read my calendar",
            "can you read my calendar?",
        } or compact in {
            "아침브리핑가능해",
            "이메일확인가능해",
            "리마인더가능해",
            "타이머설정가능해",
            "캘린더볼수있어",
        }:
            command = "capability map productivity"
        elif low in {
            "can jarvis check weather",
            "can jarvis check weather?",
            "can you check weather",
            "can you check weather?",
            "can you define words",
            "can you define words?",
            "can you get weather",
            "can you get weather?",
            "can you look up wikipedia",
            "can you look up wikipedia?",
            "can you read news",
            "can you read news?",
            "can you summarize news",
            "can you summarize news?",
            "can you tell me weather",
            "can you tell me weather?",
        } or compact in {
            "날씨확인가능해",
            "뉴스읽을수있어",
            "뉴스요약가능해",
            "위키찾아볼수있어",
        }:
            command = "capability map info"
        elif low in {
            "can you check stock prices",
            "can you check stock prices?",
            "can you tell me bitcoin price",
            "can you tell me bitcoin price?",
        } or compact in {
            "비트코인가격볼수있어",
            "주식확인가능해",
        }:
            command = "capability map markets"
        elif low in {
            "can you calculate things",
            "can you calculate things?",
            "can you convert currency",
            "can you convert currency?",
            "can you translate",
            "can you translate?",
        } or compact in {
            "계산가능해",
            "번역가능해",
            "환율계산가능해",
        }:
            command = "capability map utilities"
        elif low in {
            "can you browse the web",
            "can you browse the web?",
            "can you research the web",
            "can you research the web?",
            "can you search the internet",
            "can you search the internet?",
        } or compact in {
            "웹검색가능해",
        }:
            command = "capability map research"
        elif low in {
            "can you write emails",
            "can you write emails?",
            "can you write text",
            "can you write text?",
        } or compact in {
            "글쓰기가능해",
        }:
            command = "capability map writing"
        elif low in {
            "can you remember things",
            "can you remember things?",
            "can you search memory",
            "can you search memory?",
        } or compact in {
            "기억할수있어",
        }:
            command = "capability map memory"
        elif low in {
            "can you take notes",
            "can you take notes?",
        } or compact in {
            "메모가능해",
        }:
            command = "capability map notes"
        elif low in {
            "can you read files",
            "can you read files?",
        } or compact in {
            "파일읽을수있어",
        }:
            command = "safety status"
        elif low in {
            "can you run commands",
            "can you run commands?",
        } or compact in {
            "명령실행가능해",
        }:
            command = "risk matrix"
        elif low in {
            "can i talk to you",
            "can i talk to you?",
            "can you hear me",
            "can you hear me?",
            "can you use voice",
            "can you use voice?",
            "how do i use voice",
            "how do i use voice?",
        } or compact in {
            "말로할수있어",
            "음성가능해",
        }:
            command = "voice command cockpit"
        elif low in {
            "check what claude has left",
            "check what claude left",
            "handoff",
            "handoff brief",
            "handoff notice",
            "show handoff notice",
            "what claude said",
            "what did claude leave",
            "what did claude leave for codex",
            "what did claude leave for codex?",
            "what did claude say",
            "what has claude left",
        } or compact in {
            "claude가뭐남겼어",
            "claude가뭐라고했어",
            "claude핸드오프",
            "claude인수인계",
            "클로드가뭐남겼어",
            "클로드가뭐라고했어",
            "클로드핸드오프",
            "클로드인수인계",
            "인수인계",
            "핸드오프",
        }:
            command = "handoff brief"
        elif low in {
            "agent landscape research",
            "agent research",
            "agent research note",
            "ai agent landscape",
            "ai agent landscape research",
            "ai agent research",
            "ai agent research note",
            "all ai agent research",
            "all ai agents research",
            "devin research",
            "every ai agent research",
            "every ai agents research",
            "how do we unlock the moat",
            "jarvis direction",
            "jarvis moat",
            "jarvis models research",
            "jarvis product strategy",
            "jarvis strategy",
            "jarvis three month plan",
            "lindy research",
            "manus research",
            "other ai agent research",
            "other ai agents research",
            "other jarvis models",
            "other jarvis models research",
            "what did the agent research say",
            "what did research say to avoid",
            "what did you find about ai agents",
            "what did you find about other ai agents",
            "what did you find about other jarvis models",
            "what did you find about zoey",
            "what did you find about zoey?",
            "what did you learn from zoey",
            "what did you learn from zoey?",
            "what about other jarvis models",
            "what is jarvis moat",
            "what is the three month plan",
            "what is the three month plan for jarvis",
            "what is the jarvis strategy",
            "what is the jarvis moat",
            "what is the right move for jarvis",
            "what is the strategy after the research",
            "what should jarvis build before integrations",
            "what should jarvis focus on for the next three months",
            "zoey research",
            "zoey os research",
            "openclaw research",
            "hermes research",
            "three month plan",
            "three months of work",
            "what are the three months of work",
            "what should jarvis avoid from ai agents",
            "what should jarvis avoid from zoey",
            "what should jarvis borrow from lindy",
            "what should jarvis borrow from openclaw",
            "what should jarvis borrow from zoey",
            "what should jarvis borrow from zoey?",
            "what should jarvis build after zoey research",
            "what should jarvis copy from lindy",
            "what should jarvis copy from openclaw",
            "what should jarvis copy from zoey",
            "what should jarvis copy from zoey?",
            "what should jarvis copy from zoey os",
            "what should jarvis do after the research",
            "what should jarvis do after the research?",
            "what should we do after zoey research",
            "what should we do after zoey research?",
            "what was the ai agent research recommendation",
            "what was the ai agent research recommendation?",
            "what unlocks jarvis moat",
            "what unlocks jarvis's moat",
            "what unlocks the moat",
            "when should jarvis add integrations",
            "what did claude say about zoey",
            "what did claude say about zoey?",
            "what did claude recommend about companions",
            "what did claude recommend about companions?",
            "why no companion personas",
            "why no companions",
            "why not companion personas",
            "should jarvis add companion personas",
            "should jarvis add companions",
            "should jarvis build companion personas",
            "should jarvis build companions",
            "should jarvis copy lindy",
            "should jarvis copy openclaw",
            "should jarvis copy zoey",
            "should jarvis copy zoey?",
            "should jarvis copy zoey os",
            "should jarvis add integrations now",
            "should jarvis have companion personas",
            "should jarvis have companions",
            "should jarvis use companion personas",
            "should jarvis use companions",
            "work queue",
            "check codex tasks",
            "check codex.md tasks",
            "codex work queue",
            "current work queue",
            "current instructions",
            "what does codex tasks say",
            "what does codex tasks say?",
        } or compact in {
            "조이조사결과",
            "zoey조사결과",
            "에이전트조사결과",
            "ai에이전트조사",
            "ai에이전트랜드스케이프",
            "모든ai에이전트조사",
            "전체ai에이전트조사",
            "다른ai에이전트조사",
            "다른자비스모델",
            "다른jarvis모델",
            "자비스모델조사",
            "다른자비스모델조사",
            "자비스3개월계획",
            "자비스세달계획",
            "자비스다음3개월뭐해",
            "자비스차별점",
            "자비스해자",
            "자비스해자어떻게열어",
            "해자어떻게열어",
            "조이에서피해야할것",
            "조이따라가야해",
            "조이따라야해",
            "조이따라할까",
            "조이따라해야해",
            "자비스동반자해야해",
            "자비스방향",
            "자비스전략",
            "자비스조이이후뭐만들까",
            "자비스조이이후뭐만들어",
            "자비스조이따라가야해",
            "자비스조이따라야해",
            "자비스조이따라할까",
            "자비스조이따라해야해",
            "자비스컴패니언해야해",
            "조이이후뭐만들까",
            "조이이후뭐만들어",
            "린디연구",
            "마누스연구",
            "오픈클로연구",
            "헤르메스연구",
            "왜동반자안해",
            "왜컴패니언안해",
            "동반자페르소나피해야해",
            "동반자레이어해야해",
            "컴패니언페르소나피해야해",
            "컴패니언레이어해야해",
            "통합지금추가해도돼",
            "통합언제추가해",
            "자비스통합언제추가해",
            "코덱스작업큐",
            "코덱스작업목록",
            "코덱스작업뭐야",
            "코덱스작업상태",
            "작업큐",
            "작업큐보여줘",
            "작업목록",
            "작업목록보여줘",
            "현재작업큐",
            "현재작업목록",
            "현재지시사항",
            "whatdoescodextaskssay",
        }:
            command = "work queue"
        elif low in {
            "code changes status",
            "diff status",
            "patch status",
            "show changed files",
            "show modified files",
            "what changed after codex",
            "what changed after codex?",
            "what changed in code",
            "what changed in jarvis",
            "what did codex change",
            "what did codex just change",
            "what did codex just do",
            "what did codex modify",
            "what did codex modify?",
            "what did you edit",
            "what files changed",
            "what files did codex edit",
            "what should claude review",
            "what should i review in this change",
            "which files changed",
            "which files did you edit",
        } or compact in {
            "뭐수정했어",
            "뭘바꿨어",
            "어떤파일바꿨어",
            "변경파일보여줘",
            "수정파일보여줘",
            "코덱스가뭐바꿨어",
            "코덱스변경사항",
            "최근변경사항",
            "패치상태",
            "변경사항확인",
        }:
            command = "what changed in Jarvis"
        elif low in {
            "can jarvis claim completion",
            "can jarvis claim completion?",
            "can jarvis claim done",
            "can jarvis claim done?",
            "completion claim",
            "completion claim gate",
            "completion gate",
            "completion status",
            "is jarvis complete",
            "is jarvis complete?",
            "is jarvis done",
            "is jarvis done?",
            "is jarvis finished",
            "is jarvis finished?",
            "jarvis completion status",
            "jarvis done status",
            "jarvis finished status",
            "what blocks completion",
            "what blocks completion?",
            "what blocks jarvis completion",
            "what blocks jarvis completion?",
            "what blocks jarvis from being done",
            "what blocks jarvis from being done?",
            "what is blocking completion",
            "what is blocking completion?",
            "when is jarvis done",
            "when is jarvis done?",
            "why isn't jarvis done",
            "why isn't jarvis done?",
            "why is jarvis not done",
            "why is jarvis not done?",
            "are you done",
            "are you done?",
            "are we done",
            "are we done?",
        } or compact in {
            "자비스끝났어",
            "자비스끝났니",
            "자비스다끝났어",
            "자비스완료됐어",
            "자비스완료됬어",
            "자비스완료상태",
            "자비스완성됐어",
            "완료주장",
            "완료게이트",
            "완료클레임",
            "완료뭐막혀",
            "완료가왜안돼",
            "완료왜안돼",
            "완료막힘",
            "완료블로커",
            "자비스완료뭐막혀",
            "자비스왜안끝났어",
            "자비스왜아직안끝났어",
            "완료주장게이트",
        }:
            command = "completion claim gate"
        elif low in {
            "agi progress",
            "build progress",
            "build status",
            "how is jarvis going",
            "how is jarvis going?",
            "jarvis build progress",
            "jarvis progress",
            "progress",
            "progress report",
            "project status",
            "show build progress",
            "what did you build",
            "what did you build?",
            "what have you built",
            "what have you built?",
            "what is the build progress",
            "what is the build progress?",
        } or compact in {
            "agi진행상황",
            "agi진행",
            "빌드상태",
            "빌드진행상황",
            "빌드진행",
            "자비스진행상황",
            "자비스진행",
            "진행상황",
            "진행보고",
            "뭐만들었어",
            "무엇을만들었어",
        }:
            command = "build progress"
        elif low in {
            "agi roadmap",
            "jarvis roadmap",
            "roadmap",
            "show roadmap",
            "show the roadmap",
            "what is next on the roadmap",
            "what is next on the roadmap?",
            "what is the roadmap",
            "what is the roadmap?",
            "what should jarvis build next",
            "what should jarvis build next?",
        } or compact in {
            "agi로드맵",
            "로드맵",
            "로드맵보여줘",
            "자비스로드맵",
            "다음로드맵",
            "다음에뭐만들어",
            "다음에무엇을만들어",
        }:
            command = "roadmap"
        elif low in {
            "agi gate status",
            "agi gates",
            "agi readiness",
            "agi readiness status",
            "agi status",
            "agent harness readiness",
            "agent harness status",
            "harness readiness",
            "harness readiness status",
            "harness status",
        } or compact in {
            "agi게이트",
            "agi게이트상태",
            "agi상태",
            "agi준비",
            "agi준비상태",
            "에이지아이상태",
            "에이지아이준비",
            "하네스상태",
            "하네스준비",
            "하네스준비상태",
        }:
            command = "agi gates"
        elif low in {
            "chat latency",
            "chat latency acceptance status",
            "chat latency status",
            "chat acceptance latency",
            "chat history window",
            "chat history messages",
            "chat max history messages",
            "chat max reply tokens",
            "chat p95",
            "chat p95 status",
            "chat p95 target",
            "chat reply token cap",
            "chat response length status",
            "chat speed",
            "chat speed acceptance status",
            "chat speed status",
            "chat speed settings",
            "chat token cap",
            "chat tuning",
            "chat tuning knobs",
            "conversation acceptance latency",
            "conversation latency",
            "conversation latency status",
            "conversation p95 status",
            "conversation speed status",
            "8 second chat target",
            "8 second conversation target",
            "8s chat target",
            "8s conversation target",
            "did chat meet the 8 second target",
            "did chat pass latency",
            "did mixed conversation pass latency",
            "how do i make chat faster",
            "how do i make chat faster?",
            "is chat under 8 seconds",
            "is jarvis under 8 seconds",
            "is chat slow",
            "is chat slow?",
            "is mixed conversation under 8 seconds",
            "jarvis latency status",
            "jarvis speed status",
            "latency status",
            "lower chat history window",
            "lower chat token cap",
            "lower reply token cap",
            "make chat faster",
            "make jarvis faster",
            "make jarvis replies shorter",
            "mixed conversation latency",
            "mixed conversation latency status",
            "mixed conversation p95",
            "mixed conversation p95 status",
            "mixed conversation acceptance latency",
            "mixed conversation acceptance status",
            "p95 chat target",
            "p95 latency status",
            "reduce chat history",
            "reduce reply tokens",
            "reply length status",
            "response length status",
            "should we lower chat history",
            "should we lower reply tokens",
            "speed up chat",
            "speed up jarvis",
            "tune chat speed",
            "tune jarvis speed",
            "under 8 seconds status",
            "what are the chat speed knobs",
            "what are the chat speed knobs?",
            "what are the chat tuning knobs",
            "what are the chat tuning knobs?",
            "why are chat replies slow",
            "why are chat replies slow?",
            "why is chat slow",
            "why is chat slow?",
            "why is jarvis slow",
            "why is jarvis slow?",
            "why are replies slow",
            "why are replies slow?",
        } or compact in {
            "대화느려",
            "대화속도",
            "대화속도상태",
            "대화p95상태",
            "대화8초목표",
            "대화지연",
            "대화지연상태",
            "대화지연확인",
            "8초대화상태",
            "8초채팅상태",
            "답변느려",
            "답변길이상태",
            "답변속도",
            "응답느려",
            "응답길이상태",
            "응답속도",
            "대화기록상태",
            "채팅p95상태",
            "채팅8초목표",
            "채팅기록창",
            "채팅히스토리상태",
            "채팅느려",
            "채팅속도",
            "채팅속도상태",
            "채팅속도설정",
            "채팅토큰상태",
            "채팅튜닝",
            "채팅지연",
            "채팅지연상태",
            "혼합대화p95상태",
            "혼합대화지연상태",
            "혼합대화속도상태",
            "자비스느려",
            "자비스속도상태",
            "자비스왜느려",
        }:
            command = "model routing status"
        elif low in {
            "active goals",
            "goals",
            "goal status",
            "goals status",
            "list goals",
            "show goals",
            "show my goals",
            "what are my goals",
            "what are my goals?",
            "what goals do i have",
            "what goals do i have?",
        } or compact in {
            "목표",
            "목표상태",
            "목표보여줘",
            "목표목록",
            "자비스목표",
            "활성목표",
        }:
            command = "list goals"
        elif low in {
            "can i trust jarvis",
            "can i trust jarvis?",
            "can i trust you",
            "can i trust you?",
            "can jarvis be trusted",
            "can jarvis be trusted?",
            "earned trust",
            "earned trust checklist",
            "earned trust status",
            "is jarvis reliable",
            "is jarvis reliable?",
            "is jarvis trustworthy",
            "is jarvis trustworthy?",
            "jarvis trust status",
            "reliability health",
            "reliability report",
            "reliability status",
            "show reliability status",
            "show trust status",
            "trust health",
            "trust checklist",
            "trust report",
            "trust status",
            "what makes jarvis reliable",
            "what makes jarvis reliable?",
            "what makes jarvis trustworthy",
            "what makes jarvis trustworthy?",
            "why can i trust jarvis",
            "why can i trust jarvis?",
            "why should i trust jarvis",
            "why should i trust jarvis?",
        } or compact in {
            "믿어도돼",
            "믿어도되나요",
            "믿어도되는이유",
            "믿을만해",
            "신뢰가능",
            "신뢰보고서",
            "신뢰체크리스트",
            "신뢰할수있어",
            "신뢰할수있나요",
            "신뢰도상태",
            "신뢰도보고서",
            "자비스믿어도돼",
            "자비스믿어도되나요",
            "자비스믿어도되는이유",
            "자비스믿을만해",
            "자비스신뢰가능",
            "자비스신뢰체크리스트",
            "자비스신뢰상태",
            "왜자비스믿어도돼",
            "왜자비스를믿어도돼",
        }:
            try:
                return self._format_cockpit_summary(), None
            except Exception as e:
                return _runtime_error_reply(e), None
        elif low in {
            "approval-gated commands",
            "approval gated commands",
            "approval required commands",
            "can i use jarvis safely",
            "can i use jarvis safely?",
            "can jarvis refuse",
            "can jarvis refuse?",
            "commands that need approval",
            "commands that require approval",
            "is jarvis safe to use",
            "is jarvis safe to use?",
            "safe to use jarvis",
            "safe to use jarvis?",
            "safety",
            "safety status",
            "what are jarvis boundaries",
            "what are jarvis boundaries?",
            "what are jarvis limits",
            "what are jarvis limits?",
            "what are the approval boundaries",
            "what are the approval boundaries?",
            "what are the safety boundaries",
            "what are the safety boundaries?",
            "what are your boundaries",
            "what are your boundaries?",
            "what are your limits",
            "what are your limits?",
            "what are you allowed to do",
            "what are you allowed to do?",
            "what are you not allowed to do",
            "what are you not allowed to do?",
            "what actions need approval",
            "what actions need approval?",
            "what commands need approval",
            "what commands need approval?",
            "what can jarvis not do",
            "what can jarvis not do?",
            "what can you not do",
            "what can you not do?",
            "what cant jarvis do",
            "what cant jarvis do?",
            "what cant you do",
            "what cant you do?",
            "what is approval gated",
            "what is approval gated?",
            "what is approval-gated",
            "what is approval-gated?",
            "what needs approval",
            "what needs approval?",
            "what needs my permission",
            "what needs my permission?",
            "what requires approval",
            "what requires approval?",
            "what requires my permission",
            "what requires my permission?",
            "what tools require approval",
            "what tools require approval?",
            "which tools require approval",
            "which tools require approval?",
            "/safety",
        } or compact in {
            "안전",
            "안전상태",
            "안전하게써도돼",
            "안전하게써도되나요",
            "경계",
            "못하는것",
            "무엇을못해",
            "무엇을할수없어",
            "뭐못해",
            "뭘못해",
            "뭘할수없어",
            "한계",
            "권한필요",
            "권한필요한것",
            "뭐승인필요",
            "승인필요",
            "승인필요한것",
            "승인필요한명령",
            "승인이필요한것",
            "승인이필요한명령",
            "어떤도구가승인필요",
            "어떤명령이승인필요",
            "자비스경계",
            "자비스한계",
            "자비스뭐못해",
            "자비스뭘못해",
            "자비스못하는것",
            "자비스안전하게써도돼",
            "자비스안전하게써도되나요",
        }:
            command = "safety status"
        elif low in {"privacy", "privacy report", "/privacy"} or compact in {"개인정보", "프라이버시"}:
            command = "privacy report"
        elif low in {
            "high risk commands",
            "high risk tools",
            "risk",
            "risk matrix",
            "what is high risk",
            "what is high risk?",
            "what permissions do you need",
            "what permissions do you need?",
            "which actions are risky",
            "which actions are risky?",
            "which commands are risky",
            "which commands are risky?",
            "which tools are high risk",
            "which tools are high risk?",
            "/risk",
        } or compact in {
            "고위험도구",
            "고위험명령",
            "리스크",
            "위험",
            "위험한도구",
            "위험한명령",
        }:
            command = "risk matrix"
        elif low in {
            "audit trail",
            "execution audit",
            "latest failure",
            "last error",
            "last failed tool",
            "latest error",
            "latest failed tool",
            "last tool run",
            "latest tool run",
            "recent tool run",
            "show last error",
            "show last failure",
            "show latest error",
            "show audit trail",
            "show execution audit",
            "tool execution history",
            "tool run history",
            "what broke",
            "what broke?",
            "what did jarvis run last",
            "what errors are there",
            "what errors happened",
            "what failed last",
            "what is broken",
            "what is broken?",
            "what ran last",
            "what tool ran last",
            "what was the last error",
            "what was the last failure",
            "what's broken",
            "/broke",
        } or compact in {
            "감사로그",
            "감사상태",
            "도구실행기록",
            "뭐가고장났어",
            "뭐가고장났나요",
            "뭐가문제야",
            "뭐고장났어",
            "뭐문제있어",
            "마지막도구실행",
            "마지막실행",
            "문제있어",
            "고장났어",
            "실행기록",
            "최근도구실행",
            "최근실행기록",
            "최근오류",
            "최근오류보여줘",
            "최근에러",
            "최근에러보여줘",
            "마지막오류",
            "마지막오류보여줘",
            "마지막에러",
            "마지막에러보여줘",
            "무슨오류있어",
        }:
            command = "recent tool runs"
        elif low in {"runtime trace latest"} or compact in {"런타임추적"}:
            command = "runtime trace receipt"
        elif low in {"last execution receipt", "latest execution receipt"} or compact in {"실행영수증"}:
            command = "verification receipt"
        elif low in {
            "anything failing",
            "failure details",
            "failure report",
            "failure status",
            "repeated failure status",
            "repeated failures status",
            "show failures",
            "show me failures",
            "what failed",
            "what failed recently",
            "what is failing",
        }:
            command = "repeated failure clusters" if "repeated failure" in low else "execution health report"
        elif compact in {
            "마지막실패",
            "마지막실패보여줘",
            "뭐가실패했어",
            "뭐실패했어",
            "실패한거있어",
            "최근실패",
            "최근실패보여줘",
            "실패상태",
        }:
            command = "execution health report"
        elif compact in {"반복실패상태"}:
            command = "repeated failure clusters"
        elif low in {
            "how do i recover",
            "how do i recover?",
            "is recovery debt closed",
            "is recovery debt closed?",
            "recovery closure status",
            "recovery debt status",
            "recovery learning status",
            "recovery plan",
            "recovery status",
            "what needs recovery",
            "what needs recovery?",
        } or compact in {
            "복구계획",
            "복구마감상태",
            "복구부채상태",
            "복구상태",
            "복구종료상태",
            "복구클로저상태",
            "복구학습상태",
            "뭐복구해야해",
            "무엇을복구해야해",
        }:
            command = "recovery closure checklist"
        elif low in {"brief status", "/brief"} or compact in {
            "브리핑상태",
            "브리핑확인",
            "모닝브리핑상태",
            "아침브리핑상태",
            "모닝브리프상태",
        }:
            command = "list scheduled jobs"
        elif low in {"channels", "channel health", "channel status", "/channels"} or compact in {
            "채널",
            "채널상태",
            "채널확인",
            "채널헬스",
            "diditsend",
            "diditgothrough",
            "wasitsent",
            "didmessagesend",
            "didthemessagesend",
            "didmymessagesend",
            "wasmessagesent",
            "wasthemessagesent",
            "messagesent",
            "didtelegramsend",
            "didtelegramgothrough",
            "wastelegramsent",
            "telegramsent",
            "telegramdelivered",
            "didkakaosend",
            "didkakaogothrough",
            "waskakaosent",
            "kakaosent",
            "didkakaotalksend",
            "didkakaotalkgothrough",
            "waskakaotalksent",
            "kakaotalksent",
            "didimessagesend",
            "didimessagegothrough",
            "wasimessagesent",
            "imessagesent",
            "didinstagramsend",
            "didinstagramgothrough",
            "wasinstagramsent",
            "instagramsent",
            "didinstasend",
            "didinstagothrough",
            "wasinstasent",
            "instasent",
            "didcallgothrough",
            "didthecallgothrough",
            "didphonecallgothrough",
            "wascallconnected",
            "메세지갔어",
            "메세지갔나요",
            "메세지보내졌어",
            "메세지보내졌나요",
            "메세지보냈어",
            "메세지보냈나요",
            "메시지갔어",
            "메시지갔나요",
            "메시지보내졌어",
            "메시지보내졌나요",
            "메시지보냈어",
            "메시지보냈나요",
            "문자갔어",
            "문자갔나요",
            "문자보내졌어",
            "문자보내졌나요",
            "문자보냈어",
            "문자보냈나요",
            "전송갔어",
            "전송갔나요",
            "전송됐어",
            "전송됐나요",
            "전송됬어",
            "전송됬나요",
            "텔레그램갔어",
            "텔레그램갔나요",
            "텔레그램보내졌어",
            "텔레그램보내졌나요",
            "텔레그램보냈어",
            "텔레그램보냈나요",
            "카카오갔어",
            "카카오갔나요",
            "카카오보내졌어",
            "카카오보내졌나요",
            "카카오보냈어",
            "카카오보냈나요",
            "카톡갔어",
            "카톡갔나요",
            "카톡보내졌어",
            "카톡보내졌나요",
            "카톡보냈어",
            "카톡보냈나요",
            "아이메시지갔어",
            "아이메시지갔나요",
            "아이메시지보내졌어",
            "아이메시지보내졌나요",
            "아이메시지보냈어",
            "아이메시지보냈나요",
            "인스타그램갔어",
            "인스타그램갔나요",
            "인스타그램보내졌어",
            "인스타그램보내졌나요",
            "인스타그램보냈어",
            "인스타그램보냈나요",
            "인스타갔어",
            "인스타갔나요",
            "인스타보내졌어",
            "인스타보내졌나요",
            "인스타보냈어",
            "인스타보냈나요",
            "전화연결됐어",
            "전화연결됐나요",
            "전화연결됬어",
            "전화연결됬나요",
            "전화됐어",
            "전화됐나요",
            "전화됬어",
            "전화됬나요",
            "전화걸렸어",
            "전화걸렸나요",
            "통화연결됐어",
            "통화연결됐나요",
            "통화연결됬어",
            "통화연결됬나요",
            "통화됐어",
            "통화됐나요",
            "통화됬어",
            "통화됬나요",
            "페이스타임연결됐어",
            "페이스타임연결됐나요",
            "페이스타임연결됬어",
            "페이스타임연결됬나요",
            "페이스타임됐어",
            "페이스타임됐나요",
            "페이스타임됬어",
            "페이스타임됬나요",
        }:
            command = "channel health"
        elif low in {
            "approvals",
            "pending",
            "pending approvals",
            "what approvals are pending",
            "what needs my approval",
            "show approval queue",
            "approval queue",
            "/approvals",
        } or compact in {
            "승인",
            "승인대기",
            "승인대기뭐있어",
            "승인뭐남았어",
            "대기중인승인",
        }:
            command = "pending approvals"
        elif low in {"readiness", "ready", "ready?", "am i ready", "/ready"} or compact in {
            "준비",
            "준비상태",
            "준비됐어",
            "레디니스",
        }:
            command = "readiness report"
        elif low in {
            "memory",
            "memory health",
            "memory report",
            "memory status",
            "memory stats",
            "knowledge status",
        } or compact in {
            "기억",
            "기억상태",
            "기억확인",
            "메모리",
            "메모리상태",
            "메모리확인",
        }:
            command = "memory stats"
        elif low in {
            "after action learning status",
            "after-action learning status",
            "what did jarvis learn from last run",
            "what did jarvis learn from the last failure",
            "what did jarvis learn from the last run",
        }:
            command = "after-action learning packet"
        elif low in {
            "failure learning status",
        } or compact in {"실패학습상태"}:
            command = "failure learning cockpit"
        elif low in {
            "is learning debt closed",
            "is learning debt closed?",
            "learning debt status",
            "learning loop proof",
            "learning proof matrix",
            "what learning debt is open",
        } or compact in {
            "학습부채상태",
            "학습부채닫혔어",
            "학습증명매트릭스",
        }:
            command = "execution learning closure"
        elif low in {
            "learning",
            "learning health",
            "learning loop",
            "learning loop status",
            "learning report",
            "learning review",
            "learning status",
            "what did you learn",
            "what did you learn?",
            "what have you learned",
            "what have you learned?",
        } or compact in {
            "학습",
            "학습루프상태",
            "학습상태",
            "학습확인",
            "학습리뷰",
            "뭘배웠어",
            "무엇을배웠어",
            "배운거",
        }:
            command = "learning review"
        elif low in {"setup", "setup check", "/setup"} or compact in {"설정", "설정확인"}:
            command = "setup check"
        elif low in {"doctor", "jarvis doctor", "health", "health check", "/doctor"} or compact in {
            "진단",
            "닥터",
            "헬스체크",
        }:
            command = "jarvis doctor"
        elif low in {
            "next cockpit action",
            "cockpit next action",
            "next action",
            "next safe action",
            "what command should i run next",
            "what command should i run now",
            "what is jarvis next step",
            "what is next for jarvis",
            "what should i check next",
            "what should i check now",
            "what should i do next",
            "what should i do next?",
            "what should i run next",
            "what should i run now",
            "what should jarvis do next",
            "what's next for jarvis",
            "what is next",
            "what next",
            "/next",
        } or compact in {
            "다음행동",
            "다음액션",
            "다음명령",
            "다음콕핏액션",
            "다음할일",
            "다음뭐해",
            "다음뭐해야해",
            "다음에뭘실행해",
            "뭘하면돼",
            "무슨명령실행해",
            "자비스다음단계",
            "자비스다음뭐해",
            "자비스뭐부터해",
        }:
            try:
                return self._format_next_cockpit_action(), None
            except Exception as e:
                return _runtime_error_reply(e), None
        elif low in {
            "cockpit summary",
            "capability summary",
            "capabilities summary",
            "control plane summary",
            "control-plane summary",
            "jarvis cockpit summary",
            "jarvis control plane summary",
            "/cockpit-summary",
            "/summary",
        } or compact in {
            "콕핏요약",
            "기능요약",
            "능력요약",
            "컨트롤요약",
            "컨트롤플레인요약",
            "자비스요약",
            "자비스콕핏요약",
        }:
            try:
                return self._format_cockpit_summary(), None
            except Exception as e:
                return _runtime_error_reply(e), None
        elif low in {
            "proofs",
            "proof status",
            "proof coverage",
            "evidence",
            "evidence status",
            "evidence coverage",
            "what proof do we have",
            "what proof do we have?",
            "what evidence do we have",
            "what evidence do we have?",
            "what is proven",
            "what is proven?",
            "what is verified",
            "what is verified?",
            "show proofs",
            "show evidence",
            "/proofs",
            "/evidence",
        } or compact in {
            "증거",
            "증거상태",
            "증거요약",
            "검증",
            "검증상태",
            "검증요약",
            "증명",
            "증명상태",
            "프루프",
            "프루프상태",
            "무엇이검증됐어",
            "뭐가검증됐어",
            "뭐가증명됐어",
        }:
            try:
                return self._format_cockpit_proofs(), None
            except Exception as e:
                return _runtime_error_reply(e), None
        elif low in {
            "trust tests",
            "trust test status",
            "eval pack plan",
            "operator eval pack plan",
            "operator trust tests",
            "operator workflow evals",
            "operator workflow tests",
            "operator real workflow tests",
            "real operator workflow tests",
            "what is eval pack",
            "what is the eval pack",
            "what is the operator eval pack",
            "what real workflows should jarvis prove",
            "what real workflows are tested",
            "what real workflows are covered",
            "what should jarvis prove live",
            "can jarvis safely send korean messages",
            "hangul message proof",
            "hangul send proof",
            "is hangul messaging covered",
            "is korean message covered",
            "is korean messaging covered",
            "is korean messaging tested",
            "is korean telegram covered",
            "korean message eval",
            "korean message proof",
            "korean message test",
            "korean send proof",
            "korean telegram eval",
            "korean telegram proof",
            "korean telegram test",
            "morning brief eval",
            "phone control evals",
            "phone eval pack",
            "workflow proof status",
        } or compact in {
            "신뢰테스트",
            "신뢰테스트상태",
            "자비스신뢰테스트",
            "제임스워크플로우평가",
            "제임스워크플로우테스트",
            "실제워크플로우평가",
            "실제워크플로우테스트",
            "평가팩계획",
            "제임스평가팩계획",
            "실제워크플로우뭐증명해",
            "자비스뭐증명해야해",
            "한국어메시지평가",
            "한국어메시지검증",
            "한국어메시지증명",
            "한국어메시지테스트됐어",
            "한국어텔레그램평가",
            "한국어텔레그램검증",
            "한국어텔레그램안전해",
            "한국어텔레그램증명",
            "한글메시지증명",
            "한글전송증명",
            "가상연락처이메시지테스트",
            "가상연락처이전송증명",
            "가상연락처이전송테스트",
            "모닝브리핑평가",
            "폰컨트롤평가",
        }:
            try:
                return self._format_cockpit_lane("operator_workflow_evals"), None
            except Exception as e:
                return _runtime_error_reply(e), None
        elif low in {
            "acceptance next proof",
            "next acceptance proof",
            "next acceptance test",
            "next live proof",
            "next live test",
            "next proof",
            "next proof status",
            "next proof to run",
            "what is the next acceptance proof",
            "what is the next acceptance proof?",
            "what proof is next",
            "what proof is next?",
            "what proof should i do next",
            "what proof should i do next?",
            "what proof should i run next",
            "what proof should i run next?",
            "what should i prove next",
            "what should i prove next?",
            "what should i test next",
            "what should i test next?",
            "what should operator test next",
            "what should operator test next?",
            "which proof is next",
            "which proof is next?",
        } or compact in {
            "acceptancenextproof",
            "다음라이브증명",
            "다음라이브증명상태",
            "다음라이브테스트",
            "다음검증",
            "다음검증상태",
            "다음인수증명",
            "다음에뭐증명해",
            "다음에뭐테스트해",
            "다음엔뭐증명해",
            "다음증명",
            "다음증명뭐야",
            "다음증명상태",
            "다음프루프",
        }:
            try:
                result = self._runtime_instance().handle(
                    "completion_next_proof_packet",
                    request_token=request_token,
                )
                return _result_response_text(result), None
            except Exception as e:
                return _runtime_error_reply(e), None
        elif low in {
            "acceptance",
            "acceptance harness",
            "acceptance harness status",
            "acceptance status",
            "acceptance test status",
            "acceptance gaps",
            "acceptance gaps status",
            "acceptance coverage",
            "acceptance coverage drift",
            "acceptance coverage status",
            "acceptance harness rows",
            "acceptance next",
            "acceptance next proof",
            "acceptance next status",
            "august acceptance checklist",
            "definition of done",
            "definition of done status",
            "done checklist",
            "finish checklist",
            "finish line",
            "finish line status",
            "daemon startup",
            "daemon startup status",
            "daemon status",
            "live acceptance status",
            "live check",
            "live check coverage",
            "live check missing rows",
            "live check one screen table",
            "live check rows",
            "live check status",
            "live_check",
            "live_check coverage",
            "live_check missing rows",
            "live_check one screen table",
            "live_check rows",
            "live_check status",
            "live proof status",
            "launchagent status",
            "launchd status",
            "one screen acceptance",
            "one screen acceptance table",
            "one screen live check table",
            "one screen live_check table",
            "one-screen acceptance",
            "one-screen acceptance table",
            "one-screen live_check table",
            "open acceptance gaps",
            "open dod gaps",
            "open gaps",
            "next acceptance",
            "next acceptance proof",
            "next live proof",
            "next proof",
            "next proof status",
            "aggregate smoke",
            "aggregate smoke status",
            "are tests green",
            "are tests green?",
            "are the tests green",
            "are the tests green?",
            "full smoke status",
            "latest aggregate smoke proof",
            "reboot status",
            "reboot survival",
            "reboot survival status",
            "smoke lock status",
            "smoke status",
            "smoke suite lock status",
            "smoke suite status",
            "smoke test status",
            "smoke tests status",
            "startup status",
            "suite lock status",
            "test status",
            "tests green",
            "tests green?",
            "what is missing",
            "what is still missing",
            "what is unfinished",
            "is korean voice proven",
            "is korean voice proven?",
            "is korean voice tested",
            "is korean voice tested?",
            "is live check missing rows",
            "is live_check missing rows",
            "is telegram voice proven",
            "is telegram voice proven?",
            "is telegram voice tested",
            "is telegram voice tested?",
            "is voice proven",
            "is voice proven?",
            "is voice tested",
            "is voice tested?",
            "korean voice proof",
            "korean voice proof status",
            "telegram voice proof",
            "telegram voice proof status",
            "what remains unfinished",
            "voice acceptance status",
            "voice proof",
            "voice proof status",
            "voice test status",
            "will jarvis restart after reboot",
            "will jarvis restart after reboot?",
            "will jarvis start after reboot",
            "will jarvis start after reboot?",
            "will jarvis survive reboot",
            "will jarvis survive reboot?",
            "what is done for jarvis",
            "what is done for jarvis?",
            "what is the definition of done",
            "what is the definition of done?",
            "what rows are in live check",
            "what rows are in live_check",
            "what rows does live check show",
            "what rows does live_check show",
            "does live check cover acceptance",
            "does live check cover every checklist section",
            "does live check cover the august checklist",
            "does live_check cover acceptance",
            "does live_check cover every checklist section",
            "does live_check cover the august checklist",
            "what is the finish line",
            "what is the finish line?",
            "what is left",
            "what is left?",
            "what is left before jarvis is done",
            "what is left before jarvis is done?",
            "what is left to finish",
            "what is left to finish?",
            "what remains",
            "what remains?",
            "what remains before jarvis is done",
            "what remains before jarvis is done?",
            "what still needs proof",
            "what still needs proof?",
            "what's left",
            "what's left?",
            "what's left to finish",
            "what's left to finish?",
            "what proof is next",
            "what proof is next?",
            "what proof should i do next",
            "what proof should i do next?",
            "what should i prove next",
            "what should i prove next?",
            "/acceptance",
            "/live-check",
        } or compact in {
            "acceptancenext",
            "acceptancenextproof",
            "acceptancenextstatus",
            "acceptancegaps",
            "acceptancegapsstatus",
            "aggregatesmoke",
            "aggregatesmokestatus",
            "aretestsgreen",
            "arethetestsgreen",
            "augustacceptancechecklist",
            "definitionofdone",
            "definitionofdonestatus",
            "donechecklist",
            "finishchecklist",
            "finishline",
            "finishlinestatus",
            "fullsmokestatus",
            "opengaps",
            "openacceptancegaps",
            "opendodgaps",
            "remainingacceptancegaps",
            "인수",
            "인수상태",
            "인수하네스",
            "인수테스트상태",
            "검수체크리스트",
            "끝나는기준",
            "마감기준",
            "남은기준",
            "남은검수",
            "남은완료기준",
            "남은완료조건",
            "남은일",
            "남은증명",
            "음성검증",
            "음성검증상태",
            "음성증명",
            "음성증명상태",
            "음성테스트상태",
            "완료기준",
            "완료기준상태",
            "완료조건",
            "완료체크리스트",
            "데몬상태",
            "런치에이전트상태",
            "재부팅상태",
            "재부팅증명",
            "재부팅후실행",
            "재부팅후자비스",
            "자비스재부팅",
            "자비스재부팅상태",
            "자비스재시작상태",
            "자비스다시켜져",
            "자비스다시켜져?",
            "자비스완료기준",
            "자비스완료조건",
            "라이브체크",
            "라이브체크누락",
            "라이브체크빠진항목",
            "라이브체크상태",
            "라이브체크커버리지",
            "라이브체크표",
            "라이브체크항목",
            "라이브검증상태",
            "라이브증명상태",
            "텔레그램음성검증",
            "한국어음성검증",
            "한국어음성증명",
            "한국어음성테스트",
            "다음증명",
            "다음증명상태",
            "다음라이브증명",
            "다음라이브증명상태",
            "다음검증",
            "다음검증상태",
            "다음인수증명",
            "다음프루프",
            "뭐증명해야해",
            "뭐남았어",
            "무엇이남았어",
            "뭐가남았어",
            "자비스뭐남았어",
            "자비스남은일",
            "남은게뭐야",
            "남은거뭐야",
            "무엇을증명해야해",
            "어떤증명해야해",
            "다음에뭐증명해",
            "다음엔뭐증명해",
            "스모크",
            "스모크상태",
            "스모크테스트",
            "스모크테스트상태",
            "테스트상태",
            "테스트초록",
            "테스트초록?",
            "테스트통과",
            "테스트통과했어",
            "테스트통과했어?",
            "스모크통과",
            "스모크통과했어",
            "스모크통과했어?",
            "smokestatus",
            "smoketeststatus",
            "smoketestsstatus",
            "teststatus",
            "testsgreen",
            "원스크린인수",
            "원스크린인수상태",
            "원스크린인수표",
            "인수커버리지",
            "인수표",
        }:
            try:
                return self._format_cockpit_lane_or_text("acceptance_harness"), None
            except Exception as e:
                return _runtime_error_reply(e), None
        elif low in {
            "agent fleet health",
            "agent fleet readiness",
            "agent health",
            "agent readiness",
            "agents health",
            "are agents ready",
            "are agents ready?",
            "are my agents ready",
            "are my agents ready?",
            "how many agents are ready",
            "how many agents are ready?",
            "internal agents status",
            "internal worker status",
            "parallel agent status",
            "parallel agents status",
            "ready agent count",
            "ready agents status",
            "subagent fleet health",
            "subagent fleet readiness",
            "subagent health",
            "tool orchestration status",
            "worker fleet health",
            "worker health",
            "worker readiness",
        } or compact in {
            "내부워커상태",
            "병렬에이전트상태",
            "서브에이전트준비상태",
            "서브에이전트헬스",
            "에이전트준비상태",
            "에이전트헬스",
            "워커준비상태",
            "워커헬스",
            "준비된에이전트개수",
            "준비된에이전트몇개",
        }:
            try:
                return self._format_cockpit_lane("orchestration"), None
            except Exception as e:
                return _runtime_error_reply(e), None
        elif lane_key := _cockpit_lane_key_for_command(low, compact):
            try:
                return self._format_cockpit_lane(lane_key), None
            except Exception as e:
                return _runtime_error_reply(e), None
        elif low in {
            "cockpit attention",
            "attention",
            "attention status",
            "what needs attention",
            "what needs attention?",
            "what needs review",
            "what needs review?",
            "what needs my attention",
            "what needs my attention?",
            "which lanes need attention",
            "/attention",
        } or compact in {
            "주의상태",
            "주의필요",
            "검토필요",
            "콕핏주의",
            "콕핏주의상태",
            "봐야할것",
            "내가봐야할것",
            "내가볼것",
            "검토할것",
        }:
            try:
                return self._format_cockpit_attention(), None
            except Exception as e:
                return _runtime_error_reply(e), None
        elif low in {
            "cockpit",
            "capability cockpit",
            "capability cockpit plan",
            "guardrails",
            "show guardrails",
            "build guardrails",
            "can codex edit planner",
            "can codex edit send code",
            "can codex edit the planner",
            "can codex edit the send code",
            "can you edit send code",
            "can you edit the planner",
            "can you edit the send code",
            "freeze status",
            "freeze list",
            "frozen files",
            "codex guardrails",
            "live freeze list",
            "live proof freeze",
            "live proof status",
            "live proof test status",
            "live channel proof status",
            "conversation research proof matrix",
            "conversation research proof result format",
            "conversation and research proof matrix",
            "conversation and research result format",
            "research proof matrix",
            "research proof result format",
            "web research proof matrix",
            "chat research proof matrix",
            "section a proof matrix",
            "what conversation research should i test",
            "what research proof should i test",
            "how should i report conversation research proof",
            "how should i report research proof",
            "mixed conversation proof matrix",
            "mixed conversation proof result format",
            "conversation proof matrix",
            "conversation proof result format",
            "chat latency proof matrix",
            "chat latency proof result format",
            "what mixed conversation should i test",
            "what conversation should i test",
            "what chat proof should i run",
            "how should i report mixed conversation proof",
            "how should i report conversation proof",
            "how should i report chat latency proof",
            "voice proof matrix",
            "voice proof result format",
            "voice live proof format",
            "telegram voice proof matrix",
            "telegram voice proof result format",
            "local voice proof matrix",
            "local voice proof result format",
            "korean voice proof matrix",
            "korean voice proof result format",
            "push to talk proof matrix",
            "push-to-talk proof matrix",
            "talk.py proof matrix",
            "talk py proof matrix",
            "spoken reply proof matrix",
            "voice speak proof matrix",
            "what voice should i test",
            "what voice proofs should i test",
            "what local voice should i test",
            "what telegram voice should i test",
            "what korean voice should i test",
            "how should i report voice proof",
            "how should i report local voice proof",
            "how should i report telegram voice proof",
            "how should i report korean voice proof",
            "how should i report personal proof results",
            "how do i report personal proof results",
            "pending live proofs",
            "pending live tests",
            "personal live proof format",
            "personal proof matrix",
            "personal proof result format",
            "personal proofs matrix",
            "personal proofs result format",
            "personal integration proof matrix",
            "personal integration result format",
            "personal integrations proof matrix",
            "personal integrations result format",
            "calendar create update delete proof matrix",
            "calendar write proof matrix",
            "calendar write result format",
            "what calendar write should i test",
            "how should i report calendar write proof",
            "email read proof matrix",
            "email search proof matrix",
            "email send proof matrix",
            "what email read should i test",
            "what email search should i test",
            "what email send should i test",
            "how should i report email read proof",
            "how should i report email search proof",
            "how should i report email send proof",
            "reminder proof matrix",
            "set reminder proof matrix",
            "what reminder proof should i test",
            "what set reminder proof should i test",
            "how should i report reminder proof",
            "how should i report set reminder proof",
            "contact lookup proof matrix",
            "what contact lookup should i test",
            "how should i report contact lookup proof",
            "캘린더 쓰기 증명 매트릭스",
            "이메일 읽기 증명",
            "이메일 검색 증명",
            "이메일 보내기 증명",
            "리마인더 증명 매트릭스",
            "연락처 조회 증명 매트릭스",
            "scheduler proof matrix",
            "scheduler proof result format",
            "scheduled jobs proof matrix",
            "scheduled jobs result format",
            "schedule proof matrix",
            "schedule proof result format",
            "daily value proof matrix",
            "daily value proof result format",
            "daily value result format",
            "morning brief proof matrix",
            "morning brief proof result format",
            "daily brief proof matrix",
            "brief delivery proof matrix",
            "brief proof matrix",
            "what daily value should i test",
            "what morning brief should i test",
            "how should i report daily value proof",
            "how should i report morning brief proof",
            "morning brief delivery proof matrix",
            "7 day scheduler proof",
            "7-day scheduler proof",
            "jobs 7 day proof",
            "jobs seven day proof",
            "scheduled jobs 7 day proof",
            "scheduled jobs streak",
            "scheduled job streak status",
            "scheduler streak status",
            "did scheduled jobs run for 7 days",
            "are scheduled jobs running daily",
            "what is the 7 day scheduler proof",
            "what scheduler proof do you need",
            "what scheduled jobs should i test",
            "what scheduler streak should i test",
            "how should i report scheduler proof",
            "how should i report scheduled job results",
            "how should i report scheduled job streak proof",
            "daily streak proof",
            "daily job streak proof",
            "job streak proof matrix",
            "scheduled delivery proof matrix",
            "approval proof matrix",
            "approval proof result format",
            "approval flow proof matrix",
            "approval flow result format",
            "phone approval proof",
            "phone approval proof matrix",
            "phone approval result format",
            "approval button proof matrix",
            "approval button result format",
            "approval buttons proof matrix",
            "approval callback proof matrix",
            "approval callback result format",
            "approval from phone proof matrix",
            "approval queue button proof matrix",
            "approve deny proof matrix",
            "approve dismiss proof matrix",
            "approve from phone proof matrix",
            "deny from phone proof matrix",
            "how should i report approval button proof",
            "how should i report approval callback proof",
            "how should i report approve deny proof",
            "how should i report phone approval proof",
            "phone approval buttons proof matrix",
            "phone approval-flow proof report format",
            "phone approve deny proof matrix",
            "risky action approval proof matrix",
            "telegram approval buttons proof matrix",
            "telegram approval result format",
            "what approval buttons should i test",
            "what approval callback should i test",
            "what approve deny buttons should i test",
            "what phone approval should i test",
            "what phone approvals should i test",
            "what approval proof do you need",
            "what approval flow should i test",
            "how should i report approval proof",
            "how should i report phone approval results",
            "phone control proof matrix",
            "phone control proof result format",
            "phone control live proof format",
            "what broke proof matrix",
            "cockpit proof matrix",
            "cockpit approvals proof matrix",
            "what phone control should i test",
            "what phone shortcuts should i test",
            "how should i report phone control proof",
            "how should i report phone shortcut proof",
            "reboot proof matrix",
            "reboot proof result format",
            "daemon proof matrix",
            "daemon proof result format",
            "daemon startup proof matrix",
            "launchagent proof matrix",
            "launchd proof matrix",
            "post reboot proof matrix",
            "post-reboot proof matrix",
            "network recovery proof matrix",
            "network recovery proof result format",
            "network loss proof matrix",
            "network loss proof result format",
            "telegram control daemon status",
            "telegram control proof matrix",
            "status server proof matrix",
            "scheduler daemon proof matrix",
            "what reboot proof should i test",
            "what daemon proof should i test",
            "what network recovery should i test",
            "what should i test after reboot",
            "how should i report reboot proof",
            "how should i report daemon proof",
            "how should i report network recovery proof",
            "error guidance proof matrix",
            "error guidance proof result format",
            "error message proof matrix",
            "error message proof result format",
            "recovery guidance proof matrix",
            "user error proof matrix",
            "what error messages should i test",
            "what error messages should i test?",
            "how should i report error guidance proof",
            "how should i report error guidance proof?",
            "are error messages actionable",
            "connector error proof matrix",
            "connector recovery proof matrix",
            "do error messages name the fix",
            "do user errors name the fix",
            "error messages status",
            "error recovery proof",
            "error recovery status",
            "how should i report error messages",
            "how should i report error messages?",
            "how should i report error recovery proof",
            "how should i report error recovery proof?",
            "is error guidance proven",
            "recovery guidance status",
            "show error guidance",
            "show recovery guidance",
            "user visible error proof matrix",
            "user-visible error proof matrix",
            "what recovery guidance should i test",
            "what recovery guidance should i test?",
            "what user facing errors need proof",
            "what user facing errors need proof?",
            "what user-facing errors need proof",
            "what user-facing errors need proof?",
            "오류 안내 증명 매트릭스",
            "오류 안내 증명",
            "오류 안내 형식",
            "오류 메시지 상태",
            "오류 복구 상태",
            "복구 안내 증명",
            "에러 안내 증명",
            "proofs pending",
            "safe lane",
            "test matrix status",
            "tests pending",
            "what personal integrations should i test",
            "what personal proof do you need",
            "what personal proofs are pending",
            "what personal proofs should i test",
            "what can codex touch",
            "what can you edit",
            "what channels need proof",
            "what channels should i test",
            "what live channels are pending",
            "what live proof do you need",
            "what live proofs are pending",
            "what live tests are pending",
            "what proof is pending",
            "what proofs are still pending",
            "which channels need live proof",
            "which channels should i test",
            "how do i report live test results",
            "how should i report live test results",
            "how should i report the live matrix",
            "how to report live test results",
            "live test result format",
            "live test results format",
            "live matrix result format",
            "report live matrix results",
            "report live test results",
            "what format should i use for live test results",
            "what should i send after testing",
            "phone control center plan",
            "what is capability cockpit plan",
            "what is phone control center",
            "what is the capability cockpit plan",
            "what is the phone control center",
            "what is frozen",
            "what files are frozen",
            "what is the freeze list",
            "what is the live test matrix",
            "what results do you need from me",
            "what should codex avoid",
            "what should codex leave alone",
            "what should codex not edit",
            "what should codex not touch",
            "what should you not edit",
            "what should you not touch",
            "what should i test",
            "what should operator test",
            "what tests should i run",
            "why frozen",
            "/cockpit",
            "/freeze",
            "/guardrails",
        } or compact in {
            "콕핏",
            "능력콕핏계획",
            "폰컨트롤센터계획",
            "가드레일",
            "가드레일상태",
            "동결상태",
            "동결파일",
            "프리즈리스트",
            "프리즈상태",
            "수정금지파일",
            "검증대기",
            "검증대기뭐야",
            "남은라이브증거",
            "남은라이브테스트",
            "라이브검증대기",
            "라이브검증상태",
            "라이브증거상태",
            "라이브증명상태",
            "라이브채널대기",
            "라이브채널뭐테스트해",
            "라이브채널어떤거테스트해",
            "라이브테스트결과",
            "라이브테스트결과보고",
            "라이브테스트대기",
            "라이브테스트뭐해야해",
            "라이브테스트상태",
            "개인연동뭐테스트해",
            "개인증명결과형식",
            "개인증명매트릭스",
            "개인증명뭐해야해",
            "개인증명어떻게보고해",
            "개인증명형식",
            "스케줄증명결과형식",
            "스케줄증명매트릭스",
            "스케줄증명뭐해야해",
            "스케줄증명어떻게보고해",
            "예약작업증명형식",
            "예약작업뭐테스트해",
            "예약작업결과형식",
            "승인증명결과형식",
            "승인결과형식",
            "승인증명매트릭스",
            "승인증명뭐해야해",
            "승인증명어떻게보고해",
            "전화승인결과형식",
            "전화승인증명",
            "승인흐름증명",
            "승인흐름뭐테스트해",
            "승인거절증명",
            "승인거절증명매트릭스",
            "승인버튼결과형식",
            "승인버튼증명",
            "승인버튼증명매트릭스",
            "승인콜백결과형식",
            "승인콜백증명",
            "승인콜백증명매트릭스",
            "텔레그램승인결과형식",
            "텔레그램승인증명",
            "폰승인버튼증명",
            "폰승인버튼증명매트릭스",
            "approvalbuttonproofmatrix",
            "approvalbuttonresultformat",
            "approvalbuttonsproofmatrix",
            "approvalcallbackproofmatrix",
            "approvalcallbackresultformat",
            "approvalfromphoneproofmatrix",
            "approvalqueuebuttonproofmatrix",
            "approvedenyproofmatrix",
            "approvedismissproofmatrix",
            "approvefromphoneproofmatrix",
            "denyfromphoneproofmatrix",
            "howshouldireportapprovalbuttonproof",
            "howshouldireportapprovalcallbackproof",
            "howshouldireportapprovedenyproof",
            "howshouldireportphoneapprovalproof",
            "phoneapprovalbuttonsproofmatrix",
            "phoneapprovalflowproofreportformat",
            "phoneapprovedenyproofmatrix",
            "riskyactionapprovalproofmatrix",
            "telegramapprovalbuttonsproofmatrix",
            "telegramapprovalresultformat",
            "whatapprovalbuttonsshoulditest",
            "whatapprovalcallbackshoulditest",
            "whatapprovedenybuttonsshoulditest",
            "whatphoneapprovalshoulditest",
            "whatphoneapprovalsshoulditest",
            "폰컨트롤증명매트릭스",
            "폰제어증명형식",
            "폰컨트롤결과형식",
            "뭐가고장났어증명형식",
            "콕핏증명매트릭스",
            "콕핏승인증명형식",
            "폰컨트롤뭐테스트해",
            "폰제어뭐테스트해",
            "폰컨트롤어떻게보고해",
            "폰제어어떻게보고해",
            "phonecontrolproofmatrix",
            "phonecontrolproofresultformat",
            "phonecontrolliveproofformat",
            "whatbrokeproofmatrix",
            "cockpitproofmatrix",
            "cockpitapprovalsproofmatrix",
            "whatphonecontrolshoulditest",
            "whatphoneshortcutsshoulditest",
            "howshouldireportphonecontrolproof",
            "howshouldireportphoneshortcutproof",
            "재부팅증명매트릭스",
            "재부팅증명형식",
            "데몬증명매트릭스",
            "데몬증명형식",
            "데몬시작증명",
            "런치에이전트증명",
            "네트워크복구증명",
            "네트워크끊김증명",
            "재부팅뭐테스트해",
            "데몬뭐테스트해",
            "재부팅어떻게보고해",
            "데몬어떻게보고해",
            "rebootproofmatrix",
            "rebootproofresultformat",
            "daemonproofmatrix",
            "daemonproofresultformat",
            "daemonstartupproofmatrix",
            "launchagentproofmatrix",
            "networkrecoveryproofmatrix",
            "networklossproofmatrix",
            "whatrebootproofshoulditest",
            "whatdaemonproofshoulditest",
            "howshouldireportrebootproof",
            "howshouldireportdaemonproof",
            "오류안내증명매트릭스",
            "오류안내형식",
            "에러안내증명",
            "복구안내증명",
            "오류메시지증명",
            "오류뭐테스트해",
            "오류어떻게보고해",
            "errorguidanceproofmatrix",
            "errorguidanceproofresultformat",
            "errormessageproofmatrix",
            "errormessageproofresultformat",
            "recoveryguidanceproofmatrix",
            "usererrorproofmatrix",
            "whaterrormessagesshoulditest",
            "howshouldireporterrorguidanceproof",
            "areerrormessagesactionable",
            "connectorerrorproofmatrix",
            "connectorrecoveryproofmatrix",
            "doerrormessagesnamethefix",
            "dousererrorsnamethefix",
            "errormessagesstatus",
            "errorrecoveryproof",
            "errorrecoverystatus",
            "howshouldireporterrormessages",
            "howshouldireporterrorrecoveryproof",
            "iserrorguidanceproven",
            "recoveryguidancestatus",
            "showerrorguidance",
            "showrecoveryguidance",
            "uservisibleerrorproofmatrix",
            "whatrecoveryguidanceshoulditest",
            "whatuserfacingerrorsneedproof",
            "복구안내증명",
            "오류안내증명",
            "오류메시지상태",
            "오류복구상태",
            "라이브결과보고방법",
            "라이브결과형식",
            "라이브테스트결과형식",
            "뭐테스트해야해",
            "뭘테스트해야해",
            "무슨테스트해야해",
            "어떤테스트해야해",
            "어떤채널검증해야해",
            "어떤채널테스트해",
            "채널검증뭐남았어",
            "채널뭐테스트해",
            "테스트결과보고방법",
            "테스트결과어떻게보고해",
            "테스트결과어떻게보내",
            "테스트결과형식",
            "테스트매트릭스",
            "테스트매트릭스보여줘",
            "테스트매트릭스상태",
            "테스트대기",
            "테스트해야할거",
            "liveproofstatus",
            "liveproofteststatus",
            "howshouldireportpersonalproofresults",
            "howdoireportpersonalproofresults",
            "personalliveproofformat",
            "personalproofmatrix",
            "personalproofresultformat",
            "personalproofsmatrix",
            "personalproofsresultformat",
            "personalintegrationproofmatrix",
            "personalintegrationresultformat",
            "schedulerproofmatrix",
            "schedulerproofresultformat",
            "scheduledjobsproofmatrix",
            "scheduledjobsresultformat",
            "scheduleproofmatrix",
            "scheduleproofresultformat",
            "7dayschedulerproof",
            "7dayschedulerproof",
            "whatschedulerproofdoyouneed",
            "whatscheduledjobsshoulditest",
            "howshouldireportschedulerproof",
            "howshouldireportscheduledjobresults",
            "approvalproofmatrix",
            "approvalproofresultformat",
            "approvalflowproofmatrix",
            "approvalflowresultformat",
            "phoneapprovalproof",
            "phoneapprovalproofmatrix",
            "phoneapprovalresultformat",
            "whatapprovalproofdoyouneed",
            "whatapprovalflowshoulditest",
            "howshouldireportapprovalproof",
            "howshouldireportphoneapprovalresults",
            "mixedconversationproofmatrix",
            "mixedconversationproofresultformat",
            "conversationproofmatrix",
            "conversationproofresultformat",
            "chatlatencyproofmatrix",
            "chatlatencyproofresultformat",
            "whatmixedconversationshoulditest",
            "whatconversationshoulditest",
            "whatchatproofshouldirun",
            "howshouldireportmixedconversationproof",
            "howshouldireportconversationproof",
            "howshouldireportchatlatencyproof",
            "대화증명결과형식",
            "대화증명매트릭스",
            "대화증명뭐해야해",
            "대화증명어떻게보고해",
            "혼합대화증명",
            "혼합대화뭐테스트해",
            "대화지연증명형식",
            "챗지연증명형식",
            "voiceproofmatrix",
            "voiceproofresultformat",
            "voiceliveproofformat",
            "telegramvoiceproofmatrix",
            "telegramvoiceproofresultformat",
            "localvoiceproofmatrix",
            "localvoiceproofresultformat",
            "koreanvoiceproofmatrix",
            "koreanvoiceproofresultformat",
            "pushtotalkproofmatrix",
            "talkpyproofmatrix",
            "spokenreplyproofmatrix",
            "voicespeakproofmatrix",
            "whatvoiceshoulditest",
            "whatvoiceproofsshoulditest",
            "whatlocalvoiceshoulditest",
            "whattelegramvoiceshoulditest",
            "whatkoreanvoiceshoulditest",
            "howshouldireportvoiceproof",
            "howshouldireportlocalvoiceproof",
            "howshouldireporttelegramvoiceproof",
            "howshouldireportkoreanvoiceproof",
            "conversationresearchproofmatrix",
            "conversationresearchproofresultformat",
            "conversationandresearchproofmatrix",
            "conversationandresearchresultformat",
            "researchproofmatrix",
            "researchproofresultformat",
            "webresearchproofmatrix",
            "chatresearchproofmatrix",
            "sectionaproofmatrix",
            "whatconversationresearchshoulditest",
            "whatresearchproofshoulditest",
            "howshouldireportconversationresearchproof",
            "howshouldireportresearchproof",
            "대화연구증명매트릭스",
            "대화연구증명형식",
            "연구증명매트릭스",
            "연구증명형식",
            "리서치증명매트릭스",
            "리서치증명형식",
            "검색증명매트릭스",
            "대화연구뭐테스트해",
            "연구증명뭐테스트해",
            "연구증명어떻게보고해",
            "dailyvalueproofmatrix",
            "dailyvalueproofresultformat",
            "dailyvalueresultformat",
            "morningbriefproofmatrix",
            "morningbriefproofresultformat",
            "dailybriefproofmatrix",
            "briefdeliveryproofmatrix",
            "briefproofmatrix",
            "morningbriefdeliveryproofmatrix",
            "jobs7dayproof",
            "jobssevendayproof",
            "scheduledjobs7dayproof",
            "scheduledjobsstreak",
            "scheduledjobstreakstatus",
            "schedulerstreakstatus",
            "didscheduledjobsrunfor7days",
            "arescheduledjobsrunningdaily",
            "whatisthe7dayschedulerproof",
            "whatschedulerstreakshoulditest",
            "howshouldireportscheduledjobstreakproof",
            "dailystreakproof",
            "dailyjobstreakproof",
            "jobstreakproofmatrix",
            "scheduleddeliveryproofmatrix",
            "whatdailyvalueshoulditest",
            "whatmorningbriefshoulditest",
            "howshouldireportdailyvalueproof",
            "howshouldireportmorningbriefproof",
            "일일가치증명매트릭스",
            "일일가치증명형식",
            "아침브리핑증명매트릭스",
            "아침브리핑증명형식",
            "브리핑증명매트릭스",
            "브리핑증명형식",
            "브리핑전송증명",
            "브리핑뭐테스트해",
            "아침브리핑뭐테스트해",
            "일일가치뭐테스트해",
            "브리핑어떻게보고해",
            "음성증명결과형식",
            "음성증명매트릭스",
            "음성증명뭐해야해",
            "음성증명어떻게보고해",
            "음성뭐테스트해",
            "로컬음성증명",
            "텔레그램음성증명",
            "한국어음성증명형식",
            "텔레그램음성증명형식",
            "pendingliveproofs",
            "pendinglivetests",
            "proofspending",
            "safelane",
            "showguardrails",
            "testmatrixstatus",
            "testspending",
            "codexguardrails",
            "cancodexeditplanner",
            "cancodexeditsendcode",
            "cancodexedittheplanner",
            "cancodexeditthesendcode",
            "canyoueditsendcode",
            "canyouedittheplanner",
            "canyoueditthesendcode",
            "livechannelproofstatus",
            "livefreezelist",
            "whatpersonalintegrationsshoulditest",
            "whatpersonalproofdoyouneed",
            "whatpersonalproofsarepending",
            "whatpersonalproofsshoulditest",
            "whatcancodextouch",
            "whatcanyouedit",
            "whatisthelivetestmatrix",
            "whatliveproofdoyouneed",
            "whatliveproofsarepending",
            "whatlivetestsarepending",
            "whatchannelsneedproof",
            "whatchannelsshoulditest",
            "whatlivechannelsarepending",
            "whatproofispending",
            "whatproofsarestillpending",
            "whichchannelsneedliveproof",
            "whichchannelsshoulditest",
            "whatresultsdoyouneedfromme",
            "howdoireportlivetestresults",
            "howshouldireportlivetestresults",
            "howshouldireportthelivematrix",
            "howtoreportlivetestresults",
            "livetestresultformat",
            "livetestresultsformat",
            "livematrixresultformat",
            "reportlivematrixresults",
            "reportlivetestresults",
            "whatformatshouldiuseforlivetestresults",
            "whatshouldisendaftertesting",
            "launchdproofmatrix",
            "postrebootproofmatrix",
            "networkrecoveryproofresultformat",
            "networklossproofresultformat",
            "telegramcontroldaemonstatus",
            "telegramcontrolproofmatrix",
            "statusserverproofmatrix",
            "schedulerdaemonproofmatrix",
            "whatnetworkrecoveryshoulditest",
            "whatshoulditestafterreboot",
            "howshouldireportnetworkrecoveryproof",
            "whatshouldcodexavoid",
            "whatshouldcodexleavealone",
            "whatshouldcodexnotedit",
            "whatshouldcodexnottouch",
            "whatshouldyounotedit",
            "whatshouldyounottouch",
            "whatshoulditest",
            "whatshouldoperatortest",
            "whattestsshouldirun",
            "whyfrozen",
        }:
            try:
                if low in {
                    "how do i report live test results",
                    "how should i report live test results",
                    "how should i report the live matrix",
                    "how to report live test results",
                    "how should i report personal proof results",
                    "how do i report personal proof results",
                    "live test result format",
                    "live test results format",
                    "live matrix result format",
                    "conversation research proof matrix",
                    "conversation research proof result format",
                    "conversation and research proof matrix",
                    "conversation and research result format",
                    "research proof matrix",
                    "research proof result format",
                    "web research proof matrix",
                    "chat research proof matrix",
                    "section a proof matrix",
                    "what conversation research should i test",
                    "what research proof should i test",
                    "how should i report conversation research proof",
                    "how should i report research proof",
                    "mixed conversation proof matrix",
                    "mixed conversation proof result format",
                    "conversation proof matrix",
                    "conversation proof result format",
                    "chat latency proof matrix",
                    "chat latency proof result format",
                    "what mixed conversation should i test",
                    "what conversation should i test",
                    "what chat proof should i run",
                    "how should i report mixed conversation proof",
                    "how should i report conversation proof",
                    "how should i report chat latency proof",
                    "voice proof matrix",
                    "voice proof result format",
                    "voice live proof format",
                    "telegram voice proof matrix",
                    "telegram voice proof result format",
                    "local voice proof matrix",
                    "local voice proof result format",
                    "korean voice proof matrix",
                    "korean voice proof result format",
                    "push to talk proof matrix",
                    "push-to-talk proof matrix",
                    "talk.py proof matrix",
                    "talk py proof matrix",
                    "spoken reply proof matrix",
                    "voice speak proof matrix",
                    "what voice should i test",
                    "what voice proofs should i test",
                    "what local voice should i test",
                    "what telegram voice should i test",
                    "what korean voice should i test",
                    "how should i report voice proof",
                    "how should i report local voice proof",
                    "how should i report telegram voice proof",
                    "how should i report korean voice proof",
                    "personal live proof format",
                    "personal proof matrix",
                    "personal proof result format",
                    "personal proofs matrix",
                    "personal proofs result format",
                    "personal integration proof matrix",
                    "personal integration result format",
                    "personal integrations proof matrix",
                    "personal integrations result format",
                    "calendar create update delete proof matrix",
                    "calendar write proof matrix",
                    "calendar write result format",
                    "what calendar write should i test",
                    "how should i report calendar write proof",
                    "email read proof matrix",
                    "email search proof matrix",
                    "email send proof matrix",
                    "what email read should i test",
                    "what email search should i test",
                    "what email send should i test",
                    "how should i report email read proof",
                    "how should i report email search proof",
                    "how should i report email send proof",
                    "reminder proof matrix",
                    "set reminder proof matrix",
                    "what reminder proof should i test",
                    "what set reminder proof should i test",
                    "how should i report reminder proof",
                    "how should i report set reminder proof",
                    "contact lookup proof matrix",
                    "what contact lookup should i test",
                    "how should i report contact lookup proof",
                    "캘린더 쓰기 증명 매트릭스",
                    "이메일 읽기 증명",
                    "이메일 검색 증명",
                    "이메일 보내기 증명",
                    "리마인더 증명 매트릭스",
                    "연락처 조회 증명 매트릭스",
                    "scheduler proof matrix",
                    "scheduler proof result format",
                    "scheduled jobs proof matrix",
                    "scheduled jobs result format",
                    "schedule proof matrix",
                    "schedule proof result format",
                    "daily value proof matrix",
                    "daily value proof result format",
                    "daily value result format",
                    "morning brief proof matrix",
                    "morning brief proof result format",
                    "daily brief proof matrix",
                    "brief delivery proof matrix",
                    "brief proof matrix",
                    "what daily value should i test",
                    "what morning brief should i test",
                    "how should i report daily value proof",
                    "how should i report morning brief proof",
                    "morning brief delivery proof matrix",
                    "7 day scheduler proof",
                    "7-day scheduler proof",
                    "jobs 7 day proof",
                    "jobs seven day proof",
                    "scheduled jobs 7 day proof",
                    "scheduled jobs streak",
                    "scheduled job streak status",
                    "scheduler streak status",
                    "did scheduled jobs run for 7 days",
                    "are scheduled jobs running daily",
                    "what is the 7 day scheduler proof",
                    "what scheduler proof do you need",
                    "what scheduled jobs should i test",
                    "what scheduler streak should i test",
                    "how should i report scheduler proof",
                    "how should i report scheduled job results",
                    "how should i report scheduled job streak proof",
                    "daily streak proof",
                    "daily job streak proof",
                    "job streak proof matrix",
                    "scheduled delivery proof matrix",
                    "approval proof matrix",
                    "approval proof result format",
                    "approval flow proof matrix",
                    "approval flow result format",
                    "phone approval proof",
                    "phone approval proof matrix",
                    "phone approval result format",
                    "approval button proof matrix",
                    "approval button result format",
                    "approval buttons proof matrix",
                    "approval callback proof matrix",
                    "approval callback result format",
                    "approval from phone proof matrix",
                    "approval queue button proof matrix",
                    "approve deny proof matrix",
                    "approve dismiss proof matrix",
                    "approve from phone proof matrix",
                    "deny from phone proof matrix",
                    "how should i report approval button proof",
                    "how should i report approval callback proof",
                    "how should i report approve deny proof",
                    "how should i report phone approval proof",
                    "phone approval buttons proof matrix",
                    "phone approval-flow proof report format",
                    "phone approve deny proof matrix",
                    "risky action approval proof matrix",
                    "telegram approval buttons proof matrix",
                    "telegram approval result format",
                    "what approval buttons should i test",
                    "what approval callback should i test",
                    "what approve deny buttons should i test",
                    "what phone approval should i test",
                    "what phone approvals should i test",
                    "what approval proof do you need",
                    "what approval flow should i test",
                    "how should i report approval proof",
                    "how should i report phone approval results",
                    "phone control proof matrix",
                    "phone control proof result format",
                    "phone control live proof format",
                    "what broke proof matrix",
                    "cockpit proof matrix",
                    "cockpit approvals proof matrix",
                    "what phone control should i test",
                    "what phone shortcuts should i test",
                    "how should i report phone control proof",
                    "how should i report phone shortcut proof",
                    "reboot proof matrix",
                    "reboot proof result format",
                    "daemon proof matrix",
                    "daemon proof result format",
                    "daemon startup proof matrix",
                    "launchagent proof matrix",
                    "launchd proof matrix",
                    "post reboot proof matrix",
                    "post-reboot proof matrix",
                    "network recovery proof matrix",
                    "network recovery proof result format",
                    "network loss proof matrix",
                    "network loss proof result format",
                    "telegram control daemon status",
                    "telegram control proof matrix",
                    "status server proof matrix",
                    "scheduler daemon proof matrix",
                    "what reboot proof should i test",
                    "what daemon proof should i test",
                    "what network recovery should i test",
                    "what should i test after reboot",
                    "how should i report reboot proof",
                    "how should i report daemon proof",
                    "how should i report network recovery proof",
                    "error guidance proof matrix",
                    "error guidance proof result format",
                    "error message proof matrix",
                    "error message proof result format",
                    "recovery guidance proof matrix",
                    "user error proof matrix",
                    "what error messages should i test",
                    "what error messages should i test?",
                    "how should i report error guidance proof",
                    "how should i report error guidance proof?",
                    "are error messages actionable",
                    "connector error proof matrix",
                    "connector recovery proof matrix",
                    "do error messages name the fix",
                    "do user errors name the fix",
                    "error messages status",
                    "error recovery proof",
                    "error recovery status",
                    "how should i report error messages",
                    "how should i report error messages?",
                    "how should i report error recovery proof",
                    "how should i report error recovery proof?",
                    "is error guidance proven",
                    "recovery guidance status",
                    "show error guidance",
                    "show recovery guidance",
                    "user visible error proof matrix",
                    "user-visible error proof matrix",
                    "what recovery guidance should i test",
                    "what recovery guidance should i test?",
                    "what user facing errors need proof",
                    "what user facing errors need proof?",
                    "what user-facing errors need proof",
                    "what user-facing errors need proof?",
                    "오류 안내 증명 매트릭스",
                    "오류 안내 증명",
                    "오류 안내 형식",
                    "오류 메시지 상태",
                    "오류 복구 상태",
                    "복구 안내 증명",
                    "에러 안내 증명",
                    "report live matrix results",
                    "report live test results",
                    "test matrix status",
                    "what personal integrations should i test",
                    "what personal proof do you need",
                    "what personal proofs are pending",
                    "what personal proofs should i test",
                    "what channels need proof",
                    "what channels should i test",
                    "what is the live test matrix",
                    "what live channels are pending",
                    "what live proof do you need",
                    "what live proofs are pending",
                    "what live tests are pending",
                    "what results do you need from me",
                    "what should i send after testing",
                    "what should i test",
                    "what should operator test",
                    "what tests should i run",
                    "which channels need live proof",
                    "which channels should i test",
                } or compact in {
                    "검증대기",
                    "검증대기뭐야",
                    "남은라이브증거",
                    "남은라이브테스트",
                    "라이브검증대기",
                    "라이브채널대기",
                    "라이브채널뭐테스트해",
                    "라이브채널어떤거테스트해",
                    "라이브테스트결과",
                    "라이브테스트결과보고",
                    "라이브테스트대기",
                    "라이브테스트뭐해야해",
                    "라이브테스트상태",
                    "개인연동뭐테스트해",
                    "개인증명결과형식",
                    "개인증명매트릭스",
                    "개인증명뭐해야해",
                    "개인증명어떻게보고해",
                    "개인증명형식",
                    "스케줄증명결과형식",
                    "스케줄증명매트릭스",
                    "스케줄증명뭐해야해",
                    "스케줄증명어떻게보고해",
                    "예약작업증명형식",
                    "예약작업뭐테스트해",
                    "예약작업결과형식",
                    "승인증명결과형식",
                    "승인결과형식",
                    "승인증명매트릭스",
                    "승인증명뭐해야해",
                    "승인증명어떻게보고해",
                    "전화승인결과형식",
                    "전화승인증명",
                    "승인흐름증명",
                    "승인흐름뭐테스트해",
                    "라이브결과보고방법",
                    "라이브결과형식",
                    "라이브테스트결과형식",
                    "뭐테스트해야해",
                    "뭘테스트해야해",
                    "무슨테스트해야해",
                    "어떤테스트해야해",
                    "어떤채널검증해야해",
                    "어떤채널테스트해",
                    "채널검증뭐남았어",
                    "채널뭐테스트해",
                    "테스트결과보고방법",
                    "테스트결과어떻게보고해",
                    "테스트결과어떻게보내",
                    "테스트결과형식",
                    "테스트매트릭스",
                    "테스트매트릭스보여줘",
                    "테스트매트릭스상태",
                    "테스트대기",
                    "테스트해야할거",
                    "howdoireportlivetestresults",
                    "howshouldireportlivetestresults",
                    "howshouldireportthelivematrix",
                    "howtoreportlivetestresults",
                    "howshouldireportpersonalproofresults",
                    "howdoireportpersonalproofresults",
                    "livetestresultformat",
                    "livetestresultsformat",
                    "livematrixresultformat",
                    "personalliveproofformat",
                    "personalproofmatrix",
                    "personalproofresultformat",
                    "personalproofsmatrix",
                    "personalproofsresultformat",
                    "personalintegrationproofmatrix",
                    "personalintegrationresultformat",
                    "schedulerproofmatrix",
                    "schedulerproofresultformat",
                    "scheduledjobsproofmatrix",
                    "scheduledjobsresultformat",
                    "scheduleproofmatrix",
                    "scheduleproofresultformat",
                    "conversationresearchproofmatrix",
                    "conversationresearchproofresultformat",
                    "conversationandresearchproofmatrix",
                    "conversationandresearchresultformat",
                    "researchproofmatrix",
                    "researchproofresultformat",
                    "webresearchproofmatrix",
                    "chatresearchproofmatrix",
                    "sectionaproofmatrix",
                    "whatconversationresearchshoulditest",
                    "whatresearchproofshoulditest",
                    "howshouldireportconversationresearchproof",
                    "howshouldireportresearchproof",
                    "대화연구증명매트릭스",
                    "대화연구증명형식",
                    "연구증명매트릭스",
                    "연구증명형식",
                    "리서치증명매트릭스",
                    "리서치증명형식",
                    "검색증명매트릭스",
                    "대화연구뭐테스트해",
                    "연구증명뭐테스트해",
                    "연구증명어떻게보고해",
                    "dailyvalueproofmatrix",
                    "dailyvalueproofresultformat",
                    "dailyvalueresultformat",
                    "morningbriefproofmatrix",
                    "morningbriefproofresultformat",
                    "dailybriefproofmatrix",
                    "briefdeliveryproofmatrix",
                    "briefproofmatrix",
                    "whatdailyvalueshoulditest",
                    "whatmorningbriefshoulditest",
                    "howshouldireportdailyvalueproof",
                    "howshouldireportmorningbriefproof",
                    "일일가치증명매트릭스",
                    "일일가치증명형식",
                    "아침브리핑증명매트릭스",
                    "아침브리핑증명형식",
                    "브리핑증명매트릭스",
                    "브리핑증명형식",
                    "브리핑전송증명",
                    "브리핑뭐테스트해",
                    "아침브리핑뭐테스트해",
                    "일일가치뭐테스트해",
                    "브리핑어떻게보고해",
                    "7dayschedulerproof",
                    "jobs7dayproof",
                    "jobssevendayproof",
                    "scheduledjobs7dayproof",
                    "scheduledjobsstreak",
                    "scheduledjobstreakstatus",
                    "schedulerstreakstatus",
                    "didscheduledjobsrunfor7days",
                    "arescheduledjobsrunningdaily",
                    "whatisthe7dayschedulerproof",
                    "whatschedulerproofdoyouneed",
                    "whatscheduledjobsshoulditest",
                    "whatschedulerstreakshoulditest",
                    "howshouldireportschedulerproof",
                    "howshouldireportscheduledjobresults",
                    "howshouldireportscheduledjobstreakproof",
                    "dailystreakproof",
                    "dailyjobstreakproof",
                    "jobstreakproofmatrix",
                    "scheduleddeliveryproofmatrix",
                    "approvalproofmatrix",
                    "approvalproofresultformat",
                    "approvalflowproofmatrix",
                    "approvalflowresultformat",
                    "phoneapprovalproof",
                    "phoneapprovalproofmatrix",
                    "phoneapprovalresultformat",
                    "whatapprovalproofdoyouneed",
                    "whatapprovalflowshoulditest",
                    "approvalbuttonproofmatrix",
                    "approvalbuttonresultformat",
                    "approvalbuttonsproofmatrix",
                    "approvalcallbackproofmatrix",
                    "approvalcallbackresultformat",
                    "approvalfromphoneproofmatrix",
                    "approvalqueuebuttonproofmatrix",
                    "approvedenyproofmatrix",
                    "approvedismissproofmatrix",
                    "approvefromphoneproofmatrix",
                    "denyfromphoneproofmatrix",
                    "howshouldireportapprovalbuttonproof",
                    "howshouldireportapprovalcallbackproof",
                    "howshouldireportapprovedenyproof",
                    "howshouldireportphoneapprovalproof",
                    "phoneapprovalbuttonsproofmatrix",
                    "phoneapprovalflowproofreportformat",
                    "phoneapprovedenyproofmatrix",
                    "riskyactionapprovalproofmatrix",
                    "telegramapprovalbuttonsproofmatrix",
                    "telegramapprovalresultformat",
                    "whatapprovalbuttonsshoulditest",
                    "whatapprovalcallbackshoulditest",
                    "whatapprovedenybuttonsshoulditest",
                    "whatphoneapprovalshoulditest",
                    "whatphoneapprovalsshoulditest",
                    "승인거절증명",
                    "승인거절증명매트릭스",
                    "승인버튼결과형식",
                    "승인버튼증명",
                    "승인버튼증명매트릭스",
                    "승인콜백결과형식",
                    "승인콜백증명",
                    "승인콜백증명매트릭스",
                    "텔레그램승인결과형식",
                    "텔레그램승인증명",
                    "폰승인버튼증명",
                    "폰승인버튼증명매트릭스",
                    "howshouldireportapprovalproof",
                    "howshouldireportphoneapprovalresults",
                    "폰컨트롤증명매트릭스",
                    "폰제어증명형식",
                    "폰컨트롤결과형식",
                    "뭐가고장났어증명형식",
                    "콕핏증명매트릭스",
                    "콕핏승인증명형식",
                    "폰컨트롤뭐테스트해",
                    "폰제어뭐테스트해",
                    "폰컨트롤어떻게보고해",
                    "폰제어어떻게보고해",
                    "phonecontrolproofmatrix",
                    "phonecontrolproofresultformat",
                    "phonecontrolliveproofformat",
                    "whatbrokeproofmatrix",
                    "cockpitproofmatrix",
                    "cockpitapprovalsproofmatrix",
                    "whatphonecontrolshoulditest",
                    "whatphoneshortcutsshoulditest",
                    "howshouldireportphonecontrolproof",
                    "howshouldireportphoneshortcutproof",
                    "재부팅증명매트릭스",
                    "재부팅증명형식",
                    "데몬증명매트릭스",
                    "데몬증명형식",
                    "데몬시작증명",
                    "런치에이전트증명",
                    "네트워크복구증명",
                    "네트워크끊김증명",
                    "재부팅뭐테스트해",
                    "데몬뭐테스트해",
                    "재부팅어떻게보고해",
                    "데몬어떻게보고해",
                    "rebootproofmatrix",
                    "rebootproofresultformat",
                    "daemonproofmatrix",
                    "daemonproofresultformat",
                    "daemonstartupproofmatrix",
                    "launchagentproofmatrix",
                    "networkrecoveryproofmatrix",
                    "networklossproofmatrix",
                    "whatrebootproofshoulditest",
                    "whatdaemonproofshoulditest",
                    "whatnetworkrecoveryshoulditest",
                    "whatshoulditestafterreboot",
                    "howshouldireportrebootproof",
                    "howshouldireportdaemonproof",
                    "howshouldireportnetworkrecoveryproof",
                    "대화증명결과형식",
                    "대화증명매트릭스",
                    "대화증명뭐해야해",
                    "대화증명어떻게보고해",
                    "혼합대화증명",
                    "혼합대화뭐테스트해",
                    "대화지연증명형식",
                    "챗지연증명형식",
                    "mixedconversationproofmatrix",
                    "mixedconversationproofresultformat",
                    "conversationproofmatrix",
                    "conversationproofresultformat",
                    "chatlatencyproofmatrix",
                    "chatlatencyproofresultformat",
                    "whatmixedconversationshoulditest",
                    "whatconversationshoulditest",
                    "whatchatproofshouldirun",
                    "howshouldireportmixedconversationproof",
                    "howshouldireportconversationproof",
                    "howshouldireportchatlatencyproof",
                    "voiceproofmatrix",
                    "voiceproofresultformat",
                    "voiceliveproofformat",
                    "telegramvoiceproofmatrix",
                    "telegramvoiceproofresultformat",
                    "localvoiceproofmatrix",
                    "localvoiceproofresultformat",
                    "koreanvoiceproofmatrix",
                    "koreanvoiceproofresultformat",
                    "pushtotalkproofmatrix",
                    "talkpyproofmatrix",
                    "spokenreplyproofmatrix",
                    "voicespeakproofmatrix",
                    "whatvoiceshoulditest",
                    "whatvoiceproofsshoulditest",
                    "whatlocalvoiceshoulditest",
                    "whattelegramvoiceshoulditest",
                    "whatkoreanvoiceshoulditest",
                    "howshouldireportvoiceproof",
                    "howshouldireportlocalvoiceproof",
                    "howshouldireporttelegramvoiceproof",
                    "howshouldireportkoreanvoiceproof",
                    "음성증명결과형식",
                    "음성증명매트릭스",
                    "음성증명뭐해야해",
                    "음성증명어떻게보고해",
                    "음성뭐테스트해",
                    "로컬음성증명",
                    "텔레그램음성증명",
                    "한국어음성증명형식",
                    "텔레그램음성증명형식",
                    "reportlivematrixresults",
                    "reportlivetestresults",
                    "testmatrixstatus",
                    "whatformatshouldiuseforlivetestresults",
                    "whatpersonalintegrationsshoulditest",
                    "whatpersonalproofdoyouneed",
                    "whatpersonalproofsarepending",
                    "whatpersonalproofsshoulditest",
                    "whatchannelsneedproof",
                    "whatchannelsshoulditest",
                    "whatisthelivetestmatrix",
                    "whatlivechannelsarepending",
                    "whatliveproofdoyouneed",
                    "whatliveproofsarepending",
                    "whatlivetestsarepending",
                    "whatresultsdoyouneedfromme",
                    "whatshouldisendaftertesting",
                    "whatshoulditest",
                    "whatshouldoperatortest",
                    "whattestsshouldirun",
                    "whichchannelsneedliveproof",
                    "whichchannelsshoulditest",
                }:
                    if low in {
                        "conversation research proof matrix",
                        "conversation research proof result format",
                        "conversation and research proof matrix",
                        "conversation and research result format",
                        "research proof matrix",
                        "research proof result format",
                        "web research proof matrix",
                        "chat research proof matrix",
                        "section a proof matrix",
                        "what conversation research should i test",
                        "what research proof should i test",
                        "how should i report conversation research proof",
                        "how should i report research proof",
                    } or compact in {
                        "conversationresearchproofmatrix",
                        "conversationresearchproofresultformat",
                        "conversationandresearchproofmatrix",
                        "conversationandresearchresultformat",
                        "researchproofmatrix",
                        "researchproofresultformat",
                        "webresearchproofmatrix",
                        "chatresearchproofmatrix",
                        "sectionaproofmatrix",
                        "whatconversationresearchshoulditest",
                        "whatresearchproofshoulditest",
                        "howshouldireportconversationresearchproof",
                        "howshouldireportresearchproof",
                        "대화연구증명매트릭스",
                        "대화연구증명형식",
                        "연구증명매트릭스",
                        "연구증명형식",
                        "리서치증명매트릭스",
                        "리서치증명형식",
                        "검색증명매트릭스",
                        "대화연구뭐테스트해",
                        "연구증명뭐테스트해",
                        "연구증명어떻게보고해",
                    }:
                        return self._format_conversation_research_proof_matrix(), None
                    if low in {
                        "phone control proof matrix",
                        "phone control proof result format",
                        "phone control live proof format",
                        "what broke proof matrix",
                        "cockpit proof matrix",
                        "cockpit approvals proof matrix",
                        "what phone control should i test",
                        "what phone shortcuts should i test",
                        "how should i report phone control proof",
                        "how should i report phone shortcut proof",
                    } or compact in {
                        "폰컨트롤증명매트릭스",
                        "폰제어증명형식",
                        "폰컨트롤결과형식",
                        "뭐가고장났어증명형식",
                        "콕핏증명매트릭스",
                        "콕핏승인증명형식",
                        "폰컨트롤뭐테스트해",
                        "폰제어뭐테스트해",
                        "폰컨트롤어떻게보고해",
                        "폰제어어떻게보고해",
                        "phonecontrolproofmatrix",
                        "phonecontrolproofresultformat",
                        "phonecontrolliveproofformat",
                        "whatbrokeproofmatrix",
                        "cockpitproofmatrix",
                        "cockpitapprovalsproofmatrix",
                        "whatphonecontrolshoulditest",
                        "whatphoneshortcutsshoulditest",
                        "howshouldireportphonecontrolproof",
                        "howshouldireportphoneshortcutproof",
                    }:
                        return self._format_phone_control_proof_matrix(), None
                    if low in {
                        "reboot proof matrix",
                        "reboot proof result format",
                        "daemon proof matrix",
                        "daemon proof result format",
                        "daemon startup proof matrix",
                        "launchagent proof matrix",
                        "launchd proof matrix",
                        "post reboot proof matrix",
                        "post-reboot proof matrix",
                        "network recovery proof matrix",
                        "network recovery proof result format",
                        "network loss proof matrix",
                        "network loss proof result format",
                        "telegram control daemon status",
                        "telegram control proof matrix",
                        "status server proof matrix",
                        "scheduler daemon proof matrix",
                        "what reboot proof should i test",
                        "what daemon proof should i test",
                        "what network recovery should i test",
                        "what should i test after reboot",
                        "how should i report reboot proof",
                        "how should i report daemon proof",
                        "how should i report network recovery proof",
                    } or compact in {
                        "재부팅증명매트릭스",
                        "재부팅증명형식",
                        "데몬증명매트릭스",
                        "데몬증명형식",
                        "데몬시작증명",
                        "런치에이전트증명",
                        "네트워크복구증명",
                        "네트워크끊김증명",
                        "재부팅뭐테스트해",
                        "데몬뭐테스트해",
                        "재부팅어떻게보고해",
                        "데몬어떻게보고해",
                        "rebootproofmatrix",
                        "rebootproofresultformat",
                        "daemonproofmatrix",
                        "daemonproofresultformat",
                        "daemonstartupproofmatrix",
                        "launchagentproofmatrix",
                        "launchdproofmatrix",
                        "postrebootproofmatrix",
                        "networkrecoveryproofmatrix",
                        "networkrecoveryproofresultformat",
                        "networklossproofmatrix",
                        "networklossproofresultformat",
                        "telegramcontroldaemonstatus",
                        "telegramcontrolproofmatrix",
                        "statusserverproofmatrix",
                        "schedulerdaemonproofmatrix",
                        "whatrebootproofshoulditest",
                        "whatdaemonproofshoulditest",
                        "whatnetworkrecoveryshoulditest",
                        "whatshoulditestafterreboot",
                        "howshouldireportrebootproof",
                        "howshouldireportdaemonproof",
                        "howshouldireportnetworkrecoveryproof",
                    }:
                        return self._format_reboot_proof_matrix(), None
                    if low in {
                        "error guidance proof matrix",
                        "error guidance proof result format",
                        "error message proof matrix",
                        "error message proof result format",
                        "recovery guidance proof matrix",
                        "user error proof matrix",
                        "what error messages should i test",
                        "what error messages should i test?",
                        "how should i report error guidance proof",
                        "how should i report error guidance proof?",
                        "are error messages actionable",
                        "connector error proof matrix",
                        "connector recovery proof matrix",
                        "do error messages name the fix",
                        "do user errors name the fix",
                        "error messages status",
                        "error recovery proof",
                        "error recovery status",
                        "how should i report error messages",
                        "how should i report error messages?",
                        "how should i report error recovery proof",
                        "how should i report error recovery proof?",
                        "is error guidance proven",
                        "recovery guidance status",
                        "show error guidance",
                        "show recovery guidance",
                        "user visible error proof matrix",
                        "user-visible error proof matrix",
                        "what recovery guidance should i test",
                        "what recovery guidance should i test?",
                        "what user facing errors need proof",
                        "what user facing errors need proof?",
                        "what user-facing errors need proof",
                        "what user-facing errors need proof?",
                        "오류 안내 증명 매트릭스",
                        "오류 안내 증명",
                        "오류 안내 형식",
                        "오류 메시지 상태",
                        "오류 복구 상태",
                        "복구 안내 증명",
                        "에러 안내 증명",
                    } or compact in {
                        "오류안내증명매트릭스",
                        "오류안내형식",
                        "에러안내증명",
                        "복구안내증명",
                        "오류메시지증명",
                        "오류뭐테스트해",
                        "오류어떻게보고해",
                        "errorguidanceproofmatrix",
                        "errorguidanceproofresultformat",
                        "errormessageproofmatrix",
                        "errormessageproofresultformat",
                        "recoveryguidanceproofmatrix",
                        "usererrorproofmatrix",
                        "whaterrormessagesshoulditest",
                        "howshouldireporterrorguidanceproof",
                        "areerrormessagesactionable",
                        "connectorerrorproofmatrix",
                        "connectorrecoveryproofmatrix",
                        "doerrormessagesnamethefix",
                        "dousererrorsnamethefix",
                        "errormessagesstatus",
                        "errorrecoveryproof",
                        "errorrecoverystatus",
                        "howshouldireporterrormessages",
                        "howshouldireporterrorrecoveryproof",
                        "iserrorguidanceproven",
                        "recoveryguidancestatus",
                        "showerrorguidance",
                        "showrecoveryguidance",
                        "uservisibleerrorproofmatrix",
                        "whatrecoveryguidanceshoulditest",
                        "whatuserfacingerrorsneedproof",
                        "복구안내증명",
                        "오류안내증명",
                        "오류메시지상태",
                        "오류복구상태",
                    }:
                        return self._format_error_guidance_proof_matrix(), None
                    if low in {
                        "daily value proof matrix",
                        "daily value proof result format",
                        "daily value result format",
                        "morning brief proof matrix",
                        "morning brief proof result format",
                        "daily brief proof matrix",
                        "brief delivery proof matrix",
                        "brief proof matrix",
                        "what daily value should i test",
                        "what morning brief should i test",
                        "how should i report daily value proof",
                        "how should i report morning brief proof",
                    } or compact in {
                        "dailyvalueproofmatrix",
                        "dailyvalueproofresultformat",
                        "dailyvalueresultformat",
                        "morningbriefproofmatrix",
                        "morningbriefproofresultformat",
                        "dailybriefproofmatrix",
                        "briefdeliveryproofmatrix",
                        "briefproofmatrix",
                        "whatdailyvalueshoulditest",
                        "whatmorningbriefshoulditest",
                        "howshouldireportdailyvalueproof",
                        "howshouldireportmorningbriefproof",
                        "일일가치증명매트릭스",
                        "일일가치증명형식",
                        "아침브리핑증명매트릭스",
                        "아침브리핑증명형식",
                        "브리핑증명매트릭스",
                        "브리핑증명형식",
                        "브리핑전송증명",
                        "브리핑뭐테스트해",
                        "아침브리핑뭐테스트해",
                        "일일가치뭐테스트해",
                        "브리핑어떻게보고해",
                    }:
                        return self._format_daily_value_proof_matrix(), None
                    if low in {
                        "approval proof matrix",
                        "approval proof result format",
                        "approval flow proof matrix",
                        "approval flow result format",
                        "phone approval proof",
                        "phone approval proof matrix",
                        "phone approval result format",
                        "approval button proof matrix",
                        "approval button result format",
                        "approval buttons proof matrix",
                        "approval callback proof matrix",
                        "approval callback result format",
                        "approval from phone proof matrix",
                        "approval queue button proof matrix",
                        "approve deny proof matrix",
                        "approve dismiss proof matrix",
                        "approve from phone proof matrix",
                        "deny from phone proof matrix",
                        "how should i report approval button proof",
                        "how should i report approval callback proof",
                        "how should i report approve deny proof",
                        "how should i report phone approval proof",
                        "phone approval buttons proof matrix",
                        "phone approval-flow proof report format",
                        "phone approve deny proof matrix",
                        "risky action approval proof matrix",
                        "telegram approval buttons proof matrix",
                        "telegram approval result format",
                        "what approval buttons should i test",
                        "what approval callback should i test",
                        "what approve deny buttons should i test",
                        "what phone approval should i test",
                        "what phone approvals should i test",
                        "what approval proof do you need",
                        "what approval flow should i test",
                        "how should i report approval proof",
                        "how should i report phone approval results",
                    } or compact in {
                        "승인증명결과형식",
                        "승인결과형식",
                        "승인증명매트릭스",
                        "승인증명뭐해야해",
                        "승인증명어떻게보고해",
                        "전화승인결과형식",
                        "전화승인증명",
                        "승인흐름증명",
                        "승인흐름뭐테스트해",
                        "approvalproofmatrix",
                        "approvalproofresultformat",
                        "approvalflowproofmatrix",
                        "approvalflowresultformat",
                        "phoneapprovalproof",
                        "phoneapprovalproofmatrix",
                        "phoneapprovalresultformat",
                        "approvalbuttonproofmatrix",
                        "approvalbuttonresultformat",
                        "approvalbuttonsproofmatrix",
                        "approvalcallbackproofmatrix",
                        "approvalcallbackresultformat",
                        "approvalfromphoneproofmatrix",
                        "approvalqueuebuttonproofmatrix",
                        "approvedenyproofmatrix",
                        "approvedismissproofmatrix",
                        "approvefromphoneproofmatrix",
                        "denyfromphoneproofmatrix",
                        "howshouldireportapprovalbuttonproof",
                        "howshouldireportapprovalcallbackproof",
                        "howshouldireportapprovedenyproof",
                        "howshouldireportphoneapprovalproof",
                        "phoneapprovalbuttonsproofmatrix",
                        "phoneapprovalflowproofreportformat",
                        "phoneapprovedenyproofmatrix",
                        "riskyactionapprovalproofmatrix",
                        "telegramapprovalbuttonsproofmatrix",
                        "telegramapprovalresultformat",
                        "whatapprovalbuttonsshoulditest",
                        "whatapprovalcallbackshoulditest",
                        "whatapprovedenybuttonsshoulditest",
                        "whatphoneapprovalshoulditest",
                        "whatphoneapprovalsshoulditest",
                        "승인거절증명",
                        "승인거절증명매트릭스",
                        "승인버튼결과형식",
                        "승인버튼증명",
                        "승인버튼증명매트릭스",
                        "승인콜백결과형식",
                        "승인콜백증명",
                        "승인콜백증명매트릭스",
                        "텔레그램승인결과형식",
                        "텔레그램승인증명",
                        "폰승인버튼증명",
                        "폰승인버튼증명매트릭스",
                        "whatapprovalproofdoyouneed",
                        "whatapprovalflowshoulditest",
                        "howshouldireportapprovalproof",
                        "howshouldireportphoneapprovalresults",
                    }:
                        return self._format_approval_proof_matrix(), None
                    if low in {
                        "mixed conversation proof matrix",
                        "mixed conversation proof result format",
                        "conversation proof matrix",
                        "conversation proof result format",
                        "chat latency proof matrix",
                        "chat latency proof result format",
                        "what mixed conversation should i test",
                        "what conversation should i test",
                        "what chat proof should i run",
                        "how should i report mixed conversation proof",
                        "how should i report conversation proof",
                        "how should i report chat latency proof",
                    } or compact in {
                        "대화증명결과형식",
                        "대화증명매트릭스",
                        "대화증명뭐해야해",
                        "대화증명어떻게보고해",
                        "혼합대화증명",
                        "혼합대화뭐테스트해",
                        "대화지연증명형식",
                        "챗지연증명형식",
                        "mixedconversationproofmatrix",
                        "mixedconversationproofresultformat",
                        "conversationproofmatrix",
                        "conversationproofresultformat",
                        "chatlatencyproofmatrix",
                        "chatlatencyproofresultformat",
                        "whatmixedconversationshoulditest",
                        "whatconversationshoulditest",
                        "whatchatproofshouldirun",
                        "howshouldireportmixedconversationproof",
                        "howshouldireportconversationproof",
                        "howshouldireportchatlatencyproof",
                    }:
                        return self._format_mixed_conversation_proof_matrix(), None
                    if low in {
                        "voice proof matrix",
                        "voice proof result format",
                        "voice live proof format",
                        "telegram voice proof matrix",
                        "telegram voice proof result format",
                        "local voice proof matrix",
                        "local voice proof result format",
                        "korean voice proof matrix",
                        "korean voice proof result format",
                        "push to talk proof matrix",
                        "push-to-talk proof matrix",
                        "talk.py proof matrix",
                        "talk py proof matrix",
                        "spoken reply proof matrix",
                        "voice speak proof matrix",
                        "what voice should i test",
                        "what voice proofs should i test",
                        "what local voice should i test",
                        "what telegram voice should i test",
                        "what korean voice should i test",
                        "how should i report voice proof",
                        "how should i report local voice proof",
                        "how should i report telegram voice proof",
                        "how should i report korean voice proof",
                    } or compact in {
                        "음성증명결과형식",
                        "음성증명매트릭스",
                        "음성증명뭐해야해",
                        "음성증명어떻게보고해",
                        "음성뭐테스트해",
                        "로컬음성증명",
                        "텔레그램음성증명",
                        "한국어음성증명형식",
                        "텔레그램음성증명형식",
                        "voiceproofmatrix",
                        "voiceproofresultformat",
                        "voiceliveproofformat",
                        "telegramvoiceproofmatrix",
                        "telegramvoiceproofresultformat",
                        "localvoiceproofmatrix",
                        "localvoiceproofresultformat",
                        "koreanvoiceproofmatrix",
                        "koreanvoiceproofresultformat",
                        "pushtotalkproofmatrix",
                        "talkpyproofmatrix",
                        "spokenreplyproofmatrix",
                        "voicespeakproofmatrix",
                        "whatvoiceshoulditest",
                        "whatvoiceproofsshoulditest",
                        "whatlocalvoiceshoulditest",
                        "whattelegramvoiceshoulditest",
                        "whatkoreanvoiceshoulditest",
                        "howshouldireportvoiceproof",
                        "howshouldireportlocalvoiceproof",
                        "howshouldireporttelegramvoiceproof",
                        "howshouldireportkoreanvoiceproof",
                    }:
                        return self._format_voice_proof_matrix(), None
                    if low in {
                        "how should i report personal proof results",
                        "how do i report personal proof results",
                        "personal live proof format",
                        "personal proof matrix",
                        "personal proof result format",
                        "personal proofs matrix",
                        "personal proofs result format",
                        "personal integration proof matrix",
                        "personal integration result format",
                        "personal integrations proof matrix",
                        "personal integrations result format",
                        "calendar create update delete proof matrix",
                        "calendar write proof matrix",
                        "calendar write result format",
                        "what calendar write should i test",
                        "how should i report calendar write proof",
                        "email read proof matrix",
                        "email search proof matrix",
                        "email send proof matrix",
                        "what email read should i test",
                        "what email search should i test",
                        "what email send should i test",
                        "how should i report email read proof",
                        "how should i report email search proof",
                        "how should i report email send proof",
                        "reminder proof matrix",
                        "set reminder proof matrix",
                        "what reminder proof should i test",
                        "what set reminder proof should i test",
                        "how should i report reminder proof",
                        "how should i report set reminder proof",
                        "contact lookup proof matrix",
                        "what contact lookup should i test",
                        "how should i report contact lookup proof",
                        "캘린더 쓰기 증명 매트릭스",
                        "이메일 읽기 증명",
                        "이메일 검색 증명",
                        "이메일 보내기 증명",
                        "리마인더 증명 매트릭스",
                        "연락처 조회 증명 매트릭스",
                        "what personal integrations should i test",
                        "what personal proof do you need",
                        "what personal proofs are pending",
                        "what personal proofs should i test",
                    } or compact in {
                        "개인연동뭐테스트해",
                        "개인증명결과형식",
                        "개인증명매트릭스",
                        "개인증명뭐해야해",
                        "개인증명어떻게보고해",
                        "개인증명형식",
                        "howshouldireportpersonalproofresults",
                        "howdoireportpersonalproofresults",
                        "personalliveproofformat",
                        "personalproofmatrix",
                        "personalproofresultformat",
                        "personalproofsmatrix",
                        "personalproofsresultformat",
                        "personalintegrationproofmatrix",
                        "personalintegrationresultformat",
                        "personalintegrationsproofmatrix",
                        "personalintegrationsresultformat",
                        "calendarcreateupdatedeleteproofmatrix",
                        "calendarwriteproofmatrix",
                        "calendarwriteresultformat",
                        "whatcalendarwriteshoulditest",
                        "howshouldireportcalendarwriteproof",
                        "emailreadproofmatrix",
                        "emailsearchproofmatrix",
                        "emailsendproofmatrix",
                        "whatemailreadshoulditest",
                        "whatemailsearchshoulditest",
                        "whatemailsendshoulditest",
                        "howshouldireportemailreadproof",
                        "howshouldireportemailsearchproof",
                        "howshouldireportemailsendproof",
                        "reminderproofmatrix",
                        "setreminderproofmatrix",
                        "whatreminderproofshoulditest",
                        "whatsetreminderproofshoulditest",
                        "howshouldireportreminderproof",
                        "howshouldireportsetreminderproof",
                        "contactlookupproofmatrix",
                        "whatcontactlookupshoulditest",
                        "howshouldireportcontactlookupproof",
                        "캘린더쓰기증명매트릭스",
                        "이메일읽기증명",
                        "이메일검색증명",
                        "이메일보내기증명",
                        "리마인더증명매트릭스",
                        "연락처조회증명매트릭스",
                        "whatpersonalintegrationsshoulditest",
                        "whatpersonalproofdoyouneed",
                        "whatpersonalproofsarepending",
                        "whatpersonalproofsshoulditest",
                    }:
                        return self._format_personal_proof_matrix(), None
                    if low in {
                        "scheduler proof matrix",
                        "scheduler proof result format",
                        "scheduled jobs proof matrix",
                        "scheduled jobs result format",
                        "schedule proof matrix",
                        "schedule proof result format",
                        "morning brief delivery proof matrix",
                        "7 day scheduler proof",
                        "7-day scheduler proof",
                        "jobs 7 day proof",
                        "jobs seven day proof",
                        "scheduled jobs 7 day proof",
                        "scheduled jobs streak",
                        "scheduled job streak status",
                        "scheduler streak status",
                        "did scheduled jobs run for 7 days",
                        "are scheduled jobs running daily",
                        "what is the 7 day scheduler proof",
                        "what scheduler proof do you need",
                        "what scheduled jobs should i test",
                        "what scheduler streak should i test",
                        "how should i report scheduler proof",
                        "how should i report scheduled job results",
                        "how should i report scheduled job streak proof",
                        "daily streak proof",
                        "daily job streak proof",
                        "job streak proof matrix",
                        "scheduled delivery proof matrix",
                    } or compact in {
                        "스케줄증명결과형식",
                        "스케줄증명매트릭스",
                        "스케줄증명뭐해야해",
                        "스케줄증명어떻게보고해",
                        "예약작업증명형식",
                        "예약작업뭐테스트해",
                        "예약작업결과형식",
                        "schedulerproofmatrix",
                        "schedulerproofresultformat",
                        "scheduledjobsproofmatrix",
                        "scheduledjobsresultformat",
                        "scheduleproofmatrix",
                        "scheduleproofresultformat",
                        "morningbriefdeliveryproofmatrix",
                        "7dayschedulerproof",
                        "jobs7dayproof",
                        "jobssevendayproof",
                        "scheduledjobs7dayproof",
                        "scheduledjobsstreak",
                        "scheduledjobstreakstatus",
                        "schedulerstreakstatus",
                        "didscheduledjobsrunfor7days",
                        "arescheduledjobsrunningdaily",
                        "whatisthe7dayschedulerproof",
                        "whatschedulerproofdoyouneed",
                        "whatscheduledjobsshoulditest",
                        "whatschedulerstreakshoulditest",
                        "howshouldireportschedulerproof",
                        "howshouldireportscheduledjobresults",
                        "howshouldireportscheduledjobstreakproof",
                        "dailystreakproof",
                        "dailyjobstreakproof",
                        "jobstreakproofmatrix",
                        "scheduleddeliveryproofmatrix",
                    }:
                        return self._format_scheduler_proof_matrix(), None
                    return self._format_live_test_matrix(), None
                return self._format_cockpit_text(), None
            except Exception as e:
                return _runtime_error_reply(e), None
        try:
            result = self._runtime_instance().handle(
                command,
                request_token=request_token,
            )
            approval_id = _approval_id_from_result(result)
            reply_markup = _approval_keyboard(approval_id) if approval_id is not None else None
            return _result_response_text(result), reply_markup
        except Exception as e:
            return _runtime_error_reply(e), None

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop = stop_event or _STOP
        while not stop.is_set():
            self.process_once()
            stop.wait(0.5)


def start_telegram_control() -> str:
    global _THREAD
    token_status = _token_status()
    owner_status = _owner_status()
    if not token_status["configured"]:
        return "Telegram control disabled: TELEGRAM_BOT_TOKEN is not set."
    if not token_status["valid"]:
        return "Telegram control disabled: TELEGRAM_BOT_TOKEN is invalid (value hidden)."
    if not owner_status["configured"]:
        return "Telegram control disabled: JARVIS_OWNER_TELEGRAM is not set."
    if not owner_status["valid"]:
        return "Telegram control disabled: JARVIS_OWNER_TELEGRAM is invalid (value hidden)."
    if _THREAD is not None and _THREAD.is_alive():
        return "Telegram control already running."
    _STOP.clear()
    bridge = TelegramCommandBridge()
    _THREAD = threading.Thread(target=bridge.run_forever, name="JarvisV2TelegramControl", daemon=True)
    _THREAD.start()
    return "Telegram control started."


def stop_telegram_control() -> str:
    _STOP.set()
    return "Telegram control stopped."
