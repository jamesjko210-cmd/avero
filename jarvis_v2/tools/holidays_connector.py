"""Public holidays for Jarvis V2 (nager.date, free, no key)."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import date
from typing import Any

from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools._http import http_get, http_get_json, friendly_http_error


FALLBACK_COUNTRY = "KR"
DEFAULT_COUNTRY = FALLBACK_COUNTRY
MAX_RAW_COUNTRY_CHARS = 80
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
COUNTRY_ALIASES = {
    "korea": "KR", "korean": "KR", "kr": "KR", "south korea": "KR",
    "us": "US", "usa": "US", "america": "US", "american": "US",
    "japan": "JP", "japanese": "JP", "jp": "JP", "uk": "GB", "britain": "GB", "british": "GB", "england": "GB",
    "united kingdom": "GB", "great britain": "GB", "gb": "GB",
    "france": "FR", "french": "FR", "fr": "FR",
    "germany": "DE", "german": "DE", "de": "DE",
    "canada": "CA", "canadian": "CA", "ca": "CA",
    "australia": "AU", "australian": "AU", "au": "AU",
    "china": "CN", "chinese": "CN", "cn": "CN",
    "taiwan": "TW", "tw": "TW",
    "hong kong": "HK", "hk": "HK",
    "singapore": "SG", "sg": "SG",
    "india": "IN", "indian": "IN", "in": "IN",
    "italy": "IT", "italian": "IT", "it": "IT",
    "spain": "ES", "spanish": "ES", "es": "ES",
    "mexico": "MX", "mexican": "MX", "mx": "MX",
    "brazil": "BR", "brazilian": "BR", "br": "BR",
    "philippines": "PH", "philippine": "PH", "ph": "PH",
    "vietnam": "VN", "viet nam": "VN", "vn": "VN",
    "thailand": "TH", "thai": "TH", "th": "TH",
    "indonesia": "ID", "indonesian": "ID", "id": "ID",
    "netherlands": "NL", "holland": "NL", "nl": "NL",
}


def _env_country(name: str, fallback: str = FALLBACK_COUNTRY) -> str:
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    alias = COUNTRY_ALIASES.get(raw.lower())
    country = alias or raw.upper()
    if not re.fullmatch(r"[A-Z]{2}", country):
        return fallback
    return country


DEFAULT_COUNTRY = _env_country("JARVIS_HOLIDAY_COUNTRY")


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


def _holiday_boundaries(*, calls_external_service: bool = False) -> dict[str, bool]:
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


def _short_raw(value: Any, limit: int = MAX_RAW_COUNTRY_CHARS) -> str:
    text = "" if value is None else str(value)
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _holiday_rows(items: list[dict[str, Any]], *, limit: int = 4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items[:limit]:
        rows.append(
            {
                "date": str(item.get("date") or ""),
                "name": str(item.get("name") or "")[:120],
                "local_name": str(item.get("localName") or "")[:120],
            }
        )
    return rows


def _holidays_handoff(
    *,
    country_metadata: dict[str, Any],
    status: str,
    reason: str = "",
    items: list[dict[str, Any]] | None = None,
    today_holiday: dict[str, Any] | None = None,
    calls_external_service: bool = False,
    exception_type: str = "",
) -> dict[str, Any]:
    rows = _holiday_rows(items or [])
    next_safe_command = f"next holidays {country_metadata.get('country') or '<country>'}"
    boundaries = _holiday_boundaries(calls_external_service=calls_external_service)
    handoff = {
        "source": "next_holidays",
        "holidays_handoff_ready": True,
        "handoff_ready": True,
        "ready_for_operator": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "status": status,
        "reason": reason,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "country": country_metadata.get("country", ""),
        "country_source": country_metadata.get("source", ""),
        "raw_country": country_metadata.get("raw_country", ""),
        "holiday_count": len(items or []),
        "preview_rows": rows,
        "preview_row_count": len(rows),
        "today_holiday": bool(today_holiday),
        "today_holiday_name": str((today_holiday or {}).get("name") or ""),
        "exception_type": exception_type,
        "content_in_metadata": False,
        "retry_safe": status in {"empty", "unavailable"},
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "holidays_handoff_ready": True,
        "holidays_ready_for_operator": handoff["ready_for_operator"],
        "holidays_state_changed": handoff["state_changed"],
        "holidays_changed": handoff["changed"],
        "holidays_content_in_handoff": handoff["content_in_handoff"],
        "holidays_content_in_metadata": handoff["content_in_metadata"],
        "holidays_next_safe_command": handoff["next_safe_command"],
        "holidays_next_safe_commands": handoff["next_safe_commands"],
        "holidays_next_safe_command_count": handoff["next_safe_command_count"],
        "holidays_authorizes_execution": handoff["authorizes_execution"],
        "holidays_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "holidays_approval_granted": handoff["approval_granted"],
        "holidays_boundaries": boundaries,
        "holidays_handoff": handoff,
    }


def _fetch_next(country: str) -> list:
    url = f"https://date.nager.at/api/v3/NextPublicHolidays/{country}"
    return http_get_json(url, headers={"User-Agent": "Mozilla/5.0"})


def _holidays_recovery_message(prefix: str) -> str:
    return f"{prefix} Check network access to Nager.Date holidays, run setup check, then retry in a moment."


def _holidays_error_message(e: Exception, *, country: str) -> str:
    friendly = friendly_http_error(e, subject=f"holidays for {country}", service="The holidays service")
    lowered = friendly.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return _holidays_recovery_message("The holidays request timed out.")
    if friendly.startswith("I couldn't find"):
        return friendly
    if friendly.startswith("Error:") or "having trouble" in lowered or "try again" in lowered:
        return _holidays_recovery_message("The holidays service is having trouble right now.")
    return friendly


def _clean_country_text(value: str) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    text = re.sub(r"^[\"'`]*(?:the\s+)?", "", text)
    text = re.sub(r"[\"'`.,!?;:]+$", "", text).strip()
    return text


def _country_code(value: str) -> str | None:
    text = _clean_country_text(value)
    if not text:
        return None
    alias = COUNTRY_ALIASES.get(text)
    if alias:
        return alias
    if re.fullmatch(r"[a-z]{2}", text):
        return text.upper()
    return None


def _explicit_country_from_text(text: str) -> tuple[str, str] | None:
    low = " ".join(str(text or "").lower().split())
    patterns = [
        r"\b(?:holidays?|public holidays?)\s+(?:in|for)\s+(?P<country>[a-z][a-z .'-]{1,40})\b",
        r"\b(?:in|for)\s+(?P<country>[a-z][a-z .'-]{1,40})\s+(?:holidays?|public holidays?)\b",
        r"^(?:next|upcoming)\s+(?P<country>[a-z][a-z .'-]{1,40})\s+(?:public\s+)?holidays?\b",
        r"^is\s+(?:today|it)\s+a\s+holiday\s+(?:in|for)\s+(?P<country>[a-z][a-z .'-]{1,40})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, low)
        if match:
            raw = _clean_country_text(match.group("country"))
            raw = re.sub(r"\s+(?:today|this year|tomorrow|now)$", "", raw).strip()
            if raw:
                return raw, (_country_code(raw) or "")
    return None


def _resolve_country(args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    raw_country = args.get("country")
    country_text = str(raw_country or "").strip()
    if country_text:
        country = _country_code(country_text) or country_text.upper()
        metadata = {"raw_country": _short_raw(raw_country), "country": country, "source": "country"}
        if not re.fullmatch(r"[A-Z]{2}", country):
            return None, metadata
        return country, metadata
    raw_text = str(args.get("text") or "")
    explicit = _explicit_country_from_text(raw_text)
    if explicit:
        raw, country = explicit
        metadata = {"raw_country": _short_raw(raw), "country": country, "source": "text"}
        if country:
            return country, metadata
        return None, metadata
    country = _country_code(raw_text) or DEFAULT_COUNTRY
    return country, {"raw_country": "", "country": country, "source": "text_or_default"}


def make_holidays_tools(config: JarvisConfig):
    def next_holidays(args: dict[str, Any]) -> ToolResult:
        country, country_metadata = _resolve_country(args)
        metadata = dict(country_metadata)
        if not country:
            reason = "unknown_country" if country_metadata.get("source") == "text" else "invalid_country"
            output = (
                "I don't know that holiday country yet. Try a common country like Korea, Japan, the US, France, Germany, or Canada."
                if reason == "unknown_country"
                else "Country must be a two-letter country code, like KR or US."
            )
            return ToolResult(
                "next_holidays",
                False,
                output,
                _safe_metadata(
                    **metadata,
                    calls_external_service=False,
                    reason=reason,
                    **_holidays_handoff(
                        country_metadata=country_metadata,
                        status="refused",
                        reason=reason,
                    ),
                ),
            )
        try:
            items = _fetch_next(country)
            if not items:
                return ToolResult(
                    "next_holidays",
                    True,
                    f"No upcoming holidays found for {country}.",
                    _safe_metadata(
                        **metadata,
                        count=0,
                        **_holidays_handoff(
                            country_metadata=country_metadata,
                            status="empty",
                            reason="no_holidays",
                            items=[],
                            calls_external_service=True,
                        ),
                    ),
                )
            today = date.today().isoformat()
            is_today = [h for h in items if h.get("date") == today]
            lines = []
            if is_today:
                lines.append(f"Today is {is_today[0].get('name')} ({is_today[0].get('localName')}).")
            lines.append(f"Upcoming {country} holidays:")
            for h in items[:4]:
                local = h.get("localName")
                extra = f" ({local})" if local and local != h.get("name") else ""
                lines.append(f"  • {h.get('date')}: {h.get('name')}{extra}")
            return ToolResult(
                "next_holidays",
                True,
                "\n".join(lines),
                _safe_metadata(
                    **metadata,
                    count=len(items),
                    today_holiday=bool(is_today),
                    **_holidays_handoff(
                        country_metadata=country_metadata,
                        status="ok",
                        items=items,
                        today_holiday=is_today[0] if is_today else None,
                        calls_external_service=True,
                    ),
                ),
            )
        except Exception as e:
            return ToolResult(
                "next_holidays",
                False,
                _holidays_error_message(e, country=country),
                _safe_metadata(
                    **metadata,
                    exception_type=type(e).__name__,
                    **_holidays_handoff(
                        country_metadata=country_metadata,
                        status="unavailable",
                        reason="fetch_error",
                        calls_external_service=True,
                        exception_type=type(e).__name__,
                    ),
                ),
            )

    from jarvis_v2.tools.registry import Tool, _tool_argument_contract
    return [
        Tool(
            "next_holidays",
            "Upcoming public holidays (default Korea). Args: country (e.g. 'US') or text.",
            RiskLevel.LOCAL_SAFE,
            next_holidays,
            "personal",
            argument_contract=_tool_argument_contract(optional_strings=("text", "country")),
        ),
    ]
