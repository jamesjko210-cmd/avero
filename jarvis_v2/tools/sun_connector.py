"""Sunrise/sunset times for Jarvis V2 (sunrise-sunset.org, free, no key)."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from jarvis_v2.agent.failure_guidance import (
    EXTERNAL_INFORMATION_RECOVERY_ACTION,
    declare_retryable_external_information_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools._http import http_get, http_get_json, friendly_http_error


SEOUL_LAT = 37.5665
SEOUL_LNG = 126.9780
MAX_RAW_COORD_CHARS = 80
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
# (display name, lat, lng, IANA timezone). The timezone is used to convert the
# UTC sunrise/sunset API response into that CITY's own local time -- previously
# `_local()` converted to the machine's system timezone regardless of which city
# was asked about, so every city other than the machine's own timezone (and ones
# that happen to share it, e.g. Tokyo/Osaka sharing Seoul's UTC+9) silently
# returned wrong, Seoul-shifted times (real bug found live: London came back as
# "sunrise 12:51, sunset 05:19" -- sunrise later than sunset, both implausible).
CITY_COORDS = {
    "seoul": ("Seoul", SEOUL_LAT, SEOUL_LNG, "Asia/Seoul"),
    "busan": ("Busan", 35.1796, 129.0756, "Asia/Seoul"),
    "tokyo": ("Tokyo", 35.6762, 139.6503, "Asia/Tokyo"),
    "osaka": ("Osaka", 34.6937, 135.5023, "Asia/Tokyo"),
    "new york": ("New York", 40.7128, -74.0060, "America/New_York"),
    "nyc": ("New York", 40.7128, -74.0060, "America/New_York"),
    "london": ("London", 51.5072, -0.1276, "Europe/London"),
    "los angeles": ("Los Angeles", 34.0522, -118.2437, "America/Los_Angeles"),
    "la": ("Los Angeles", 34.0522, -118.2437, "America/Los_Angeles"),
    "san francisco": ("San Francisco", 37.7749, -122.4194, "America/Los_Angeles"),
    "sf": ("San Francisco", 37.7749, -122.4194, "America/Los_Angeles"),
    "paris": ("Paris", 48.8566, 2.3522, "Europe/Paris"),
    "singapore": ("Singapore", 1.3521, 103.8198, "Asia/Singapore"),
    "hong kong": ("Hong Kong", 22.3193, 114.1694, "Asia/Hong_Kong"),
    "taipei": ("Taipei", 25.0330, 121.5654, "Asia/Taipei"),
}


def _env_coord(name: str, fallback: float, *, low: float, high: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return fallback
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed) or parsed < low or parsed > high:
        return fallback
    return parsed


DEFAULT_LAT = _env_coord("JARVIS_LAT", SEOUL_LAT, low=-90.0, high=90.0)
DEFAULT_LNG = _env_coord("JARVIS_LNG", SEOUL_LNG, low=-180.0, high=180.0)


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


def _sun_boundaries(*, calls_external_service: bool = False) -> dict[str, bool]:
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


def _short_raw(value: Any, limit: int = MAX_RAW_COORD_CHARS) -> str:
    text = "" if value is None else str(value)
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _clean_location(value: Any) -> str:
    text = LOCAL_PATH_RE.sub("<local-path>", str(value or ""))
    return re.sub(r"\s+", " ", text).strip(" ?.!").lower()


def _location_from_text(text: Any) -> str:
    raw = _clean_location(text)
    patterns = (
        r"\b(?:sunrise|sunset)\s+(?:times?\s+)?(?:in|at|for)\s+(.{2,60})$",
        r"\b(?:what time is|when is|when does)\s+(?:the\s+)?(?:sun\s+)?(?:rise|set|sunrise|sunset)\s+(?:in|at|for)\s+(.{2,60})$",
        r"^(?:sunrise|sunset)\s+(.{2,60})$",
    )
    for pattern in patterns:
        match = re.search(pattern, raw)
        if not match:
            continue
        location = re.sub(r"\b(?:today|tomorrow|now|right now)\b", "", match.group(1)).strip(" ?.!").lower()
        if location:
            return location
    return ""


def _coords_for_location(location: str) -> tuple[str, float, float, str] | None:
    cleaned = _clean_location(location)
    return CITY_COORDS.get(cleaned)


def _parse_coord(value: Any, *, default: float, low: float, high: float) -> tuple[float | None, dict[str, Any]]:
    raw = value
    metadata = {"raw": _short_raw(raw), "used_default": raw in (None, "")}
    try:
        parsed = default if raw in (None, "") else float(raw)
    except (TypeError, ValueError):
        return None, metadata
    if not math.isfinite(parsed) or parsed < low or parsed > high:
        return None, metadata
    return parsed, {**metadata, "value": parsed}


def _location_metadata(raw_location: str, resolved_location: str = "") -> dict[str, Any]:
    return {
        "raw": _short_raw(raw_location),
        "resolved_location": resolved_location,
        "used_location": bool(resolved_location),
        "known_location": bool(resolved_location),
    }


def _sun_times_handoff(
    *,
    lat_metadata: dict[str, Any],
    lng_metadata: dict[str, Any],
    day: str,
    status: str,
    reason: str = "",
    calls_external_service: bool = False,
    sunrise: str = "",
    sunset: str = "",
    exception_type: str = "",
    location_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    next_safe_command = "sunset today"
    boundaries = _sun_boundaries(calls_external_service=calls_external_service)
    handoff = {
        "source": "get_sun_times",
        "sun_times_handoff_ready": True,
        "handoff_ready": True,
        "ready_for_operator": True,
        "status": status,
        "reason": reason,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "day": day,
        "location": location_metadata or _location_metadata(""),
        "lat": lat_metadata,
        "lng": lng_metadata,
        "sunrise": sunrise,
        "sunset": sunset,
        "times_available": bool(sunrise and sunset),
        "exception_type": exception_type,
        "content_in_metadata": False,
        "retry_safe": status in {"missing_data", "unavailable"},
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "sun_times_handoff_ready": True,
        "sun_times_ready_for_operator": handoff["ready_for_operator"],
        "sun_times_state_changed": handoff["state_changed"],
        "sun_times_changed": handoff["changed"],
        "sun_times_content_in_handoff": handoff["content_in_handoff"],
        "sun_times_content_in_metadata": handoff["content_in_metadata"],
        "sun_times_next_safe_command": handoff["next_safe_command"],
        "sun_times_next_safe_commands": handoff["next_safe_commands"],
        "sun_times_next_safe_command_count": handoff["next_safe_command_count"],
        "sun_times_authorizes_execution": handoff["authorizes_execution"],
        "sun_times_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "sun_times_approval_granted": handoff["approval_granted"],
        "sun_times_boundaries": boundaries,
        "sun_times_location": handoff["location"],
        "sun_times_handoff": handoff,
    }


def _fetch(lat: float, lng: float, day: str) -> dict:
    params = urllib.parse.urlencode({"lat": lat, "lng": lng, "formatted": 0, "date": day})
    url = "https://api.sunrise-sunset.org/json?" + params
    return http_get_json(url, headers={"User-Agent": "Mozilla/5.0"})


def _sun_recovery_message(prefix: str) -> str:
    return f"{prefix} Check network access to sunrise-sunset.org, run setup check, then retry in a moment."


def _sun_error_message(e: Exception) -> str:
    friendly = friendly_http_error(e, service="The sunrise/sunset service")
    lowered = friendly.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return _sun_recovery_message("The sunrise/sunset request timed out.")
    if friendly.startswith("I couldn't find"):
        return friendly
    if friendly.startswith("Error:") or "having trouble" in lowered or "try again" in lowered:
        return _sun_recovery_message("The sunrise/sunset service is having trouble right now.")
    return friendly


def _local(iso_utc: str, tz_name: str = "") -> str:
    try:
        dt = datetime.fromisoformat(iso_utc)
        if tz_name:
            try:
                return dt.astimezone(ZoneInfo(tz_name)).strftime("%H:%M")
            except Exception:
                pass
        # No known city timezone (e.g. a raw lat/lng not resolved through
        # CITY_COORDS) -- best effort, falls back to the machine's own timezone.
        return dt.astimezone().strftime("%H:%M")
    except Exception:
        return iso_utc


def make_sun_tools(config: JarvisConfig):
    def get_sun_times(args: dict[str, Any]) -> ToolResult:
        text = str(args.get("text") or "")
        raw_location = str(args.get("location") or args.get("city") or _location_from_text(text) or "").strip()
        resolved_location = ""
        resolved_tz = ""
        if raw_location and args.get("lat") in (None, "") and args.get("lng") in (None, ""):
            coords = _coords_for_location(raw_location)
            if not coords:
                loc_metadata = _location_metadata(raw_location)
                lat_metadata = {"raw": "", "used_default": False}
                lng_metadata = {"raw": "", "used_default": False}
                day = "tomorrow" if "tomorrow" in text.lower() else "today"
                metadata = {"day": day, "lat": lat_metadata, "lng": lng_metadata, "location": loc_metadata}
                return ToolResult(
                    "get_sun_times",
                    False,
                    "I don't know that sunrise/sunset location yet. Try a common city like Seoul, Tokyo, Busan, London, or New York.",
                    _safe_metadata(
                        **metadata,
                        calls_external_service=False,
                        reason="unknown_location",
                        **_sun_times_handoff(
                            lat_metadata=lat_metadata,
                            lng_metadata=lng_metadata,
                            day=day,
                            status="refused",
                            reason="unknown_location",
                            location_metadata=loc_metadata,
                        ),
                    ),
                )
            resolved_location, location_lat, location_lng, resolved_tz = coords
            lat, lat_metadata = location_lat, {"raw": _short_raw(raw_location), "used_default": False, "value": location_lat, "source": "location", "location": resolved_location}
            lng, lng_metadata = location_lng, {"raw": _short_raw(raw_location), "used_default": False, "value": location_lng, "source": "location", "location": resolved_location}
        else:
            lat, lat_metadata = _parse_coord(args.get("lat"), default=DEFAULT_LAT, low=-90.0, high=90.0)
            lng, lng_metadata = _parse_coord(args.get("lng"), default=DEFAULT_LNG, low=-180.0, high=180.0)
        day = "tomorrow" if "tomorrow" in text.lower() else "today"
        loc_metadata = _location_metadata(raw_location, resolved_location)
        metadata = {"day": day, "lat": lat_metadata, "lng": lng_metadata, "location": loc_metadata}
        if lat is None or lng is None:
            return ToolResult(
                "get_sun_times",
                False,
                "Latitude must be between -90 and 90, and longitude must be between -180 and 180.",
                _safe_metadata(
                    **metadata,
                    calls_external_service=False,
                    reason="invalid_coordinates",
                    **_sun_times_handoff(
                        lat_metadata=lat_metadata,
                        lng_metadata=lng_metadata,
                        day=day,
                        status="refused",
                        reason="invalid_coordinates",
                        location_metadata=loc_metadata,
                    ),
                ),
            )
        try:
            data = _fetch(lat, lng, day)
            r = data.get("results") or {}
            if not r.get("sunrise"):
                failure_output = (
                    f"{_sun_recovery_message('Sunrise/sunset data was incomplete.')} "
                    f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "get_sun_times",
                    False,
                    failure_output,
                    declare_retryable_external_information_failure(
                        _safe_metadata(
                            **metadata,
                            reason="missing_sunrise",
                            **_sun_times_handoff(
                                lat_metadata=lat_metadata,
                                lng_metadata=lng_metadata,
                                day=day,
                                status="missing_data",
                                reason="missing_sunrise",
                                calls_external_service=True,
                                location_metadata=loc_metadata,
                            ),
                        ),
                        output=failure_output,
                        action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                        commands=("setup check",),
                    ),
                )
            label = "Tomorrow" if day == "tomorrow" else "Today"
            sunrise = _local(r["sunrise"], resolved_tz)
            sunset = _local(r["sunset"], resolved_tz)
            place = f"{resolved_location} " if resolved_location else ""
            return ToolResult(
                "get_sun_times", True,
                f"{place}{label}: sunrise {sunrise}, sunset {sunset}.",
                _safe_metadata(
                    **metadata,
                    sunrise=sunrise,
                    sunset=sunset,
                    **_sun_times_handoff(
                        lat_metadata=lat_metadata,
                        lng_metadata=lng_metadata,
                        day=day,
                        status="ok",
                        calls_external_service=True,
                        sunrise=sunrise,
                        sunset=sunset,
                        location_metadata=loc_metadata,
                    ),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_sun_error_message(e)} "
                f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_sun_times",
                False,
                failure_output,
                declare_retryable_external_information_failure(
                    _safe_metadata(
                        **metadata,
                        exception_type=type(e).__name__,
                        **_sun_times_handoff(
                            lat_metadata=lat_metadata,
                            lng_metadata=lng_metadata,
                            day=day,
                            status="unavailable",
                            reason="fetch_error",
                            calls_external_service=True,
                            exception_type=type(e).__name__,
                            location_metadata=loc_metadata,
                        ),
                    ),
                    output=failure_output,
                    action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    from jarvis_v2.tools.registry import (
        TOOL_ARGUMENT_CONTRACT_VERSION,
        Tool,
        ToolArgumentContract,
        ToolArgumentSpec,
        ToolArgumentType,
    )
    return [
        Tool(
            "get_sun_times",
            "Get sunrise and sunset times (default Seoul). Args: text (today/tomorrow), lat, lng.",
            RiskLevel.LOCAL_SAFE,
            get_sun_times,
            "personal",
            argument_contract=ToolArgumentContract(
                TOOL_ARGUMENT_CONTRACT_VERSION,
                (
                    ToolArgumentSpec("text", frozenset({ToolArgumentType.STRING}), False),
                    ToolArgumentSpec("location", frozenset({ToolArgumentType.STRING}), False),
                    ToolArgumentSpec("city", frozenset({ToolArgumentType.STRING}), False),
                    ToolArgumentSpec(
                        "lat",
                        frozenset({ToolArgumentType.NUMBER, ToolArgumentType.STRING}),
                        False,
                    ),
                    ToolArgumentSpec(
                        "lng",
                        frozenset({ToolArgumentType.NUMBER, ToolArgumentType.STRING}),
                        False,
                    ),
                ),
                False,
            ),
        ),
    ]
