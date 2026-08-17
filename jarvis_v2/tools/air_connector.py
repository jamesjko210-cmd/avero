"""Air quality for Jarvis V2 (Open-Meteo air-quality API, free, no key)."""

from __future__ import annotations

import json
import math
import os
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


SEOUL_LAT = 37.5665
SEOUL_LNG = 126.9780
DEFAULT_LAT = SEOUL_LAT
DEFAULT_LNG = SEOUL_LNG
MAX_RAW_COORD_CHARS = 80
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
CITY_COORDS = {
    "seoul": ("Seoul", SEOUL_LAT, SEOUL_LNG),
    "busan": ("Busan", 35.1796, 129.0756),
    "tokyo": ("Tokyo", 35.6762, 139.6503),
    "osaka": ("Osaka", 34.6937, 135.5023),
    "new york": ("New York", 40.7128, -74.0060),
    "nyc": ("New York", 40.7128, -74.0060),
    "london": ("London", 51.5072, -0.1276),
    "los angeles": ("Los Angeles", 34.0522, -118.2437),
    "la": ("Los Angeles", 34.0522, -118.2437),
    "san francisco": ("San Francisco", 37.7749, -122.4194),
    "sf": ("San Francisco", 37.7749, -122.4194),
    "paris": ("Paris", 48.8566, 2.3522),
    "singapore": ("Singapore", 1.3521, 103.8198),
    "hong kong": ("Hong Kong", 22.3193, 114.1694),
    "taipei": ("Taipei", 25.0330, 121.5654),
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


def _air_boundaries(*, calls_external_service: bool = False) -> dict[str, bool]:
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
    city_pattern = "|".join(re.escape(city) for city in sorted(CITY_COORDS, key=len, reverse=True))
    patterns = (
        r"\b(?:air quality|aqi|fine dust|pollution)\s+(?:in|at|for)\s+(.{2,60})$",
        r"\b(?:air quality|aqi|fine dust|pollution)\s+(.{2,60})$",
        r"\b(?:how(?:'s| is)|is)\s+(?:the\s+)?air(?:\s+(?:bad|clean|good|safe))?\s+(?:in|at|for)\s+(.{2,60})$",
        r"\bhow\s+(?:bad|clean|good|safe)\s+is\s+(?:the\s+)?air\s+(?:in|at|for)\s+(.{2,60})$",
        rf"^air\s+({city_pattern})(?:\s+(?:today|tomorrow|now|right now))?$",
        rf"^({city_pattern})\s+air(?:\s+(?:today|tomorrow|now|right now))?$",
        rf"^how(?:'s| is)\s+({city_pattern})\s+air(?:\s+(?:today|tomorrow|now|right now))?$",
    )
    for pattern in patterns:
        match = re.search(pattern, raw)
        if not match:
            continue
        location = re.sub(r"\b(?:today|tomorrow|now|right now)\b", "", match.group(1)).strip(" ?.!").lower()
        if location:
            return location
    return ""


def _coords_for_location(location: str) -> tuple[str, float, float] | None:
    return CITY_COORDS.get(_clean_location(location))


def _air_quality_handoff(
    *,
    lat_metadata: dict[str, Any],
    lng_metadata: dict[str, Any],
    status: str,
    reason: str = "",
    calls_external_service: bool = False,
    aqi: float | None = None,
    pm25: float | None = None,
    pm10: float | None = None,
    exception_type: str = "",
    location_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    next_safe_command = "air quality"
    boundaries = _air_boundaries(calls_external_service=calls_external_service)
    handoff = {
        "source": "get_air_quality",
        "air_quality_handoff_ready": True,
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
        "lat": lat_metadata,
        "lng": lng_metadata,
        "location": location_metadata or _location_metadata(""),
        "aqi": aqi,
        "aqi_label": _aqi_label(aqi),
        "pm25": pm25,
        "pm10": pm10,
        "measurement_available": aqi is not None,
        "exception_type": exception_type,
        "content_in_metadata": False,
        "retry_safe": status in {"missing_data", "unavailable"},
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "air_quality_handoff_ready": True,
        "air_quality_ready_for_operator": handoff["ready_for_operator"],
        "air_quality_state_changed": handoff["state_changed"],
        "air_quality_changed": handoff["changed"],
        "air_quality_content_in_handoff": handoff["content_in_handoff"],
        "air_quality_content_in_metadata": handoff["content_in_metadata"],
        "air_quality_next_safe_command": handoff["next_safe_command"],
        "air_quality_next_safe_commands": handoff["next_safe_commands"],
        "air_quality_next_safe_command_count": handoff["next_safe_command_count"],
        "air_quality_authorizes_execution": handoff["authorizes_execution"],
        "air_quality_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "air_quality_approval_granted": handoff["approval_granted"],
        "air_quality_boundaries": boundaries,
        "air_quality_location": handoff["location"],
        "air_quality_handoff": handoff,
    }


def _location_metadata(raw_location: str, resolved_location: str = "") -> dict[str, Any]:
    return {
        "raw": _short_raw(raw_location),
        "resolved_location": resolved_location,
        "used_location": bool(resolved_location),
        "known_location": bool(resolved_location),
    }


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


def _aqi_label(aqi: float | None) -> str:
    if aqi is None:
        return ""
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for sensitive groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very unhealthy"
    return "Hazardous"


def _fetch(lat: float, lng: float) -> dict:
    params = urllib.parse.urlencode({"latitude": lat, "longitude": lng, "current": "us_aqi,pm2_5,pm10"})
    url = "https://air-quality-api.open-meteo.com/v1/air-quality?" + params
    return http_get_json(url, headers={"User-Agent": "Mozilla/5.0"})


def _air_recovery_message(prefix: str) -> str:
    return f"{prefix} Check network access to Open-Meteo air quality, run setup check, then retry in a moment."


def _air_error_message(e: Exception) -> str:
    friendly = friendly_http_error(e, service="The air-quality service")
    lowered = friendly.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return _air_recovery_message("The air-quality request timed out.")
    if friendly.startswith("I couldn't find"):
        return friendly
    if friendly.startswith("Error:") or "having trouble" in lowered or "try again" in lowered:
        return _air_recovery_message("The air-quality service is having trouble right now.")
    return friendly


def make_air_tools(config: JarvisConfig):
    def get_air_quality(args: dict[str, Any]) -> ToolResult:
        text = str(args.get("text") or "")
        raw_location = str(args.get("location") or args.get("city") or _location_from_text(text) or "").strip()
        resolved_location = ""
        if raw_location and args.get("lat") in (None, "") and args.get("lng") in (None, ""):
            coords = _coords_for_location(raw_location)
            if not coords:
                loc_metadata = _location_metadata(raw_location)
                lat_metadata = {"raw": "", "used_default": False}
                lng_metadata = {"raw": "", "used_default": False}
                metadata = {"lat": lat_metadata, "lng": lng_metadata, "location": loc_metadata}
                return ToolResult(
                    "get_air_quality",
                    False,
                    "I don't know that air-quality location yet. Try a common city like Seoul, Tokyo, Busan, London, or New York.",
                    _safe_metadata(
                        **metadata,
                        calls_external_service=False,
                        reason="unknown_location",
                        **_air_quality_handoff(
                            lat_metadata=lat_metadata,
                            lng_metadata=lng_metadata,
                            status="refused",
                            reason="unknown_location",
                            location_metadata=loc_metadata,
                        ),
                    ),
                )
            resolved_location, location_lat, location_lng = coords
            lat, lat_metadata = location_lat, {"raw": _short_raw(raw_location), "used_default": False, "value": location_lat, "source": "location", "location": resolved_location}
            lng, lng_metadata = location_lng, {"raw": _short_raw(raw_location), "used_default": False, "value": location_lng, "source": "location", "location": resolved_location}
        else:
            lat, lat_metadata = _parse_coord(args.get("lat"), default=DEFAULT_LAT, low=-90.0, high=90.0)
            lng, lng_metadata = _parse_coord(args.get("lng"), default=DEFAULT_LNG, low=-180.0, high=180.0)
        loc_metadata = _location_metadata(raw_location, resolved_location)
        metadata = {"lat": lat_metadata, "lng": lng_metadata, "location": loc_metadata}
        if lat is None or lng is None:
            return ToolResult(
                "get_air_quality",
                False,
                "Latitude must be between -90 and 90, and longitude must be between -180 and 180.",
                _safe_metadata(
                    **metadata,
                    calls_external_service=False,
                    reason="invalid_coordinates",
                    **_air_quality_handoff(
                        lat_metadata=lat_metadata,
                        lng_metadata=lng_metadata,
                        status="refused",
                        reason="invalid_coordinates",
                        location_metadata=loc_metadata,
                    ),
                ),
            )
        try:
            cur = (_fetch(lat, lng).get("current") or {})
            aqi = cur.get("us_aqi")
            if aqi is None:
                failure_output = (
                    f"{_air_recovery_message('Air-quality data was incomplete.')} "
                    f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "get_air_quality",
                    False,
                    failure_output,
                    declare_retryable_external_information_failure(
                        _safe_metadata(
                            **metadata,
                            reason="missing_aqi",
                            **_air_quality_handoff(
                                lat_metadata=lat_metadata,
                                lng_metadata=lng_metadata,
                                status="missing_data",
                                reason="missing_aqi",
                                calls_external_service=True,
                                location_metadata=loc_metadata,
                            ),
                        ),
                        output=failure_output,
                        action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                        commands=("setup check",),
                    ),
                )
            pm25 = cur.get("pm2_5")
            pm10 = cur.get("pm10")
            return ToolResult(
                "get_air_quality", True,
                f"{resolved_location + ' ' if resolved_location else ''}Air quality: AQI {aqi} ({_aqi_label(aqi)}). PM2.5 {pm25}, PM10 {pm10} µg/m³.",
                _safe_metadata(
                    **metadata,
                    aqi=aqi,
                    pm25=pm25,
                    pm10=pm10,
                    **_air_quality_handoff(
                        lat_metadata=lat_metadata,
                        lng_metadata=lng_metadata,
                        status="ok",
                        calls_external_service=True,
                        aqi=aqi,
                        pm25=pm25,
                        pm10=pm10,
                        location_metadata=loc_metadata,
                    ),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_air_error_message(e)} "
                f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_air_quality",
                False,
                failure_output,
                declare_retryable_external_information_failure(
                    _safe_metadata(
                        **metadata,
                        exception_type=type(e).__name__,
                        **_air_quality_handoff(
                            lat_metadata=lat_metadata,
                            lng_metadata=lng_metadata,
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
            "get_air_quality",
            "Get current air quality / fine dust (default Seoul). Args: text, location/city, lat, lng.",
            RiskLevel.LOCAL_SAFE,
            get_air_quality,
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
