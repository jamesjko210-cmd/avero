"""Live weather connector for Jarvis V2.

Uses wttr.in — free, no API key, accepts a city name. Read-only network call,
so this is LOCAL_SAFE and runs without an approval prompt.
"""

from __future__ import annotations

import json
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


MAX_LOCATION_CHARS = 80
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


def _weather_boundaries(*, calls_external_service: bool) -> dict[str, bool]:
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


def _weather_handoff(
    *,
    location: str,
    location_metadata: dict[str, Any],
    status: str,
    reason: str = "",
    resolved_location: str = "",
    temp_c: Any = None,
    feels_like_c: Any = None,
    high_c: Any = None,
    low_c: Any = None,
    description: str = "",
    humidity_pct: Any = None,
    wind_kmph: Any = None,
    calls_external_service: bool,
) -> dict[str, Any]:
    next_safe_command = f"weather in {location}" if location else "weather"
    boundaries = _weather_boundaries(calls_external_service=calls_external_service)
    handoff = {
        "source": "get_weather",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "location": location,
        "location_chars": location_metadata.get("location_chars", len(location)),
        "location_truncated": bool(location_metadata.get("location_truncated")),
        "resolved_location": resolved_location,
        "status": status,
        "reason": reason,
        "weather_available": status == "ok",
        "temp_c": temp_c,
        "feels_like_c": feels_like_c,
        "high_c": high_c,
        "low_c": low_c,
        "description": description,
        "humidity_pct": humidity_pct,
        "wind_kmph": wind_kmph,
        "content_in_handoff": False,
        "prose_in_metadata": False,
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "retry_safe": status != "invalid_input",
        "boundaries": boundaries,
    }
    return {
        "weather_handoff_ready": True,
        "weather_ready_for_operator": handoff["ready_for_operator"],
        "weather_state_changed": handoff["state_changed"],
        "weather_changed": handoff["changed"],
        "weather_content_in_handoff": handoff["content_in_handoff"],
        "weather_prose_in_metadata": handoff["prose_in_metadata"],
        "weather_next_safe_command": handoff["next_safe_command"],
        "weather_next_safe_commands": handoff["next_safe_commands"],
        "weather_next_safe_command_count": handoff["next_safe_command_count"],
        "weather_authorizes_execution": handoff["authorizes_execution"],
        "weather_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "weather_approval_granted": handoff["approval_granted"],
        "weather_boundaries": boundaries,
        "weather_handoff": handoff,
    }


def _redact_local_paths(value: Any) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", str(value or ""))


def _clean(value: Any, limit: int) -> str:
    text = " ".join(_redact_local_paths(value).split())
    return text[:limit].strip()


def _location_metadata(location: str, *, requested: Any) -> dict[str, Any]:
    raw_text = " ".join(_redact_local_paths(requested).split())
    return {
        "location": location,
        "location_chars": len(location),
        "location_truncated": bool(raw_text and len(raw_text) > MAX_LOCATION_CHARS),
    }


def _looks_like_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _default_location() -> str:
    raw = os.getenv("JARVIS_WEATHER_LOCATION", "")
    location = _clean(raw, MAX_LOCATION_CHARS)
    if not location:
        return "Seoul"
    if _looks_like_local_path(raw) or not any(ch.isalnum() for ch in location):
        return "Seoul"
    return location


def _fetch(location: str) -> dict:
    url = "https://wttr.in/" + urllib.parse.quote(location) + "?format=j1"
    return http_get_json(url, headers={"User-Agent": "curl/8"})


def _weather_recovery_message(prefix: str) -> str:
    return f"{prefix} Check network access to wttr.in, run setup check, then retry in a moment."


def make_weather_tools(config: JarvisConfig):
    def get_weather(args: dict[str, Any]) -> ToolResult:
        requested_location = args.get("location") or args.get("city")
        location = _clean(requested_location, MAX_LOCATION_CHARS) or _default_location()
        location_metadata = _location_metadata(location, requested=requested_location)
        if requested_location not in (None, "") and (_looks_like_local_path(requested_location) or not any(ch.isalnum() for ch in location)):
            return ToolResult(
                "get_weather",
                False,
                "Please give me a real location for weather.",
                _safe_metadata(
                    **location_metadata,
                    reason="invalid_location",
                    calls_external_service=False,
                    **_weather_handoff(
                        location=location,
                        location_metadata=location_metadata,
                        status="invalid_input",
                        reason="invalid_location",
                        calls_external_service=False,
                    ),
                ),
            )
        try:
            data = _fetch(location)
            current = (data.get("current_condition") or [{}])[0]
            today = (data.get("weather") or [{}])[0]
            area = (data.get("nearest_area") or [{}])[0]
            name = _clean((area.get("areaName") or [{}])[0].get("value"), 60) or location

            temp = current.get("temp_C")
            feels = current.get("FeelsLikeC")
            desc = _clean((current.get("weatherDesc") or [{}])[0].get("value"), 60)
            humidity = current.get("humidity")
            wind = current.get("windspeedKmph")
            hi = today.get("maxtempC")
            lo = today.get("mintempC")
            if temp in (None, ""):
                return ToolResult(
                    "get_weather",
                    False,
                    _weather_recovery_message(f"No current weather returned for '{location}'."),
                    _safe_metadata(
                        **location_metadata,
                        reason="missing_current_temperature",
                        **_weather_handoff(
                            location=location,
                            location_metadata=location_metadata,
                            status="unavailable",
                            reason="missing_current_temperature",
                            resolved_location=name,
                            temp_c=temp,
                            feels_like_c=feels,
                            high_c=hi,
                            low_c=lo,
                            description=desc,
                            humidity_pct=humidity,
                            wind_kmph=wind,
                            calls_external_service=True,
                        ),
                    ),
                )

            parts = [f"{name}: {temp}°C"]
            if feels and feels != temp:
                parts[0] += f" (feels {feels}°C)"
            if desc:
                parts.append(desc)
            tail = []
            if hi and lo:
                tail.append(f"High {hi}° / Low {lo}°")
            if humidity:
                tail.append(f"humidity {humidity}%")
            if wind:
                tail.append(f"wind {wind} km/h")
            line = ", ".join(parts)
            if tail:
                line += ". " + ", ".join(tail) + "."
            return ToolResult(
                "get_weather",
                True,
                line,
                _safe_metadata(
                    **{**location_metadata, "resolved_location": name},
                    **_weather_handoff(
                        location=location,
                        location_metadata=location_metadata,
                        status="ok",
                        resolved_location=name,
                        temp_c=temp,
                        feels_like_c=feels,
                        high_c=hi,
                        low_c=lo,
                        description=desc,
                        humidity_pct=humidity,
                        wind_kmph=wind,
                        calls_external_service=True,
                    ),
                ),
            )
        except Exception as e:
            msg = str(e)
            status = getattr(e, "status", None)
            if status == 404 or "404" in msg:
                return ToolResult(
                    "get_weather",
                    False,
                    f"I couldn't find weather for '{location}'. Try a city name.",
                    _safe_metadata(
                        **location_metadata,
                        reason="not_found",
                        **_weather_handoff(
                            location=location,
                            location_metadata=location_metadata,
                            status="unavailable",
                            reason="not_found",
                            calls_external_service=True,
                        ),
                    ),
                )
            if status is not None and 500 <= status < 600 or "500" in msg:
                failure_output = (
                    f"{_weather_recovery_message('The weather service is having trouble right now.')} "
                    f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "get_weather",
                    False,
                    failure_output,
                    declare_retryable_external_information_failure(
                        _safe_metadata(
                            **location_metadata,
                            reason="service_error",
                            **_weather_handoff(
                                location=location,
                                location_metadata=location_metadata,
                                status="unavailable",
                                reason="service_error",
                                calls_external_service=True,
                            ),
                        ),
                        output=failure_output,
                        action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                        commands=("setup check",),
                    ),
                )
            friendly = friendly_http_error(e, subject=f"weather for '{location}'", service="The weather service")
            friendly_lower = friendly.lower()
            if "timed out" in friendly_lower or "timeout" in friendly_lower:
                friendly = _weather_recovery_message("The weather request timed out.")
            elif friendly.startswith("Error:") or "weather service is having trouble" in friendly_lower:
                friendly = _weather_recovery_message("The weather service is having trouble right now.")
            failure_output = (
                f"{friendly} {EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_weather",
                False,
                failure_output,
                declare_retryable_external_information_failure(
                    _safe_metadata(
                        **location_metadata,
                        reason="fetch_error",
                        **_weather_handoff(
                            location=location,
                            location_metadata=location_metadata,
                            status="unavailable",
                            reason="fetch_error",
                            calls_external_service=True,
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
            "get_weather",
            "Get current weather + today's high/low for a location (default Seoul). Args: location.",
            RiskLevel.LOCAL_SAFE,
            get_weather,
            "personal",
            argument_contract=_tool_argument_contract(optional_strings=("location", "city")),
        ),
    ]
