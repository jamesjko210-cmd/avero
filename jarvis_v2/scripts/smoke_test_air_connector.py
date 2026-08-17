"""Smoke tests for the air-quality connector (mocked fetch, no network)."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import air_connector as ac
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry


NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


def _tools():
    return {t.name: t for t in ac.make_air_tools(load_config())}


def _assert_air_recovery_message(output: str, label: str) -> None:
    lowered = output.lower()
    for fragment in [
        "network access to open-meteo air quality",
        "setup check",
        "retry in a moment",
    ]:
        if fragment not in lowered:
            raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")
    for forbidden in ["http error", "offline", "traceback", "/users/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in lowered:
            raise SystemExit(f"{label} leaked raw backend text {forbidden!r}: {output}")


def _assert_air_handoff(
    metadata: dict[str, Any],
    label: str,
    *,
    status: str,
    reason: str = "",
    calls_external_service: bool = False,
    aqi: float | None = None,
    retry_safe: bool = False,
) -> dict[str, Any]:
    handoff = metadata.get("air_quality_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing air_quality_handoff: {metadata}")
    if handoff.get("source") != "get_air_quality" or handoff.get("status") != status:
        raise SystemExit(f"{label} has wrong handoff source/status: {handoff}")
    if metadata.get("air_quality_handoff_ready") is not True or handoff.get("air_quality_handoff_ready") is not True:
        raise SystemExit(f"{label} should expose flat and nested handoff readiness: {metadata} vs {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should expose nested handoff_ready: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should be operator-ready: {handoff}")
    if metadata.get("air_quality_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} ready alias parity failed: {metadata} vs {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} has wrong handoff reason: {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} should not claim changed state: {handoff}")
    if metadata.get("air_quality_state_changed") != handoff.get("state_changed") or metadata.get("air_quality_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} state alias parity failed: {metadata} vs {handoff}")
    if handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep output content out of handoff: {handoff}")
    if metadata.get("air_quality_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} content-in-handoff alias parity failed: {metadata} vs {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
        alias = f"air_quality_{key}"
        if metadata.get(alias) is not expected_value or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if handoff.get("lat") != metadata.get("lat") or handoff.get("lng") != metadata.get("lng"):
        raise SystemExit(f"{label} should mirror coordinate metadata: {handoff} vs {metadata}")
    if "aqi" in metadata and handoff.get("aqi") != metadata.get("aqi"):
        raise SystemExit(f"{label} should mirror AQI: {handoff} vs {metadata}")
    if "pm25" in metadata and handoff.get("pm25") != metadata.get("pm25"):
        raise SystemExit(f"{label} should mirror PM2.5: {handoff} vs {metadata}")
    if "pm10" in metadata and handoff.get("pm10") != metadata.get("pm10"):
        raise SystemExit(f"{label} should mirror PM10: {handoff} vs {metadata}")
    if handoff.get("exception_type", "") != metadata.get("exception_type", ""):
        raise SystemExit(f"{label} should mirror exception type: {handoff} vs {metadata}")
    if handoff.get("aqi") != aqi:
        raise SystemExit(f"{label} has wrong handoff AQI: {handoff}")
    if handoff.get("measurement_available") is not (aqi is not None):
        raise SystemExit(f"{label} has wrong measurement availability: {handoff}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} should mark content exclusion: {handoff}")
    if metadata.get("air_quality_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} content-in-metadata alias parity failed: {metadata} vs {handoff}")
    if handoff.get("retry_safe") is not retry_safe:
        raise SystemExit(f"{label} has wrong retry state: {handoff}")
    if handoff.get("next_safe_command") != "air quality":
        raise SystemExit(f"{label} has wrong next command: {handoff}")
    expected_commands = ["air quality"]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} has wrong next safe command list: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} has wrong next safe command count: {handoff}")
    if metadata.get("air_quality_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("air_quality_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("air_quality_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} next-command-count alias parity failed: {metadata} vs {handoff}")
    for coord_key in ("lat", "lng"):
        raw = str(handoff.get(coord_key, {}).get("raw", ""))
        if "/\x55sers/" in raw or "/private/" in raw or "/var/folders" in raw or "/tmp/" in raw:
            raise SystemExit(f"{label} leaked local path in handoff: {handoff}")

    boundaries = handoff.get("boundaries")
    expected = {
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
        **NO_AUTHORITY_FLAGS,
    }
    if boundaries != expected:
        raise SystemExit(f"{label} has wrong boundaries: {boundaries}")
    if metadata.get("air_quality_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary alias parity failed: {metadata} vs {handoff}")
    if metadata.get("calls_external_service") is not calls_external_service:
        raise SystemExit(f"{label} flat external-service flag should match handoff: {metadata}")
    if metadata.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} flat plural external-service flag should match handoff: {metadata}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["get_air_quality"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("get_air_quality should be LOCAL_SAFE")


def test_air_quality_argument_contract_fences_malformed_input() -> None:
    fetch_calls: list[tuple[float, float]] = []
    original_fetch = ac._fetch

    def mocked_fetch(lat: float, lng: float) -> dict:
        fetch_calls.append((lat, lng))
        return {"current": {"us_aqi": 42, "pm2_5": 8.0, "pm10": 12.0}}

    try:
        ac._fetch = mocked_fetch  # type: ignore[assignment]
        private_sentinel = "private-air-contract-sentinel"
        tool = _tools()["get_air_quality"]
        contract = tool.argument_contract
        actual_shape = (
            tuple(
                (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                for field in contract.fields
            )
            if contract is not None
            else ()
        )
        expected_shape = (
            ("text", ("string",), False),
            ("location", ("string",), False),
            ("city", ("string",), False),
            ("lat", ("number", "string"), False),
            ("lng", ("number", "string"), False),
        )
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown is not False
            or actual_shape != expected_shape
        ):
            raise SystemExit(f"get_air_quality should have an exact strict argument contract: {contract}")

        registry = ToolRegistry()
        registry.register(tool)
        executor = Executor(registry, PermissionPolicy())
        malformed_args: list[object] = [
            None,
            ["not", "an", "object"],
            {"text": {"private": private_sentinel}},
            {"location": True},
            {"city": 7},
            {"lat": {"private": private_sentinel}},
            {"lng": [private_sentinel]},
            {"location": "Seoul", "unexpected": private_sentinel},
        ]
        rejected_surfaces: list[object] = []
        for args in malformed_args:
            result = executor.execute(
                PlannedAction("get_air_quality", args, "air-quality contract smoke")  # type: ignore[arg-type]
            )
            if (
                result.ok
                or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
                or result.metadata.get("requires_confirmation") is not False
            ):
                raise SystemExit(f"malformed air-quality input passed the executor: {result}")
            rejected_surfaces.append({"output": result.output, "metadata": result.metadata})
        if fetch_calls:
            raise SystemExit(f"malformed air-quality input reached Open-Meteo: {fetch_calls}")
        if private_sentinel in str(rejected_surfaces):
            raise SystemExit("air-quality argument rejection leaked a supplied private value")

        valid_cases = (
            ({"location": "Tokyo"}, ac.CITY_COORDS["tokyo"][1:]),
            ({"city": "Busan"}, ac.CITY_COORDS["busan"][1:]),
            ({"text": "air quality in Seoul"}, ac.CITY_COORDS["seoul"][1:]),
            ({"lat": 35.0, "lng": 127.0}, (35.0, 127.0)),
            ({"lat": "35.0", "lng": "127.0"}, (35.0, 127.0)),
        )
        for args, expected_coords in valid_cases:
            result = executor.execute(
                PlannedAction("get_air_quality", args, "valid air-quality contract smoke")
            )
            if not result.ok or fetch_calls[-1] != expected_coords:
                raise SystemExit(f"valid air-quality input changed: {args} {result} {fetch_calls}")
        empty = executor.execute(
            PlannedAction("get_air_quality", {}, "empty air-quality contract smoke")
        )
        if (
            not empty.ok
            or empty.metadata.get("handler_invoked") is not True
            or fetch_calls[-1] != (ac.DEFAULT_LAT, ac.DEFAULT_LNG)
        ):
            raise SystemExit(f"empty air-quality compatibility changed: {empty}")
    finally:
        ac._fetch = original_fetch  # type: ignore[assignment]


def test_labels_aqi() -> None:
    if ac._aqi_label(40) != "Good" or ac._aqi_label(84) != "Moderate" or ac._aqi_label(250) != "Very unhealthy":
        raise SystemExit("AQI labels wrong")


def test_reports_air_quality() -> None:
    ac._fetch = lambda lat, lng: {"current": {"us_aqi": 84, "pm2_5": 28.3, "pm10": 32.2}}  # type: ignore
    out = _tools()["get_air_quality"].handler({})
    if not out.ok or "AQI 84" not in out.output or "Moderate" not in out.output or "PM2.5" not in out.output:
        raise SystemExit(f"air quality output wrong: {out.output}")
    if out.metadata.get("lat", {}).get("value") != ac.DEFAULT_LAT or out.metadata.get("lng", {}).get("value") != ac.DEFAULT_LNG:
        raise SystemExit(f"air quality should preserve coordinate metadata: {out.metadata}")
    handoff = _assert_air_handoff(
        out.metadata,
        "air quality success",
        status="ok",
        calls_external_service=True,
        aqi=84,
    )
    if handoff.get("aqi_label") != "Moderate" or handoff.get("pm25") != 28.3 or handoff.get("pm10") != 32.2:
        raise SystemExit(f"air quality handoff should preserve readings: {handoff}")


def test_resolves_common_city_from_text() -> None:
    seen = {}

    def fake_fetch(lat, lng):
        seen["lat"] = lat
        seen["lng"] = lng
        return {"current": {"us_aqi": 84, "pm2_5": 28.3, "pm10": 32.2}}

    ac._fetch = fake_fetch  # type: ignore
    for text, expected_name in {
        "air quality in Tokyo": "Tokyo",
        "air Seoul": "Seoul",
        "Seoul air": "Seoul",
        "how is Seoul air": "Seoul",
        "Busan air today": "Busan",
        "how bad is the air in Seoul": "Seoul",
        "how clean is the air in Busan today": "Busan",
    }.items():
        seen.clear()
        out = _tools()["get_air_quality"].handler({"text": text})
        if not out.ok or f"{expected_name} Air quality" not in out.output:
            raise SystemExit(f"city air quality output wrong for {text!r}: {out.output}")
        handoff = _assert_air_handoff(out.metadata, f"city air quality {text!r}", status="ok", calls_external_service=True, aqi=84)
        location = handoff.get("location") or {}
        if location.get("resolved_location") != expected_name or out.metadata.get("air_quality_location", {}).get("resolved_location") != expected_name:
            raise SystemExit(f"city air quality should preserve resolved location for {text!r}: {out.metadata}")
        if out.metadata.get("lat", {}).get("source") != "location" or out.metadata.get("lng", {}).get("source") != "location":
            raise SystemExit(f"city air quality should mark coordinate source for {text!r}: {out.metadata}")


def test_unknown_explicit_location_never_fetches() -> None:
    calls = []

    def fail_fetch(lat, lng):
        calls.append((lat, lng))
        raise AssertionError("unknown explicit location should not fetch")

    ac._fetch = fail_fetch  # type: ignore
    out = _tools()["get_air_quality"].handler({"text": "air quality Atlantis"})
    if out.ok or "don't know that air-quality location" not in out.output:
        raise SystemExit(f"unknown air location should be refused cleanly: {out.output}")
    handoff = _assert_air_handoff(
        out.metadata,
        "unknown air location",
        status="refused",
        reason="unknown_location",
        calls_external_service=False,
    )
    if handoff.get("location", {}).get("raw") != "atlantis":
        raise SystemExit(f"unknown air location should preserve bounded raw value: {handoff}")
    if calls:
        raise SystemExit(f"unknown air location unexpectedly fetched: {calls}")


def test_invalid_coordinates_never_fetch() -> None:
    calls = []

    def fail_fetch(lat, lng):
        calls.append((lat, lng))
        raise AssertionError("invalid coordinates should not fetch")

    ac._fetch = fail_fetch  # type: ignore
    for args in [
        {"lat": "north"},
        {"lat": 91},
        {"lng": 181},
        {"lat": "nan"},
        {"lat": "/\x55sers/example/private/air-lat"},
        {"lng": "/private/tmp/air-lng"},
        {"lat": "/var/folders/zc/jarvis-air-lat"},
        {"lng": "/tmp/jarvis-air-lng"},
    ]:
        out = _tools()["get_air_quality"].handler(args)
        if out.ok or "latitude" not in out.output.lower():
            raise SystemExit(f"invalid coordinates should be rejected locally: {args!r} -> {out.output}")
        if out.metadata.get("reason") != "invalid_coordinates":
            raise SystemExit(f"invalid coordinates should include reason metadata: {out.metadata}")
        if out.metadata.get("calls_external_service") is not False:
            raise SystemExit(f"invalid coordinates should not claim external calls: {out.metadata}")
        _assert_air_handoff(
            out.metadata,
            f"invalid coordinates {args!r}",
            status="refused",
            reason="invalid_coordinates",
            calls_external_service=False,
        )
        for coord_key in ("lat", "lng"):
            raw = str(out.metadata.get(coord_key, {}).get("raw", ""))
            if "/\x55sers/" in raw or "/private/" in raw or "/var/folders" in raw or "/tmp/" in raw:
                raise SystemExit(f"invalid coordinates leaked local path metadata: {out.metadata}")
    if calls:
        raise SystemExit(f"invalid coordinates unexpectedly fetched: {calls}")


def test_invalid_env_defaults_do_not_crash_import() -> None:
    env = os.environ.copy()
    env["JARVIS_LAT"] = "north"
    env["JARVIS_LNG"] = "inf"
    code = (
        "from jarvis_v2.tools import air_connector as ac; "
        "assert ac.DEFAULT_LAT == ac.SEOUL_LAT; "
        "assert ac.DEFAULT_LNG == ac.SEOUL_LNG"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise SystemExit(f"invalid env defaults should not crash air import: {result.stderr or result.stdout}")


def test_handles_error() -> None:
    def boom(lat, lng):
        raise RuntimeError("offline")

    ac._fetch = boom  # type: ignore
    out = _tools()["get_air_quality"].handler({})
    if out.ok or not out.output.strip():
        raise SystemExit(f"error not handled: {out.output}")
    if "offline" in out.output or out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"air error should be friendly with bounded exception metadata: {out.output} {out.metadata}")
    _assert_air_recovery_message(out.output, "air generic fetch error")
    _assert_air_handoff(
        out.metadata,
        "air fetch error",
        status="unavailable",
        reason="fetch_error",
        calls_external_service=True,
        retry_safe=True,
    )

    def server_error(lat, lng):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    ac._fetch = server_error  # type: ignore
    server_out = _tools()["get_air_quality"].handler({})
    if server_out.ok or server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"HTTP 500 should preserve bounded exception metadata: {server_out.output} {server_out.metadata}")
    _assert_air_recovery_message(server_out.output, "air HTTP 500")
    _assert_air_handoff(
        server_out.metadata,
        "air HTTP 500 fetch error",
        status="unavailable",
        reason="fetch_error",
        calls_external_service=True,
        retry_safe=True,
    )

    def timeout_error(lat, lng):
        raise TimeoutError("request timed out near /\x55sers/example/private")

    ac._fetch = timeout_error  # type: ignore
    timeout_out = _tools()["get_air_quality"].handler({})
    if timeout_out.ok or timeout_out.metadata.get("exception_type") != "TimeoutError":
        raise SystemExit(f"timeout should preserve bounded exception metadata: {timeout_out.output} {timeout_out.metadata}")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"timeout recovery should preserve timeout cause: {timeout_out.output}")
    _assert_air_recovery_message(timeout_out.output, "air timeout")
    _assert_air_handoff(
        timeout_out.metadata,
        "air timeout fetch error",
        status="unavailable",
        reason="fetch_error",
        calls_external_service=True,
        retry_safe=True,
    )


def test_handles_missing_aqi() -> None:
    ac._fetch = lambda lat, lng: {"current": {"pm2_5": 12.0, "pm10": 18.0}}  # type: ignore
    out = _tools()["get_air_quality"].handler({})
    if out.ok or "air-quality data was incomplete" not in out.output.lower():
        raise SystemExit(f"missing AQI should be a clean failure: {out.output}")
    _assert_air_recovery_message(out.output, "air missing AQI")
    if out.metadata.get("reason") != "missing_aqi":
        raise SystemExit(f"missing AQI should preserve reason metadata: {out.metadata}")
    _assert_air_handoff(
        out.metadata,
        "missing AQI",
        status="missing_data",
        reason="missing_aqi",
        calls_external_service=True,
        retry_safe=True,
    )


def test_planner_routes_air() -> None:
    p = RuleBasedPlanner()
    for q in [
        "air quality",
        "how's the air",
        "is the air bad",
        "how bad is the air in Seoul",
        "how clean is the air in Busan today",
        "fine dust",
        "air quality in Tokyo",
        "air Seoul",
        "Seoul air",
        "how is Seoul air",
        "Busan air today",
        "aqi Busan",
        "pollution in London",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["get_air_quality"]:
            raise SystemExit(f"air route missed: {q!r}")
    if [a.tool_name for a in p.plan("open air app").actions] != ["open_application"]:
        raise SystemExit("app launch phrase wrongly routed to air quality")


def main() -> None:
    test_is_local_safe()
    test_air_quality_argument_contract_fences_malformed_input()
    test_labels_aqi()
    test_reports_air_quality()
    test_resolves_common_city_from_text()
    test_unknown_explicit_location_never_fetches()
    test_invalid_coordinates_never_fetch()
    test_invalid_env_defaults_do_not_crash_import()
    test_handles_error()
    test_handles_missing_aqi()
    test_planner_routes_air()
    print("Air connector smoke passed")


if __name__ == "__main__":
    main()
