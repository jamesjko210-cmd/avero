"""Smoke tests for the sunrise/sunset connector (mocked fetch, no network)."""

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
from jarvis_v2.tools import sun_connector as sc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry


NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


def _tools():
    return {t.name: t for t in sc.make_sun_tools(load_config())}


def _assert_sun_recovery_message(output: str, label: str) -> None:
    lowered = output.lower()
    for fragment in [
        "network access to sunrise-sunset.org",
        "setup check",
        "retry in a moment",
    ]:
        if fragment not in lowered:
            raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")
    for forbidden in ["http error", "offline", "traceback", "/users/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in lowered:
            raise SystemExit(f"{label} leaked raw backend text {forbidden!r}: {output}")


def _assert_sun_handoff(
    metadata: dict[str, Any],
    label: str,
    *,
    status: str,
    reason: str = "",
    calls_external_service: bool = True,
    retry_safe: bool = False,
) -> dict[str, Any]:
    handoff = metadata.get("sun_times_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing sun_times_handoff: {metadata}")
    if handoff.get("source") != "get_sun_times" or handoff.get("status") != status:
        raise SystemExit(f"{label} has wrong source/status: {handoff}")
    if metadata.get("sun_times_handoff_ready") is not True or handoff.get("sun_times_handoff_ready") is not True:
        raise SystemExit(f"{label} should expose flat and nested handoff readiness: {metadata} vs {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should expose nested handoff_ready: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should be operator-ready: {handoff}")
    if metadata.get("sun_times_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} ready alias parity failed: {metadata} vs {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} has wrong reason: {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} should not claim changed state: {handoff}")
    if metadata.get("sun_times_state_changed") != handoff.get("state_changed") or metadata.get("sun_times_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} state alias parity failed: {metadata} vs {handoff}")
    if handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep output content out of handoff: {handoff}")
    if metadata.get("sun_times_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} content-in-handoff alias parity failed: {metadata} vs {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
        alias = f"sun_times_{key}"
        if metadata.get(alias) is not expected_value or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    for key in ("day", "lat", "lng", "exception_type"):
        if handoff.get(key) != metadata.get(key, ""):
            raise SystemExit(f"{label} should mirror {key}: {handoff} vs {metadata}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} should mark content exclusion: {handoff}")
    if metadata.get("sun_times_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} content-in-metadata alias parity failed: {metadata} vs {handoff}")
    if handoff.get("retry_safe") is not retry_safe:
        raise SystemExit(f"{label} has wrong retry safety: {handoff}")
    if handoff.get("next_safe_command") != "sunset today":
        raise SystemExit(f"{label} has wrong next safe command: {handoff}")
    expected_commands = ["sunset today"]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} has wrong next safe command list: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} has wrong next safe command count: {handoff}")
    if metadata.get("sun_times_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("sun_times_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("sun_times_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} next-command-count alias parity failed: {metadata} vs {handoff}")
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
    if metadata.get("sun_times_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary alias parity failed: {metadata} vs {handoff}")
    if metadata.get("calls_external_service") is not calls_external_service:
        raise SystemExit(f"{label} flat external-call flag should match handoff: {metadata}")
    if metadata.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} flat plural external-call flag should match handoff: {metadata}")
    combined = f"{metadata}\n{handoff}".lower()
    for fragment in ("/users/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in combined:
            raise SystemExit(f"{label} leaked local path fragment {fragment!r}: {metadata} {handoff}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["get_sun_times"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("get_sun_times should be LOCAL_SAFE")


def test_sun_argument_contract_fences_malformed_input() -> None:
    fetch_calls: list[tuple[float, float, str]] = []
    original_fetch = sc._fetch

    def mocked_fetch(lat: float, lng: float, day: str) -> dict:
        fetch_calls.append((lat, lng, day))
        return {
            "results": {
                "sunrise": "2026-06-16T20:00:00+00:00",
                "sunset": "2026-06-17T11:00:00+00:00",
            }
        }

    try:
        sc._fetch = mocked_fetch  # type: ignore[assignment]
        private_sentinel = "private-sun-contract-sentinel"
        tool = _tools()["get_sun_times"]
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
            raise SystemExit(f"get_sun_times should have an exact strict argument contract: {contract}")

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
                PlannedAction("get_sun_times", args, "sun contract smoke")  # type: ignore[arg-type]
            )
            if (
                result.ok
                or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
                or result.metadata.get("requires_confirmation") is not False
            ):
                raise SystemExit(f"malformed sun-times input passed the executor: {result}")
            rejected_surfaces.append({"output": result.output, "metadata": result.metadata})
        if fetch_calls:
            raise SystemExit(f"malformed sun-times input reached sunrise-sunset.org: {fetch_calls}")
        if private_sentinel in str(rejected_surfaces):
            raise SystemExit("sun-times argument rejection leaked a supplied private value")

        valid_cases = (
            ({"location": "Tokyo"}, sc.CITY_COORDS["tokyo"][1:3], "today"),
            ({"city": "Busan"}, sc.CITY_COORDS["busan"][1:3], "today"),
            ({"text": "sunrise in Seoul tomorrow"}, sc.CITY_COORDS["seoul"][1:3], "tomorrow"),
            ({"lat": "35.0", "lng": 129.0}, (35.0, 129.0), "today"),
            ({}, (sc.DEFAULT_LAT, sc.DEFAULT_LNG), "today"),
        )
        for args, expected_coords, expected_day in valid_cases:
            result = executor.execute(
                PlannedAction("get_sun_times", args, "valid sun contract smoke")
            )
            if (
                not result.ok
                or fetch_calls[-1][:2] != expected_coords
                or fetch_calls[-1][2] != expected_day
            ):
                raise SystemExit(f"valid sun-times input changed: {args} {result} {fetch_calls}")
    finally:
        sc._fetch = original_fetch  # type: ignore[assignment]


def test_reports_times() -> None:
    sc._fetch = lambda lat, lng, day: {  # type: ignore
        "results": {"sunrise": "2026-06-15T20:08:50+00:00", "sunset": "2026-06-16T10:56:43+00:00"}
    }
    out = _tools()["get_sun_times"].handler({"text": "sunset today"})
    if not out.ok or "sunrise" not in out.output or "sunset" not in out.output:
        raise SystemExit(f"sun times output wrong: {out.output}")
    if out.metadata.get("lat", {}).get("value") != sc.DEFAULT_LAT or out.metadata.get("lng", {}).get("value") != sc.DEFAULT_LNG:
        raise SystemExit(f"sun times should preserve coordinate metadata: {out.metadata}")
    handoff = _assert_sun_handoff(out.metadata, "sun times success", status="ok")
    if not handoff.get("times_available") or not handoff.get("sunrise") or not handoff.get("sunset"):
        raise SystemExit(f"sun times success should preserve local time fields: {handoff}")


def test_resolves_common_city_from_text() -> None:
    seen = {}

    def fake_fetch(lat, lng, day):
        seen["lat"] = lat
        seen["lng"] = lng
        seen["day"] = day
        return {"results": {"sunrise": "2026-06-15T20:08:50+00:00", "sunset": "2026-06-16T10:56:43+00:00"}}

    sc._fetch = fake_fetch  # type: ignore
    out = _tools()["get_sun_times"].handler({"text": "sunrise in Tokyo"})
    if not out.ok or "Tokyo Today" not in out.output:
        raise SystemExit(f"city sun time output wrong: {out.output}")
    if seen.get("lat") == sc.DEFAULT_LAT or seen.get("lng") == sc.DEFAULT_LNG:
        raise SystemExit(f"city sun time should not use default coordinates: {seen}")
    handoff = _assert_sun_handoff(out.metadata, "city sun times", status="ok")
    location = handoff.get("location") or {}
    if location.get("resolved_location") != "Tokyo" or out.metadata.get("sun_times_location", {}).get("resolved_location") != "Tokyo":
        raise SystemExit(f"city sun time should preserve resolved location: {out.metadata}")
    if out.metadata.get("lat", {}).get("source") != "location" or out.metadata.get("lng", {}).get("source") != "location":
        raise SystemExit(f"city sun time should mark coordinate source: {out.metadata}")


def test_resolved_city_uses_its_own_timezone_not_the_machines() -> None:
    """Real bug found live 2026-07-08/09: `_local()` converted the API's UTC
    response to the machine's own system timezone regardless of which city was
    asked about, so every city other than the machine's timezone (and ones that
    happen to share it) returned wrong times -- observed live: London came back
    as "sunrise 12:51, sunset 05:19" (sunrise later than sunset, both
    implausible). Uses a January UTC timestamp (London is on plain GMT/UTC+0
    then, no DST ambiguity) so the correct London local time trivially equals
    the UTC clock time -- proving the conversion used London's own timezone and
    not some other offset (e.g. the machine's, which is UTC+9 in this repo's
    dev environment and would show a very different, wrong hour)."""
    sc._fetch = lambda lat, lng, day: {  # type: ignore
        "results": {"sunrise": "2026-01-15T08:30:00+00:00", "sunset": "2026-01-15T16:45:00+00:00"}
    }
    out = _tools()["get_sun_times"].handler({"text": "sunrise in london"})
    if not out.ok or "sunrise 08:30" not in out.output or "sunset 16:45" not in out.output:
        raise SystemExit(f"London sun times should use London's own GMT timezone, not the machine's: {out.output}")


def test_sunset_time_in_city_phrasing_resolves_location() -> None:
    """Real bug found live 2026-07-08/09: "sunset TIME in london" (the extra
    word "time") fell through to an overly greedy fallback pattern that
    captured "time in london" itself as the location string, which then failed
    the exact-match city lookup -- even though bare "sunset in london" (no
    "time") already worked and London is listed as a supported example city in
    the tool's own unknown-location message."""
    sc._fetch = lambda lat, lng, day: {  # type: ignore
        "results": {"sunrise": "2026-01-15T08:30:00+00:00", "sunset": "2026-01-15T16:45:00+00:00"}
    }
    for text in ("sunset time in london", "sunrise time in london", "sunset times in london"):
        out = _tools()["get_sun_times"].handler({"text": text})
        if not out.ok or "London Today" not in out.output:
            raise SystemExit(f"{text!r} should resolve London despite the extra 'time' word: {out.output}")


def test_unknown_explicit_location_never_fetches() -> None:
    calls = []

    def fail_fetch(lat, lng, day):
        calls.append((lat, lng, day))
        raise AssertionError("unknown explicit location should not fetch")

    sc._fetch = fail_fetch  # type: ignore
    out = _tools()["get_sun_times"].handler({"text": "sunrise in Atlantis"})
    if out.ok or "don't know that sunrise/sunset location" not in out.output:
        raise SystemExit(f"unknown location should be refused cleanly: {out.output}")
    handoff = _assert_sun_handoff(
        out.metadata,
        "unknown sun location",
        status="refused",
        reason="unknown_location",
        calls_external_service=False,
    )
    if handoff.get("location", {}).get("raw") != "atlantis":
        raise SystemExit(f"unknown location should preserve bounded raw value: {handoff}")
    if calls:
        raise SystemExit(f"unknown location unexpectedly fetched: {calls}")
    path_out = _tools()["get_sun_times"].handler({"text": "sunrise in /USERS/OPERATOR/PRIVATE/SUN-CITY"})
    if path_out.ok or "don't know that sunrise/sunset location" not in path_out.output:
        raise SystemExit(f"path-shaped location should be refused cleanly: {path_out.output}")
    path_handoff = _assert_sun_handoff(
        path_out.metadata,
        "path-shaped sun location",
        status="refused",
        reason="unknown_location",
        calls_external_service=False,
    )
    if path_handoff.get("location", {}).get("raw") != "<local-path>":
        raise SystemExit(f"path-shaped location should be redacted in handoff: {path_handoff}")
    if calls:
        raise SystemExit(f"path-shaped location unexpectedly fetched: {calls}")


def test_invalid_coordinates_never_fetch() -> None:
    calls = []

    def fail_fetch(lat, lng, day):
        calls.append((lat, lng, day))
        raise AssertionError("invalid coordinates should not fetch")

    sc._fetch = fail_fetch  # type: ignore
    for args in [
        {"lat": "north"},
        {"lat": 91},
        {"lng": 181},
        {"lng": "inf"},
        {"lat": "/\x55sers/example/private/sun-lat"},
        {"lng": "/private/tmp/sun-lng"},
        {"lat": "/var/folders/zc/jarvis-sun-lat"},
        {"lng": "/tmp/jarvis-sun-lng"},
        {"lat": "/USERS/OPERATOR/PRIVATE/SUN-LAT"},
        {"lng": "/TMP/JARVIS-SUN-LNG"},
    ]:
        out = _tools()["get_sun_times"].handler(args)
        if out.ok or "latitude" not in out.output.lower():
            raise SystemExit(f"invalid coordinates should be rejected locally: {args!r} -> {out.output}")
        if out.metadata.get("reason") != "invalid_coordinates":
            raise SystemExit(f"invalid coordinates should include reason metadata: {out.metadata}")
        _assert_sun_handoff(
            out.metadata,
            f"invalid coordinates {args!r}",
            status="refused",
            reason="invalid_coordinates",
            calls_external_service=False,
        )
        for coord_key in ("lat", "lng"):
            raw = str(out.metadata.get(coord_key, {}).get("raw", "")).lower()
            if "/users/" in raw or "/private/" in raw or "/var/folders" in raw or "/tmp/" in raw:
                raise SystemExit(f"invalid coordinates leaked local path metadata: {out.metadata}")
    if calls:
        raise SystemExit(f"invalid coordinates unexpectedly fetched: {calls}")


def test_invalid_env_defaults_do_not_crash_import() -> None:
    env = os.environ.copy()
    env["JARVIS_LAT"] = "nan"
    env["JARVIS_LNG"] = "east"
    code = (
        "from jarvis_v2.tools import sun_connector as sc; "
        "assert sc.DEFAULT_LAT == sc.SEOUL_LAT; "
        "assert sc.DEFAULT_LNG == sc.SEOUL_LNG"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise SystemExit(f"invalid env defaults should not crash sun import: {result.stderr or result.stdout}")


def test_tomorrow_passed_to_fetch() -> None:
    seen = {}

    def fake_fetch(lat, lng, day):
        seen["day"] = day
        return {"results": {"sunrise": "2026-06-16T20:00:00+00:00", "sunset": "2026-06-17T11:00:00+00:00"}}

    sc._fetch = fake_fetch  # type: ignore
    out = _tools()["get_sun_times"].handler({"text": "when is sunrise tomorrow"})
    if seen.get("day") != "tomorrow":
        raise SystemExit(f"tomorrow not passed to fetch: {seen}")
    handoff = _assert_sun_handoff(out.metadata, "tomorrow sun times", status="ok")
    if handoff.get("day") != "tomorrow":
        raise SystemExit(f"tomorrow should be preserved in handoff: {handoff}")


def test_missing_sunrise_is_clean() -> None:
    sc._fetch = lambda lat, lng, day: {"results": {}}  # type: ignore
    out = _tools()["get_sun_times"].handler({"text": "sunset"})
    if out.ok or "sunrise/sunset data was incomplete" not in out.output.lower():
        raise SystemExit(f"missing sunrise should be clean failure: {out.output}")
    _assert_sun_recovery_message(out.output, "missing sunrise")
    _assert_sun_handoff(
        out.metadata,
        "missing sunrise",
        status="missing_data",
        reason="missing_sunrise",
        retry_safe=True,
    )


def test_handles_error() -> None:
    def boom(lat, lng, day):
        raise RuntimeError("offline")

    sc._fetch = boom  # type: ignore
    out = _tools()["get_sun_times"].handler({"text": "sunset"})
    if out.ok or not out.output.strip():
        raise SystemExit(f"error not handled: {out.output}")
    if "offline" in out.output or out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"sun error should be friendly with bounded exception metadata: {out.output} {out.metadata}")
    _assert_sun_recovery_message(out.output, "sun generic fetch error")
    _assert_sun_handoff(
        out.metadata,
        "sun fetch error",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )

    def server_error(lat, lng, day):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    sc._fetch = server_error  # type: ignore
    server_out = _tools()["get_sun_times"].handler({"text": "sunset"})
    if server_out.ok or server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"HTTP 500 should preserve bounded exception metadata: {server_out.output} {server_out.metadata}")
    _assert_sun_recovery_message(server_out.output, "sun HTTP 500")
    _assert_sun_handoff(
        server_out.metadata,
        "sun HTTP 500 fetch error",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )

    def timeout_error(lat, lng, day):
        raise TimeoutError("request timed out near /\x55sers/example/private")

    sc._fetch = timeout_error  # type: ignore
    timeout_out = _tools()["get_sun_times"].handler({"text": "sunset"})
    if timeout_out.ok or timeout_out.metadata.get("exception_type") != "TimeoutError":
        raise SystemExit(f"timeout should preserve bounded exception metadata: {timeout_out.output} {timeout_out.metadata}")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"timeout recovery should preserve timeout cause: {timeout_out.output}")
    _assert_sun_recovery_message(timeout_out.output, "sun timeout")
    _assert_sun_handoff(
        timeout_out.metadata,
        "sun timeout fetch error",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )


def test_planner_routes_sun() -> None:
    p = RuleBasedPlanner()
    for q in [
        "what time is sunset",
        "when is sunrise",
        "sunset today",
        "sunrise in Tokyo",
        "sunset Busan",
        "what time does the sun set in Tokyo",
        "what time does the sun rise in Seoul",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["get_sun_times"]:
            raise SystemExit(f"sun route missed: {q!r}")
    if [a.tool_name for a in p.plan("what time is it in Tokyo").actions] != ["current_time"]:
        raise SystemExit("world-time route should still handle ordinary time lookups")


def main() -> None:
    test_is_local_safe()
    test_sun_argument_contract_fences_malformed_input()
    test_reports_times()
    test_resolves_common_city_from_text()
    test_resolved_city_uses_its_own_timezone_not_the_machines()
    test_sunset_time_in_city_phrasing_resolves_location()
    test_unknown_explicit_location_never_fetches()
    test_invalid_coordinates_never_fetch()
    test_invalid_env_defaults_do_not_crash_import()
    test_tomorrow_passed_to_fetch()
    test_missing_sunrise_is_clean()
    test_handles_error()
    test_planner_routes_sun()
    print("Sun connector smoke passed")


if __name__ == "__main__":
    main()
