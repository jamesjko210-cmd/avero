"""Smoke tests for the weather connector (mocked fetch, no network)."""

from __future__ import annotations

import os

from jarvis_v2.agent.failure_guidance import EXTERNAL_INFORMATION_RECOVERY_ACTION
from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools import weather_connector as wc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry

NO_AUTHORITY_FLAGS = (
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
)


_SAMPLE = {
    "current_condition": [{
        "temp_C": "20", "FeelsLikeC": "18", "humidity": "60", "windspeedKmph": "9",
        "weatherDesc": [{"value": "Partly cloudy"}],
    }],
    "weather": [{"maxtempC": "24", "mintempC": "15"}],
    "nearest_area": [{"areaName": [{"value": "Seoul"}]}],
}


def _tools():
    return {t.name: t for t in wc.make_weather_tools(load_config())}


def assert_no_raw_local_path(value: object, label: str) -> None:
    text = str(value)
    if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"{label} leaked a raw local path: {value}")


def assert_weather_recovery_message(output: str, label: str) -> None:
    lowered = output.lower()
    for fragment in [
        "network access to wttr.in",
        "setup check",
        "retry in a moment",
    ]:
        if fragment not in lowered:
            raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")
    for forbidden in ["http error", "network down", "traceback"]:
        if forbidden in lowered:
            raise SystemExit(f"{label} leaked raw backend text {forbidden!r}: {output}")


def assert_external_information_guidance(result, label: str) -> None:
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    if EXTERNAL_INFORMATION_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} hid the canonical recovery action: {result.output}")
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} recovery field {key} drifted: {result.metadata}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": EXTERNAL_INFORMATION_RECOVERY_ACTION,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")


def assert_weather_handoff(metadata: dict, label: str, *, status: str, calls_external_service: bool) -> dict:
    if metadata.get("weather_handoff_ready") is not True:
        raise SystemExit(f"{label} missed weather_handoff_ready: {metadata}")
    handoff = metadata.get("weather_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed weather handoff: {metadata}")
    if handoff.get("source") != "get_weather":
        raise SystemExit(f"{label} handoff source wrong: {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} nested handoff_ready missing: {handoff}")
    if handoff.get("status") != status:
        raise SystemExit(f"{label} handoff status wrong: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} handoff should be operator-ready: {handoff}")
    if metadata.get("weather_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} weather ready alias parity failed: {metadata} vs {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} handoff should declare no state changes: {handoff}")
    if metadata.get("weather_state_changed") != handoff.get("state_changed") or metadata.get("weather_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} weather state alias parity failed: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False in flat and nested metadata: {metadata}")
        alias = f"weather_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} weather no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if handoff.get("weather_available") != (status == "ok"):
        raise SystemExit(f"{label} handoff availability wrong: {handoff}")
    if handoff.get("location") != metadata.get("location"):
        raise SystemExit(f"{label} handoff location parity failed: {metadata}")
    if handoff.get("location_chars") != metadata.get("location_chars"):
        raise SystemExit(f"{label} handoff location size parity failed: {metadata}")
    if handoff.get("location_truncated") != metadata.get("location_truncated"):
        raise SystemExit(f"{label} handoff truncation parity failed: {metadata}")
    if handoff.get("content_in_handoff") is not False or handoff.get("prose_in_metadata") is not False:
        raise SystemExit(f"{label} handoff should keep prose out of metadata: {handoff}")
    if metadata.get("weather_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} weather content alias parity failed: {metadata} vs {handoff}")
    if metadata.get("weather_prose_in_metadata") != handoff.get("prose_in_metadata"):
        raise SystemExit(f"{label} weather prose alias parity failed: {metadata} vs {handoff}")
    if not isinstance(handoff.get("next_safe_command"), str) or not handoff["next_safe_command"].startswith("weather"):
        raise SystemExit(f"{label} handoff missed retry/navigation command: {handoff}")
    expected_commands = [handoff["next_safe_command"]]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} weather safe-command list parity failed: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} weather safe-command count failed: {handoff}")
    if metadata.get("weather_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} weather next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("weather_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} weather next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("weather_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} weather next-command count alias parity failed: {metadata} vs {handoff}")
    assert_no_raw_local_path(handoff, f"{label} handoff")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} handoff missed boundaries: {handoff}")
    if boundaries.get("calls_external_service") is not calls_external_service or boundaries.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} handoff external-call boundary wrong: {handoff}")
    if metadata.get("weather_boundaries") != boundaries:
        raise SystemExit(f"{label} weather boundary alias parity failed: {metadata} vs {handoff}")
    for key in [
        "calls_model",
        "executes_tools",
        *NO_AUTHORITY_FLAGS,
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "requires_approval",
        "controls_computer",
    ]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} should not perform {key}: {metadata}")
    if metadata.get("calls_external_service") is not calls_external_service or metadata.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} flat external-call flags should match handoff: {metadata}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["get_weather"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("get_weather should be LOCAL_SAFE (read-only network)")


def test_weather_argument_contract_fences_malformed_input() -> None:
    fetch_calls: list[str] = []
    original_fetch = wc._fetch

    def mocked_fetch(location: str) -> dict:
        fetch_calls.append(location)
        return _SAMPLE

    try:
        wc._fetch = mocked_fetch  # type: ignore[assignment]
        private_sentinel = "private-weather-contract-sentinel"
        tool = _tools()["get_weather"]
        contract = tool.argument_contract
        actual_shape = (
            tuple(
                (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                for field in contract.fields
            )
            if contract is not None
            else ()
        )
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown is not False
            or actual_shape != (("location", ("string",), False), ("city", ("string",), False))
        ):
            raise SystemExit(f"get_weather should have an exact strict location/city contract: {contract}")

        registry = ToolRegistry()
        registry.register(tool)
        executor = Executor(registry, PermissionPolicy())
        malformed_args: list[object] = [
            None,
            ["not", "an", "object"],
            {"location": {"private": private_sentinel}},
            {"location": True},
            {"location": [private_sentinel]},
            {"city": 7},
            {"location": "Seoul", "unexpected": private_sentinel},
        ]
        rejected_surfaces: list[object] = []
        for args in malformed_args:
            result = executor.execute(PlannedAction("get_weather", args, "weather contract smoke"))  # type: ignore[arg-type]
            if (
                result.ok
                or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
                or result.metadata.get("requires_confirmation") is not False
            ):
                raise SystemExit(f"malformed weather input passed the executor: {result}")
            rejected_surfaces.append({"output": result.output, "metadata": result.metadata})
        if fetch_calls:
            raise SystemExit(f"malformed weather input reached wttr.in: {fetch_calls}")
        if private_sentinel in str(rejected_surfaces):
            raise SystemExit("weather argument rejection leaked a supplied private value")

        for args, expected_location in (({"location": "Busan"}, "Busan"), ({"city": "Busan"}, "Busan")):
            result = executor.execute(PlannedAction("get_weather", args, "valid weather contract smoke"))
            if not result.ok or fetch_calls[-1] != expected_location:
                raise SystemExit(f"valid weather input did not preserve the location alias: {args} {result}")
        empty = executor.execute(PlannedAction("get_weather", {}, "empty weather contract smoke"))
        if not empty.ok or empty.metadata.get("handler_invoked") is not True or fetch_calls[-1] != "Seoul":
            raise SystemExit(f"empty weather compatibility changed: {empty}")
    finally:
        wc._fetch = original_fetch  # type: ignore[assignment]


def test_formats_current_and_highlow() -> None:
    wc._fetch = lambda location: _SAMPLE  # type: ignore
    out = _tools()["get_weather"].handler({})
    if not out.ok:
        raise SystemExit(f"weather failed: {out.output}")
    for fragment in ["Seoul", "20°C", "feels 18°C", "Partly cloudy", "High 24", "Low 15", "humidity 60%"]:
        if fragment not in out.output:
            raise SystemExit(f"missing {fragment!r} in: {out.output}")
    handoff = assert_weather_handoff(out.metadata, "weather success", status="ok", calls_external_service=True)
    expected = {
        "resolved_location": "Seoul",
        "temp_c": "20",
        "feels_like_c": "18",
        "high_c": "24",
        "low_c": "15",
        "description": "Partly cloudy",
        "humidity_pct": "60",
        "wind_kmph": "9",
    }
    for key, value in expected.items():
        if handoff.get(key) != value:
            raise SystemExit(f"weather success handoff missed {key}: {handoff}")


def test_passes_location_through() -> None:
    seen = {}

    def fake_fetch(location):
        seen["loc"] = location
        return _SAMPLE

    wc._fetch = fake_fetch  # type: ignore
    out = _tools()["get_weather"].handler({"location": "Busan"})
    if seen.get("loc") != "Busan":
        raise SystemExit(f"location not passed to fetch: {seen}")
    if out.metadata.get("location") != "Busan" or out.metadata.get("resolved_location") != "Seoul":
        raise SystemExit(f"weather metadata should preserve cleaned and resolved locations: {out.metadata}")
    if out.metadata.get("location_chars") != 5 or out.metadata.get("location_truncated"):
        raise SystemExit(f"weather metadata should report location size without truncation: {out.metadata}")


def test_bounds_long_location_without_raw_echo() -> None:
    seen = {}

    def fake_fetch(location):
        seen["loc"] = location
        return _SAMPLE

    wc._fetch = fake_fetch  # type: ignore
    raw = "x" * 120
    out = _tools()["get_weather"].handler({"location": raw})
    if seen.get("loc") != "x" * wc.MAX_LOCATION_CHARS:
        raise SystemExit(f"long location was not bounded before fetch: {seen}")
    if out.metadata.get("location") != "x" * wc.MAX_LOCATION_CHARS:
        raise SystemExit(f"weather metadata should preserve cleaned bounded location: {out.metadata}")
    if out.metadata.get("location_chars") != wc.MAX_LOCATION_CHARS or out.metadata.get("location_truncated") is not True:
        raise SystemExit(f"weather metadata should report bounded location diagnostics: {out.metadata}")
    if "raw_location" in out.metadata:
        raise SystemExit(f"weather metadata should not echo raw location input: {out.metadata}")


def test_invalid_location_never_fetches() -> None:
    calls = []

    def fail_fetch(location):
        calls.append(location)
        raise AssertionError("invalid location should not fetch")

    wc._fetch = fail_fetch  # type: ignore
    for raw_location, expected_location, should_redact in (
        ("---", "---", False),
        ("/\x55sers/example/private/weather-location", "<local-path>", True),
        ("/private/tmp/weather-location", "<local-path>", True),
        ("/var/folders/zc/weather-location", "<local-path>", True),
        ("/tmp/weather-location", "<local-path>", True),
    ):
        out = _tools()["get_weather"].handler({"location": raw_location})
        if out.ok or "real location" not in out.output.lower():
            raise SystemExit(f"invalid location should ask for a real place: {out.output}")
        if out.metadata.get("reason") != "invalid_location" or out.metadata.get("location") != expected_location:
            raise SystemExit(f"invalid location should preserve safe bounded metadata: {out.metadata}")
        handoff = assert_weather_handoff(out.metadata, f"invalid weather location {raw_location!r}", status="invalid_input", calls_external_service=False)
        if handoff.get("reason") != "invalid_location" or handoff.get("retry_safe") is not False:
            raise SystemExit(f"invalid weather handoff should preserve invalid-input state: {handoff}")
        assert_no_raw_local_path(out.output, "invalid weather location output")
        assert_no_raw_local_path(out.metadata, "invalid weather location metadata")
        if should_redact and "<local-path>" not in str(out.metadata):
            raise SystemExit(f"invalid path-shaped location should expose a redacted marker: {out.metadata}")
    if calls:
        raise SystemExit(f"invalid location unexpectedly fetched: {calls}")


def test_default_location_reads_environment_at_call_time() -> None:
    seen = {}
    old = os.environ.get("JARVIS_WEATHER_LOCATION")

    def fake_fetch(location):
        seen["loc"] = location
        return _SAMPLE

    try:
        os.environ["JARVIS_WEATHER_LOCATION"] = "Incheon"
        wc._fetch = fake_fetch  # type: ignore
        _tools()["get_weather"].handler({})
        if seen.get("loc") != "Incheon":
            raise SystemExit(f"default location did not read env at call time: {seen}")
    finally:
        if old is None:
            os.environ.pop("JARVIS_WEATHER_LOCATION", None)
        else:
            os.environ["JARVIS_WEATHER_LOCATION"] = old


def test_invalid_default_location_falls_back_locally() -> None:
    seen = {}
    old = os.environ.get("JARVIS_WEATHER_LOCATION")

    def fake_fetch(location):
        seen["loc"] = location
        return _SAMPLE

    try:
        os.environ["JARVIS_WEATHER_LOCATION"] = "/\x55sers/example/private/weather-default"
        wc._fetch = fake_fetch  # type: ignore
        out = _tools()["get_weather"].handler({})
        if not out.ok:
            raise SystemExit(f"weather should fall back from invalid default env: {out.output}")
        if seen.get("loc") != "Seoul":
            raise SystemExit(f"path-shaped default weather env should fall back before fetch: {seen}")
        if out.metadata.get("location") != "Seoul" or out.metadata.get("location_chars") != len("Seoul"):
            raise SystemExit(f"fallback metadata should report safe default location: {out.metadata}")
    finally:
        if old is None:
            os.environ.pop("JARVIS_WEATHER_LOCATION", None)
        else:
            os.environ["JARVIS_WEATHER_LOCATION"] = old


def test_temp_path_default_location_falls_back_locally() -> None:
    seen = {}
    old = os.environ.get("JARVIS_WEATHER_LOCATION")

    def fake_fetch(location):
        seen["loc"] = location
        return _SAMPLE

    try:
        for raw_default in ("/var/folders/zc/weather-default", "/tmp/weather-default"):
            seen.clear()
            os.environ["JARVIS_WEATHER_LOCATION"] = raw_default
            wc._fetch = fake_fetch  # type: ignore
            out = _tools()["get_weather"].handler({})
            if not out.ok:
                raise SystemExit(f"weather should fall back from temp-path default env: {out.output}")
            if seen.get("loc") != "Seoul":
                raise SystemExit(f"temp-path default weather env should fall back before fetch: {seen}")
            if out.metadata.get("location") != "Seoul" or out.metadata.get("location_chars") != len("Seoul"):
                raise SystemExit(f"fallback metadata should report safe default location: {out.metadata}")
    finally:
        if old is None:
            os.environ.pop("JARVIS_WEATHER_LOCATION", None)
        else:
            os.environ["JARVIS_WEATHER_LOCATION"] = old


def test_handles_fetch_error() -> None:
    def boom(location):
        raise RuntimeError("network down")

    wc._fetch = boom  # type: ignore
    out = _tools()["get_weather"].handler({"location": "  Busan   beach  "})
    if out.ok:
        raise SystemExit(f"fetch error not handled: {out.output}")
    assert_weather_recovery_message(out.output, "generic weather fetch error")
    assert_external_information_guidance(out, "generic weather fetch error")
    if out.metadata.get("location") != "Busan beach":
        raise SystemExit(f"weather error metadata should preserve cleaned location: {out.metadata}")
    if out.metadata.get("location_chars") != len("Busan beach") or out.metadata.get("location_truncated"):
        raise SystemExit(f"weather error metadata should report location diagnostics: {out.metadata}")
    handoff = assert_weather_handoff(out.metadata, "weather fetch error", status="unavailable", calls_external_service=True)
    if handoff.get("reason") != "fetch_error" or handoff.get("retry_safe") is not True:
        raise SystemExit(f"weather fetch error handoff should preserve retryable service state: {handoff}")


def test_http_status_errors_are_friendly() -> None:
    def server_error(location):
        raise RuntimeError("HTTP Error 500: Internal Server Error")

    wc._fetch = server_error  # type: ignore
    server_out = _tools()["get_weather"].handler({"location": "Seoul"})
    if server_out.ok:
        raise SystemExit(f"HTTP 500 should be a service-trouble message: {server_out.output}")
    assert_weather_recovery_message(server_out.output, "weather HTTP 500")
    assert_external_information_guidance(server_out, "weather HTTP 500")

    def not_found(location):
        raise RuntimeError("HTTP Error 404: Not Found")

    wc._fetch = not_found  # type: ignore
    not_found_out = _tools()["get_weather"].handler({"location": "badcity"})
    if not_found_out.ok or "couldn't find weather for 'badcity'" not in not_found_out.output:
        raise SystemExit(f"HTTP 404 should be a friendly lookup miss: {not_found_out.output}")
    if "HTTP Error" in not_found_out.output:
        raise SystemExit(f"HTTP 404 should not leak raw error prefix: {not_found_out.output}")
    if "setup check" in not_found_out.output.lower() or "network access" in not_found_out.output.lower():
        raise SystemExit(f"HTTP 404 should preserve precise not-found guidance without setup noise: {not_found_out.output}")
    handoff = assert_weather_handoff(not_found_out.metadata, "weather not found", status="unavailable", calls_external_service=True)
    if handoff.get("reason") != "not_found":
        raise SystemExit(f"weather 404 handoff should preserve not-found reason: {handoff}")


def test_missing_current_temperature_is_not_success() -> None:
    wc._fetch = lambda location: {"current_condition": [{}], "weather": [{}], "nearest_area": []}  # type: ignore
    out = _tools()["get_weather"].handler({"location": "Nowhere"})
    if out.ok or "No current weather returned" not in out.output:
        raise SystemExit(f"missing temperature should not format as success: {out.output}")
    assert_weather_recovery_message(out.output, "weather missing current temperature")
    if out.metadata.get("location") != "Nowhere" or out.metadata.get("location_chars") != len("Nowhere"):
        raise SystemExit(f"missing temperature metadata should preserve cleaned location diagnostics: {out.metadata}")
    handoff = assert_weather_handoff(out.metadata, "weather missing temperature", status="unavailable", calls_external_service=True)
    if handoff.get("reason") != "missing_current_temperature":
        raise SystemExit(f"missing temperature handoff should preserve no-data reason: {handoff}")


def test_planner_routes_weather() -> None:
    p = RuleBasedPlanner()
    for q in [
        "what's the weather",
        "weather in Tokyo",
        "is it going to rain",
        "will it rain today",
        "will it snow tomorrow",
        "what's the temperature",
        "what's the temperature outside",
        "current temperature",
        "temperature outside",
        "temp today",
        "how hot will it be today",
        "how cold will it be tomorrow",
        "will it be cold tomorrow",
        "will it be hot today",
        "do I need an umbrella today",
        "should I bring an umbrella",
        # Real gap found live 2026-07-10: "what's the weather like in 3 days"
        # was misrouted to the generic calculate() tool (an unrelated, much
        # broader "what's <alnum expression with a digit>" matcher checked
        # earlier in the planner chain treated "the weather like in 3 days" as
        # a calculator expression, since it contains a digit and no
        # arithmetic symbol) instead of get_weather. Fails safely (a clean
        # refusal, not a wrong number) but is the wrong tool for a weather
        # question.
        "what's the weather like in 3 days",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["get_weather"]:
            raise SystemExit(f"planner missed weather route: {q!r}")
    if p.plan("weather in Tokyo").actions[0].args.get("location") != "tokyo":
        raise SystemExit("planner did not extract weather location")
    for q, expected in {
        "weather Seoul": "seoul",
        "forecast Busan": "busan",
        "weather Seoul today": "seoul",
        "Seoul weather": "seoul",
        "Tokyo forecast": "tokyo",
        "Busan weather today": "busan",
        "rain forecast Seoul": "seoul",
        "Seoul rain forecast": "seoul",
        "rain Seoul": "seoul",
        "snow Tokyo": "tokyo",
        "umbrella Busan": "busan",
        "should I bring an umbrella to Seoul today": "seoul",
        "should I bring umbrella to Gangnam": "gangnam",
        "weather tomorrow Seoul": "seoul",
        "forecast tomorrow Seoul": "seoul",
        "tomorrow weather Seoul": "seoul",
        "temperature in Seoul": "seoul",
        "temp in Busan": "busan",
        "temperature Seoul": "seoul",
        "Busan temperature": "busan",
        # Real gap found live 2026-07-10: in a compound sentence, the lazy
        # location group couldn't find its terminator within 40 chars
        # starting at "in seoul" (no temporal/punctuation word followed), so
        # the match failed there and `re.search` fell back to a LATER,
        # unrelated "to X" clause -- "check the weather in seoul and then add
        # a task to bring an umbrella" silently asked for weather in "bring
        # an umbrella" instead of seoul.
        "can you check the weather in seoul and then add a task to bring an umbrella": "seoul",
        "what's the weather in tokyo and seoul": "tokyo",
    }.items():
        action = p.plan(q).actions[0]
        if action.tool_name != "get_weather" or action.args.get("location") != expected:
            raise SystemExit(f"planner did not extract terse weather location for {q!r}: {action}")
    if p.plan("weather").actions[0].args:
        raise SystemExit("bare weather should keep using the configured default location")
    if p.plan("the weather").actions[0].args:
        raise SystemExit("generic 'the weather' should keep using the configured default location")
    if p.plan("weather tomorrow").actions[0].args:
        raise SystemExit("generic 'weather tomorrow' should keep using the configured default location")
    for q in [
        "rain tomorrow",
        "snow tomorrow",
        "umbrella tomorrow",
        "weather this weekend",
        "weekend weather",
        "weather tonight",
        "tonight weather",
        "weather later",
        "weather this afternoon",
    ]:
        action = p.plan(q).actions[0]
        if action.tool_name != "get_weather" or action.args:
            raise SystemExit(f"temporal weather phrase should use default location for {q!r}: {action}")
    for q in [
        "tell me about the weather",
        "what is the weather",
        "tell me weather",
        "tell me about weather",
    ]:
        action = p.plan(q).actions[0]
        if action.tool_name != "get_weather" or action.args:
            raise SystemExit(f"generic weather phrase should use default location for {q!r}: {action}")
    for q, expected in {
        "tell me about the weather in Seoul": "seoul",
        "tell me about Seoul weather": "seoul",
        "what is the weather in Busan": "busan",
    }.items():
        action = p.plan(q).actions[0]
        if action.tool_name != "get_weather" or action.args.get("location") != expected:
            raise SystemExit(f"natural weather phrase did not extract location for {q!r}: {action}")


def main() -> None:
    test_is_local_safe()
    test_weather_argument_contract_fences_malformed_input()
    test_formats_current_and_highlow()
    test_passes_location_through()
    test_bounds_long_location_without_raw_echo()
    test_invalid_location_never_fetches()
    test_default_location_reads_environment_at_call_time()
    test_invalid_default_location_falls_back_locally()
    test_temp_path_default_location_falls_back_locally()
    test_handles_fetch_error()
    test_http_status_errors_are_friendly()
    test_missing_current_temperature_is_not_success()
    test_planner_routes_weather()
    print("Weather connector smoke passed")


if __name__ == "__main__":
    main()
