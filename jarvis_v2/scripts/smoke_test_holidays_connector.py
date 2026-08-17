"""Smoke tests for the public-holidays connector (mocked fetch, no network)."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from typing import Any

from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import holidays_connector as hc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry


_SAMPLE = [
    {"date": "2026-07-17", "name": "Constitution Day", "localName": "제헌절"},
    {"date": "2026-08-15", "name": "Liberation Day", "localName": "광복절"},
]


NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


def _tools():
    return {t.name: t for t in hc.make_holidays_tools(load_config())}


def _assert_holidays_recovery_message(output: str, label: str) -> None:
    lowered = output.lower()
    for fragment in [
        "network access to nager.date holidays",
        "setup check",
        "retry in a moment",
    ]:
        if fragment not in lowered:
            raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")
    for forbidden in ["http error", "offline", "traceback", "/users/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in lowered:
            raise SystemExit(f"{label} leaked raw backend text {forbidden!r}: {output}")


def _assert_holidays_handoff(
    metadata: dict[str, Any],
    label: str,
    *,
    status: str,
    reason: str = "",
    calls_external_service: bool = False,
    count: int = 0,
    retry_safe: bool = False,
) -> dict[str, Any]:
    handoff = metadata.get("holidays_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing holidays_handoff: {metadata}")
    if handoff.get("source") != "next_holidays" or handoff.get("status") != status:
        raise SystemExit(f"{label} has wrong handoff source/status: {handoff}")
    if metadata.get("holidays_handoff_ready") is not True or handoff.get("holidays_handoff_ready") is not True:
        raise SystemExit(f"{label} should expose flat and nested handoff readiness: {metadata} vs {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should expose nested handoff_ready: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should be operator-ready: {handoff}")
    if metadata.get("holidays_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} ready alias parity failed: {metadata} vs {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
        alias = f"holidays_{key}"
        if metadata.get(alias) is not expected_value or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} has wrong handoff reason: {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} should not claim changed state: {handoff}")
    if metadata.get("holidays_state_changed") != handoff.get("state_changed") or metadata.get("holidays_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} state alias parity failed: {metadata} vs {handoff}")
    if handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep output content out of handoff: {handoff}")
    if metadata.get("holidays_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} content-in-handoff alias parity failed: {metadata} vs {handoff}")
    if handoff.get("country") != metadata.get("country") or handoff.get("country_source") != metadata.get("source"):
        raise SystemExit(f"{label} should mirror country metadata: {handoff} vs {metadata}")
    if handoff.get("raw_country") != metadata.get("raw_country", ""):
        raise SystemExit(f"{label} should mirror raw country: {handoff} vs {metadata}")
    if handoff.get("holiday_count") != count:
        raise SystemExit(f"{label} has wrong holiday count: {handoff}")
    if "count" in metadata and handoff.get("holiday_count") != metadata.get("count"):
        raise SystemExit(f"{label} should mirror count metadata: {handoff} vs {metadata}")
    if handoff.get("exception_type", "") != metadata.get("exception_type", ""):
        raise SystemExit(f"{label} should mirror exception type: {handoff} vs {metadata}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} should mark content exclusion: {handoff}")
    if metadata.get("holidays_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} content-in-metadata alias parity failed: {metadata} vs {handoff}")
    if handoff.get("retry_safe") is not retry_safe:
        raise SystemExit(f"{label} has wrong retry safety: {handoff}")
    if not str(handoff.get("next_safe_command") or "").startswith("next holidays"):
        raise SystemExit(f"{label} has wrong next safe command: {handoff}")
    expected_commands = [handoff["next_safe_command"]]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} has wrong next safe command list: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} has wrong next safe command count: {handoff}")
    if metadata.get("holidays_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("holidays_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("holidays_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} next-command-count alias parity failed: {metadata} vs {handoff}")
    raw_country = str(handoff.get("raw_country") or "")
    if any(fragment in raw_country for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
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
    if metadata.get("holidays_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary alias parity failed: {metadata} vs {handoff}")
    if metadata.get("calls_external_service") is not calls_external_service:
        raise SystemExit(f"{label} flat external-call flag should match handoff: {metadata}")
    if metadata.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} flat plural external-call flag should match handoff: {metadata}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["next_holidays"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("next_holidays should be LOCAL_SAFE")


def test_holidays_argument_contract_fences_malformed_input() -> None:
    fetch_calls: list[str] = []
    original_fetch = hc._fetch_next

    def mocked_fetch(country: str) -> list[dict[str, str]]:
        fetch_calls.append(country)
        return _SAMPLE

    try:
        hc._fetch_next = mocked_fetch  # type: ignore[assignment]
        private_sentinel = "private-holiday-contract-sentinel"
        tool = _tools()["next_holidays"]
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
            or actual_shape
            != (("text", ("string",), False), ("country", ("string",), False))
        ):
            raise SystemExit(f"next_holidays should have an exact strict text/country contract: {contract}")

        registry = ToolRegistry()
        registry.register(tool)
        executor = Executor(registry, PermissionPolicy())
        malformed_args: list[object] = [
            None,
            ["not", "an", "object"],
            {"text": {"private": private_sentinel}},
            {"text": True},
            {"country": 7},
            {"country": [private_sentinel]},
            {"country": "KR", "unexpected": private_sentinel},
        ]
        rejected_surfaces: list[object] = []
        for args in malformed_args:
            result = executor.execute(
                PlannedAction("next_holidays", args, "holiday contract smoke")  # type: ignore[arg-type]
            )
            if (
                result.ok
                or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
                or result.metadata.get("requires_confirmation") is not False
            ):
                raise SystemExit(f"malformed holiday input passed the executor: {result}")
            rejected_surfaces.append({"output": result.output, "metadata": result.metadata})
        if fetch_calls:
            raise SystemExit(f"malformed holiday input reached Nager.Date: {fetch_calls}")
        if private_sentinel in str(rejected_surfaces):
            raise SystemExit("holiday argument rejection leaked a supplied private value")

        valid_cases = (
            ({"country": "US"}, "US"),
            ({"text": "next holidays in Japan"}, "JP"),
            ({}, hc.DEFAULT_COUNTRY),
        )
        for args, expected_country in valid_cases:
            result = executor.execute(
                PlannedAction("next_holidays", args, "valid holiday contract smoke")
            )
            if not result.ok or fetch_calls[-1] != expected_country:
                raise SystemExit(f"valid holiday input changed: {args} {result} {fetch_calls}")
    finally:
        hc._fetch_next = original_fetch  # type: ignore[assignment]


def test_lists_upcoming() -> None:
    hc._fetch_next = lambda country: _SAMPLE  # type: ignore
    out = _tools()["next_holidays"].handler({})
    if not out.ok or "Constitution Day" not in out.output or "제헌절" not in out.output:
        raise SystemExit(f"holidays output wrong: {out.output}")
    if out.metadata.get("country") != hc.DEFAULT_COUNTRY or out.metadata.get("source") != "text_or_default":
        raise SystemExit(f"holidays should preserve country metadata: {out.metadata}")
    handoff = _assert_holidays_handoff(
        out.metadata,
        "holiday success",
        status="ok",
        calls_external_service=True,
        count=len(_SAMPLE),
    )
    if handoff.get("preview_row_count") != 2 or handoff.get("preview_rows", [{}])[0].get("name") != "Constitution Day":
        raise SystemExit(f"holiday success should include preview rows: {handoff}")


def test_country_from_text() -> None:
    seen = {}

    def fake_fetch(country):
        seen["country"] = country
        return _SAMPLE

    hc._fetch_next = fake_fetch  # type: ignore
    out = _tools()["next_holidays"].handler({"text": "next holidays in the US"})
    if seen.get("country") != "US":
        raise SystemExit(f"country not resolved from text: {seen}")
    _assert_holidays_handoff(out.metadata, "country from text", status="ok", calls_external_service=True, count=len(_SAMPLE))


def test_common_country_names_from_text() -> None:
    cases = {
        "next holidays in France": "FR",
        "holidays in Canada": "CA",
        "is today a holiday in Germany": "DE",
        "next Japanese holidays": "JP",
    }
    for text, expected_country in cases.items():
        seen = {}

        def fake_fetch(country):
            seen["country"] = country
            return _SAMPLE

        hc._fetch_next = fake_fetch  # type: ignore
        out = _tools()["next_holidays"].handler({"text": text})
        if not out.ok:
            raise SystemExit(f"{text!r} should resolve successfully: {out.output} {out.metadata}")
        if seen.get("country") != expected_country:
            raise SystemExit(f"{text!r} resolved wrong country: {seen}")
        if out.metadata.get("country") != expected_country or out.metadata.get("source") != "text":
            raise SystemExit(f"{text!r} should preserve text country metadata: {out.metadata}")
        _assert_holidays_handoff(
            out.metadata,
            f"country name from text {text!r}",
            status="ok",
            calls_external_service=True,
            count=len(_SAMPLE),
        )


def test_unknown_explicit_country_never_fetches() -> None:
    calls = []

    def fail_fetch(country):
        calls.append(country)
        raise AssertionError("unknown explicit country should not fetch")

    hc._fetch_next = fail_fetch  # type: ignore
    out = _tools()["next_holidays"].handler({"text": "next holidays in Atlantis"})
    if out.ok or "don't know that holiday country" not in out.output:
        raise SystemExit(f"unknown explicit country should be rejected locally: {out.output}")
    if out.metadata.get("reason") != "unknown_country":
        raise SystemExit(f"unknown explicit country should include reason metadata: {out.metadata}")
    if out.metadata.get("raw_country") != "atlantis" or out.metadata.get("source") != "text":
        raise SystemExit(f"unknown explicit country should preserve bounded raw country: {out.metadata}")
    if out.metadata.get("calls_external_service") is not False:
        raise SystemExit(f"unknown explicit country should not claim external calls: {out.metadata}")
    _assert_holidays_handoff(
        out.metadata,
        "unknown explicit country",
        status="refused",
        reason="unknown_country",
        calls_external_service=False,
    )
    if calls:
        raise SystemExit(f"unknown explicit country unexpectedly fetched: {calls}")


def test_invalid_country_never_fetches() -> None:
    calls = []

    def fail_fetch(country):
        calls.append(country)
        raise AssertionError("invalid country should not fetch")

    hc._fetch_next = fail_fetch  # type: ignore
    for country in [
        "United States!",
        "USA1",
        "KOR",
        "/\x55sers/example/private/holiday-country",
        "/var/folders/zc/holiday-country",
        "/tmp/holiday-country",
    ]:
        out = _tools()["next_holidays"].handler({"country": country})
        if out.ok or "two-letter country code" not in out.output:
            raise SystemExit(f"invalid country should be rejected locally: {country!r} -> {out.output}")
        if out.metadata.get("reason") != "invalid_country":
            raise SystemExit(f"invalid country should include reason metadata: {out.metadata}")
        if out.metadata.get("calls_external_service") is not False:
            raise SystemExit(f"invalid country should not claim external calls: {out.metadata}")
        _assert_holidays_handoff(
            out.metadata,
            f"invalid country {country!r}",
            status="refused",
            reason="invalid_country",
            calls_external_service=False,
        )
        raw_country = str(out.metadata.get("raw_country"))
        if any(fragment in raw_country for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
            raise SystemExit(f"invalid country leaked local path metadata: {out.metadata}")
    if calls:
        raise SystemExit(f"invalid country unexpectedly fetched: {calls}")


def test_invalid_env_default_country_falls_back() -> None:
    code = (
        "from jarvis_v2.tools import holidays_connector as hc; "
        "assert hc.DEFAULT_COUNTRY == hc.FALLBACK_COUNTRY"
    )
    for raw_country in ("United States!", "/var/folders/zc/holiday-country", "/tmp/holiday-country"):
        env = os.environ.copy()
        env["JARVIS_HOLIDAY_COUNTRY"] = raw_country
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise SystemExit(f"invalid holiday env default should fall back cleanly: {result.stderr or result.stdout}")


def test_env_default_country_accepts_alias() -> None:
    env = os.environ.copy()
    env["JARVIS_HOLIDAY_COUNTRY"] = "japan"
    code = "from jarvis_v2.tools import holidays_connector as hc; assert hc.DEFAULT_COUNTRY == 'JP'"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise SystemExit(f"holiday env default should accept aliases: {result.stderr or result.stdout}")


def test_handles_error() -> None:
    def boom(country):
        raise RuntimeError("offline")

    hc._fetch_next = boom  # type: ignore
    out = _tools()["next_holidays"].handler({})
    if out.ok or not out.output.strip():
        raise SystemExit(f"error not handled: {out.output}")
    if "offline" in out.output or out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"holiday error should be friendly with bounded exception metadata: {out.output} {out.metadata}")
    _assert_holidays_recovery_message(out.output, "holidays generic fetch error")
    _assert_holidays_handoff(
        out.metadata,
        "holiday fetch error",
        status="unavailable",
        reason="fetch_error",
        calls_external_service=True,
        retry_safe=True,
    )

    def server_error(country):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    hc._fetch_next = server_error  # type: ignore
    server_out = _tools()["next_holidays"].handler({})
    if server_out.ok or server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"HTTP 500 should preserve bounded exception metadata: {server_out.output} {server_out.metadata}")
    _assert_holidays_recovery_message(server_out.output, "holidays HTTP 500")
    _assert_holidays_handoff(
        server_out.metadata,
        "holiday HTTP 500 fetch error",
        status="unavailable",
        reason="fetch_error",
        calls_external_service=True,
        retry_safe=True,
    )

    def timeout_error(country):
        raise TimeoutError("request timed out near /\x55sers/example/private")

    hc._fetch_next = timeout_error  # type: ignore
    timeout_out = _tools()["next_holidays"].handler({})
    if timeout_out.ok or timeout_out.metadata.get("exception_type") != "TimeoutError":
        raise SystemExit(f"timeout should preserve bounded exception metadata: {timeout_out.output} {timeout_out.metadata}")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"timeout recovery should preserve timeout cause: {timeout_out.output}")
    _assert_holidays_recovery_message(timeout_out.output, "holidays timeout")
    _assert_holidays_handoff(
        timeout_out.metadata,
        "holiday timeout fetch error",
        status="unavailable",
        reason="fetch_error",
        calls_external_service=True,
        retry_safe=True,
    )


def test_unknown_remote_country_is_clean() -> None:
    def not_found(country):
        raise HttpError(404, "HTTP Error 404: Not Found")

    hc._fetch_next = not_found  # type: ignore
    out = _tools()["next_holidays"].handler({"country": "ZZ"})
    if out.ok or "I couldn't find holidays for ZZ" not in out.output:
        raise SystemExit(f"remote 404 should be a clean not-found message: {out.output}")
    if "setup check" in out.output.lower() or "network access" in out.output.lower() or "http error" in out.output.lower():
        raise SystemExit(f"remote 404 should not add noisy generic setup guidance: {out.output}")
    _assert_holidays_handoff(
        out.metadata,
        "remote holiday 404",
        status="unavailable",
        reason="fetch_error",
        calls_external_service=True,
        retry_safe=True,
    )


def test_handles_empty_holiday_response() -> None:
    hc._fetch_next = lambda country: []  # type: ignore
    out = _tools()["next_holidays"].handler({"country": "US"})
    if not out.ok or "No upcoming holidays" not in out.output:
        raise SystemExit(f"empty holidays should be a clean success: {out.output}")
    _assert_holidays_handoff(
        out.metadata,
        "empty holidays",
        status="empty",
        reason="no_holidays",
        calls_external_service=True,
        retry_safe=True,
    )


def test_marks_today_holiday() -> None:
    today_item = {"date": date.today().isoformat(), "name": "Today Holiday", "localName": "Today Local"}
    hc._fetch_next = lambda country: [today_item]  # type: ignore
    out = _tools()["next_holidays"].handler({"country": "US"})
    if not out.ok or "Today is Today Holiday" not in out.output:
        raise SystemExit(f"today holiday should be called out: {out.output}")
    handoff = _assert_holidays_handoff(
        out.metadata,
        "today holiday",
        status="ok",
        calls_external_service=True,
        count=1,
    )
    if handoff.get("today_holiday") is not True or handoff.get("today_holiday_name") != "Today Holiday":
        raise SystemExit(f"today holiday handoff wrong: {handoff}")


def test_planner_routes_holidays() -> None:
    p = RuleBasedPlanner()
    for q in [
        "next holidays",
        "when is the next holiday",
        "upcoming holidays",
        "next holidays in France",
        "holidays in Canada",
        "is today a holiday in Germany",
        "next holidays in Atlantis",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["next_holidays"]:
            raise SystemExit(f"holidays route missed: {q!r}")


def main() -> None:
    test_is_local_safe()
    test_holidays_argument_contract_fences_malformed_input()
    test_lists_upcoming()
    test_country_from_text()
    test_common_country_names_from_text()
    test_unknown_explicit_country_never_fetches()
    test_invalid_country_never_fetches()
    test_invalid_env_default_country_falls_back()
    test_env_default_country_accepts_alias()
    test_handles_error()
    test_unknown_remote_country_is_clean()
    test_handles_empty_holiday_response()
    test_marks_today_holiday()
    test_planner_routes_holidays()
    print("Holidays connector smoke passed")


if __name__ == "__main__":
    main()
