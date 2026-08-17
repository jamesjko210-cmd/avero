"""Focused offline proof for currency failure guidance and read-only boundaries."""

from __future__ import annotations

import urllib.parse
from unittest.mock import patch

from jarvis_v2.agent.failure_guidance import FAILURE_GUIDANCE_VERSION
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.config import load_config
from jarvis_v2.tools import currency_connector as cc
from jarvis_v2.tools._http import HttpError


PRIVATE_VALUES = (
    "/\x55sers/example/private/currency.txt",
    "/private/tmp/currency.txt",
    "/var/folders/zz/private-currency.txt",
    "/tmp/private-currency.txt",
    "sk_" + "live_currencyPrivateToken123456",
    "123456:telegramCurrencyPrivateToken1234567890",
)


def _tool():
    return {tool.name: tool for tool in cc.make_currency_tools(load_config())}["convert_currency"]


def _assert_canonical_failure(result: ToolResult, label: str, *, external: bool) -> None:
    if result.ok:
        raise SystemExit(f"{label} should fail: {result}")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("version") != FAILURE_GUIDANCE_VERSION:
        raise SystemExit(f"{label} lacks canonical failure guidance: {result.metadata}")
    action = guidance.get("action")
    if not isinstance(action, str) or action not in result.output:
        raise SystemExit(f"{label} guidance is not user-visible: {result}")
    for key, expected in {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
    }.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} has wrong {key}: {result.metadata}")
    if result.metadata.get("calls_external_service") is not external:
        raise SystemExit(f"{label} has wrong external-service truth: {result.metadata}")
    if result.metadata.get("calls_external_services") is not external:
        raise SystemExit(f"{label} has wrong plural external-service truth: {result.metadata}")
    handoff = result.metadata.get("currency_handoff")
    if not isinstance(handoff, dict) or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} lacks no-state-change handoff truth: {result.metadata}")
    if handoff.get("boundaries", {}).get("calls_external_service") is not external:
        raise SystemExit(f"{label} handoff has wrong external-service truth: {handoff}")
    surface = result.output + repr(result.metadata)
    for private in PRIVATE_VALUES:
        if private in surface:
            raise SystemExit(f"{label} leaked private input: {surface}")


def test_local_validation_failures_are_canonical_and_do_not_fetch() -> None:
    cases = (
        ({}, "missing_conversion"),
        ({"text": "convert usd"}, "missing_conversion"),
        ({"amount": 10, "from": "NOPE", "to": "KRW"}, "missing_currency"),
        ({"amount": "not-a-number", "from": "USD", "to": "KRW"}, "bad_amount"),
        ({"amount": "nan", "from": "USD", "to": "KRW"}, "non_finite_amount"),
        ({"amount": "inf", "from": "USD", "to": "KRW"}, "non_finite_amount"),
        ({"text": f"convert {PRIVATE_VALUES[0]} 10 usd to krw"}, "local_path_request"),
    )
    with patch.object(cc, "_fetch", side_effect=AssertionError("validation reached network seam")):
        for args, reason in cases:
            result = _tool().handler(args)
            _assert_canonical_failure(result, f"local validation {reason}", external=False)
            if result.metadata.get("reason") != reason:
                raise SystemExit(f"local validation lost its reason: {result.metadata}")
            if result.metadata.get("exception_type"):
                raise SystemExit(f"local validation invented an exception: {result.metadata}")


def test_private_values_are_redacted_from_local_failure_metadata() -> None:
    with patch.object(cc, "_fetch", side_effect=AssertionError("private input reached network seam")):
        for private in PRIVATE_VALUES:
            direct_code = _tool().handler({"amount": 10, "from": private, "to": "KRW"})
            _assert_canonical_failure(direct_code, "private currency code", external=False)
            direct_amount = _tool().handler({"amount": private, "from": "USD", "to": "KRW"})
            _assert_canonical_failure(direct_amount, "private currency amount", external=False)
            text = _tool().handler({"text": private})
            _assert_canonical_failure(text, "private currency text", external=False)
            surface = repr((direct_code.metadata, direct_amount.metadata, text.metadata))
            expected_marker = "<local-path>" if private.startswith("/") else "<secret>"
            if expected_marker not in surface:
                raise SystemExit(f"private currency input lost its redaction marker: {surface}")


def test_external_api_parse_and_rate_failures_are_canonical() -> None:
    failures = (
        (TimeoutError(f"timeout near {PRIVATE_VALUES[0]}"), "TimeoutError"),
        (HttpError(503, f"private backend {PRIVATE_VALUES[4]}"), "HttpError"),
        (ValueError(f"malformed response {PRIVATE_VALUES[5]}"), "ValueError"),
    )
    for exception, expected_type in failures:
        with patch.object(cc, "_fetch", side_effect=exception):
            result = _tool().handler({"amount": 12.5, "from": "USD", "to": "KRW"})
        _assert_canonical_failure(result, f"external {expected_type}", external=True)
        if result.metadata.get("reason") != "fetch_error":
            raise SystemExit(f"external failure lost fetch_error reason: {result.metadata}")
        if result.metadata.get("exception_type") != expected_type:
            raise SystemExit(f"external failure lost bounded exception class: {result.metadata}")
        if result.metadata.get("recovery_commands") != ["setup check"]:
            raise SystemExit(f"external failure lost setup recovery command: {result.metadata}")

    for response in ({"rates": {}}, {"rates": None}):
        with patch.object(cc, "_fetch", return_value=response):
            result = _tool().handler({"amount": 7, "from": "USD", "to": "KRW"})
        _assert_canonical_failure(result, "missing rate", external=True)
        if result.metadata.get("reason") != "missing_rate":
            raise SystemExit(f"missing rate lost its reason: {result.metadata}")
        if result.metadata.get("recovery_commands") != ["setup check"]:
            raise SystemExit(f"missing rate lost setup recovery command: {result.metadata}")

    with patch.object(cc, "_fetch", return_value=[]):
        malformed = _tool().handler({"amount": 9, "from": "USD", "to": "KRW"})
    _assert_canonical_failure(malformed, "malformed response shape", external=True)
    if malformed.metadata.get("exception_type") != "AttributeError":
        raise SystemExit(f"malformed response lost parse provenance: {malformed.metadata}")

    with patch.object(cc, "_fetch", return_value={"rates": {"KRW": "not-a-number"}}):
        invalid_rate = _tool().handler({"amount": 9, "from": "USD", "to": "KRW"})
    _assert_canonical_failure(invalid_rate, "invalid rate value", external=True)
    if invalid_rate.metadata.get("exception_type") != "ValueError":
        raise SystemExit(f"invalid rate lost parse provenance: {invalid_rate.metadata}")

    for rate in (float("nan"), float("inf"), float("-inf")):
        with patch.object(cc, "_fetch", return_value={"rates": {"KRW": rate}}):
            non_finite_rate = _tool().handler({"amount": 9, "from": "USD", "to": "KRW"})
        _assert_canonical_failure(non_finite_rate, "non-finite rate value", external=True)
        if non_finite_rate.metadata.get("exception_type") != "ValueError":
            raise SystemExit(f"non-finite rate lost bounded parse provenance: {non_finite_rate.metadata}")


def test_success_and_latest_rate_request_provenance_stay_stable() -> None:
    with patch.object(cc, "_fetch", return_value={"date": "2026-08-01", "rates": {"KRW": 13500.0}}):
        result = _tool().handler({"amount": 10, "from": "USD", "to": "KRW"})
    if not result.ok or result.output != "10.00 USD = 13,500.00 KRW":
        raise SystemExit(f"currency success behavior drifted: {result}")
    if result.metadata.get("amount") != 10.0 or result.metadata.get("result") != 13500.0:
        raise SystemExit(f"currency success lost bounded values: {result.metadata}")
    for key in ("writes_files", "writes_database", "writes_memory", "writes_notes", "executes_side_effect"):
        if result.metadata.get(key) is not False:
            raise SystemExit(f"currency success falsely claimed a side effect: {result.metadata}")

    seen: dict[str, object] = {}

    def fake_get_json(url: str, *, headers: dict[str, str]):
        seen.update(url=url, headers=headers)
        return {"date": "2026-08-01", "rates": {"KRW": 1350.0}}

    with patch.object(cc, "http_get_json", side_effect=fake_get_json):
        payload = cc._fetch(1.25, "USD", "KRW")
    parsed = urllib.parse.urlparse(str(seen.get("url")))
    query = urllib.parse.parse_qs(parsed.query)
    if parsed.scheme != "https" or parsed.netloc != "api.frankfurter.app" or parsed.path != "/latest":
        raise SystemExit(f"currency source drifted from Frankfurter latest rates: {seen}")
    if query != {"amount": ["1.25"], "from": ["USD"], "to": ["KRW"]}:
        raise SystemExit(f"currency request parameters drifted: {query}")
    if seen.get("headers") != {"User-Agent": "Mozilla/5.0"} or payload.get("date") != "2026-08-01":
        raise SystemExit(f"currency latest-rate provenance drifted: {seen} {payload}")


def main() -> None:
    test_local_validation_failures_are_canonical_and_do_not_fetch()
    test_private_values_are_redacted_from_local_failure_metadata()
    test_external_api_parse_and_rate_failures_are_canonical()
    test_success_and_latest_rate_request_provenance_stay_stable()
    print("Currency failure guidance smoke passed")


if __name__ == "__main__":
    main()
