"""Smoke tests for the currency connector (mocked fetch, no network)."""

from __future__ import annotations

from jarvis_v2.agent.failure_guidance import EXTERNAL_INFORMATION_RECOVERY_ACTION
from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import currency_connector as cc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry

NO_AUTHORITY_FLAGS = (
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
)


def _tools():
    return {t.name: t for t in cc.make_currency_tools(load_config())}


def assert_currency_recovery_message(output: str, label: str) -> None:
    lowered = output.lower()
    for fragment in [
        "network access to frankfurter",
        "setup check",
        "retry in a moment",
    ]:
        if fragment not in lowered:
            raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")
    for forbidden in ["http error", "offline", "traceback", "/users/", "/private/", "/var/folders/", "/tmp/"]:
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


def assert_currency_handoff(metadata: dict, label: str, *, status: str, reason: str = "", calls_external_service: bool) -> dict:
    if metadata.get("currency_handoff_ready") is not True:
        raise SystemExit(f"{label} missed currency_handoff_ready: {metadata}")
    handoff = metadata.get("currency_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed currency_handoff: {metadata}")
    if handoff.get("source") != "convert_currency" or handoff.get("status") != status:
        raise SystemExit(f"{label} has wrong source/status: {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} has wrong reason: {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should expose nested handoff_ready: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} handoff should be operator-ready: {handoff}")
    if metadata.get("currency_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} ready alias parity failed: {metadata} vs {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} handoff should declare no state changes: {handoff}")
    if metadata.get("currency_state_changed") != handoff.get("state_changed") or metadata.get("currency_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} state alias parity failed: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False in flat and nested metadata: {metadata}")
        alias = f"currency_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if handoff.get("content_in_handoff") is not False or handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} handoff should be content-free: {handoff}")
    if metadata.get("currency_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} content-in-handoff alias parity failed: {metadata} vs {handoff}")
    if metadata.get("currency_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} content-in-metadata alias parity failed: {metadata} vs {handoff}")
    for key in ("src", "dst", "exception_type"):
        if handoff.get(key) != (metadata.get(key) or ""):
            raise SystemExit(f"{label} handoff should mirror {key}: {handoff} vs {metadata}")
    for key in ("amount", "result"):
        if key in metadata and handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff should mirror {key}: {handoff} vs {metadata}")
    for key in ("raw_amount", "raw_from", "raw_to", "raw_text"):
        if key in metadata and handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff should mirror {key}: {handoff} vs {metadata}")
    if "/\x55sers/" in str(handoff) or "/private/" in str(handoff) or "/var/folders/" in str(handoff) or "/tmp/" in str(handoff):
        raise SystemExit(f"{label} handoff leaked local path: {handoff}")
    if not isinstance(handoff.get("next_safe_command"), str) or not handoff["next_safe_command"].startswith("convert"):
        raise SystemExit(f"{label} handoff missed next safe command: {handoff}")
    expected_commands = [handoff["next_safe_command"]]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} has wrong next safe command list: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} has wrong next safe command count: {handoff}")
    if metadata.get("currency_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("currency_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("currency_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} next-command-count alias parity failed: {metadata} vs {handoff}")
    boundaries = handoff.get("boundaries")
    expected = {
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
    if boundaries != expected:
        raise SystemExit(f"{label} has wrong boundaries: {boundaries}")
    if metadata.get("currency_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary alias parity failed: {metadata} vs {handoff}")
    if metadata.get("calls_external_service") is not calls_external_service or metadata.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} flat external-call flags should match handoff: {metadata}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["convert_currency"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("convert_currency should be LOCAL_SAFE")


def test_currency_argument_contract_fences_malformed_input() -> None:
    fetch_calls: list[tuple[float, str, str]] = []
    original_fetch = cc._fetch

    def mocked_fetch(amount: float, src: str, dst: str) -> dict:
        fetch_calls.append((amount, src, dst))
        return {"rates": {dst: amount * 2}}

    try:
        cc._fetch = mocked_fetch  # type: ignore[assignment]
        tool = _tools()["convert_currency"]
        contract = tool.argument_contract
        actual_shape = (
            tuple(
                (
                    field.name,
                    tuple(sorted(kind.value for kind in field.types)),
                    field.required,
                    field.minimum,
                    field.maximum,
                )
                for field in contract.fields
            )
            if contract is not None
            else ()
        )
        expected_shape = (
            ("text", ("string",), False, None, None),
            ("request", ("string",), False, None, None),
            ("from", ("string",), False, None, None),
            ("to", ("string",), False, None, None),
            ("amount", ("number", "string"), False, None, None),
        )
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown is not False
            or actual_shape != expected_shape
        ):
            raise SystemExit(f"convert_currency should have an exact strict argument contract: {contract}")

        registry = ToolRegistry()
        registry.register(tool)
        executor = Executor(registry, PermissionPolicy())
        private_sentinel = "private-currency-contract-sentinel"
        malformed_args: list[object] = [
            {"text": {"value": private_sentinel}},
            {"request": [private_sentinel]},
            {"from": True},
            {"to": 123},
            {"amount": False},
            {"amount": {"value": private_sentinel}},
            {"unknown": private_sentinel},
            ["text", private_sentinel],
        ]
        for args in malformed_args:
            result = executor.execute(
                PlannedAction("convert_currency", args, "currency contract smoke")  # type: ignore[arg-type]
            )
            if (
                result.ok
                or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
            ):
                raise SystemExit(f"malformed convert_currency arguments crossed the executor fence: {args!r} -> {result}")
            if private_sentinel in f"{result.output}\n{result.metadata}":
                raise SystemExit(f"convert_currency rejection leaked an argument value: {result}")
        if fetch_calls:
            raise SystemExit(f"malformed convert_currency arguments reached the network seam: {fetch_calls}")

        valid_args = (
            {"text": "convert 5 usd to krw"},
            {"request": "convert 6 eur to usd"},
            {"amount": 7, "from": "USD", "to": "KRW"},
            {"amount": "8.5", "from": "EUR", "to": "USD"},
        )
        for args in valid_args:
            result = executor.execute(PlannedAction("convert_currency", args, "valid currency contract smoke"))
            if not result.ok:
                raise SystemExit(f"valid convert_currency arguments were rejected: {args!r} -> {result}")
        if len(fetch_calls) != len(valid_args):
            raise SystemExit(f"valid convert_currency arguments did not each reach the mocked fetch seam: {fetch_calls}")
    finally:
        cc._fetch = original_fetch  # type: ignore[assignment]


def test_parse_aliases_and_amounts() -> None:
    cases = {
        "convert 100 usd to krw": (100.0, "USD", "KRW"),
        "100 dollars in won": (100.0, "USD", "KRW"),
        "how much is 50 euros in dollars": (50.0, "EUR", "USD"),
        "convert 1,000 yen to gbp": (1000.0, "JPY", "GBP"),
        # Real bug found live 2026-07-08/09: "how many X is/are/in Y Z" mentions
        # the TARGET currency first and the source (amount's own) currency
        # second -- the reverse of every other phrasing tested above, where
        # left-to-right token order already matches source-then-target. The
        # naive left-to-right scan alone got this backwards ("how many usd is
        # 100 eur" returned "100 USD = 87.69 EUR" instead of converting the 100
        # EUR into USD).
        "how many usd is 100 eur": (100.0, "EUR", "USD"),
        "how many dollars are 50 euros": (50.0, "EUR", "USD"),
        "how many usd in a euro": (1.0, "EUR", "USD"),
        "how many usd are in a euro": (1.0, "EUR", "USD"),
    }
    for text, expected in cases.items():
        if cc.parse_conversion(text) != expected:
            raise SystemExit(f"parse wrong for {text!r}: {cc.parse_conversion(text)}")
    if cc.parse_conversion("hello there") is not None:
        raise SystemExit("non-conversion text should not parse")


def test_converts_with_mocked_rate() -> None:
    seen = {}

    def fake_fetch(amount, src, dst):
        seen.update({"amount": amount, "src": src, "dst": dst})
        return {"rates": {dst: amount * 1350.0}}

    cc._fetch = fake_fetch  # type: ignore
    out = _tools()["convert_currency"].handler({"text": "convert 100 usd to krw"})
    if not out.ok or "100.00 USD" not in out.output or "135,000.00 KRW" not in out.output:
        raise SystemExit(f"conversion output wrong: {out.output}")
    if seen != {"amount": 100.0, "src": "USD", "dst": "KRW"}:
        raise SystemExit(f"fetch args wrong: {seen}")
    handoff = assert_currency_handoff(out.metadata, "currency success", status="ok", calls_external_service=True)
    if handoff.get("amount") != 100.0 or handoff.get("src") != "USD" or handoff.get("dst") != "KRW" or handoff.get("result") != 135000.0:
        raise SystemExit(f"currency success handoff missed conversion fields: {handoff}")


def test_direct_zero_amount_and_bad_currency_validation() -> None:
    seen = {}

    def fake_fetch(amount, src, dst):
        seen.update({"amount": amount, "src": src, "dst": dst})
        return {"rates": {dst: 0.0}}

    cc._fetch = fake_fetch  # type: ignore
    zero = _tools()["convert_currency"].handler({"amount": 0, "from": "USD", "to": "KRW"})
    if not zero.ok or "0.00 USD" not in zero.output:
        raise SystemExit(f"direct zero amount should convert instead of falling through: {zero.output}")
    if seen != {"amount": 0.0, "src": "USD", "dst": "KRW"}:
        raise SystemExit(f"direct zero amount fetch args wrong: {seen}")
    assert_currency_handoff(zero.metadata, "direct zero amount", status="ok", calls_external_service=True)

    called = {"fetch": False}

    def forbidden_fetch(amount, src, dst):
        called["fetch"] = True
        return {"rates": {dst: 1.0}}

    cc._fetch = forbidden_fetch  # type: ignore
    bad = _tools()["convert_currency"].handler({"amount": 100, "from": "BAD", "to": "KRW"})
    if bad.ok or "supported currencies" not in bad.output:
        raise SystemExit(f"bad direct currency should fail locally: {bad.output}")
    if called["fetch"]:
        raise SystemExit("bad direct currency should not call fetch")
    if bad.metadata.get("reason") != "missing_currency" or bad.metadata.get("raw_from") != "BAD" or bad.metadata.get("dst") != "KRW":
        raise SystemExit(f"bad direct currency should preserve bounded metadata: {bad.metadata}")
    assert_currency_handoff(bad.metadata, "bad direct currency", status="refused", reason="missing_currency", calls_external_service=False)
    long_bad = _tools()["convert_currency"].handler({"amount": 100, "from": "x" * 200, "to": ""})
    if long_bad.metadata.get("raw_from") != ("x" * 79 + "…"):
        raise SystemExit(f"bad direct currency should bound raw code metadata: {long_bad.metadata}")
    assert_currency_handoff(long_bad.metadata, "long bad direct currency", status="refused", reason="missing_currency", calls_external_service=False)
    path_bad = _tools()["convert_currency"].handler({"amount": 100, "from": "/\x55sers/example/private/currency-code", "to": "KRW"})
    if path_bad.ok or path_bad.metadata.get("raw_from") != "<local-path>":
        raise SystemExit(f"bad direct currency should redact path-shaped raw code metadata: {path_bad.metadata}")
    assert_currency_handoff(path_bad.metadata, "path bad direct currency", status="refused", reason="missing_currency", calls_external_service=False)
    temp_path_bad = _tools()["convert_currency"].handler({"amount": 100, "from": "/var/folders/zc/currency-code", "to": "KRW"})
    if temp_path_bad.ok or temp_path_bad.metadata.get("raw_from") != "<local-path>":
        raise SystemExit(f"bad direct currency should redact temp-root raw code metadata: {temp_path_bad.metadata}")
    tmp_path_bad = _tools()["convert_currency"].handler({"amount": 100, "from": "/tmp/currency-code", "to": "KRW"})
    if tmp_path_bad.ok or tmp_path_bad.metadata.get("raw_from") != "<local-path>":
        raise SystemExit(f"bad direct currency should redact tmp-root raw code metadata: {tmp_path_bad.metadata}")


def test_path_shaped_text_request_does_not_fetch() -> None:
    called = {"fetch": False}

    def forbidden_fetch(amount, src, dst):
        called["fetch"] = True
        return {"rates": {dst: 1.0}}

    cc._fetch = forbidden_fetch  # type: ignore
    for text in [
        "convert /\x55sers/example/private/currency-note 100 usd to krw",
        "convert /var/folders/zc/currency-note 100 usd to krw",
        "convert /tmp/currency-note 100 usd to krw",
    ]:
        out = _tools()["convert_currency"].handler({"text": text})
        if out.ok or out.metadata.get("reason") != "local_path_request" or out.metadata.get("raw_text") is None:
            raise SystemExit(f"path-shaped currency text should reject locally: {out.output} {out.metadata}")
        if "<local-path>" not in out.metadata.get("raw_text", ""):
            raise SystemExit(f"path-shaped currency text should preserve redacted raw metadata: {out.metadata}")
        if any(fragment in out.output or fragment in str(out.metadata) for fragment in ["/\x55sers/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"path-shaped currency text leaked raw local path: {out.output} {out.metadata}")
        assert_currency_handoff(out.metadata, "path-shaped currency text", status="refused", reason="local_path_request", calls_external_service=False)
    if called["fetch"]:
        raise SystemExit("path-shaped currency text should not call fetch")


def test_handles_error() -> None:
    def boom(amount, src, dst):
        raise RuntimeError("offline")

    cc._fetch = boom  # type: ignore
    out = _tools()["convert_currency"].handler({"text": "convert 5 gbp to usd"})
    if out.ok or not out.output.strip():
        raise SystemExit(f"error not handled: {out.output}")
    if out.metadata.get("amount") != 5.0 or out.metadata.get("src") != "GBP" or out.metadata.get("dst") != "USD":
        raise SystemExit(f"currency error metadata should preserve attempted conversion: {out.metadata}")
    assert_currency_recovery_message(out.output, "generic currency fetch error")
    assert_external_information_guidance(out, "generic currency fetch error")
    if out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"currency error should be friendly with bounded exception metadata: {out.output} {out.metadata}")
    assert_currency_handoff(out.metadata, "currency fetch error", status="unavailable", reason="fetch_error", calls_external_service=True)

    def server_error(amount, src, dst):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    cc._fetch = server_error  # type: ignore
    server_out = _tools()["convert_currency"].handler({"text": "convert 5 gbp to usd"})
    if server_out.ok:
        raise SystemExit(f"HTTP 500 should be a service-trouble message: {server_out.output}")
    assert_currency_recovery_message(server_out.output, "currency HTTP 500")
    if server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"currency HTTP error should keep bounded exception metadata: {server_out.metadata}")
    assert_currency_handoff(server_out.metadata, "currency HTTP 500", status="unavailable", reason="fetch_error", calls_external_service=True)

    def timeout_error(amount, src, dst):
        raise TimeoutError("request timed out near /\x55sers/example/private")

    cc._fetch = timeout_error  # type: ignore
    timeout_out = _tools()["convert_currency"].handler({"text": "convert 5 gbp to usd"})
    if timeout_out.ok:
        raise SystemExit(f"timeout should be a retryable service message: {timeout_out.output}")
    assert_currency_recovery_message(timeout_out.output, "currency timeout")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"timeout message should preserve the safe cause: {timeout_out.output}")
    assert_currency_handoff(timeout_out.metadata, "currency timeout", status="unavailable", reason="fetch_error", calls_external_service=True)


def test_missing_rate_preserves_metadata() -> None:
    def fake_fetch(amount, src, dst):
        return {"rates": {}}

    cc._fetch = fake_fetch  # type: ignore
    out = _tools()["convert_currency"].handler({"text": "convert 7 usd to krw"})
    if out.ok or "Couldn't get a USD->KRW rate" not in out.output:
        raise SystemExit(f"missing rate not handled: {out.output}")
    assert_currency_recovery_message(out.output, "missing currency rate")
    if out.metadata.get("amount") != 7.0 or out.metadata.get("src") != "USD" or out.metadata.get("dst") != "KRW":
        raise SystemExit(f"missing rate metadata should preserve attempted conversion: {out.metadata}")
    handoff = assert_currency_handoff(out.metadata, "missing currency rate", status="empty", reason="missing_rate", calls_external_service=True)
    if handoff.get("result_available") is not False:
        raise SystemExit(f"missing currency rate should mark result unavailable: {handoff}")


def test_invalid_direct_amount_does_not_fetch() -> None:
    called = {"fetch": False}

    def fake_fetch(amount, src, dst):
        called["fetch"] = True
        return {"rates": {dst: 1300.0}}

    cc._fetch = fake_fetch  # type: ignore
    out = _tools()["convert_currency"].handler({"amount": "not-a-number", "from": "USD", "to": "KRW"})
    if out.ok or "amount must be a number" not in out.output.lower():
        raise SystemExit(f"invalid amount should be a local validation error: {out.output}")
    if called["fetch"]:
        raise SystemExit("invalid direct amount should not call fetch")
    if out.metadata.get("raw_amount") != "not-a-number" or out.metadata.get("src") != "USD" or out.metadata.get("dst") != "KRW":
        raise SystemExit(f"invalid amount should preserve bounded raw amount and currency metadata: {out.metadata}")
    assert_currency_handoff(out.metadata, "invalid currency amount", status="refused", reason="bad_amount", calls_external_service=False)

    long_out = _tools()["convert_currency"].handler({"amount": "a" * 200, "from": "USD", "to": "KRW"})
    if long_out.ok or long_out.metadata.get("raw_amount") != ("a" * 79 + "…"):
        raise SystemExit(f"invalid amount should bound raw amount metadata: {long_out.metadata}")
    assert_currency_handoff(long_out.metadata, "long invalid currency amount", status="refused", reason="bad_amount", calls_external_service=False)
    path_out = _tools()["convert_currency"].handler({"amount": "/private/tmp/jarvis-currency-amount", "from": "USD", "to": "KRW"})
    if path_out.ok or path_out.metadata.get("raw_amount") != "<local-path>":
        raise SystemExit(f"invalid amount should redact path-shaped raw amount metadata: {path_out.metadata}")
    temp_path_out = _tools()["convert_currency"].handler({"amount": "/var/folders/zc/jarvis-currency-amount", "from": "USD", "to": "KRW"})
    if temp_path_out.ok or temp_path_out.metadata.get("raw_amount") != "<local-path>":
        raise SystemExit(f"invalid amount should redact temp-root raw amount metadata: {temp_path_out.metadata}")
    tmp_path_out = _tools()["convert_currency"].handler({"amount": "/tmp/jarvis-currency-amount", "from": "USD", "to": "KRW"})
    if tmp_path_out.ok or tmp_path_out.metadata.get("raw_amount") != "<local-path>":
        raise SystemExit(f"invalid amount should redact tmp-root raw amount metadata: {tmp_path_out.metadata}")


def test_non_finite_direct_amount_does_not_fetch() -> None:
    called = {"fetch": False}

    def fake_fetch(amount, src, dst):
        called["fetch"] = True
        return {"rates": {dst: 1300.0}}

    cc._fetch = fake_fetch  # type: ignore
    for amount in ["nan", "inf", "-inf"]:
        out = _tools()["convert_currency"].handler({"amount": amount, "from": "USD", "to": "KRW"})
        if out.ok or "amount must be finite" not in out.output.lower():
            raise SystemExit(f"non-finite amount should be a local validation error: {amount!r} -> {out.output}")
        if out.metadata.get("raw_amount") != amount or out.metadata.get("src") != "USD" or out.metadata.get("dst") != "KRW":
            raise SystemExit(f"non-finite amount should preserve bounded metadata: {out.metadata}")
        assert_currency_handoff(out.metadata, f"non-finite currency amount {amount}", status="refused", reason="non_finite_amount", calls_external_service=False)
    if called["fetch"]:
        raise SystemExit("non-finite direct amount should not call fetch")


def test_planner_routes_currency_not_units() -> None:
    p = RuleBasedPlanner()
    for q in [
        "convert 100 usd to krw",
        "100 dollars in won",
        "how much is 50 euros in dollars",
        "what is 100 euros in dollars",
        "what's 100 eur in usd",
        "how much is 100 eur in usd",
        "usd to krw",
        "100 usd krw",
        "currency 100 usd krw",
        "currency usd krw",
        "exchange rate usd krw",
        "100 dollars won",
        # Real bug found live 2026-07-08/09: these "how many X is/are/in Y Z"
        # phrasings were routed to convert_units (which doesn't recognize
        # currency codes as units, e.g. "Unknown unit: eur") instead of
        # convert_currency -- the two dedicated "how many" planner patterns
        # lacked the currency-code check that the general convert_match pattern
        # already had.
        "how many usd is 100 eur",
        "how many dollars are 50 euros",
        "how many usd in a euro",
        "how many usd are in a euro",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["convert_currency"]:
            raise SystemExit(f"currency route missed: {q!r}")
    # real unit conversions must not be hijacked
    if [a.tool_name for a in p.plan("convert 5 miles to km").actions] != ["convert_units"]:
        raise SystemExit("unit conversion wrongly routed to currency")
    if [a.tool_name for a in p.plan("how many ml are a quarter cup").actions] != ["convert_units"]:
        raise SystemExit("how-many unit conversion wrongly routed to currency")
    if [a.tool_name for a in p.plan("how many km is three miles").actions] != ["convert_units"]:
        raise SystemExit("how-many unit conversion (in-form) wrongly routed to currency")


def main() -> None:
    test_is_local_safe()
    test_currency_argument_contract_fences_malformed_input()
    test_parse_aliases_and_amounts()
    test_converts_with_mocked_rate()
    test_direct_zero_amount_and_bad_currency_validation()
    test_path_shaped_text_request_does_not_fetch()
    test_handles_error()
    test_missing_rate_preserves_metadata()
    test_invalid_direct_amount_does_not_fetch()
    test_non_finite_direct_amount_does_not_fetch()
    test_planner_routes_currency_not_units()
    print("Currency connector smoke passed")


if __name__ == "__main__":
    main()
