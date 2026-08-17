"""Smoke tests for market price tools (mocked fetches, no network)."""

from __future__ import annotations

from jarvis_v2.agent.failure_guidance import EXTERNAL_INFORMATION_RECOVERY_ACTION
from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import markets_connector as mc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry

NO_AUTHORITY_FLAGS = [
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
]


def _tools():
    return {tool.name: tool for tool in mc.make_markets_tools(load_config())}


def assert_market_recovery_message(output: str, service: str, label: str) -> None:
    for expected in [
        f"{service} is having trouble",
        f"network access to {service}",
        "setup check",
        "retry",
    ]:
        if expected not in output:
            raise SystemExit(f"{label} missed actionable {service} recovery text {expected!r}: {output}")
    for forbidden in ["Error:", "HTTP Error", "Traceback", "offline", "/\x55sers/"]:
        if forbidden in output:
            raise SystemExit(f"{label} leaked raw backend detail {forbidden!r}: {output}")


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


def test_market_error_preserves_not_found_without_setup_noise() -> None:
    message = mc._market_error(
        HttpError(404, "HTTP Error 404: Not Found"),
        subject="a price for 'missing'",
        service="Yahoo Finance",
    )
    if message != "I couldn't find a price for 'missing'.":
        raise SystemExit(f"404 market errors should stay precise: {message}")
    for unexpected in ["setup check", "network access", "having trouble"]:
        if unexpected in message:
            raise SystemExit(f"404 market errors should not include generic recovery noise: {message}")


def test_markets_metadata_bool_is_exact() -> None:
    if mc._metadata_bool(True) is not True:
        raise SystemExit("markets exact metadata bool rejected True")
    if mc._metadata_bool(False) is not False:
        raise SystemExit("markets exact metadata bool rejected False")
    for value in ("true", "false", "yes", "no", 1, 0, [True], {"crypto_only": True}, None):
        if mc._metadata_bool(value) is not False:
            raise SystemExit(f"markets exact metadata bool accepted malformed truthy value: {value!r}")
    if mc._metadata_bool("false", default=True) is not True:
        raise SystemExit("markets exact metadata bool did not preserve explicit default")


def assert_market_price_handoff(
    metadata: dict,
    label: str,
    *,
    asset_type: str,
    ok: bool,
    external: bool = True,
) -> dict:
    specific_key = f"{asset_type}_price_handoff"
    if metadata.get(f"{asset_type}_price_handoff_ready") is not True or metadata.get("market_price_handoff_ready") is not True:
        raise SystemExit(f"{label} missed market price handoff readiness: {metadata}")
    handoff = metadata.get(specific_key)
    if not isinstance(handoff, dict) or metadata.get("market_price_handoff") != handoff:
        raise SystemExit(f"{label} missed market price handoff aliases: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} nested handoff should report handoff_ready=True: {handoff}")
    if handoff.get("asset_type") != asset_type:
        raise SystemExit(f"{label} handoff asset type wrong: {handoff}")
    expected_status = "ok" if ok else "unavailable"
    if handoff.get("status") != expected_status:
        raise SystemExit(f"{label} handoff status wrong: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} handoff should be operator-ready: {handoff}")
    for prefix in [f"{asset_type}_price", "market_price"]:
        if metadata.get(f"{prefix}_ready_for_operator") is not True:
            raise SystemExit(f"{label} missed flat operator-ready alias {prefix}: {metadata}")
        if metadata.get(f"{prefix}_state_changed") is not False or metadata.get(f"{prefix}_changed") != []:
            raise SystemExit(f"{label} missed flat unchanged-state aliases {prefix}: {metadata}")
        if metadata.get(f"{prefix}_content_in_handoff") is not False:
            raise SystemExit(f"{label} missed flat content-free alias {prefix}: {metadata}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} handoff should declare no state changes: {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False in metadata and handoff: {metadata}")
    for prefix in [f"{asset_type}_price", "market_price"]:
        for key in NO_AUTHORITY_FLAGS:
            alias = f"{prefix}_{key}"
            if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
                raise SystemExit(f"{label} should mirror false authority alias {alias}: {metadata}")
    if handoff.get("content_in_handoff") is not False or handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} handoff should declare content-free metadata: {handoff}")
    next_safe_commands = handoff.get("next_safe_commands")
    if (
        not isinstance(next_safe_commands, list)
        or next_safe_commands != [handoff.get("next_safe_command")]
        or handoff.get("next_safe_command_count") != len(next_safe_commands)
    ):
        raise SystemExit(f"{label} missed nested next-safe-command list/count: {handoff}")
    hidden_next_command_count = handoff.get("hidden_next_command_count", 0)
    if not isinstance(hidden_next_command_count, int) or hidden_next_command_count < 0:
        raise SystemExit(f"{label} hidden next-command count should be a non-negative integer: {handoff}")
    for prefix in [f"{asset_type}_price", "market_price"]:
        if metadata.get(f"{prefix}_next_safe_command") != handoff.get("next_safe_command"):
            raise SystemExit(f"{label} missed flat next-safe-command alias {prefix}: {metadata}")
        if metadata.get(f"{prefix}_next_safe_commands") != next_safe_commands:
            raise SystemExit(f"{label} missed flat next-safe-commands alias {prefix}: {metadata}")
        if metadata.get(f"{prefix}_next_safe_command_count") != len(next_safe_commands):
            raise SystemExit(f"{label} missed flat next-safe-command count alias {prefix}: {metadata}")
        if metadata.get(f"{prefix}_hidden_next_command_count", 0) != hidden_next_command_count:
            raise SystemExit(f"{label} missed hidden next-command count alias {prefix}: {metadata}")
    handoff_text = str(handoff).lower()
    metadata_text = str(metadata).lower()
    if "/users/" in handoff_text or "/private/" in handoff_text or "/var/folders/" in handoff_text or "/tmp/" in handoff_text:
        raise SystemExit(f"{label} handoff leaked a local path: {handoff}")
    if "/users/" in metadata_text or "/private/" in metadata_text or "/var/folders/" in metadata_text or "/tmp/" in metadata_text:
        raise SystemExit(f"{label} metadata leaked a local path: {metadata}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} handoff missed boundaries: {handoff}")
    if boundaries.get("calls_external_service") is not external or boundaries.get("calls_external_services") is not external:
        raise SystemExit(f"{label} handoff has wrong external-service read truth: {handoff}")
    for prefix in [f"{asset_type}_price", "market_price"]:
        if metadata.get(f"{prefix}_boundaries") != boundaries:
            raise SystemExit(f"{label} missed flat boundary alias {prefix}: {metadata}")
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
    if metadata.get("calls_external_service") is not external or metadata.get("calls_external_services") is not external:
        raise SystemExit(f"{label} metadata has wrong external-service read parity: {metadata}")
    return handoff


def assert_markets_overview_handoff(
    metadata: dict,
    label: str,
    *,
    instruments: list[str],
    crypto_only: bool,
    partial: bool,
    response_language: str = "en",
) -> dict:
    if metadata.get("markets_overview_handoff_ready") is not True:
        raise SystemExit(f"{label} missed markets overview handoff readiness: {metadata}")
    handoff = metadata.get("markets_overview_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed markets overview handoff: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} overview nested handoff should report handoff_ready=True: {handoff}")
    rows = handoff.get("rows")
    if not isinstance(rows, list):
        raise SystemExit(f"{label} overview handoff missed rows: {handoff}")
    if handoff.get("source") != "get_markets_overview" or handoff.get("crypto_only") != crypto_only:
        raise SystemExit(f"{label} overview handoff source/crypto_only wrong: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} overview handoff should be operator-ready: {handoff}")
    if metadata.get("markets_overview_ready_for_operator") is not True:
        raise SystemExit(f"{label} overview missed flat operator-ready alias: {metadata}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} overview handoff should declare no state changes: {handoff}")
    if metadata.get("markets_overview_state_changed") is not False or metadata.get("markets_overview_changed") != []:
        raise SystemExit(f"{label} overview missed flat unchanged-state aliases: {metadata}")
    for key in NO_AUTHORITY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} overview should keep {key}=False in metadata and handoff: {metadata}")
        alias = f"markets_overview_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} overview should mirror false authority alias {alias}: {metadata}")
    if handoff.get("instruments") != instruments or metadata.get("instruments") != instruments:
        raise SystemExit(f"{label} overview handoff instrument parity failed: {metadata}")
    if handoff.get("instrument_count") != len(instruments) or metadata.get("instrument_count") != len(instruments):
        raise SystemExit(f"{label} overview handoff count parity failed: {metadata}")
    if [row.get("instrument") for row in rows] != instruments:
        raise SystemExit(f"{label} overview row instruments diverged: {handoff}")
    if handoff.get("partial") != partial or metadata.get("partial") != partial:
        raise SystemExit(f"{label} overview partial parity failed: {metadata}")
    if (
        handoff.get("response_language") != response_language
        or metadata.get("response_language") != response_language
        or metadata.get("markets_overview_response_language") != response_language
    ):
        raise SystemExit(f"{label} overview response language parity failed: {metadata}")
    if handoff.get("not_financial_advice") is not True or handoff.get("content_in_handoff") is not False or handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} overview missed advice/content flags: {handoff}")
    if metadata.get("markets_overview_content_in_handoff") is not False:
        raise SystemExit(f"{label} overview missed flat content-free alias: {metadata}")
    next_safe_commands = handoff.get("next_safe_commands")
    if (
        not isinstance(next_safe_commands, list)
        or next_safe_commands != [handoff.get("next_safe_command")]
        or handoff.get("next_safe_command_count") != len(next_safe_commands)
    ):
        raise SystemExit(f"{label} overview missed nested next-safe-command list/count: {handoff}")
    if metadata.get("markets_overview_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} overview missed flat next-safe-command alias: {metadata}")
    if metadata.get("markets_overview_next_safe_commands") != next_safe_commands:
        raise SystemExit(f"{label} overview missed flat next-safe-commands alias: {metadata}")
    if metadata.get("markets_overview_next_safe_command_count") != len(next_safe_commands):
        raise SystemExit(f"{label} overview missed flat next-safe-command count alias: {metadata}")
    if "/\x55sers/" in str(handoff) or "/private/" in str(handoff) or "/var/folders/" in str(handoff) or "/tmp/" in str(handoff):
        raise SystemExit(f"{label} overview handoff leaked a local path: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict) or not boundaries.get("calls_external_service") or not boundaries.get("calls_external_services"):
        raise SystemExit(f"{label} overview handoff missed external boundary: {handoff}")
    if metadata.get("markets_overview_boundaries") != boundaries:
        raise SystemExit(f"{label} overview missed flat boundary alias: {metadata}")
    for key in ["calls_model", "executes_tools", *NO_AUTHORITY_FLAGS, "reads_personal_data", "reads_private_data", "executes_side_effect", "external_side_effect", "writes_files", "writes_database", "writes_memory", "writes_notes", "queues_approval", "requires_approval", "controls_computer"]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} overview should not perform {key}: {metadata}")
    if metadata.get("calls_external_service") is not True or metadata.get("calls_external_services") is not True:
        raise SystemExit(f"{label} overview metadata should declare external-service read parity: {metadata}")
    return handoff


def test_risk_levels() -> None:
    tools = _tools()
    if tools["get_crypto_price"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("get_crypto_price should be LOCAL_SAFE")
    if tools["get_stock_price"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("get_stock_price should be LOCAL_SAFE")
    if tools["get_markets_overview"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("get_markets_overview should be LOCAL_SAFE")


def test_market_argument_contracts_fence_malformed_input() -> None:
    crypto_calls: list[object] = []
    stock_calls: list[str] = []
    original_crypto = mc._fetch_crypto
    original_crypto_many = mc._fetch_crypto_many
    original_stock = mc._fetch_stock

    def mocked_crypto(coin_id: str) -> dict:
        crypto_calls.append(coin_id)
        return {coin_id: {"usd": 100.0, "usd_24h_change": 1.0}}

    def mocked_crypto_many(coin_ids: list[str]) -> dict:
        crypto_calls.append(tuple(coin_ids))
        return {coin_id: {"usd": 100.0, "usd_24h_change": 1.0} for coin_id in coin_ids}

    def mocked_stock(symbol: str) -> dict:
        stock_calls.append(symbol)
        return {"price": 200.0, "prev": 198.0, "symbol": symbol}

    try:
        mc._fetch_crypto = mocked_crypto  # type: ignore[assignment]
        mc._fetch_crypto_many = mocked_crypto_many  # type: ignore[assignment]
        mc._fetch_stock = mocked_stock  # type: ignore[assignment]
        tools = _tools()
        expected_shapes = {
            "get_crypto_price": tuple(
                (name, ("string",), False, None, None) for name in ("coin", "symbol", "text")
            ),
            "get_stock_price": tuple(
                (name, ("string",), False, None, None) for name in ("symbol", "ticker", "text")
            ),
            "get_markets_overview": (
                ("text", ("string",), False, None, None),
                ("language", ("string",), False, None, None),
                ("lang", ("string",), False, None, None),
                ("locale", ("string",), False, None, None),
                ("crypto_only", ("boolean",), False, None, None),
            ),
        }
        registry = ToolRegistry()
        for name, expected_shape in expected_shapes.items():
            tool = tools[name]
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
            if (
                contract is None
                or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
                or contract.allow_unknown is not False
                or actual_shape != expected_shape
            ):
                raise SystemExit(f"{name} should have an exact strict argument contract: {contract}")
            registry.register(tool)

        executor = Executor(registry, PermissionPolicy())
        private_sentinel = "private-market-contract-sentinel"
        malformed = {
            "get_crypto_price": (
                {"coin": {"value": private_sentinel}},
                {"symbol": [private_sentinel]},
                {"text": True},
                {"unknown": private_sentinel},
                ["coin", private_sentinel],
            ),
            "get_stock_price": (
                {"symbol": {"value": private_sentinel}},
                {"ticker": [private_sentinel]},
                {"text": False},
                {"unknown": private_sentinel},
                ["symbol", private_sentinel],
            ),
            "get_markets_overview": (
                {"text": {"value": private_sentinel}},
                {"language": [private_sentinel]},
                {"lang": True},
                {"locale": 123},
                {"crypto_only": "false"},
                {"crypto_only": 1},
                {"unknown": private_sentinel},
                ["text", private_sentinel],
            ),
        }
        for tool_name, cases in malformed.items():
            for args in cases:
                result = executor.execute(
                    PlannedAction(tool_name, args, "market contract smoke")  # type: ignore[arg-type]
                )
                if (
                    result.ok
                    or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or result.metadata.get("handler_invoked") is not False
                    or result.metadata.get("executed_handler") is not False
                ):
                    raise SystemExit(f"malformed {tool_name} arguments crossed the executor fence: {args!r} -> {result}")
                if private_sentinel in f"{result.output}\n{result.metadata}":
                    raise SystemExit(f"{tool_name} rejection leaked an argument value: {result}")
        if crypto_calls or stock_calls:
            raise SystemExit(f"malformed market arguments reached adapters: crypto={crypto_calls}, stock={stock_calls}")

        valid = (
            ("get_crypto_price", {"coin": "btc"}),
            ("get_crypto_price", {"symbol": "eth"}),
            ("get_crypto_price", {"text": "what is sol worth"}),
            ("get_stock_price", {"symbol": "AAPL"}),
            ("get_stock_price", {"ticker": "MSFT"}),
            ("get_stock_price", {"text": "stock price of NVDA"}),
            ("get_markets_overview", {"text": "how are the markets"}),
            ("get_markets_overview", {"crypto_only": True, "language": "korean"}),
            ("get_markets_overview", {"lang": "ko", "locale": "ko-KR"}),
        )
        for tool_name, args in valid:
            result = executor.execute(PlannedAction(tool_name, args, "valid market contract smoke"))
            if not result.ok:
                raise SystemExit(f"valid {tool_name} arguments were rejected: {args!r} -> {result}")
        if len(crypto_calls) != 6 or len(stock_calls) != 7:
            raise SystemExit(f"valid market arguments reached adapters incorrectly: crypto={crypto_calls}, stock={stock_calls}")
    finally:
        mc._fetch_crypto = original_crypto  # type: ignore[assignment]
        mc._fetch_crypto_many = original_crypto_many  # type: ignore[assignment]
        mc._fetch_stock = original_stock  # type: ignore[assignment]


def test_crypto_price_with_mocked_api() -> None:
    seen: dict[str, str] = {}

    def fake_fetch_crypto(coin_id: str) -> dict:
        seen["coin_id"] = coin_id
        return {"bitcoin": {"usd": 65000.5, "usd_24h_change": -2.25}}

    mc._fetch_crypto = fake_fetch_crypto  # type: ignore[method-assign]
    result = _tools()["get_crypto_price"].handler({"text": "what is BTC worth"})
    if not result.ok or "Bitcoin: $65,000.50" not in result.output or "2.2% 24h" not in result.output:
        raise SystemExit(f"crypto price output wrong: {result.output}")
    if seen != {"coin_id": "bitcoin"}:
        raise SystemExit(f"crypto fetch args wrong: {seen}")
    if result.metadata.get("coin") != "bitcoin" or result.metadata.get("external_side_effect"):
        raise SystemExit(f"crypto metadata unsafe or incomplete: {result.metadata}")
    handoff = assert_market_price_handoff(result.metadata, "crypto price success", asset_type="crypto", ok=True)
    if handoff.get("instrument") != "bitcoin" or handoff.get("price_usd") != 65000.5 or handoff.get("change_pct") != -2.25:
        raise SystemExit(f"crypto success handoff missed quote fields: {handoff}")


def test_crypto_price_handles_missing_data() -> None:
    def fake_fetch_crypto(coin_id: str) -> dict:
        return {}

    mc._fetch_crypto = fake_fetch_crypto  # type: ignore[method-assign]
    result = _tools()["get_crypto_price"].handler({"coin": "btc"})
    if result.ok or "Couldn't find a price" not in result.output:
        raise SystemExit(f"missing crypto data not handled: {result.output}")
    if result.metadata.get("coin") != "bitcoin" or result.metadata.get("requested_coin") != "btc":
        raise SystemExit(f"missing crypto metadata should preserve attempted coin: {result.metadata}")
    handoff = assert_market_price_handoff(result.metadata, "missing crypto price", asset_type="crypto", ok=False)
    if handoff.get("reason") != "missing_price" or handoff.get("price_available"):
        raise SystemExit(f"missing crypto handoff should preserve missing-price state: {handoff}")


def test_crypto_price_handles_malformed_price_data() -> None:
    def fake_fetch_crypto(coin_id: str) -> dict:
        return {coin_id: {"usd": "not-a-number", "usd_24h_change": "also-bad"}}

    mc._fetch_crypto = fake_fetch_crypto  # type: ignore[method-assign]
    result = _tools()["get_crypto_price"].handler({"coin": "btc"})
    if result.ok or "usable price" not in result.output:
        raise SystemExit(f"malformed crypto price should be a clean refusal: {result.output}")
    if result.metadata.get("reason") != "bad_price_data" or result.metadata.get("coin") != "bitcoin":
        raise SystemExit(f"malformed crypto price metadata wrong: {result.metadata}")


def test_crypto_price_fetch_error_is_friendly() -> None:
    def boom(coin_id: str) -> dict:
        raise RuntimeError("offline")

    mc._fetch_crypto = boom  # type: ignore[method-assign]
    result = _tools()["get_crypto_price"].handler({"coin": "btc"})
    if result.ok or "CoinGecko is having trouble" not in result.output:
        raise SystemExit(f"crypto fetch error should be a friendly refusal: {result.output}")
    assert_market_recovery_message(result.output, "CoinGecko", "crypto fetch error")
    assert_external_information_guidance(result, "crypto fetch error")
    if result.metadata.get("coin") != "bitcoin":
        raise SystemExit(f"crypto fetch error should preserve coin metadata: {result.metadata}")


def test_unknown_direct_crypto_never_fetches() -> None:
    calls = []

    def fail_fetch_crypto(coin_id: str) -> dict:
        calls.append(coin_id)
        raise AssertionError("unsupported coin should not fetch")

    mc._fetch_crypto = fail_fetch_crypto  # type: ignore[method-assign]
    result = _tools()["get_crypto_price"].handler({"coin": "unknowncoin"})
    if result.ok or "Unsupported coin" not in result.output:
        raise SystemExit(f"unsupported coin should be rejected locally: {result.output}")
    if result.metadata.get("reason") != "unsupported_coin" or result.metadata.get("requested_coin") != "unknowncoin":
        raise SystemExit(f"unsupported coin should preserve bounded metadata: {result.metadata}")
    assert_market_price_handoff(
        result.metadata,
        "unsupported crypto",
        asset_type="crypto",
        ok=False,
        external=False,
    )
    if calls:
        raise SystemExit(f"unsupported coin unexpectedly fetched: {calls}")

    path_result = _tools()["get_crypto_price"].handler({"coin": "/\x55sers/example/private/coin"})
    if path_result.ok or path_result.metadata.get("requested_coin") != "<local-path>":
        raise SystemExit(f"path-shaped unsupported coin should redact raw metadata: {path_result.metadata}")
    if calls:
        raise SystemExit(f"path-shaped unsupported coin unexpectedly fetched: {calls}")

    temp_path_result = _tools()["get_crypto_price"].handler({"coin": "/var/folders/zc/jarvis-market-coin"})
    if temp_path_result.ok or temp_path_result.metadata.get("requested_coin") != "<local-path>":
        raise SystemExit(f"temp-root unsupported coin should redact raw metadata: {temp_path_result.metadata}")
    tmp_path_result = _tools()["get_crypto_price"].handler({"coin": "/tmp/jarvis-market-coin"})
    if tmp_path_result.ok or tmp_path_result.metadata.get("requested_coin") != "<local-path>":
        raise SystemExit(f"tmp-root unsupported coin should redact raw metadata: {tmp_path_result.metadata}")
    if calls:
        raise SystemExit(f"temp-root unsupported coin unexpectedly fetched: {calls}")


def test_stock_price_with_mocked_quote() -> None:
    seen: dict[str, str] = {}

    def fake_fetch_stock(symbol: str) -> dict:
        seen["symbol"] = symbol
        return {"price": 200.25, "prev": 198.0, "symbol": "AAPL"}

    mc._fetch_stock = fake_fetch_stock  # type: ignore[method-assign]
    result = _tools()["get_stock_price"].handler({"text": "stock price of AAPL"})
    if not result.ok or "AAPL: $200.25" not in result.output or "▲" not in result.output:
        raise SystemExit(f"stock price output wrong: {result.output}")
    if seen != {"symbol": "AAPL"}:
        raise SystemExit(f"stock fetch args wrong: {seen}")
    if result.metadata.get("symbol") != "AAPL" or result.metadata.get("external_side_effect"):
        raise SystemExit(f"stock metadata unsafe or incomplete: {result.metadata}")
    handoff = assert_market_price_handoff(result.metadata, "stock price success", asset_type="stock", ok=True)
    if handoff.get("instrument") != "AAPL" or handoff.get("price_usd") != 200.25 or handoff.get("prev_close_usd") != 198.0:
        raise SystemExit(f"stock success handoff missed quote fields: {handoff}")


def test_market_price_handoff_suppresses_unsafe_next_command_instrument() -> None:
    metadata = mc._safe_metadata(
        **mc._price_handoff(
            tool="get_stock_price",
            asset_type="stock",
            requested="/\x55sers/example/private/AAPL",
            instrument="/USERS/OPERATOR/PRIVATE/AAPL",
            service="Yahoo Finance",
            ok=True,
            price_usd=200.25,
            prev_close_usd=198.0,
        )
    )
    handoff = assert_market_price_handoff(metadata, "unsafe stock instrument handoff", asset_type="stock", ok=True)
    if handoff.get("instrument") != "<local-path>":
        raise SystemExit(f"unsafe stock handoff should redact the instrument display: {handoff}")
    if handoff.get("next_safe_command") != "stock price" or handoff.get("next_safe_commands") != ["stock price"]:
        raise SystemExit(f"unsafe stock handoff should fall back to generic next command: {handoff}")
    if handoff.get("hidden_next_command_count") != 1 or metadata.get("market_price_hidden_next_command_count") != 1:
        raise SystemExit(f"unsafe stock handoff should count hidden command candidates: {metadata}")


def test_stock_price_redacts_upstream_path_symbol() -> None:
    def fake_fetch_stock(symbol: str) -> dict:
        return {"price": 200.25, "prev": 198.0, "symbol": "/\x55sers/example/private/AAPL"}

    mc._fetch_stock = fake_fetch_stock  # type: ignore[method-assign]
    result = _tools()["get_stock_price"].handler({"symbol": "AAPL"})
    combined = f"{result.output}\n{result.metadata}".lower()
    if not result.ok or "aapl: $200.25" not in result.output.lower():
        raise SystemExit(f"path-shaped upstream symbol should fall back to requested ticker label: {result.output}")
    if "/users/" in combined or "/private/" in combined:
        raise SystemExit(f"path-shaped upstream symbol leaked through output/metadata: {result.output} / {result.metadata}")
    handoff = assert_market_price_handoff(result.metadata, "path-shaped upstream stock symbol", asset_type="stock", ok=True)
    if handoff.get("instrument") != "AAPL" or handoff.get("next_safe_command") != "stock price of AAPL":
        raise SystemExit(f"path-shaped upstream symbol should fall back to safe ticker in handoff: {handoff}")


def test_stock_price_extracts_phone_quote_phrases() -> None:
    seen: list[str] = []

    def fake_fetch_stock(symbol: str) -> dict:
        seen.append(symbol)
        return {"price": 200.25, "prev": 198.0, "symbol": symbol}

    mc._fetch_stock = fake_fetch_stock  # type: ignore[method-assign]
    cases = {
        "quote AAPL": "AAPL",
        "price AAPL": "AAPL",
        "AAPL quote": "AAPL",
        "what is AAPL trading at": "AAPL",
        "how much is NVDA trading at": "NVDA",
        "what is Tesla stock at": "TSLA",
    }
    for text, expected_symbol in cases.items():
        result = _tools()["get_stock_price"].handler({"text": text})
        if not result.ok or result.metadata.get("symbol") != expected_symbol:
            raise SystemExit(f"stock phrase {text!r} should resolve {expected_symbol}: {result}")
    if seen != list(cases.values()):
        raise SystemExit(f"phone quote phrases fetched wrong symbols: {seen}")


def test_stock_price_validates_ticker() -> None:
    result = _tools()["get_stock_price"].handler({"text": "stock price"})
    if result.ok or "Which ticker" not in result.output:
        raise SystemExit(f"missing ticker should ask for ticker: {result.output}")
    if result.metadata.get("reason") != "missing_ticker" or result.metadata.get("symbol") is not None:
        raise SystemExit(f"missing ticker should include stable refusal metadata: {result.metadata}")
    if result.metadata.get("raw_symbol") != "stock price":
        raise SystemExit(f"missing ticker should preserve bounded raw prompt metadata: {result.metadata}")
    handoff = assert_market_price_handoff(
        result.metadata,
        "missing ticker",
        asset_type="stock",
        ok=False,
        external=False,
    )
    if handoff.get("reason") != "missing_ticker" or handoff.get("instrument") != "":
        raise SystemExit(f"missing ticker handoff should preserve refusal state: {handoff}")


def test_stock_price_refusals_preserve_symbol_metadata() -> None:
    mc._fetch_stock = lambda symbol: {"price": None, "prev": None, "symbol": symbol}  # type: ignore[method-assign]
    missing = _tools()["get_stock_price"].handler({"symbol": "MSFT"})
    if missing.ok or "No price" not in missing.output:
        raise SystemExit(f"missing stock price should be a refusal: {missing.output}")
    if missing.metadata.get("symbol") != "MSFT" or missing.metadata.get("reason") != "missing_price":
        raise SystemExit(f"missing stock price should preserve symbol metadata: {missing.metadata}")

    mc._fetch_stock = lambda symbol: {"price": "not-a-number", "prev": "also-bad", "symbol": symbol}  # type: ignore[method-assign]
    malformed = _tools()["get_stock_price"].handler({"symbol": "MSFT"})
    if malformed.ok or "No price" not in malformed.output:
        raise SystemExit(f"malformed stock price should be a clean refusal: {malformed.output}")
    if malformed.metadata.get("symbol") != "MSFT" or malformed.metadata.get("reason") != "missing_price":
        raise SystemExit(f"malformed stock price metadata wrong: {malformed.metadata}")

    def boom(symbol: str) -> dict:
        raise RuntimeError("offline")

    mc._fetch_stock = boom  # type: ignore[method-assign]
    error = _tools()["get_stock_price"].handler({"symbol": "MSFT"})
    if error.ok or "Yahoo Finance is having trouble" not in error.output:
        raise SystemExit(f"stock fetch error should be a friendly refusal: {error.output}")
    assert_market_recovery_message(error.output, "Yahoo Finance", "stock fetch error")
    assert_external_information_guidance(error, "stock fetch error")
    if error.metadata.get("symbol") != "MSFT":
        raise SystemExit(f"stock fetch error should preserve symbol metadata: {error.metadata}")


def test_invalid_direct_ticker_does_not_fetch() -> None:
    called = {"fetch": False}

    def fake_fetch_stock(symbol: str) -> str:
        called["fetch"] = True
        return "Symbol,Date,Time,Open,High,Low,Close,Volume\nBAD.US,2026-06-15,22:00,1,1,1,1,1"

    mc._fetch_stock = fake_fetch_stock  # type: ignore[method-assign]
    result = _tools()["get_stock_price"].handler({"symbol": "AAPL!!"})
    if result.ok or "Ticker must be 1-5 letters" not in result.output:
        raise SystemExit(f"invalid direct ticker should be rejected locally: {result.output}")
    if called["fetch"]:
        raise SystemExit("invalid direct ticker should not call fetch")
    if result.metadata.get("reason") != "bad_ticker" or result.metadata.get("raw_symbol") != "AAPL!!":
        raise SystemExit(f"invalid direct ticker should preserve bounded raw metadata: {result.metadata}")

    long_result = _tools()["get_stock_price"].handler({"symbol": "x" * 120})
    if long_result.ok or long_result.metadata.get("raw_symbol") != ("x" * 79 + "…"):
        raise SystemExit(f"long invalid ticker should be bounded in metadata: {long_result.metadata}")

    path_result = _tools()["get_stock_price"].handler({"symbol": "/private/tmp/jarvis-market-symbol"})
    if path_result.ok or path_result.metadata.get("raw_symbol") != "<local-path>":
        raise SystemExit(f"path-shaped invalid ticker should redact raw metadata: {path_result.metadata}")
    if called["fetch"]:
        raise SystemExit("path-shaped invalid ticker should not call fetch")

    temp_path_result = _tools()["get_stock_price"].handler({"ticker": "/var/folders/zc/jarvis-market-symbol"})
    if temp_path_result.ok or temp_path_result.metadata.get("raw_symbol") != "<local-path>":
        raise SystemExit(f"temp-root invalid ticker should redact raw metadata: {temp_path_result.metadata}")
    tmp_path_result = _tools()["get_stock_price"].handler({"text": "/tmp/jarvis-market-symbol"})
    if tmp_path_result.ok or tmp_path_result.metadata.get("raw_symbol") != "<local-path>":
        raise SystemExit(f"tmp-root missing ticker metadata should redact raw text: {tmp_path_result.metadata}")
    if called["fetch"]:
        raise SystemExit("temp-root invalid ticker should not call fetch")


def test_markets_overview_with_mocked_quotes() -> None:
    crypto_seen: dict[str, list[str]] = {}
    stock_seen: list[str] = []

    def fake_fetch_crypto_many(coin_ids: list[str]) -> dict:
        crypto_seen["coin_ids"] = coin_ids
        return {
            "bitcoin": {"usd": 65000.0, "usd_24h_change": 1.2},
            "ethereum": {"usd": 3500.5, "usd_24h_change": -0.4},
        }

    def fake_fetch_stock(symbol: str) -> dict:
        stock_seen.append(symbol)
        prices = {
            "AAPL": {"price": 200.0, "prev": 198.0, "symbol": "AAPL"},
            "NVDA": {"price": 120.0, "prev": 125.0, "symbol": "NVDA"},
        }
        return prices[symbol]

    mc._fetch_crypto_many = fake_fetch_crypto_many  # type: ignore[method-assign]
    mc._fetch_stock = fake_fetch_stock  # type: ignore[method-assign]
    result = _tools()["get_markets_overview"].handler({"text": "how are the markets"})
    expected_parts = ["Markets snapshot:", "BTC: $65,000.00", "ETH: $3,500.50", "AAPL: $200.00", "NVDA: $120.00", "not financial advice"]
    if not result.ok or any(part not in result.output for part in expected_parts):
        raise SystemExit(f"markets overview output wrong: {result.output}")
    if crypto_seen != {"coin_ids": ["bitcoin", "ethereum"]} or stock_seen != ["AAPL", "NVDA"]:
        raise SystemExit(f"markets overview fetch args wrong: crypto={crypto_seen}, stocks={stock_seen}")
    if result.metadata.get("instruments") != ["BTC", "ETH", "AAPL", "NVDA"] or result.metadata.get("partial"):
        raise SystemExit(f"markets overview metadata wrong: {result.metadata}")
    handoff = assert_markets_overview_handoff(
        result.metadata,
        "markets overview",
        instruments=["BTC", "ETH", "AAPL", "NVDA"],
        crypto_only=False,
        partial=False,
    )
    rows = handoff["rows"]
    if rows[0].get("price_usd") != 65000.0 or rows[2].get("prev_close_usd") != 198.0:
        raise SystemExit(f"markets overview handoff missed quote details: {handoff}")


def test_markets_overview_korean_response_with_mocked_quotes() -> None:
    def fake_fetch_crypto_many(coin_ids: list[str]) -> dict:
        if coin_ids != ["bitcoin", "ethereum"]:
            raise AssertionError(f"unexpected crypto ids: {coin_ids}")
        return {
            "bitcoin": {"usd": 65000.0, "usd_24h_change": 1.2},
            "ethereum": {"usd": 3500.5, "usd_24h_change": -0.4},
        }

    def fake_fetch_stock(symbol: str) -> dict:
        prices = {
            "AAPL": {"price": 200.0, "prev": 198.0, "symbol": "AAPL"},
            "NVDA": {"price": 120.0, "prev": 125.0, "symbol": "NVDA"},
        }
        return prices[symbol]

    mc._fetch_crypto_many = fake_fetch_crypto_many  # type: ignore[method-assign]
    mc._fetch_stock = fake_fetch_stock  # type: ignore[method-assign]
    result = _tools()["get_markets_overview"].handler({"text": "markets overview in Korean"})
    expected_parts = ["시장 스냅샷:", "BTC: $65,000.00", "AAPL: $200.00", "정보 제공용이며 투자 조언이 아니에요."]
    if not result.ok or any(part not in result.output for part in expected_parts):
        raise SystemExit(f"Korean markets overview output wrong: {result.output}")
    if "Markets snapshot:" in result.output or "Informational only, not financial advice." in result.output:
        raise SystemExit(f"Korean markets overview should not use English framing: {result.output}")
    if result.metadata.get("response_language") != "ko":
        raise SystemExit(f"Korean markets overview should mark response_language=ko: {result.metadata}")
    assert_markets_overview_handoff(
        result.metadata,
        "Korean markets overview",
        instruments=["BTC", "ETH", "AAPL", "NVDA"],
        crypto_only=False,
        partial=False,
        response_language="ko",
    )


def test_crypto_prices_overview_with_mocked_quotes() -> None:
    stock_called = {"value": False}

    def fake_fetch_crypto_many(coin_ids: list[str]) -> dict:
        if coin_ids != ["bitcoin", "ethereum", "solana"]:
            raise AssertionError(f"unexpected crypto ids: {coin_ids}")
        return {
            "bitcoin": {"usd": 65000.0, "usd_24h_change": 1.2},
            "ethereum": {"usd": 3500.5, "usd_24h_change": -0.4},
            "solana": {"usd": 150.0, "usd_24h_change": 3.0},
        }

    def fail_fetch_stock(symbol: str) -> dict:
        stock_called["value"] = True
        raise AssertionError("crypto-only overview should not fetch stocks")

    mc._fetch_crypto_many = fake_fetch_crypto_many  # type: ignore[method-assign]
    mc._fetch_stock = fail_fetch_stock  # type: ignore[method-assign]
    result = _tools()["get_markets_overview"].handler({"text": "crypto prices", "crypto_only": True})
    expected_parts = ["Crypto prices:", "BTC: $65,000.00", "ETH: $3,500.50", "SOL: $150.00"]
    if not result.ok or any(part not in result.output for part in expected_parts):
        raise SystemExit(f"crypto prices overview output wrong: {result.output}")
    if stock_called["value"]:
        raise SystemExit("crypto prices overview unexpectedly fetched stocks")
    if result.metadata.get("instruments") != ["BTC", "ETH", "SOL"] or not result.metadata.get("crypto_only"):
        raise SystemExit(f"crypto prices overview metadata wrong: {result.metadata}")
    assert_markets_overview_handoff(
        result.metadata,
        "crypto prices overview",
        instruments=["BTC", "ETH", "SOL"],
        crypto_only=True,
        partial=False,
    )


def test_markets_overview_rejects_malformed_crypto_only_arg() -> None:
    crypto_seen: dict[str, list[str]] = {}
    stock_seen: list[str] = []

    def fake_fetch_crypto_many(coin_ids: list[str]) -> dict:
        crypto_seen["coin_ids"] = coin_ids
        return {
            "bitcoin": {"usd": 65000.0, "usd_24h_change": 1.2},
            "ethereum": {"usd": 3500.5, "usd_24h_change": -0.4},
        }

    def fake_fetch_stock(symbol: str) -> dict:
        stock_seen.append(symbol)
        return {"price": 100.0, "prev": 99.0, "symbol": symbol}

    mc._fetch_crypto_many = fake_fetch_crypto_many  # type: ignore[method-assign]
    mc._fetch_stock = fake_fetch_stock  # type: ignore[method-assign]
    result = _tools()["get_markets_overview"].handler({"text": "how are the markets", "crypto_only": "false"})
    if not result.ok:
        raise SystemExit(f"malformed crypto_only overview should still succeed: {result.output}")
    if crypto_seen != {"coin_ids": ["bitcoin", "ethereum"]} or stock_seen != ["AAPL", "NVDA"]:
        raise SystemExit(f"malformed crypto_only should keep full market snapshot: crypto={crypto_seen}, stocks={stock_seen}")
    if result.metadata.get("crypto_only") is not False:
        raise SystemExit(f"malformed crypto_only should fail closed to False: {result.metadata}")
    assert_markets_overview_handoff(
        result.metadata,
        "malformed crypto_only overview",
        instruments=["BTC", "ETH", "AAPL", "NVDA"],
        crypto_only=False,
        partial=False,
    )


def test_markets_overview_partial_errors_are_friendly() -> None:
    def boom_crypto(coin_ids: list[str]) -> dict:
        raise RuntimeError("offline crypto")

    def boom_stock(symbol: str) -> dict:
        raise RuntimeError("offline stock")

    mc._fetch_crypto_many = boom_crypto  # type: ignore[method-assign]
    mc._fetch_stock = boom_stock  # type: ignore[method-assign]
    result = _tools()["get_markets_overview"].handler({"text": "how are the markets"})
    if not result.ok or not result.metadata.get("partial"):
        raise SystemExit(f"partial market overview should still return a snapshot: {result.output} / {result.metadata}")
    if "CoinGecko is having trouble" not in result.output or "Yahoo Finance is having trouble" not in result.output:
        raise SystemExit(f"partial market overview should summarize friendly service errors: {result.output}")
    assert_market_recovery_message(result.output, "CoinGecko", "partial crypto market overview")
    assert_market_recovery_message(result.output, "Yahoo Finance", "partial stock market overview")
    handoff = assert_markets_overview_handoff(
        result.metadata,
        "partial markets overview",
        instruments=["BTC", "ETH", "AAPL", "NVDA"],
        crypto_only=False,
        partial=True,
    )
    if handoff.get("error_count") != 3 or not all(row.get("status") == "unavailable" for row in handoff["rows"]):
        raise SystemExit(f"partial overview handoff should preserve unavailable rows and error count: {handoff}")


def test_planner_routes_market_prices() -> None:
    planner = RuleBasedPlanner()
    # Regression: "how's <anything> today" must NOT be a markets snapshot — the
    # old pattern swallowed "how's the weather today?" (live bug in the HUD).
    for phrase, wrong_tool in [
        ("how's the weather today?", "get_markets_overview"),
        ("how is the weather today", "get_markets_overview"),
        ("how's your day today", "get_markets_overview"),
    ]:
        plan = planner.plan(phrase)
        tools = [a.tool_name for a in (plan.actions or [])]
        if wrong_tool in tools:
            raise SystemExit(f"{phrase!r} must not route to markets: {tools}")
    weather_plan = planner.plan("how's the weather today?")
    if not weather_plan.actions or weather_plan.actions[0].tool_name != "get_weather":
        raise SystemExit(f"'how's the weather today?' must route to weather: {weather_plan}")
    cases = {
        "how are the markets": "get_markets_overview",
        "how's the market today": "get_markets_overview",
        "how are markets doing": "get_markets_overview",
        "show me the markets": "get_markets_overview",
        "markets": "get_markets_overview",
        "markets please": "get_markets_overview",
        "show markets": "get_markets_overview",
        "show the markets please": "get_markets_overview",
        "market": "get_markets_overview",
        "market overview": "get_markets_overview",
        "markets overview in Korean": "get_markets_overview",
        "market update": "get_markets_overview",
        "markets today": "get_markets_overview",
        "stock market please": "get_markets_overview",
        "show stock market": "get_markets_overview",
        "stocks please": "get_markets_overview",
        "show stocks please": "get_markets_overview",
        "stocks today": "get_markets_overview",
        "stock market update": "get_markets_overview",
        "how are stocks": "get_markets_overview",
        "crypto please": "get_markets_overview",
        "show crypto": "get_markets_overview",
        "crypto prices": "get_markets_overview",
        "crypto market": "get_markets_overview",
        "crypto market today": "get_markets_overview",
        "what is bitcoin worth": "get_crypto_price",
        "what's BTC at": "get_crypto_price",
        "SOL price": "get_crypto_price",
        "sol price": "get_crypto_price",
        "crypto price": "get_crypto_price",
        "stock price of AAPL": "get_stock_price",
        "AAPL stock": "get_stock_price",
        "AAPL price": "get_stock_price",
        "aapl price": "get_stock_price",
        "NVDA quote": "get_stock_price",
        "quote AAPL": "get_stock_price",
        "price AAPL": "get_stock_price",
        "what is AAPL trading at": "get_stock_price",
        "how much is NVDA trading at": "get_stock_price",
        "what is Tesla stock at": "get_stock_price",
        # Real gap found live 2026-07-10: the bare word "google" was an
        # unanchored web-search trigger in planner.py's lookup_m, so it fired
        # on "google" appearing ANYWHERE (not just as a leading verb) --
        # "compare apple and google stock" became a web search for just
        # "stock" (discarding "compare apple and google" entirely) instead of
        # reaching this stock-price route. Fixed by anchoring the bare
        # "google" trigger to the start of the command only.
        "compare apple and google stock": "get_stock_price",
    }
    for text, expected_tool in cases.items():
        actions = planner.plan(text).actions
        if [action.tool_name for action in actions] != [expected_tool]:
            raise SystemExit(f"planner missed market route for {text!r}: {actions}")
    for text in ["crypto please", "show crypto", "crypto prices", "crypto market", "crypto market today"]:
        action = planner.plan(text).actions[0]
        if action.tool_name != "get_markets_overview" or action.args.get("crypto_only") is not True:
            raise SystemExit(f"crypto market overview should request crypto-only snapshot for {text!r}: {action}")
    for text in ["markets", "markets please", "show markets", "market", "market overview", "stocks please"]:
        action = planner.plan(text).actions[0]
        if action.tool_name != "get_markets_overview" or action.args.get("crypto_only") is not False:
            raise SystemExit(f"general market overview should request full snapshot for {text!r}: {action}")
    korean_action = planner.plan("markets overview in Korean").actions[0]
    if korean_action.args.get("text") != "markets overview in Korean":
        raise SystemExit(f"Korean market route must preserve original text for localized output: {korean_action}")
    for text in ["house price", "gas price", "price of milk", "quote about courage", "what is milk trading at"]:
        actions = planner.plan(text).actions
        if [action.tool_name for action in actions] == ["get_stock_price"]:
            raise SystemExit(f"stock price route should not hijack {text!r}: {actions}")


def main() -> None:
    test_market_error_preserves_not_found_without_setup_noise()
    test_markets_metadata_bool_is_exact()
    test_risk_levels()
    test_market_argument_contracts_fence_malformed_input()
    test_crypto_price_with_mocked_api()
    test_crypto_price_handles_missing_data()
    test_crypto_price_handles_malformed_price_data()
    test_crypto_price_fetch_error_is_friendly()
    test_unknown_direct_crypto_never_fetches()
    test_stock_price_with_mocked_quote()
    test_market_price_handoff_suppresses_unsafe_next_command_instrument()
    test_stock_price_redacts_upstream_path_symbol()
    test_stock_price_extracts_phone_quote_phrases()
    test_stock_price_validates_ticker()
    test_stock_price_refusals_preserve_symbol_metadata()
    test_invalid_direct_ticker_does_not_fetch()
    test_markets_overview_with_mocked_quotes()
    test_markets_overview_korean_response_with_mocked_quotes()
    test_crypto_prices_overview_with_mocked_quotes()
    test_markets_overview_rejects_malformed_crypto_only_arg()
    test_markets_overview_partial_errors_are_friendly()
    test_planner_routes_market_prices()
    print("Markets connector smoke passed")


if __name__ == "__main__":
    main()
