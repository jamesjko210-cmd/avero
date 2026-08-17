"""Focused offline proof for canonical market failure guidance and redaction."""

from __future__ import annotations

from typing import Any

from jarvis_v2.agent.failure_guidance import EXTERNAL_INFORMATION_RECOVERY_ACTION
from jarvis_v2.config import load_config
from jarvis_v2.tools import markets_connector as mc
from jarvis_v2.tools._http import HttpError


PRIVATE_MARKERS = (
    "/users/",
    "/private/",
    "/var/folders/",
    "/tmp/",
    "sk_" + "test_private-market-secret",
    "ghp" + "_private-market-secret",
)


def _tools():
    return {tool.name: tool for tool in mc.make_markets_tools(load_config())}


def _assert_read_only_boundaries(result: Any, label: str, *, external: bool = True) -> None:
    metadata = result.metadata
    if metadata.get("calls_external_service") is not external or metadata.get("calls_external_services") is not external:
        raise SystemExit(f"{label} has wrong external-read provenance: {metadata}")
    for key in (
        "calls_model",
        "executes_tools",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
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
    ):
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False: {metadata}")


def _assert_private_safe(result: Any, label: str) -> None:
    public = f"{result.output}\n{result.metadata}".lower()
    for marker in PRIVATE_MARKERS:
        if marker in public:
            raise SystemExit(f"{label} leaked private-looking detail {marker!r}: {public}")


def _assert_guidance(
    result: Any,
    label: str,
    *,
    action: str,
    commands: list[str],
    external: bool,
) -> None:
    expected = {
        "version": 1,
        "action": action,
        "commands": commands,
    }
    metadata = result.metadata
    if action not in result.output or metadata.get("recovery_guidance") != expected:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result}")
    for key, value in {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }.items():
        if metadata.get(key) != value:
            raise SystemExit(f"{label} recovery truth drifted at {key}: {metadata}")
    if commands:
        if metadata.get("next_command") != commands[0] or metadata.get("recovery_commands") != commands:
            raise SystemExit(f"{label} command guidance drifted: {metadata}")
    _assert_read_only_boundaries(result, label, external=external)
    _assert_private_safe(result, label)


def _assert_local_input(result: Any, label: str) -> None:
    _assert_guidance(
        result,
        label,
        action=mc.MARKET_INPUT_RECOVERY_ACTION,
        commands=[],
        external=False,
    )


def _assert_external(result: Any, label: str) -> None:
    _assert_guidance(
        result,
        label,
        action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
        commands=["setup check"],
        external=True,
    )


def test_local_inputs_are_rejected_before_network_with_private_safe_guidance() -> None:
    calls: list[tuple[str, str]] = []

    def crypto_fetch(coin_id: str) -> dict[str, Any]:
        calls.append(("crypto", coin_id))
        return {coin_id: {"usd": 100.0, "usd_24h_change": 1.0}}

    def stock_fetch(symbol: str) -> dict[str, Any]:
        calls.append(("stock", symbol))
        return {"price": 100.0, "prev": 99.0, "symbol": symbol}

    mc._fetch_crypto = crypto_fetch  # type: ignore[assignment]
    mc._fetch_stock = stock_fetch  # type: ignore[assignment]
    tools = _tools()
    cases = (
        (
            "get_crypto_price",
            {"coin": "sk_" + "test_private-market-secret"},
            "unsupported secret coin",
        ),
        ("get_crypto_price", {"coin": "/\x55sers/example/private/coin"}, "path coin"),
        ("get_stock_price", {"text": "stock price"}, "missing ticker"),
        ("get_stock_price", {"symbol": "ghp" + "_private-market-secret"}, "secret ticker"),
        ("get_stock_price", {"symbol": "/private/tmp/market-symbol"}, "path ticker"),
    )
    for tool_name, args, label in cases:
        result = tools[tool_name].handler(args)
        if result.ok:
            raise SystemExit(f"{label} should be rejected locally: {result}")
        _assert_local_input(result, label)
    if calls:
        raise SystemExit(f"invalid market inputs reached network seams: {calls}")


def test_missing_crypto_input_keeps_documented_bitcoin_default() -> None:
    seen: list[str] = []

    def crypto_fetch(coin_id: str) -> dict[str, Any]:
        seen.append(coin_id)
        return {coin_id: {"usd": 65000.0, "usd_24h_change": 1.0}}

    mc._fetch_crypto = crypto_fetch  # type: ignore[assignment]
    result = _tools()["get_crypto_price"].handler({})
    if not result.ok or seen != ["bitcoin"] or result.metadata.get("coin") != "bitcoin":
        raise SystemExit(f"empty crypto input should retain the Bitcoin default: {result} / {seen}")
    if "recovery_guidance" in result.metadata:
        raise SystemExit(f"successful default lookup should not claim failure guidance: {result.metadata}")
    _assert_read_only_boundaries(result, "default Bitcoin lookup")


def test_crypto_external_failures_are_canonical_and_offline() -> None:
    tools = _tools()
    failures = (
        (lambda _coin_id: {}, "no-result crypto response"),
        (lambda coin_id: {coin_id: {"usd": "bad", "usd_24h_change": "bad"}}, "malformed crypto response"),
        (lambda _coin_id: (_ for _ in ()).throw(HttpError(503)), "crypto HTTP failure"),
        (lambda _coin_id: (_ for _ in ()).throw(RuntimeError("API unavailable")), "crypto API failure"),
        (lambda _coin_id: (_ for _ in ()).throw(ValueError("parse failure")), "crypto parse failure"),
    )
    for fetch, label in failures:
        mc._fetch_crypto = fetch  # type: ignore[assignment]
        result = tools["get_crypto_price"].handler({"coin": "btc"})
        if result.ok:
            raise SystemExit(f"{label} should refuse the lookup: {result}")
        _assert_external(result, label)


def test_stock_external_failures_are_canonical_and_offline() -> None:
    tools = _tools()
    failures = (
        (lambda symbol: {"price": None, "prev": None, "symbol": symbol}, "no-result stock response"),
        (lambda _symbol: (_ for _ in ()).throw(HttpError(404)), "stock not found"),
        (lambda _symbol: (_ for _ in ()).throw(HttpError(503)), "stock HTTP failure"),
        (lambda _symbol: (_ for _ in ()).throw(RuntimeError("API unavailable")), "stock API failure"),
        (lambda _symbol: (_ for _ in ()).throw(ValueError("parse failure")), "stock parse failure"),
    )
    for fetch, label in failures:
        mc._fetch_stock = fetch  # type: ignore[assignment]
        result = tools["get_stock_price"].handler({"symbol": "AAPL"})
        if result.ok:
            raise SystemExit(f"{label} should refuse the lookup: {result}")
        _assert_external(result, label)


def test_partial_overview_exposes_canonical_recovery_without_side_effects() -> None:
    mc._fetch_crypto_many = lambda _ids: {}  # type: ignore[assignment]
    mc._fetch_stock = lambda symbol: {"price": "bad", "prev": None, "symbol": symbol}  # type: ignore[assignment]
    result = _tools()["get_markets_overview"].handler({"text": "how are the markets"})
    if not result.ok or result.metadata.get("partial") is not True:
        raise SystemExit(f"no-result overview should remain a partial read-only snapshot: {result}")
    if result.metadata.get("markets_overview_handoff", {}).get("error_count") != 4:
        raise SystemExit(f"no-result overview should identify all unavailable rows: {result.metadata}")
    _assert_external(result, "partial no-result overview")


def test_successes_preserve_behavior_without_failure_claims_or_secret_leaks() -> None:
    mc._fetch_crypto = lambda coin_id: {coin_id: {"usd": 65000.0, "usd_24h_change": -1.5}}  # type: ignore[assignment]
    crypto = _tools()["get_crypto_price"].handler({"symbol": "btc"})
    if not crypto.ok or "Bitcoin: $65,000.00" not in crypto.output:
        raise SystemExit(f"crypto success drifted: {crypto}")

    mc._fetch_stock = lambda symbol: {  # type: ignore[assignment]
        "price": 200.0,
        "prev": 198.0,
        "symbol": "sk_" + "test_private-market-secret",
    }
    stock = _tools()["get_stock_price"].handler({"ticker": "AAPL"})
    if not stock.ok or "AAPL: $200.00" not in stock.output:
        raise SystemExit(f"stock success/fallback label drifted: {stock}")

    mc._fetch_crypto_many = lambda ids: {  # type: ignore[assignment]
        coin_id: {"usd": 100.0, "usd_24h_change": 1.0} for coin_id in ids
    }
    mc._fetch_stock = lambda symbol: {"price": 200.0, "prev": 198.0, "symbol": symbol}  # type: ignore[assignment]
    overview = _tools()["get_markets_overview"].handler({"text": "how are the markets"})
    for result, label in ((crypto, "crypto success"), (stock, "stock success"), (overview, "overview success")):
        if not result.ok or "recovery_guidance" in result.metadata:
            raise SystemExit(f"{label} should not carry failure guidance: {result}")
        _assert_read_only_boundaries(result, label)
        _assert_private_safe(result, label)


def main() -> None:
    originals = (mc._fetch_crypto, mc._fetch_crypto_many, mc._fetch_stock)
    try:
        test_local_inputs_are_rejected_before_network_with_private_safe_guidance()
        test_missing_crypto_input_keeps_documented_bitcoin_default()
        test_crypto_external_failures_are_canonical_and_offline()
        test_stock_external_failures_are_canonical_and_offline()
        test_partial_overview_exposes_canonical_recovery_without_side_effects()
        test_successes_preserve_behavior_without_failure_claims_or_secret_leaks()
    finally:
        mc._fetch_crypto, mc._fetch_crypto_many, mc._fetch_stock = originals
    print("PASS: markets failures have canonical private-safe read-only recovery guidance")


if __name__ == "__main__":
    main()
