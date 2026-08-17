"""Live crypto & stock prices for Jarvis V2 (CoinGecko + Stooq, free, no keys)."""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    EXTERNAL_INFORMATION_RECOVERY_ACTION,
    declare_retryable_external_information_failure,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig


CRYPTO_IDS = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "cardano": "cardano", "ada": "cardano",
    "ripple": "ripple", "xrp": "ripple",
    "bnb": "binancecoin", "binance": "binancecoin",
}
STOCK_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "google": "GOOG",
    "alphabet": "GOOG",
    "meta": "META",
    "facebook": "META",
    "amazon": "AMZN",
}
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
SECRET_VALUE_RE = re.compile(
    r"(?:"
    r"\bsk_(?:live|test)_[A-Za-z0-9_-]+"
    r"|\bgh[pousr]_[A-Za-z0-9_-]+"
    r"|\bxox[baprs]-[A-Za-z0-9-]+"
    r"|\bAIza[A-Za-z0-9_-]{16,}"
    r"|\b\d{6,}:[A-Za-z0-9_-]{20,}"
    r")",
    re.IGNORECASE,
)
COMMAND_INSTRUMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,23}")
MARKET_INPUT_RECOVERY_ACTION = (
    "Provide a supported public market identifier, then retry the read-only lookup."
)


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


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _market_boundaries(*, calls_external_service: bool = True) -> dict[str, bool]:
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


def _price_handoff(
    *,
    tool: str,
    asset_type: str,
    requested: str | None,
    instrument: str | None,
    service: str,
    ok: bool,
    calls_external_service: bool = True,
    reason: str = "",
    price_usd: Any = None,
    change_pct: Any = None,
    prev_close_usd: Any = None,
) -> dict[str, Any]:
    price_num = _as_float(price_usd)
    change_num = _as_float(change_pct)
    prev_num = _as_float(prev_close_usd)
    command_instrument = _safe_command_instrument(instrument)
    hidden_next_command_count = 1 if instrument and not command_instrument else 0
    generic_next_command = "crypto price" if asset_type == "crypto" else "stock price"
    next_safe_command = (
        f"{'crypto price' if asset_type == 'crypto' else 'stock price of'} {command_instrument}"
        if command_instrument
        else generic_next_command
    )
    handoff = {
        "source": tool,
        "asset_type": asset_type,
        "service": service,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "requested": _short_raw(requested),
        "instrument": _short_raw(instrument),
        "status": "ok" if ok else "unavailable",
        "reason": reason,
        "price_available": price_num is not None,
        "price_usd": price_num,
        "content_in_handoff": False,
        "content_in_metadata": False,
        "next_safe_command": next_safe_command,
        "hidden_next_command_count": hidden_next_command_count,
        "boundaries": _market_boundaries(calls_external_service=calls_external_service),
    }
    handoff["next_safe_commands"] = [handoff["next_safe_command"]]
    handoff["next_safe_command_count"] = len(handoff["next_safe_commands"])
    if change_num is not None:
        handoff["change_pct"] = change_num
        handoff["change_window"] = "24h"
    if prev_num is not None:
        handoff["prev_close_usd"] = prev_num
    return {
        f"{asset_type}_price_handoff_ready": True,
        f"{asset_type}_price_handoff": handoff,
        f"{asset_type}_price_ready_for_operator": handoff["ready_for_operator"],
        f"{asset_type}_price_state_changed": handoff["state_changed"],
        f"{asset_type}_price_changed": handoff["changed"],
        f"{asset_type}_price_content_in_handoff": handoff["content_in_handoff"],
        f"{asset_type}_price_next_safe_command": handoff["next_safe_command"],
        f"{asset_type}_price_next_safe_commands": handoff["next_safe_commands"],
        f"{asset_type}_price_next_safe_command_count": handoff["next_safe_command_count"],
        f"{asset_type}_price_hidden_next_command_count": hidden_next_command_count,
        f"{asset_type}_price_authorizes_execution": handoff["authorizes_execution"],
        f"{asset_type}_price_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        f"{asset_type}_price_approval_granted": handoff["approval_granted"],
        f"{asset_type}_price_boundaries": handoff["boundaries"],
        "market_price_handoff_ready": True,
        "market_price_handoff": handoff,
        "market_price_ready_for_operator": handoff["ready_for_operator"],
        "market_price_state_changed": handoff["state_changed"],
        "market_price_changed": handoff["changed"],
        "market_price_content_in_handoff": handoff["content_in_handoff"],
        "market_price_next_safe_command": handoff["next_safe_command"],
        "market_price_next_safe_commands": handoff["next_safe_commands"],
        "market_price_next_safe_command_count": handoff["next_safe_command_count"],
        "market_price_hidden_next_command_count": hidden_next_command_count,
        "market_price_authorizes_execution": handoff["authorizes_execution"],
        "market_price_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "market_price_approval_granted": handoff["approval_granted"],
        "market_price_boundaries": handoff["boundaries"],
    }


def _overview_handoff(
    *,
    rows: list[dict[str, Any]],
    crypto_only: bool,
    errors: list[str],
    response_language: str,
) -> dict[str, Any]:
    next_safe_command = "crypto prices" if crypto_only else "how are the markets"
    handoff = {
        "source": "get_markets_overview",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "crypto_only": crypto_only,
        "instrument_count": len(rows),
        "instruments": [row["instrument"] for row in rows],
        "rows": rows,
        "partial": bool(errors),
        "error_count": len(errors),
        "errors": errors[:4],
        "response_language": response_language,
        "content_in_handoff": False,
        "content_in_metadata": False,
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "not_financial_advice": True,
        "boundaries": _market_boundaries(),
    }
    return {
        "markets_overview_handoff_ready": True,
        "markets_overview_handoff": handoff,
        "markets_overview_ready_for_operator": handoff["ready_for_operator"],
        "markets_overview_state_changed": handoff["state_changed"],
        "markets_overview_changed": handoff["changed"],
        "markets_overview_content_in_handoff": handoff["content_in_handoff"],
        "markets_overview_next_safe_command": handoff["next_safe_command"],
        "markets_overview_next_safe_commands": handoff["next_safe_commands"],
        "markets_overview_next_safe_command_count": handoff["next_safe_command_count"],
        "markets_overview_response_language": handoff["response_language"],
        "markets_overview_authorizes_execution": handoff["authorizes_execution"],
        "markets_overview_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "markets_overview_approval_granted": handoff["approval_granted"],
        "markets_overview_boundaries": handoff["boundaries"],
    }


def _short_raw(value: Any, *, limit: int = 80) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = SECRET_VALUE_RE.sub("<redacted-secret>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_command_instrument(value: Any) -> str:
    text = "" if value is None else " ".join(str(value).strip().split())
    if not text or LOCAL_PATH_RE.search(text) or SECRET_VALUE_RE.search(text):
        return ""
    return text if COMMAND_INSTRUMENT_RE.fullmatch(text) else ""


def _safe_instrument_label(value: Any, *, fallback: str = "") -> str:
    label = _short_raw(value, limit=32)
    if label and label not in {"<local-path>", "<redacted-secret>"}:
        return label
    return _short_raw(fallback, limit=32)


def _declare_market_input_failure(
    metadata: dict[str, Any],
    *,
    output: str,
) -> dict[str, Any]:
    return declare_retryable_local_read_failure(
        metadata,
        output=output,
        action=MARKET_INPUT_RECOVERY_ACTION,
    )


def _declare_market_lookup_failure(
    metadata: dict[str, Any],
    *,
    output: str,
) -> dict[str, Any]:
    return declare_retryable_external_information_failure(
        metadata,
        output=output,
        action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
        commands=("setup check",),
    )


def _market_response_language(args: dict[str, Any], text: str) -> str:
    requested = " ".join(str(args.get(key) or "") for key in ("language", "lang", "locale"))
    combined = f"{requested} {text}".lower()
    if re.search(r"\b(?:korean|ko)\b", combined) or any(
        marker in combined for marker in ("한국어", "한국말", "한글", "한국어로", "한국말로", "한글로")
    ):
        return "ko"
    return "en"


def _get(url: str) -> bytes:
    from jarvis_v2.tools._http import http_get
    return http_get(url)


def _get_json(url: str) -> Any:
    from jarvis_v2.tools._http import http_get_json
    return http_get_json(url)


def _fetch_crypto(coin_id: str) -> dict:
    return _fetch_crypto_many([coin_id])


def _fetch_crypto_many(coin_ids: list[str]) -> dict:
    ids = ",".join(dict.fromkeys(coin_ids))
    params = urllib.parse.urlencode({"ids": ids, "vs_currencies": "usd", "include_24hr_change": "true"})
    data = _get_json("https://api.coingecko.com/api/v3/simple/price?" + params)
    return data if isinstance(data, dict) else {}


def _fetch_stock(symbol: str) -> dict:
    """Return {'price': float, 'prev': float, 'symbol': str} from Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?interval=1d&range=1d"
    data = json.loads(_get(url).decode("utf-8"))
    meta = ((data.get("chart") or {}).get("result") or [{}])[0].get("meta") or {}
    return {
        "price": meta.get("regularMarketPrice"),
        "prev": meta.get("chartPreviousClose"),
        "symbol": meta.get("symbol") or symbol.upper(),
    }


def _format_price_line(label: str, price: Any, prev: Any = None, change: Any = None, *, change_window: str = "") -> str:
    price_num = _as_float(price)
    if price_num is None:
        return f"{label}: unavailable"
    change_str = ""
    change_num = _as_float(change)
    prev_num = _as_float(prev)
    if change_num is not None:
        arrow = "▲" if change_num >= 0 else "▼"
        suffix = f" {change_window}" if change_window else ""
        change_str = f" ({arrow}{abs(change_num):.1f}%{suffix})"
    elif prev_num:
        pct = (price_num - prev_num) / prev_num * 100
        arrow = "▲" if pct >= 0 else "▼"
        change_str = f" ({arrow}{abs(pct):.1f}%)"
    return f"{label}: ${price_num:,.2f}{change_str}"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_error(e: Exception, *, subject: str, service: str) -> str:
    from jarvis_v2.tools._http import friendly_http_error

    friendly = friendly_http_error(e, subject=subject, service=service)
    if friendly.startswith("I couldn't find"):
        return friendly
    return (
        f"{service} is having trouble right now. Check network access to {service}, "
        "run setup check, then retry in a moment."
    )


def make_markets_tools(config: JarvisConfig):
    def get_crypto_price(args: dict[str, Any]) -> ToolResult:
        raw_coin = args.get("coin") or args.get("symbol")
        name = str(raw_coin or "").strip().lower()
        if not name:
            m = re.search(r"\b(bitcoin|btc|ethereum|eth|solana|sol|dogecoin|doge|cardano|ada|ripple|xrp|bnb|binance)\b",
                          str(args.get("text") or "").lower())
            name = m.group(1) if m else "bitcoin"
        coin_id = CRYPTO_IDS.get(name)
        if not coin_id:
            failure_output = (
                "Unsupported coin. Try bitcoin, ethereum, solana, dogecoin, cardano, ripple, or bnb. "
                f"{MARKET_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_crypto_price",
                False,
                failure_output,
                _declare_market_input_failure(
                    _safe_metadata(
                        calls_external_service=False,
                        reason="unsupported_coin",
                        coin=None,
                        requested_coin=_short_raw(raw_coin or args.get("text") or name),
                        **_price_handoff(
                            tool="get_crypto_price",
                            asset_type="crypto",
                            requested=raw_coin or args.get("text") or name,
                            instrument=None,
                            service="CoinGecko",
                            ok=False,
                            calls_external_service=False,
                            reason="unsupported_coin",
                        ),
                    ),
                    output=failure_output,
                ),
            )
        try:
            data = _fetch_crypto(coin_id)
            row = data.get(coin_id)
            if not row:
                failure_output = (
                    f"Couldn't find a price for '{name}'. "
                    f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "get_crypto_price",
                    False,
                    failure_output,
                    _declare_market_lookup_failure(
                        _safe_metadata(
                            coin=coin_id,
                            requested_coin=name,
                            reason="missing_price",
                            **_price_handoff(
                                tool="get_crypto_price",
                                asset_type="crypto",
                                requested=name,
                                instrument=coin_id,
                                service="CoinGecko",
                                ok=False,
                                reason="missing_price",
                            ),
                        ),
                        output=failure_output,
                    ),
                )
            price = row.get("usd")
            change = row.get("usd_24h_change")
            line = _format_price_line(coin_id.capitalize(), price, change=change, change_window="24h")
            if "unavailable" in line:
                failure_output = (
                    f"Couldn't find a usable price for '{name}'. "
                    f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "get_crypto_price",
                    False,
                    failure_output,
                    _declare_market_lookup_failure(
                        _safe_metadata(
                            coin=coin_id,
                            requested_coin=name,
                            reason="bad_price_data",
                            **_price_handoff(
                                tool="get_crypto_price",
                                asset_type="crypto",
                                requested=name,
                                instrument=coin_id,
                                service="CoinGecko",
                                ok=False,
                                reason="bad_price_data",
                                price_usd=price,
                                change_pct=change,
                            ),
                        ),
                        output=failure_output,
                    ),
                )
            return ToolResult(
                "get_crypto_price",
                True,
                line,
                _safe_metadata(
                    coin=coin_id,
                    requested_coin=name,
                    **_price_handoff(
                        tool="get_crypto_price",
                        asset_type="crypto",
                        requested=name,
                        instrument=coin_id,
                        service="CoinGecko",
                        ok=True,
                        price_usd=price,
                        change_pct=change,
                    ),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_market_error(e, subject=f'a price for {name!r}', service='CoinGecko')} "
                f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_crypto_price",
                False,
                failure_output,
                declare_retryable_external_information_failure(
                    _safe_metadata(
                        coin=coin_id,
                        requested_coin=name,
                        reason="fetch_error",
                        **_price_handoff(
                            tool="get_crypto_price",
                            asset_type="crypto",
                            requested=name,
                            instrument=coin_id,
                            service="CoinGecko",
                            ok=False,
                            reason="fetch_error",
                        ),
                    ),
                    output=failure_output,
                    action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    def get_stock_price(args: dict[str, Any]) -> ToolResult:
        symbol = str(args.get("symbol") or args.get("ticker") or "").strip()
        if not symbol:
            text = str(args.get("text") or "").lower()
            for company, ticker in STOCK_ALIASES.items():
                if re.search(rf"\b{re.escape(company)}\b.{{0,16}}\b(?:stock|share|quote|price|trading|at)\b", text):
                    symbol = ticker
                    break
            patterns = [
                r"\b(?:stock|share)\s+price\s+(?:of|for)\s+([a-z]{1,5})\b",
                r"\b(?:stock|ticker)\s+(?:of\s+|for\s+)?(?!price\b|quote\b)([a-z]{1,5})\b",
                r"\b([a-z]{1,5})\s+(?:stock|shares?)\b",
                r"\bprice\s+(?:of|for)\s+([a-z]{1,5})\s+stock\b",
                r"\b(?:quote|price)\s+([a-z]{1,5})\b",
                r"\b([a-z]{1,5})\s+(?:quote|price)\b",
                r"\b(?:what'?s|what is|how much is)\s+([a-z]{1,5})\s+(?:trading\s+)?(?:at|worth)\b",
            ]
            if not symbol:
                for pattern in patterns:
                    m = re.search(pattern, text)
                    if m and m.group(1) not in {"at", "for", "of", "the", "stock", "share", "price", "quote"}:
                        symbol = m.group(1)
                        break
        if not symbol:
            failure_output = (
                "Which ticker? e.g. 'stock price of AAPL'. "
                f"{MARKET_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_stock_price",
                False,
                failure_output,
                _declare_market_input_failure(
                    _safe_metadata(
                        calls_external_service=False,
                        reason="missing_ticker",
                        symbol=None,
                        raw_symbol=_short_raw(args.get("symbol") or args.get("ticker") or args.get("text")),
                        **_price_handoff(
                            tool="get_stock_price",
                            asset_type="stock",
                            requested=args.get("symbol") or args.get("ticker") or args.get("text"),
                            instrument=None,
                            service="Yahoo Finance",
                            ok=False,
                            calls_external_service=False,
                            reason="missing_ticker",
                        ),
                    ),
                    output=failure_output,
                ),
            )
        symbol = symbol.upper()
        if not re.fullmatch(r"[A-Z]{1,5}", symbol):
            failure_output = (
                "Ticker must be 1-5 letters, e.g. 'AAPL'. "
                f"{MARKET_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_stock_price",
                False,
                failure_output,
                _declare_market_input_failure(
                    _safe_metadata(
                        calls_external_service=False,
                        reason="bad_ticker",
                        symbol=None,
                        raw_symbol=_short_raw(args.get("symbol") or args.get("ticker") or symbol),
                        **_price_handoff(
                            tool="get_stock_price",
                            asset_type="stock",
                            requested=args.get("symbol") or args.get("ticker") or symbol,
                            instrument=None,
                            service="Yahoo Finance",
                            ok=False,
                            calls_external_service=False,
                            reason="bad_ticker",
                        ),
                    ),
                    output=failure_output,
                ),
            )
        try:
            quote = _fetch_stock(symbol)
            price = quote.get("price")
            if _as_float(price) is None:
                failure_output = (
                    f"No price for '{symbol}'. "
                    f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "get_stock_price",
                    False,
                    failure_output,
                    _declare_market_lookup_failure(
                        _safe_metadata(
                            symbol=symbol,
                            reason="missing_price",
                            **_price_handoff(
                                tool="get_stock_price",
                                asset_type="stock",
                                requested=symbol,
                                instrument=symbol,
                                service="Yahoo Finance",
                                ok=False,
                                reason="missing_price",
                                price_usd=price,
                                prev_close_usd=quote.get("prev"),
                            ),
                        ),
                        output=failure_output,
                    ),
                )
            prev = quote.get("prev")
            actual_symbol = _safe_instrument_label(str(quote.get("symbol") or symbol).upper(), fallback=symbol)
            return ToolResult(
                "get_stock_price", True,
                _format_price_line(actual_symbol, price, prev=prev),
                _safe_metadata(
                    symbol=actual_symbol,
                    **_price_handoff(
                        tool="get_stock_price",
                        asset_type="stock",
                        requested=symbol,
                        instrument=actual_symbol,
                        service="Yahoo Finance",
                        ok=True,
                        price_usd=price,
                        prev_close_usd=prev,
                    ),
                ),
            )
        except Exception as e:
            if "404" in str(e):
                failure_output = (
                    f"No price found for '{symbol}' — is that a valid ticker? "
                    f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "get_stock_price",
                    False,
                    failure_output,
                    _declare_market_lookup_failure(
                        _safe_metadata(
                            symbol=symbol,
                            reason="not_found",
                            **_price_handoff(
                                tool="get_stock_price",
                                asset_type="stock",
                                requested=symbol,
                                instrument=symbol,
                                service="Yahoo Finance",
                                ok=False,
                                reason="not_found",
                            ),
                        ),
                        output=failure_output,
                    ),
                )
            failure_output = (
                f"{_market_error(e, subject=f'a price for {symbol!r}', service='Yahoo Finance')} "
                f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "get_stock_price",
                False,
                failure_output,
                declare_retryable_external_information_failure(
                    _safe_metadata(
                        symbol=symbol,
                        reason="fetch_error",
                        **_price_handoff(
                            tool="get_stock_price",
                            asset_type="stock",
                            requested=symbol,
                            instrument=symbol,
                            service="Yahoo Finance",
                            ok=False,
                            reason="fetch_error",
                        ),
                    ),
                    output=failure_output,
                    action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )

    def get_markets_overview(args: dict[str, Any]) -> ToolResult:
        raw_text = str(args.get("text") or "").strip()
        text = raw_text.lower()
        crypto_only = _metadata_bool(args.get("crypto_only")) or "crypto" in text and "market" not in text
        response_language = _market_response_language(args, raw_text)
        korean = response_language == "ko"
        crypto_ids = ["bitcoin", "ethereum", "solana"] if crypto_only else ["bitcoin", "ethereum"]
        stock_symbols = [] if crypto_only else ["AAPL", "NVDA"]
        lines = (
            ["암호화폐 가격:" if crypto_only else "시장 스냅샷:"]
            if korean
            else ["Crypto prices:" if crypto_only else "Markets snapshot:"]
        )
        instruments: list[str] = []
        rows: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            crypto_data = _fetch_crypto_many(crypto_ids)
            for coin_id in crypto_ids:
                row = crypto_data.get(coin_id) or {}
                label = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}.get(coin_id, coin_id.upper())
                lines.append("- " + _format_price_line(label, row.get("usd"), change=row.get("usd_24h_change"), change_window="24h"))
                instruments.append(label)
                price_available = _as_float(row.get("usd")) is not None
                if not price_available:
                    errors.append(f"CoinGecko returned no usable price for {label}.")
                rows.append(
                    {
                        "instrument": label,
                        "asset_type": "crypto",
                        "service": "CoinGecko",
                        "status": "ok" if price_available else "unavailable",
                        "price_available": price_available,
                        "price_usd": _as_float(row.get("usd")),
                        "change_pct": _as_float(row.get("usd_24h_change")),
                        "change_window": "24h",
                    }
                )
        except Exception as e:
            errors.append(_market_error(e, subject="crypto prices", service="CoinGecko"))
            for coin_id in crypto_ids:
                label = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}.get(coin_id, coin_id.upper())
                lines.append(f"- {label}: unavailable")
                instruments.append(label)
                rows.append(
                    {
                        "instrument": label,
                        "asset_type": "crypto",
                        "service": "CoinGecko",
                        "status": "unavailable",
                        "price_available": False,
                        "price_usd": None,
                    }
                )

        for symbol in stock_symbols:
            try:
                quote = _fetch_stock(symbol)
                actual_symbol = _safe_instrument_label(str(quote.get("symbol") or symbol).upper(), fallback=symbol)
                lines.append("- " + _format_price_line(actual_symbol, quote.get("price"), prev=quote.get("prev")))
                instruments.append(actual_symbol)
                price_available = _as_float(quote.get("price")) is not None
                if not price_available:
                    errors.append(f"Yahoo Finance returned no usable price for {actual_symbol}.")
                rows.append(
                    {
                        "instrument": actual_symbol,
                        "asset_type": "stock",
                        "service": "Yahoo Finance",
                        "status": "ok" if price_available else "unavailable",
                        "price_available": price_available,
                        "price_usd": _as_float(quote.get("price")),
                        "prev_close_usd": _as_float(quote.get("prev")),
                    }
                )
            except Exception as e:
                errors.append(_market_error(e, subject=f"a price for '{symbol}'", service="Yahoo Finance"))
                lines.append(f"- {symbol}: unavailable")
                instruments.append(symbol)
                rows.append(
                    {
                        "instrument": symbol,
                        "asset_type": "stock",
                        "service": "Yahoo Finance",
                        "status": "unavailable",
                        "price_available": False,
                        "price_usd": None,
                    }
                )

        if errors:
            lines.append("")
            prefix = "일부 시장 데이터를 가져오지 못했어요: " if korean else "Some market data was unavailable: "
            lines.append(prefix + " | ".join(dict.fromkeys(errors)))
            lines.append(EXTERNAL_INFORMATION_RECOVERY_ACTION)
        lines.append("")
        lines.append("정보 제공용이며 투자 조언이 아니에요." if korean else "Informational only, not financial advice.")
        output = "\n".join(lines)
        metadata = _safe_metadata(
            instruments=instruments,
            crypto_only=crypto_only,
            partial=bool(errors),
            errors=errors[:4],
            instrument_count=len(rows),
            response_language=response_language,
            **_overview_handoff(
                rows=rows,
                crypto_only=crypto_only,
                errors=errors,
                response_language=response_language,
            ),
        )
        if errors:
            metadata = _declare_market_lookup_failure(metadata, output=output)
        return ToolResult(
            "get_markets_overview",
            True,
            output,
            metadata,
        )

    from jarvis_v2.tools.registry import Tool, _tool_argument_contract
    return [
        Tool(
            "get_crypto_price",
            "Get a cryptocurrency price in USD (free). Args: coin (e.g. 'bitcoin') or text.",
            RiskLevel.LOCAL_SAFE,
            get_crypto_price,
            "personal",
            argument_contract=_tool_argument_contract(optional_strings=("coin", "symbol", "text")),
        ),
        Tool(
            "get_stock_price",
            "Get a US stock price (free). Args: symbol (e.g. 'AAPL') or text.",
            RiskLevel.LOCAL_SAFE,
            get_stock_price,
            "personal",
            argument_contract=_tool_argument_contract(optional_strings=("symbol", "ticker", "text")),
        ),
        Tool(
            "get_markets_overview",
            "Get a compact crypto and stock market snapshot. Args: text or crypto_only.",
            RiskLevel.LOCAL_SAFE,
            get_markets_overview,
            "personal",
            argument_contract=_tool_argument_contract(
                optional_strings=("text", "language", "lang", "locale"),
                optional_booleans=("crypto_only",),
            ),
        ),
    ]
