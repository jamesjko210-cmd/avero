"""Live currency conversion for Jarvis V2 via Frankfurter (ECB rates, no API key)."""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    EXTERNAL_INFORMATION_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    declare_retryable_external_information_failure,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools._http import http_get, http_get_json, friendly_http_error


CURRENCY_ALIASES = {
    "usd": "USD", "dollar": "USD", "dollars": "USD", "buck": "USD", "bucks": "USD", "$": "USD",
    "krw": "KRW", "won": "KRW", "원": "KRW",
    "eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR",
    "jpy": "JPY", "yen": "JPY", "엔": "JPY",
    "gbp": "GBP", "pound": "GBP", "pounds": "GBP", "£": "GBP",
    "cny": "CNY", "yuan": "CNY", "rmb": "CNY",
    "aud": "AUD", "cad": "CAD", "chf": "CHF", "hkd": "HKD", "sgd": "SGD", "inr": "INR",
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


def _currency_boundaries(*, calls_external_service: bool) -> dict[str, bool]:
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


def _currency_handoff(
    *,
    status: str,
    reason: str = "",
    amount: Any = None,
    src: str | None = None,
    dst: str | None = None,
    result: Any = None,
    raw_amount: Any = None,
    raw_from: Any = None,
    raw_to: Any = None,
    raw_text: Any = None,
    calls_external_service: bool,
    exception_type: str = "",
) -> dict[str, Any]:
    amount_num = _as_float(amount)
    result_num = _as_float(result)
    next_safe_command = f"convert {amount_num:g} {src} to {dst}" if amount_num is not None and src and dst else "convert <amount> <from> to <to>"
    boundaries = _currency_boundaries(calls_external_service=calls_external_service)
    handoff = {
        "source": "convert_currency",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "amount": amount_num,
        "src": src or "",
        "dst": dst or "",
        "result_available": result_num is not None,
        "result": result_num,
        "raw_amount": _short_raw(raw_amount),
        "raw_from": _short_raw(raw_from),
        "raw_to": _short_raw(raw_to),
        "raw_text": _short_raw(raw_text),
        "exception_type": exception_type,
        "content_in_handoff": False,
        "content_in_metadata": False,
        "retry_safe": status in {"unavailable", "empty"},
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "currency_handoff_ready": True,
        "currency_ready_for_operator": handoff["ready_for_operator"],
        "currency_state_changed": handoff["state_changed"],
        "currency_changed": handoff["changed"],
        "currency_content_in_handoff": handoff["content_in_handoff"],
        "currency_content_in_metadata": handoff["content_in_metadata"],
        "currency_next_safe_command": handoff["next_safe_command"],
        "currency_next_safe_commands": handoff["next_safe_commands"],
        "currency_next_safe_command_count": handoff["next_safe_command_count"],
        "currency_authorizes_execution": handoff["authorizes_execution"],
        "currency_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "currency_approval_granted": handoff["approval_granted"],
        "currency_boundaries": boundaries,
        "currency_handoff": handoff,
    }


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short_raw(value: Any, limit: int = 80) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = SECRET_VALUE_RE.sub("<secret>", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _has_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _code(token: str) -> str | None:
    token = (token or "").strip().lower()
    if token.upper() in set(CURRENCY_ALIASES.values()):
        return token.upper()
    return CURRENCY_ALIASES.get(token)


def parse_conversion(text: str) -> tuple[float, str, str] | None:
    low = " ".join(text.lower().split())
    amount_m = re.search(r"(\d[\d,]*(?:\.\d+)?)", low)
    amount = float(amount_m.group(1).replace(",", "")) if amount_m else 1.0
    # currency tokens, including symbols
    tokens = re.findall(r"[$€£]|[a-z원엔]{2,}", low)
    codes: list[str] = []
    for tok in tokens:
        c = _code(tok)
        if c and (not codes or codes[-1] != c):
            codes.append(c)
    if len(codes) < 2:
        return None
    # "how many X is/are/in Y Z" mentions the TARGET currency first and the
    # amount's own (source) currency second -- the reverse of every other
    # supported phrasing ("50 usd to eur", "how much is 50 euros in dollars",
    # etc.), where left-to-right token order already matches source-then-target.
    # Real bug found live: "how many usd is 100 eur" returned "100.00 USD =
    # 87.69 EUR" (backwards), and "how many usd in a euro" had the same issue,
    # before this check.
    if re.search(r"^how many\s+.+?\s+(?:is|are|in)\b", low):
        return amount, codes[1], codes[0]
    return amount, codes[0], codes[1]


def _fetch(amount: float, src: str, dst: str) -> dict:
    params = urllib.parse.urlencode({"amount": amount, "from": src, "to": dst})
    url = "https://api.frankfurter.app/latest?" + params
    return http_get_json(url, headers={"User-Agent": "Mozilla/5.0"})


def _currency_recovery_message(prefix: str) -> str:
    return f"{prefix} Check network access to Frankfurter, run setup check, then retry in a moment."


def _currency_error_message(e: Exception) -> str:
    friendly = friendly_http_error(e, service="The currency service")
    lowered = friendly.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return _currency_recovery_message("The currency request timed out.")
    if friendly.startswith("Error:") or "having trouble" in lowered or "try again" in lowered:
        return _currency_recovery_message("The currency service is having trouble right now.")
    return friendly


def make_currency_tools(config: JarvisConfig):
    def convert_currency(args: dict[str, Any]) -> ToolResult:
        amount = args.get("amount")
        raw_src = args.get("from")
        raw_dst = args.get("to")
        src = _code(str(raw_src or ""))
        dst = _code(str(raw_dst or ""))
        direct_args_present = amount is not None or raw_src is not None or raw_dst is not None
        if direct_args_present and amount is not None and (not src or not dst):
            failure_output = (
                "Tell me two supported currencies, e.g. from USD to KRW. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "convert_currency",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _safe_metadata(
                        reason="missing_currency",
                        raw_from=_short_raw(raw_src),
                        raw_to=_short_raw(raw_dst),
                        src=src,
                        dst=dst,
                        calls_external_service=False,
                        **_currency_handoff(
                            status="refused",
                            reason="missing_currency",
                            amount=amount,
                            src=src,
                            dst=dst,
                            raw_amount=amount,
                            raw_from=raw_src,
                            raw_to=raw_dst,
                            calls_external_service=False,
                        ),
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        if amount is None or not (src and dst):
            raw_text = str(args.get("text") or args.get("request") or "")
            if _has_local_path(raw_text):
                failure_output = (
                    "Currency requests cannot contain local file paths. "
                    f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "convert_currency",
                    False,
                    failure_output,
                    declare_retryable_local_read_failure(
                        _safe_metadata(
                            reason="local_path_request",
                            raw_text=_short_raw(raw_text),
                            src=src,
                            dst=dst,
                            calls_external_service=False,
                            **_currency_handoff(
                                status="refused",
                                reason="local_path_request",
                                src=src,
                                dst=dst,
                                raw_from=raw_src,
                                raw_to=raw_dst,
                                raw_text=raw_text,
                                calls_external_service=False,
                            ),
                        ),
                        output=failure_output,
                        action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                    ),
                )
            parsed = parse_conversion(raw_text)
            if parsed is None:
                failure_output = (
                    "Tell me an amount and two currencies, e.g. 'convert 100 USD to KRW'. "
                    f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "convert_currency",
                    False,
                    failure_output,
                    declare_retryable_local_read_failure(
                        _safe_metadata(
                            reason="missing_conversion",
                            raw_from=_short_raw(raw_src),
                            raw_to=_short_raw(raw_dst),
                            calls_external_service=False,
                            **_currency_handoff(
                                status="refused",
                                reason="missing_conversion",
                                raw_from=raw_src,
                                raw_to=raw_dst,
                                raw_text=raw_text,
                                calls_external_service=False,
                            ),
                        ),
                        output=failure_output,
                        action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                    ),
                )
            amount, src, dst = parsed
        try:
            amount_value = float(amount)
        except (TypeError, ValueError):
            failure_output = (
                "Currency amount must be a number. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "convert_currency",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _safe_metadata(
                        raw_amount=_short_raw(amount),
                        src=src,
                        dst=dst,
                        calls_external_service=False,
                        reason="bad_amount",
                        **_currency_handoff(
                            status="refused",
                            reason="bad_amount",
                            src=src,
                            dst=dst,
                            raw_amount=amount,
                            raw_from=raw_src,
                            raw_to=raw_dst,
                            calls_external_service=False,
                        ),
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        if not math.isfinite(amount_value):
            failure_output = (
                "Currency amount must be finite. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "convert_currency",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _safe_metadata(
                        raw_amount=_short_raw(amount),
                        src=src,
                        dst=dst,
                        calls_external_service=False,
                        reason="non_finite_amount",
                        **_currency_handoff(
                            status="refused",
                            reason="non_finite_amount",
                            amount=amount,
                            src=src,
                            dst=dst,
                            raw_amount=amount,
                            raw_from=raw_src,
                            raw_to=raw_dst,
                            calls_external_service=False,
                        ),
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        try:
            data = _fetch(amount_value, src, dst)
            rate = (data.get("rates") or {}).get(dst)
            if rate is None:
                missing_rate_message = _currency_recovery_message(
                    f"Couldn't get a {src}->{dst} rate."
                )
                failure_output = (
                    f"{missing_rate_message} "
                    f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
                )
                return ToolResult(
                    "convert_currency",
                    False,
                    failure_output,
                    declare_retryable_external_information_failure(
                        _safe_metadata(
                            amount=amount_value,
                            src=src,
                            dst=dst,
                            reason="missing_rate",
                            **_currency_handoff(
                                status="empty",
                                reason="missing_rate",
                                amount=amount_value,
                                src=src,
                                dst=dst,
                                raw_amount=amount,
                                raw_from=raw_src,
                                raw_to=raw_dst,
                                calls_external_service=True,
                            ),
                        ),
                        output=failure_output,
                        action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
                        commands=("setup check",),
                    ),
                )
            result_value = float(rate)
            if not math.isfinite(result_value):
                raise ValueError("currency service returned a non-finite rate")
            return ToolResult(
                "convert_currency", True,
                f"{amount_value:,.2f} {src} = {result_value:,.2f} {dst}",
                _safe_metadata(
                    amount=amount_value,
                    src=src,
                    dst=dst,
                    result=result_value,
                    **_currency_handoff(
                        status="ok",
                        amount=amount_value,
                        src=src,
                        dst=dst,
                        result=result_value,
                        raw_amount=amount,
                        raw_from=raw_src,
                        raw_to=raw_dst,
                        calls_external_service=True,
                    ),
                ),
            )
        except Exception as e:
            failure_output = (
                f"{_currency_error_message(e)} "
                f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
            )
            return ToolResult(
                "convert_currency",
                False,
                failure_output,
                declare_retryable_external_information_failure(
                    _safe_metadata(
                        amount=amount_value,
                        src=src,
                        dst=dst,
                        exception_type=type(e).__name__,
                        reason="fetch_error",
                        **_currency_handoff(
                            status="unavailable",
                            reason="fetch_error",
                            amount=amount_value,
                            src=src,
                            dst=dst,
                            raw_amount=amount,
                            raw_from=raw_src,
                            raw_to=raw_dst,
                            calls_external_service=True,
                            exception_type=type(e).__name__,
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
            "convert_currency",
            "Convert between currencies at the latest rate. Args: text (e.g. 'convert 100 USD to KRW') or amount/from/to.",
            RiskLevel.LOCAL_SAFE,
            convert_currency,
            "personal",
            argument_contract=_tool_argument_contract(
                optional_strings=("text", "request", "from", "to"),
                optional_numeric_inputs=("amount",),
            ),
        ),
    ]
