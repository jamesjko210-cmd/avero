"""Opt-in live smoke test: hit every live connector's real API once.

This is a DIAGNOSTIC, not part of smoke_test_all — it makes real network calls
and never fails CI. It just prints a ✓/✗ row per tool so you can see which
external services are up.

Run:
    JARVIS_LIVE_SMOKE=1 python3 -m jarvis_v2.scripts.smoke_test_live
"""

from __future__ import annotations

import os
import sys

from jarvis_v2.config import load_config


# (label, module, factory, tool_name, args)
LIVE_CHECKS = [
    ("weather", "weather_connector", "make_weather_tools", "get_weather", {}),
    ("news", "news_connector", "make_news_tools", "get_news", {"limit": 2}),
    ("air quality", "air_connector", "make_air_tools", "get_air_quality", {}),
    ("sun times", "sun_connector", "make_sun_tools", "get_sun_times", {"text": "sunset"}),
    ("on this day", "history_connector", "make_history_tools", "on_this_day", {}),
    ("holidays", "holidays_connector", "make_holidays_tools", "next_holidays", {}),
    ("dictionary", "dictionary_connector", "make_dictionary_tools", "define", {"word": "lucid"}),
    ("wikipedia", "wikipedia_connector", "make_wikipedia_tools", "wiki_summary", {"query": "Seoul"}),
    ("currency", "currency_connector", "make_currency_tools", "convert_currency", {"text": "convert 1 usd to krw"}),
    ("translate", "translate_connector", "make_translate_tools", "translate", {"request": "translate water to korean"}),
    ("crypto", "markets_connector", "make_markets_tools", "get_crypto_price", {"text": "bitcoin price"}),
    ("stock", "markets_connector", "make_markets_tools", "get_stock_price", {"text": "stock price of AAPL"}),
    ("web lookup", "research_connector", "make_research_tools", "web_lookup", {"query": "python release"}),
    ("joke", "fun_connector", "make_fun_tools", "tell_joke", {}),
]


def main() -> None:
    if os.getenv("JARVIS_LIVE_SMOKE") != "1":
        print("Live smoke is opt-in. Run with: JARVIS_LIVE_SMOKE=1 python3 -m jarvis_v2.scripts.smoke_test_live")
        return

    import importlib

    config = load_config()
    up = 0
    print(f"{'TOOL':<14} STATUS  DETAIL")
    print("-" * 72)
    for label, module, factory, tool_name, args in LIVE_CHECKS:
        try:
            mod = importlib.import_module(f"jarvis_v2.tools.{module}")
            tool = {t.name: t for t in getattr(mod, factory)(config)}[tool_name]
            result = tool.handler(args)
            ok = result.ok
            detail = (result.output or "").splitlines()[0][:48]
        except Exception as e:  # never raise — this is a diagnostic
            ok = False
            detail = f"EXCEPTION: {e}"[:48]
        if ok:
            up += 1
        print(f"{label:<14} {'✓ up' if ok else '✗ down':<7} {detail}")
    print("-" * 72)
    print(f"{up}/{len(LIVE_CHECKS)} live services reachable.")
    # Diagnostic: exit 0 regardless so it never breaks automation.
    sys.exit(0)


if __name__ == "__main__":
    main()
