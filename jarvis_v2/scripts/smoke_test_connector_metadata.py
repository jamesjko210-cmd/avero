"""Smoke tests for shared connector metadata invariants."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jarvis_v2.tools import (
    air_connector,
    currency_connector,
    dictionary_connector,
    history_connector,
    holidays_connector,
    markets_connector,
    news_connector,
    sun_connector,
    translate_connector,
    weather_connector,
    wikipedia_connector,
)


SAFE_METADATA_HELPERS: dict[str, Callable[..., dict[str, Any]]] = {
    "air": air_connector._safe_metadata,
    "currency": currency_connector._safe_metadata,
    "dictionary": dictionary_connector._safe_metadata,
    "history": history_connector._safe_metadata,
    "holidays": holidays_connector._safe_metadata,
    "markets": markets_connector._safe_metadata,
    "news": news_connector._safe_metadata,
    "sun": sun_connector._safe_metadata,
    "translate": translate_connector._safe_metadata,
    "weather": weather_connector._safe_metadata,
    "wikipedia": wikipedia_connector._safe_metadata,
}


def assert_external_service_alias_is_exact() -> None:
    malformed_values = ("true", "false", "yes", "no", 1, 0, [True], {"calls": True}, None)
    for label, helper in SAFE_METADATA_HELPERS.items():
        exact_true = helper(calls_external_service=True)
        if exact_true.get("calls_external_service") is not True or exact_true.get("calls_external_services") is not True:
            raise SystemExit(f"{label} should mirror exact True to plural external-service alias: {exact_true}")
        exact_false = helper(calls_external_service=False)
        if exact_false.get("calls_external_service") is not False or exact_false.get("calls_external_services") is not False:
            raise SystemExit(f"{label} should mirror exact False to plural external-service alias: {exact_false}")
        for value in malformed_values:
            metadata = helper(calls_external_service=value)
            if metadata.get("calls_external_services") is not False:
                raise SystemExit(f"{label} accepted malformed external-service alias value {value!r}: {metadata}")


def main() -> None:
    assert_external_service_alias_is_exact()
    print("Connector metadata smoke passed")


if __name__ == "__main__":
    main()
