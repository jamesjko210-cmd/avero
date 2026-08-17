"""Deterministic smoke tests for the shared HTTP helper (no network)."""

from __future__ import annotations

import email.utils
import math
import traceback
import urllib.error
from unittest.mock import patch

import jarvis_v2.tools._http as http_module
from jarvis_v2.tools._http import HttpError, friendly_http_error, http_get, http_get_json


class FakeClock:
    def __init__(self, *, monotonic: float = 0.0, wall_time: float = 1_700_000_000.0):
        self.now = monotonic
        self.wall_epoch = wall_time - monotonic
        self.sleeps: list[float] = []
        self.events: list[str] = []

    def monotonic(self) -> float:
        return self.now

    def wall_time(self) -> float:
        return self.wall_epoch + self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.events.append(f"sleep:{seconds:g}")
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        events: list[str] | None = None,
        *,
        max_chunk: int | None = None,
    ):
        self.payload = payload
        self.events = events
        self.max_chunk = max_chunk
        self.offset = 0
        self.read_sizes: list[int] = []
        self.read1_sizes: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.events is not None:
            self.events.append("response_close")
        return False

    def _read(self, size: int) -> bytes:
        amount = len(self.payload) - self.offset if size < 0 else size
        if self.max_chunk is not None:
            amount = min(amount, self.max_chunk)
        chunk = self.payload[self.offset : self.offset + amount]
        self.offset += len(chunk)
        return chunk

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self._read(size)

    def read1(self, size: int = -1) -> bytes:
        self.read1_sizes.append(size)
        return self._read(size)


class TrackingBody:
    def __init__(self, events: list[str], *, fail_close: bool = False):
        self.events = events
        self.fail_close = fail_close

    def close(self) -> None:
        self.events.append("close")
        if self.fail_close:
            self.fail_close = False
            raise OSError("close failed at /\x55sers/private/close")


def _http_error(
    status: int,
    *,
    retry_after: str | None = None,
    events: list[str] | None = None,
    hostile: bool = False,
) -> urllib.error.HTTPError:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    path = "/\x55sers/private/token" if hostile else "/resource"
    reason = "secret reason /private/tmp/credential" if hostile else "failure"
    body = TrackingBody(events, fail_close=hostile) if events is not None else None
    return urllib.error.HTTPError(f"https://example.test{path}", status, reason, headers, body)


def _request_kwargs(clock: FakeClock) -> dict:
    return {
        "_sleep": clock.sleep,
        "_monotonic": clock.monotonic,
        "_wall_time": clock.wall_time,
    }


def _assert_no_raw_prefix(message: str) -> None:
    forbidden = ["Error:", "HTTP Error", "offline"]
    for fragment in forbidden:
        if fragment in message:
            raise SystemExit(f"friendly_http_error leaked raw text {fragment!r}: {message}")


def _capture_http_error(callable_) -> HttpError:
    try:
        callable_()
    except HttpError as exc:
        return exc
    raise SystemExit("HTTP helper did not raise HttpError")


def test_status_errors_are_friendly() -> None:
    not_found = friendly_http_error(HttpError(404, "HTTP Error 404: Not Found"), subject="weather for 'badcity'", service="Weather")
    if "I couldn't find weather for 'badcity'." != not_found:
        raise SystemExit(f"404 status message wrong: {not_found}")
    _assert_no_raw_prefix(not_found)

    server = friendly_http_error(HttpError(503, "HTTP Error 503: Service Unavailable"), service="Weather")
    if "Weather is having trouble" not in server:
        raise SystemExit(f"5xx status message wrong: {server}")
    _assert_no_raw_prefix(server)


def test_string_only_http_errors_are_friendly() -> None:
    not_found = friendly_http_error(RuntimeError("HTTP Error 404: Not Found"), subject="a definition for 'zzxqq'", service="Dictionary")
    if "I couldn't find a definition for 'zzxqq'." != not_found:
        raise SystemExit(f"string 404 message wrong: {not_found}")
    _assert_no_raw_prefix(not_found)

    server = friendly_http_error(RuntimeError("HTTP Error 500: Internal Server Error"), service="CoinGecko")
    if "CoinGecko is having trouble" not in server:
        raise SystemExit(f"string 5xx message wrong: {server}")
    _assert_no_raw_prefix(server)


def test_timeout_and_generic_errors_are_friendly() -> None:
    timeout = friendly_http_error(TimeoutError("timed out"), service="The news service")
    if timeout != "The request timed out — try again.":
        raise SystemExit(f"timeout message wrong: {timeout}")
    _assert_no_raw_prefix(timeout)

    generic = friendly_http_error(RuntimeError("offline"), service="The news service")
    if "The news service is having trouble" not in generic:
        raise SystemExit(f"generic message wrong: {generic}")
    _assert_no_raw_prefix(generic)


def test_option_validation_table() -> None:
    invalid_timeouts = [0, -1, math.inf, -math.inf, math.nan, True, "12", None]
    invalid_retries = [-1, 4, 1.0, True, "1", None]

    for timeout in invalid_timeouts:
        with patch.object(http_module.urllib.request, "urlopen") as opener:
            try:
                http_get("https://example.test/validation", timeout=timeout)  # type: ignore[arg-type]
            except ValueError:
                pass
            else:
                raise SystemExit(f"HTTP helper accepted invalid timeout {timeout!r}")
        if opener.called:
            raise SystemExit(f"Invalid timeout {timeout!r} reached urlopen")

    for retries in invalid_retries:
        with patch.object(http_module.urllib.request, "urlopen") as opener:
            try:
                http_get("https://example.test/validation", retries=retries)  # type: ignore[arg-type]
            except ValueError:
                pass
            else:
                raise SystemExit(f"HTTP helper accepted invalid retries {retries!r}")
        if opener.called:
            raise SystemExit(f"Invalid retries {retries!r} reached urlopen")

    for retries in range(4):
        clock = FakeClock()
        with patch.object(http_module.urllib.request, "urlopen", return_value=FakeResponse(b"ok")):
            if http_get("https://example.test/valid", timeout=1.5, retries=retries, **_request_kwargs(clock)) != b"ok":
                raise SystemExit(f"HTTP helper rejected valid retry count {retries}")


def test_retryable_status_table() -> None:
    cases = [(301, False), (400, False), (404, False), (408, False), (429, True), (500, True), (503, True), (599, True), (600, False)]
    for status, should_retry in cases:
        clock = FakeClock()
        side_effect = [_http_error(status), FakeResponse(b"ok")]
        with patch.object(http_module.urllib.request, "urlopen", side_effect=side_effect) as opener:
            if should_retry:
                result = http_get("https://example.test/status", timeout=10, retries=1, **_request_kwargs(clock))
                if result != b"ok":
                    raise SystemExit(f"Retryable HTTP {status} did not recover")
            else:
                exc = _capture_http_error(
                    lambda: http_get("https://example.test/status", timeout=10, retries=1, **_request_kwargs(clock))
                )
                if exc.status != status or str(exc) != f"HTTP {status}":
                    raise SystemExit(f"HTTP {status} used an unsafe or incorrect error: {exc!r}")
        expected_calls = 2 if should_retry else 1
        expected_sleeps = [1.0] if should_retry else []
        if opener.call_count != expected_calls or clock.sleeps != expected_sleeps:
            raise SystemExit(f"HTTP {status} retry behavior wrong: calls={opener.call_count}, sleeps={clock.sleeps}")


def test_exact_retry_count_and_fallback_table() -> None:
    for retries in range(4):
        clock = FakeClock()
        failures = [OSError(f"network failure {index}") for index in range(retries + 1)]
        with patch.object(http_module.urllib.request, "urlopen", side_effect=failures) as opener:
            exc = _capture_http_error(
                lambda: http_get("https://example.test/retries", timeout=20, retries=retries, **_request_kwargs(clock))
            )
        if str(exc) != "network request failed" or exc.status is not None:
            raise SystemExit(f"Transport failure was not sanitized: {exc!r}")
        expected_sleeps = [1.0, 2.0, 4.0][:retries]
        if opener.call_count != retries + 1 or clock.sleeps != expected_sleeps:
            raise SystemExit(
                f"retries={retries} used calls={opener.call_count}, sleeps={clock.sleeps}; expected {retries + 1}, {expected_sleeps}"
            )


def test_retry_after_forms_and_caps_table() -> None:
    clock = FakeClock()
    future = email.utils.formatdate(clock.wall_time() + 4, usegmt=True)
    future_over_cap = email.utils.formatdate(clock.wall_time() + 6, usegmt=True)
    past = email.utils.formatdate(clock.wall_time() - 20, usegmt=True)
    cases = [
        ("delta", "3", 10, [3.0], True),
        ("whitespace delta", " 2 ", 10, [2.0], True),
        ("HTTP-date", future, 10, [4.0], True),
        ("past HTTP-date", past, 10, [0.0], True),
        ("malformed", "next Tuesday", 10, [1.0], True),
        ("negative malformed", "-1", 10, [1.0], True),
        ("exact cap", "5", 10, [5.0], True),
        ("over cap", "6", 10, [], False),
        ("HTTP-date over cap", future_over_cap, 10, [], False),
    ]

    for label, retry_after, timeout, expected_sleeps, should_retry in cases:
        case_clock = FakeClock()
        side_effect = [_http_error(429, retry_after=retry_after), FakeResponse(b"ok")]
        with patch.object(http_module.urllib.request, "urlopen", side_effect=side_effect) as opener:
            if should_retry:
                result = http_get("https://example.test/rate", timeout=timeout, retries=1, **_request_kwargs(case_clock))
                if result != b"ok":
                    raise SystemExit(f"{label} Retry-After did not recover")
            else:
                exc = _capture_http_error(
                    lambda: http_get("https://example.test/rate", timeout=timeout, retries=1, **_request_kwargs(case_clock))
                )
                if exc.status != 429:
                    raise SystemExit(f"{label} Retry-After lost HTTP status: {exc!r}")
        expected_calls = 2 if should_retry else 1
        if opener.call_count != expected_calls or case_clock.sleeps != expected_sleeps:
            raise SystemExit(
                f"{label} Retry-After behavior wrong: calls={opener.call_count}, sleeps={case_clock.sleeps}"
            )

    budget_clock = FakeClock()

    def consume_budget(*args, **kwargs):
        budget_clock.advance(3)
        raise _http_error(429, retry_after="3")

    with patch.object(http_module.urllib.request, "urlopen", side_effect=consume_budget) as opener:
        exc = _capture_http_error(
            lambda: http_get("https://example.test/budget", timeout=5, retries=1, **_request_kwargs(budget_clock))
        )
    if exc.status != 429 or opener.call_count != 1 or budget_clock.sleeps:
        raise SystemExit("Valid Retry-After exceeding the remaining budget must not be shortened or retried")


def test_overall_budget_shrinks_urlopen_timeout() -> None:
    clock = FakeClock()
    timeouts: list[float] = []
    calls = 0

    def timed_open(*args, **kwargs):
        nonlocal calls
        calls += 1
        timeouts.append(kwargs["timeout"])
        if calls < 3:
            clock.advance(2)
            raise OSError("temporary network failure")
        return FakeResponse(b"ok")

    with patch.object(http_module.urllib.request, "urlopen", side_effect=timed_open):
        result = http_get("https://example.test/shrinking", timeout=10, retries=3, **_request_kwargs(clock))
    if result != b"ok" or timeouts != [10.0, 7.0, 3.0] or clock.sleeps != [1.0, 2.0]:
        raise SystemExit(f"Overall deadline did not shrink request timeouts: timeouts={timeouts}, sleeps={clock.sleeps}")

    exhausted_clock = FakeClock()
    with patch.object(http_module.urllib.request, "urlopen", side_effect=OSError("temporary")) as opener:
        exc = _capture_http_error(
            lambda: http_get("https://example.test/no-room", timeout=1, retries=3, **_request_kwargs(exhausted_clock))
        )
    if str(exc) != "network request failed" or opener.call_count != 1 or exhausted_clock.sleeps:
        raise SystemExit("HTTP helper slept or retried when the fallback delay consumed the remaining budget")


def test_http_error_bodies_close_before_sleep_or_raise() -> None:
    retry_clock = FakeClock()
    retry_events = retry_clock.events
    unavailable = _http_error(503, events=retry_events)
    recovered = FakeResponse(b"ok", retry_events)

    def open_with_retry(*args, **kwargs):
        retry_events.append("request")
        if retry_events == ["request"]:
            raise unavailable
        return recovered

    with patch.object(http_module.urllib.request, "urlopen", side_effect=open_with_retry) as opener:
        if http_get("https://example.test/retry", timeout=10, retries=1, **_request_kwargs(retry_clock)) != b"ok":
            raise SystemExit("HTTP helper did not recover after closing a retryable error body")
    expected = ["request", "close", "sleep:1", "request", "response_close"]
    if retry_events != expected or opener.call_count != 2:
        raise SystemExit(f"Retry close/sleep ordering was wrong: {retry_events}")

    terminal_events: list[str] = []
    not_found = _http_error(404, events=terminal_events, hostile=True)

    def terminal_open(*args, **kwargs):
        terminal_events.append("request")
        raise not_found

    with patch.object(http_module.urllib.request, "urlopen", side_effect=terminal_open) as opener:
        exc = _capture_http_error(lambda: http_get("https://example.test/missing", retries=3))
        terminal_events.append("caught")
    if exc.status != 404 or terminal_events != ["request", "close", "caught"] or opener.call_count != 1:
        raise SystemExit(f"Terminal body was not closed before raising: {terminal_events}")


def test_known_failures_suppress_hostile_details_and_context() -> None:
    cases = [
        (_http_error(404, events=[], hostile=True), "HTTP 404", 404),
        (urllib.error.URLError("secret host /\x55sers/private/key"), "network request failed", None),
        (OSError("socket path /private/tmp/key"), "network request failed", None),
        (TimeoutError("timed out at /\x55sers/private/key"), "request timed out", None),
    ]
    forbidden = ["secret", "/\x55sers/private/", "/private/tmp/", "credential", "socket path"]

    for source, expected_message, expected_status in cases:
        clock = FakeClock()
        with patch.object(http_module.urllib.request, "urlopen", side_effect=source):
            exc = _capture_http_error(
                lambda: http_get("https://example.test/safe", retries=0, **_request_kwargs(clock))
            )
        rendered = "\n".join(
            (
                "".join(traceback.format_exception_only(exc)),
                repr(exc),
                repr(exc.args),
                repr(exc.__dict__),
            )
        )
        if str(exc) != expected_message or exc.status != expected_status:
            raise SystemExit(f"Known failure used the wrong fixed error: {exc!r}")
        if exc.__cause__ is not None or exc.__context__ is not None or not exc.__suppress_context__:
            raise SystemExit(f"Known failure retained exception context: cause={exc.__cause__!r}, context={exc.__context__!r}")
        if any(fragment in rendered for fragment in forbidden):
            raise SystemExit(f"Known failure leaked hostile source text: {rendered}")


def test_unexpected_programming_errors_propagate() -> None:
    programming_error = ValueError("unknown URL type")
    with patch.object(http_module.urllib.request, "urlopen", side_effect=programming_error) as opener:
        try:
            http_get("invalid://example.test", retries=3)
        except ValueError as exc:
            if exc is not programming_error:
                raise SystemExit("HTTP helper replaced an unexpected programming error")
        else:
            raise SystemExit("HTTP helper swallowed an unexpected programming error")
    if opener.call_count != 1:
        raise SystemExit("Unexpected programming errors should not be retried")

    clock = FakeClock()

    def broken_sleep(seconds: float) -> None:
        raise RuntimeError("sleep hook bug")

    with patch.object(http_module.urllib.request, "urlopen", side_effect=_http_error(503)):
        try:
            http_get(
                "https://example.test/sleep-bug",
                retries=1,
                _sleep=broken_sleep,
                _monotonic=clock.monotonic,
                _wall_time=clock.wall_time,
            )
        except RuntimeError as exc:
            if str(exc) != "sleep hook bug":
                raise
        else:
            raise SystemExit("HTTP helper swallowed an unexpected sleep-hook error")


def test_http_response_body_is_bounded_before_parsing() -> None:
    exact = FakeResponse(b"12345678")
    with patch.object(http_module, "MAX_HTTP_RESPONSE_BYTES", 8), patch.object(
        http_module.urllib.request, "urlopen", return_value=exact
    ) as opener:
        if http_get("https://example.test/exact") != b"12345678":
            raise SystemExit("HTTP helper rejected a response exactly at its byte ceiling")
    if exact.read_sizes or exact.read1_sizes != [9, 1] or opener.call_count != 1:
        raise SystemExit(
            f"HTTP helper did not prefer bounded read1 calls through EOF: "
            f"read={exact.read_sizes}, read1={exact.read1_sizes}"
        )

    oversized = FakeResponse(b"123456789")
    with patch.object(http_module, "MAX_HTTP_RESPONSE_BYTES", 8), patch.object(
        http_module.urllib.request, "urlopen", return_value=oversized
    ) as opener:
        exc = _capture_http_error(lambda: http_get("https://example.test/oversized", retries=3))
    if exc.status is not None or str(exc) != "response too large":
        raise SystemExit(f"Oversized HTTP response used the wrong fixed diagnostic: {exc!r}")
    if oversized.read_sizes or oversized.read1_sizes != [9] or opener.call_count != 1:
        raise SystemExit("Oversized HTTP response should fail immediately without retries")


def test_http_response_short_reads_continue_until_eof() -> None:
    response = FakeResponse(b"short-read-body", max_chunk=2)
    with patch.object(http_module, "MAX_HTTP_RESPONSE_BYTES", 32), patch.object(
        http_module.urllib.request, "urlopen", return_value=response
    ):
        payload = http_get("https://example.test/short-read")
    if payload != b"short-read-body":
        raise SystemExit(f"HTTP helper treated a short read as EOF: {payload!r}")
    if response.read_sizes or len(response.read1_sizes) < 3:
        raise SystemExit(
            f"HTTP helper did not continue with bounded read1 calls: {response.read1_sizes}"
        )


def test_http_response_trickle_obeys_overall_deadline() -> None:
    clock = FakeClock()

    class TrickleResponse(FakeResponse):
        def read1(self, size: int = -1) -> bytes:
            clock.advance(0.6)
            return super().read1(size)

    response = TrickleResponse(b"abc", max_chunk=1)
    with patch.object(http_module.urllib.request, "urlopen", return_value=response):
        exc = _capture_http_error(
            lambda: http_get(
                "https://example.test/trickle",
                timeout=1.0,
                retries=3,
                **_request_kwargs(clock),
            )
        )
    if str(exc) != "request timed out" or exc.status is not None:
        raise SystemExit(f"Cooperative trickle used the wrong deadline failure: {exc!r}")
    if len(response.read1_sizes) != 2 or response.offset != 2:
        raise SystemExit(
            f"HTTP helper did not stop immediately after the deadline-crossing chunk: "
            f"reads={response.read1_sizes}, offset={response.offset}"
        )


def test_http_get_json_forwards_all_options() -> None:
    sleep = lambda seconds: None
    monotonic = lambda: 1.0
    wall_time = lambda: 2.0
    kwargs = {
        "timeout": 9,
        "headers": {"Accept": "application/json"},
        "retries": 3,
        "_sleep": sleep,
        "_monotonic": monotonic,
        "_wall_time": wall_time,
    }
    with patch.object(http_module, "http_get", return_value=b'{"ok": true}') as getter:
        result = http_get_json("https://example.test/data", **kwargs)
    if result != {"ok": True}:
        raise SystemExit(f"http_get_json parsed the wrong result: {result!r}")
    if getter.call_args.args != ("https://example.test/data",) or getter.call_args.kwargs != kwargs:
        raise SystemExit(f"http_get_json did not forward all options exactly: {getter.call_args!r}")


def main() -> None:
    test_status_errors_are_friendly()
    test_string_only_http_errors_are_friendly()
    test_timeout_and_generic_errors_are_friendly()
    test_option_validation_table()
    test_retryable_status_table()
    test_exact_retry_count_and_fallback_table()
    test_retry_after_forms_and_caps_table()
    test_overall_budget_shrinks_urlopen_timeout()
    test_http_error_bodies_close_before_sleep_or_raise()
    test_known_failures_suppress_hostile_details_and_context()
    test_unexpected_programming_errors_propagate()
    test_http_response_body_is_bounded_before_parsing()
    test_http_response_short_reads_continue_until_eof()
    test_http_response_trickle_obeys_overall_deadline()
    test_http_get_json_forwards_all_options()
    print("HTTP helper smoke passed")


if __name__ == "__main__":
    main()
