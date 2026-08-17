"""Shared bounded HTTP helper for Jarvis V2 connectors."""

from __future__ import annotations

import email.utils
import json
import math
import time
import urllib.error
import urllib.request
from datetime import timezone
from typing import Any, Callable

DEFAULT_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)"
DEFAULT_TIMEOUT = 12
MAX_HTTP_RESPONSE_BYTES = 5_000_000
MAX_RETRIES = 3
MAX_RETRY_WAIT_SECONDS = 5.0
_FALLBACK_RETRY_DELAYS = (1.0, 2.0, 4.0)
_RESPONSE_READ_CHUNK_BYTES = 64 * 1024


class HttpError(Exception):
    def __init__(self, status: int | None, message: str = ""):
        self.status = status
        super().__init__(message or (f"HTTP {status}" if status else "HTTP error"))


def _validate_options(timeout: float, retries: int) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a finite positive number")
    timeout_value = float(timeout)
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ValueError("timeout must be a finite positive number")
    if type(retries) is not int or not 0 <= retries <= MAX_RETRIES:
        raise ValueError(f"retries must be an integer from 0 to {MAX_RETRIES}")
    return timeout_value


def _retry_after_seconds(value: object, *, wall_time: Callable[[], float]) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.isascii() and text.isdigit():
        return float(text)
    try:
        parsed = email.utils.parsedate_to_datetime(text)
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        retry_at = parsed.timestamp()
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    delay = retry_at - wall_time()
    if not math.isfinite(delay):
        return None
    return max(0.0, delay)


def _http_failure(error: urllib.error.HTTPError) -> tuple[HttpError, bool, object]:
    status = error.code if type(error.code) is int else None
    retryable = status == 429 or (status is not None and 500 <= status < 600)
    retry_after = error.headers.get("Retry-After") if error.headers is not None else None
    return HttpError(status), retryable, retry_after


def _transport_failure(error: Exception) -> HttpError:
    timed_out = isinstance(error, TimeoutError)
    if isinstance(error, urllib.error.URLError):
        timed_out = timed_out or isinstance(error.reason, TimeoutError)
    return HttpError(None, "request timed out" if timed_out else "network request failed")


def _close_http_error(error: urllib.error.HTTPError) -> None:
    try:
        if error.fp is not None:
            error.fp.close()
    except Exception:
        pass


def _read_response_body(
    response: object,
    *,
    deadline: float,
    monotonic: Callable[[], float],
) -> bytes:
    read = getattr(response, "read1", None)
    if not callable(read):
        read = getattr(response, "read")

    chunks: list[bytes] = []
    total = 0
    while total <= MAX_HTTP_RESPONSE_BYTES:
        if monotonic() >= deadline:
            raise HttpError(None, "request timed out")
        amount = min(
            _RESPONSE_READ_CHUNK_BYTES,
            MAX_HTTP_RESPONSE_BYTES + 1 - total,
        )
        chunk = read(amount)
        if monotonic() >= deadline:
            raise HttpError(None, "request timed out")
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_HTTP_RESPONSE_BYTES:
            raise HttpError(None, "response too large")

    if monotonic() >= deadline:
        raise HttpError(None, "request timed out")
    return b"".join(chunks)


def http_get(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict | None = None,
    retries: int = 1,
    _sleep: Callable[[float], None] = time.sleep,
    _monotonic: Callable[[], float] = time.monotonic,
    _wall_time: Callable[[], float] = time.time,
) -> bytes:
    timeout_value = _validate_options(timeout, retries)
    hdrs = {"User-Agent": DEFAULT_UA}
    if headers:
        hdrs.update(headers)
    deadline = _monotonic() + timeout_value

    for attempt in range(retries + 1):
        failure: HttpError | None = None
        retryable = False
        retry_after_value: object = None

        remaining = deadline - _monotonic()
        if remaining <= 0:
            raise HttpError(None, "request timed out") from None
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=remaining) as resp:
                return _read_response_body(
                    resp,
                    deadline=deadline,
                    monotonic=_monotonic,
                )
        except urllib.error.HTTPError as error:
            failure, retryable, retry_after_value = _http_failure(error)
            _close_http_error(error)
        except HttpError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            failure = _transport_failure(error)
            retryable = True
        except Exception:
            raise

        if failure is None:
            raise RuntimeError("HTTP helper reached an invalid failure state")
        if not retryable or attempt >= retries:
            raise failure from None

        remaining = deadline - _monotonic()
        retry_after = _retry_after_seconds(retry_after_value, wall_time=_wall_time)
        wait = _FALLBACK_RETRY_DELAYS[attempt] if retry_after is None else retry_after
        if wait > MAX_RETRY_WAIT_SECONDS or wait >= remaining:
            raise failure from None
        _sleep(wait)

    raise RuntimeError("HTTP helper exhausted an unreachable retry loop")


def http_get_json(url: str, **kwargs: Any) -> Any:
    return json.loads(http_get(url, **kwargs).decode("utf-8"))


def friendly_http_error(e: Exception, *, subject: str = "that", service: str = "the service") -> str:
    status = getattr(e, "status", None)
    msg = str(e).lower()
    if status == 404:
        return f"I couldn't find {subject}."
    if status is not None and 500 <= status < 600:
        return f"{service} is having trouble right now — try again in a moment."
    if "404" in msg:
        return f"I couldn't find {subject}."
    if "http error 5" in msg or "http 5" in msg:
        return f"{service} is having trouble right now — try again in a moment."
    if "timed out" in msg or "timeout" in msg:
        return "The request timed out — try again."
    return f"{service} is having trouble right now — try again in a moment."
