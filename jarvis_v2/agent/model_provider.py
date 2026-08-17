from __future__ import annotations

import json
import ipaddress
import math
import os
import queue
import re
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_SINGLE_OWNER_SAFETY_IDENTIFIER = "jarvis-v3-single-owner"
OPENAI_SAFETY_IDENTIFIER_MAX_CHARS = 64
MAX_OPENAI_REQUEST_BYTES = 1_000_000
MAX_OPENAI_RESPONSE_BYTES = 1_000_000
OPENAI_BLOCKING_WORKER_LIMIT = 4
MAX_REPORTED_MODEL_TOKENS = 1_000_000_000
SUPPORTED_MODEL_PROVIDERS = {"ollama", "openai"}
SUPPORTED_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
FALSE_ENV_VALUES = {"0", "false", "no", "off"}
KNOWN_OLLAMA_CLOUD_ALIASES = {"kimi-k2-thinking"}
OLLAMA_UNVERIFIED_PERSONAL_CONTEXT_ENV = (
    "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT"
)
MAX_OLLAMA_PROBE_RESPONSE_BYTES = 1_000_000
MAX_OLLAMA_REQUEST_BYTES = 1_000_000
MAX_OLLAMA_RESPONSE_BYTES = 1_000_000
OLLAMA_BLOCKING_WORKER_LIMIT = 4
# Ollama falls back to a small per-model default (often 4096 tokens) unless a
# request sets num_ctx explicitly. The planner prompt (tool registry + system
# prompt) measured 15,502 tokens against the live registry; benchmarking
# showed models silently truncated or outright rejecting requests at the
# default, and accuracy roughly doubled once num_ctx covered the real prompt.
OLLAMA_NUM_CTX = 24_000
# Keep local generation comfortably below the requested context window. The
# daemon does not attest that it honored num_ctx or whether it truncated the
# prompt, so this is an output ceiling rather than a compatibility claim.
OLLAMA_MAX_OUTPUT_TOKENS = 4_096
SAFE_OLLAMA_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,239}")
SAFE_OLLAMA_KEEP_ALIVE_RE = re.compile(r"(?:0|[1-9][0-9]{0,8})(?:ms|s|m|h)?")


_OPENAI_BLOCKING_WORKER_SLOTS = threading.BoundedSemaphore(OPENAI_BLOCKING_WORKER_LIMIT)
_OLLAMA_BLOCKING_WORKER_SLOTS = threading.BoundedSemaphore(OLLAMA_BLOCKING_WORKER_LIMIT)


@dataclass
class ModelProviderError(Exception):
    diagnostic: str
    recovery_hint: str

    def __str__(self) -> str:
        return self.diagnostic


@dataclass(frozen=True)
class OllamaDestination:
    allowed: bool
    base_url: str
    configured: bool
    source: str
    scheme: str
    address_family: str
    port: int
    diagnostic: str

    def receipt(self) -> dict[str, object]:
        return {
            "ollama_destination_allowed": self.allowed,
            "ollama_destination_configured": self.configured,
            "ollama_destination_source": self.source,
            "ollama_destination_scheme": self.scheme,
            "ollama_destination_address_family": self.address_family,
            "ollama_destination_port": self.port,
            "ollama_destination_diagnostic": self.diagnostic,
            "ollama_destination_value_exposed": False,
            "ollama_redirects_allowed": False,
            "ollama_proxy_environment_allowed": False,
        }


@dataclass(frozen=True)
class OllamaLocalOnlyPolicy:
    configured: bool
    valid: bool
    requested: bool
    cloud_model_alias: bool
    personal_context_allowed: bool
    diagnostic: str
    unverified_context_consent_configured: bool
    unverified_context_consent_valid: bool
    unverified_context_consent_allowed: bool

    def receipt(self) -> dict[str, object]:
        return {
            "ollama_no_cloud_configured": self.configured,
            "ollama_no_cloud_valid": self.valid,
            "ollama_no_cloud_requested": self.requested,
            "ollama_cloud_model_alias": self.cloud_model_alias,
            "ollama_personal_context_allowed": self.personal_context_allowed,
            "ollama_unverified_context_consent_configured": self.unverified_context_consent_configured,
            "ollama_unverified_context_consent_valid": self.unverified_context_consent_valid,
            "ollama_unverified_context_consent_allowed": self.unverified_context_consent_allowed,
            "ollama_unverified_context_consent_env": OLLAMA_UNVERIFIED_PERSONAL_CONTEXT_ENV,
            "ollama_execution_locality_verified": False,
            "ollama_cloud_policy_diagnostic": self.diagnostic,
            "ollama_cloud_policy_value_exposed": False,
        }


def ollama_local_only_policy(
    model: object = "",
    environ: dict[str, str] | None = None,
) -> OllamaLocalOnlyPolicy:
    source = os.environ if environ is None else environ
    raw = str(source.get("OLLAMA_NO_CLOUD") or "").strip().casefold()
    configured = bool(raw)
    valid = not configured or raw in TRUE_ENV_VALUES or raw in FALSE_ENV_VALUES
    requested = bool(valid and raw in TRUE_ENV_VALUES)
    consent_raw = str(
        source.get(OLLAMA_UNVERIFIED_PERSONAL_CONTEXT_ENV) or ""
    ).strip().casefold()
    consent_configured = bool(consent_raw)
    consent_valid = bool(
        not consent_configured
        or consent_raw in TRUE_ENV_VALUES
        or consent_raw in FALSE_ENV_VALUES
    )
    consent_allowed = bool(consent_valid and consent_raw in TRUE_ENV_VALUES)
    normalized_model = str(model or "").strip().casefold()
    model_basename = normalized_model.rsplit("/", 1)[-1].split(":", 1)[0]
    cloud_model_alias = bool(
        model_basename in KNOWN_OLLAMA_CLOUD_ALIASES
        or re.search(r"(?:^|[:_-])cloud(?:$|[:_-])", normalized_model)
    )
    if not valid:
        diagnostic = "invalid_no_cloud_setting"
    elif not consent_valid:
        diagnostic = "invalid_unverified_context_consent"
    elif cloud_model_alias:
        diagnostic = "cloud_model_alias_blocked"
    elif requested:
        diagnostic = (
            "unverified_local_only_consent_granted"
            if consent_allowed
            else "unverified_daemon_consent_required"
        )
    else:
        diagnostic = "cloud_mode_not_disabled"
    return OllamaLocalOnlyPolicy(
        configured,
        valid,
        requested,
        cloud_model_alias,
        bool(requested and consent_allowed and not cloud_model_alias),
        diagnostic,
        consent_configured,
        consent_valid,
        consent_allowed,
    )


def resolve_ollama_destination(
    environ: dict[str, str] | None = None,
) -> OllamaDestination:
    source = os.environ if environ is None else environ
    raw_value = source.get("OLLAMA_HOST")
    configured = raw_value is not None and bool(str(raw_value).strip())
    if not configured:
        return OllamaDestination(
            True,
            "http://127.0.0.1:11434",
            False,
            "default",
            "http",
            "ipv4_loopback",
            11434,
            "local_loopback",
        )

    raw = str(raw_value)
    if (
        raw != raw.strip()
        or len(raw) > 300
        or any(character.isspace() or ord(character) < 32 for character in raw)
        or "\\" in raw
    ):
        return OllamaDestination(False, "", True, "env-invalid", "", "", 0, "invalid_format")

    explicit_scheme = "://" in raw
    candidate = raw
    if not explicit_scheme:
        if raw.startswith(":"):
            candidate = "127.0.0.1" + raw
        candidate = "http://" + candidate
    try:
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.casefold()
        port = parsed.port
    except (TypeError, ValueError):
        return OllamaDestination(False, "", True, "env-invalid", "", "", 0, "invalid_format")
    if scheme not in {"http", "https"}:
        return OllamaDestination(False, "", True, "env-invalid", scheme, "", 0, "unsupported_scheme")
    if parsed.netloc.endswith(":"):
        return OllamaDestination(False, "", True, "env-invalid", scheme, "", 0, "invalid_port")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        return OllamaDestination(False, "", True, "env-invalid", scheme, "", 0, "userinfo_not_allowed")
    if parsed.query or parsed.fragment:
        return OllamaDestination(False, "", True, "env-invalid", scheme, "", 0, "query_fragment_not_allowed")
    if parsed.path not in {"", "/"}:
        return OllamaDestination(False, "", True, "env-invalid", scheme, "", 0, "path_not_allowed")

    host = (parsed.hostname or "").casefold()
    if not host:
        return OllamaDestination(False, "", True, "env-invalid", scheme, "", 0, "host_missing")
    if "%" in host:
        return OllamaDestination(False, "", True, "env-invalid", scheme, "", 0, "scoped_address_not_allowed")
    if host == "localhost":
        normalized_host = "127.0.0.1"
        address_family = "localhost_pinned_ipv4"
    else:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return OllamaDestination(False, "", True, "env-invalid", scheme, "", 0, "non_loopback_host")
        if not address.is_loopback:
            return OllamaDestination(False, "", True, "env-invalid", scheme, "", 0, "non_loopback_address")
        if isinstance(address, ipaddress.IPv6Address):
            if address.ipv4_mapped is not None:
                return OllamaDestination(False, "", True, "env-invalid", scheme, "", 0, "mapped_address_not_allowed")
            normalized_host = f"[{address.compressed}]"
            address_family = "ipv6_loopback"
        else:
            normalized_host = str(address)
            address_family = "ipv4_loopback"

    if port is None:
        port = (443 if scheme == "https" else 80) if explicit_scheme else 11434
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        return OllamaDestination(False, "", True, "env-invalid", scheme, address_family, 0, "invalid_port")
    return OllamaDestination(
        True,
        f"{scheme}://{normalized_host}:{port}",
        True,
        "env",
        scheme,
        address_family,
        port,
        "local_loopback",
    )


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe_ollama_models(
    environ: dict[str, str] | None = None,
    *,
    timeout_seconds: float = 5.0,
    opener: object | None = None,
) -> tuple[bool, list[str], str, str]:
    """Read model names from a validated loopback daemon without redirects or proxies."""
    destination = resolve_ollama_destination(environ)
    if not destination.allowed:
        return False, [], "ollama_destination_not_local", "OllamaDestinationBlocked"
    request = Request(
        destination.base_url.rstrip("/") + "/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    client = opener or build_opener(ProxyHandler({}), _NoRedirectHandler())
    try:
        response = client.open(request, timeout=max(0.1, float(timeout_seconds)))
        with response:
            raw = response.read(MAX_OLLAMA_PROBE_RESPONSE_BYTES + 1)
        if len(raw) > MAX_OLLAMA_PROBE_RESPONSE_BYTES:
            return False, [], "ollama_probe_response_too_large", "ResponseTooLarge"
        payload = json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        diagnostic = (
            "ollama_probe_redirect_blocked"
            if 300 <= int(getattr(exc, "code", 0) or 0) < 400
            else "ollama_probe_http_error"
        )
        return False, [], diagnostic, type(exc).__name__
    except Exception as exc:
        return False, [], "ollama_probe_unavailable", type(exc).__name__
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return False, [], "ollama_probe_invalid_response", "InvalidResponse"
    models: list[str] = []
    for item in payload["models"][:1000]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if name and SAFE_OLLAMA_MODEL_NAME_RE.fullmatch(name):
            models.append(name)
    return True, models, "ollama_probe_ok", ""


def normalized_model_provider(value: object) -> str:
    provider = str(value or "").strip().lower()
    if not provider:
        return "ollama"
    return provider if provider in SUPPORTED_MODEL_PROVIDERS else "invalid"


def normalized_reasoning_effort(value: object, default: str = "medium") -> str:
    effort = str(value or "").strip().lower()
    if effort in SUPPORTED_REASONING_EFFORTS:
        return effort
    return default if default in SUPPORTED_REASONING_EFFORTS else "medium"


def openai_api_key_configured(environ: dict[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    return bool(str(source.get("OPENAI_API_KEY") or "").strip())


def provider_output_token_limit(
    provider: object,
    *,
    local_output_tokens: int,
    openai_output_tokens: int,
) -> int:
    """Return the correct token ceiling for local output or remote reasoning + output."""
    selected = normalized_model_provider(provider)
    configured = openai_output_tokens if selected == "openai" else local_output_tokens
    try:
        return max(1, int(configured))
    except (TypeError, ValueError):
        return 1


def safe_model_usage_receipt(value: object, provider: object) -> dict[str, object]:
    selected = normalized_model_provider(provider)
    ollama_context: dict[str, object] = {}
    if selected == "ollama":
        requested_num_ctx = (
            value.get("ollama_num_ctx_requested")
            if isinstance(value, dict)
            and value.get("ollama_num_ctx_requested_available") is True
            else None
        )
        requested_available = bool(
            type(requested_num_ctx) is int
            and requested_num_ctx == OLLAMA_NUM_CTX
        )
        ollama_context = {
            "ollama_num_ctx_requested": (
                requested_num_ctx if requested_available else None
            ),
            "ollama_num_ctx_requested_available": requested_available,
            "ollama_num_ctx_effective": None,
            "ollama_num_ctx_effective_available": False,
            "ollama_prompt_truncated": None,
            "ollama_prompt_truncation_verified": False,
            "ollama_context_compatibility_verified": False,
        }
    empty = {
        "model_usage_provider": selected,
        "model_usage_available": False,
        "model_usage_consistent": False,
        "model_input_tokens": 0,
        "model_cached_input_tokens": 0,
        "model_cached_input_tokens_available": False,
        "model_output_tokens": 0,
        "model_reasoning_tokens": 0,
        "model_reasoning_tokens_available": False,
        "model_total_tokens": 0,
        "model_usage_content_recorded": False,
        **ollama_context,
    }
    if not isinstance(value, dict) or value.get("model_usage_available") is not True:
        return empty

    input_tokens = _safe_token_count(value.get("model_input_tokens"))
    output_tokens = _safe_token_count(value.get("model_output_tokens"))
    total_tokens = _safe_token_count(value.get("model_total_tokens"))
    if (
        input_tokens is None
        or output_tokens is None
        or total_tokens is None
        or total_tokens != input_tokens + output_tokens
    ):
        return empty

    cached_available = value.get("model_cached_input_tokens_available") is True
    cached_tokens = _safe_token_count(value.get("model_cached_input_tokens")) if cached_available else 0
    reasoning_available = value.get("model_reasoning_tokens_available") is True
    reasoning_tokens = _safe_token_count(value.get("model_reasoning_tokens")) if reasoning_available else 0
    if (
        cached_tokens is None
        or reasoning_tokens is None
        or cached_tokens > input_tokens
        or reasoning_tokens > output_tokens
    ):
        return empty

    return {
        **empty,
        "model_usage_available": True,
        "model_usage_consistent": True,
        "model_input_tokens": input_tokens,
        "model_cached_input_tokens": cached_tokens,
        "model_cached_input_tokens_available": cached_available,
        "model_output_tokens": output_tokens,
        "model_reasoning_tokens": reasoning_tokens,
        "model_reasoning_tokens_available": reasoning_available,
        "model_total_tokens": total_tokens,
    }


def _safe_token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > MAX_REPORTED_MODEL_TOKENS:
        return None
    return value


def _set_model_usage_receipt(
    usage_metadata: dict[str, object] | None,
    receipt: object,
    provider: object,
) -> None:
    if usage_metadata is None:
        return
    usage_metadata.clear()
    usage_metadata.update(safe_model_usage_receipt(receipt, provider))


def _openai_usage_receipt(data: object) -> dict[str, object]:
    if not isinstance(data, dict):
        return {}
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return {}
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    cached_available = isinstance(input_details, dict) and "cached_tokens" in input_details
    reasoning_available = isinstance(output_details, dict) and "reasoning_tokens" in output_details
    return {
        "model_usage_available": True,
        "model_input_tokens": usage.get("input_tokens"),
        "model_cached_input_tokens": input_details.get("cached_tokens") if isinstance(input_details, dict) else 0,
        "model_cached_input_tokens_available": cached_available,
        "model_output_tokens": usage.get("output_tokens"),
        "model_reasoning_tokens": output_details.get("reasoning_tokens") if isinstance(output_details, dict) else 0,
        "model_reasoning_tokens_available": reasoning_available,
        "model_total_tokens": usage.get("total_tokens"),
    }


def _ollama_usage_receipt(data: object) -> dict[str, object]:
    context_receipt = _ollama_context_receipt()
    if not isinstance(data, dict):
        return context_receipt
    prompt_tokens = data.get("prompt_eval_count")
    output_tokens = data.get("eval_count")
    if _safe_token_count(prompt_tokens) is None or _safe_token_count(output_tokens) is None:
        return context_receipt
    return {
        **context_receipt,
        "model_usage_available": True,
        "model_input_tokens": prompt_tokens,
        "model_cached_input_tokens": 0,
        "model_cached_input_tokens_available": False,
        "model_output_tokens": output_tokens,
        "model_reasoning_tokens": 0,
        "model_reasoning_tokens_available": False,
        "model_total_tokens": prompt_tokens + output_tokens,
    }


def _ollama_context_receipt() -> dict[str, object]:
    """Report the request while keeping daemon-effective context truth unknown."""
    return {
        "ollama_num_ctx_requested": OLLAMA_NUM_CTX,
        "ollama_num_ctx_requested_available": True,
        "ollama_num_ctx_effective": None,
        "ollama_num_ctx_effective_available": False,
        "ollama_prompt_truncated": None,
        "ollama_prompt_truncation_verified": False,
        "ollama_context_compatibility_verified": False,
    }


def generate_model_text(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    reasoning_effort: str = "medium",
    keep_alive: str = "30m",
    environ: dict[str, str] | None = None,
    urlopen_impl: Callable[..., Any] | None = None,
    usage_metadata: dict[str, object] | None = None,
) -> str:
    selected = normalized_model_provider(provider)
    _set_model_usage_receipt(usage_metadata, {}, selected)
    if selected not in SUPPORTED_MODEL_PROVIDERS:
        raise ModelProviderError(
            "model_provider_invalid",
            "Set JARVIS_MODEL_PROVIDER to ollama or openai, then retry `model routing status`.",
        )
    if selected == "openai":
        return _openai_response_text(
            model=model,
            messages=messages,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            environ=environ,
            urlopen_impl=urlopen_impl,
            usage_metadata=usage_metadata,
        )

    destination = resolve_ollama_destination(environ)
    if not destination.allowed:
        raise ModelProviderError(
            "ollama_destination_not_local",
            "Set OLLAMA_HOST to a numeric loopback address such as "
            "http://127.0.0.1:11434, then retry `model routing status`.",
        )
    local_only_policy = ollama_local_only_policy(model, environ)
    if local_only_policy.cloud_model_alias:
        raise ModelProviderError(
            "ollama_cloud_model_blocked",
            "Choose an on-device Ollama model and configure the Ollama daemon with "
            "OLLAMA_NO_CLOUD=1 before retrying `model routing status`.",
        )

    return _ollama_response_text(
        destination=destination,
        model=model,
        messages=messages,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        keep_alive=keep_alive,
        urlopen_impl=urlopen_impl,
        usage_metadata=usage_metadata,
    )


def _ollama_response_text(
    *,
    destination: OllamaDestination,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
    max_output_tokens: int | None,
    temperature: float | None,
    keep_alive: str,
    urlopen_impl: Callable[..., Any] | None,
    usage_metadata: dict[str, object] | None,
) -> str:
    payload = _ollama_request_payload(
        model=model,
        messages=messages,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        keep_alive=keep_alive,
    )
    _set_model_usage_receipt(
        usage_metadata,
        _ollama_context_receipt(),
        "ollama",
    )
    try:
        encoded_payload = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ModelProviderError(
            "ollama_invalid_request",
            "The Ollama request is malformed. Check the configured model and retry `model routing status`.",
        ) from None
    if len(encoded_payload) > MAX_OLLAMA_REQUEST_BYTES:
        raise ModelProviderError(
            "ollama_request_too_large",
            "The Ollama request is too large. Shorten the input or reduce "
            "JARVIS_CHAT_MAX_HISTORY_MESSAGES, then retry.",
        )

    request = Request(
        destination.base_url.rstrip("/") + "/api/chat",
        data=encoded_payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Connection": "close",
        },
        method="POST",
    )
    opener = (
        build_opener(ProxyHandler({}), _NoRedirectHandler()).open
        if urlopen_impl is None
        else urlopen_impl
    )
    request_timeout = _safe_request_timeout(timeout_seconds)
    deadline = time.monotonic() + request_timeout
    try:
        raw_response = _bounded_ollama_read(opener, request, request_timeout)
        if len(raw_response) > MAX_OLLAMA_RESPONSE_BYTES:
            raise ModelProviderError(
                "ollama_response_too_large",
                "Ollama returned an unexpectedly large response. Lower the output token limit, "
                "then retry `model routing status` if it persists.",
            )
        data = json.loads(raw_response.decode("utf-8"))
        if time.monotonic() > deadline:
            raise TimeoutError
    except HTTPError as exc:
        try:
            exc.close()
        except Exception:
            pass
        raise _ollama_http_error(int(getattr(exc, "code", 0) or 0)) from None
    except (TimeoutError, socket.timeout):
        raise ModelProviderError(
            "ollama_timeout",
            "Ollama took too long to respond. Try again, or raise the configured model timeout before retrying.",
        ) from None
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            diagnostic = "ollama_timeout"
            recovery = (
                "Ollama took too long to respond. Try again, or raise the configured model timeout before retrying."
            )
        elif isinstance(reason, ssl.SSLError):
            diagnostic = "ollama_tls_failed"
            recovery = (
                "The loopback Ollama TLS connection could not be verified. Correct its certificate configuration; "
                "do not disable certificate verification."
            )
        else:
            diagnostic = "ollama_unreachable"
            recovery = "Ollama does not appear to be running. Start it, then run `model status` in Jarvis."
        raise ModelProviderError(diagnostic, recovery) from None
    except OSError:
        raise ModelProviderError(
            "ollama_stream_failed",
            "The Ollama response stream failed. Check the local daemon, then retry `model routing status`.",
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ModelProviderError(
            "ollama_invalid_response",
            "Ollama returned an unreadable response. Retry once, then run `model routing status` if it persists.",
        ) from None

    if not isinstance(data, dict):
        raise ModelProviderError(
            "ollama_invalid_response",
            "Ollama returned an unreadable response. Retry once, then run `model routing status` if it persists.",
        )
    _set_model_usage_receipt(usage_metadata, _ollama_usage_receipt(data), "ollama")
    response_error = data.get("error")
    if response_error is not None and response_error != "":
        raise ModelProviderError(
            "ollama_request_rejected",
            "Ollama rejected the request. Run `model status` in Jarvis for details.",
        )
    if data.get("done") is not True:
        raise ModelProviderError(
            "ollama_output_incomplete",
            "Ollama returned an incomplete response. Retry once; no partial answer was used.",
        )
    done_reason = data.get("done_reason")
    if done_reason == "length":
        raise ModelProviderError(
            "ollama_output_incomplete_max_tokens",
            "Ollama reached the output token limit before finishing. Ask for a shorter answer or raise the local output limit.",
        )
    if done_reason is not None and not isinstance(done_reason, str):
        raise ModelProviderError(
            "ollama_invalid_response",
            "Ollama returned an unreadable completion state. Retry once, then run `model routing status` if it persists.",
        )
    message = data.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ModelProviderError(
            "ollama_invalid_response",
            "Ollama returned an unreadable response. Retry once, then run `model routing status` if it persists.",
        )
    text = message["content"].strip()
    if not text:
        raise ModelProviderError(
            "ollama_empty_response",
            "Ollama returned no answer text. Retry once, then run `model routing status` if it persists.",
        )
    return text


def _ollama_request_payload(
    *,
    model: object,
    messages: object,
    max_output_tokens: object,
    temperature: object,
    keep_alive: object,
) -> dict[str, object]:
    if type(model) is not str or SAFE_OLLAMA_MODEL_NAME_RE.fullmatch(model) is None:
        raise ModelProviderError(
            "ollama_invalid_request",
            "The configured Ollama model name is invalid. Choose a local model, then retry `model routing status`.",
        )
    if type(messages) is not list:
        raise ModelProviderError(
            "ollama_invalid_request",
            "The Ollama message request is malformed. Retry the request without changing any approval state.",
        )
    normalized_messages: list[dict[str, str]] = []
    for item in messages:
        if (
            type(item) is not dict
            or type(item.get("role")) is not str
            or not item["role"]
            or type(item.get("content")) is not str
        ):
            raise ModelProviderError(
                "ollama_invalid_request",
                "The Ollama message request is malformed. Retry the request without changing any approval state.",
            )
        normalized_messages.append({"role": item["role"], "content": item["content"]})
    if type(keep_alive) is not str or SAFE_OLLAMA_KEEP_ALIVE_RE.fullmatch(keep_alive) is None:
        raise ModelProviderError(
            "ollama_invalid_request",
            "The Ollama keep-alive setting is invalid. Correct the local model configuration, then retry.",
        )
    options: dict[str, object] = {}
    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
        ):
            raise ModelProviderError(
                "ollama_invalid_request",
                "The Ollama temperature setting is invalid. Correct the local model configuration, then retry.",
            )
        options["temperature"] = temperature
    if max_output_tokens is not None:
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 1 <= max_output_tokens <= MAX_REPORTED_MODEL_TOKENS
        ):
            raise ModelProviderError(
                "ollama_invalid_request",
                "The Ollama output token limit is invalid. Correct the local model configuration, then retry.",
            )
        if (
            max_output_tokens > OLLAMA_MAX_OUTPUT_TOKENS
            or max_output_tokens >= OLLAMA_NUM_CTX
        ):
            raise ModelProviderError(
                "ollama_output_token_limit_exceeded",
                f"Reduce the Ollama output token limit to {OLLAMA_MAX_OUTPUT_TOKENS} or fewer, then retry.",
            )
        options["num_predict"] = max_output_tokens
    options["num_ctx"] = OLLAMA_NUM_CTX
    return {
        "model": model,
        "messages": normalized_messages,
        "stream": False,
        "options": options,
        "keep_alive": keep_alive,
    }


def _safe_request_timeout(value: object) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        timeout = 0.1
    if not math.isfinite(timeout):
        timeout = 0.1
    return max(0.1, timeout)


def _ollama_http_error(status: int) -> ModelProviderError:
    if 300 <= status <= 399:
        return ModelProviderError(
            "ollama_redirect_blocked",
            "Ollama attempted to redirect the local model request. Check OLLAMA_HOST and the daemon configuration.",
        )
    if status in {401, 403}:
        return ModelProviderError(
            "ollama_access_denied",
            "The local Ollama daemon denied access. Check its local access policy, then retry `model routing status`.",
        )
    if status == 404:
        return ModelProviderError(
            "ollama_model_unavailable",
            "The configured Ollama model is not pulled. Pull it outside Jarvis, then retry `model routing status`.",
        )
    if status in {408, 504}:
        return ModelProviderError(
            "ollama_timeout",
            "Ollama took too long to respond. Try again, or raise the configured model timeout before retrying.",
        )
    if status == 429:
        return ModelProviderError(
            "ollama_busy",
            "Ollama is temporarily busy. Wait for existing local model work to finish, then retry.",
        )
    if status in {400, 422}:
        return ModelProviderError(
            "ollama_request_rejected",
            "Ollama rejected the request. Run `model status` in Jarvis for details.",
        )
    if 500 <= status <= 599:
        return ModelProviderError(
            "ollama_service_error",
            "The local Ollama daemon could not complete the request. Retry later; no Jarvis tool action was performed.",
        )
    return ModelProviderError(
        "ollama_request_failed",
        f"Ollama rejected the local request (HTTP {status}). Run `model routing status` before retrying.",
    )


def _bounded_ollama_read(
    opener: Callable[..., Any],
    request: Request,
    timeout_seconds: float,
) -> bytes:
    if not _OLLAMA_BLOCKING_WORKER_SLOTS.acquire(blocking=False):
        raise ModelProviderError(
            "ollama_request_capacity_exhausted",
            "Ollama request capacity is temporarily full. Wait for existing requests to finish, then retry.",
        )

    completed: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    response_lock = threading.Lock()
    active_response: list[Any] = []

    def worker() -> None:
        try:
            try:
                with opener(request, timeout=timeout_seconds) as response:
                    with response_lock:
                        active_response.append(response)
                    try:
                        raw = response.read(MAX_OLLAMA_RESPONSE_BYTES + 1)
                    finally:
                        with response_lock:
                            active_response.clear()
                completed.put_nowait(("ok", raw))
            except Exception as exc:
                try:
                    completed.put_nowait(("error", exc))
                except queue.Full:
                    pass
        finally:
            _OLLAMA_BLOCKING_WORKER_SLOTS.release()

    thread = threading.Thread(target=worker, name="jarvis-ollama-request", daemon=True)
    try:
        thread.start()
    except Exception:
        _OLLAMA_BLOCKING_WORKER_SLOTS.release()
        raise
    try:
        status, value = completed.get(timeout=timeout_seconds)
    except queue.Empty:
        with response_lock:
            response = active_response[0] if active_response else None
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        raise TimeoutError from None
    thread.join()
    if status == "error":
        if isinstance(value, BaseException):
            raise value
        raise TypeError("Ollama request failed without an exception")
    if type(value) is not bytes:
        raise TypeError("Ollama response body must be bytes")
    return value


def _openai_response_text(
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
    max_output_tokens: int | None,
    reasoning_effort: str,
    environ: dict[str, str] | None,
    urlopen_impl: Callable[..., Any] | None,
    usage_metadata: dict[str, object] | None,
) -> str:
    source = os.environ if environ is None else environ
    api_key = str(source.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ModelProviderError(
            "openai_api_key_missing",
            "Set OPENAI_API_KEY in Jarvis's local environment, then retry `model routing status`.",
        )

    payload: dict[str, object] = {
        "model": model,
        "input": messages,
        "reasoning": {"effort": normalized_reasoning_effort(reasoning_effort)},
        "safety_identifier": OPENAI_SINGLE_OWNER_SAFETY_IDENTIFIER,
        "store": False,
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max(1, int(max_output_tokens))

    encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(encoded_payload) > MAX_OPENAI_REQUEST_BYTES:
        raise ModelProviderError(
            "openai_request_too_large",
            "The OpenAI request is too large. Shorten the input or reduce "
            "JARVIS_CHAT_MAX_HISTORY_MESSAGES, then retry.",
        )

    request = Request(
        OPENAI_RESPONSES_URL,
        data=encoded_payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = (
        build_opener(ProxyHandler({}), _NoRedirectHandler()).open
        if urlopen_impl is None
        else urlopen_impl
    )
    request_timeout = max(0.1, float(timeout_seconds))
    deadline = time.monotonic() + request_timeout
    try:
        raw_response = _bounded_openai_read(opener, request, request_timeout)
        if len(raw_response) > MAX_OPENAI_RESPONSE_BYTES:
            raise ModelProviderError(
                "openai_response_too_large",
                "OpenAI returned an unexpectedly large response. Lower the output token limit, "
                "then retry `model routing status` if it persists.",
            )
        data = json.loads(raw_response.decode("utf-8"))
        if time.monotonic() > deadline:
            raise TimeoutError
    except HTTPError as exc:
        raise _openai_http_error(exc.code, model) from None
    except (TimeoutError, socket.timeout):
        raise ModelProviderError(
            "openai_timeout",
            "The OpenAI model timed out. Try again, or raise the configured model timeout before retrying.",
        ) from None
    except URLError:
        raise ModelProviderError(
            "openai_unreachable",
            "OpenAI could not be reached. Check the network connection, then retry `model routing status`.",
        ) from None
    except OSError:
        raise ModelProviderError(
            "openai_stream_failed",
            "The OpenAI response stream failed. Check the network connection, then retry `model routing status`.",
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ModelProviderError(
            "openai_invalid_response",
            "OpenAI returned an unreadable response. Retry once, then run `model routing status` if it persists.",
        ) from None

    _set_model_usage_receipt(usage_metadata, _openai_usage_receipt(data), "openai")
    _validate_openai_response_status(data)
    text = _extract_openai_output_text(data)
    if not text:
        raise ModelProviderError(
            "openai_empty_response",
            "OpenAI returned no text. Retry once, then run `model routing status` if it persists.",
        )
    return text


def _bounded_openai_read(
    opener: Callable[..., Any],
    request: Request,
    timeout_seconds: float,
) -> bytes:
    if not _OPENAI_BLOCKING_WORKER_SLOTS.acquire(blocking=False):
        raise ModelProviderError(
            "openai_request_capacity_exhausted",
            "OpenAI request capacity is temporarily full. Wait for existing requests to finish, then retry.",
        )

    completed: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    response_lock = threading.Lock()
    active_response: list[Any] = []

    def worker() -> None:
        try:
            try:
                with opener(request, timeout=timeout_seconds) as response:
                    with response_lock:
                        active_response.append(response)
                    try:
                        raw = response.read(MAX_OPENAI_RESPONSE_BYTES + 1)
                    finally:
                        with response_lock:
                            active_response.clear()
                completed.put_nowait(("ok", raw))
            except Exception as exc:
                try:
                    completed.put_nowait(("error", exc))
                except queue.Full:
                    pass
        finally:
            _OPENAI_BLOCKING_WORKER_SLOTS.release()

    thread = threading.Thread(target=worker, name="jarvis-openai-request", daemon=True)
    try:
        thread.start()
    except Exception:
        _OPENAI_BLOCKING_WORKER_SLOTS.release()
        raise
    try:
        status, value = completed.get(timeout=timeout_seconds)
    except queue.Empty:
        with response_lock:
            response = active_response[0] if active_response else None
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        raise TimeoutError from None
    thread.join()
    if status == "error":
        if isinstance(value, BaseException):
            raise value
        raise TypeError("OpenAI request failed without an exception")
    if type(value) is not bytes:
        raise TypeError("OpenAI response body must be bytes")
    return value


def _validate_openai_response_status(data: object) -> None:
    if not isinstance(data, dict):
        return
    raw_status = data.get("status")
    if raw_status is None:
        raise ModelProviderError(
            "openai_invalid_response_status",
            "OpenAI returned a response without a completion state. Retry once, then run "
            "`model routing status` if it persists.",
        )
    status = str(raw_status).strip().lower()
    if status == "completed":
        return
    if status == "incomplete":
        details = data.get("incomplete_details")
        reason = ""
        if isinstance(details, dict):
            reason = str(details.get("reason") or "").strip().lower()
        if reason in {"max_tokens", "max_output_tokens"}:
            raise ModelProviderError(
                "openai_output_incomplete_max_tokens",
                "OpenAI reached the total reasoning/output token limit before finishing. Ask for a "
                "shorter answer or raise JARVIS_OPENAI_MAX_OUTPUT_TOKENS, then retry.",
            )
        if reason == "content_filter":
            raise ModelProviderError(
                "openai_output_filtered",
                "OpenAI could not complete that response under its content policy. Rephrase the request "
                "and retry; no partial answer was used.",
            )
        raise ModelProviderError(
            "openai_output_incomplete",
            "OpenAI returned an incomplete response. Retry once, then run `model routing status` if it persists; "
            "no partial answer was used.",
        )
    if status == "failed":
        raise ModelProviderError(
            "openai_response_failed",
            "OpenAI could not complete the response. Retry once, then run `model routing status` if it persists.",
        )
    if status == "cancelled":
        raise ModelProviderError(
            "openai_response_cancelled",
            "The OpenAI response was cancelled before completion. Retry the request.",
        )
    if status in {"queued", "in_progress"}:
        raise ModelProviderError(
            "openai_response_not_ready",
            "OpenAI returned before the response was complete. Retry once, then run `model routing status` if it persists.",
        )
    raise ModelProviderError(
        "openai_invalid_response_status",
        "OpenAI returned an unrecognized response state. Retry once, then run `model routing status` if it persists.",
    )


def _openai_http_error(status: int, model: str) -> ModelProviderError:
    safe_model = str(model or "configured model")[:128]
    if status == 401:
        return ModelProviderError(
            "openai_authentication_failed",
            "OPENAI_API_KEY was rejected. Replace the local key, then retry `model routing status`.",
        )
    if status == 403:
        return ModelProviderError(
            "openai_access_denied",
            f"This OpenAI project cannot use {safe_model}. Check model access, then retry `model routing status`.",
        )
    if status == 404:
        return ModelProviderError(
            "openai_model_unavailable",
            f"{safe_model} is not available to this OpenAI project. Choose an available model, then retry `model routing status`.",
        )
    if status == 429:
        return ModelProviderError(
            "openai_rate_limited",
            "OpenAI rate or spending limits were reached. Check the API project limits, then retry later.",
        )
    if 500 <= status <= 599:
        return ModelProviderError(
            "openai_service_error",
            "OpenAI is having trouble. Retry later; no Jarvis approval or tool action was performed.",
        )
    return ModelProviderError(
        "openai_request_failed",
        f"OpenAI rejected the request (HTTP {status}). Run `model routing status` before retrying.",
    )


def _extract_openai_output_text(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    output = data.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()
