from __future__ import annotations

import json
import io
import os
import sys
import threading
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

import jarvis_v2.agent.chat as chat_module
import jarvis_v2.agent.model_planner as model_planner_module
import jarvis_v2.agent.model_provider as model_provider_module
import jarvis_v2.tools.model_status as model_status_module
import jarvis_v2.tools.compose_connector as compose_connector_module
import jarvis_v2.tools.audit as audit_module
import jarvis_v2.tools.research_connector as research_connector_module
from jarvis_v2.automations.compaction import _default_summarize
from jarvis_v2.agent.chat import ChatBrain
from jarvis_v2.agent.model_planner import ModelBackedPlanner
from jarvis_v2.agent.model_provider import (
    ModelProviderError,
    OPENAI_SAFETY_IDENTIFIER_MAX_CHARS,
    OPENAI_SINGLE_OWNER_SAFETY_IDENTIFIER,
    generate_model_text,
)
from jarvis_v2.config import (
    DEFAULT_OPENAI_CHAT_MODEL,
    DEFAULT_OPENAI_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_PLANNER_MODEL,
    MAX_OPENAI_MAX_OUTPUT_TOKENS,
    MIN_OPENAI_MAX_OUTPUT_TOKENS,
    JarvisConfig,
    load_config,
)
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.tools.doctor import make_doctor_tool
from jarvis_v2.tools.conversation import make_conversation_tools
from jarvis_v2.tools.compose_connector import _generate_text as compose_generate_text
from jarvis_v2.tools.model_status import (
    make_model_status_tool,
    make_specialist_execution_readiness_tool,
    make_specialist_model_draft_tool,
    make_specialist_proposal_gate_tool,
)
from jarvis_v2.tools.readiness import make_readiness_tools
from jarvis_v2.tools.research_connector import _synthesize
from jarvis_v2.tools.registry import ToolRegistry
from jarvis_v2.tools.system import setup_check
from jarvis_v2.scripts.run_imessage_control import _warm_model as warm_imessage_model
from jarvis_v2.scripts.run_telegram_control import _warm_model as warm_telegram_model


OLLAMA_LOOPBACK_HOST = "http://127.0.0.1:11434"
EXPECTED_OLLAMA_NUM_CTX = 24_000
OLLAMA_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _synthetic_openai_key(label: str) -> str:
    """Create a secret-shaped test value without storing a static signature."""
    return "s" + "k-" + label


@contextmanager
def _local_ollama_env(*, stored_history: bool):
    keys = (
        "OLLAMA_HOST",
        "OLLAMA_NO_CLOUD",
        "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT",
        *OLLAMA_PROXY_ENV_KEYS,
    )
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["OLLAMA_HOST"] = OLLAMA_LOOPBACK_HOST
    for key in OLLAMA_PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    if stored_history:
        os.environ["OLLAMA_NO_CLOUD"] = "1"
        os.environ["JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT"] = "1"
    else:
        os.environ.pop("OLLAMA_NO_CLOUD", None)
        os.environ.pop("JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT", None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _size: int = -1) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class FakeReadResponse:
    def __init__(self, read_impl):
        self.read_impl = read_impl

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.read_impl(size)


def _completed_openai_response(text: str = "mocked OpenAI response") -> FakeResponse:
    return FakeResponse(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        }
    )


def test_openai_default_transport_ignores_proxy_environment() -> None:
    proxy_environment = {
        "HTTP_PROXY": "http://upper-http-proxy.private.invalid:24101",
        "HTTPS_PROXY": "http://upper-https-proxy.private.invalid:24102",
        "ALL_PROXY": "socks5://upper-all-proxy.private.invalid:24103",
        "http_proxy": "http://lower-http-proxy.private.invalid:24104",
        "https_proxy": "http://lower-https-proxy.private.invalid:24105",
        "all_proxy": "socks5://lower-all-proxy.private.invalid:24106",
    }
    calls: list[tuple[object, float]] = []
    opener_contract: list[tuple[object, ...]] = []

    class MockPrivateOpener:
        def open(self, request, *, timeout: float):
            calls.append((request, timeout))
            return _completed_openai_response()

    def fake_build_opener(*handlers):
        opener_contract.append(handlers)
        proxy_handlers = [
            handler
            for handler in handlers
            if isinstance(handler, model_provider_module.ProxyHandler)
        ]
        if len(proxy_handlers) != 1 or proxy_handlers[0].proxies:
            raise AssertionError(
                f"OpenAI transport must install one empty proxy handler: {handlers!r}"
            )
        if not any(
            type(handler).__name__ == "_NoRedirectHandler" for handler in handlers
        ):
            raise AssertionError(
                f"OpenAI transport must install its no-redirect handler: {handlers!r}"
            )
        return MockPrivateOpener()

    environment = {"OPENAI_API_KEY": _synthetic_openai_key("proxy-smoke"), **proxy_environment}
    with (
        patch.dict(os.environ, environment, clear=True),
        patch.object(model_provider_module, "build_opener", fake_build_opener),
        patch.object(
            model_provider_module,
            "urlopen",
            side_effect=AssertionError("OpenAI default transport used global urlopen"),
        ),
    ):
        answer = generate_model_text(
            provider="openai",
            model="gpt-5.6-terra",
            messages=[{"role": "user", "content": "proxy isolation smoke"}],
            timeout_seconds=2.5,
        )

    if answer != "mocked OpenAI response":
        raise SystemExit(f"OpenAI private opener response drifted: {answer!r}")
    if len(opener_contract) != 1 or len(calls) != 1:
        raise SystemExit(
            f"OpenAI private opener call contract drifted: {opener_contract!r} / {calls!r}"
        )
    request, timeout = calls[0]
    if request.full_url != model_provider_module.OPENAI_RESPONSES_URL or timeout != 2.5:
        raise SystemExit(f"OpenAI private opener target/timeout drifted: {calls!r}")


def test_openai_default_transport_rejects_redirect_without_replay() -> None:
    secret = _synthetic_openai_key("redirect-smoke-must-not-replay")
    redirect_target = "https://redirect-attacker.private.invalid/collect"

    class MockRedirectOpener:
        def __init__(self, no_redirect_handler, status: int, outbound_requests: list[object]):
            self.no_redirect_handler = no_redirect_handler
            self.status = status
            self.outbound_requests = outbound_requests

        def open(self, request, *, timeout: float):
            self.outbound_requests.append(request)
            redirected = self.no_redirect_handler.redirect_request(
                request,
                None,
                self.status,
                "Redirect",
                {"Location": redirect_target},
                redirect_target,
            )
            if redirected is not None:
                self.outbound_requests.append(redirected)
            raise HTTPError(
                request.full_url,
                self.status,
                "Redirect",
                {"Location": redirect_target},
                None,
            )

    for status in (301, 302, 303, 307, 308):
        outbound_requests: list[object] = []

        def fake_build_opener(*handlers):
            no_redirect_handlers = [
                handler
                for handler in handlers
                if type(handler).__name__ == "_NoRedirectHandler"
            ]
            if len(no_redirect_handlers) != 1:
                raise AssertionError(
                    f"OpenAI transport missed its no-redirect handler: {handlers!r}"
                )
            return MockRedirectOpener(
                no_redirect_handlers[0], status, outbound_requests
            )

        try:
            with patch.object(model_provider_module, "build_opener", fake_build_opener):
                generate_model_text(
                    provider="openai",
                    model="gpt-5.6-terra",
                    messages=[{"role": "user", "content": "redirect payload must not replay"}],
                    timeout_seconds=2.5,
                    environ={"OPENAI_API_KEY": secret},
                )
        except ModelProviderError as exc:
            if exc.diagnostic != "openai_request_failed":
                raise SystemExit(
                    f"OpenAI {status} redirect diagnostic drifted: {exc.diagnostic!r}"
                )
            if secret in str(exc) or redirect_target in str(exc):
                raise SystemExit("OpenAI redirect error exposed a secret or redirect target")
        else:
            raise SystemExit(f"OpenAI default transport accepted a {status} redirect")

        if len(outbound_requests) != 1:
            raise SystemExit(
                f"OpenAI {status} redirect replayed the authorization header or POST payload: "
                f"{len(outbound_requests)} outbound requests"
            )
        original = outbound_requests[0]
        if original.get_header("Authorization") != f"Bearer {secret}" or not original.data:
            raise SystemExit("OpenAI redirect smoke did not exercise an authenticated POST body")
        if original.full_url != model_provider_module.OPENAI_RESPONSES_URL:
            raise SystemExit(f"OpenAI redirect smoke target drifted: {original.full_url!r}")


def test_openai_request_is_bounded_private_and_unicode_safe() -> None:
    captured: dict[str, object] = {}
    usage_metadata: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "안녕하세요, the operator."}],
                    }
                ],
                "usage": {
                    "input_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 40},
                    "output_tokens": 80,
                    "output_tokens_details": {"reasoning_tokens": 50},
                    "total_tokens": 200,
                },
            }
        )

    secret = _synthetic_openai_key("smoke-must-never-leak")
    answer = generate_model_text(
        provider="openai",
        model="gpt-5.6-terra",
        messages=[{"role": "user", "content": "한국어로 인사해줘"}],
        timeout_seconds=9.5,
        max_output_tokens=120,
        reasoning_effort="low",
        environ={
            "OPENAI_API_KEY": secret,
            "JARVIS_OWNER_TELEGRAM": "987654321",
            "JARVIS_OWNER_IMESSAGE": "owner@example.invalid",
            "USER": "private-local-user",
        },
        urlopen_impl=fake_urlopen,
        usage_metadata=usage_metadata,
    )
    if answer != "안녕하세요, the operator.":
        raise SystemExit(f"OpenAI adapter did not extract output text: {answer!r}")
    payload = captured.get("payload")
    if not isinstance(payload, dict):
        raise SystemExit(f"OpenAI adapter did not send JSON: {captured}")
    if payload.get("model") != "gpt-5.6-terra":
        raise SystemExit(f"OpenAI adapter model drifted: {payload}")
    if payload.get("store") is not False:
        raise SystemExit(f"OpenAI adapter must send the store=false request flag: {payload}")
    if payload.get("safety_identifier") != OPENAI_SINGLE_OWNER_SAFETY_IDENTIFIER:
        raise SystemExit(f"OpenAI adapter missed the stable single-owner safety identifier: {payload}")
    if not 0 < len(OPENAI_SINGLE_OWNER_SAFETY_IDENTIFIER) <= OPENAI_SAFETY_IDENTIFIER_MAX_CHARS:
        raise SystemExit("OpenAI safety identifier violated the API's 64-character boundary")
    if payload.get("reasoning") != {"effort": "low"} or payload.get("max_output_tokens") != 120:
        raise SystemExit(f"OpenAI adapter missed bounded reasoning/output settings: {payload}")
    if payload.get("input") != [{"role": "user", "content": "한국어로 인사해줘"}]:
        raise SystemExit(f"OpenAI adapter corrupted Unicode input: {payload}")
    if secret in json.dumps(payload, ensure_ascii=False) or secret in answer:
        raise SystemExit("OpenAI adapter leaked its API key into the request body or answer")
    if captured.get("url") != "https://api.openai.com/v1/responses" or captured.get("method") != "POST":
        raise SystemExit(f"OpenAI adapter used an unexpected endpoint: {captured}")
    expected_usage = {
        "model_usage_provider": "openai",
        "model_usage_available": True,
        "model_usage_consistent": True,
        "model_input_tokens": 120,
        "model_cached_input_tokens": 40,
        "model_cached_input_tokens_available": True,
        "model_output_tokens": 80,
        "model_reasoning_tokens": 50,
        "model_reasoning_tokens_available": True,
        "model_total_tokens": 200,
        "model_usage_content_recorded": False,
    }
    if usage_metadata != expected_usage:
        raise SystemExit(f"OpenAI adapter missed its content-free usage receipt: {usage_metadata}")
    for forbidden in (
        secret,
        "987654321",
        "owner@example.invalid",
        "private-local-user",
    ):
        if forbidden in json.dumps(payload, ensure_ascii=False):
            raise SystemExit(f"OpenAI request leaked personal or secret identifier material: {payload}")
    for forbidden in (secret, "한국어로 인사해줘", "안녕하세요"):
        if forbidden in str(usage_metadata):
            raise SystemExit(f"OpenAI usage receipt leaked request/response content: {usage_metadata}")


def test_model_usage_receipts_fail_closed_and_preserve_local_parity() -> None:
    private_marker = "private-usage-marker-must-never-leak"
    malformed_cases = [
        {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 999,
        },
        {
            "input_tokens": True,
            "input_tokens_details": {"cached_tokens": private_marker},
            "output_tokens": -1,
            "output_tokens_details": {"reasoning_tokens": 2_000_000_000},
            "total_tokens": private_marker,
        },
    ]
    for usage in malformed_cases:
        def malformed_response(_request, timeout, *, usage=usage):
            return FakeResponse(
                {
                    "status": "completed",
                    "output_text": "safe answer",
                    "usage": usage,
                }
            )

        receipt: dict[str, object] = {"preexisting": private_marker}
        answer = generate_model_text(
            provider="openai",
            model="gpt-5.6-terra",
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=1,
            environ={"OPENAI_API_KEY": _synthetic_openai_key("usage-smoke-must-never-leak")},
            urlopen_impl=malformed_response,
            usage_metadata=receipt,
        )
        if answer != "safe answer" or receipt.get("model_usage_available") is not False:
            raise SystemExit(f"Malformed usage should not invalidate safe text or become trusted: {receipt}")
        if any(receipt.get(key) != 0 for key in (
            "model_input_tokens",
            "model_cached_input_tokens",
            "model_output_tokens",
            "model_reasoning_tokens",
            "model_total_tokens",
        )):
            raise SystemExit(f"Malformed usage counters were not zeroed: {receipt}")
        if private_marker in str(receipt) or "preexisting" in receipt:
            raise SystemExit(f"Malformed usage receipt kept raw or preexisting content: {receipt}")

    incomplete_receipt: dict[str, object] = {}

    def incomplete_response(_request, timeout):
        return FakeResponse(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": private_marker,
                "usage": {
                    "input_tokens": 30,
                    "input_tokens_details": {"cached_tokens": 10},
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 100},
                    "total_tokens": 130,
                },
            }
        )

    try:
        generate_model_text(
            provider="openai",
            model="gpt-5.6-terra",
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=1,
            environ={"OPENAI_API_KEY": _synthetic_openai_key("usage-smoke-must-never-leak")},
            urlopen_impl=incomplete_response,
            usage_metadata=incomplete_receipt,
        )
    except ModelProviderError as exc:
        if exc.diagnostic != "openai_output_incomplete_max_tokens":
            raise SystemExit(f"Incomplete response diagnostic drifted: {exc!r}")
    else:
        raise SystemExit("Incomplete response should still fail closed")
    if incomplete_receipt.get("model_total_tokens") != 130 or incomplete_receipt.get("model_reasoning_tokens") != 100:
        raise SystemExit(f"Incomplete paid response lost its content-free usage receipt: {incomplete_receipt}")
    if private_marker in str(incomplete_receipt):
        raise SystemExit(f"Incomplete usage receipt leaked partial output: {incomplete_receipt}")

    local_calls: list[tuple[object, float]] = []

    def fake_ollama_open(request, timeout):
        local_calls.append((request, timeout))
        return FakeResponse(
            {
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "local answer"},
                "prompt_eval_count": 25,
                "eval_count": 15,
            }
        )

    local_receipt: dict[str, object] = {}
    with (
        _local_ollama_env(stored_history=False),
        patch.dict(sys.modules, {"ollama": None, "httpx": None, "pydantic": None}),
    ):
        answer = generate_model_text(
            provider="ollama",
            model="local-model",
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=1,
            urlopen_impl=fake_ollama_open,
            usage_metadata=local_receipt,
        )
    if answer != "local answer" or local_receipt.get("model_usage_provider") != "ollama":
        raise SystemExit(f"Ollama usage routing drifted: {answer!r} {local_receipt}")
    if local_receipt.get("model_input_tokens") != 25 or local_receipt.get("model_output_tokens") != 15:
        raise SystemExit(f"Ollama native token counts were not preserved: {local_receipt}")
    if local_receipt.get("model_reasoning_tokens_available") is not False:
        raise SystemExit(f"Ollama receipt invented a reasoning-token breakdown: {local_receipt}")
    expected_context_receipt = {
        "ollama_num_ctx_requested": EXPECTED_OLLAMA_NUM_CTX,
        "ollama_num_ctx_requested_available": True,
        "ollama_num_ctx_effective": None,
        "ollama_num_ctx_effective_available": False,
        "ollama_prompt_truncated": None,
        "ollama_prompt_truncation_verified": False,
        "ollama_context_compatibility_verified": False,
    }
    actual_context_receipt = {
        key: local_receipt.get(key) for key in expected_context_receipt
    }
    if actual_context_receipt != expected_context_receipt:
        raise SystemExit(
            "Ollama context receipt overclaimed daemon-effective context or truncation state: "
            f"{local_receipt}"
        )
    if len(local_calls) != 1:
        raise SystemExit(f"Ollama stdlib adapter call count drifted: {local_calls}")
    local_request, local_timeout = local_calls[0]
    local_payload = json.loads(local_request.data.decode("utf-8"))
    if (
        local_request.full_url != "http://127.0.0.1:11434/api/chat"
        or local_request.get_method() != "POST"
        or local_timeout != 1
        or local_payload
        != {
            "model": "local-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "options": {"num_ctx": EXPECTED_OLLAMA_NUM_CTX},
            "keep_alive": "30m",
        }
    ):
        raise SystemExit(
            f"Ollama stdlib request schema drifted: {local_request.full_url!r} "
            f"{local_request.get_method()!r} {local_timeout!r} {local_payload!r}"
        )


def test_ollama_stdlib_transport_is_private_bounded_and_dependency_free() -> None:
    proxy_environment = {
        "HTTP_PROXY": "http://private-proxy.invalid:24101",
        "HTTPS_PROXY": "http://private-proxy.invalid:24102",
        "ALL_PROXY": "socks5://private-proxy.invalid:24103",
        "OLLAMA_API_KEY": "private-ollama-key-must-not-leave-loopback",
    }
    calls: list[tuple[object, float]] = []
    opener_contract: list[tuple[object, ...]] = []

    class BoundedResponse(FakeResponse):
        def __init__(self):
            super().__init__(
                {
                    "done": True,
                    "done_reason": "stop",
                    "message": {"role": "assistant", "content": "안전한 로컬 답변"},
                }
            )
            self.read_sizes: list[int] = []

        def read(self, size: int = -1) -> bytes:
            self.read_sizes.append(size)
            return super().read(size)

    response = BoundedResponse()

    class MockPrivateOpener:
        def open(self, request, *, timeout: float):
            calls.append((request, timeout))
            return response

    def fake_build_opener(*handlers):
        opener_contract.append(handlers)
        proxies = [
            handler
            for handler in handlers
            if isinstance(handler, model_provider_module.ProxyHandler)
        ]
        if len(proxies) != 1 or proxies[0].proxies:
            raise AssertionError("Ollama stdlib transport did not install one empty proxy handler")
        if not any(type(handler).__name__ == "_NoRedirectHandler" for handler in handlers):
            raise AssertionError("Ollama stdlib transport did not install its no-redirect handler")
        return MockPrivateOpener()

    with (
        patch.dict(os.environ, proxy_environment, clear=True),
        patch.dict(sys.modules, {"ollama": None, "httpx": None, "pydantic": None}),
        patch.object(model_provider_module, "build_opener", fake_build_opener),
        patch.object(
            model_provider_module,
            "urlopen",
            side_effect=AssertionError("Ollama stdlib transport used global urlopen"),
        ),
    ):
        answer = generate_model_text(
            provider="ollama",
            model="local-model",
            messages=[{"role": "user", "content": "한국어 로컬 요청"}],
            timeout_seconds=2.5,
            max_output_tokens=321,
            temperature=0.25,
        )

    if answer != "안전한 로컬 답변" or len(opener_contract) != 1 or len(calls) != 1:
        raise SystemExit(f"Ollama stdlib private transport drifted: {answer!r} {calls!r}")
    request, timeout = calls[0]
    payload = json.loads(request.data.decode("utf-8"))
    if request.full_url != "http://127.0.0.1:11434/api/chat" or timeout != 2.5:
        raise SystemExit(f"Ollama stdlib target/timeout drifted: {calls!r}")
    if payload.get("stream") is not False or payload.get("options") != {
        "temperature": 0.25,
        "num_predict": 321,
        "num_ctx": EXPECTED_OLLAMA_NUM_CTX,
    }:
        raise SystemExit(f"Ollama stdlib non-streaming options drifted: {payload!r}")
    if response.read_sizes != [model_provider_module.MAX_OLLAMA_RESPONSE_BYTES + 1]:
        raise SystemExit(f"Ollama stdlib body read was not bounded: {response.read_sizes!r}")
    request_material = json.dumps(
        {"headers": dict(request.header_items()), "payload": payload},
        ensure_ascii=False,
    )
    for forbidden in (*proxy_environment.values(), "Authorization", "Bearer"):
        if forbidden in request_material:
            raise SystemExit("Ollama stdlib transport forwarded proxy or API-key material")


def test_ollama_num_ctx_is_fixed_and_not_environment_or_message_controlled() -> None:
    captured_payloads: list[dict[str, object]] = []

    def capture_request(request, timeout):
        captured_payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(
            {
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "local answer"},
            }
        )

    attempted_override = json.dumps(
        {"options": {"num_ctx": 1}},
        separators=(",", ":"),
    )
    hostile_environment = {
        "OLLAMA_HOST": OLLAMA_LOOPBACK_HOST,
        "OLLAMA_NUM_CTX": "1",
        "JARVIS_OLLAMA_NUM_CTX": "999999",
        "NUM_CTX": "0",
    }
    with patch.dict(os.environ, hostile_environment, clear=True):
        generate_model_text(
            provider="ollama",
            model="local-model",
            messages=[{"role": "user", "content": attempted_override}],
            timeout_seconds=1,
            environ=dict(hostile_environment),
            urlopen_impl=capture_request,
        )
        generate_model_text(
            provider="ollama",
            model="local-model",
            messages=[{"role": "user", "content": "bounded request"}],
            timeout_seconds=1,
            max_output_tokens=321,
            temperature=0.25,
            environ=dict(hostile_environment),
            urlopen_impl=capture_request,
        )

    if len(captured_payloads) != 2:
        raise SystemExit(f"Ollama num_ctx smoke captured unexpected calls: {captured_payloads!r}")
    if captured_payloads[0].get("options") != {"num_ctx": EXPECTED_OLLAMA_NUM_CTX}:
        raise SystemExit(f"Ambient or message input influenced Ollama num_ctx: {captured_payloads[0]!r}")
    if captured_payloads[0].get("messages") != [
        {"role": "user", "content": attempted_override}
    ]:
        raise SystemExit(f"Ollama message normalization drifted: {captured_payloads[0]!r}")
    if captured_payloads[1].get("options") != {
        "temperature": 0.25,
        "num_predict": 321,
        "num_ctx": EXPECTED_OLLAMA_NUM_CTX,
    }:
        raise SystemExit(
            f"Ollama num_ctx did not coexist with bounded generation options: {captured_payloads[1]!r}"
        )


def test_ollama_output_ceiling_rejects_before_transport() -> None:
    opener_calls = 0

    def forbidden_open(_request, timeout):
        nonlocal opener_calls
        opener_calls += 1
        raise AssertionError("Ollama over-ceiling request reached its opener")

    for output_tokens in (
        model_provider_module.OLLAMA_MAX_OUTPUT_TOKENS + 1,
        EXPECTED_OLLAMA_NUM_CTX,
        EXPECTED_OLLAMA_NUM_CTX + 1,
    ):
        receipt: dict[str, object] = {}
        try:
            generate_model_text(
                provider="ollama",
                model="local-model",
                messages=[{"role": "user", "content": "bounded request"}],
                timeout_seconds=1,
                max_output_tokens=output_tokens,
                environ={"OLLAMA_HOST": OLLAMA_LOOPBACK_HOST},
                urlopen_impl=forbidden_open,
                usage_metadata=receipt,
            )
        except ModelProviderError as exc:
            if exc.diagnostic != "ollama_output_token_limit_exceeded":
                raise SystemExit(
                    f"Ollama output-ceiling diagnostic drifted for {output_tokens}: {exc!r}"
                )
        else:
            raise SystemExit(f"Ollama accepted over-ceiling output limit {output_tokens}")
        if receipt.get("ollama_num_ctx_requested_available") is not False:
            raise SystemExit(
                "Rejected Ollama request claimed a context request was sent: "
                f"{output_tokens} {receipt!r}"
            )
        if any(
            receipt.get(key) is not None
            for key in ("ollama_num_ctx_effective", "ollama_prompt_truncated")
        ):
            raise SystemExit(
                f"Rejected Ollama request invented effective context evidence: {receipt!r}"
            )
    if opener_calls != 0:
        raise SystemExit(f"Ollama output-ceiling refusals reached transport {opener_calls} time(s)")

    hostile_receipt = model_provider_module.safe_model_usage_receipt(
        {
            "model_usage_available": False,
            "ollama_num_ctx_requested": EXPECTED_OLLAMA_NUM_CTX,
            "ollama_num_ctx_requested_available": True,
            "ollama_num_ctx_effective": EXPECTED_OLLAMA_NUM_CTX,
            "ollama_num_ctx_effective_available": True,
            "ollama_prompt_truncated": False,
            "ollama_prompt_truncation_verified": True,
            "ollama_context_compatibility_verified": True,
        },
        "ollama",
    )
    for key, expected in {
        "ollama_num_ctx_requested": EXPECTED_OLLAMA_NUM_CTX,
        "ollama_num_ctx_requested_available": True,
        "ollama_num_ctx_effective": None,
        "ollama_num_ctx_effective_available": False,
        "ollama_prompt_truncated": None,
        "ollama_prompt_truncation_verified": False,
        "ollama_context_compatibility_verified": False,
    }.items():
        if hostile_receipt.get(key) != expected:
            raise SystemExit(
                f"Ollama safe receipt trusted unverified daemon context field {key}: {hostile_receipt!r}"
            )


def test_ollama_redirects_are_blocked_without_replay() -> None:
    redirect_marker = "private-ollama-redirect-marker"
    for status in (301, 302, 303, 307, 308):
        outbound: list[object] = []

        class RedirectOpener:
            def open(self, request, *, timeout: float):
                outbound.append(request)
                raise HTTPError(
                    request.full_url,
                    status,
                    redirect_marker,
                    {"Location": f"https://remote.invalid/{redirect_marker}"},
                    None,
                )

        with patch.object(
            model_provider_module,
            "build_opener",
            return_value=RedirectOpener(),
        ):
            try:
                generate_model_text(
                    provider="ollama",
                    model="local-model",
                    messages=[{"role": "user", "content": redirect_marker}],
                    timeout_seconds=1,
                    environ={"OLLAMA_HOST": "127.0.0.1:22115"},
                )
            except ModelProviderError as exc:
                combined = f"{exc.diagnostic} {exc.recovery_hint}"
                if exc.diagnostic != "ollama_redirect_blocked" or redirect_marker in combined:
                    raise SystemExit(f"Ollama redirect diagnostic leaked or drifted: {exc!r}")
            else:
                raise SystemExit(f"Ollama stdlib transport accepted HTTP {status} redirect")
        if len(outbound) != 1 or outbound[0].full_url != "http://127.0.0.1:22115/api/chat":
            raise SystemExit(f"Ollama redirect replayed or changed its POST: {outbound!r}")


def test_ollama_request_response_and_completion_bounds_fail_closed() -> None:
    private_marker = "private-ollama-boundary-marker-must-not-leak"
    opener_called = False

    def forbidden_open(_request, timeout):
        nonlocal opener_called
        opener_called = True
        raise AssertionError("oversized Ollama request reached its opener")

    try:
        generate_model_text(
            provider="ollama",
            model="local-model",
            messages=[
                {
                    "role": "user",
                    "content": private_marker + "x" * model_provider_module.MAX_OLLAMA_REQUEST_BYTES,
                }
            ],
            timeout_seconds=1,
            urlopen_impl=forbidden_open,
        )
    except ModelProviderError as exc:
        if exc.diagnostic != "ollama_request_too_large" or private_marker in str(exc.recovery_hint):
            raise SystemExit(f"Ollama oversized request diagnostic drifted: {exc!r}")
    else:
        raise SystemExit("Ollama stdlib adapter accepted an oversized request")
    if opener_called:
        raise SystemExit("Ollama stdlib adapter opened a connection for an oversized request")

    cases = (
        (
            b"x" * (model_provider_module.MAX_OLLAMA_RESPONSE_BYTES + 1),
            "ollama_response_too_large",
        ),
        (b"{not-json", "ollama_invalid_response"),
        (json.dumps([private_marker]).encode("utf-8"), "ollama_invalid_response"),
        (
            json.dumps(
                {"done": False, "message": {"content": private_marker}},
            ).encode("utf-8"),
            "ollama_output_incomplete",
        ),
        (
            json.dumps(
                {
                    "done": True,
                    "done_reason": "length",
                    "message": {"content": private_marker},
                }
            ).encode("utf-8"),
            "ollama_output_incomplete_max_tokens",
        ),
        (
            json.dumps(
                {"done": True, "message": {"content": ""}, "thinking": private_marker},
            ).encode("utf-8"),
            "ollama_empty_response",
        ),
        (
            json.dumps(
                {"done": True, "error": {"private": private_marker}, "message": {"content": private_marker}},
            ).encode("utf-8"),
            "ollama_request_rejected",
        ),
    )
    for raw, expected in cases:
        usage: dict[str, object] = {"preexisting": private_marker}

        def response_open(_request, timeout, *, raw=raw):
            return FakeReadResponse(lambda size: raw[:size])

        try:
            generate_model_text(
                provider="ollama",
                model="local-model",
                messages=[{"role": "user", "content": "safe request"}],
                timeout_seconds=1,
                urlopen_impl=response_open,
                usage_metadata=usage,
            )
        except ModelProviderError as exc:
            combined = f"{exc.diagnostic} {exc.recovery_hint}"
            if exc.diagnostic != expected or private_marker in combined:
                raise SystemExit(f"Ollama response refusal drifted for {expected}: {exc!r}")
        else:
            raise SystemExit(f"Ollama stdlib adapter accepted invalid response: {expected}")
        if private_marker in str(usage) or "preexisting" in usage:
            raise SystemExit(f"Ollama refusal retained private usage material: {usage!r}")


def test_ollama_timeout_is_an_end_to_end_body_deadline() -> None:
    class SlowResponse:
        def __init__(self) -> None:
            self.closed = False
            self.finished = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

        def close(self) -> None:
            self.closed = True

        def read(self, _size: int = -1) -> bytes:
            try:
                time.sleep(0.4)
                return json.dumps(
                    {"done": True, "message": {"content": "late private Ollama answer"}}
                ).encode("utf-8")
            finally:
                self.finished.set()

    response = SlowResponse()
    usage: dict[str, object] = {"preexisting": "private"}
    started = time.monotonic()
    try:
        generate_model_text(
            provider="ollama",
            model="local-model",
            messages=[{"role": "user", "content": "private timeout prompt"}],
            timeout_seconds=0.1,
            urlopen_impl=lambda _request, timeout: response,
            usage_metadata=usage,
        )
    except ModelProviderError as exc:
        elapsed = time.monotonic() - started
        if exc.diagnostic != "ollama_timeout" or elapsed >= 0.3:
            raise SystemExit(f"Ollama body read exceeded its deadline: {elapsed:.3f}s {exc!r}")
    else:
        raise SystemExit("Slow Ollama response escaped its configured deadline")
    if response.closed is not True:
        raise SystemExit("Ollama timeout did not close the live response")
    if usage.get("model_usage_available") is not False or "preexisting" in usage:
        raise SystemExit(f"Ollama timeout published or retained usage: {usage!r}")
    if not response.finished.wait(timeout=1):
        raise SystemExit("Timed-out Ollama worker did not finish after its bounded test response")


def test_openai_failures_name_safe_recovery_without_raw_secret() -> None:
    try:
        generate_model_text(
            provider="openai",
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=1,
            environ={},
        )
    except ModelProviderError as exc:
        if exc.diagnostic != "openai_api_key_missing" or "OPENAI_API_KEY" not in exc.recovery_hint:
            raise SystemExit(f"Missing-key recovery is not actionable: {exc!r}")
    else:
        raise SystemExit("OpenAI adapter should fail closed without OPENAI_API_KEY")

    def denied(request, timeout):
        raise HTTPError(request.full_url, 403, "raw secret server text", None, None)

    secret = _synthetic_openai_key("denied-smoke-must-never-leak")
    try:
        generate_model_text(
            provider="openai",
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=1,
            environ={"OPENAI_API_KEY": secret},
            urlopen_impl=denied,
        )
    except ModelProviderError as exc:
        combined = f"{exc.diagnostic} {exc.recovery_hint}"
        if exc.diagnostic != "openai_access_denied" or "model access" not in combined.lower():
            raise SystemExit(f"Access-denied recovery is not actionable: {exc!r}")
        for forbidden in (secret, "raw secret server text"):
            if forbidden in combined:
                raise SystemExit(f"OpenAI failure leaked sensitive raw detail: {combined}")
    else:
        raise SystemExit("OpenAI adapter should fail closed on HTTP 403")


def test_openai_timeout_is_an_end_to_end_body_deadline() -> None:
    class SlowResponse:
        def __init__(self) -> None:
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

        def close(self) -> None:
            self.closed = True

        def read(self, _size: int = -1) -> bytes:
            time.sleep(0.4)
            return json.dumps({"status": "completed", "output_text": "late private answer"}).encode("utf-8")

    response = SlowResponse()
    usage: dict[str, object] = {"preexisting": "private"}

    def slow_body(_request, timeout):
        return response

    started = time.monotonic()
    try:
        generate_model_text(
            provider="openai",
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "private timeout prompt"}],
            timeout_seconds=0.1,
            environ={"OPENAI_API_KEY": _synthetic_openai_key("timeout-smoke-must-never-leak")},
            urlopen_impl=slow_body,
            usage_metadata=usage,
        )
    except ModelProviderError as exc:
        elapsed = time.monotonic() - started
        if exc.diagnostic != "openai_timeout" or elapsed >= 0.3:
            raise SystemExit(f"OpenAI body read exceeded its end-to-end deadline: {elapsed:.3f}s / {exc!r}")
    else:
        raise SystemExit("Slow OpenAI response body escaped the configured deadline")
    if response.closed is not True:
        raise SystemExit("OpenAI timeout did not close the live response")
    if usage.get("model_usage_available") is not False or "preexisting" in usage:
        raise SystemExit(f"Timed-out OpenAI body published usage or retained stale metadata: {usage}")
    if "late private answer" in str(usage) or "private timeout prompt" in str(usage):
        raise SystemExit(f"Timed-out OpenAI body leaked content into usage metadata: {usage}")


def test_openai_hung_workers_have_process_wide_bounded_admission() -> None:
    release_workers = threading.Event()
    state_changed = threading.Condition()
    entered = 0
    exited = 0

    class LateResponse(FakeResponse):
        def __exit__(self, exc_type, exc, tb):
            nonlocal exited
            with state_changed:
                exited += 1
                state_changed.notify_all()
            return False

    def hung_opener(_request, timeout):
        nonlocal entered
        with state_changed:
            entered += 1
            state_changed.notify_all()
        release_workers.wait()
        return LateResponse({"status": "completed", "output_text": "late private capacity answer"})

    secret = _synthetic_openai_key("capacity-smoke-must-never-leak")
    prompt = "private capacity prompt must never leak"
    limit = model_provider_module.OPENAI_BLOCKING_WORKER_LIMIT
    try:
        for _ in range(limit):
            try:
                generate_model_text(
                    provider="openai",
                    model="gpt-5.6-sol",
                    messages=[{"role": "user", "content": prompt}],
                    timeout_seconds=0.1,
                    environ={"OPENAI_API_KEY": secret},
                    urlopen_impl=hung_opener,
                )
            except ModelProviderError as exc:
                if exc.diagnostic != "openai_timeout":
                    raise SystemExit(f"OpenAI worker slot was lost before the configured cap: {exc!r}")
            else:
                raise SystemExit("Hung OpenAI opener escaped its configured deadline")

        with state_changed:
            if not state_changed.wait_for(lambda: entered == limit, timeout=1):
                raise SystemExit(f"Expected {limit} blocked OpenAI workers, observed {entered}")

        active_workers = sum(
            thread.name == "jarvis-openai-request" and thread.is_alive()
            for thread in threading.enumerate()
        )
        if active_workers != limit:
            raise SystemExit(f"OpenAI blocking worker cap drifted: {active_workers} active / {limit} allowed")

        for _ in range(limit * 3):
            saturated_usage: dict[str, object] = {"preexisting": prompt}
            started = time.monotonic()
            try:
                generate_model_text(
                    provider="openai",
                    model="gpt-5.6-sol",
                    messages=[{"role": "user", "content": prompt}],
                    timeout_seconds=0.5,
                    environ={"OPENAI_API_KEY": secret},
                    urlopen_impl=hung_opener,
                    usage_metadata=saturated_usage,
                )
            except ModelProviderError as exc:
                elapsed = time.monotonic() - started
                combined = f"{exc.diagnostic} {exc.recovery_hint}"
                if exc.diagnostic != "openai_request_capacity_exhausted" or elapsed >= 0.1:
                    raise SystemExit(f"Saturated OpenAI provider did not fail promptly: {elapsed:.3f}s / {exc!r}")
                if secret in combined or prompt in combined or "late private" in combined:
                    raise SystemExit(f"OpenAI capacity error leaked request or response content: {combined}")
                if saturated_usage.get("model_usage_available") is not False:
                    raise SystemExit(f"OpenAI capacity error published model usage: {saturated_usage}")
                if prompt in str(saturated_usage) or "preexisting" in saturated_usage:
                    raise SystemExit(f"OpenAI capacity error retained private usage metadata: {saturated_usage}")
            else:
                raise SystemExit("Saturated OpenAI provider admitted a worker beyond its cap")
        if entered != limit:
            raise SystemExit(f"Saturated calls created extra OpenAI workers: {entered} / {limit}")
    finally:
        release_workers.set()

    with state_changed:
        if not state_changed.wait_for(lambda: exited == limit, timeout=2):
            raise SystemExit(f"Timed-out OpenAI workers did not finish after release: {exited} / {limit}")

    answer = generate_model_text(
        provider="openai",
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "normal request"}],
        timeout_seconds=1,
        environ={"OPENAI_API_KEY": secret},
        urlopen_impl=lambda _request, timeout: FakeResponse(
            {"status": "completed", "output_text": "normal answer"}
        ),
    )
    if answer != "normal answer":
        raise SystemExit(f"Released OpenAI worker permits were not reusable: {answer!r}")


def test_openai_body_bounds_fail_closed_without_raw_content() -> None:
    opener_called = False

    def must_not_open(request, timeout):
        nonlocal opener_called
        opener_called = True
        raise AssertionError("oversized request reached the network opener")

    private_marker = "private-request-marker-must-never-leak"
    oversized_input = private_marker + ("x" * model_provider_module.MAX_OPENAI_REQUEST_BYTES)
    try:
        generate_model_text(
            provider="openai",
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": oversized_input}],
            timeout_seconds=1,
            environ={"OPENAI_API_KEY": _synthetic_openai_key("bounds-smoke-must-never-leak")},
            urlopen_impl=must_not_open,
        )
    except ModelProviderError as exc:
        combined = f"{exc.diagnostic} {exc.recovery_hint}"
        if exc.diagnostic != "openai_request_too_large":
            raise SystemExit(f"Oversized request diagnostic drifted: {exc!r}")
        if private_marker in combined:
            raise SystemExit(f"Oversized request leaked input content: {combined}")
    else:
        raise SystemExit("OpenAI adapter should reject an oversized request")
    if opener_called:
        raise SystemExit("OpenAI adapter opened the network for an oversized request")

    response_marker = b"private-response-marker-must-never-leak"

    def oversized_response(_request, timeout):
        def read(_size: int) -> bytes:
            return response_marker + (
                b"x" * (model_provider_module.MAX_OPENAI_RESPONSE_BYTES + 1)
            )

        return FakeReadResponse(read)

    try:
        generate_model_text(
            provider="openai",
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=1,
            environ={"OPENAI_API_KEY": _synthetic_openai_key("bounds-smoke-must-never-leak")},
            urlopen_impl=oversized_response,
        )
    except ModelProviderError as exc:
        combined = f"{exc.diagnostic} {exc.recovery_hint}"
        if exc.diagnostic != "openai_response_too_large":
            raise SystemExit(f"Oversized response diagnostic drifted: {exc!r}")
        if response_marker.decode("utf-8") in combined:
            raise SystemExit(f"Oversized response leaked response content: {combined}")
    else:
        raise SystemExit("OpenAI adapter should reject an oversized response")

    stream_marker = "private-stream-path-/\x55sers/example/must-never-leak"

    def failed_stream(_request, timeout):
        def read(_size: int) -> bytes:
            raise OSError(stream_marker)

        return FakeReadResponse(read)

    try:
        generate_model_text(
            provider="openai",
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=1,
            environ={"OPENAI_API_KEY": _synthetic_openai_key("bounds-smoke-must-never-leak")},
            urlopen_impl=failed_stream,
        )
    except ModelProviderError as exc:
        combined = f"{exc.diagnostic} {exc.recovery_hint}"
        if exc.diagnostic != "openai_stream_failed":
            raise SystemExit(f"OpenAI stream-failure diagnostic drifted: {exc!r}")
        if stream_marker in combined or "/" + "Users/" in combined:
            raise SystemExit(f"OpenAI stream failure leaked raw local detail: {combined}")
    else:
        raise SystemExit("OpenAI adapter should fail closed on a response stream error")


def test_openai_rejects_partial_and_noncompleted_responses() -> None:
    partial_marker = "private-partial-output-must-never-leak"
    backend_marker = "private-backend-error-must-never-leak"

    cases = [
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_tokens"},
                "output_text": partial_marker,
            },
            "openai_output_incomplete_max_tokens",
            "shorter answer",
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": partial_marker,
            },
            "openai_output_incomplete_max_tokens",
            "JARVIS_OPENAI_MAX_OUTPUT_TOKENS",
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "output_text": partial_marker,
            },
            "openai_output_filtered",
            "content policy",
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": backend_marker},
                "output_text": partial_marker,
            },
            "openai_output_incomplete",
            "no partial answer was used",
        ),
        (
            {
                "status": "failed",
                "error": {"message": backend_marker},
                "output_text": partial_marker,
            },
            "openai_response_failed",
            "could not complete",
        ),
        (
            {"status": "cancelled", "output_text": partial_marker},
            "openai_response_cancelled",
            "cancelled",
        ),
        (
            {"status": "in_progress", "output_text": partial_marker},
            "openai_response_not_ready",
            "before the response was complete",
        ),
        (
            {"status": "queued", "output_text": partial_marker},
            "openai_response_not_ready",
            "before the response was complete",
        ),
        (
            {"status": backend_marker, "output_text": partial_marker},
            "openai_invalid_response_status",
            "unrecognized response state",
        ),
        (
            {"output_text": partial_marker},
            "openai_invalid_response_status",
            "without a completion state",
        ),
    ]

    for payload, diagnostic, recovery_fragment in cases:
        def response_for_case(_request, timeout, *, payload=payload):
            return FakeResponse(payload)

        try:
            generate_model_text(
                provider="openai",
                model="gpt-5.6-sol",
                messages=[{"role": "user", "content": "hello"}],
                timeout_seconds=1,
                environ={"OPENAI_API_KEY": _synthetic_openai_key("status-smoke-must-never-leak")},
                urlopen_impl=response_for_case,
            )
        except ModelProviderError as exc:
            combined = f"{exc.diagnostic} {exc.recovery_hint}"
            if exc.diagnostic != diagnostic or recovery_fragment not in exc.recovery_hint:
                raise SystemExit(
                    f"OpenAI response-state recovery drifted for {payload.get('status')!r}: {exc!r}"
                )
            for forbidden in (
                partial_marker,
                backend_marker,
                _synthetic_openai_key("status-smoke-must-never-leak"),
            ):
                if forbidden in combined:
                    raise SystemExit(f"OpenAI response-state failure leaked raw content: {combined}")
        else:
            raise SystemExit(
                f"OpenAI adapter accepted noncompleted response state {payload.get('status')!r}"
            )


def test_openai_config_is_explicit_and_keeps_ollama_default() -> None:
    with TemporaryDirectory(prefix="jarvis-openai-config-") as temp:
        root = Path(temp)
        empty_env = root / "empty.env"
        empty_env.write_text("", encoding="utf-8")
        empty_env.chmod(0o600)
        common = {
            "JARVIS_V3_ENV": str(empty_env),
            "JARVIS_DATA_DIR": str(root / "data"),
            "JARVIS_DB_PATH": str(root / "data" / "jarvis.sqlite"),
            "JARVIS_OBSIDIAN_VAULT": str(root / "vault"),
            "JARVIS_CHAT_MODEL": "",
            "JARVIS_PLANNER_MODEL": "",
            "JARVIS_MODEL_TIMEOUT_SECONDS": "",
            "JARVIS_CHAT_TIMEOUT_SECONDS": "",
            "JARVIS_CHAT_REASONING_EFFORT": "",
            "JARVIS_PLANNER_REASONING_EFFORT": "",
            "JARVIS_OPENAI_MAX_OUTPUT_TOKENS": "",
        }
        with patch.dict(os.environ, {**common, "JARVIS_MODEL_PROVIDER": "openai"}, clear=False):
            config = load_config()
        if config.model_provider != "openai":
            raise SystemExit(f"OpenAI provider opt-in was ignored: {config}")
        if config.chat_model != DEFAULT_OPENAI_CHAT_MODEL or config.planner_model != DEFAULT_OPENAI_PLANNER_MODEL:
            raise SystemExit(f"OpenAI GPT-5.6 defaults drifted: {config}")
        if config.model_timeout_seconds != 12.0 or config.chat_timeout_seconds != 60.0:
            raise SystemExit(f"OpenAI timeout defaults should allow remote reasoning latency: {config}")
        if config.chat_reasoning_effort != "low":
            raise SystemExit(f"OpenAI chat should default to low reasoning for routine latency/cost: {config}")
        if config.openai_max_output_tokens != DEFAULT_OPENAI_MAX_OUTPUT_TOKENS:
            raise SystemExit(f"OpenAI reasoning/output ceiling default drifted: {config}")

        for raw_value, expected in (
            ("500", MIN_OPENAI_MAX_OUTPUT_TOKENS),
            ("999999", MAX_OPENAI_MAX_OUTPUT_TOKENS),
            ("not-an-int", DEFAULT_OPENAI_MAX_OUTPUT_TOKENS),
            ("32000", 32_000),
        ):
            with patch.dict(
                os.environ,
                {
                    **common,
                    "JARVIS_MODEL_PROVIDER": "openai",
                    "JARVIS_OPENAI_MAX_OUTPUT_TOKENS": raw_value,
                },
                clear=False,
            ):
                bounded = load_config()
            if bounded.openai_max_output_tokens != expected:
                raise SystemExit(
                    f"OpenAI reasoning/output ceiling did not bound {raw_value!r}: {bounded}"
                )

        with patch.dict(
            os.environ,
            {**common, "JARVIS_MODEL_PROVIDER": "not-a-provider", "OLLAMA_MODEL": "ollama-smoke"},
            clear=False,
        ):
            fallback = load_config()
        if fallback.model_provider != "invalid" or fallback.chat_model != "ollama-smoke":
            raise SystemExit(
                "Invalid provider should remain explicit for caller-side blocking: "
                f"{fallback}"
            )


def test_runtime_brains_receive_openai_provider_without_live_call() -> None:
    with TemporaryDirectory(prefix="jarvis-openai-runtime-") as temp:
        root = Path(temp)
        store = MemoryStore(root / "jarvis.sqlite")
        store.init()
        brain = ChatBrain(
            "gpt-5.6-terra",
            store,
            provider="openai",
            reasoning_effort="medium",
            model_timeout_seconds=30,
            openai_max_output_tokens=25_000,
        )
        def mocked_openai_chat(**kwargs):
            kwargs["usage_metadata"].update(
                {
                    "model_usage_available": True,
                    "model_input_tokens": 120,
                    "model_cached_input_tokens": 40,
                    "model_cached_input_tokens_available": True,
                    "model_output_tokens": 80,
                    "model_reasoning_tokens": 50,
                    "model_reasoning_tokens_available": True,
                    "model_total_tokens": 200,
                }
            )
            return "Hello from GPT-5.6"

        with patch.object(chat_module, "generate_model_text", side_effect=mocked_openai_chat) as call:
            answer = brain.respond("hello Jarvis")
        if not answer.startswith("Hello from GPT-5.6") or (
            "I couldn't fully check profile for this answer" not in answer
            or "I did not assume the missing context was empty" not in answer
        ):
            raise SystemExit(f"ChatBrain did not use the mocked OpenAI model: {answer!r}")
        kwargs = call.call_args.kwargs
        if kwargs.get("provider") != "openai" or kwargs.get("model") != "gpt-5.6-terra":
            raise SystemExit(f"ChatBrain missed OpenAI provider/model routing: {kwargs}")
        if kwargs.get("max_output_tokens") != 25_000:
            raise SystemExit(f"ChatBrain reused the local visible-output cap for GPT-5.6: {kwargs}")
        model_messages = kwargs.get("messages")
        if not isinstance(model_messages, list) or not any(
            "Remote personal-context policy for this turn" in str(message.get("content") or "")
            and "deliberately kept local" in str(message.get("content") or "")
            and "do not invent personal context" in str(message.get("content") or "")
            for message in model_messages
        ):
            raise SystemExit(f"ChatBrain did not disclose the default-off personal-context policy: {kwargs}")
        if any(
            "Personal context availability for this turn" in str(message.get("content") or "")
            for message in model_messages
        ):
            raise SystemExit(f"ChatBrain leaked local source-health details while context was withheld: {kwargs}")
        if brain.last_turn_metadata.get("model_provider") != "openai":
            raise SystemExit(f"ChatBrain metadata missed provider: {brain.last_turn_metadata}")
        if brain.last_turn_metadata.get("context_unavailable_sources") != ["profile"]:
            raise SystemExit(f"ChatBrain metadata hid missing profile context: {brain.last_turn_metadata}")
        if brain.last_turn_metadata.get("openai_total_output_token_ceiling") != 25_000:
            raise SystemExit(f"ChatBrain metadata hid the GPT-5.6 total-token ceiling: {brain.last_turn_metadata}")
        for key, expected in {
            "model_usage_available": True,
            "model_input_tokens": 120,
            "model_cached_input_tokens": 40,
            "model_output_tokens": 80,
            "model_reasoning_tokens": 50,
            "model_total_tokens": 200,
            "model_usage_content_recorded": False,
        }.items():
            if brain.last_turn_metadata.get(key) != expected:
                raise SystemExit(f"ChatBrain usage receipt missed {key}: {brain.last_turn_metadata}")
        for key in ("model_call_attempted", "calls_model", "calls_external_service", "shares_conversation_with_external_model"):
            if brain.last_turn_metadata.get(key) is not True:
                raise SystemExit(f"ChatBrain missed OpenAI disclosure {key}: {brain.last_turn_metadata}")
        for key in (
            "current_user_message_shared_with_external_model",
            "personal_context_omitted_from_remote_model",
        ):
            if brain.last_turn_metadata.get(key) is not True:
                raise SystemExit(f"ChatBrain missed OpenAI privacy receipt {key}: {brain.last_turn_metadata}")
        for key in (
            "shares_stored_personal_context_with_external_model",
            "shares_history_with_external_model",
            "remote_personal_context_allowed",
        ):
            if brain.last_turn_metadata.get(key) is not False:
                raise SystemExit(f"ChatBrain overclaimed remote personal-context sharing for {key}: {brain.last_turn_metadata}")
        if brain.last_turn_metadata.get("remote_personal_context_policy") != "disabled":
            raise SystemExit(f"ChatBrain hid the default-off remote personal-context policy: {brain.last_turn_metadata}")
        for key in ("external_side_effect", "model_request_content_in_metadata", "model_response_content_in_metadata"):
            if brain.last_turn_metadata.get(key) is not False:
                raise SystemExit(f"ChatBrain disclosure should keep {key}=False: {brain.last_turn_metadata}")

        local_brain = ChatBrain("local-model", store, provider="ollama", max_reply_tokens=300)
        with (
            _local_ollama_env(stored_history=False),
            patch.object(chat_module, "generate_model_text", return_value="Local answer") as local_chat_call,
        ):
            local_brain.respond("hello local Jarvis")
        if local_chat_call.call_args.kwargs.get("max_output_tokens") != 300:
            raise SystemExit(f"Ollama chat lost its existing local output cap: {local_chat_call.call_args.kwargs}")
        if local_brain.last_turn_metadata.get("openai_total_output_token_ceiling") != 0:
            raise SystemExit(f"Local chat falsely reported an OpenAI token ceiling: {local_brain.last_turn_metadata}")

        openai_turn_metadata = dict(brain.last_turn_metadata)
        vault = ObsidianVault(root / "Vault", "Jarvis")
        vault.init()
        store.log_message(
            "openai-receipt",
            "assistant",
            "mocked reply content must not appear in health metadata",
            {"runtime_route": "chat", "chat_response": openai_turn_metadata},
        )
        conversation_handlers = make_conversation_tools(store, vault, "openai-receipt")
        health = conversation_handlers[2]({"limit": 10})
        for expected in (
            "external model attempts: 1",
            "model provider: openai",
            "external model call: yes",
            "conversation shared with external model: yes",
            "prompt/response content copied into health metadata: no",
            "recorded input/output/total tokens: 120 / 80 / 200",
            "cached input tokens: 40",
            "reasoning tokens: 50",
        ):
            if expected not in health.output:
                raise SystemExit(f"Chat health missed OpenAI disclosure {expected!r}: {health.output}")
        if health.metadata.get("latest_calls_external_service") is not True:
            raise SystemExit(f"Chat health metadata missed the OpenAI call: {health.metadata}")
        if health.metadata.get("latest_model_total_tokens") != 200:
            raise SystemExit(f"Chat health metadata missed the OpenAI usage receipt: {health.metadata}")
        if "mocked reply content" in str(health.metadata):
            raise SystemExit(f"Chat health metadata copied reply content: {health.metadata}")

        brain.respond("what do you remember about me?")
        for key in ("model_call_attempted", "calls_model", "calls_external_service", "shares_conversation_with_external_model"):
            if brain.last_turn_metadata.get(key) is not False:
                raise SystemExit(f"Grounded-memory reply falsely claimed external model use for {key}: {brain.last_turn_metadata}")

        def mocked_incomplete_chat(**kwargs):
            kwargs["usage_metadata"].update(
                {
                    "model_usage_available": True,
                    "model_input_tokens": 30,
                    "model_cached_input_tokens": 0,
                    "model_cached_input_tokens_available": True,
                    "model_output_tokens": 100,
                    "model_reasoning_tokens": 100,
                    "model_reasoning_tokens_available": True,
                    "model_total_tokens": 130,
                }
            )
            raise ModelProviderError(
                "openai_output_incomplete_max_tokens",
                "OpenAI reached the output token limit before finishing. Ask for a shorter answer and retry.",
            )

        with patch.object(chat_module, "generate_model_text", side_effect=mocked_incomplete_chat):
            incomplete_answer = brain.respond("Tell me a short story about a lighthouse.")
        if "did not produce a usable complete response" not in incomplete_answer:
            raise SystemExit(f"Chat fallback mislabeled incomplete OpenAI output: {incomplete_answer}")
        if "local model" in incomplete_answer.lower() or "configured chat model" not in incomplete_answer:
            raise SystemExit(f"Chat fallback made a contradictory provider claim: {incomplete_answer}")
        if "Diagnostic: openai_output_incomplete_max_tokens" not in incomplete_answer:
            raise SystemExit(
                "Chat fallback hid the incomplete-output recovery diagnostic: "
                f"{incomplete_answer}\nmetadata={brain.last_turn_metadata}"
            )
        if brain.last_turn_metadata.get("model_error") != "openai_output_incomplete_max_tokens":
            raise SystemExit(f"Chat metadata hid the incomplete-output diagnostic: {brain.last_turn_metadata}")
        if brain.last_turn_metadata.get("calls_external_service") is not True:
            raise SystemExit(f"Chat metadata hid the incomplete remote attempt: {brain.last_turn_metadata}")
        if (
            brain.last_turn_metadata.get("model_execution_status") != "outcome_unknown"
            or brain.last_turn_metadata.get("model_execution_occurred") is not None
            or brain.last_turn_metadata.get("external_processing_status") != "unknown"
            or brain.last_turn_metadata.get("external_processing_occurred") is not None
        ):
            raise SystemExit(
                f"Chat metadata overclaimed incomplete remote execution: {brain.last_turn_metadata}"
            )
        if brain.last_turn_metadata.get("model_total_tokens") != 130:
            raise SystemExit(f"Chat metadata lost paid usage on incomplete output: {brain.last_turn_metadata}")

        planner = ModelBackedPlanner(
            "gpt-5.6-luna",
            ToolRegistry(),
            provider="openai",
            reasoning_effort="low",
            timeout_seconds=12,
            openai_max_output_tokens=25_000,
        )
        def mocked_planner_chat(**kwargs):
            kwargs["usage_metadata"].update(
                {
                    "model_usage_available": True,
                    "model_input_tokens": 300,
                    "model_cached_input_tokens": 200,
                    "model_cached_input_tokens_available": True,
                    "model_output_tokens": 70,
                    "model_reasoning_tokens": 40,
                    "model_reasoning_tokens_available": True,
                    "model_total_tokens": 370,
                }
            )
            return '{"mode":"chat","goal":"talk","actions":[]}'

        with patch.object(
            model_planner_module,
            "generate_model_text",
            side_effect=mocked_planner_chat,
        ) as planner_call:
            plan = planner.plan("please frobnicate workspace")
        if plan.metadata.get("model_planner_provider") != "openai":
            raise SystemExit(f"Planner metadata missed OpenAI provider: {plan.metadata}")
        for key in ("model_planner_calls_model", "model_planner_calls_external_service", "model_planner_shares_request_with_external_model"):
            if plan.metadata.get(key) is not True:
                raise SystemExit(f"Planner metadata missed OpenAI disclosure {key}: {plan.metadata}")
        if (
            plan.metadata.get("model_planner_model_execution_status") != "response_received"
            or plan.metadata.get("model_planner_model_execution_occurred") is not True
            or plan.metadata.get("model_planner_external_processing_status") != "confirmed"
            or plan.metadata.get("model_planner_external_processing_occurred") is not True
        ):
            raise SystemExit(f"Planner metadata missed confirmed OpenAI processing: {plan.metadata}")
        for key in ("model_planner_external_side_effect", "model_planner_request_content_in_metadata", "model_planner_response_content_in_metadata"):
            if plan.metadata.get(key) is not False:
                raise SystemExit(f"Planner metadata should keep {key}=False: {plan.metadata}")
        if planner_call.call_args.kwargs.get("reasoning_effort") != "low":
            raise SystemExit(f"Planner missed OpenAI reasoning effort: {planner_call.call_args.kwargs}")
        if planner_call.call_args.kwargs.get("max_output_tokens") != 25_000:
            raise SystemExit(f"Planner reused its local JSON cap for GPT-5.6 reasoning: {planner_call.call_args.kwargs}")
        if plan.metadata.get("model_planner_max_output_tokens") != 25_000:
            raise SystemExit(f"Planner metadata hid the GPT-5.6 token ceiling: {plan.metadata}")
        if plan.metadata.get("model_planner_total_tokens") != 370 or plan.metadata.get("model_planner_reasoning_tokens") != 40:
            raise SystemExit(f"Planner metadata missed the content-free usage receipt: {plan.metadata}")
        if plan.metadata.get("model_planner_usage_content_recorded") is not False:
            raise SystemExit(f"Planner metadata claimed usage content was stored: {plan.metadata}")
        planner_trace = audit_module._planner_trace_metadata(plan.metadata)
        if planner_trace.get("model_planner_total_tokens") != 370:
            raise SystemExit(f"Runtime audit scrubber lost the planner usage receipt: {planner_trace}")
        if planner_trace.get("model_planner_usage_content_recorded") is not False:
            raise SystemExit(f"Runtime audit scrubber claimed planner content was stored: {planner_trace}")

        hostile_usage = dict(plan.metadata)
        hostile_usage.update(
            {
                "model_planner_input_tokens": "300 private-marker",
                "model_planner_cached_input_tokens": -1,
                "model_planner_output_tokens": True,
                "model_planner_reasoning_tokens": 2_000_000_000,
                "model_planner_total_tokens": "370",
            }
        )
        hostile_trace = audit_module._planner_trace_metadata(hostile_usage)
        if hostile_trace.get("model_planner_usage_available") is not False:
            raise SystemExit(f"Runtime audit scrubber trusted malformed planner usage: {hostile_trace}")
        if "private-marker" in str(hostile_trace):
            raise SystemExit(f"Runtime audit scrubber leaked malformed usage content: {hostile_trace}")

        local_planner = ModelBackedPlanner("local-model", ToolRegistry(), provider="ollama")
        with (
            _local_ollama_env(stored_history=False),
            patch.object(
                model_planner_module,
                "generate_model_text",
                return_value='{"mode":"chat","goal":"talk","actions":[]}',
            ) as local_planner_call,
        ):
            local_planner.plan("please frobnicate local workspace")
        if local_planner_call.call_args.kwargs.get("max_output_tokens") != model_planner_module.MAX_PLANNER_OUTPUT_TOKENS:
            raise SystemExit(f"Ollama planner lost its existing local output cap: {local_planner_call.call_args.kwargs}")

        with patch.object(
            model_planner_module,
            "generate_model_text",
            side_effect=ModelProviderError(
                "openai_timeout",
                "The OpenAI model timed out. Try again before retrying.",
            ),
        ):
            failed_plan = planner.plan("please frobnicate workspace again")
        if failed_plan.metadata.get("model_planner_fell_back") is not True:
            raise SystemExit(f"Planner failure did not preserve fallback: {failed_plan.metadata}")
        if failed_plan.metadata.get("model_planner_calls_external_service") is not True:
            raise SystemExit(f"Failed planner call hid the remote attempt: {failed_plan.metadata}")
        if (
            failed_plan.metadata.get("model_planner_model_execution_status") != "outcome_unknown"
            or failed_plan.metadata.get("model_planner_model_execution_occurred") is not None
            or failed_plan.metadata.get("model_planner_external_processing_status") != "unknown"
            or failed_plan.metadata.get("model_planner_external_processing_occurred") is not None
        ):
            raise SystemExit(
                f"Failed planner call overclaimed remote execution: {failed_plan.metadata}"
            )
        if failed_plan.metadata.get("model_planner_request_content_in_metadata") is not False:
            raise SystemExit(f"Failed planner call copied request content into metadata: {failed_plan.metadata}")


def test_openai_model_status_is_read_only_and_never_displays_key() -> None:
    config = JarvisConfig(
        data_dir=Path("/tmp/jarvis-openai-status"),
        db_path=Path("/tmp/jarvis-openai-status/jarvis.sqlite"),
        obsidian_vault=Path("/tmp/jarvis-openai-status/vault"),
        model_provider="openai",
        chat_model="gpt-5.6-terra",
        planner_model="gpt-5.6-luna",
        chat_reasoning_effort="medium",
        planner_reasoning_effort="low",
    )
    secret = _synthetic_openai_key("status-smoke-must-never-leak")
    with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=False):
        result = make_model_status_tool(config)({})
    if not result.ok or result.metadata.get("model_provider") != "openai":
        raise SystemExit(f"OpenAI model status did not report provider readiness: {result}")
    if result.metadata.get("openai_api_key_configured") is not True:
        raise SystemExit(f"OpenAI model status missed configured-key boolean: {result.metadata}")
    if result.metadata.get("model_access_live_verified") is not False:
        raise SystemExit(f"OpenAI status must not imply a paid live proof: {result.metadata}")
    for key in ("openai_account_retention_controls_checked", "openai_zero_data_retention_verified"):
        if result.metadata.get(key) is not False:
            raise SystemExit(f"OpenAI status overclaimed account retention proof for {key}: {result.metadata}")
    if result.metadata.get("openai_request_store_flag") is not False:
        raise SystemExit(f"OpenAI status lost the store=false request contract: {result.metadata}")
    if result.metadata.get("openai_default_abuse_monitoring_may_retain_content") is not True:
        raise SystemExit(f"OpenAI status hid default abuse-monitoring retention: {result.metadata}")
    if result.metadata.get("openai_default_abuse_monitoring_max_days") != 30:
        raise SystemExit(f"OpenAI status retention bound drifted: {result.metadata}")
    for expected in (
        "safety identifier: stable single-owner pseudonym sent",
        "Responses persistence request flag: disabled (`store: false`)",
        "not a zero-retention guarantee",
        "may retain prompts/responses for up to 30 days",
        "account retention controls: not checked by Jarvis",
        "total reasoning/output ceiling per OpenAI request: 25,000 tokens",
        "actual usage may be lower",
    ):
        if expected not in result.output:
            raise SystemExit(f"OpenAI status missed retention disclosure {expected!r}: {result.output}")
    if "API storage: disabled" in result.output:
        raise SystemExit(f"OpenAI status kept the overbroad storage claim: {result.output}")
    if result.metadata.get("authorizes_model_call") is not False:
        raise SystemExit(f"OpenAI status must not authorize a model call: {result.metadata}")
    if result.metadata.get("openai_max_output_tokens") != 25_000:
        raise SystemExit(f"OpenAI status hid the total-token cost ceiling: {result.metadata}")
    if result.metadata.get("openai_output_ceiling_is_usage_target") is not False:
        raise SystemExit(f"OpenAI status mislabeled the ceiling as expected usage: {result.metadata}")
    expected_safety_metadata = {
        "openai_safety_identifier_sent": True,
        "openai_safety_identifier_scope": "single_owner",
        "openai_safety_identifier_max_chars": 64,
        "openai_safety_identifier_value_exposed": False,
        "openai_safety_identifier_uses_personal_data": False,
        "openai_safety_identifier_uses_api_key": False,
    }
    for key, expected in expected_safety_metadata.items():
        if result.metadata.get(key) != expected:
            raise SystemExit(f"OpenAI status safety-identifier metadata drifted for {key}: {result.metadata}")
    if secret in result.output or secret in str(result.metadata):
        raise SystemExit("OpenAI model status exposed the API key")


def test_openai_specialist_draft_preserves_remote_attempt_receipts() -> None:
    config = JarvisConfig(
        data_dir=Path("/tmp/jarvis-openai-specialist"),
        db_path=Path("/tmp/jarvis-openai-specialist/jarvis.sqlite"),
        obsidian_vault=Path("/tmp/jarvis-openai-specialist/vault"),
        model_provider="openai",
        chat_model="gpt-5.6-terra",
        planner_model="gpt-5.6-luna",
        chat_reasoning_effort="medium",
        planner_reasoning_effort="low",
    )
    verifier_names = ("verification_packet", "chat_response_health", "runtime_trace_receipt")
    draft_tool = make_specialist_model_draft_tool(
        config,
        lambda: [SimpleNamespace(name=name) for name in verifier_names],
    )
    secret = _synthetic_openai_key("specialist-smoke-must-never-leak")
    generated = "Mocked GPT-5.6 specialist draft"
    with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=False), patch.object(
        model_status_module,
        "generate_model_text",
        return_value=generated,
    ) as specialist_call:
        success = draft_tool({"request": "summarize this work history into a brief"})
    if not success.ok or success.metadata.get("draft_state") != "DRAFT_READY":
        raise SystemExit(f"OpenAI specialist draft did not run through the mocked provider: {success}")
    for key in (
        "calls_model",
        "calls_external_service",
        "model_call_attempted",
        "external_model_call_attempted",
        "shares_specialist_request_with_external_model",
    ):
        if success.metadata.get(key) is not True:
            raise SystemExit(f"OpenAI specialist success missed disclosure {key}: {success.metadata}")
    if success.metadata.get("model_provider") != "openai":
        raise SystemExit(f"OpenAI specialist success missed provider: {success.metadata}")
    if specialist_call.call_args.kwargs.get("max_output_tokens") != 25_000:
        raise SystemExit(f"OpenAI specialist reused its local draft cap: {specialist_call.call_args.kwargs}")
    if success.metadata.get("model_response_content_in_metadata") is not False or generated in str(success.metadata):
        raise SystemExit(f"OpenAI specialist success copied generated content into metadata: {success.metadata}")
    if secret in success.output or secret in str(success.metadata):
        raise SystemExit("OpenAI specialist success exposed the API key")

    with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=False), patch.object(
        model_status_module,
        "generate_model_text",
        side_effect=ModelProviderError(
            "openai_timeout",
            "The OpenAI model timed out. Try again before retrying.",
        ),
    ):
        failed = draft_tool({"request": "summarize this work history into a brief"})
    if failed.metadata.get("draft_state") != "DRAFT_FALLBACK_PREVIEW" or failed.metadata.get("draft_produced") is not False:
        raise SystemExit(f"OpenAI specialist failure did not fall back cleanly: {failed}")
    for key in ("calls_model", "calls_external_service", "model_call_attempted", "external_model_call_attempted"):
        if failed.metadata.get(key) is not True:
            raise SystemExit(f"OpenAI specialist failure hid attempted call {key}: {failed.metadata}")
    if failed.metadata.get("model_response_content_in_metadata") is not False:
        raise SystemExit(f"OpenAI specialist failure copied response content into metadata: {failed.metadata}")


def test_specialist_gates_use_the_configured_provider_consistently() -> None:
    request = "summarize this work history into a brief"
    verifier_names = sorted(
        {
            name
            for names in model_status_module.SPECIALIST_VERIFIER_TOOLS.values()
            for name in names
        }
    )
    def list_tools():
        return [SimpleNamespace(name=name) for name in verifier_names]

    def config_for(provider: str, target_model: str) -> JarvisConfig:
        return JarvisConfig(
            data_dir=Path(f"/tmp/jarvis-{provider}-specialist-consistency"),
            db_path=Path(f"/tmp/jarvis-{provider}-specialist-consistency/jarvis.sqlite"),
            obsidian_vault=Path(f"/tmp/jarvis-{provider}-specialist-consistency/vault"),
            model_provider=provider,
            chat_model=target_model,
            planner_model=f"{target_model}-planner",
        )

    def specialist_results(config: JarvisConfig, generated: str):
        readiness = make_specialist_execution_readiness_tool(config, list_tools)
        proposal = make_specialist_proposal_gate_tool(config, list_tools)
        draft = make_specialist_model_draft_tool(config, list_tools)
        with patch.object(model_status_module, "generate_model_text", return_value=generated):
            return readiness({"request": request}), proposal({"request": request}), draft({"request": request})

    def assert_metadata_agrees(results, *, provider: str, target_model: str, ready: bool, live_verified: bool) -> None:
        expected = {
            "model_provider": provider,
            "target_model": target_model,
            "model_ready": ready,
            "model_access_live_verified": live_verified,
        }
        for label, result in zip(("readiness", "proposal", "draft"), results):
            actual = {key: result.metadata.get(key) for key in expected}
            if actual != expected:
                raise SystemExit(
                    f"{label} specialist provider metadata disagreed: expected={expected}, actual={actual}"
                )

    openai_target = "same-target-smoke"
    openai_config = config_for("openai", openai_target)
    with patch.object(model_status_module, "openai_api_key_configured", return_value=True), patch.object(
        model_status_module,
        "_ollama_models",
        side_effect=AssertionError("OpenAI specialist gates must not probe Ollama"),
    ) as openai_ollama_probe:
        openai_results = specialist_results(openai_config, "mocked OpenAI specialist draft")
    if openai_ollama_probe.call_count:
        raise SystemExit("Configured OpenAI specialist gates probed Ollama")
    openai_readiness, openai_proposal, openai_draft = openai_results
    if openai_readiness.metadata.get("verdict") != "READY_FOR_SPECIALIST_MODEL_DRAFT":
        raise SystemExit(f"Configured OpenAI readiness was not ready: {openai_readiness.metadata}")
    if openai_proposal.metadata.get("proposal_gate_state") != "PROPOSAL_DRAFT_READY":
        raise SystemExit(f"Configured OpenAI proposal was not ready: {openai_proposal.metadata}")
    if openai_draft.metadata.get("draft_state") != "DRAFT_READY":
        raise SystemExit(f"Configured OpenAI draft was not ready: {openai_draft.metadata}")
    assert_metadata_agrees(
        openai_results,
        provider="openai",
        target_model=openai_target,
        ready=True,
        live_verified=False,
    )

    with patch.object(model_status_module, "openai_api_key_configured", return_value=False), patch.object(
        model_status_module,
        "_ollama_models",
        return_value=(True, [openai_target], "same target is locally available", ""),
    ) as missing_key_ollama_probe, patch.object(
        model_status_module,
        "generate_model_text",
        side_effect=AssertionError("Missing OpenAI credentials must hold before a model call"),
    ):
        missing_key_results = specialist_results(openai_config, "must not be generated")
    if missing_key_ollama_probe.call_count:
        raise SystemExit("Missing OpenAI credentials incorrectly fell back to Ollama readiness")
    missing_readiness, missing_proposal, missing_draft = missing_key_results
    if missing_readiness.metadata.get("verdict") != "HOLD_FOR_MODEL_SETUP":
        raise SystemExit(f"Missing-key OpenAI readiness did not hold: {missing_readiness.metadata}")
    if missing_proposal.metadata.get("proposal_gate_state") != "PROPOSAL_FALLBACK_NO_MODEL":
        raise SystemExit(f"Missing-key OpenAI proposal did not hold: {missing_proposal.metadata}")
    if missing_draft.metadata.get("draft_state") != "DRAFT_FALLBACK_PREVIEW":
        raise SystemExit(f"Missing-key OpenAI draft did not hold: {missing_draft.metadata}")
    assert_metadata_agrees(
        missing_key_results,
        provider="openai",
        target_model=openai_target,
        ready=False,
        live_verified=False,
    )

    ollama_target = "local-specialist-smoke"
    ollama_config = config_for("ollama", ollama_target)
    with (
        _local_ollama_env(stored_history=False),
        patch.object(
            model_status_module,
            "_ollama_models",
            return_value=(True, [f"{ollama_target}:latest"], "", ""),
        ) as local_probe,
    ):
        ollama_results = specialist_results(ollama_config, "mocked local specialist draft")
    if local_probe.call_count != 3:
        raise SystemExit(f"Each local specialist tool should preserve its Ollama probe: {local_probe.call_count}")
    local_readiness, local_proposal, local_draft = ollama_results
    if local_readiness.metadata.get("verdict") != "READY_FOR_SPECIALIST_MODEL_DRAFT":
        raise SystemExit(f"Local readiness behavior regressed: {local_readiness.metadata}")
    if local_proposal.metadata.get("proposal_gate_state") != "PROPOSAL_DRAFT_READY":
        raise SystemExit(f"Local proposal behavior regressed: {local_proposal.metadata}")
    if local_draft.metadata.get("draft_state") != "DRAFT_READY":
        raise SystemExit(f"Local draft behavior regressed: {local_draft.metadata}")
    assert_metadata_agrees(
        ollama_results,
        provider="ollama",
        target_model=ollama_target,
        ready=True,
        live_verified=False,
    )


def test_secondary_model_paths_follow_provider_and_skip_remote_warmup() -> None:
    config = SimpleNamespace(
        model_provider="openai",
        chat_model="gpt-5.6-terra",
        chat_reasoning_effort="medium",
        chat_timeout_seconds=60.0,
        openai_max_output_tokens=25_000,
    )
    with patch.object(model_provider_module, "generate_model_text", return_value="compact summary") as call:
        answer = _default_summarize(config)("user: 오래된 대화")
    if answer != "compact summary" or call.call_args.kwargs.get("provider") != "openai":
        raise SystemExit(f"Conversation compaction missed OpenAI routing: {call.call_args}")
    if call.call_args.kwargs.get("max_output_tokens") != 25_000:
        raise SystemExit(f"Conversation compaction reused its local cap for GPT-5.6: {call.call_args}")

    with patch.object(model_provider_module, "generate_model_text", return_value="research summary") as call:
        answer = _synthesize(config, "what changed?", "Source: official notes")
    if answer != "research summary" or call.call_args.kwargs.get("provider") != "openai":
        raise SystemExit(f"Research synthesis missed OpenAI routing: {call.call_args}")
    if call.call_args.kwargs.get("max_output_tokens") != 25_000:
        raise SystemExit(f"Research synthesis reused its local cap for GPT-5.6: {call.call_args}")

    with patch.object(compose_connector_module, "generate_model_text", return_value="composed text") as call:
        answer = compose_generate_text(config, "write a short note")
    if answer != "composed text" or call.call_args.kwargs.get("provider") != "openai":
        raise SystemExit(f"Compose-and-write missed OpenAI routing: {call.call_args}")
    if call.call_args.kwargs.get("max_output_tokens") != 25_000:
        raise SystemExit(f"Compose-and-write reused its local draft cap for GPT-5.6: {call.call_args}")

    for warm in (warm_telegram_model, warm_imessage_model):
        output = io.StringIO()
        with redirect_stdout(output):
            warm(config)
        if (
            "warmup skipped" not in output.getvalue()
            or "no request was sent" not in output.getvalue()
        ):
            raise SystemExit(f"Remote provider should skip Ollama warmup: {output.getvalue()!r}")


def test_openai_setup_doctor_and_readiness_do_not_require_ollama() -> None:
    with TemporaryDirectory(prefix="jarvis-openai-readiness-") as temp:
        root = Path(temp)
        empty_env = root / "empty.env"
        empty_env.write_text("", encoding="utf-8")
        empty_env.chmod(0o600)
        vault = ObsidianVault(root / "Vault", "Jarvis")
        vault.init()
        store = MemoryStore(root / "jarvis.sqlite")
        store.init()
        secret = _synthetic_openai_key("readiness-smoke-must-never-leak")
        env = {
            "JARVIS_V3_ENV": str(empty_env),
            "JARVIS_DATA_DIR": str(root),
            "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
            "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
            "JARVIS_MODEL_PROVIDER": "openai",
            "JARVIS_CHAT_MODEL": "gpt-5.6-terra",
            "JARVIS_PLANNER_MODEL": "gpt-5.6-luna",
            "OPENAI_API_KEY": secret,
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            setup = setup_check({})
            doctor = make_doctor_tool(store, vault, config, list_tools=lambda: [])({})
            readiness, prototype = make_readiness_tools(
                store,
                vault,
                config,
                list_tools=lambda: [],
            )
            readiness_result = readiness({})
            prototype_result = prototype({})

        for label, result in [
            ("setup", setup),
            ("doctor", doctor),
            ("readiness", readiness_result),
            ("prototype", prototype_result),
        ]:
            combined = f"{result.output} {result.metadata}"
            if secret in combined:
                raise SystemExit(f"{label} exposed OPENAI_API_KEY")
            if result.metadata.get("model_provider") != "openai":
                raise SystemExit(f"{label} missed OpenAI provider metadata: {result.metadata}")
            if result.metadata.get("ollama_required") is not False:
                raise SystemExit(f"{label} should not require Ollama for OpenAI: {result.metadata}")
            if result.metadata.get("openai_api_key_configured") is not True:
                raise SystemExit(f"{label} missed configured key boolean: {result.metadata}")
            if result.metadata.get("openai_api_key_value_exposed") is not False:
                raise SystemExit(f"{label} key-exposure boundary drifted: {result.metadata}")
            if label == "setup":
                if result.metadata.get("openai_api_key_value_inspected") is not False:
                    raise SystemExit(f"setup inspected OPENAI_API_KEY: {result.metadata}")
                if result.metadata.get("openai_api_key_validation_performed") is not False:
                    raise SystemExit(f"setup overclaimed OPENAI_API_KEY validation: {result.metadata}")
            for key, expected in {
                "openai_safety_identifier_sent": True,
                "openai_safety_identifier_scope": "single_owner",
                "openai_safety_identifier_value_exposed": False,
                "openai_safety_identifier_uses_personal_data": False,
                "openai_safety_identifier_uses_api_key": False,
            }.items():
                if result.metadata.get(key) != expected:
                    raise SystemExit(f"{label} safety-identifier metadata drifted for {key}: {result.metadata}")
            for key in ("openai_account_retention_controls_checked", "openai_zero_data_retention_verified"):
                if result.metadata.get(key) is not False:
                    raise SystemExit(f"{label} overclaimed OpenAI retention proof for {key}: {result.metadata}")
            if result.metadata.get("openai_request_store_flag") is not False:
                raise SystemExit(f"{label} lost the store=false request contract: {result.metadata}")
            if result.metadata.get("openai_default_abuse_monitoring_may_retain_content") is not True:
                raise SystemExit(f"{label} hid default OpenAI abuse-monitoring retention: {result.metadata}")
            if result.metadata.get("openai_default_abuse_monitoring_max_days") != 30:
                raise SystemExit(f"{label} OpenAI retention bound drifted: {result.metadata}")
            if label in {"setup", "doctor"}:
                if result.metadata.get("remote_conversation_compaction_enabled") is not False:
                    raise SystemExit(f"{label} should keep remote conversation compaction disabled by default: {result.metadata}")
                if result.metadata.get("remote_compaction_sends_history_to_external_model") is not False:
                    raise SystemExit(f"{label} overclaimed remote history export: {result.metadata}")
                if result.metadata.get("remote_personal_context_allowed") is not False:
                    raise SystemExit(f"{label} should keep remote personal context disabled by default: {result.metadata}")
                if result.metadata.get("remote_personal_context_sends_stored_context_to_external_model") is not False:
                    raise SystemExit(f"{label} overclaimed stored personal-context export: {result.metadata}")
                if result.metadata.get("remote_personal_context_current_message_may_be_external") is not True:
                    raise SystemExit(f"{label} hid current-message OpenAI disclosure: {result.metadata}")
                if result.metadata.get("remote_personal_context_content_in_metadata") is not False:
                    raise SystemExit(f"{label} copied personal-context content into setup metadata: {result.metadata}")

        for label, output in [("setup", setup.output), ("doctor", doctor.output)]:
            if "not required; OpenAI provider selected" not in output:
                raise SystemExit(f"{label} did not suppress the Ollama requirement: {output}")
            if "Start Ollama" in output or "ollama pull gpt-5.6" in output:
                raise SystemExit(f"{label} emitted contradictory Ollama recovery: {output}")
            if "OpenAI safety identifier" not in output or "not derived from personal data or the API key" not in output:
                raise SystemExit(f"{label} missed the non-personal safety-identifier disclosure: {output}")
        if "remote history summarization disabled by default" not in setup.output:
            raise SystemExit(f"setup check missed the remote-compaction safe default: {setup.output}")
        if "disabled by default; no old conversation batches are sent to OpenAI" not in doctor.output:
            raise SystemExit(f"doctor missed the remote-compaction privacy boundary: {doctor.output}")
        if "stored profile, preferences, memory, skills, and prior chat history stay local by default" not in setup.output:
            raise SystemExit(f"setup check missed the remote personal-context safe default: {setup.output}")
        if "stored personal context stays local while the current message may still be sent to OpenAI" not in doctor.output:
            raise SystemExit(f"doctor missed the interactive OpenAI privacy boundary: {doctor.output}")
        for label, output in (("setup", setup.output), ("doctor", doctor.output)):
            for expected in (
                "store=false",
                "account retention controls",
                "may retain prompts/responses for up to 30 days",
            ):
                if expected not in output:
                    raise SystemExit(f"{label} missed OpenAI retention disclosure {expected!r}: {output}")
        if "OpenAI API configuration: ok" not in readiness_result.output:
            raise SystemExit(f"readiness report missed OpenAI route readiness: {readiness_result.output}")
        if "Ollama reachable: needs attention" in readiness_result.output:
            raise SystemExit(f"readiness report still treated Ollama as required: {readiness_result.output}")
        if any("OpenAI model route needs" in item for item in prototype_result.metadata.get("optional_attention", [])):
            raise SystemExit(f"prototype readiness falsely flagged configured OpenAI route: {prototype_result.metadata}")

        missing_env = dict(env)
        missing_env.pop("OPENAI_API_KEY")
        with patch.dict(os.environ, missing_env, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            missing_setup = setup_check({})
        if "key not present; set OPENAI_API_KEY locally (value not inspected)" not in missing_setup.output:
            raise SystemExit(f"setup check missed OpenAI key recovery: {missing_setup.output}")
        if "OPENAI_API_KEY" not in missing_setup.metadata.get("setup_attention", []):
            raise SystemExit(f"setup check missed OpenAI key attention metadata: {missing_setup.metadata}")
        if missing_setup.metadata.get("openai_api_key_configured") is not False:
            raise SystemExit(f"setup check should report missing OpenAI key: {missing_setup.metadata}")


def main() -> None:
    test_openai_default_transport_ignores_proxy_environment()
    test_openai_default_transport_rejects_redirect_without_replay()
    test_openai_request_is_bounded_private_and_unicode_safe()
    test_model_usage_receipts_fail_closed_and_preserve_local_parity()
    test_ollama_stdlib_transport_is_private_bounded_and_dependency_free()
    test_ollama_num_ctx_is_fixed_and_not_environment_or_message_controlled()
    test_ollama_output_ceiling_rejects_before_transport()
    test_ollama_redirects_are_blocked_without_replay()
    test_ollama_request_response_and_completion_bounds_fail_closed()
    test_ollama_timeout_is_an_end_to_end_body_deadline()
    test_openai_failures_name_safe_recovery_without_raw_secret()
    test_openai_hung_workers_have_process_wide_bounded_admission()
    test_openai_timeout_is_an_end_to_end_body_deadline()
    test_openai_body_bounds_fail_closed_without_raw_content()
    test_openai_rejects_partial_and_noncompleted_responses()
    test_openai_config_is_explicit_and_keeps_ollama_default()
    test_runtime_brains_receive_openai_provider_without_live_call()
    test_openai_model_status_is_read_only_and_never_displays_key()
    test_openai_specialist_draft_preserves_remote_attempt_receipts()
    test_specialist_gates_use_the_configured_provider_consistently()
    test_secondary_model_paths_follow_provider_and_skip_remote_warmup()
    test_openai_setup_doctor_and_readiness_do_not_require_ollama()
    print("Model provider smoke passed")


if __name__ == "__main__":
    main()
