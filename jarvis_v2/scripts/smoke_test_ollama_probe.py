"""Deterministic coverage for the central Ollama model probe (no network)."""

from __future__ import annotations

import json
from urllib.error import HTTPError
from unittest.mock import patch

import jarvis_v2.agent.model_provider as model_provider_module
from jarvis_v2.agent.model_provider import (
    MAX_OLLAMA_PROBE_RESPONSE_BYTES,
    probe_ollama_models,
)


RAW_HOST_MARKER = "RAW_OLLAMA_HOST_SECRET"
MODEL_PAYLOAD_MARKER = "PRIVATE_MODEL_PAYLOAD_SECRET"
REDIRECT_MARKER = "PRIVATE_REDIRECT_SECRET"


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.read_sizes: list[int] = []
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        return False

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.payload if size < 0 else self.payload[:size]


class FakeOpener:
    def __init__(self, outcome: FakeResponse | BaseException):
        self.outcome = outcome
        self.calls: list[tuple[object, float]] = []

    def open(self, request, *, timeout: float):
        self.calls.append((request, timeout))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class ForbiddenOpener:
    def __init__(self):
        self.calls = 0

    def open(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("blocked Ollama destination reached the opener")


def _forbid_network(*_args, **_kwargs):
    raise AssertionError("Ollama probe smoke attempted a real network call")


def _probe(environ: dict[str, str], opener: object):
    with (
        patch.object(model_provider_module, "build_opener", side_effect=_forbid_network),
        patch.object(model_provider_module.socket, "getaddrinfo", side_effect=_forbid_network),
        patch.object(model_provider_module.socket, "create_connection", side_effect=_forbid_network),
        patch.object(model_provider_module, "urlopen", side_effect=_forbid_network),
    ):
        return probe_ollama_models(
            environ,
            timeout_seconds=2.5,
            opener=opener,
        )


def _assert_diagnostics_are_content_free(
    result: tuple[bool, list[str], str, str],
    *forbidden: str,
) -> None:
    diagnostic_text = json.dumps(result[2:], sort_keys=True)
    for marker in forbidden:
        if marker and marker in diagnostic_text:
            raise AssertionError("Ollama probe diagnostics exposed private content")


def test_localhost_is_pinned_and_models_are_bounded() -> None:
    expected_models = [f"model-{index}:latest" for index in range(1000)]
    payload = json.dumps(
        {"models": [{"name": name} for name in expected_models + ["beyond-bound"]]}
    ).encode("utf-8")
    response = FakeResponse(payload)
    opener = FakeOpener(response)

    result = _probe(
        {"OLLAMA_HOST": "http://localhost:22114"},
        opener,
    )

    if result != (True, expected_models, "ollama_probe_ok", ""):
        raise AssertionError("Ollama probe did not return the bounded valid model list")
    if len(opener.calls) != 1:
        raise AssertionError(f"Ollama probe opener call count drifted: {len(opener.calls)}")
    request, timeout = opener.calls[0]
    if request.full_url != "http://127.0.0.1:22114/api/tags":
        raise AssertionError(f"Ollama probe URL was not pinned and exact: {request.full_url!r}")
    if request.get_method() != "GET" or request.get_header("Accept") != "application/json":
        raise AssertionError("Ollama probe request method or Accept header drifted")
    if timeout != 2.5:
        raise AssertionError(f"Ollama probe timeout drifted: {timeout!r}")
    if response.read_sizes != [MAX_OLLAMA_PROBE_RESPONSE_BYTES + 1]:
        raise AssertionError(f"Ollama probe read was not bounded: {response.read_sizes}")
    if not response.entered or not response.exited:
        raise AssertionError("Ollama probe did not close its response context")
    _assert_diagnostics_are_content_free(
        result,
        "localhost",
        "127.0.0.1",
        expected_models[0],
    )


def test_redirect_is_blocked_without_following() -> None:
    redirect = HTTPError(
        f"http://127.0.0.1:22115/api/tags?{REDIRECT_MARKER}",
        302,
        REDIRECT_MARKER,
        {"Location": f"https://remote.example.invalid/{REDIRECT_MARKER}"},
        None,
    )
    opener = FakeOpener(redirect)

    result = _probe({"OLLAMA_HOST": "localhost:22115"}, opener)

    if result != (False, [], "ollama_probe_redirect_blocked", "HTTPError"):
        raise AssertionError(f"Ollama redirect failure contract drifted: {result!r}")
    if len(opener.calls) != 1:
        raise AssertionError(f"Ollama redirect was retried or followed: {len(opener.calls)}")
    request, _timeout = opener.calls[0]
    if request.full_url != "http://127.0.0.1:22115/api/tags":
        raise AssertionError(f"Ollama redirect request URL drifted: {request.full_url!r}")
    _assert_diagnostics_are_content_free(result, REDIRECT_MARKER, "remote.example.invalid")


def test_remote_and_malformed_destinations_never_reach_opener() -> None:
    cases = (
        f"http://{RAW_HOST_MARKER}.example.invalid:22116",
        f" http://127.0.0.1:22117/{RAW_HOST_MARKER}",
        f"http://127.0.0.1:22118\n{RAW_HOST_MARKER}",
    )
    for raw_host in cases:
        opener = ForbiddenOpener()
        result = _probe({"OLLAMA_HOST": raw_host}, opener)
        if result != (
            False,
            [],
            "ollama_destination_not_local",
            "OllamaDestinationBlocked",
        ):
            raise AssertionError("Blocked Ollama destination failure contract drifted")
        if opener.calls:
            raise AssertionError("Blocked Ollama destination called the opener")
        _assert_diagnostics_are_content_free(result, raw_host, RAW_HOST_MARKER)


def test_oversized_invalid_json_and_invalid_roots_fail_content_free() -> None:
    cases = (
        (
            "oversized",
            (MODEL_PAYLOAD_MARKER.encode("utf-8") + b"x" * MAX_OLLAMA_PROBE_RESPONSE_BYTES),
            (False, [], "ollama_probe_response_too_large", "ResponseTooLarge"),
        ),
        (
            "invalid JSON",
            f'{{"models":["{MODEL_PAYLOAD_MARKER}"]'.encode("utf-8"),
            (False, [], "ollama_probe_unavailable", "JSONDecodeError"),
        ),
        (
            "invalid root",
            json.dumps([MODEL_PAYLOAD_MARKER]).encode("utf-8"),
            (False, [], "ollama_probe_invalid_response", "InvalidResponse"),
        ),
        (
            "invalid models root",
            json.dumps({"models": {"private": MODEL_PAYLOAD_MARKER}}).encode("utf-8"),
            (False, [], "ollama_probe_invalid_response", "InvalidResponse"),
        ),
    )
    for label, payload, expected in cases:
        response = FakeResponse(payload)
        opener = FakeOpener(response)
        result = _probe({"OLLAMA_HOST": "127.0.0.1:22119"}, opener)
        if result != expected:
            raise AssertionError(f"Ollama probe {label} failure contract drifted: {result!r}")
        if len(opener.calls) != 1:
            raise AssertionError(f"Ollama probe {label} call count drifted")
        _assert_diagnostics_are_content_free(
            result,
            MODEL_PAYLOAD_MARKER,
            payload.decode("utf-8", errors="ignore"),
        )


def test_hostile_model_names_are_dropped_without_diagnostic_leaks() -> None:
    hostile_path = f"../../{MODEL_PAYLOAD_MARKER}"
    hostile_control = f"model\n{MODEL_PAYLOAD_MARKER}"
    hostile_nul = f"model\x00{MODEL_PAYLOAD_MARKER}"
    payload = json.dumps(
        {
            "models": [
                {"name": "safe-model:latest"},
                {"model": "fallback_model-2"},
                {"name": hostile_path},
                {"name": hostile_control},
                {"name": hostile_nul},
                {"name": "x" * 241},
                MODEL_PAYLOAD_MARKER,
                {"name": []},
            ]
        }
    ).encode("utf-8")
    opener = FakeOpener(FakeResponse(payload))

    result = _probe({"OLLAMA_HOST": "[::1]:22120"}, opener)

    if result != (
        True,
        ["safe-model:latest", "fallback_model-2"],
        "ollama_probe_ok",
        "",
    ):
        raise AssertionError(f"Ollama probe retained hostile model names: {result!r}")
    _assert_diagnostics_are_content_free(
        result,
        MODEL_PAYLOAD_MARKER,
        hostile_path,
        hostile_control,
        hostile_nul,
    )


def main() -> None:
    test_localhost_is_pinned_and_models_are_bounded()
    test_redirect_is_blocked_without_following()
    test_remote_and_malformed_destinations_never_reach_opener()
    test_oversized_invalid_json_and_invalid_roots_fail_content_free()
    test_hostile_model_names_are_dropped_without_diagnostic_leaks()
    print("Ollama probe smoke passed")


if __name__ == "__main__":
    main()
