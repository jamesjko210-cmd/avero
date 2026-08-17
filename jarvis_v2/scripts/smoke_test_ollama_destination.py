from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jarvis_v2.agent.chat as chat_module
import jarvis_v2.agent.model_provider as model_provider_module
from jarvis_v2.agent.chat import ChatBrain
from jarvis_v2.agent.model_provider import (
    ModelProviderError,
    generate_model_text,
    resolve_ollama_destination,
)
from jarvis_v2.memory.obsidian import ProfileGroundingView
from jarvis_v2.memory.store import ProfileKnowledgeSnapshot


PROFILE_MARKER = "PRIVATE_PROFILE_DESTINATION_MARKER"
PREFERENCE_MARKER = "PRIVATE_PREFERENCE_DESTINATION_MARKER"
MEMORY_MARKER = "PRIVATE_MEMORY_DESTINATION_MARKER"
SKILL_MARKER = "PRIVATE_SKILL_DESTINATION_MARKER"
ELIGIBLE_HISTORY_MARKER = "ELIGIBLE_PRIVATE_HISTORY_DESTINATION_MARKER"
LEGACY_HISTORY_MARKER = "LEGACY_PRIVATE_HISTORY_DESTINATION_MARKER"
CURRENT_PROMPT_MARKER = "CURRENT_DESTINATION_PROMPT_MARKER"
STORED_CONTEXT_MARKERS = (
    PROFILE_MARKER,
    PREFERENCE_MARKER,
    MEMORY_MARKER,
    SKILL_MARKER,
)
PERSONAL_MARKERS = (*STORED_CONTEXT_MARKERS, ELIGIBLE_HISTORY_MARKER, LEGACY_HISTORY_MARKER)
OLLAMA_UNVERIFIED_LOOPBACK_DESTINATION_CLASS = "loopback_daemon_unverified"


class _DurableDisclosurePolicy:
    def __init__(self, brain: ChatBrain) -> None:
        decision = brain._history_policy_decision(commit=False)
        if (
            decision.provider != "ollama"
            or decision.destination_class != OLLAMA_UNVERIFIED_LOOPBACK_DESTINATION_CLASS
            or decision.stored_context_allowed is not True
        ):
            raise AssertionError(f"durable policy installed for the wrong route: {decision}")
        self.brain = brain
        self.active_epoch_id = decision.epoch_id
        self.active_fingerprint = decision.fingerprint
        self.binding = object()
        self.prepared: dict[str, tuple[object, str, str]] = {}
        self.finalized: list[str] = []

    def prepare(self, payload: dict[str, object]) -> object:
        decision = self.brain._history_policy_decision(commit=False)
        source_count = payload.get("source_count")
        source_digest = payload.get("source_digest")
        if (
            set(payload)
            != {
                "session_generation",
                "policy_epoch_id",
                "policy_fingerprint",
                "provider",
                "destination_class",
                "source_count",
                "source_digest",
            }
            or payload.get("session_generation") != self.brain._session_generation
            or payload.get("policy_epoch_id") != self.active_epoch_id
            or payload.get("policy_epoch_id") != decision.epoch_id
            or payload.get("policy_fingerprint") != self.active_fingerprint
            or payload.get("policy_fingerprint") != decision.fingerprint
            or payload.get("provider") != "ollama"
            or payload.get("destination_class")
            != OLLAMA_UNVERIFIED_LOOPBACK_DESTINATION_CLASS
            or type(source_count) is not int
            or source_count < 1
            or type(source_digest) is not str
            or len(source_digest) != 64
            or any(character not in "0123456789abcdef" for character in source_digest)
        ):
            raise ValueError("disclosure payload does not match the exact active epoch")
        leaked = [marker for marker in PERSONAL_MARKERS if marker in repr(payload)]
        if leaked:
            raise AssertionError(f"durable disclosure callback received content: {leaked}")
        receipt_id = f"{len(self.prepared) + len(self.finalized) + 1:064x}"
        handle = (self.binding, self.active_epoch_id, receipt_id)
        self.prepared[receipt_id] = handle
        return handle

    def validate(self, handle: object) -> bool:
        if type(handle) is not tuple or len(handle) != 3:
            return False
        binding, epoch_id, receipt_id = handle
        return bool(
            binding is self.binding
            and epoch_id == self.active_epoch_id
            and type(receipt_id) is str
            and self.prepared.get(receipt_id) == handle
        )

    def finalize(self, handle: object, outcome: str) -> str:
        if outcome not in {"confirmed", "blocked", "uncertain"} or not self.validate(handle):
            raise ValueError("disclosure finalization lost its active epoch")
        receipt_id = handle[2]
        del self.prepared[receipt_id]
        self.finalized.append(outcome)
        return outcome


def _install_durable_disclosure_policy(brain: ChatBrain) -> _DurableDisclosurePolicy:
    policy = _DurableDisclosurePolicy(brain)
    brain.install_history_disclosure_callbacks(
        prepare=policy.prepare,
        validate=policy.validate,
        finalize=policy.finalize,
        fence=lambda _handle: nullcontext(),
    )
    return policy


def _fail_network(*_args, **_kwargs):
    raise AssertionError("Ollama destination smoke attempted a real network call")


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]):
        self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.payload if size < 0 else self.payload[:size]


def _message_text(call: MagicMock) -> str:
    messages = call.call_args.kwargs.get("messages")
    if not isinstance(messages, list):
        raise AssertionError(f"ChatBrain missed inspectable attempted messages: {call.call_args}")
    return "\n".join(str(message.get("content") or "") for message in messages)


def _private_brain(*, model: str = "mocked-ollama-model") -> ChatBrain:
    store = MagicMock()
    vault = MagicMock()
    profile_snapshot = ProfileKnowledgeSnapshot(
        store_identity="0" * 32,
        notes=(),
        source_count=0,
        invalid_count=0,
        truncated=False,
    )
    store.read_profile_knowledge_snapshot.return_value = profile_snapshot

    @contextmanager
    def hold_profile_snapshot():
        yield profile_snapshot

    @contextmanager
    def hold_profile_file():
        yield

    @contextmanager
    def hold_profile_ownership(memory_ids):
        yield store.profile_owned_memory_ids(memory_ids)

    store.hold_profile_knowledge_snapshot.side_effect = hold_profile_snapshot
    vault.profile_grounding_evidence_lock.side_effect = hold_profile_file
    store.profile_memory_ownership_egress_fence.side_effect = hold_profile_ownership
    store.profile_owned_memory_ids.return_value = frozenset()
    vault.read_profile_grounding.return_value = ProfileGroundingView(
        text=f"# Profile\n\n{PROFILE_MARKER}",
        verified_source_keys=(),
        invalid=False,
        truncated=False,
    )
    store.list_preferences.return_value = [
        {
            "category": "destination-smoke",
            "key": "private-preference",
            "value": PREFERENCE_MARKER,
        }
    ]
    store.search_memories.return_value = [
        {
            "id": 101,
            "category": "destination-smoke",
            "title": "Private memory",
            "body": MEMORY_MARKER,
        }
    ]
    store.recent_memories.return_value = []
    store.list_active_skills.return_value = [
        {
            "name": "Private destination skill",
            "trigger": "approach destination project",
            "body": SKILL_MARKER,
            "tags": "destination smoke",
            "origin": "user_authored",
        }
    ]
    brain = ChatBrain(model, store, vault, provider="ollama")
    brain.history = [{"role": "assistant", "content": LEGACY_HISTORY_MARKER}]
    return brain


def test_loopback_destinations_are_normalized_and_pinned_into_client() -> None:
    cases = (
        (
            "default",
            {},
            "http://127.0.0.1:11434",
            False,
            "default",
            "ipv4_loopback",
            11434,
        ),
        (
            "localhost pinned",
            {"OLLAMA_HOST": "localhost:21435"},
            "http://127.0.0.1:21435",
            True,
            "env",
            "localhost_pinned_ipv4",
            21435,
        ),
        (
            "127/8 loopback",
            {"OLLAMA_HOST": "http://127.83.14.9:21436"},
            "http://127.83.14.9:21436",
            True,
            "env",
            "ipv4_loopback",
            21436,
        ),
        (
            "IPv6 loopback",
            {"OLLAMA_HOST": "https://[0:0:0:0:0:0:0:1]:21437"},
            "https://[::1]:21437",
            True,
            "env",
            "ipv6_loopback",
            21437,
        ),
    )

    for label, environ, expected_host, configured, source, family, port in cases:
        destination = resolve_ollama_destination(environ)
        if (
            destination.allowed is not True
            or destination.base_url != expected_host
            or destination.configured is not configured
            or destination.source != source
            or destination.address_family != family
            or destination.port != port
            or destination.diagnostic != "local_loopback"
        ):
            raise AssertionError(f"{label} destination normalization drifted: {destination}")

        http_calls: list[tuple[object, float]] = []

        def fake_open(request, timeout):
            http_calls.append((request, timeout))
            return _FakeHTTPResponse(
                {
                    "done": True,
                    "done_reason": "stop",
                    "message": {"role": "assistant", "content": f"mocked {label} answer"},
                }
            )

        with (
            patch.dict(sys.modules, {"ollama": None, "httpx": None, "pydantic": None}),
            patch.object(model_provider_module.socket, "getaddrinfo", side_effect=_fail_network),
            patch.object(model_provider_module.socket, "create_connection", side_effect=_fail_network),
        ):
            answer = generate_model_text(
                provider="ollama",
                model="mocked-loopback-model",
                messages=[{"role": "user", "content": "mock-only request"}],
                timeout_seconds=3.25,
                environ=environ,
                urlopen_impl=fake_open,
            )

        if answer != f"mocked {label} answer" or len(http_calls) != 1:
            raise AssertionError(
                f"{label} did not complete through exactly one mocked stdlib HTTP call: "
                f"{answer!r} {http_calls}"
            )
        request, timeout = http_calls[0]
        payload = json.loads(request.data.decode("utf-8"))
        if (
            request.full_url != expected_host + "/api/chat"
            or request.get_method() != "POST"
            or timeout != 3.25
            or payload.get("stream") is not False
            or payload.get("messages") != [{"role": "user", "content": "mock-only request"}]
        ):
            raise AssertionError(
                f"{label} stdlib HTTP boundary drifted: {request.full_url!r} "
                f"{request.get_method()!r} {timeout!r} {payload!r}"
            )


def test_adversarial_destinations_fail_before_client_or_network() -> None:
    cases = (
        ("remote hostname", "private-remote-host.example.invalid:22100", "private-remote-host.example.invalid"),
        ("remote IPv4", "198.51.100.77:22101", "198.51.100.77"),
        ("IPv4 wildcard", "0.0.0.0:22102", "0.0.0.0"),
        ("IPv6 wildcard", "[::]:22103", "::"),
        ("mapped IPv6", "[::ffff:127.0.0.1]:22104", "::ffff:127.0.0.1"),
        ("userinfo", "http://private-user@127.0.0.1:22105", "private-user"),
        ("path", "http://127.0.0.1:22106/private-path-marker", "private-path-marker"),
        ("query", "http://127.0.0.1:22107?private-query-marker=1", "private-query-marker"),
        ("fragment", "http://127.0.0.1:22108#private-fragment-marker", "private-fragment-marker"),
        ("unsupported scheme", "ftp://127.0.0.1:22109", "ftp://127.0.0.1"),
        ("leading whitespace", " http://127.0.0.1:22110", "127.0.0.1:22110"),
        ("embedded whitespace", "http://127.0.0.1:22111 private-space-marker", "private-space-marker"),
        ("control character", "http://127.0.0.1:22112\nprivate-control-marker", "private-control-marker"),
        ("nonnumeric port", "127.0.0.1:private-port-marker", "private-port-marker"),
        ("zero port", "127.0.0.1:0", "127.0.0.1:0"),
        ("oversized port", "127.0.0.1:65536", "127.0.0.1:65536"),
    )

    client_calls: list[dict[str, object]] = []
    network_calls: list[str] = []

    class ForbiddenClient:
        def __init__(self, **kwargs):
            client_calls.append(dict(kwargs))
            raise AssertionError("blocked Ollama destination constructed a client")

    def forbidden_network(*_args, **_kwargs):
        network_calls.append("called")
        raise AssertionError("blocked Ollama destination reached a network primitive")

    with (
        patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=ForbiddenClient)}),
        patch.object(model_provider_module.socket, "getaddrinfo", side_effect=forbidden_network),
        patch.object(model_provider_module.socket, "create_connection", side_effect=forbidden_network),
    ):
        for label, raw_host, private_fragment in cases:
            environ = {"OLLAMA_HOST": raw_host}
            destination = resolve_ollama_destination(environ)
            receipt_text = json.dumps(destination.receipt(), sort_keys=True)
            destination_text = f"{destination!r} {receipt_text}"
            if destination.allowed is not False or destination.base_url:
                raise AssertionError(f"{label} destination was not blocked: {destination}")
            if raw_host in destination_text or private_fragment in destination_text:
                raise AssertionError(f"{label} destination receipt leaked the configured host: {destination_text}")
            if destination.receipt().get("ollama_destination_value_exposed") is not False:
                raise AssertionError(f"{label} destination receipt claimed value exposure: {destination.receipt()}")

            try:
                generate_model_text(
                    provider="ollama",
                    model="mocked-blocked-model",
                    messages=[{"role": "user", "content": "private request body"}],
                    timeout_seconds=1,
                    environ=environ,
                    urlopen_impl=forbidden_network,
                )
            except ModelProviderError as exc:
                error_text = f"{exc.diagnostic} {exc.recovery_hint}"
                if exc.diagnostic != "ollama_destination_not_local":
                    raise AssertionError(f"{label} failure diagnostic drifted: {exc!r}")
                if raw_host in error_text or private_fragment in error_text:
                    raise AssertionError(f"{label} failure leaked the configured host: {error_text}")
            else:
                raise AssertionError(f"{label} destination escaped the provider boundary")

    if client_calls or network_calls:
        raise AssertionError(
            f"blocked destinations created client/network calls: clients={client_calls}, network={network_calls}"
        )


def test_parser_edge_cases_are_blocked_without_value_disclosure() -> None:
    cases = (
        ("trailing colon", "127.83.14.9:", "127.83.14.9:", "invalid_port"),
        ("scheme only", "HTTPS://", "HTTPS://", "host_missing"),
        (
            "duplicate port",
            "127.83.14.10:21438:private-duplicate-port-marker",
            "private-duplicate-port-marker",
            "invalid_format",
        ),
        (
            "IPv6 zone",
            "[::1%private-zone-marker]:21439",
            "private-zone-marker",
            "scoped_address_not_allowed",
        ),
        (
            "localhost trailing dot",
            "localhost.:21440",
            "localhost.",
            "non_loopback_host",
        ),
    )

    client_calls: list[dict[str, object]] = []
    network_calls: list[str] = []

    class ForbiddenClient:
        def __init__(self, **kwargs):
            client_calls.append(dict(kwargs))
            raise AssertionError("malformed Ollama destination constructed a client")

    def forbidden_network(*_args, **_kwargs):
        network_calls.append("called")
        raise AssertionError("malformed Ollama destination reached a network primitive")

    policy_violations: list[str] = []
    with (
        patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=ForbiddenClient)}),
        patch.object(model_provider_module.socket, "getaddrinfo", side_effect=forbidden_network),
        patch.object(model_provider_module.socket, "create_connection", side_effect=forbidden_network),
    ):
        for label, raw_host, private_fragment, expected_diagnostic in cases:
            environ = {"OLLAMA_HOST": raw_host}
            destination = resolve_ollama_destination(environ)
            receipt_text = json.dumps(destination.receipt(), sort_keys=True)
            diagnostic_text = f"{destination.diagnostic} {receipt_text}"
            if destination.allowed is not False or destination.base_url:
                policy_violations.append(f"{label} was accepted")
                continue
            if destination.diagnostic != expected_diagnostic:
                policy_violations.append(
                    f"{label} diagnostic {destination.diagnostic!r} != {expected_diagnostic!r}"
                )
                continue
            if raw_host in diagnostic_text or private_fragment in diagnostic_text:
                raise AssertionError(f"{label} parser diagnostic exposed its raw value")
            if destination.receipt().get("ollama_destination_value_exposed") is not False:
                raise AssertionError(f"{label} parser receipt claimed value exposure")

            try:
                generate_model_text(
                    provider="ollama",
                    model="llama3.1",
                    messages=[{"role": "user", "content": "private parser request"}],
                    timeout_seconds=1,
                    environ=environ,
                    urlopen_impl=forbidden_network,
                )
            except ModelProviderError as exc:
                error_text = f"{exc.diagnostic} {exc.recovery_hint}"
                if exc.diagnostic != "ollama_destination_not_local":
                    raise AssertionError(f"{label} provider diagnostic drifted: {exc!r}")
                if raw_host in error_text or private_fragment in error_text:
                    raise AssertionError(f"{label} provider diagnostic exposed its raw value")
            else:
                raise AssertionError(f"{label} parser edge escaped the provider boundary")

    if client_calls or network_calls:
        raise AssertionError(
            f"parser edge cases created client/network calls: {client_calls} {network_calls}"
        )
    if policy_violations:
        raise AssertionError("parser edge cases escaped the boundary: " + ", ".join(policy_violations))


def test_cloud_aliases_fail_before_client_construction() -> None:
    aliases = (
        "gpt-oss:120b-cloud",
        "glm-4.7:cloud",
        "kimi-k2-thinking",
        "kimi-k2-thinking:latest",
    )
    client_calls: list[dict[str, object]] = []
    network_calls: list[str] = []

    class ForbiddenClient:
        def __init__(self, **kwargs):
            client_calls.append(dict(kwargs))
            raise AssertionError("cloud Ollama alias constructed a client")

    def forbidden_network(*_args, **_kwargs):
        network_calls.append("called")
        raise AssertionError("cloud Ollama alias reached a network primitive")

    environments = (
        {"OLLAMA_HOST": "127.0.0.1:11434"},
        {"OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_NO_CLOUD": "1"},
        {
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_NO_CLOUD": "1",
            "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
        },
    )
    with (
        patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=ForbiddenClient)}),
        patch.object(model_provider_module.socket, "getaddrinfo", side_effect=forbidden_network),
        patch.object(model_provider_module.socket, "create_connection", side_effect=forbidden_network),
    ):
        for environ in environments:
            for alias in aliases:
                try:
                    generate_model_text(
                        provider="ollama",
                        model=alias,
                        messages=[{"role": "user", "content": "private cloud-alias request"}],
                        timeout_seconds=1,
                        environ=environ,
                        urlopen_impl=forbidden_network,
                    )
                except ModelProviderError as exc:
                    if exc.diagnostic != "ollama_cloud_model_blocked":
                        raise AssertionError(f"{alias} cloud policy diagnostic drifted: {exc!r}")
                    if alias in f"{exc.diagnostic} {exc.recovery_hint}":
                        raise AssertionError(f"{alias} cloud policy diagnostic exposed the alias")
                else:
                    raise AssertionError(f"{alias} cloud alias escaped the provider boundary")

    if client_calls or network_calls:
        raise AssertionError(
            f"cloud aliases created client/network calls: {client_calls} {network_calls}"
        )


def test_chatbrain_cloud_aliases_omit_context_and_report_no_execution() -> None:
    aliases = (
        "gpt-oss:120b-cloud",
        "glm-4.7:cloud",
        "kimi-k2-thinking",
        "kimi-k2-thinking:latest",
    )
    environ = {
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_NO_CLOUD": "1",
        "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
    }
    client_calls: list[dict[str, object]] = []

    class ForbiddenClient:
        def __init__(self, **kwargs):
            client_calls.append(dict(kwargs))
            raise AssertionError("cloud-alias ChatBrain constructed an Ollama client")

    real_generate_model_text = model_provider_module.generate_model_text
    prompt = f"How should I approach this destination project? {CURRENT_PROMPT_MARKER}"
    with patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=ForbiddenClient)}):
        for alias in aliases:
            brain = _private_brain(model=alias)
            with patch.dict(os.environ, environ, clear=True):
                preview = brain.preview_loop(prompt)
            expected_preview = {
                "model_request_would_be_blocked_by_cloud_policy": True,
                "would_make_model_network_request": False,
                "stored_personal_context_explicit_consent": True,
                "stored_personal_context_policy_satisfied": False,
                "stored_personal_context_sources_to_include": [],
                "remote_personal_context_policy": "blocked_cloud_model",
                "external_processing_if_executed": "not_applicable",
                "history_retained_messages": 1,
                "history_eligible_messages": 0,
                "history_withheld_messages": 1,
                "history_withheld_reason_counts": {"legacy_unmarked": 1},
            }
            wrong_preview = {
                key: preview.get(key)
                for key, expected in expected_preview.items()
                if preview.get(key) != expected
            }
            if wrong_preview:
                raise AssertionError(
                    f"{alias} ChatBrain preview drifted: {wrong_preview} / {preview}"
                )

            with (
                patch.dict(os.environ, environ, clear=True),
                patch.object(
                    chat_module,
                    "generate_model_text",
                    side_effect=real_generate_model_text,
                ) as model_call,
            ):
                brain.respond(prompt)
            if model_call.call_count != 1:
                raise AssertionError(f"{alias} ChatBrain attempt count drifted")
            message_text = _message_text(model_call)
            leaked = [marker for marker in PERSONAL_MARKERS if marker in message_text]
            if leaked:
                raise AssertionError(f"{alias} cloud request contained stored context: {leaked}")
            expected_metadata = {
                "model_request_blocked_by_cloud_policy": True,
                "model_execution_status": "not_executed",
                "model_execution_occurred": False,
                "external_processing_status": "not_executed",
                "external_processing_occurred": False,
                "stored_personal_context_in_model_request": False,
                "personal_context_sources_in_model_request": [],
                "history_in_model_request": False,
                "remote_personal_context_policy": "blocked_cloud_model",
                "history_retained_messages": 1,
                "history_eligible_messages": 0,
                "history_withheld_messages": 1,
                "history_withheld_reason_counts": {"legacy_unmarked": 1},
            }
            wrong_metadata = {
                key: brain.last_turn_metadata.get(key)
                for key, expected in expected_metadata.items()
                if brain.last_turn_metadata.get(key) != expected
            }
            if wrong_metadata:
                raise AssertionError(
                    f"{alias} ChatBrain metadata drifted: "
                    f"{wrong_metadata} / {brain.last_turn_metadata}"
                )
    if client_calls:
        raise AssertionError(f"cloud-alias ChatBrain created clients: {client_calls}")


def test_ordinary_ollama_context_requires_no_cloud_and_locality_stays_unverified() -> None:
    cases = (
        (
            "stateless without no-cloud",
            {"OLLAMA_HOST": "127.0.0.1:11434"},
            False,
            "cloud_mode_not_disabled",
            "local_provider_stateless",
        ),
        (
            "no-cloud intent without explicit consent",
            {"OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_NO_CLOUD": "1"},
            False,
            "unverified_daemon_consent_required",
            "local_provider_unverified_context_consent_required",
        ),
        (
            "personal context with explicit consent",
            {
                "OLLAMA_HOST": "127.0.0.1:11434",
                "OLLAMA_NO_CLOUD": "1",
                "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
            },
            True,
            "unverified_local_only_consent_granted",
            "local_provider_unverified_context_consent_granted",
        ),
    )

    for label, environ, expect_personal_context, policy_diagnostic, context_policy in cases:
        client_calls: list[dict[str, object]] = []
        chat_calls: list[dict[str, object]] = []

        class FakeOpener:
            def open(self, request, *, timeout):
                payload = json.loads(request.data.decode("utf-8"))
                chat_calls.append(payload)
                if expect_personal_context and len(chat_calls) == 1:
                    content = ELIGIBLE_HISTORY_MARKER
                else:
                    content = f"mocked {label} answer"
                return _FakeHTTPResponse(
                    {
                        "done": True,
                        "done_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                )

        def fake_build_opener(*handlers):
            client_calls.append({"handlers": handlers})
            return FakeOpener()

        brain = _private_brain(model="llama3.1")
        policy: _DurableDisclosurePolicy | None = None
        prompt = f"How should I approach this destination project? {CURRENT_PROMPT_MARKER}"
        with (
            patch.dict(os.environ, environ, clear=True),
            patch.dict(sys.modules, {"ollama": None, "httpx": None, "pydantic": None}),
            patch.object(model_provider_module, "build_opener", fake_build_opener),
            patch.object(model_provider_module.socket, "getaddrinfo", side_effect=_fail_network),
            patch.object(model_provider_module.socket, "create_connection", side_effect=_fail_network),
        ):
            if expect_personal_context:
                policy = _install_durable_disclosure_policy(brain)
                seed_answer = brain.respond("Why is the sky blue?")
                if seed_answer != ELIGIBLE_HISTORY_MARKER:
                    raise AssertionError(f"{label} eligible-history seed failed: {seed_answer!r}")
            answer = brain.respond(prompt)

        expected_calls = 2 if expect_personal_context else 1
        if (
            answer != f"mocked {label} answer"
            or len(client_calls) != expected_calls
            or len(chat_calls) != expected_calls
        ):
            raise AssertionError(
                f"{label} did not complete one mocked request: "
                f"{answer!r} {client_calls} {chat_calls}"
            )
        messages = chat_calls[-1].get("messages")
        if not isinstance(messages, list):
            raise AssertionError(f"{label} missed inspectable messages: {chat_calls[0]}")
        message_text = "\n".join(str(message.get("content") or "") for message in messages)
        if CURRENT_PROMPT_MARKER not in message_text:
            raise AssertionError(f"{label} dropped the current prompt: {message_text}")
        present_personal_markers = [marker for marker in PERSONAL_MARKERS if marker in message_text]
        if expect_personal_context:
            expected_markers = (*STORED_CONTEXT_MARKERS, ELIGIBLE_HISTORY_MARKER)
            missing_markers = [marker for marker in expected_markers if marker not in message_text]
            if missing_markers:
                raise AssertionError(f"{label} omitted stored personal context: {missing_markers}")
            duplicated_markers = [
                marker for marker in expected_markers if message_text.count(marker) != 1
            ]
            if duplicated_markers:
                raise AssertionError(
                    f"{label} did not include stored context exactly once: {duplicated_markers}"
                )
            if LEGACY_HISTORY_MARKER in message_text:
                raise AssertionError(f"{label} disclosed legacy history: {message_text}")
        elif present_personal_markers:
            raise AssertionError(f"{label} leaked stored personal context: {present_personal_markers}")

        metadata = brain.last_turn_metadata
        expected_metadata = {
            "source": "model",
            "model_provider": "ollama",
            "model_destination_allowed": True,
            "model_request_blocked_by_destination_policy": False,
            "model_request_blocked_by_cloud_policy": False,
            "model_execution_location_policy": "loopback_daemon_execution_unverified",
            "model_execution_locality_verified": False,
            "current_message_on_device_verified": False,
            "ollama_no_cloud_configured": "OLLAMA_NO_CLOUD" in environ,
            "ollama_no_cloud_valid": True,
            "ollama_no_cloud_requested": "OLLAMA_NO_CLOUD" in environ,
            "ollama_cloud_model_alias": False,
            "ollama_personal_context_allowed": expect_personal_context,
            "ollama_unverified_context_consent_configured": (
                "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT" in environ
            ),
            "ollama_unverified_context_consent_valid": True,
            "ollama_unverified_context_consent_allowed": expect_personal_context,
            "ollama_execution_locality_verified": False,
            "ollama_cloud_policy_diagnostic": policy_diagnostic,
            "ollama_cloud_policy_value_exposed": False,
            "remote_personal_context_policy": context_policy,
            "stored_personal_context_explicit_consent": expect_personal_context,
            "stored_personal_context_policy_satisfied": expect_personal_context,
            "stored_personal_context_in_model_request": expect_personal_context,
            "history_in_model_request": expect_personal_context,
            "history_retained_messages": 3 if expect_personal_context else 1,
            "history_eligible_messages": 2 if expect_personal_context else 0,
            "history_withheld_messages": 1,
            "history_withheld_reason_counts": {"legacy_unmarked": 1},
            "external_processing_status": "unknown",
            "external_processing_occurred": None,
            "stored_personal_context_processed_externally": (
                None if expect_personal_context else False
            ),
            "personal_context_omitted_by_cloud_policy": not expect_personal_context,
            "model_request_content_in_metadata": False,
            "model_response_content_in_metadata": False,
        }
        wrong_metadata = {
            key: metadata.get(key)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if wrong_metadata:
            raise AssertionError(f"{label} metadata drifted: {wrong_metadata} / {metadata}")
        metadata_text = json.dumps(metadata, sort_keys=True)
        leaked_metadata_markers = [marker for marker in PERSONAL_MARKERS if marker in metadata_text]
        if leaked_metadata_markers:
            raise AssertionError(f"{label} metadata exposed personal context: {leaked_metadata_markers}")
        if expect_personal_context and (
            policy is None
            or policy.finalized != ["confirmed", "confirmed"]
            or policy.prepared
        ):
            raise AssertionError(
                f"{label} durable receipts did not finalize exactly: "
                f"{None if policy is None else policy.__dict__}"
            )


def test_chatbrain_blocked_destination_preview_and_fallback_are_private() -> None:
    raw_host = "private-chat-destination.example.invalid:22999"
    prompt = f"How should I approach this destination project? {CURRENT_PROMPT_MARKER}"
    brain = _private_brain()

    with patch.dict(os.environ, {"OLLAMA_HOST": raw_host}, clear=False):
        preview = brain.preview_loop(prompt)

    expected_preview = {
        "would_call_model": True,
        "would_make_model_network_request": False,
        "model_destination_allowed": False,
        "model_request_would_be_blocked_by_destination_policy": True,
        "model_destination_policy": "ollama_blocked_nonlocal",
        "ollama_destination_diagnostic": "non_loopback_host",
        "ollama_destination_value_exposed": False,
        "would_call_external_service": False,
        "would_share_current_message_with_external_model": False,
        "would_share_stored_personal_context_with_external_model": False,
        "would_share_history_with_external_model": False,
        "remote_personal_context_policy": "blocked_destination",
        "remote_personal_context_sources_to_share": [],
        "history_retained_messages": 1,
        "history_eligible_messages": 0,
        "history_withheld_messages": 1,
        "history_withheld_reason_counts": {"legacy_unmarked": 1},
    }
    wrong_preview = {
        key: preview.get(key)
        for key, expected in expected_preview.items()
        if preview.get(key) != expected
    }
    if wrong_preview:
        raise AssertionError(f"blocked ChatBrain preview metadata drifted: {wrong_preview} / {preview}")
    if raw_host in json.dumps(preview, sort_keys=True):
        raise AssertionError(f"blocked ChatBrain preview leaked the configured host: {preview}")

    client_calls: list[dict[str, object]] = []
    network_calls: list[str] = []

    class ForbiddenClient:
        def __init__(self, **kwargs):
            client_calls.append(dict(kwargs))
            raise AssertionError("blocked ChatBrain response constructed an Ollama client")

    def forbidden_network(*_args, **_kwargs):
        network_calls.append("called")
        raise AssertionError("blocked ChatBrain response reached the network")

    real_generate_model_text = model_provider_module.generate_model_text

    def capture_attempt(**kwargs):
        return real_generate_model_text(**kwargs)

    with (
        patch.dict(os.environ, {"OLLAMA_HOST": raw_host}, clear=False),
        patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=ForbiddenClient)}),
        patch.object(model_provider_module.socket, "getaddrinfo", side_effect=forbidden_network),
        patch.object(model_provider_module.socket, "create_connection", side_effect=forbidden_network),
        patch.object(chat_module, "generate_model_text", side_effect=capture_attempt) as model_call,
    ):
        answer = brain.respond(prompt)

    if model_call.call_count != 1:
        raise AssertionError(f"blocked ChatBrain response attempt count drifted: {model_call.call_count}")
    attempted_message_text = _message_text(model_call)
    if CURRENT_PROMPT_MARKER not in attempted_message_text:
        raise AssertionError(f"blocked ChatBrain attempt dropped the current message: {attempted_message_text}")
    leaked_markers = [marker for marker in PERSONAL_MARKERS if marker in attempted_message_text]
    if leaked_markers:
        raise AssertionError(f"blocked ChatBrain attempt included personal context: {leaked_markers}")
    if client_calls or network_calls:
        raise AssertionError(
            f"blocked ChatBrain response created client/network calls: {client_calls} {network_calls}"
        )

    expected_fallback = {
        "source": "fallback",
        "used_model": False,
        "used_fallback": True,
        "model_provider": "ollama",
        "model_call_attempted": True,
        "calls_model": True,
        "calls_external_service": False,
        "external_model_request_attempted": False,
        "external_model_response_received": False,
        "model_destination_allowed": False,
        "model_request_blocked_by_destination_policy": True,
        "model_destination_policy": "ollama_blocked_nonlocal",
        "ollama_destination_diagnostic": "non_loopback_host",
        "ollama_destination_value_exposed": False,
        "ollama_redirects_allowed": False,
        "ollama_proxy_environment_allowed": False,
        "shares_conversation_with_external_model": False,
        "current_user_message_shared_with_external_model": False,
        "shares_stored_personal_context_with_external_model": False,
        "shares_history_with_external_model": False,
        "personal_context_omitted_from_remote_model": False,
        "personal_context_omitted_by_destination_policy": True,
        "remote_personal_context_policy": "blocked_destination",
        "remote_personal_context_sources_shared": [],
        "history_retained_messages": 1,
        "history_eligible_messages": 0,
        "history_withheld_messages": 1,
        "history_withheld_reason_counts": {"legacy_unmarked": 1},
        "model_request_content_in_metadata": False,
        "model_response_content_in_metadata": False,
        "model_error": "ollama_destination_not_local",
        "model_exception_type": "ModelProviderError",
    }
    metadata = brain.last_turn_metadata
    wrong_fallback = {
        key: metadata.get(key)
        for key, expected in expected_fallback.items()
        if metadata.get(key) != expected
    }
    if wrong_fallback:
        raise AssertionError(f"blocked ChatBrain fallback metadata drifted: {wrong_fallback} / {metadata}")
    metadata_text = json.dumps(metadata, sort_keys=True)
    if raw_host in answer or raw_host in metadata_text:
        raise AssertionError("blocked ChatBrain fallback leaked the configured host")
    for expected in ("OLLAMA_HOST", "loopback", "Diagnostic: ollama_destination_not_local"):
        if expected not in answer:
            raise AssertionError(f"blocked ChatBrain fallback missed destination recovery guidance {expected!r}: {answer!r}")
    for forbidden in ("Start Ollama", "ollama pull", "Diagnostic: model_unavailable"):
        if forbidden in answer:
            raise AssertionError(f"blocked ChatBrain fallback gave misleading Ollama recovery guidance {forbidden!r}: {answer!r}")
    leaked_metadata_markers = [marker for marker in PERSONAL_MARKERS if marker in metadata_text]
    if leaked_metadata_markers:
        raise AssertionError(f"blocked ChatBrain metadata included personal content: {leaked_metadata_markers}")


def main() -> None:
    test_loopback_destinations_are_normalized_and_pinned_into_client()
    test_adversarial_destinations_fail_before_client_or_network()
    test_parser_edge_cases_are_blocked_without_value_disclosure()
    test_cloud_aliases_fail_before_client_construction()
    test_chatbrain_cloud_aliases_omit_context_and_report_no_execution()
    test_ordinary_ollama_context_requires_no_cloud_and_locality_stays_unverified()
    test_chatbrain_blocked_destination_preview_and_fallback_are_private()
    print("Ollama destination smoke passed")


if __name__ == "__main__":
    main()
