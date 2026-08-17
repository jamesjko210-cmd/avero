from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from jarvis_v2.agent import chat as chat_module
from jarvis_v2.agent.chat import ChatBrain
from jarvis_v2.memory.obsidian import ProfileGroundingView
from jarvis_v2.memory.store import ProfileKnowledgeSnapshot
from jarvis_v2.tools.conversation import make_conversation_tools


CURRENT_PROMPT_MARKER = "CURRENT_PROMPT_PRIVACY_MARKER"
PROFILE_MARKER = "PROFILE_PRIVACY_MARKER"
PREFERENCE_MARKER = "PREFERENCE_PRIVACY_MARKER"
MEMORY_MARKER = "MEMORY_PRIVACY_MARKER"
SKILL_MARKER = "SKILL_PRIVACY_MARKER"
ELIGIBLE_HISTORY_MARKER = "ELIGIBLE_HISTORY_PRIVACY_MARKER"
LEGACY_HISTORY_MARKER = "LEGACY_HISTORY_PRIVACY_MARKER"
PERSONAL_MARKERS = (
    PROFILE_MARKER,
    PREFERENCE_MARKER,
    MEMORY_MARKER,
    SKILL_MARKER,
    ELIGIBLE_HISTORY_MARKER,
    LEGACY_HISTORY_MARKER,
)
ELIGIBLE_PERSONAL_MARKERS = (
    PROFILE_MARKER,
    PREFERENCE_MARKER,
    MEMORY_MARKER,
    SKILL_MARKER,
    ELIGIBLE_HISTORY_MARKER,
)
PROMPT = f"Help me approach this project. {CURRENT_PROMPT_MARKER}"
CONTEXT_SOURCES = ["profile", "preferences", "memory", "skills"]
OPENAI_DESTINATION_CLASS = "external_provider"
OLLAMA_UNVERIFIED_LOOPBACK_DESTINATION_CLASS = "loopback_daemon_unverified"
OLLAMA_LOOPBACK_HOST = "http://127.0.0.1:11434"
OLLAMA_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class _DurableDisclosurePolicy:
    def __init__(
        self,
        brain: ChatBrain,
        *,
        provider: str,
        destination_class: str,
    ) -> None:
        decision = brain._history_policy_decision(commit=False)
        if (
            decision.provider != provider
            or decision.destination_class != destination_class
            or decision.stored_context_allowed is not True
        ):
            raise AssertionError(f"durable policy installed for the wrong route: {decision}")
        self.brain = brain
        self.provider = provider
        self.destination_class = destination_class
        self.active_epoch_id = decision.epoch_id
        self.active_fingerprint = decision.fingerprint
        self.binding = object()
        self.prepared: dict[str, tuple[object, str, str]] = {}
        self.finalized: list[str] = []

    def prepare(self, payload: dict[str, object]) -> object:
        expected_keys = {
            "session_generation",
            "policy_epoch_id",
            "policy_fingerprint",
            "provider",
            "destination_class",
            "source_count",
            "source_digest",
        }
        decision = self.brain._history_policy_decision(commit=False)
        source_count = payload.get("source_count")
        source_digest = payload.get("source_digest")
        if (
            type(payload) is not dict
            or set(payload) != expected_keys
            or payload.get("session_generation") != self.brain._session_generation
            or payload.get("policy_epoch_id") != self.active_epoch_id
            or payload.get("policy_epoch_id") != decision.epoch_id
            or payload.get("policy_fingerprint") != self.active_fingerprint
            or payload.get("policy_fingerprint") != decision.fingerprint
            or payload.get("provider") != self.provider
            or payload.get("provider") != decision.provider
            or payload.get("destination_class") != self.destination_class
            or payload.get("destination_class") != decision.destination_class
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

    def rotate_epoch(self) -> None:
        replacement = "f" * 64
        self.active_epoch_id = replacement if replacement != self.active_epoch_id else "e" * 64


def _install_durable_disclosure_policy(
    brain: ChatBrain,
    *,
    provider: str,
    destination_class: str,
) -> _DurableDisclosurePolicy:
    policy = _DurableDisclosurePolicy(
        brain,
        provider=provider,
        destination_class=destination_class,
    )
    brain.install_history_disclosure_callbacks(
        prepare=policy.prepare,
        validate=policy.validate,
        finalize=policy.finalize,
        fence=lambda _handle: nullcontext(),
    )
    return policy


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


def _brain(*, provider: str, allow_remote_personal_context: bool = False) -> ChatBrain:
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
        text=f"# Profile\n\n## Privacy smoke\n{PROFILE_MARKER}",
        verified_source_keys=(),
        invalid=False,
        truncated=False,
    )
    store.list_preferences.return_value = [
        {
            "category": "privacy-smoke",
            "key": "response-style",
            "value": PREFERENCE_MARKER,
        }
    ]
    store.search_memories.return_value = [
        {
            "id": 101,
            "category": "privacy-smoke",
            "title": "Provider boundary",
            "body": MEMORY_MARKER,
        }
    ]
    store.recent_memories.return_value = []
    store.list_active_skills.return_value = [
        {
            "name": "Project approach",
            "trigger": "approach this project",
            "body": SKILL_MARKER,
            "tags": "privacy smoke",
            "origin": "user_authored",
        }
    ]
    brain = ChatBrain(
        "mocked-chat-model",
        store,
        vault,
        provider=provider,
        allow_remote_personal_context=allow_remote_personal_context,
    )
    brain.history = [{"role": "assistant", "content": LEGACY_HISTORY_MARKER}]
    return brain


def _seed_eligible_history(brain: ChatBrain) -> None:
    with patch.object(
        chat_module,
        "generate_model_text",
        return_value=ELIGIBLE_HISTORY_MARKER,
    ) as call:
        answer = brain.respond("Why is the sky blue?")
    if answer != ELIGIBLE_HISTORY_MARKER or call.call_count != 1:
        raise AssertionError("eligible-history seed did not complete one mocked ChatBrain turn")


def _message_text(call: MagicMock) -> str:
    messages = call.call_args.kwargs.get("messages")
    if not isinstance(messages, list):
        raise AssertionError(f"chat provider call missed inspectable messages: {call.call_args}")
    return "\n".join(str(message.get("content") or "") for message in messages)


def _assert_metadata(brain: ChatBrain, expected: dict[str, object], label: str) -> None:
    wrong = {
        key: brain.last_turn_metadata.get(key)
        for key, value in expected.items()
        if brain.last_turn_metadata.get(key) != value
    }
    if wrong:
        raise AssertionError(f"{label} metadata drifted: {wrong} / {brain.last_turn_metadata}")


def test_openai_gate_off_keeps_only_current_prompt() -> None:
    brain = _brain(provider="openai")
    with patch.object(chat_module, "generate_model_text", return_value="mocked remote answer") as call:
        answer = brain.respond(PROMPT)

    if answer != "mocked remote answer" or call.call_count != 1:
        raise AssertionError(f"OpenAI gate-off chat did not use exactly one mocked call: {answer!r}")
    message_text = _message_text(call)
    if CURRENT_PROMPT_MARKER not in message_text:
        raise AssertionError(f"OpenAI gate off dropped the current prompt: {message_text}")
    leaked = [marker for marker in PERSONAL_MARKERS if marker in message_text]
    if leaked:
        raise AssertionError(f"OpenAI gate off leaked stored personal context: {leaked}")
    if "stored profile, preferences, memory, saved skills" not in message_text:
        raise AssertionError(f"OpenAI gate off missed its remote-context instruction: {message_text}")
    _assert_metadata(
        brain,
        {
            "source": "model",
            "model_provider": "openai",
            "calls_external_service": True,
            "model_execution_status": "response_received",
            "model_execution_occurred": True,
            "external_processing_status": "confirmed",
            "external_processing_occurred": True,
            "current_user_message_processed_externally": True,
            "current_user_message_shared_with_external_model": True,
            "shares_stored_personal_context_with_external_model": False,
            "shares_history_with_external_model": False,
            "remote_personal_context_allowed": False,
            "remote_personal_context_policy": "disabled",
            "remote_personal_context_sources_shared": [],
            "personal_context_omitted_from_remote_model": True,
            "stored_personal_context_processed_externally": False,
            "history_retained_messages": 1,
            "history_eligible_messages": 0,
            "history_withheld_messages": 1,
            "history_withheld_reason_counts": {"legacy_unmarked": 1},
        },
        "OpenAI gate off",
    )


def test_openai_gate_on_includes_only_eligible_history() -> None:
    brain = _brain(provider="openai", allow_remote_personal_context=True)
    policy = _install_durable_disclosure_policy(
        brain,
        provider="openai",
        destination_class=OPENAI_DESTINATION_CLASS,
    )
    _seed_eligible_history(brain)
    with patch.object(chat_module, "generate_model_text", return_value="mocked opted-in answer") as call:
        brain.respond(PROMPT)

    if call.call_count != 1:
        raise AssertionError("OpenAI gate-on chat did not use exactly one mocked call")
    message_text = _message_text(call)
    missing = [
        marker
        for marker in (CURRENT_PROMPT_MARKER, *ELIGIBLE_PERSONAL_MARKERS)
        if marker not in message_text
    ]
    if missing:
        raise AssertionError(f"OpenAI gate on omitted opted-in context: {missing}")
    if message_text.count(ELIGIBLE_HISTORY_MARKER) != 1:
        raise AssertionError(f"OpenAI gate on duplicated eligible history: {message_text}")
    if LEGACY_HISTORY_MARKER in message_text:
        raise AssertionError(f"OpenAI gate on disclosed legacy history: {message_text}")
    _assert_metadata(
        brain,
        {
            "shares_stored_personal_context_with_external_model": True,
            "shares_history_with_external_model": True,
            "remote_personal_context_allowed": True,
            "remote_personal_context_policy": "enabled",
            "remote_personal_context_sources_shared": CONTEXT_SOURCES,
            "personal_context_omitted_from_remote_model": False,
            "external_processing_status": "confirmed",
            "external_processing_occurred": True,
            "stored_personal_context_processed_externally": True,
            "history_processed_externally": True,
            "history_retained_messages": 3,
            "history_eligible_messages": 2,
            "history_withheld_messages": 1,
            "history_withheld_reason_counts": {"legacy_unmarked": 1},
            "history_disclosure_prepare_status": "prepared",
            "history_disclosure_finalize_status": "confirmed",
        },
        "OpenAI gate on",
    )
    if policy.finalized != ["confirmed", "confirmed"] or policy.prepared:
        raise AssertionError(f"OpenAI durable receipts did not finalize exactly: {policy.__dict__}")


def test_profile_change_after_receipt_prepare_strips_stale_context() -> None:
    brain = _brain(provider="openai", allow_remote_personal_context=True)
    policy = _install_durable_disclosure_policy(
        brain,
        provider="openai",
        destination_class=OPENAI_DESTINATION_CLASS,
    )
    original_prepare = policy.prepare

    def prepare_then_change_profile(payload: dict[str, object]) -> object:
        handle = original_prepare(payload)
        brain.vault.read_profile_grounding.return_value = ProfileGroundingView(
            text="# Profile\n\n## Changed during prepare\nnew snapshot",
            verified_source_keys=(),
            invalid=False,
            truncated=False,
        )
        return handle

    brain.install_history_disclosure_callbacks(
        prepare=prepare_then_change_profile,
        validate=policy.validate,
        finalize=policy.finalize,
        fence=lambda _handle: nullcontext(),
    )
    with patch.object(
        chat_module,
        "generate_model_text",
        return_value="mocked stateless retry",
    ) as call:
        answer = brain.respond(PROMPT)
    if answer != "mocked stateless retry" or call.call_count != 1:
        raise AssertionError(f"profile snapshot drift did not retry statelessly: {answer!r}")
    message_text = _message_text(call)
    leaked = [marker for marker in ELIGIBLE_PERSONAL_MARKERS if marker in message_text]
    if leaked or CURRENT_PROMPT_MARKER not in message_text:
        raise AssertionError(f"profile snapshot drift leaked stale context: {leaked} / {message_text}")
    if policy.finalized != ["blocked"] or policy.prepared:
        raise AssertionError(f"profile snapshot drift receipt was not blocked: {policy.__dict__}")
    metadata = brain.last_turn_metadata
    if (
        metadata.get("history_disclosure_prepare_status") != "failed_stripped"
        or metadata.get("history_disclosure_finalize_status") != "blocked"
        or metadata.get("stored_personal_context_processed_externally") is not False
    ):
        raise AssertionError(f"profile snapshot drift metadata claimed disclosure: {metadata}")


def test_ollama_explicit_consent_includes_context_exactly_once() -> None:
    brain = _brain(provider="ollama")
    with _local_ollama_env(stored_history=True):
        policy = _install_durable_disclosure_policy(
            brain,
            provider="ollama",
            destination_class=OLLAMA_UNVERIFIED_LOOPBACK_DESTINATION_CLASS,
        )
        _seed_eligible_history(brain)
    with (
        _local_ollama_env(stored_history=True),
        patch.object(chat_module, "generate_model_text", return_value="mocked local answer") as call,
    ):
        brain.respond(PROMPT)

    if call.call_count != 1:
        raise AssertionError("Ollama chat did not use exactly one mocked call")
    message_text = _message_text(call)
    missing = [
        marker
        for marker in (CURRENT_PROMPT_MARKER, *ELIGIBLE_PERSONAL_MARKERS)
        if marker not in message_text
    ]
    if missing:
        raise AssertionError(f"Ollama lost its existing local context: {missing}")
    duplicated = [
        marker for marker in ELIGIBLE_PERSONAL_MARKERS if message_text.count(marker) != 1
    ]
    if duplicated:
        raise AssertionError(f"Ollama did not include consented context exactly once: {duplicated}")
    if LEGACY_HISTORY_MARKER in message_text:
        raise AssertionError(f"Ollama disclosed legacy history: {message_text}")
    _assert_metadata(
        brain,
        {
            "model_provider": "ollama",
            "calls_external_service": False,
            "shares_conversation_with_external_model": False,
            "shares_stored_personal_context_with_external_model": False,
            "shares_history_with_external_model": False,
            "stored_personal_context_explicit_consent": True,
            "stored_personal_context_policy_satisfied": True,
            "stored_personal_context_in_model_request": True,
            "personal_context_sources_in_model_request": CONTEXT_SOURCES,
            "history_in_model_request": True,
            "external_processing_status": "unknown",
            "external_processing_occurred": None,
            "stored_personal_context_processed_externally": None,
            "remote_personal_context_policy": "local_provider_unverified_context_consent_granted",
            "remote_personal_context_sources_shared": [],
            "personal_context_omitted_from_remote_model": False,
            "history_retained_messages": 3,
            "history_eligible_messages": 2,
            "history_withheld_messages": 1,
            "history_withheld_reason_counts": {"legacy_unmarked": 1},
            "history_disclosure_prepare_status": "prepared",
            "history_disclosure_finalize_status": "confirmed",
        },
        "Ollama local context",
    )
    if policy.finalized != ["confirmed", "confirmed"] or policy.prepared:
        raise AssertionError(f"Ollama durable receipts did not finalize exactly: {policy.__dict__}")


def test_missing_or_stale_durable_policy_strips_all_personal_context() -> None:
    cases = (
        ("OpenAI missing callbacks", "openai", None, False),
        ("OpenAI stale epoch", "openai", OPENAI_DESTINATION_CLASS, True),
        ("Ollama missing callbacks", "ollama", None, False),
        (
            "Ollama stale epoch",
            "ollama",
            OLLAMA_UNVERIFIED_LOOPBACK_DESTINATION_CLASS,
            True,
        ),
    )
    for label, provider, destination_class, rotate_epoch in cases:
        brain = _brain(
            provider=provider,
            allow_remote_personal_context=provider == "openai",
        )
        environment = (
            _local_ollama_env(stored_history=True)
            if provider == "ollama"
            else nullcontext()
        )
        with environment:
            policy = (
                _install_durable_disclosure_policy(
                    brain,
                    provider=provider,
                    destination_class=str(destination_class),
                )
                if destination_class is not None
                else None
            )
            _seed_eligible_history(brain)
            if policy is not None and rotate_epoch:
                policy.rotate_epoch()
            with patch.object(
                chat_module,
                "generate_model_text",
                return_value="mocked stripped answer",
            ) as call:
                brain.respond(PROMPT)

        message_text = _message_text(call)
        if CURRENT_PROMPT_MARKER not in message_text:
            raise AssertionError(f"{label} dropped the current prompt: {message_text}")
        leaked = [marker for marker in PERSONAL_MARKERS if marker in message_text]
        if leaked:
            raise AssertionError(f"{label} disclosed stored personal context: {leaked}")
        _assert_metadata(
            brain,
            {
                "stored_personal_context_in_model_request": False,
                "personal_context_sources_in_model_request": [],
                "history_in_model_request": False,
                "history_disclosure_prepare_status": "failed_stripped",
                "history_disclosure_receipt_prepared": False,
                "history_disclosure_callback_content_in_metadata": False,
            },
            label,
        )


def test_invalid_provider_blocks_before_provider_call_and_previews_explicitly() -> None:
    brain = _brain(provider="not-a-provider", allow_remote_personal_context=True)
    body, preview = brain.preview_loop_text(PROMPT)
    expected_preview = {
        "model_provider": "invalid",
        "model_provider_valid": False,
        "model_request_would_be_blocked_by_invalid_provider": True,
        "model_request_would_be_blocked_by_destination_policy": False,
        "model_destination_policy": "invalid_provider",
        "model_execution_location_policy": "not_applicable_invalid_provider",
        "would_make_model_network_request": False,
        "external_processing_if_executed": "not_applicable",
        "stored_personal_context_policy_satisfied": False,
        "stored_personal_context_sources_to_include": [],
        "history_retained_messages": 1,
        "history_eligible_messages": 0,
        "history_withheld_messages": 1,
        "history_withheld_reason_counts": {"legacy_unmarked": 1},
    }
    wrong = {
        key: preview.get(key)
        for key, expected in expected_preview.items()
        if preview.get(key) != expected
    }
    if wrong:
        raise AssertionError(f"invalid-provider preview drifted: {wrong} / {preview}")
    for expected_text in (
        "model destination policy: invalid_provider",
        "model execution location policy: not_applicable_invalid_provider",
        "preview execution status: not_executed",
    ):
        if expected_text not in body:
            raise AssertionError(f"invalid-provider preview hid {expected_text!r}: {body}")

    with patch.object(
        chat_module,
        "generate_model_text",
        side_effect=AssertionError("invalid provider reached provider function"),
    ) as call:
        answer = brain.respond(PROMPT)
    if call.call_count:
        raise AssertionError("invalid provider called the provider function")
    if "configured provider is invalid" not in answer or "no model request was made" not in answer:
        raise AssertionError(f"invalid-provider fallback was not truthful: {answer}")
    _assert_metadata(
        brain,
        {
            "source": "fallback",
            "model_provider": "invalid",
            "model_provider_valid": False,
            "model_request_blocked_by_invalid_provider": True,
            "model_request_blocked_by_destination_policy": False,
            "model_call_attempted": False,
            "calls_model": False,
            "calls_external_service": False,
            "model_execution_status": "not_executed",
            "model_execution_occurred": False,
            "external_processing_status": "not_executed",
            "external_processing_occurred": False,
            "stored_personal_context_in_model_request": False,
            "personal_context_sources_in_model_request": [],
            "history_in_model_request": False,
            "history_retained_messages": 1,
            "history_eligible_messages": 0,
            "history_withheld_messages": 1,
            "history_withheld_reason_counts": {"legacy_unmarked": 1},
            "model_error": "model_provider_invalid",
        },
        "invalid provider",
    )


def test_deterministic_routes_never_call_provider() -> None:
    cases = (
        (
            _brain(provider="openai"),
            "What do you know about me and this project?",
            "grounded_memory",
        ),
        (
            _brain(provider="openai", allow_remote_personal_context=True),
            "Please run this project script",
            "safety_preflight_guidance",
        ),
    )
    with patch.object(
        chat_module,
        "generate_model_text",
        side_effect=AssertionError("deterministic chat route called the provider"),
    ) as call:
        for brain, prompt, expected_source in cases:
            brain.respond(prompt)
            _assert_metadata(
                brain,
                {
                    "source": expected_source,
                    "model_call_attempted": False,
                    "calls_model": False,
                    "calls_external_service": False,
                    "model_execution_status": "not_executed",
                    "model_execution_occurred": False,
                    "external_processing_status": "not_executed",
                    "external_processing_occurred": False,
                    "shares_conversation_with_external_model": False,
                    "current_user_message_shared_with_external_model": False,
                    "shares_stored_personal_context_with_external_model": False,
                    "shares_history_with_external_model": False,
                    "remote_personal_context_sources_shared": [],
                    "personal_context_omitted_from_remote_model": False,
                },
                expected_source,
            )
    if call.call_count:
        raise AssertionError(f"deterministic routes called the provider {call.call_count} time(s)")


def test_preview_matrix_reports_provider_boundary_truthfully() -> None:
    openai_gate_on = _brain(provider="openai", allow_remote_personal_context=True)
    _install_durable_disclosure_policy(
        openai_gate_on,
        provider="openai",
        destination_class=OPENAI_DESTINATION_CLASS,
    )
    _seed_eligible_history(openai_gate_on)
    cases = (
        (
            "OpenAI gate off",
            _brain(provider="openai"),
            PROMPT,
            {
                "would_call_model": True,
                "would_call_external_service": True,
                "would_share_current_message_with_external_model": True,
                "would_share_stored_personal_context_with_external_model": False,
                "would_share_history_with_external_model": False,
                "remote_personal_context_policy": "disabled",
                "remote_personal_context_sources_to_share": [],
            },
        ),
        (
            "OpenAI gate on",
            openai_gate_on,
            PROMPT,
            {
                "would_call_model": True,
                "would_call_external_service": True,
                "would_share_current_message_with_external_model": True,
                "would_share_stored_personal_context_with_external_model": True,
                "would_share_history_with_external_model": True,
                "remote_personal_context_policy": "enabled",
                "remote_personal_context_sources_to_share": CONTEXT_SOURCES,
                "history_retained_messages": 3,
                "history_eligible_messages": 2,
                "history_withheld_messages": 1,
                "history_withheld_reason_counts": {"legacy_unmarked": 1},
            },
        ),
        (
            "Ollama local",
            _brain(provider="ollama"),
            PROMPT,
            {
                "would_call_model": True,
                "would_call_external_service": False,
                "would_share_current_message_with_external_model": False,
                "would_share_stored_personal_context_with_external_model": False,
                "would_share_history_with_external_model": False,
                "remote_personal_context_policy": "local_provider_stateless",
                "remote_personal_context_sources_to_share": [],
            },
        ),
        (
            "OpenAI deterministic grounded",
            _brain(provider="openai", allow_remote_personal_context=True),
            "What do you know about me and this project?",
            {
                "would_call_model": False,
                "would_call_external_service": False,
                "would_share_current_message_with_external_model": False,
                "would_share_stored_personal_context_with_external_model": False,
                "would_share_history_with_external_model": False,
                "remote_personal_context_policy": "enabled",
                "remote_personal_context_sources_to_share": [],
            },
        ),
        (
            "OpenAI deterministic safety",
            _brain(provider="openai"),
            "Please run this project script",
            {
                "would_call_model": False,
                "would_call_external_service": False,
                "would_share_current_message_with_external_model": False,
                "would_share_stored_personal_context_with_external_model": False,
                "would_share_history_with_external_model": False,
                "remote_personal_context_policy": "disabled",
                "remote_personal_context_sources_to_share": [],
            },
        ),
    )
    for label, brain, prompt, expected in cases:
        environment = (
            _local_ollama_env(stored_history=False)
            if brain.provider == "ollama"
            else nullcontext()
        )
        with environment:
            body, metadata = brain.preview_loop_text(prompt)
        wrong = {key: metadata.get(key) for key, value in expected.items() if metadata.get(key) != value}
        if wrong:
            raise AssertionError(f"{label} preview metadata drifted: {wrong} / {metadata}")
        text_expectations = {
            "configured provider": metadata["model_provider"],
            "would call an external model": "yes" if expected["would_call_external_service"] else "no",
            "would share this current message externally": (
                "yes" if expected["would_share_current_message_with_external_model"] else "no"
            ),
            "would share stored profile/preferences/memory/skills externally": (
                "yes"
                if expected["would_share_stored_personal_context_with_external_model"]
                else "no"
            ),
            "would share prior chat history externally": (
                "yes" if expected["would_share_history_with_external_model"] else "no"
            ),
            "remote stored-context opt-in": expected["remote_personal_context_policy"],
        }
        for field, value in text_expectations.items():
            if f"- {field}: {value}" not in body:
                raise AssertionError(f"{label} text preview misstated {field!r}: {body}")


def test_registered_previews_use_live_provider_gate_and_history() -> None:
    for allowed in (False, True):
        brain = _brain(
            provider="openai",
            allow_remote_personal_context=allowed,
        )
        if allowed:
            _install_durable_disclosure_policy(
                brain,
                provider="openai",
                destination_class=OPENAI_DESTINATION_CLASS,
            )
            _seed_eligible_history(brain)
        config = SimpleNamespace(
            chat_model=brain.model,
            chat_timeout_seconds=brain.model_timeout_seconds,
            chat_max_reply_tokens=brain.max_reply_tokens,
            chat_max_history_messages=brain.max_history_messages,
            model_provider=brain.provider,
            chat_reasoning_effort=brain.reasoning_effort,
            openai_max_output_tokens=brain.openai_max_output_tokens,
            allow_remote_personal_context=allowed,
        )
        handlers = make_conversation_tools(
            brain.store,
            brain.vault,
            "privacy-preview-session",
            config,
            lambda: brain,
        )
        by_name = {handler.__name__: handler for handler in handlers}
        with patch.object(
            chat_module,
            "generate_model_text",
            side_effect=AssertionError("registered preview called the provider"),
        ) as call:
            loop_result = by_name["chat_loop_preview"]({"prompt": PROMPT})
            prompt_result = by_name["chat_prompt_preview"]({"prompt": PROMPT})
        if call.call_count:
            raise AssertionError("registered chat preview called the provider")
        for label, result in (("loop", loop_result), ("prompt", prompt_result)):
            if not result.ok:
                raise AssertionError(f"registered {label} preview failed: {result.output}")
            if result.metadata.get("model_provider") != "openai":
                raise AssertionError(f"registered {label} preview lost runtime provider: {result.metadata}")
            if result.metadata.get("would_share_stored_personal_context_with_external_model") is not allowed:
                raise AssertionError(f"registered {label} preview misstated context gate: {result.metadata}")
            if result.metadata.get("would_share_history_with_external_model") is not allowed:
                raise AssertionError(f"registered {label} preview ignored live history/gate: {result.metadata}")
            if label == "loop":
                expected_history = (3, 2, 1) if allowed else (1, 0, 1)
                actual_history = (
                    result.metadata.get("history_retained_messages"),
                    result.metadata.get("history_eligible_messages"),
                    result.metadata.get("history_withheld_messages"),
                )
                if actual_history != expected_history:
                    raise AssertionError(
                        f"registered {label} preview history counts drifted: "
                        f"{actual_history} != {expected_history}"
                    )
                if result.metadata.get("history_withheld_reason_counts") != {
                    "legacy_unmarked": 1
                }:
                    raise AssertionError(
                        f"registered {label} preview history reasons drifted: {result.metadata}"
                    )
            if f"remote stored-context opt-in: {'enabled' if allowed else 'disabled'}" not in result.output:
                raise AssertionError(f"registered {label} preview hid gate state: {result.output}")


def main() -> None:
    test_openai_gate_off_keeps_only_current_prompt()
    test_openai_gate_on_includes_only_eligible_history()
    test_profile_change_after_receipt_prepare_strips_stale_context()
    test_ollama_explicit_consent_includes_context_exactly_once()
    test_missing_or_stale_durable_policy_strips_all_personal_context()
    test_invalid_provider_blocks_before_provider_call_and_previews_explicitly()
    test_deterministic_routes_never_call_provider()
    test_preview_matrix_reports_provider_boundary_truthfully()
    test_registered_previews_use_live_provider_gate_and_history()
    print("chat provider privacy smoke passed")


if __name__ == "__main__":
    main()
