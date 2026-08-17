from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import threading
from unittest.mock import MagicMock, patch

from jarvis_v2.agent import chat as chat_module
from jarvis_v2.agent.chat import (
    MAX_PREFERENCE_CONTEXT_CHARS,
    MAX_SKILL_CONTEXT_CHARS,
    MAX_STORED_GROUNDING_CHARS,
    ChatBrain,
)
from jarvis_v2.agent.model_provider import ModelProviderError
from jarvis_v2.memory.obsidian import ProfileGroundingView
from jarvis_v2.memory.obsidian import ProfileEvidenceRevalidationError
from jarvis_v2.memory.store import MemoryRecord, ProfileKnowledgeSnapshot
from jarvis_v2.scripts.test_runtime import make_temp_runtime


PROMPT = "Help me approach this project with durable context."
MEMORY_MARKER = "EGRESS_MEMORY_MARKER"
OLLAMA_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class _ReceiptPolicy:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []
        self.active: set[object] = set()
        self.finalized: list[str] = []

    def prepare(self, payload: dict[str, object]) -> object:
        handle = object()
        self.payloads.append(dict(payload))
        self.active.add(handle)
        return handle

    def validate(self, handle: object) -> bool:
        return handle in self.active

    def finalize(self, handle: object, outcome: str) -> str:
        if handle not in self.active:
            raise AssertionError("receipt handle was not active")
        self.active.remove(handle)
        self.finalized.append(outcome)
        return outcome


def _install_policy(
    brain: ChatBrain,
    *,
    fence=nullcontext,
) -> _ReceiptPolicy:
    policy = _ReceiptPolicy()
    brain.install_history_disclosure_callbacks(
        prepare=policy.prepare,
        validate=policy.validate,
        finalize=policy.finalize,
        fence=fence,
    )
    return policy


def _base_brain(*, provider: str = "openai") -> ChatBrain:
    store = MagicMock()
    vault = MagicMock()
    store.read_profile_knowledge_snapshot.return_value = ProfileKnowledgeSnapshot(
        store_identity="0" * 32,
        notes=(),
        source_count=0,
        invalid_count=0,
        truncated=False,
    )
    @contextmanager
    def profile_snapshot_fence():
        yield store.read_profile_knowledge_snapshot()

    store.hold_profile_knowledge_snapshot.side_effect = profile_snapshot_fence
    vault.read_profile_grounding.return_value = ProfileGroundingView(
        text="# Profile",
        verified_source_keys=(),
        invalid=False,
        truncated=False,
    )
    @contextmanager
    def profile_file_fence():
        yield

    vault.profile_grounding_evidence_lock.side_effect = profile_file_fence
    store.list_preferences.return_value = []
    store.search_memories.return_value = []
    store.recent_memories.return_value = []
    store.profile_owned_memory_ids.return_value = frozenset()
    @contextmanager
    def ownership_fence(memory_ids):
        yield store.profile_owned_memory_ids(memory_ids)

    store.profile_memory_ownership_egress_fence.side_effect = ownership_fence
    store.list_active_skills.return_value = []
    return ChatBrain(
        "egress-smoke-model",
        store,
        vault,
        provider=provider,
        allow_remote_personal_context=provider == "openai",
    )


def _message_text(call: MagicMock) -> str:
    messages = call.call_args.kwargs.get("messages")
    if type(messages) is not list:
        raise AssertionError("provider call did not expose a message list")
    return "\n".join(str(message.get("content") or "") for message in messages)


def _seed_history(brain: ChatBrain) -> None:
    decision = brain._history_policy_decision(commit=True)
    brain.history = [
        brain._make_history_entry(
            role="user",
            content="eligible history one",
            decision=decision,
            turn_generation=1,
            personal_derived=False,
            source_count=0,
            source_digest=None,
        ),
        brain._make_history_entry(
            role="assistant",
            content="eligible history two",
            decision=decision,
            turn_generation=1,
            personal_derived=False,
            source_count=0,
            source_digest=None,
        ),
    ]


def test_provider_revocation_at_fence_never_dispatches_stale_route() -> None:
    brain = _base_brain()
    brain.store.list_preferences.return_value = [
        {"category": "privacy", "key": "style", "value": "direct"}
    ]

    @contextmanager
    def revoke_provider(_handle: object):
        brain.provider = "revoked-provider"
        yield

    policy = _install_policy(brain, fence=revoke_provider)
    with patch.object(chat_module, "generate_model_text") as call:
        answer = brain.respond(PROMPT)

    if call.call_count:
        raise AssertionError("a revoked provider decision reached generate_model_text")
    if policy.finalized != ["blocked"] or policy.active:
        raise AssertionError(f"revoked route receipt was not blocked exactly: {policy.__dict__}")
    metadata = brain.last_turn_metadata
    if (
        metadata.get("model_call_attempted") is not False
        or metadata.get("model_error") != "chat_policy_changed_before_dispatch"
    ):
        raise AssertionError(f"provider race metadata was not fail-closed: {metadata}")
    for expected in ("OpenAI routing note", "did not receive a model request", "chat_policy_changed_before_dispatch"):
        if expected not in answer:
            raise AssertionError(f"provider race fallback answer missed frozen-route truth {expected!r}: {answer!r}")
    if "configured provider is invalid" in answer or "Local model note" in answer:
        raise AssertionError(f"provider race fallback answer reread mutated provider state: {answer!r}")


def test_policy_mutation_inside_output_limit_never_dispatches_stale_route() -> None:
    brain = _base_brain()
    brain.store.list_preferences.return_value = [
        {"category": "privacy", "key": "style", "value": "direct"}
    ]
    policy = _install_policy(brain)

    def mutate_policy(*_args, **_kwargs) -> int:
        brain.provider = "ollama"
        brain.model = "mutated-model"
        brain.allow_remote_personal_context = False
        os.environ["OLLAMA_HOST"] = "private.invalid:22999"
        return 300

    previous_host = os.environ.get("OLLAMA_HOST")
    try:
        with (
            patch.object(
                chat_module,
                "provider_output_token_limit",
                side_effect=mutate_policy,
            ),
            patch.object(chat_module, "generate_model_text") as call,
        ):
            brain.respond(PROMPT)
    finally:
        if previous_host is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = previous_host

    if call.call_count:
        raise AssertionError("policy mutation in the output-limit window reached dispatch")
    if policy.finalized != ["blocked"] or policy.active:
        raise AssertionError(f"mutated policy receipt was not blocked: {policy.__dict__}")
    if brain.last_turn_metadata.get("model_error") != "chat_policy_changed_before_dispatch":
        raise AssertionError(f"output-limit policy race did not fail closed: {brain.last_turn_metadata}")


def test_profile_ownership_mutation_inside_output_limit_strips_memory() -> None:
    brain = _base_brain()
    brain.store.search_memories.return_value = [
        {
            "id": 71,
            "source": "manual",
            "category": "project",
            "title": "Captured memory",
            "body": MEMORY_MARKER,
        }
    ]
    ownership = {"ids": frozenset()}
    brain.store.profile_owned_memory_ids.side_effect = lambda memory_ids: frozenset(
        set(memory_ids) & ownership["ids"]
    )
    policy = _install_policy(brain)

    def mutate_ownership(*_args, **_kwargs) -> int:
        ownership["ids"] = frozenset({71})
        return 300

    with (
        patch.object(
            chat_module,
            "provider_output_token_limit",
            side_effect=mutate_ownership,
        ),
        patch.object(
            chat_module,
            "generate_model_text",
            return_value="stateless response",
        ) as call,
    ):
        answer = brain.respond(PROMPT)

    if answer != "stateless response" or call.call_count != 1:
        raise AssertionError("memory ownership race did not take one stateless retry")
    if MEMORY_MARKER in _message_text(call):
        raise AssertionError("newly profile-owned memory crossed the provider boundary")
    if policy.finalized != ["blocked"] or policy.active:
        raise AssertionError(f"memory ownership receipt was not blocked: {policy.__dict__}")
    if brain.last_turn_metadata.get("personal_context_sources_in_model_request") != []:
        raise AssertionError("memory ownership race retained stored context metadata")


def test_policy_setter_blocks_while_provider_call_is_held() -> None:
    brain = _base_brain()
    _install_policy(brain)
    provider_entered = threading.Event()
    release_provider = threading.Event()
    setter_started = threading.Event()
    setter_finished = threading.Event()
    failures: list[BaseException] = []

    def held_provider(**_kwargs) -> str:
        provider_entered.set()
        if not release_provider.wait(2):
            raise AssertionError("held provider was not released")
        return "held response"

    def run_response() -> None:
        try:
            brain.respond(PROMPT)
        except BaseException as exc:
            failures.append(exc)

    def set_provider() -> None:
        setter_started.set()
        brain.provider = "ollama"
        setter_finished.set()

    with patch.object(chat_module, "generate_model_text", side_effect=held_provider):
        response_thread = threading.Thread(target=run_response)
        response_thread.start()
        if not provider_entered.wait(2):
            raise AssertionError("provider call did not enter")
        setter_thread = threading.Thread(target=set_provider)
        setter_thread.start()
        if not setter_started.wait(2):
            raise AssertionError("policy setter did not start")
        if setter_finished.wait(0.1):
            raise AssertionError("policy setter escaped custody during provider dispatch")
        release_provider.set()
        response_thread.join(2)
        setter_thread.join(2)

    if response_thread.is_alive() or setter_thread.is_alive() or failures:
        raise AssertionError(f"held provider concurrency did not finish cleanly: {failures}")
    if not setter_finished.is_set() or brain.provider != "ollama":
        raise AssertionError("policy setter did not resume after provider custody released")


def test_policy_aba_advances_epoch_and_invalidates_prior_history() -> None:
    brain = _base_brain()
    first = brain._history_policy_decision(commit=True)
    brain.history = [
        brain._make_history_entry(
            role="user",
            content="prior policy generation",
            decision=first,
            turn_generation=1,
            personal_derived=False,
            source_count=0,
            source_digest=None,
        )
    ]
    brain.provider = "ollama"
    brain.provider = "openai"
    brain.model = "temporary-model"
    brain.model = "egress-smoke-model"
    brain.allow_remote_personal_context = False
    brain.allow_remote_personal_context = True
    second = brain._history_policy_decision(commit=True)
    if second.epoch <= first.epoch or second.signature == first.signature:
        raise AssertionError(f"policy ABA reused an older epoch: {first} / {second}")
    if brain._eligible_history(second).messages:
        raise AssertionError("prior history survived a provider/model/consent ABA transition")


def test_ollama_environment_aba_advances_epoch_and_invalidates_prior_history() -> None:
    denied_values = (
        ("OLLAMA_HOST", "http://192.0.2.1:11434"),
        ("OLLAMA_NO_CLOUD", "0"),
        ("JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT", "0"),
    )
    for key, denied_value in denied_values:
        with _ollama_context_env():
            brain = _base_brain(provider="ollama")
            first = brain._history_policy_decision(commit=True)
            if not first.stored_context_allowed:
                raise AssertionError(f"Ollama ABA fixture was not initially allowed for {key}")
            brain.history = [
                brain._make_history_entry(
                    role="user",
                    content=f"prior Ollama environment generation for {key}",
                    decision=first,
                    turn_generation=1,
                    personal_derived=False,
                    source_count=0,
                    source_digest=None,
                )
            ]

            allowed_value = os.environ[key]
            os.environ[key] = denied_value
            denied = brain._history_policy_decision(commit=True)
            if denied.stored_context_allowed:
                raise AssertionError(f"Ollama environment denial was not observed for {key}")

            os.environ[key] = allowed_value
            second = brain._history_policy_decision(commit=True)
            if not second.stored_context_allowed:
                raise AssertionError(f"Ollama environment allowance was not restored for {key}")
            if (
                denied.epoch <= first.epoch
                or second.epoch <= denied.epoch
                or denied.epoch_id == first.epoch_id
                or second.epoch_id in {first.epoch_id, denied.epoch_id}
                or second.signature == first.signature
            ):
                raise AssertionError(
                    f"Ollama environment ABA reused policy identity for {key}: "
                    f"{first.epoch}/{denied.epoch}/{second.epoch}"
                )
            if brain._eligible_history(second).messages:
                raise AssertionError(f"prior history survived Ollama environment ABA for {key}")


def test_provider_environment_snapshot_does_not_iterate_process_environment() -> None:
    class GetOnlyEnvironment:
        values = {
            "OPENAI_API_KEY": "test-key",
            "OLLAMA_HOST": "127.0.0.1:11434",
        }

        def get(self, key, default=None):
            return self.values.get(key, default)

        def __iter__(self):
            raise KeyError("concurrent environment iteration")

    with patch.object(chat_module.os, "environ", GetOnlyEnvironment()):
        snapshot = ChatBrain._provider_environment_snapshot()
    if snapshot != GetOnlyEnvironment.values:
        raise AssertionError(f"provider environment snapshot was not allowlisted: {snapshot}")


def test_profile_evidence_is_held_through_dispatch_and_drift_is_uncertain() -> None:
    brain, _data = _digest_fixture()
    policy = _install_policy(brain)
    lock_active = False
    entries = 0

    @contextmanager
    def drifting_profile_fence():
        nonlocal lock_active, entries
        entries += 1
        lock_active = True
        try:
            yield
        finally:
            lock_active = False
        if entries >= 2:
            raise ProfileEvidenceRevalidationError("profile changed during dispatch")

    brain.vault.profile_grounding_evidence_lock.side_effect = drifting_profile_fence

    def provider(**_kwargs) -> str:
        if not lock_active:
            raise AssertionError("provider dispatch escaped the profile evidence lock")
        return "profile-backed response"

    with patch.object(chat_module, "generate_model_text", side_effect=provider) as call:
        answer = brain.respond(PROMPT)
    if call.call_count != 1 or answer == "profile-backed response":
        raise AssertionError("profile drift was accepted as a confirmed model response")
    if policy.finalized != ["uncertain"] or policy.active:
        raise AssertionError(f"profile drift receipt was not uncertain: {policy.__dict__}")


def test_profile_evidence_teardown_preserves_provider_diagnostic() -> None:
    brain, _data = _digest_fixture()
    policy = _install_policy(brain)
    observed: list[str] = []

    @contextmanager
    def exception_aware_profile_fence():
        try:
            yield
        except BaseException as exc:
            observed.append(type(exc).__name__)
            raise
        else:
            raise ProfileEvidenceRevalidationError(
                "profile evidence teardown did not receive the provider exception"
            )

    brain.vault.profile_grounding_evidence_lock.side_effect = exception_aware_profile_fence

    def provider(**_kwargs) -> str:
        raise ModelProviderError(
            "openai_timeout",
            "OpenAI timed out before returning a complete answer.",
        )

    with patch.object(chat_module, "generate_model_text", side_effect=provider) as call:
        answer = brain.respond(PROMPT)
    if call.call_count != 1 or observed != ["ModelProviderError"]:
        raise AssertionError(f"profile evidence teardown missed provider exception: {observed}")
    if "Diagnostic: openai_timeout" not in answer:
        raise AssertionError(f"profile evidence teardown masked provider diagnostic: {answer}")
    if brain.last_turn_metadata.get("model_error") != "openai_timeout":
        raise AssertionError(
            f"profile evidence teardown masked diagnostic metadata: {brain.last_turn_metadata}"
        )
    if policy.finalized != ["blocked"] or policy.active:
        raise AssertionError(
            "stored-context receipt was not blocked before the current-message-only retry: "
            f"{policy.__dict__} metadata={brain.last_turn_metadata}"
        )


def test_direct_sql_profile_drift_during_dispatch_is_uncertain() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-egress-drift-") as temp:
        runtime = make_temp_runtime(Path(temp))
        profile_path = runtime.vault.root_path / "Profile.md"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text("# Profile\n", encoding="utf-8")
        added = runtime.registry.get("add_profile_note").handler(
            {
                "heading": "Egress custody",
                "body": "Use only profile evidence that remains current through dispatch.",
                "category": "privacy",
            }
        )
        memory_id = added.metadata.get("memory_id")
        if not added.ok or type(memory_id) is not int:
            raise AssertionError(f"direct SQL drift fixture failed: {added}")

        brain = ChatBrain(
            "egress-smoke-model",
            runtime.store,
            runtime.vault,
            provider="openai",
            allow_remote_personal_context=True,
        )
        policy = _install_policy(brain)

        def provider(**_kwargs) -> str:
            with sqlite3.connect(str(runtime.store.db_path), timeout=10.0) as conn:
                conn.execute(
                    "UPDATE memories SET body = ?, revision = revision + 1 WHERE id = ?",
                    ("DIRECT_SQL_PROFILE_DRIFT", memory_id),
                )
            return "stale profile-backed response"

        with patch.object(chat_module, "generate_model_text", side_effect=provider) as call:
            answer = brain.respond(PROMPT)
        if call.call_count != 1 or answer == "stale profile-backed response":
            raise AssertionError("direct SQL profile drift was accepted as a model response")
        if policy.finalized != ["uncertain"] or policy.active:
            raise AssertionError(
                f"direct SQL profile drift receipt was not uncertain: {policy.__dict__}"
            )


def test_direct_sql_source_flip_during_dispatch_is_uncertain() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-source-flip-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = runtime.store.add_memory(
            MemoryRecord(
                "project",
                "Durable project context",
                "EGRESS_SOURCE_FLIP_MARKER helps approach this project with durable context.",
                "manual",
                1.0,
            )
        )
        brain = ChatBrain(
            "egress-smoke-model",
            runtime.store,
            runtime.vault,
            provider="openai",
            allow_remote_personal_context=True,
        )
        policy = _install_policy(brain)

        def provider(**kwargs) -> str:
            request_text = "\n".join(
                str(message.get("content") or "") for message in kwargs["messages"]
            )
            if "EGRESS_SOURCE_FLIP_MARKER" not in request_text:
                raise AssertionError("ordinary memory source-flip fixture was not disclosed")
            with sqlite3.connect(str(runtime.store.db_path), timeout=10.0) as conn:
                conn.execute("UPDATE memories SET source = 'profile' WHERE id = ?", (memory_id,))
            return "stale source-flip response"

        with patch.object(chat_module, "generate_model_text", side_effect=provider) as call:
            answer = brain.respond(PROMPT)
        if call.call_count != 1 or answer == "stale source-flip response":
            raise AssertionError("direct SQL profile source flip was accepted as a model response")
        if policy.finalized != ["uncertain"] or policy.active:
            raise AssertionError(
                f"direct SQL profile source-flip receipt was not uncertain: {policy.__dict__}"
            )


def test_inflight_sql_source_flip_is_observed_before_acceptance() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-source-inflight-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = runtime.store.add_memory(
            MemoryRecord(
                "project",
                "In-flight durable context",
                "EGRESS_INFLIGHT_FLIP_MARKER helps approach this project with durable context.",
                "manual",
                1.0,
            )
        )
        brain = ChatBrain(
            "egress-smoke-model",
            runtime.store,
            runtime.vault,
            provider="openai",
            allow_remote_personal_context=True,
        )
        policy = _install_policy(brain)
        writer_started = threading.Event()
        release_writer = threading.Event()
        writer_finished = threading.Event()
        writer_failures: list[BaseException] = []

        def raw_writer() -> None:
            try:
                with sqlite3.connect(str(runtime.store.db_path), timeout=10.0) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        "UPDATE memories SET source = 'profile' WHERE id = ?",
                        (memory_id,),
                    )
                    writer_started.set()
                    if not release_writer.wait(2):
                        raise AssertionError("in-flight raw writer was not released")
                    conn.commit()
            except BaseException as exc:
                writer_failures.append(exc)
            finally:
                writer_finished.set()

        writer = threading.Thread(target=raw_writer, name="raw-profile-source-writer")

        def provider(**kwargs) -> str:
            request_text = "\n".join(
                str(message.get("content") or "") for message in kwargs["messages"]
            )
            if "EGRESS_INFLIGHT_FLIP_MARKER" not in request_text:
                raise AssertionError("in-flight source-flip fixture was not disclosed")
            writer.start()
            if not writer_started.wait(2):
                raise AssertionError("in-flight raw writer did not acquire its transaction")
            threading.Timer(0.05, release_writer.set).start()
            return "stale in-flight source-flip response"

        with patch.object(chat_module, "generate_model_text", side_effect=provider) as call:
            answer = brain.respond(PROMPT)
        writer.join(2)
        if writer.is_alive() or not writer_finished.is_set() or writer_failures:
            raise AssertionError(f"in-flight raw writer did not finish cleanly: {writer_failures}")
        if call.call_count != 1 or answer == "stale in-flight source-flip response":
            raise AssertionError("in-flight SQL profile source flip was accepted")
        if policy.finalized != ["uncertain"] or policy.active:
            raise AssertionError(
                f"in-flight SQL source-flip receipt was not uncertain: {policy.__dict__}"
            )


def _digest_fixture() -> tuple[ChatBrain, dict[str, object]]:
    brain = _base_brain()
    profile = {"text": "# Profile\nProfile fact one\nProfile fact two"}
    preferences = [
        {"category": "style", "key": "tone", "value": "direct"},
        {"category": "format", "key": "length", "value": "short"},
    ]
    memories = [
        {
            "id": 81,
            "source": "manual",
            "category": "project",
            "title": "Ordinary memory",
            "body": "ordinary body",
        },
        {
            "id": 82,
            "source": "conversation_compaction",
            "category": "conversation-digest",
            "title": "Compaction memory",
            "body": "compaction body",
        },
    ]
    skills = [
        {
            "name": "Approach alpha",
            "trigger": "approach project",
            "body": "alpha procedure",
            "tags": "project",
            "origin": "user_authored",
        },
        {
            "name": "Approach beta",
            "trigger": "approach context",
            "body": "beta procedure",
            "tags": "project",
            "origin": "user_authored",
        },
    ]
    provenance = {
        "source_digest": "a" * 64,
    }
    brain.vault.read_profile_grounding.side_effect = lambda *_args, **_kwargs: ProfileGroundingView(
        text=str(profile["text"]),
        verified_source_keys=(),
        invalid=False,
        truncated=False,
    )
    brain.store.list_preferences.side_effect = lambda **_kwargs: preferences
    brain.store.search_memories.side_effect = lambda *_args, **_kwargs: memories
    brain.store.list_active_skills.side_effect = lambda **_kwargs: skills

    def read_provenance(memory_id: int) -> dict[str, object]:
        return {
            "memory_id": memory_id,
            "lineage_state": "complete",
            "recorded_remote_eligible": True,
            "remote_eligible": True,
            "epoch_active": True,
            "destination_class": "external_provider",
            "source_count": 1,
            "source_digest": provenance["source_digest"],
            "policy_epoch_id": "b" * 64,
            "created_at": "2026-07-13T00:00:00Z",
        }

    brain.store.read_memory_provenance.side_effect = read_provenance
    _seed_history(brain)
    return brain, {
        "profile": profile,
        "preferences": preferences,
        "memories": memories,
        "skills": skills,
        "provenance": provenance,
    }


def _disclosure_material(brain: ChatBrain) -> tuple[int, str]:
    decision = brain._history_policy_decision(commit=False)
    context = brain._capture_context_snapshot(PROMPT, decision=decision)
    availability_instruction = brain._source_health_availability_instruction(
        [
            source
            for source, state in context.source_states().items()
            if state == "unavailable"
        ]
    )
    return brain._stored_context_disclosure_material(
        decision=decision,
        context=context,
        context_sources=["profile", "preferences", "memory", "skills"],
        history_message_count=2,
        availability_instruction=availability_instruction,
    )


def test_receipt_digest_and_count_bind_every_disclosed_item() -> None:
    brain, data = _digest_fixture()
    baseline_count, baseline_digest = _disclosure_material(brain)
    if baseline_count != 12:
        raise AssertionError(f"item-level disclosure count drifted: {baseline_count}")

    mutations = (
        (data["preferences"][0], "value", "warmer"),  # type: ignore[index]
        (data["preferences"][1], "value", "longer"),  # type: ignore[index]
        (data["memories"][0], "body", "ordinary changed"),  # type: ignore[index]
        (data["memories"][1], "body", "compaction changed"),  # type: ignore[index]
        (data["skills"][0], "body", "alpha changed"),  # type: ignore[index]
        (data["skills"][1], "body", "beta changed"),  # type: ignore[index]
    )
    for target, key, replacement in mutations:
        original = target[key]
        target[key] = replacement
        count, digest = _disclosure_material(brain)
        target[key] = original
        if count != baseline_count or digest == baseline_digest:
            raise AssertionError(f"receipt did not bind disclosed {key}: {count} / {digest}")

    profile = data["profile"]
    for old, new in (
        ("Profile fact one", "Profile fact changed"),
        ("Profile fact two", "Second profile fact changed"),
    ):
        original = profile["text"]  # type: ignore[index]
        profile["text"] = str(original).replace(old, new)  # type: ignore[index]
        count, digest = _disclosure_material(brain)
        profile["text"] = original  # type: ignore[index]
        if count != baseline_count or digest == baseline_digest:
            raise AssertionError(f"receipt did not bind verified profile content: {old}")

    provenance = data["provenance"]
    original_provenance = provenance["source_digest"]  # type: ignore[index]
    provenance["source_digest"] = "c" * 64  # type: ignore[index]
    count, digest = _disclosure_material(brain)
    provenance["source_digest"] = original_provenance  # type: ignore[index]
    if count != baseline_count or digest == baseline_digest:
        raise AssertionError("receipt did not bind compaction provenance")

    original_history = list(brain.history)
    for index, replacement in enumerate(("changed history one", "changed history two")):
        decision = brain._history_policy_decision(commit=False)
        old_entry = original_history[index]
        brain.history[index] = brain._make_history_entry(
            role=str(old_entry["role"]),
            content=replacement,
            decision=decision,
            turn_generation=1,
            personal_derived=False,
            source_count=0,
            source_digest=None,
        )
        count, digest = _disclosure_material(brain)
        brain.history = list(original_history)
        if count != baseline_count or digest == baseline_digest:
            raise AssertionError(f"receipt did not bind eligible history item {index}")


def test_source_health_only_is_exact_receipted_disclosure_material() -> None:
    brain = _base_brain()
    decision = brain._history_policy_decision(commit=False)
    expected: list[tuple[int, str, str]] = []
    for preferences_available in (True, False):
        brain.store.list_preferences.side_effect = (
            None if preferences_available else RuntimeError("source unavailable")
        )
        context = brain._capture_context_snapshot(PROMPT, decision=decision)
        health_message = brain._source_health_availability_instruction(
            [
                source
                for source, state in context.source_states().items()
                if state == "unavailable"
            ]
        )
        source_count, source_digest = brain._stored_context_disclosure_material(
            decision=decision,
            context=context,
            context_sources=[],
            history_message_count=0,
            availability_instruction=health_message,
        )
        if source_count != 1:
            raise AssertionError("health-only disclosure material did not count exactly one item")
        item_digest = chat_module._disclosure_item_digest(
            "source_health_availability_instruction_v1",
            {"content": health_message},
        )
        expected_digest = hashlib.sha256(
            json.dumps(
                {
                    "kind": "chat_stored_context_disclosure_v2",
                    "policy_epoch_id": decision.epoch_id,
                    "policy_fingerprint": decision.fingerprint,
                    "item_digests": [item_digest],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if source_digest != expected_digest:
            raise AssertionError("health-only receipt did not bind the exact rendered instruction")
        expected.append((source_count, source_digest, health_message))
    if expected[0][1] == expected[1][1]:
        raise AssertionError("source-health state change did not change the receipt digest")

    policy = _install_policy(brain)
    with patch.object(
        chat_module,
        "generate_model_text",
        return_value="health-only response",
    ) as call:
        brain.respond(PROMPT)
    if call.call_count != 1 or len(policy.payloads) != 1:
        raise AssertionError("health-only turn did not prepare exactly one provider receipt")
    payload = policy.payloads[0]
    if (
        payload.get("source_count") != expected[1][0]
        or payload.get("source_digest") != expected[1][1]
    ):
        raise AssertionError(f"prepared health-only receipt drifted: {payload}")
    health_messages = [
        str(message.get("content") or "")
        for message in call.call_args.kwargs["messages"]
        if str(message.get("content") or "").startswith(
            "Personal context availability for this turn:"
        )
    ]
    if health_messages != [expected[1][2]]:
        raise AssertionError("provider request did not use the receipted health instruction")


@contextmanager
def _ollama_context_env():
    keys = (
        "OLLAMA_HOST",
        "OLLAMA_NO_CLOUD",
        "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT",
        *OLLAMA_PROXY_ENV_KEYS,
    )
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
    os.environ["OLLAMA_NO_CLOUD"] = "1"
    os.environ["JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT"] = "1"
    for key in OLLAMA_PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_preference_skill_and_ollama_aggregate_bounds() -> None:
    with _ollama_context_env():
        brain = _base_brain(provider="ollama")
        brain.store.list_preferences.return_value = [
            {
                "category": "C" * 500,
                "key": "K" * 500,
                "value": "V" * 5000 + "PREFERENCE_TAIL",
            }
            for _index in range(20)
        ]
        brain.store.list_active_skills.return_value = [
            {
                "name": "Approach " + "N" * 1000,
                "trigger": "approach project " + "T" * 1000,
                "body": "B" * 5000 + "SKILL_TAIL",
                "tags": "project",
                "origin": "user_authored",
            }
            for _index in range(3)
        ]
        decision = brain._history_policy_decision(commit=True)
        brain.history = [
            brain._make_history_entry(
                role=role,
                content=marker + "H" * 3500,
                decision=decision,
                turn_generation=1,
                personal_derived=False,
                source_count=0,
                source_digest=None,
            )
            for role, marker in (
                ("user", "HISTORY_BOUND_ONE:"),
                ("assistant", "HISTORY_BOUND_TWO:"),
            )
        ]
        _install_policy(brain)
        with patch.object(
            chat_module,
            "generate_model_text",
            return_value="bounded ollama response",
        ) as call:
            brain.respond(PROMPT)

    if call.call_count != 1:
        raise AssertionError("bounded Ollama request did not use one mocked dispatch")
    messages = call.call_args.kwargs["messages"]
    preference_contexts = [
        message["content"]
        for message in messages
        if str(message.get("content") or "").startswith("Structured user preferences")
    ]
    skill_contexts = [
        message["content"]
        for message in messages
        if str(message.get("content") or "").startswith("Relevant saved skills")
    ]
    if not preference_contexts or len(preference_contexts[0]) > MAX_PREFERENCE_CONTEXT_CHARS:
        raise AssertionError("preference context escaped its field/aggregate bound")
    if not skill_contexts or len(skill_contexts[0]) > MAX_SKILL_CONTEXT_CHARS:
        raise AssertionError("skill context escaped its field/aggregate bound")
    stored_prefixes = (
        "Curated profile context",
        "Structured user preferences",
        "Relevant memory context",
        "Relevant saved skills",
        "HISTORY_BOUND_",
        "Personal context availability for this turn:",
    )
    stored_grounding_chars = sum(
        len(str(message.get("content") or ""))
        for message in messages
        if str(message.get("content") or "").startswith(stored_prefixes)
    )
    text = _message_text(call)
    if stored_grounding_chars > MAX_STORED_GROUNDING_CHARS:
        raise AssertionError(
            f"Ollama stored grounding exceeded aggregate bound: {stored_grounding_chars}"
        )
    if "PREFERENCE_TAIL" in text or "SKILL_TAIL" in text:
        raise AssertionError("bounded preference or skill tail reached Ollama")
    if text.count("HISTORY_BOUND_") != 1:
        raise AssertionError("Ollama aggregate bound did not retain exactly one history item")


def main() -> None:
    test_provider_revocation_at_fence_never_dispatches_stale_route()
    test_policy_mutation_inside_output_limit_never_dispatches_stale_route()
    test_profile_ownership_mutation_inside_output_limit_strips_memory()
    test_policy_setter_blocks_while_provider_call_is_held()
    test_policy_aba_advances_epoch_and_invalidates_prior_history()
    test_ollama_environment_aba_advances_epoch_and_invalidates_prior_history()
    test_provider_environment_snapshot_does_not_iterate_process_environment()
    test_receipt_digest_and_count_bind_every_disclosed_item()
    test_source_health_only_is_exact_receipted_disclosure_material()
    test_preference_skill_and_ollama_aggregate_bounds()
    test_profile_evidence_is_held_through_dispatch_and_drift_is_uncertain()
    test_profile_evidence_teardown_preserves_provider_diagnostic()
    test_direct_sql_profile_drift_during_dispatch_is_uncertain()
    test_direct_sql_source_flip_during_dispatch_is_uncertain()
    test_inflight_sql_source_flip_is_observed_before_acceptance()
    print("chat egress custody smoke passed")


if __name__ == "__main__":
    main()
