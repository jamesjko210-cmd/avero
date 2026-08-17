from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from jarvis_v2.agent import chat as chat_module
from jarvis_v2.agent.chat import (
    HISTORY_PROVENANCE_KEY,
    ChatBrain,
    _HistoryProvenance,
)
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.store import (
    MemoryRecord,
    ProfileKnowledgeSnapshot,
    history_memory_content_digest,
    history_source_digest,
)
from jarvis_v2.memory.obsidian import ProfileGroundingView


PROFILE_MARKER = "PROVENANCE_PROFILE_MARKER"
PREFERENCE_MARKER = "PROVENANCE_PREFERENCE_MARKER"
MEMORY_MARKER = "PROVENANCE_MEMORY_MARKER"
ORDINARY_SKILL_MARKER = "PROVENANCE_ORDINARY_SKILL_MARKER"
SESSION_SKILL_MARKER = "PROVENANCE_SESSION_SKILL_MARKER"
CLEANED_SESSION_SKILL_MARKER = "PROVENANCE_CLEANED_SESSION_SKILL_MARKER"
LEGACY_SKILL_MARKER = "PROVENANCE_LEGACY_SKILL_MARKER"
PERSONAL_MARKERS = (
    PROFILE_MARKER,
    PREFERENCE_MARKER,
    MEMORY_MARKER,
    ORDINARY_SKILL_MARKER,
    SESSION_SKILL_MARKER,
    CLEANED_SESSION_SKILL_MARKER,
    LEGACY_SKILL_MARKER,
)
PROVENANCE_METADATA_KEYS = {
    "role",
    "lineage_state",
    "policy_epoch_id",
    "policy_fingerprint",
    "provider",
    "destination_class",
    "future_history_allowed",
    "source_count",
    "source_digest",
    "lineage_token_digest",
}
OLLAMA_ENV_KEYS = (
    "OLLAMA_HOST",
    "OLLAMA_NO_CLOUD",
    "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT",
)


@contextmanager
def _ollama_policy(*, host: str = "http://127.0.0.1:11434", allowed: bool = True):
    previous = {key: os.environ.get(key) for key in OLLAMA_ENV_KEYS}
    os.environ["OLLAMA_HOST"] = host
    if allowed:
        os.environ["OLLAMA_NO_CLOUD"] = "1"
        os.environ["JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT"] = "1"
    else:
        os.environ["OLLAMA_NO_CLOUD"] = "0"
        os.environ["JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT"] = "0"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _brain(
    *,
    provider: str = "openai",
    allowed: bool = True,
    with_personal_context: bool = True,
) -> ChatBrain:
    store = MagicMock()
    vault = MagicMock()
    profile_text = (
        f"# Profile\n\n{PROFILE_MARKER}" if with_personal_context else "# Profile"
    )
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
        text=profile_text,
        verified_source_keys=(),
        invalid=False,
        truncated=False,
    )
    store.list_preferences.return_value = (
        [
            {
                "category": "provenance",
                "key": "style",
                "value": PREFERENCE_MARKER,
            }
        ]
        if with_personal_context
        else []
    )
    store.search_memories.return_value = (
        [
            {
                "id": 101,
                "category": "provenance",
                "title": "Architecture",
                "body": MEMORY_MARKER,
            }
        ]
        if with_personal_context
        else []
    )
    store.recent_memories.return_value = []
    store.list_active_skills.return_value = (
        [
            {
                "name": "Reviewed architecture procedure",
                "trigger": "discuss architecture",
                "body": f"Use the reviewed architecture procedure. {ORDINARY_SKILL_MARKER}",
                "tags": "reviewed,architecture",
                "origin": "user_authored",
            },
            {
                "name": "Session architecture draft",
                "trigger": "discuss architecture",
                "body": (
                    "This is a reviewable skill draft inferred from a Jarvis session.\n"
                    f"## Source Session Signals\n- {SESSION_SKILL_MARKER}"
                ),
                "tags": "draft,session,human-review-required",
                "origin": "session_history",
            },
            {
                "name": "Activated and cleaned session procedure",
                "trigger": "discuss architecture",
                "body": f"Use this clean-looking procedure. {CLEANED_SESSION_SKILL_MARKER}",
                "tags": "reviewed,architecture",
                "origin": "session_history",
            },
            {
                "name": "Legacy procedure without durable custody",
                "trigger": "discuss architecture",
                "body": f"Use this legacy procedure. {LEGACY_SKILL_MARKER}",
                "tags": "reviewed,architecture",
            },
        ]
        if with_personal_context
        else []
    )
    brain = ChatBrain(
        "provenance-model-a",
        store,
        vault,
        provider=provider,
        allow_remote_personal_context=allowed,
    )
    brain.install_history_disclosure_callbacks(
        prepare=lambda _payload: "mock-durable-receipt",
        validate=lambda handle: handle == "mock-durable-receipt",
        finalize=lambda _handle, outcome: outcome,
        fence=lambda _handle: nullcontext(),
    )
    return brain


def _payloads(call: MagicMock) -> list[list[dict[str, str]]]:
    payloads = []
    for item in call.call_args_list:
        messages = item.kwargs.get("messages")
        if not isinstance(messages, list):
            raise AssertionError(f"provider call had no inspectable messages: {item}")
        payloads.append(messages)
    return payloads


def _payload_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(str(message.get("content") or "") for message in messages)


def _assert_payload_stripped(messages: list[dict[str, str]]) -> None:
    for message in messages:
        if set(message) != {"role", "content"}:
            raise AssertionError(f"provenance escaped into provider payload: {message}")
        if HISTORY_PROVENANCE_KEY in message:
            raise AssertionError(f"internal provenance key escaped: {message}")


def _assert_turn_provenance(brain: ChatBrain) -> None:
    value = brain.last_turn_metadata.get("history_message_provenance")
    if not isinstance(value, dict) or set(value) != {"user", "assistant"}:
        raise AssertionError(f"turn provenance object is not structurally strict: {value}")
    for expected_role in ("user", "assistant"):
        entry = value[expected_role]
        if not isinstance(entry, dict) or set(entry) != PROVENANCE_METADATA_KEYS:
            raise AssertionError(f"{expected_role} provenance shape drifted: {entry}")
        if entry["role"] != expected_role:
            raise AssertionError(f"{expected_role} role drifted: {entry}")
        for opaque_key in (
            "policy_epoch_id",
            "policy_fingerprint",
            "lineage_token_digest",
        ):
            opaque = entry.get(opaque_key)
            if not isinstance(opaque, str) or len(opaque) != 64:
                raise AssertionError(f"{expected_role} {opaque_key} is not opaque: {entry}")
        if entry["lineage_state"] == "personal_derived":
            source_digest = entry.get("source_digest")
            if not isinstance(source_digest, str) or len(source_digest) != 64:
                raise AssertionError(
                    f"{expected_role} derived source digest is missing: {entry}"
                )
        elif entry.get("source_digest") is not None:
            raise AssertionError(
                f"{expected_role} direct lineage gained a source digest: {entry}"
            )
    serialized = repr(value)
    if any(marker in serialized for marker in PERSONAL_MARKERS):
        raise AssertionError(f"turn provenance contains personal content: {value}")


def test_opt_in_nonretroactivity_and_repeated_toggles() -> None:
    brain = _brain(allowed=False)
    prompts = [f"Discuss architecture toggle-{index}" for index in range(5)]
    with patch.object(
        chat_module,
        "generate_model_text",
        side_effect=[f"toggle-answer-{index}" for index in range(5)],
    ) as call:
        brain.respond(prompts[0])
        if brain.last_turn_metadata["history_message_provenance"]["user"][
            "future_history_allowed"
        ] is not False:
            raise AssertionError("gate-off current-message authorization became a future grant")
        first_epoch = brain.last_turn_metadata["history_policy_epoch"]
        gate_off_preview = brain.preview_loop("Discuss architecture gate-off-preview")
        if gate_off_preview["history_withheld_reason_counts"] != {
            "future_history_not_granted": 2
        }:
            raise AssertionError(
                f"gate-off turns became future eligible: {gate_off_preview}"
            )

        brain.allow_remote_personal_context = True
        epoch_before_preview = brain._policy_epoch
        preview = brain.preview_loop(prompts[1])
        if brain._policy_epoch != epoch_before_preview:
            raise AssertionError("preview mutated the committed policy epoch")
        if preview["history_eligible_messages"] != 0 or preview[
            "history_withheld_messages"
        ] != 2:
            raise AssertionError(f"opt-in preview resurrected gate-off history: {preview}")
        brain.respond(prompts[1])
        second_epoch = brain.last_turn_metadata["history_policy_epoch"]
        brain.respond(prompts[2])
        stable_epoch = brain.last_turn_metadata["history_policy_epoch"]

        brain.allow_remote_personal_context = False
        brain.respond(prompts[3])
        third_epoch = brain.last_turn_metadata["history_policy_epoch"]
        brain.allow_remote_personal_context = True
        brain.respond(prompts[4])
        fourth_epoch = brain.last_turn_metadata["history_policy_epoch"]

    payloads = _payloads(call)
    for messages in payloads:
        _assert_payload_stripped(messages)
    if prompts[0] in _payload_text(payloads[1]):
        raise AssertionError("opt-in retroactively disclosed a gate-off message")
    stable_text = _payload_text(payloads[2])
    if prompts[1] not in stable_text or "toggle-answer-1" not in stable_text:
        raise AssertionError(f"stable opted-in history was not eligible: {stable_text}")
    if prompts[1] in _payload_text(payloads[4]) or prompts[3] in _payload_text(payloads[4]):
        raise AssertionError("off/on/off/on resurrected a prior policy generation")
    if not (first_epoch < second_epoch == stable_epoch < third_epoch < fourth_epoch):
        raise AssertionError(
            f"policy epochs were not monotonic: {first_epoch, second_epoch, stable_epoch, third_epoch, fourth_epoch}"
        )


def test_provider_model_destination_and_no_cloud_switches() -> None:
    brain = _brain(provider="openai", allowed=True, with_personal_context=False)
    prompts = [f"Discuss architecture switch-{index}" for index in range(8)]
    with (
        _ollama_policy(),
        patch.object(
            chat_module,
            "generate_model_text",
            side_effect=[f"switch-answer-{index}" for index in range(8)],
        ) as call,
    ):
        brain.respond(prompts[0])
        epochs = [brain.last_turn_metadata["history_policy_epoch"]]

        brain.model = "provenance-model-b"
        brain.respond(prompts[1])
        epochs.append(brain.last_turn_metadata["history_policy_epoch"])
        brain.model = "provenance-model-a"
        brain.respond(prompts[2])
        epochs.append(brain.last_turn_metadata["history_policy_epoch"])

        brain.provider = "ollama"
        brain.respond(prompts[3])
        epochs.append(brain.last_turn_metadata["history_policy_epoch"])
        brain.provider = "openai"
        brain.respond(prompts[4])
        epochs.append(brain.last_turn_metadata["history_policy_epoch"])

        brain.provider = "ollama"
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11435"
        brain.respond(prompts[5])
        epochs.append(brain.last_turn_metadata["history_policy_epoch"])
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
        brain.respond(prompts[6])
        epochs.append(brain.last_turn_metadata["history_policy_epoch"])

        os.environ["OLLAMA_NO_CLOUD"] = "0"
        brain.respond(prompts[7])
        epochs.append(brain.last_turn_metadata["history_policy_epoch"])

    payloads = _payloads(call)
    for index, messages in enumerate(payloads):
        text = _payload_text(messages)
        if any(old_prompt in text for old_prompt in prompts[:index]):
            raise AssertionError(f"policy switch {index} resurrected old history: {text}")
    if any(later <= earlier for earlier, later in zip(epochs, epochs[1:])):
        raise AssertionError(f"switch epochs did not rotate monotonically: {epochs}")


def test_malformed_entries_seals_generations_order_and_payload_stripping() -> None:
    brain = _brain(allowed=True, with_personal_context=False)
    with patch.object(chat_module, "generate_model_text", return_value="seed answer"):
        brain.respond("Discuss architecture valid-seed")
    valid_user, valid_assistant = brain.history[-2:]
    tampered = dict(valid_user)
    tampered["content"] = "TAMPERED_CONTENT_MARKER"
    malformed = {
        "role": "assistant",
        "content": "MALFORMED_PROVENANCE_MARKER",
        HISTORY_PROVENANCE_KEY: {"seal": "not-a-record"},
    }
    bad_role = {"role": "system", "content": "BAD_ROLE_MARKER"}
    legacy = {"role": "user", "content": "LEGACY_MARKER"}
    brain.history = [
        valid_user,
        legacy,
        valid_assistant,
        bad_role,
        malformed,
        tampered,
    ]
    preview = brain.preview_loop("Discuss architecture after-malformed")
    if preview["history_eligible_messages"] != 2 or preview[
        "history_withheld_messages"
    ] != 4:
        raise AssertionError(f"mixed eligibility counts drifted: {preview}")
    expected_reasons = {
        "legacy_unmarked": 1,
        "invalid_role": 1,
        "malformed_provenance": 1,
        "invalid_seal": 1,
    }
    if preview["history_withheld_reason_counts"] != expected_reasons:
        raise AssertionError(f"withheld reasons drifted: {preview}")
    with patch.object(chat_module, "generate_model_text", return_value="clean answer") as call:
        brain.respond("Discuss architecture after-malformed")
    messages = _payloads(call)[0]
    _assert_payload_stripped(messages)
    text = _payload_text(messages)
    if "valid-seed" not in text or "seed answer" not in text:
        raise AssertionError(f"eligible mixed-order history was lost: {text}")
    for marker in (
        "LEGACY_MARKER",
        "BAD_ROLE_MARKER",
        "MALFORMED_PROVENANCE_MARKER",
        "TAMPERED_CONTENT_MARKER",
    ):
        if marker in text:
            raise AssertionError(f"withheld history escaped: {marker}")

    other = _brain(allowed=True, with_personal_context=False)
    other.history = [valid_user]
    generation_preview = other.preview_loop("Discuss architecture generation-check")
    if generation_preview["history_withheld_reason_counts"] != {
        "generation_mismatch": 1
    }:
        raise AssertionError(f"cross-brain seal was accepted: {generation_preview}")

    malformed_record = dict(valid_user)
    provenance = valid_user[HISTORY_PROVENANCE_KEY]
    if type(provenance) is not _HistoryProvenance:
        raise AssertionError("valid entry lost its internal provenance type")
    malformed_record[HISTORY_PROVENANCE_KEY] = replace(
        provenance, source_count=object()  # type: ignore[arg-type]
    )
    brain.history = [malformed_record]
    malformed_preview = brain.preview_loop("Discuss architecture malformed-record")
    if malformed_preview["history_withheld_reason_counts"] != {
        "malformed_provenance": 1
    }:
        raise AssertionError(f"malformed record was not withheld: {malformed_preview}")

    forged_provenance = valid_user[HISTORY_PROVENANCE_KEY]
    if type(forged_provenance) is not _HistoryProvenance:
        raise AssertionError("valid entry lost its internal provenance type")
    forged = dict(valid_user)
    forged[HISTORY_PROVENANCE_KEY] = replace(
        forged_provenance, future_history_allowed=False
    )
    brain.history = [forged]
    forged_preview = brain.preview_loop("Discuss architecture forged")
    if forged_preview["history_withheld_reason_counts"] != {
        "invalid_seal": 1
    }:
        raise AssertionError(f"forged grant did not trip the seal: {forged_preview}")


def test_lineage_metadata_skill_laundering_and_preview_parity() -> None:
    brain = _brain(allowed=True, with_personal_context=True)
    with patch.object(chat_module, "generate_model_text", return_value="lineage answer") as call:
        brain.respond("Discuss architecture lineage-source")
    text = _payload_text(_payloads(call)[0])
    for marker in (
        PROFILE_MARKER,
        PREFERENCE_MARKER,
        MEMORY_MARKER,
        ORDINARY_SKILL_MARKER,
    ):
        if marker not in text:
            raise AssertionError(f"ordinary personal source was not consumed: {marker}")
    for marker in (
        SESSION_SKILL_MARKER,
        CLEANED_SESSION_SKILL_MARKER,
        LEGACY_SKILL_MARKER,
    ):
        if marker in text:
            raise AssertionError(f"restricted skill origin was laundered into model context: {marker}")
    _assert_turn_provenance(brain)
    current = brain.last_turn_metadata["history_message_provenance"]
    if (
        current["user"]["lineage_state"] != "direct"
        or current["user"]["source_count"] != 0
        or current["user"]["source_digest"] is not None
    ):
        raise AssertionError(f"user lineage drifted: {current}")
    if current["assistant"]["lineage_state"] != "personal_derived" or current[
        "assistant"
    ]["source_count"] != 4:
        raise AssertionError(f"assistant did not inherit local personal lineage: {current}")
    first_source_digest = current["assistant"]["source_digest"]
    if not isinstance(first_source_digest, str) or len(first_source_digest) != 64:
        raise AssertionError(f"local-source digest is missing: {current}")

    brain.vault.read_profile_grounding.return_value = ProfileGroundingView(
        text="# Profile",
        verified_source_keys=(),
        invalid=False,
        truncated=False,
    )
    brain.store.list_preferences.return_value = []
    brain.store.search_memories.return_value = []
    brain.store.list_active_skills.return_value = []
    epoch_before = brain._policy_epoch
    preview = brain.preview_loop("Discuss architecture parity")
    if brain._policy_epoch != epoch_before:
        raise AssertionError("stable preview mutated the epoch")
    with patch.object(chat_module, "generate_model_text", return_value="history-derived answer"):
        brain.respond("Discuss architecture parity")
    metadata = brain.last_turn_metadata
    for key in (
        "history_policy_epoch",
        "history_policy_epoch_id",
        "history_policy_fingerprint",
        "history_retained_messages",
        "history_eligible_messages",
        "history_withheld_messages",
        "history_withheld_reason_counts",
    ):
        if metadata[key] != preview[key]:
            raise AssertionError(f"preview/actual parity drifted for {key}: {preview} / {metadata}")
    assistant = metadata["history_message_provenance"]["assistant"]
    if assistant["lineage_state"] != "personal_derived" or assistant[
        "source_count"
    ] != 1:
        raise AssertionError(f"history-derived reply lost lineage: {assistant}")
    second_source_digest = assistant["source_digest"]
    if (
        not isinstance(second_source_digest, str)
        or len(second_source_digest) != 64
        or second_source_digest == first_source_digest
    ):
        raise AssertionError(
            f"history-derived digest did not incorporate ancestor lineage: {assistant}"
        )

    with patch.object(chat_module, "generate_model_text", return_value="transitive answer"):
        brain.respond("Discuss architecture transitive-lineage")
    transitive = brain.last_turn_metadata["history_message_provenance"]["assistant"]
    transitive_digest = transitive["source_digest"]
    if (
        transitive["source_count"] != 1
        or not isinstance(transitive_digest, str)
        or len(transitive_digest) != 64
        or transitive_digest in {first_source_digest, second_source_digest}
    ):
        raise AssertionError(f"transitive history lineage was not propagated: {transitive}")
    if any(marker in repr(transitive) for marker in PERSONAL_MARKERS):
        raise AssertionError(f"transitive source digest metadata leaked content: {transitive}")


def _complete_memory_provenance(
    memory_id: int,
    source_digest: str,
    *,
    destination_class: str = "external_provider",
) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "lineage_state": "complete",
        "source_count": 2,
        "source_digest": source_digest,
        "policy_epoch_id": "e" * 64,
        "destination_class": destination_class,
        "recorded_remote_eligible": True,
        "remote_eligible": True,
        "epoch_active": True,
        "created_at": "2099-01-01T00:00:00+00:00",
    }


def test_conversation_digest_memory_requires_durable_remote_lineage() -> None:
    search_valid = "SEARCH_VALID_CONVERSATION_DIGEST"
    search_missing = "SEARCH_MISSING_PROVENANCE_DIGEST"
    search_stale = "SEARCH_STALE_PROVENANCE_DIGEST"
    ordinary = "SEARCH_ORDINARY_MEMORY"
    brain = _brain(allowed=True, with_personal_context=False)
    decision = brain._history_policy_decision(commit=False)
    if decision.destination_class != "external_provider":
        raise AssertionError(f"OpenAI destination vocabulary drifted: {decision.destination_class}")
    brain.store.search_memories.return_value = [
        {
            "id": 10,
            "category": "facts",
            "source": "manual",
            "title": "Ordinary",
            "body": ordinary,
        },
        {
            "id": 11,
            "category": "conversation-digest",
            "source": "conversation_compaction",
            "title": "Eligible digest",
            "body": search_valid,
        },
        {
            "id": 12,
            "category": "conversation-digest",
            "source": "conversation_compaction",
            "title": "Missing digest",
            "body": search_missing,
        },
        {
            "id": 13,
            "category": "other",
            "source": "conversation_compaction",
            "title": "Stale digest",
            "body": search_stale,
        },
    ]
    valid = _complete_memory_provenance(
        11,
        "1" * 64,
        destination_class=decision.destination_class,
    )
    stale = _complete_memory_provenance(13, "3" * 64)
    stale["epoch_active"] = False
    stale["remote_eligible"] = False
    brain.store.read_memory_provenance.side_effect = lambda memory_id: {
        11: valid,
        12: None,
        13: stale,
    }.get(memory_id)
    with patch.object(chat_module, "generate_model_text", return_value="search digest answer") as call:
        brain.respond("Discuss architecture digest-search")
    text = _payload_text(_payloads(call)[0])
    if ordinary not in text or search_valid not in text:
        raise AssertionError(f"ordinary or eligible search memory was omitted: {text}")
    if search_missing in text or search_stale in text:
        raise AssertionError(f"unproven search digest reached the provider: {text}")
    search_lineage = brain.last_turn_metadata["history_message_provenance"]["assistant"]
    if (
        search_lineage["source_count"] != 1
        or not isinstance(search_lineage["source_digest"], str)
        or len(search_lineage["source_digest"]) != 64
    ):
        raise AssertionError(f"eligible durable memory lost source identity: {search_lineage}")
    for marker in (search_valid, search_missing, search_stale, ordinary):
        if marker in repr(search_lineage):
            raise AssertionError(f"memory content entered lineage metadata: {search_lineage}")

    recent_valid = "RECENT_VALID_CONVERSATION_DIGEST"
    recent_malformed = "RECENT_MALFORMED_PROVENANCE_DIGEST"
    recent_brain = _brain(allowed=True, with_personal_context=False)
    recent_brain.store.search_memories.return_value = []
    recent_brain.store.recent_memories.return_value = [
        {
            "id": 21,
            "category": "conversation-digest",
            "source": "conversation_compaction",
            "title": "Recent eligible digest",
            "body": recent_valid,
        },
        {
            "id": 22,
            "category": "conversation-digest",
            "source": "conversation_compaction",
            "title": "Recent malformed digest",
            "body": recent_malformed,
        },
    ]
    malformed = _complete_memory_provenance(22, "2" * 64)
    malformed["destination_class"] = "loopback_daemon_unverified"
    recent_brain.store.read_memory_provenance.side_effect = lambda memory_id: {
        21: _complete_memory_provenance(21, "4" * 64),
        22: malformed,
    }.get(memory_id)
    with patch.object(chat_module, "generate_model_text", return_value="recent digest answer") as call:
        recent_brain.respond("Discuss architecture digest-recent")
    recent_text = _payload_text(_payloads(call)[0])
    if recent_valid not in recent_text or recent_malformed in recent_text:
        raise AssertionError(f"recent fallback provenance filter drifted: {recent_text}")

    ollama_brain = _brain(
        provider="ollama",
        allowed=True,
        with_personal_context=False,
    )
    with _ollama_policy():
        ollama_decision = ollama_brain._history_policy_decision(commit=False)
    if (
        ollama_decision.destination_class != "loopback_daemon_unverified"
        or ollama_brain._durable_memory_destination_class(ollama_decision)
        != ollama_decision.destination_class
    ):
        raise AssertionError(
            f"Ollama persisted destination vocabulary drifted: {ollama_decision}"
        )


def _real_runtime(root: Path, *, provider: str) -> JarvisRuntime:
    return JarvisRuntime(
        JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="provenance-model-a",
            model_provider=provider,
            allow_remote_personal_context=True,
            use_model_planner=False,
        )
    )


def _commit_real_conversation_digest(
    runtime: JarvisRuntime,
    *,
    expected_after_id: int,
    epoch_id: str,
    destination_class: str,
    marker: str,
) -> dict[str, object]:
    source_content = f"SOURCE_CONTENT_{marker}"
    message_id = runtime.store.log_message(
        "destination-provenance-round-trip",
        "user",
        source_content,
        provenance={
            "role": "user",
            "policy_epoch_id": epoch_id,
            "lineage_state": "direct_current",
            "destination_class": destination_class,
            "remote_eligible": True,
        },
    )
    message_provenance = runtime.store.read_message_provenance(message_id)
    if message_provenance is None:
        raise AssertionError("compaction source provenance was not persisted")
    source_digest = history_source_digest(
        [
            (
                message_id,
                str(message_provenance["lineage_token"]),
                str(message_provenance["content_digest"]),
            )
        ]
    )
    title = f"Architecture destination digest {marker}"
    body = f"- Durable architecture destination fact {marker}"
    committed = runtime.store.commit_conversation_compaction(
        expected_after_id=expected_after_id,
        first_message_id=message_id,
        last_message_id=message_id,
        record=MemoryRecord(
            "conversation-digest",
            title,
            body,
            "conversation_compaction",
            0.8,
        ),
        eligible_source_count=1,
        withheld_source_count=0,
        source_provenance_digest=source_digest,
        policy_epoch_id=epoch_id,
    )
    if committed.status != "committed" or not committed.memory_created:
        raise AssertionError(f"production compaction commit failed: {committed}")
    return {
        "memory_id": int(committed.memory_id or 0),
        "message_id": message_id,
        "source_digest": source_digest,
        "content_digest": history_memory_content_digest(title, body),
        "title": title,
        "body": body,
        "source_content": source_content,
    }


def test_real_destination_provenance_round_trip() -> None:
    cases = (
        ("openai", "external_provider", "loopback_daemon_unverified"),
        ("ollama", "loopback_daemon_unverified", "external_provider"),
    )
    for provider, current_class, mismatched_class in cases:
        policy = _ollama_policy() if provider == "ollama" else nullcontext()
        with policy, TemporaryDirectory(
            prefix=f"jarvis-chat-destination-{provider}-"
        ) as temp:
            runtime = _real_runtime(Path(temp), provider=provider)
            mismatch_epoch = runtime.store.start_history_policy_epoch(
                policy_fingerprint="a" * 64,
                provider="ollama" if provider == "openai" else "openai",
                model_identifier="mismatched-provenance-model",
                destination_class=mismatched_class,
                session_generation=2,
                explicit_consent_satisfied=True,
            )
            mismatch_marker = f"MISMATCHED_{provider.upper()}_DESTINATION_CONTENT"
            mismatched = _commit_real_conversation_digest(
                runtime,
                expected_after_id=0,
                epoch_id=mismatch_epoch,
                destination_class=mismatched_class,
                marker=mismatch_marker,
            )

            if not runtime._activate_fresh_history_policy_binding():
                raise AssertionError(f"{provider} ChatBrain policy binding was unavailable")
            active = runtime.store.read_active_history_epoch()
            if active["destination_class"] != current_class:
                raise AssertionError(f"{provider} active destination class drifted: {active}")
            exact_marker = f"EXACT_{provider.upper()}_DESTINATION_CONTENT"
            exact = _commit_real_conversation_digest(
                runtime,
                expected_after_id=int(mismatched["message_id"]),
                epoch_id=str(active["epoch_id"]),
                destination_class=current_class,
                marker=exact_marker,
            )

            exact_provenance = runtime.store.read_memory_provenance(
                int(exact["memory_id"])
            )
            mismatched_provenance = runtime.store.read_memory_provenance(
                int(mismatched["memory_id"])
            )
            for label, persisted, expected, expected_class in (
                ("exact", exact_provenance, exact, current_class),
                ("mismatched", mismatched_provenance, mismatched, mismatched_class),
            ):
                if persisted is None or any(
                    (
                        persisted["lineage_state"] != "complete",
                        persisted["source_count"] != 1,
                        persisted["source_digest"] != expected["source_digest"],
                        persisted["content_digest"] != expected["content_digest"],
                        persisted["content_matches"] is not True,
                        persisted["destination_class"] != expected_class,
                    )
                ):
                    raise AssertionError(
                        f"{provider} {label} memory provenance did not survive SQLite: {persisted}"
                    )
            if exact_provenance["remote_eligible"] is not True:
                raise AssertionError(f"{provider} exact-class digest was not current: {exact_provenance}")
            if mismatched_provenance["remote_eligible"] is not False:
                raise AssertionError(
                    f"{provider} mismatched-class digest remained eligible: {mismatched_provenance}"
                )

            with patch.object(
                chat_module,
                "generate_model_text",
                return_value=f"{provider} destination round-trip answer",
            ) as call:
                runtime.chat.respond("Discuss architecture destination provenance round-trip")
            messages = _payloads(call)[0]
            _assert_payload_stripped(messages)
            payload_text = _payload_text(messages)
            if exact_marker not in payload_text or mismatch_marker in payload_text:
                raise AssertionError(
                    f"{provider} ChatBrain destination filter drifted: {payload_text}"
                )
            assistant = runtime.chat.last_turn_metadata["history_message_provenance"][
                "assistant"
            ]
            if (
                assistant["lineage_state"] != "personal_derived"
                or assistant["source_count"] != 1
                or not isinstance(assistant["source_digest"], str)
                or len(assistant["source_digest"]) != 64
            ):
                raise AssertionError(
                    f"{provider} ChatBrain lost the persisted digest lineage: {assistant}"
                )

            with runtime.store.connect() as conn:
                receipts = [
                    dict(row)
                    for row in conn.execute(
                        "SELECT * FROM history_disclosure_receipts ORDER BY prepared_at"
                    )
                ]
            if (
                len(receipts) != 1
                or receipts[0]["state"] != "confirmed"
                or receipts[0]["destination_class"] != current_class
            ):
                raise AssertionError(f"{provider} durable receipt drifted: {receipts}")
            receipt_metadata = repr(receipts)
            forbidden_content = (
                exact_marker,
                mismatch_marker,
                exact["title"],
                exact["body"],
                exact["source_content"],
                mismatched["title"],
                mismatched["body"],
                mismatched["source_content"],
            )
            if any(str(content) in receipt_metadata for content in forbidden_content):
                raise AssertionError(
                    f"{provider} receipt metadata contained source content: {receipts}"
                )


def test_unavailable_durable_authority_strips_stored_context() -> None:
    brain = _brain(allowed=True, with_personal_context=True)
    brain.clear_history_disclosure_callbacks()
    prompt = "Discuss architecture without durable authority"
    with patch.object(chat_module, "generate_model_text", return_value="current only") as call:
        brain.respond(prompt)
    messages = _payloads(call)[0]
    _assert_payload_stripped(messages)
    text = _payload_text(messages)
    if prompt not in text or any(marker in text for marker in PERSONAL_MARKERS):
        raise AssertionError(f"unavailable durable authority disclosed stored context: {text}")
    if (
        brain.last_turn_metadata["history_disclosure_prepare_status"] != "failed_stripped"
        or brain.last_turn_metadata["stored_personal_context_in_model_request"] is not False
    ):
        raise AssertionError(
            f"unavailable durable authority metadata drifted: {brain.last_turn_metadata}"
        )


def test_crash_safe_prepare_finalize_callbacks() -> None:
    brain = _brain(allowed=True, with_personal_context=True)
    with patch.object(chat_module, "generate_model_text", return_value="callback seed"):
        brain.respond("Discuss architecture callback-seed")

    prepared: list[dict[str, object]] = []
    finalized: list[tuple[object, str]] = []

    def prepare(payload: dict[str, object]) -> object:
        prepared.append(payload)
        return "opaque-receipt-handle"

    def finalize(handle: object, outcome: str) -> None:
        finalized.append((handle, outcome))

    validate = lambda handle: handle in {
        "opaque-receipt-handle",
        "mutation-receipt",
    }
    normal_fence = lambda _handle: nullcontext()
    brain.install_history_disclosure_callbacks(
        prepare=prepare,
        validate=validate,
        finalize=finalize,
        fence=normal_fence,
    )
    with patch.object(chat_module, "generate_model_text", return_value="callback confirmed"):
        brain.respond("Discuss architecture callback-confirmed")
    if len(prepared) != 1 or set(prepared[0]) != {
        "session_generation",
        "policy_epoch_id",
        "policy_fingerprint",
        "provider",
        "destination_class",
        "source_count",
        "source_digest",
    }:
        raise AssertionError(f"prepare payload was not strict and content-free: {prepared}")
    if any(marker in repr(prepared[0]) for marker in PERSONAL_MARKERS):
        raise AssertionError(f"prepare payload contained personal content: {prepared[0]}")
    if finalized != [("opaque-receipt-handle", "confirmed")]:
        raise AssertionError(f"confirmed disclosure was not finalized: {finalized}")

    prepared.clear()
    finalized.clear()
    with patch.object(
        chat_module, "generate_model_text", side_effect=RuntimeError("provider failed")
    ):
        brain.respond("Discuss architecture callback-uncertain")
    if finalized != [("opaque-receipt-handle", "uncertain")]:
        raise AssertionError(f"failed provider disclosure was not uncertain: {finalized}")

    def failed_prepare(payload: dict[str, object]) -> object:
        prepared.append(payload)
        raise RuntimeError("receipt store unavailable")

    brain.install_history_disclosure_callbacks(
        prepare=failed_prepare,
        validate=validate,
        finalize=finalize,
        fence=normal_fence,
    )
    prepared.clear()
    finalized.clear()
    with patch.object(chat_module, "generate_model_text", return_value="stripped answer") as call:
        brain.respond("Discuss architecture callback-stripped")
    text = _payload_text(_payloads(call)[0])
    if any(marker in text for marker in PERSONAL_MARKERS) or "callback-uncertain" in text:
        raise AssertionError(f"prepare failure disclosed stale context: {text}")
    if brain.last_turn_metadata["history_in_model_request"] is not False or brain.last_turn_metadata[
        "stored_personal_context_in_model_request"
    ] is not False:
        raise AssertionError(f"prepare failure metadata claimed disclosure: {brain.last_turn_metadata}")
    if brain.last_turn_metadata["history_disclosure_prepare_status"] != "failed_stripped":
        raise AssertionError(f"prepare failure status drifted: {brain.last_turn_metadata}")
    if finalized:
        raise AssertionError(f"failed prepare unexpectedly finalized a handle: {finalized}")

    brain.install_history_disclosure_callbacks(
        prepare=prepare,
        validate=validate,
        finalize=finalize,
        fence=lambda _handle: (_ for _ in ()).throw(
            RuntimeError("synthetic fence failure")
        ),
    )
    prepared.clear()
    finalized.clear()
    with patch.object(chat_module, "generate_model_text", return_value="fence-stripped") as call:
        brain.respond("Discuss architecture broken-fence")
    text = _payload_text(_payloads(call)[0])
    if any(marker in text for marker in PERSONAL_MARKERS) or "callback-uncertain" in text:
        raise AssertionError(f"broken fence disclosed stored context: {text}")
    if finalized != [("opaque-receipt-handle", "blocked")]:
        raise AssertionError(f"broken fence receipt was not blocked: {finalized}")
    if brain.last_turn_metadata["history_disclosure_prepare_status"] != "failed_stripped":
        raise AssertionError(f"broken fence did not report stripping: {brain.last_turn_metadata}")
    def mutating_prepare(payload: dict[str, object]) -> object:
        brain.allow_remote_personal_context = False
        return "mutation-receipt"

    brain.install_history_disclosure_callbacks(
        prepare=mutating_prepare,
        validate=validate,
        finalize=finalize,
        fence=normal_fence,
    )
    finalized.clear()
    with patch.object(
        chat_module,
        "generate_model_text",
        side_effect=AssertionError("policy mutation reached provider"),
    ) as call:
        brain.respond("Discuss architecture callback-blocked")
    if call.call_count:
        raise AssertionError("policy mutation reached provider dispatch")
    if finalized != [("mutation-receipt", "blocked")]:
        raise AssertionError(f"mutated policy receipt was not blocked: {finalized}")
    if brain.last_turn_metadata["model_error"] != "chat_policy_changed_before_dispatch":
        raise AssertionError(f"policy mutation did not fail closed: {brain.last_turn_metadata}")


def test_callback_bundle_pinning_and_strict_confirmation_status() -> None:
    brain = _brain(allowed=True, with_personal_context=True)
    pinned_finalizations: list[tuple[object, str]] = []
    replacement_finalizations: list[tuple[object, str]] = []
    def pinned_finalize(handle: object, outcome: str) -> str:
        pinned_finalizations.append((handle, outcome))
        return outcome

    brain.install_history_disclosure_callbacks(
        prepare=lambda _payload: "pinned-receipt",
        validate=lambda handle: handle == "pinned-receipt",
        finalize=pinned_finalize,
        fence=lambda _handle: nullcontext(),
    )

    def swap_callbacks_during_invocation(**_kwargs) -> str:
        brain.install_history_disclosure_callbacks(
            prepare=lambda _payload: None,
            validate=lambda _handle: False,
            finalize=lambda handle, outcome: (
                replacement_finalizations.append((handle, outcome)) or outcome
            ),
            fence=lambda _handle: nullcontext(),
        )
        return "pinned callback answer"

    with patch.object(
        chat_module,
        "generate_model_text",
        side_effect=swap_callbacks_during_invocation,
    ):
        answer = brain.respond("Discuss architecture while callbacks rotate")
    if answer != "pinned callback answer":
        raise AssertionError(f"pinned callback answer drifted: {answer!r}")
    if pinned_finalizations != [("pinned-receipt", "confirmed")] or replacement_finalizations:
        raise AssertionError(
            f"provider-time callback swap redirected finalization: "
            f"{pinned_finalizations!r} / {replacement_finalizations!r}"
        )
    if brain.last_turn_metadata["history_disclosure_finalize_status"] != "confirmed":
        raise AssertionError(f"pinned confirmation metadata drifted: {brain.last_turn_metadata}")

    for label, finalizer in (
        ("missing-result", lambda _handle, _outcome: None),
        (
            "raised-finalizer",
            lambda _handle, _outcome: (_ for _ in ()).throw(
                RuntimeError("synthetic durable finalization failure")
            ),
        ),
    ):
        uncertain_brain = _brain(allowed=True, with_personal_context=True)
        uncertain_brain.install_history_disclosure_callbacks(
            prepare=lambda _payload: "uncertain-receipt",
            validate=lambda handle: handle == "uncertain-receipt",
            finalize=finalizer,
            fence=lambda _handle: nullcontext(),
        )
        with patch.object(chat_module, "generate_model_text", return_value="attempted egress"):
            uncertain_brain.respond(f"Discuss architecture with {label}")
        if uncertain_brain.last_turn_metadata["history_disclosure_finalize_status"] != "uncertain":
            raise AssertionError(
                f"{label} claimed a durable confirmation: {uncertain_brain.last_turn_metadata}"
            )


def test_deterministic_paths_never_prepare_or_call_provider() -> None:
    prepared: list[dict[str, object]] = []
    finalized: list[tuple[object, str]] = []
    for prompt, expected_source in (
        ("What do you know about me?", "grounded_memory"),
        ("Please run this architecture script", "safety_preflight_guidance"),
    ):
        brain = _brain(allowed=True)
        brain.install_history_disclosure_callbacks(
            prepare=lambda payload: prepared.append(payload) or "receipt",
            validate=lambda handle: handle == "receipt",
            finalize=lambda handle, outcome: finalized.append((handle, outcome)) or outcome,
            fence=lambda _handle: nullcontext(),
        )
        with patch.object(
            chat_module,
            "generate_model_text",
            side_effect=AssertionError("deterministic path called provider"),
        ) as call:
            brain.respond(prompt)
        if call.call_count or brain.last_turn_metadata["source"] != expected_source:
            raise AssertionError(f"deterministic route drifted: {brain.last_turn_metadata}")
        _assert_turn_provenance(brain)
    if prepared or finalized:
        raise AssertionError(f"deterministic path touched disclosure callbacks: {prepared} / {finalized}")


def main() -> None:
    test_opt_in_nonretroactivity_and_repeated_toggles()
    test_provider_model_destination_and_no_cloud_switches()
    test_malformed_entries_seals_generations_order_and_payload_stripping()
    test_lineage_metadata_skill_laundering_and_preview_parity()
    test_conversation_digest_memory_requires_durable_remote_lineage()
    test_real_destination_provenance_round_trip()
    test_unavailable_durable_authority_strips_stored_context()
    test_crash_safe_prepare_finalize_callbacks()
    test_callback_bundle_pinning_and_strict_confirmation_status()
    test_deterministic_paths_never_prepare_or_call_provider()
    print("chat history provenance smoke passed")


if __name__ == "__main__":
    main()
