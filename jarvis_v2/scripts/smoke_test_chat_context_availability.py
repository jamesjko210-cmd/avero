from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, patch

from jarvis_v2.agent.chat import ChatBrain
from jarvis_v2.memory.obsidian import ProfileGroundingView
from jarvis_v2.memory.store import ProfileKnowledgeSnapshot


PROMPT = "What do you know about me?"
PRIVATE_FAILURE = "private store failed at /\x55sers/example/secret/context.db"


def _brain() -> tuple[ChatBrain, MagicMock, MagicMock]:
    store = MagicMock()
    vault = MagicMock()
    store.read_profile_knowledge_snapshot.return_value = ProfileKnowledgeSnapshot(
        store_identity="0" * 32,
        notes=(),
        source_count=0,
        invalid_count=0,
        truncated=False,
    )
    store.profile_owned_memory_ids.return_value = frozenset()
    vault.read_profile_grounding.return_value = ProfileGroundingView(
        text="# Profile",
        verified_source_keys=(),
        invalid=False,
        truncated=False,
    )

    @contextmanager
    def hold_profile_snapshot():
        yield store.read_profile_knowledge_snapshot()

    @contextmanager
    def hold_profile_file():
        yield

    @contextmanager
    def hold_profile_ownership(memory_ids):
        yield store.profile_owned_memory_ids(memory_ids)

    store.hold_profile_knowledge_snapshot.side_effect = hold_profile_snapshot
    vault.profile_grounding_evidence_lock.side_effect = hold_profile_file
    store.profile_memory_ownership_egress_fence.side_effect = hold_profile_ownership
    brain = ChatBrain("offline-smoke-model", store, vault)
    brain.install_history_disclosure_callbacks(
        prepare=lambda _payload: "availability-receipt",
        validate=lambda handle: handle == "availability-receipt",
        finalize=lambda _handle, outcome: outcome,
        fence=lambda _handle: nullcontext(),
    )
    return brain, store, vault


def _set_empty(store: MagicMock, vault: MagicMock) -> None:
    store.read_profile_knowledge_snapshot.side_effect = None
    vault.read_profile_grounding.side_effect = None
    vault.read_profile_grounding.return_value = ProfileGroundingView(
        text="# Profile",
        verified_source_keys=(),
        invalid=False,
        truncated=False,
    )
    store.list_preferences.side_effect = None
    store.list_preferences.return_value = []
    store.search_memories.side_effect = None
    store.search_memories.return_value = []
    store.recent_memories.side_effect = None
    store.recent_memories.return_value = []
    store.list_active_skills.side_effect = None
    store.list_active_skills.return_value = []


def _set_unavailable(store: MagicMock, vault: MagicMock) -> None:
    failure = RuntimeError(PRIVATE_FAILURE)
    store.read_profile_knowledge_snapshot.side_effect = failure
    vault.read_profile_grounding.side_effect = failure
    store.list_preferences.side_effect = failure
    store.search_memories.side_effect = failure
    store.recent_memories.side_effect = failure
    store.list_active_skills.side_effect = failure


def _assert_bounded(metadata: dict[str, object], output: str) -> None:
    combined = f"{output}\n{metadata}"
    for forbidden in (PRIVATE_FAILURE, "/\x55sers/operator", "context.db"):
        if forbidden in combined:
            raise AssertionError(f"context availability leaked private failure detail: {combined}")


def test_verified_empty_stays_distinct_from_unavailable() -> None:
    brain, store, vault = _brain()
    _set_empty(store, vault)

    output = brain.respond(PROMPT)
    metadata = brain.last_turn_metadata
    expected_states = {
        "profile": "empty",
        "preferences": "empty",
        "memory": "empty",
        "skills": "empty",
    }
    if "I do not see matching saved context yet" not in output:
        raise AssertionError(f"verified-empty context lost its honest empty response: {output}")
    if metadata.get("context_source_states") != expected_states:
        raise AssertionError(f"verified-empty source states drifted: {metadata}")
    if metadata.get("context_unavailable_sources") != []:
        raise AssertionError(f"verified-empty context claimed an outage: {metadata}")
    if metadata.get("context_fully_available") is not True:
        raise AssertionError(f"verified-empty context should be fully readable: {metadata}")


def test_total_outage_never_claims_memory_is_empty() -> None:
    brain, store, vault = _brain()
    _set_unavailable(store, vault)

    output = brain.respond(PROMPT)
    metadata = brain.last_turn_metadata
    expected_sources = ["profile", "preferences", "memory", "skills"]
    if "couldn't verify saved personal context right now" not in output:
        raise AssertionError(f"context outage missed its explicit availability warning: {output}")
    if "I will not claim that nothing is saved" not in output:
        raise AssertionError(f"context outage did not protect existing personal knowledge: {output}")
    if "I do not see matching saved context yet" in output:
        raise AssertionError(f"context outage falsely reported verified-empty memory: {output}")
    if metadata.get("context_unavailable_sources") != expected_sources:
        raise AssertionError(f"context outage source list drifted: {metadata}")
    if metadata.get("context_fully_available") is not False:
        raise AssertionError(f"context outage claimed complete availability: {metadata}")
    if metadata.get("model_error"):
        raise AssertionError(f"local context outage was mislabeled as a model error: {metadata}")
    _assert_bounded(metadata, output)


def test_partial_outage_uses_only_verified_personal_context() -> None:
    brain, store, vault = _brain()
    vault.read_profile_grounding.return_value = ProfileGroundingView(
        text="# Profile\n\n## Style\nDirect and warm",
        verified_source_keys=(),
        invalid=False,
        truncated=False,
    )
    store.list_preferences.side_effect = RuntimeError(PRIVATE_FAILURE)
    store.search_memories.return_value = [
        {
            "id": 101,
            "category": "principle",
            "title": "Personal agent moat",
            "body": "Use operator-specific knowledge, with visible provenance and safety.",
        }
    ]
    store.list_active_skills.return_value = []

    output = brain.respond(PROMPT)
    metadata = brain.last_turn_metadata
    for expected in (
        "Profile context",
        "Direct and warm",
        "Relevant memory",
        "Personal agent moat",
        "I could not fully check preferences",
        "only the portion I could verify",
    ):
        if expected not in output:
            raise AssertionError(f"partial context outage missed {expected!r}: {output}")
    if metadata.get("context_unavailable_sources") != ["preferences"]:
        raise AssertionError(f"partial outage did not isolate the failed source: {metadata}")
    _assert_bounded(metadata, output)


def test_preference_overflow_never_claims_complete_context() -> None:
    brain, store, vault = _brain()
    _set_empty(store, vault)
    overflow_marker = "CRITICAL_PREFERENCE_BEYOND_DISCLOSURE_BOUND"
    rows = [
        {
            "category": "a",
            "key": f"preference-{index:02d}",
            "value": f"value-{index:02d}",
        }
        for index in range(20)
    ] + [
        {
            "category": "z",
            "key": "critical",
            "value": overflow_marker,
        }
    ]
    requested_limits: list[int] = []

    def list_preferences(*, status: str, limit: int):
        if status != "active":
            raise AssertionError(f"preference context requested wrong status: {status}")
        requested_limits.append(limit)
        return rows[:limit]

    store.list_preferences.side_effect = list_preferences
    captured: dict[str, object] = {}

    def fake_generate_model_text(**kwargs):
        captured.update(kwargs)
        return "Preference-aware answer."

    with patch("jarvis_v2.agent.chat.generate_model_text", side_effect=fake_generate_model_text):
        output = brain.respond("Help me make a decision")

    if requested_limits != [21]:
        raise AssertionError(f"preference overflow probe should request one sentinel row: {requested_limits}")
    metadata = brain.last_turn_metadata
    if metadata.get("context_unavailable_sources") != ["preferences"]:
        raise AssertionError(f"preference overflow was reported as complete context: {metadata}")
    if metadata.get("context_fully_available") is not False:
        raise AssertionError(f"preference overflow claimed full availability: {metadata}")
    if "I couldn't fully check preferences for this answer" not in output:
        raise AssertionError(f"preference overflow missed its user-visible partial-context note: {output}")
    messages = captured.get("messages")
    if not isinstance(messages, list):
        raise AssertionError(f"preference overflow did not produce inspectable model messages: {captured}")
    message_text = "\n".join(str(row.get("content") or "") for row in messages)
    if message_text.count("[a] preference-") != 20:
        raise AssertionError("preference overflow did not preserve the exact 20-item disclosure bound")
    if overflow_marker in message_text or overflow_marker in output or overflow_marker in str(metadata):
        raise AssertionError("preference overflow disclosed the sentinel row beyond the context bound")
    _assert_bounded(metadata, f"{output}\n{message_text}")


def test_source_health_recovers_without_stale_outage_state() -> None:
    brain, store, vault = _brain()
    _set_unavailable(store, vault)
    brain.respond(PROMPT)

    _set_empty(store, vault)
    preview, metadata = brain.preview_loop_text(PROMPT)
    if metadata.get("context_unavailable_sources") != []:
        raise AssertionError(f"recovered context retained stale outage state: {metadata}")
    if metadata.get("context_fully_available") is not True:
        raise AssertionError(f"recovered context did not restore readable state: {metadata}")
    for source in ("profile", "preferences", "memory", "skills"):
        if f"- {source}: empty" not in preview:
            raise AssertionError(f"context preview missed recovered {source} health: {preview}")


def test_memory_search_failure_is_not_hidden_by_recent_fallback() -> None:
    brain, store, vault = _brain()
    _set_empty(store, vault)
    store.search_memories.side_effect = RuntimeError(PRIVATE_FAILURE)
    store.recent_memories.return_value = [
        {
            "id": 102,
            "category": "recent",
            "title": "Fallback context",
            "body": "This row is readable, but relevance search is unavailable.",
        }
    ]

    output = brain.respond(PROMPT)
    metadata = brain.last_turn_metadata
    if "Fallback context" not in output:
        raise AssertionError(f"readable fallback context was discarded: {output}")
    if "I could not fully check memory" not in output:
        raise AssertionError(f"failed relevance search was hidden by fallback rows: {output}")
    if metadata.get("context_unavailable_sources") != ["memory"]:
        raise AssertionError(f"failed relevance search was marked fully available: {metadata}")
    _assert_bounded(metadata, output)


def test_short_prompt_still_checks_skill_source_health() -> None:
    brain, store, vault = _brain()
    _set_empty(store, vault)
    store.list_active_skills.side_effect = RuntimeError(PRIVATE_FAILURE)

    _preview, metadata = brain.preview_loop_text("Hi")
    if metadata.get("context_unavailable_sources") != ["skills"]:
        raise AssertionError(f"short prompt skipped skill-source health: {metadata}")
    if store.list_active_skills.call_count != 1:
        raise AssertionError("short prompt did not check the skill source exactly once")
    _assert_bounded(metadata, "")


def test_ordinary_model_reply_discloses_partial_personal_context() -> None:
    brain, store, vault = _brain()
    _set_empty(store, vault)
    store.list_preferences.side_effect = RuntimeError(PRIVATE_FAILURE)
    captured: dict[str, object] = {}

    def fake_generate_model_text(**kwargs):
        captured.update(kwargs)
        return "A grounded decision framework."

    with patch("jarvis_v2.agent.chat.generate_model_text", side_effect=fake_generate_model_text):
        output = brain.respond("Help me think through this choice")

    if "A grounded decision framework." not in output:
        raise AssertionError(f"ordinary model reply lost its answer: {output}")
    if "I couldn't fully check preferences for this answer" not in output:
        raise AssertionError(f"ordinary model reply hid unavailable personal context: {output}")
    if "I did not assume the missing context was empty" not in output:
        raise AssertionError(f"ordinary model reply missed its non-empty assumption guard: {output}")
    messages = captured.get("messages")
    if not isinstance(messages, list):
        raise AssertionError(f"ordinary model call did not expose inspectable messages: {captured}")
    message_text = "\n".join(str(row.get("content") or "") for row in messages)
    for expected in (
        "Personal context availability for this turn",
        "preferences",
        "Do not infer that their contents are empty",
        "do not invent missing personal context",
        "supersedes any older availability notice",
    ):
        if expected not in message_text:
            raise AssertionError(f"model context guard missed {expected!r}: {message_text}")
    _assert_bounded(brain.last_turn_metadata, f"{output}\n{message_text}")


def test_ordinary_fallback_reply_discloses_total_context_outage() -> None:
    brain, store, vault = _brain()
    _set_unavailable(store, vault)

    with patch(
        "jarvis_v2.agent.chat.generate_model_text",
        side_effect=RuntimeError("model failed at /\x55sers/example/secret/model.sock"),
    ):
        output = brain.respond("Can we just talk normally?")

    for expected in (
        "Personal context note",
        "profile, preferences, memory, skills",
        "it may be less personalized",
        "I did not assume the missing context was empty",
    ):
        if expected not in output:
            raise AssertionError(f"ordinary fallback outage missed {expected!r}: {output}")
    if "model.sock" in output or "/\x55sers/operator" in output:
        raise AssertionError(f"ordinary fallback leaked model/context failure detail: {output}")
    if "grounded in the memory I can read" in output:
        raise AssertionError(f"ordinary fallback contradicted its total context outage: {output}")
    if "full local model is offline" in output.lower() or "full local chat model is offline" in output.lower():
        raise AssertionError(f"ordinary fallback made a provider-specific model claim: {output}")
    _assert_bounded(brain.last_turn_metadata, output)


def test_ordinary_reply_has_no_outage_note_when_sources_are_readable() -> None:
    brain, store, vault = _brain()
    _set_empty(store, vault)
    captured: dict[str, object] = {}

    def fake_generate_model_text(**kwargs):
        captured.update(kwargs)
        return "Readable-context answer."

    with patch("jarvis_v2.agent.chat.generate_model_text", side_effect=fake_generate_model_text):
        output = brain.respond("Help me think through this choice")

    if output != "Readable-context answer.":
        raise AssertionError(f"readable personal context gained a false outage warning: {output}")
    messages = captured.get("messages")
    if not isinstance(messages, list):
        raise AssertionError(f"readable model call did not expose messages: {captured}")
    status_messages = [
        str(row.get("content") or "")
        for row in messages
        if "Personal context availability for this turn" in str(row.get("content") or "")
    ]
    if len(status_messages) != 1 or "all configured local sources were readable" not in status_messages[0]:
        raise AssertionError(f"readable personal context missed its authoritative current status: {messages}")


def test_recovery_system_status_supersedes_stale_history_warning() -> None:
    brain, store, vault = _brain()
    _set_unavailable(store, vault)
    with patch("jarvis_v2.agent.chat.generate_model_text", return_value="Degraded answer."):
        first = brain.respond("Help me think through this choice")
    if "Personal context note" not in first:
        raise AssertionError(f"outage fixture missed its history warning: {first}")

    _set_empty(store, vault)
    captured: dict[str, object] = {}

    def fake_generate_model_text(**kwargs):
        captured.update(kwargs)
        return "Recovered answer."

    with patch("jarvis_v2.agent.chat.generate_model_text", side_effect=fake_generate_model_text):
        second = brain.respond("Help me think through the next choice")

    if second != "Recovered answer.":
        raise AssertionError(f"recovered turn retained a user-visible outage warning: {second}")
    messages = captured.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise AssertionError(f"recovered model turn missed messages: {captured}")
    current_status = str(messages[-2].get("content") or "")
    if messages[-2].get("role") != "system" or (
        "all configured local sources were readable" not in current_status
        or "older availability notice" not in current_status
        or "historical" not in current_status
    ):
        raise AssertionError(f"recovered turn did not supersede stale outage history: {messages}")
    if messages[-1].get("role") != "user":
        raise AssertionError(f"current availability status was not adjacent to the user turn: {messages[-2:]}")


def main() -> None:
    with patch.dict(
        os.environ,
        {
            "OLLAMA_NO_CLOUD": "1",
            "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
        },
        clear=False,
    ):
        test_verified_empty_stays_distinct_from_unavailable()
        test_total_outage_never_claims_memory_is_empty()
        test_partial_outage_uses_only_verified_personal_context()
        test_preference_overflow_never_claims_complete_context()
        test_source_health_recovers_without_stale_outage_state()
        test_memory_search_failure_is_not_hidden_by_recent_fallback()
        test_short_prompt_still_checks_skill_source_health()
        test_ordinary_model_reply_discloses_partial_personal_context()
        test_ordinary_fallback_reply_discloses_total_context_outage()
        test_ordinary_reply_has_no_outage_note_when_sources_are_readable()
        test_recovery_system_status_supersedes_stale_history_warning()
    print("Chat context availability smoke checks passed.")


if __name__ == "__main__":
    main()
