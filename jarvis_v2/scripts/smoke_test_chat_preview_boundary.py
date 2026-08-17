from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, patch

from jarvis_v2.agent.chat import ChatBrain
from jarvis_v2.memory.obsidian import ProfileGroundingView
from jarvis_v2.memory.store import ProfileKnowledgeSnapshot
from jarvis_v2.tools.conversation import make_conversation_tools


PROMPT = "How should I approach this project?"
ELIGIBLE_HISTORY_MARKER = "ELIGIBLE_PREVIEW_HISTORY_MARKER"
LEGACY_HISTORY_MARKER = "LEGACY_PREVIEW_HISTORY_MARKER"
OLLAMA_LOOPBACK_HOST = "http://127.0.0.1:11434"


def _install_durable_policy(brain: ChatBrain) -> None:
    prepared: set[tuple[str, str]] = set()

    def prepare(payload: dict[str, object]) -> object:
        decision = brain._history_policy_decision(commit=False)
        handle = (str(payload.get("policy_epoch_id") or ""), str(payload.get("policy_fingerprint") or ""))
        if handle != (decision.epoch_id, decision.fingerprint):
            raise ValueError("stale preview disclosure policy")
        prepared.add(handle)
        return handle

    def validate(handle: object) -> bool:
        decision = brain._history_policy_decision(commit=False)
        return bool(
            type(handle) is tuple
            and handle in prepared
            and handle == (decision.epoch_id, decision.fingerprint)
        )

    def finalize(handle: object, outcome: str) -> str:
        if outcome not in {"confirmed", "blocked", "uncertain"} or not validate(handle):
            raise ValueError("invalid preview disclosure finalization")
        prepared.remove(handle)
        return outcome

    brain.install_history_disclosure_callbacks(
        prepare=prepare,
        validate=validate,
        finalize=finalize,
        fence=lambda _handle: nullcontext(),
    )


def _brain(*, durable_policy: bool = True) -> tuple[ChatBrain, MagicMock, MagicMock]:
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
    store.profile_owned_memory_ids.return_value = frozenset()
    vault.read_profile_grounding.return_value = ProfileGroundingView(
        text="# Profile\n\n## Work\nFocused builder",
        verified_source_keys=(),
        invalid=False,
        truncated=False,
    )
    store.list_preferences.return_value = [
        {"category": "communication", "key": "style", "value": "direct"}
    ]
    store.search_memories.return_value = []
    store.recent_memories.return_value = [
        {
            "id": 101,
            "category": "project",
            "title": "Snapshot",
            "body": "Use one coherent read pass.",
        }
    ]
    store.list_active_skills.return_value = []
    brain = ChatBrain(
        "offline-smoke-model",
        store,
        vault,
        model_timeout_seconds=9.0,
        max_reply_tokens=77,
    )
    if durable_policy:
        _install_durable_policy(brain)
    brain.history = [{"role": "user", "content": LEGACY_HISTORY_MARKER}]
    return brain, store, vault


def _seed_eligible_history(brain: ChatBrain, store: MagicMock, vault: MagicMock) -> None:
    with patch(
        "jarvis_v2.agent.chat.generate_model_text",
        return_value=ELIGIBLE_HISTORY_MARKER,
    ) as call:
        answer = brain.respond("Why is the sky blue?")
    if answer != ELIGIBLE_HISTORY_MARKER or call.call_count != 1:
        raise AssertionError("preview history seed did not complete one mocked ChatBrain turn")
    for mock in (
        store.read_profile_knowledge_snapshot,
        vault.read_profile_grounding,
        store.list_preferences,
        store.search_memories,
        store.recent_memories,
        store.list_active_skills,
    ):
        mock.reset_mock()


def _assert_one_context_read(
    store: MagicMock,
    vault: MagicMock,
    label: str,
    *,
    profile_snapshot_reads: int = 1,
    profile_grounding_reads: int = 2,
) -> None:
    expected = {
        "profile snapshot": store.read_profile_knowledge_snapshot.call_count,
        "profile grounding": vault.read_profile_grounding.call_count,
        "preferences": store.list_preferences.call_count,
        "memory search": store.search_memories.call_count,
        "recent-memory fallback": store.recent_memories.call_count,
        "skill search": store.list_active_skills.call_count,
    }
    expected_counts = {
        "profile snapshot": profile_snapshot_reads,
        "profile grounding": profile_grounding_reads,
        "preferences": 1,
        "memory search": 1,
        "recent-memory fallback": 1,
        "skill search": 1,
    }
    wrong = {
        name: count
        for name, count in expected.items()
        if count != expected_counts[name]
    }
    if wrong:
        raise AssertionError(f"{label} did not read each context category exactly once: {wrong}")


def test_respond_uses_one_context_snapshot() -> None:
    brain, store, vault = _brain()
    _seed_eligible_history(brain, store, vault)
    captured: dict[str, object] = {}

    def fake_generate_model_text(**kwargs):
        captured.update(kwargs)
        return "offline mocked answer"

    with patch("jarvis_v2.agent.chat.generate_model_text", side_effect=fake_generate_model_text):
        answer = brain.respond(PROMPT)

    if answer != "offline mocked answer":
        raise AssertionError(f"ordinary response missed mocked model output: {answer!r}")
    _assert_one_context_read(
        store,
        vault,
        "ordinary response",
        profile_snapshot_reads=2,
        profile_grounding_reads=4,
    )
    messages = captured.get("messages")
    if not isinstance(messages, list):
        raise AssertionError(f"ordinary response did not build model messages: {captured}")
    message_text = "\n".join(str(message.get("content") or "") for message in messages)
    for expected in ("Focused builder", "communication", "Snapshot"):
        if expected not in message_text:
            raise AssertionError(f"coherent response snapshot missed {expected!r}: {message_text}")
    if message_text.count(ELIGIBLE_HISTORY_MARKER) != 1:
        raise AssertionError(f"snapshot response duplicated eligible history: {message_text}")
    if LEGACY_HISTORY_MARKER in message_text:
        raise AssertionError(f"snapshot response disclosed legacy history: {message_text}")
    metadata = brain.last_turn_metadata
    if (
        metadata.get("source") != "model"
        or metadata.get("model_timeout_seconds") != 9.0
        or metadata.get("max_reply_tokens") != 77
        or metadata.get("external_side_effect") is not False
        or metadata.get("model_request_content_in_metadata") is not False
        or metadata.get("model_response_content_in_metadata") is not False
        or metadata.get("history_retained_messages") != 3
        or metadata.get("history_eligible_messages") != 2
        or metadata.get("history_withheld_messages") != 1
        or metadata.get("history_withheld_reason_counts") != {"legacy_unmarked": 1}
    ):
        raise AssertionError(f"ordinary response metadata contract drifted: {metadata}")


def test_missing_durable_policy_strips_stored_context() -> None:
    brain, _store, _vault = _brain(durable_policy=False)
    captured: dict[str, object] = {}

    def fake_generate_model_text(**kwargs):
        captured.update(kwargs)
        return "offline mocked answer"

    with patch("jarvis_v2.agent.chat.generate_model_text", side_effect=fake_generate_model_text):
        answer = brain.respond(PROMPT)

    if answer != "offline mocked answer":
        raise AssertionError(f"missing-policy response missed mocked output: {answer!r}")
    messages = captured.get("messages")
    if not isinstance(messages, list):
        raise AssertionError(f"missing-policy response did not build model messages: {captured}")
    message_text = "\n".join(str(message.get("content") or "") for message in messages)
    for forbidden in ("Focused builder", "communication", "Snapshot", LEGACY_HISTORY_MARKER):
        if forbidden in message_text:
            raise AssertionError(f"missing durable policy disclosed {forbidden!r}: {message_text}")
    if brain.last_turn_metadata.get("history_disclosure_prepare_status") != "failed_stripped":
        raise AssertionError(
            f"missing durable policy receipt was not explicit: {brain.last_turn_metadata}"
        )


def test_direct_previews_read_once_and_report_private_boundary() -> None:
    brain, store, vault = _brain()
    body, metadata = brain.preview_loop_text(PROMPT)
    _assert_one_context_read(store, vault, "text preview")
    if metadata.get("reads_private_data") is not True or metadata.get("reads_personal_data") is not True:
        raise AssertionError(f"text preview hid its local private reads: {metadata}")
    expected_history = {
        "history_retained_messages": 1,
        "history_eligible_messages": 0,
        "history_withheld_messages": 1,
        "history_withheld_reason_counts": {"legacy_unmarked": 1},
    }
    wrong_history = {
        key: metadata.get(key)
        for key, expected in expected_history.items()
        if metadata.get(key) != expected
    }
    if wrong_history:
        raise AssertionError(f"text preview history eligibility drifted: {wrong_history} / {metadata}")
    for expected in (
        "reads local profile, preferences, memories, saved skills",
        "does not call the model",
        "does not execute tools, write memory, control the computer, or queue approvals",
    ):
        if expected not in body:
            raise AssertionError(f"text preview missed truthful boundary {expected!r}: {body}")
    if "does not read private data" in body:
        raise AssertionError(f"text preview retained the false private-read claim: {body}")

    brain, store, vault = _brain()
    metadata = brain.preview_loop(PROMPT)
    _assert_one_context_read(store, vault, "structured preview")
    if metadata.get("reads_private_data") is not True or metadata.get("memories") != 1:
        raise AssertionError(f"structured preview metadata was not grounded truthfully: {metadata}")
    wrong_history = {
        key: metadata.get(key)
        for key, expected in expected_history.items()
        if metadata.get(key) != expected
    }
    if wrong_history:
        raise AssertionError(
            f"structured preview history eligibility drifted: {wrong_history} / {metadata}"
        )


def test_classification_only_path_reads_no_context() -> None:
    brain, store, vault = _brain()
    metadata = brain._classify_loop("Please run a script")
    if metadata.get("reply_path") != "safety_preflight_guidance":
        raise AssertionError(f"classification-only path changed safety routing: {metadata}")
    if any(
        mock.call_count
        for mock in (
            store.read_profile_knowledge_snapshot,
            vault.read_profile_grounding,
            store.list_preferences,
            store.search_memories,
            store.recent_memories,
            store.list_active_skills,
        )
    ):
        raise AssertionError("classification-only path touched private context")
    for context_key in ("has_profile", "preferences", "memories", "skills", "recent_history_messages"):
        if context_key in metadata:
            raise AssertionError(f"classification-only metadata invented {context_key}: {metadata}")


def test_tool_preview_preserves_protected_metadata() -> None:
    _brain_instance, store, vault = _brain()
    handler = next(
        tool
        for tool in make_conversation_tools(store, vault, "offline-smoke-session")
        if tool.__name__ == "chat_loop_preview"
    )
    result = handler({"prompt": PROMPT})
    metadata = result.metadata
    if not result.ok:
        raise AssertionError(f"direct chat-loop tool preview failed: {result}")
    expected = {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "writes_memory": False,
        "controls_computer": False,
        "reads_private_data": True,
        "reads_personal_data": True,
        "would_call_model": True,
    }
    wrong = {key: metadata.get(key) for key, value in expected.items() if metadata.get(key) is not value}
    if wrong:
        raise AssertionError(f"chat-loop tool preview metadata contract drifted: {wrong} / {metadata}")
    _assert_one_context_read(store, vault, "chat-loop tool preview")


def main() -> None:
    with patch.dict(
        os.environ,
        {
            "OLLAMA_HOST": OLLAMA_LOOPBACK_HOST,
            "OLLAMA_NO_CLOUD": "1",
            "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
        },
        clear=False,
    ):
        test_respond_uses_one_context_snapshot()
        test_missing_durable_policy_strips_stored_context()
        test_direct_previews_read_once_and_report_private_boundary()
        test_classification_only_path_reads_no_context()
        test_tool_preview_preserves_protected_metadata()
    print("chat preview boundary smoke passed")


if __name__ == "__main__":
    main()
