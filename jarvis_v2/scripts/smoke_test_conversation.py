from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools.conversation import _conversation_handoff_metadata, _metadata_bool


class HostileRow:
    def __init__(self, marker: str):
        self.marker = marker

    def __getitem__(self, _key):
        raise RuntimeError(self.marker)

    def keys(self):
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-row {self.marker}>"


def assert_safe(metadata: dict, label: str) -> None:
    unsafe = [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "controls_computer",
        "executes_side_effect",
    ]
    if any(metadata.get(key) for key in unsafe):
        raise SystemExit(f"{label} should stay safe/read-only for execution boundaries: {metadata}")


def assert_no_local_path(value: str, label: str) -> None:
    for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if fragment in value:
            raise SystemExit(f"{label} leaked a local path: {value}")


def assert_conversation_handoff(
    metadata: dict,
    handoff_key: str,
    label: str,
    *,
    state_changed: bool = False,
    changed: list[str] | None = None,
    content_in_handoff: bool = False,
    writes: bool = False,
) -> None:
    prefix = handoff_key.removesuffix("_handoff")
    handoff = metadata.get(handoff_key)
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed {handoff_key}: {metadata}")
    if metadata.get(f"{handoff_key}_ready") is not True:
        raise SystemExit(f"{label} did not mark {handoff_key} ready: {metadata}")
    if handoff.get("handoff_ready") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} handoff is not operator-ready: {handoff}")

    expected_changed = changed or []
    expected_next = handoff.get("next_commands") or []
    if isinstance(expected_next, dict):
        expected_next = [str(value) for value in expected_next.values() if str(value or "").strip()]
    else:
        expected_next = [str(value) for value in expected_next if str(value or "").strip()]
    expected_first = expected_next[0] if expected_next else ""

    checks = {
        "state_changed": state_changed,
        "changed": expected_changed,
        "content_in_handoff": content_in_handoff,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "next_safe_command": expected_first,
        "next_safe_commands": expected_next,
        "next_safe_command_count": len(expected_next),
    }
    for key, expected in checks.items():
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} handoff {key} mismatch: {handoff}")
        if metadata.get(key) != expected:
            raise SystemExit(f"{label} metadata {key} mismatch: {metadata}")
        prefixed_key = f"{prefix}_{key}"
        if metadata.get(prefixed_key) != expected:
            raise SystemExit(f"{label} metadata {prefixed_key} mismatch: {metadata}")

    if metadata.get(f"{prefix}_handoff_ready") is not True or metadata.get(f"{prefix}_ready_for_operator") is not True:
        raise SystemExit(f"{label} missed prefixed ready aliases: {metadata}")

    boundaries = handoff.get("boundaries") or {}
    for key in ("writes_files", "writes_memory", "writes_notes"):
        if boundaries.get(key) is not writes or metadata.get(key) is not writes:
            raise SystemExit(f"{label} write boundary {key} mismatch: handoff={handoff} metadata={metadata}")
    if boundaries.get("read_only") is not (not writes):
        raise SystemExit(f"{label} read-only boundary mismatch: {handoff}")
    unsafe = [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "controls_computer",
        "executes_side_effect",
    ]
    if any(boundaries.get(key) for key in unsafe):
        raise SystemExit(f"{label} handoff boundaries granted unsafe authority: {handoff}")


def assert_conversation_exact_metadata_bool() -> None:
    if _metadata_bool(True) is not True:
        raise SystemExit("conversation exact bool helper should preserve True.")
    if _metadata_bool(False, default=True) is not False:
        raise SystemExit("conversation exact bool helper should preserve False.")
    for value in ("true", "false", "yes", "0", 1, 0, [], ["content"], None):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"conversation exact bool helper should reject malformed handoff flags: {value!r}")
    if _metadata_bool("fallback", default=True) is not True:
        raise SystemExit("conversation exact bool helper should honor explicit malformed-value default.")


def assert_conversation_malformed_handoff_flags() -> None:
    handoff = {
        "ready_for_operator": True,
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "next_commands": ["recent conversation"],
        "boundaries": {
            "read_only": True,
            "writes_files": False,
            "writes_memory": False,
            "writes_notes": False,
            "calls_model": False,
            "executes_tools": False,
            "queues_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "controls_computer": False,
            "executes_side_effect": False,
        },
    }
    metadata = _conversation_handoff_metadata("conversation_recent_handoff", handoff)
    if metadata.get("state_changed") is not False or metadata.get("conversation_recent_state_changed") is not False:
        raise SystemExit(f"malformed conversation state_changed should not become truthy: {metadata}")
    if metadata.get("content_in_handoff") is not False or metadata.get("conversation_recent_content_in_handoff") is not False:
        raise SystemExit(f"malformed conversation content_in_handoff should not become truthy: {metadata}")


def assert_conversation_retrieval_tolerates_malformed_message_rows(runtime: JarvisRuntime) -> None:
    marker = "SHOULD_NOT_LEAK_CONVERSATION_HOSTILE_MESSAGE_ROW"

    readable_recent = runtime.store.recent_messages(limit=5, session_id=runtime.session_id)
    original_recent_messages = runtime.store.recent_messages

    def hostile_recent_messages(*_args, **_kwargs):
        return [HostileRow(marker), *readable_recent]

    try:
        runtime.store.recent_messages = hostile_recent_messages  # type: ignore[method-assign]
        recent = runtime.registry.get("recent_conversation").handler({"limit": 5})
    finally:
        runtime.store.recent_messages = original_recent_messages  # type: ignore[method-assign]

    if not recent.ok:
        raise SystemExit(f"recent_conversation should tolerate malformed message rows: {recent}")
    recent_metadata = recent.metadata
    assert_safe(recent_metadata, "recent_conversation malformed message rows")
    assert_conversation_handoff(
        recent_metadata,
        "conversation_recent_handoff",
        "recent_conversation malformed message rows",
        content_in_handoff=True,
    )
    if recent_metadata.get("readable_message_rows") != len(readable_recent) or recent_metadata.get("unreadable_message_rows") != 1:
        raise SystemExit(f"recent_conversation missed malformed-row counts: {recent_metadata}")
    recent_handoff = recent_metadata.get("conversation_recent_handoff") or {}
    if recent_handoff.get("count") != len(readable_recent):
        raise SystemExit(f"recent_conversation handoff count should track readable rows: {recent_metadata}")
    if "unreadable message row(s) hidden for safety" not in recent.output:
        raise SystemExit(f"recent_conversation should report hidden malformed rows: {recent.output}")
    recent_payload = recent.output + json.dumps(recent_metadata, sort_keys=True, default=str)
    if marker in recent_payload:
        raise SystemExit("recent_conversation leaked raw malformed message row text.")

    readable_search = runtime.store.search_messages("session", 12)
    original_search_messages = runtime.store.search_messages

    def hostile_search_messages(*_args, **_kwargs):
        return [HostileRow(marker), *readable_search]

    try:
        runtime.store.search_messages = hostile_search_messages  # type: ignore[method-assign]
        search = runtime.registry.get("search_conversations").handler({"query": "session", "limit": 12})
    finally:
        runtime.store.search_messages = original_search_messages  # type: ignore[method-assign]

    if not search.ok:
        raise SystemExit(f"search_conversations should tolerate malformed message rows: {search}")
    search_metadata = search.metadata
    assert_safe(search_metadata, "search_conversations malformed message rows")
    assert_conversation_handoff(
        search_metadata,
        "conversation_search_handoff",
        "search_conversations malformed message rows",
        content_in_handoff=bool(readable_search),
    )
    if search_metadata.get("readable_message_rows") != len(readable_search) or search_metadata.get("unreadable_message_rows") != 1:
        raise SystemExit(f"search_conversations missed malformed-row counts: {search_metadata}")
    search_handoff = search_metadata.get("conversation_search_handoff") or {}
    if search_handoff.get("count") != len(readable_search):
        raise SystemExit(f"search_conversations handoff count should track readable rows: {search_metadata}")
    if "unreadable message row(s) hidden for safety" not in search.output:
        raise SystemExit(f"search_conversations should report hidden malformed rows: {search.output}")
    search_payload = search.output + json.dumps(search_metadata, sort_keys=True, default=str)
    if marker in search_payload:
        raise SystemExit("search_conversations leaked raw malformed message row text.")


def assert_chat_diagnostics_tolerate_malformed_message_rows(runtime: JarvisRuntime) -> None:
    marker = "SHOULD_NOT_LEAK_CHAT_DIAGNOSTIC_HOSTILE_MESSAGE_ROW"
    readable_messages = runtime.store.recent_messages(limit=32, session_id=runtime.session_id)
    original_recent_messages = runtime.store.recent_messages

    def hostile_recent_messages(*_args, **_kwargs):
        return [HostileRow(marker), *readable_messages]

    try:
        runtime.store.recent_messages = hostile_recent_messages  # type: ignore[method-assign]
        health = runtime.registry.get("chat_response_health").handler({"limit": 32})
        continuity = runtime.registry.get("chat_continuity_brief").handler({"limit": 32})
    finally:
        runtime.store.recent_messages = original_recent_messages  # type: ignore[method-assign]

    for label, result in (
        ("chat_response_health", health),
        ("chat_continuity_brief", continuity),
    ):
        if not result.ok:
            raise SystemExit(f"{label} should tolerate malformed message rows: {result}")
        metadata = result.metadata
        assert_safe(metadata, f"{label} malformed message rows")
        if metadata.get("readable_message_rows") != len(readable_messages):
            raise SystemExit(f"{label} missed readable malformed-row count: {metadata}")
        if metadata.get("unreadable_message_rows") != 1:
            raise SystemExit(f"{label} missed unreadable malformed-row count: {metadata}")
        if "unreadable message row" not in result.output.lower():
            raise SystemExit(f"{label} should report hidden malformed rows: {result.output}")
        payload = result.output + json.dumps(metadata, sort_keys=True, default=str)
        if marker in payload:
            raise SystemExit(f"{label} leaked raw malformed message row text.")


def assert_chat_response_health_reports_ws4_measurement(runtime: JarvisRuntime) -> None:
    probe_session = f"{runtime.session_id}-ws4-measurement"
    seeded_responses = [
        {
            "source": "model",
            "reply_path": "model",
            "used_model": True,
            "used_fallback": False,
            "model": "mock-chat",
            "model_timeout_seconds": 2.5,
            "latency_ms": 100,
        },
        {
            "source": "model",
            "reply_path": "model",
            "used_model": True,
            "used_fallback": False,
            "model": "mock-chat",
            "model_timeout_seconds": 2.5,
            "duration_ms": 200,
        },
        {
            "source": "fallback",
            "reply_path": "fallback",
            "used_model": False,
            "used_fallback": True,
            "model": "mock-chat",
            "model_timeout_seconds": 2.5,
            "latency_ms": 300,
        },
        {
            "source": "safety_preflight_guidance",
            "reply_path": "safety_preflight_guidance",
            "used_model": False,
            "used_fallback": False,
            "model": "mock-chat",
            "model_timeout_seconds": 2.5,
            "latency_ms": 400,
        },
    ]
    for index, chat_response in enumerate(seeded_responses, start=1):
        runtime.store.log_message(probe_session, "user", f"WS4 probe turn {index}")
        runtime.store.log_message(
            probe_session,
            "assistant",
            f"WS4 probe response {index}",
            {"runtime_route": "chat", "chat_response": chat_response},
        )
    runtime.store.log_message(
        probe_session,
        "assistant",
        "Did you mean a command?",
        {"runtime_route": "chat", "chat_response": {"runtime_route": "command_suggestion"}},
    )
    runtime.store.log_message(
        probe_session,
        "assistant",
        "Tool response",
        {"runtime_route": "tools", "tool": "jarvis_status"},
    )

    health = runtime.registry.get("chat_response_health").handler({"session_id": probe_session, "limit": 20})
    if not health.ok:
        raise SystemExit(f"chat_response_health should report WS4 measurement state: {health}")
    metadata = health.metadata
    assert_safe(metadata, "chat_response_health WS4 measurement")
    expected = {
        "assistant_messages": 6,
        "chat_responses": 5,
        "ordinary_chat_responses": 4,
        "tool_responses": 1,
        "model_responses": 2,
        "fallback_responses": 1,
        "model_response_rate_pct": 50.0,
        "fallback_response_rate_pct": 25.0,
        "latency_measurement_available": True,
        "latency_sample_count": 4,
        "latency_p50_ms": 250.0,
        "latency_p95_ms": 385.0,
        "ws4_required_chat_turns": 20,
        "ws4_turns_remaining": 16,
        "ws4_latency_samples_remaining": 16,
        "ws4_measurement_ready": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SystemExit(f"chat_response_health WS4 metadata {key} mismatch: {metadata}")
    if metadata.get("source_counts") != {"fallback": 1, "model": 2, "safety_preflight_guidance": 1}:
        raise SystemExit(f"chat_response_health missed source counts: {metadata}")
    for required in [
        "ordinary chat responses: 4",
        "latency samples: 4",
        "latency p50/p95: 250.0ms / 385.0ms",
        "turns remaining: 16",
        "ready for WS4 measurement review: False",
        "reads stored response metadata only",
    ]:
        if required not in health.output:
            raise SystemExit(f"chat_response_health output missed {required!r}: {health.output}")


def assert_chat_safety_and_learning_tolerate_malformed_message_rows(runtime: JarvisRuntime) -> None:
    marker = "SHOULD_NOT_LEAK_CHAT_LEARNING_HOSTILE_MESSAGE_ROW"
    readable_messages = runtime.store.recent_messages(limit=40, session_id=runtime.session_id)
    original_recent_messages = runtime.store.recent_messages

    def hostile_recent_messages(*_args, **_kwargs):
        return [HostileRow(marker), *readable_messages]

    try:
        runtime.store.recent_messages = hostile_recent_messages  # type: ignore[method-assign]
        safety = runtime.registry.get("chat_safety_report").handler({})
        learning = runtime.registry.get("session_learning_preview").handler({"limit": 40})
    finally:
        runtime.store.recent_messages = original_recent_messages  # type: ignore[method-assign]

    for label, result in (
        ("chat_safety_report", safety),
        ("session_learning_preview", learning),
    ):
        if not result.ok:
            raise SystemExit(f"{label} should tolerate malformed message rows: {result}")
        metadata = result.metadata
        assert_safe(metadata, f"{label} malformed message rows")
        if metadata.get("readable_message_rows") != len(readable_messages):
            raise SystemExit(f"{label} missed readable malformed-row count: {metadata}")
        if metadata.get("unreadable_message_rows") != 1:
            raise SystemExit(f"{label} missed unreadable malformed-row count: {metadata}")
        if "unreadable message row(s) hidden for safety" not in result.output.lower():
            raise SystemExit(f"{label} should report hidden malformed rows: {result.output}")
        payload = result.output + json.dumps(metadata, sort_keys=True, default=str)
        if marker in payload:
            raise SystemExit(f"{label} leaked raw malformed message row text.")


def assert_chat_context_previews_tolerate_malformed_message_rows(runtime: JarvisRuntime) -> None:
    marker = "SHOULD_NOT_LEAK_CHAT_CONTEXT_HOSTILE_MESSAGE_ROW"
    readable_messages = runtime.store.recent_messages(limit=8, session_id=runtime.session_id)
    original_recent_messages = runtime.store.recent_messages

    def hostile_recent_messages(*_args, **_kwargs):
        return [HostileRow(marker), *readable_messages]

    try:
        runtime.store.recent_messages = hostile_recent_messages  # type: ignore[method-assign]
        context = runtime.registry.get("chat_context").handler({"prompt": "what should Jarvis remember", "limit": 8})
        prompt_preview = runtime.registry.get("chat_prompt_preview").handler(
            {"prompt": "what should Jarvis remember", "limit": 8}
        )
    finally:
        runtime.store.recent_messages = original_recent_messages  # type: ignore[method-assign]

    for label, result in (
        ("chat_context", context),
        ("chat_prompt_preview", prompt_preview),
    ):
        if not result.ok:
            raise SystemExit(f"{label} should tolerate malformed context rows: {result}")
        metadata = result.metadata
        assert_safe(metadata, f"{label} malformed message rows")
        if metadata.get("readable_message_rows") != len(readable_messages):
            raise SystemExit(f"{label} missed readable context-row count: {metadata}")
        if metadata.get("unreadable_message_rows") != 1:
            raise SystemExit(f"{label} missed unreadable context-row count: {metadata}")
        if "unreadable message row(s) hidden for safety" not in result.output.lower():
            raise SystemExit(f"{label} should report hidden malformed context rows: {result.output}")
        payload = result.output + json.dumps(metadata, sort_keys=True, default=str)
        if marker in payload:
            raise SystemExit(f"{label} leaked raw malformed message row text.")


def assert_conversation_session_listing_tolerates_malformed_session_rows(runtime: JarvisRuntime) -> None:
    marker = "SHOULD_NOT_LEAK_CONVERSATION_HOSTILE_SESSION_ROW"
    readable_sessions = runtime.store.list_sessions(20)
    original_list_sessions = runtime.store.list_sessions

    def hostile_list_sessions(*_args, **_kwargs):
        return [HostileRow(marker), *readable_sessions]

    try:
        runtime.store.list_sessions = hostile_list_sessions  # type: ignore[method-assign]
        result = runtime.registry.get("list_sessions").handler({"limit": 20})
    finally:
        runtime.store.list_sessions = original_list_sessions  # type: ignore[method-assign]

    if not result.ok:
        raise SystemExit(f"list_sessions should tolerate malformed session rows: {result}")
    metadata = result.metadata
    assert_safe(metadata, "list_sessions malformed session rows")
    assert_conversation_handoff(
        metadata,
        "conversation_sessions_handoff",
        "list_sessions malformed session rows",
        content_in_handoff=bool(readable_sessions),
    )
    if metadata.get("readable_session_rows") != len(readable_sessions) or metadata.get("unreadable_session_rows") != 1:
        raise SystemExit(f"list_sessions missed malformed session-row counts: {metadata}")
    handoff = metadata.get("conversation_sessions_handoff") or {}
    if handoff.get("count") != len(readable_sessions):
        raise SystemExit(f"list_sessions handoff count should track readable rows: {metadata}")
    if "unreadable session row(s) hidden for safety" not in result.output:
        raise SystemExit(f"list_sessions should report hidden malformed session rows: {result.output}")
    payload = result.output + json.dumps(metadata, sort_keys=True, default=str)
    if marker in payload:
        raise SystemExit("list_sessions leaked raw malformed session row text.")


def assert_conversation_write_tools_tolerate_malformed_message_rows(runtime: JarvisRuntime) -> None:
    marker = "SHOULD_NOT_LEAK_CONVERSATION_WRITE_HOSTILE_MESSAGE_ROW"
    readable_messages = runtime.store.recent_messages(limit=40, session_id=runtime.session_id)
    original_recent_messages = runtime.store.recent_messages

    def hostile_recent_messages(*_args, **_kwargs):
        return [HostileRow(marker), *readable_messages]

    try:
        runtime.store.recent_messages = hostile_recent_messages  # type: ignore[method-assign]
        summary = runtime.registry.get("summarize_session").handler({"limit": 40})
        export = runtime.registry.get("export_session").handler({"limit": 40})
        draft = runtime.registry.get("draft_skill_from_session").handler(
            {"name": "Malformed Conversation Row Draft", "limit": 40}
        )
    finally:
        runtime.store.recent_messages = original_recent_messages  # type: ignore[method-assign]

    cases = [
        ("summarize_session", summary, "conversation_summary_handoff", ["session_summary"], True),
        ("export_session", export, "conversation_export_handoff", ["session_export"], False),
        ("draft_skill_from_session", draft, "conversation_skill_draft_handoff", ["skill_draft"], False),
    ]
    for label, result, handoff_key, changed, content_in_handoff in cases:
        if not result.ok:
            raise SystemExit(f"{label} should tolerate malformed message rows: {result}")
        metadata = result.metadata
        assert_safe(metadata, f"{label} malformed message rows")
        assert_conversation_handoff(
            metadata,
            handoff_key,
            f"{label} malformed message rows",
            state_changed=True,
            changed=changed,
            content_in_handoff=content_in_handoff,
            writes=True,
        )
        if metadata.get("readable_message_rows") != len(readable_messages) or metadata.get("unreadable_message_rows") != 1:
            raise SystemExit(f"{label} missed malformed message-row counts: {metadata}")
        if "unreadable message row(s) hidden for safety" not in result.output:
            raise SystemExit(f"{label} should report hidden malformed rows: {result.output}")
        payload = result.output + json.dumps(metadata, sort_keys=True, default=str)
        if marker in payload:
            raise SystemExit(f"{label} leaked raw malformed message row text.")
        path = Path(metadata.get("path") or "")
        if not path.exists():
            raise SystemExit(f"{label} did not write expected note/skill path: {metadata}")
        written = path.read_text(encoding="utf-8")
        if "unreadable message row(s) hidden for safety" not in written:
            raise SystemExit(f"{label} saved artifact missed hidden-row notice: {written[:1000]}")
        if marker in written:
            raise SystemExit(f"{label} saved artifact leaked raw malformed message row text.")


def main() -> None:
    assert_conversation_exact_metadata_bool()
    assert_conversation_malformed_handoff_flags()
    with TemporaryDirectory(prefix="jarvis-conversation-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        runtime = JarvisRuntime(config)
        recent_conversation_routes = {
            "conversation please": ("recent_conversation", {}),
            "recent conversation please": ("recent_conversation", {}),
            "recent conversations please": ("recent_conversation", {}),
            "recent chat please": ("recent_conversation", {}),
            "recent chats please": ("recent_conversation", {}),
            "show recent chat please": ("recent_conversation", {}),
            "show recent conversation": ("recent_conversation", {}),
            "show latest conversation": ("recent_conversation", {}),
            "show current conversation": ("recent_conversation", {}),
            "show newest chat": ("recent_conversation", {}),
            "show last chat": ("recent_conversation", {}),
            "show me recent conversation": ("recent_conversation", {}),
            "show me latest chat": ("recent_conversation", {}),
            "conversation history please": ("recent_conversation", {}),
            "chat history please": ("recent_conversation", {}),
            "show conversation history": ("recent_conversation", {}),
            "show chat history": ("recent_conversation", {}),
            "what did we talk about please": ("recent_conversation", {}),
            "what did we just discuss": ("recent_conversation", {}),
            "what were we just talking about": ("recent_conversation", {}),
        }
        for command, (expected_tool, expected_args) in recent_conversation_routes.items():
            plan = runtime.planner.plan(command)
            if len(plan.actions) != 1:
                raise SystemExit(f"Expected exactly one action for recent conversation route {command!r}: {plan.actions}")
            action = plan.actions[0]
            if action.tool_name != expected_tool or action.args != expected_args:
                raise SystemExit(f"Recent conversation route misplanned {command!r}: {(action.tool_name, action.args)}")

        conversation_session_routes = {
            "sessions please": ("list_sessions", {}),
            "list sessions please": ("list_sessions", {}),
            "conversation sessions please": ("list_sessions", {}),
            "chat sessions please": ("list_sessions", {}),
            "show sessions": ("list_sessions", {}),
            "show conversation sessions": ("list_sessions", {}),
            "show chat sessions": ("list_sessions", {}),
            "show latest sessions": ("list_sessions", {}),
            "show current sessions": ("list_sessions", {}),
            "session list please": ("list_sessions", {}),
            "conversation session list": ("list_sessions", {}),
            "chat session list": ("list_sessions", {}),
        }
        for command, (expected_tool, expected_args) in conversation_session_routes.items():
            plan = runtime.planner.plan(command)
            if len(plan.actions) != 1:
                raise SystemExit(f"Expected exactly one action for conversation session route {command!r}: {plan.actions}")
            action = plan.actions[0]
            if action.tool_name != expected_tool or action.args != expected_args:
                raise SystemExit(f"Conversation session route misplanned {command!r}: {(action.tool_name, action.args)}")

        conversation_search_routes = {
            "search conversations for memory please": ("search_conversations", {"query": "memory"}),
            # Real privacy-relevant gap found live 2026-07-09: "search my
            # conversations for X" silently misrouted to a PUBLIC web_lookup
            # instead of the local, private search_conversations tool,
            # because the trigger regex required "conversations" to appear
            # immediately after the verb with no "my"/"the" article.
            "search my conversations for memory please": ("search_conversations", {"query": "memory"}),
            "search chat for memory please": ("search_conversations", {"query": "memory"}),
            "search conversations memory": ("search_conversations", {"query": "memory"}),
            "find chats about memory please": ("search_conversations", {"query": "memory"}),
            "show conversations about memory": ("search_conversations", {"query": "memory"}),
            "review chats for memory please": ("search_conversations", {"query": "memory"}),
            "conversation search memory please": ("search_conversations", {"query": "memory"}),
            "chat search memory please": ("search_conversations", {"query": "memory"}),
        }
        for command, (expected_tool, expected_args) in conversation_search_routes.items():
            plan = runtime.planner.plan(command)
            if len(plan.actions) != 1:
                raise SystemExit(f"Expected exactly one action for conversation search route {command!r}: {plan.actions}")
            action = plan.actions[0]
            if action.tool_name != expected_tool or action.args != expected_args:
                raise SystemExit(f"Conversation search route misplanned {command!r}: {(action.tool_name, action.args)}")
        # Real gap found live 2026-07-09: "show my sessions" fell through to
        # chat while bare "list sessions" worked.
        for command in ("list sessions", "show my sessions"):
            session_plan = runtime.planner.plan(command)
            if [a.tool_name for a in session_plan.actions] != ["list_sessions"]:
                raise SystemExit(f"list_sessions route missed {command!r}: {session_plan.actions}")
        memory_search_guard = runtime.planner.plan("search memory for conversations please")
        if len(memory_search_guard.actions) != 1 or memory_search_guard.actions[0].tool_name != "search_memory":
            raise SystemExit(f"Generic memory search should remain on search_memory: {memory_search_guard.actions}")
        # Real gap found live 2026-07-10, same privacy-relevant misroute class
        # as the round-21/26/27 notes/tasks/files word-order fixes: "search
        # for the/a memory about X" / "find the/a memory about X" (article
        # BEFORE the noun) fell through to the generic public web-search
        # fallback instead of search_memory.
        for command, expected_query in {
            "search for the memory about my trip": "my trip",
            "find the memory about my trip": "my trip",
            "search for a memory about my trip": "my trip",
        }.items():
            word_order_plan = runtime.planner.plan(command)
            if [(a.tool_name, a.args) for a in word_order_plan.actions] != [("search_memory", {"query": expected_query})]:
                raise SystemExit(f"memory search word-order route missed {command!r}: {word_order_plan.actions}")

        chat_diagnostic_routes = {
            "chat health please": ("chat_response_health", {}),
            "conversation health please": ("chat_response_health", {}),
            "show chat health": ("chat_response_health", {}),
            "show latest chat health": ("chat_response_health", {}),
            "show current chat response health": ("chat_response_health", {}),
            "show newest conversation health": ("chat_response_health", {}),
            "chat fallback report please": ("chat_response_health", {}),
            "show latest chat fallback report": ("chat_response_health", {}),
            "chat safety please": ("chat_safety_report", {}),
            "conversation safety please": ("chat_safety_report", {}),
            "show chat safety": ("chat_safety_report", {}),
            "show latest chat safety report": ("chat_safety_report", {}),
            "show current conversation safety": ("chat_safety_report", {}),
            "is chat grounded please": ("chat_safety_report", {}),
            "how is chat safe please": ("chat_safety_report", {}),
            "where did we leave off please": ("chat_continuity_brief", {}),
            "continue from last time": ("chat_continuity_brief", {}),
            "continue where we left off": ("chat_continuity_brief", {}),
            "what were we doing please": ("chat_continuity_brief", {}),
            "what were we working on": ("chat_continuity_brief", {}),
            "what were we building": ("chat_continuity_brief", {}),
            "what was I doing with Jarvis": ("chat_continuity_brief", {}),
            "show latest chat continuity brief": ("chat_continuity_brief", {}),
            "show current conversation catchup": ("chat_continuity_brief", {}),
        }
        for command, (expected_tool, expected_args) in chat_diagnostic_routes.items():
            plan = runtime.planner.plan(command)
            if len(plan.actions) != 1:
                raise SystemExit(f"Expected exactly one action for chat diagnostic route {command!r}: {plan.actions}")
            action = plan.actions[0]
            if action.tool_name != expected_tool or action.args != expected_args:
                raise SystemExit(f"Chat diagnostic route misplanned {command!r}: {(action.tool_name, action.args)}")

        chat_context_preview_routes = {
            "chat context please": ("chat_context", {"prompt": ""}),
            "show chat context": ("chat_context", {"prompt": ""}),
            "show current chat context": ("chat_context", {"prompt": ""}),
            "show latest conversation context": ("chat_context", {"prompt": ""}),
            "preview chat context please": ("chat_context", {"prompt": ""}),
            "chat context what should Jarvis do next": ("chat_context", {"prompt": "what should Jarvis do next"}),
            "chat context: what should Jarvis do next please": ("chat_context", {"prompt": "what should Jarvis do next"}),
            "chat prompt preview please": ("chat_prompt_preview", {"prompt": ""}),
            "show chat prompt preview": ("chat_prompt_preview", {"prompt": ""}),
            "show latest chat prompt preview": ("chat_prompt_preview", {"prompt": ""}),
            "preview conversation prompt please": ("chat_prompt_preview", {"prompt": ""}),
            "chat prompt preview how should Jarvis talk about memory": (
                "chat_prompt_preview",
                {"prompt": "how should Jarvis talk about memory"},
            ),
            "session learning preview please": ("session_learning_preview", {}),
            "latest session learning preview please": ("session_learning_preview", {}),
            "preview session learning please": ("session_learning_preview", {}),
            "show session learning preview": ("session_learning_preview", {}),
            "show latest session learning preview": ("session_learning_preview", {}),
            "what should jarvis learn from this session please": ("session_learning_preview", {}),
            "what should you learn from this session please": ("session_learning_preview", {}),
        }
        for command, (expected_tool, expected_args) in chat_context_preview_routes.items():
            plan = runtime.planner.plan(command)
            if len(plan.actions) != 1:
                raise SystemExit(f"Expected exactly one action for chat context/learning route {command!r}: {plan.actions}")
            action = plan.actions[0]
            if action.tool_name != expected_tool or action.args != expected_args:
                raise SystemExit(f"Chat context/learning route misplanned {command!r}: {(action.tool_name, action.args)}")

        cases = [
            "calculate 21 + 21",
            "recent conversation",
            "search conversations for 21",
            "list sessions",
            "summarize this session",
            "draft skill from this session called Conversation Review Draft",
            "get skill Conversation Review Draft",
            "export this session",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            if result.tool_results:
                metadata = result.tool_results[0].metadata
                assert_safe(metadata, case)
                expected_handoffs = {
                    "recent conversation": ("conversation_recent_handoff", False, [], True, False),
                    "search conversations for 21": ("conversation_search_handoff", False, [], True, False),
                    "list sessions": ("conversation_sessions_handoff", False, [], True, False),
                    "summarize this session": ("conversation_summary_handoff", True, ["session_summary"], True, True),
                    "draft skill from this session called Conversation Review Draft": ("conversation_skill_draft_handoff", True, ["skill_draft"], False, True),
                    "export this session": ("conversation_export_handoff", True, ["session_export"], False, True),
                }
                if case in expected_handoffs:
                    key, changed_state, changed, has_content, writes = expected_handoffs[case]
                    assert_conversation_handoff(
                        metadata,
                        key,
                        case,
                        state_changed=changed_state,
                        changed=changed,
                        content_in_handoff=has_content,
                        writes=writes,
                    )
                if case in {"recent conversation", "search conversations for 21", "list sessions"} and metadata.get("writes_memory"):
                    raise SystemExit(f"{case} should not write memory: {metadata}")
                if case in {"summarize this session", "draft skill from this session called Conversation Review Draft", "export this session"}:
                    if not metadata.get("writes_notes") or not metadata.get("writes_files"):
                        raise SystemExit(f"{case} should declare note/file writes: {metadata}")
                    path_text = str(metadata.get("path") or "")
                    path_display = metadata.get("path_display")
                    if not path_text or not Path(path_text).exists():
                        raise SystemExit(f"{case} missed exact saved-note path metadata: {metadata}")
                    if path_text in result.response:
                        raise SystemExit(f"{case} should not print the raw local note path.")
                    assert_no_local_path(result.response, f"{case} output")
                    if not isinstance(path_display, str) or "/" not in path_display:
                        raise SystemExit(f"{case} missed vault-relative display metadata: {metadata}")
                    expected_prefix = {
                        "summarize this session": "Reflections/",
                        "draft skill from this session called Conversation Review Draft": "Skills/",
                        "export this session": "Sessions/",
                    }[case]
                    if not path_display.startswith(expected_prefix):
                        raise SystemExit(f"{case} used an unexpected display path: {metadata}")
                    if path_display not in result.response:
                        raise SystemExit(f"{case} should print the vault-relative saved-note label.")
                if case == "draft skill from this session called Conversation Review Draft":
                    if metadata.get("human_review_required") is not True or metadata.get("self_modifying_code") is not False:
                        raise SystemExit(f"{case} missed human-review skill metadata: {metadata}")
                    skill_path = Path(metadata["path"])
                    skill_text = skill_path.read_text(encoding="utf-8")
                    required = [
                        "Review status: DRAFT",
                        "Human review is required",
                        "## Human Review Gate",
                        "## Failure And Stop Conditions",
                        "## Source Session Signals",
                        "Do not let this draft edit code",
                    ]
                    missing = [item for item in required if item not in skill_text]
                    if missing:
                        raise SystemExit(f"{case} saved skill missed review contract text: {missing}")

        long_result = runtime.handle("search conversations for " + ("jarvis " * 400))
        if not long_result.verified:
            raise SystemExit("Long conversation search should run safely.")
        metadata = long_result.tool_results[0].metadata
        assert_safe(metadata, "long conversation search")
        assert_conversation_handoff(metadata, "conversation_search_handoff", "long conversation search")
        if len(metadata.get("query", "")) > 620:
            raise SystemExit("Conversation search query was not bounded.")

        bool_recent = runtime.registry.get("recent_conversation").handler({"limit": False})
        if not bool_recent.ok or bool_recent.metadata.get("count", 0) <= 1:
            raise SystemExit(f"recent_conversation should treat boolean limits as malformed defaults, not one-row views: {bool_recent.metadata}")
        assert_safe(bool_recent.metadata, "recent conversation boolean limit")
        assert_conversation_handoff(bool_recent.metadata, "conversation_recent_handoff", "recent conversation boolean limit", content_in_handoff=True)

        raw_path_message = (
            "Please summarize /\x55sers/example/Desktop/Claude code/private-session-plan.md, "
            "/private/tmp/session-secret.txt, /var/folders/zc/session-cache.txt, and /tmp/session-direct.txt"
        )
        runtime.store.log_message(runtime.session_id, "user", raw_path_message)
        runtime.store.log_message(runtime.session_id, "assistant", "I saw /var/folders/zc/assistant-cache.txt and /tmp/assistant-direct.txt in old context.")

        recent_path = runtime.registry.get("recent_conversation").handler({"limit": 5})
        if "<local-path>" not in recent_path.output:
            raise SystemExit(f"recent_conversation should redact local paths: {recent_path.output}")
        assert_no_local_path(recent_path.output, "recent_conversation path output")
        assert_conversation_handoff(recent_path.metadata, "conversation_recent_handoff", "recent_conversation path output", content_in_handoff=True)

        search_path = runtime.registry.get("search_conversations").handler({"query": "session"})
        if "<local-path>" not in search_path.output:
            raise SystemExit(f"search_conversations should redact local paths: {search_path.output}")
        assert_no_local_path(search_path.output, "search_conversations path output")
        assert_conversation_handoff(search_path.metadata, "conversation_search_handoff", "search_conversations path output", content_in_handoff=True)

        assert_chat_response_health_reports_ws4_measurement(runtime)
        assert_conversation_retrieval_tolerates_malformed_message_rows(runtime)
        assert_chat_diagnostics_tolerate_malformed_message_rows(runtime)
        assert_chat_safety_and_learning_tolerate_malformed_message_rows(runtime)
        assert_chat_context_previews_tolerate_malformed_message_rows(runtime)
        assert_conversation_session_listing_tolerates_malformed_session_rows(runtime)
        assert_conversation_write_tools_tolerate_malformed_message_rows(runtime)

        chat_context_path = runtime.registry.get("chat_context").handler({"prompt": "/var/folders/zc/chat-context-prompt", "limit": 8})
        if "<local-path>" not in chat_context_path.output:
            raise SystemExit(f"chat_context should redact path-shaped prompts and recent messages: {chat_context_path.output}")
        assert_no_local_path(chat_context_path.output, "chat_context path output")
        assert_safe(chat_context_path.metadata, "chat_context path preview")

        summary_path = runtime.registry.get("summarize_session").handler({"limit": 20})
        if summary_path.metadata["path"] in summary_path.output:
            raise SystemExit("summarize_session should not print the raw local note path.")
        if "<local-path>" not in summary_path.output:
            raise SystemExit("summarize_session should redact local paths in output.")
        assert_no_local_path(summary_path.output, "summarize_session path output")
        assert_conversation_handoff(
            summary_path.metadata,
            "conversation_summary_handoff",
            "summarize_session path output",
            state_changed=True,
            changed=["session_summary"],
            content_in_handoff=True,
            writes=True,
        )
        summary_note = Path(summary_path.metadata["path"]).read_text(encoding="utf-8")
        if "<local-path>" not in summary_note:
            raise SystemExit("summarize_session should redact local paths in saved reflection.")
        assert_no_local_path(summary_note, "summarize_session saved reflection")

        export_path = runtime.registry.get("export_session").handler({"limit": 20})
        if export_path.metadata["path"] in export_path.output:
            raise SystemExit("export_session should not print the raw local note path.")
        assert_no_local_path(export_path.output, "export_session output")
        assert_conversation_handoff(
            export_path.metadata,
            "conversation_export_handoff",
            "export_session output",
            state_changed=True,
            changed=["session_export"],
            writes=True,
        )
        export_note = Path(export_path.metadata["path"]).read_text(encoding="utf-8")
        if "<local-path>" not in export_note:
            raise SystemExit("export_session should redact local paths in saved transcript.")
        assert_no_local_path(export_note, "export_session saved transcript")

        draft_path = runtime.registry.get("draft_skill_from_session").handler({"name": "Path Redaction Draft", "limit": 20})
        if draft_path.metadata["path"] in draft_path.output:
            raise SystemExit("draft_skill_from_session should not print the raw local note path.")
        assert_no_local_path(draft_path.output, "draft_skill_from_session output")
        assert_conversation_handoff(
            draft_path.metadata,
            "conversation_skill_draft_handoff",
            "draft_skill_from_session output",
            state_changed=True,
            changed=["skill_draft"],
            writes=True,
        )
        draft_note = Path(draft_path.metadata["path"]).read_text(encoding="utf-8")
        if "<local-path>" not in draft_note:
            raise SystemExit("draft_skill_from_session should redact local paths in saved skill draft.")
        assert_no_local_path(draft_note, "draft_skill_from_session saved draft")


if __name__ == "__main__":
    main()
