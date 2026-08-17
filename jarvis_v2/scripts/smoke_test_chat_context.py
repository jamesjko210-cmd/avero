from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.conversation import make_conversation_tools


class _HostileValue:
    def __bool__(self) -> bool:
        raise RuntimeError("bool trap /\x55sers/example/private SHOULD NOT APPEAR")

    def __str__(self) -> str:
        raise RuntimeError("str trap /\x55sers/example/private SHOULD NOT APPEAR")


class _HostileRow:
    def __getitem__(self, key: str) -> _HostileValue:
        return _HostileValue()

    def get(self, key: str, default: object = None) -> _HostileValue:
        return _HostileValue()


def assert_contains(text: str, expected: list[str], label: str) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"{label} missing expected text: {missing}")


def assert_no_hostile_leak(text: str, label: str) -> None:
    for forbidden in ["/\x55sers/operator", "SHOULD NOT APPEAR", "bool trap", "str trap"]:
        if forbidden in text:
            raise SystemExit(f"{label} leaked hostile row text {forbidden!r}: {text}")


def _chat_context_path(runtime) -> Path:
    return runtime.vault.root_path / "Reflections" / f"{time.strftime('%Y-%m-%d')} Chat Context.md"


def _save_chat_context(runtime, *, prompt: str = "projection ownership", limit: int = 5):
    return runtime.registry.get("save_chat_context").handler({"prompt": prompt, "limit": limit})


def _expect_projection_refusal(call, message: str) -> None:
    try:
        call()
    except (FileExistsError, OSError, RuntimeError, ValueError):
        return
    raise SystemExit(message)


def test_chat_context_projection_ownership_and_legacy_migration() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-context-unowned-") as temp:
        runtime = make_temp_runtime(Path(temp))
        path = _chat_context_path(runtime)
        unowned = b"# Chat Context\n\nThis is my personal note.\n"
        path.write_bytes(unowned)
        _expect_projection_refusal(
            lambda: _save_chat_context(runtime),
            "save chat context overwrote an unowned note",
        )
        if path.read_bytes() != unowned:
            raise SystemExit("unowned chat-context note changed after refusal")

    with TemporaryDirectory(prefix="jarvis-chat-context-owned-") as temp:
        runtime = make_temp_runtime(Path(temp))
        first = _save_chat_context(runtime, prompt="first owned snapshot")
        path = _chat_context_path(runtime)
        owner = runtime.store.get_store_identity()
        if not first.ok or not path.read_text(encoding="utf-8").startswith(
            f"---\njarvis_projection: chat_context\nstore_identity: {owner}\n---\n\n# Chat Context\n"
        ):
            raise SystemExit("fresh chat-context projection missed ownership metadata")
        second = _save_chat_context(runtime, prompt="second owned snapshot")
        text = path.read_text(encoding="utf-8")
        if not second.ok or "second owned snapshot" not in text or "first owned snapshot" in text:
            raise SystemExit("same owner could not rewrite chat-context projection")

    with TemporaryDirectory(prefix="jarvis-chat-context-legacy-") as temp:
        runtime = make_temp_runtime(Path(temp))
        preview = runtime.registry.get("chat_context").handler({"prompt": "exact legacy", "limit": 5})
        path = _chat_context_path(runtime)
        legacy = f"# Chat Context\n\n{preview.output.rstrip()}\n"
        path.write_text(legacy, encoding="utf-8")
        migrated = _save_chat_context(runtime, prompt="exact legacy")
        if not migrated.ok or not path.read_text(encoding="utf-8").startswith(
            "---\njarvis_projection: chat_context\n"
        ):
            raise SystemExit("exact historical chat-context projection did not migrate")

    with TemporaryDirectory(prefix="jarvis-chat-context-legacy-near-") as temp:
        runtime = make_temp_runtime(Path(temp))
        preview = runtime.registry.get("chat_context").handler({"prompt": "exact legacy", "limit": 5})
        path = _chat_context_path(runtime)
        near_match = f"# Chat Context\n\n{preview.output.rstrip()}\n\nPersonal addition.\n"
        path.write_text(near_match, encoding="utf-8")
        before = path.read_bytes()
        _expect_projection_refusal(
            lambda: _save_chat_context(runtime, prompt="exact legacy"),
            "near-match legacy chat context was accepted",
        )
        if path.read_bytes() != before:
            raise SystemExit("near-match legacy refusal changed the note")


def test_chat_context_projection_foreign_store_and_atomicity() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-context-foreign-") as temp:
        root = Path(temp)
        owner = make_temp_runtime(root)
        _save_chat_context(owner, prompt="owner snapshot")
        path = _chat_context_path(owner)
        before = path.read_bytes()
        foreign_root = root / "foreign"
        foreign = make_temp_runtime(foreign_root)
        foreign.vault = type(owner.vault)(owner.vault.vault_path, root=owner.vault.root)
        foreign_save = next(
            handler
            for handler in make_conversation_tools(foreign.store, foreign.vault, "smoke-session")
            if handler.__name__ == "save_chat_context"
        )
        if foreign.store.get_store_identity() == owner.store.get_store_identity():
            raise SystemExit("foreign chat-context fixture reused owner identity")
        _expect_projection_refusal(
            lambda: foreign_save({"prompt": "foreign snapshot", "limit": 5}),
            "foreign store overwrote chat-context projection",
        )
        if path.read_bytes() != before:
            raise SystemExit("foreign-store refusal changed chat-context projection")

        ambiguous = path.read_text(encoding="utf-8").replace(
            "store_identity:",
            "store_identity: ffffffffffffffffffffffffffffffff\nstore_identity:",
            1,
        )
        path.write_text(ambiguous, encoding="utf-8")
        _expect_projection_refusal(
            lambda: _save_chat_context(owner, prompt="ambiguous ownership"),
            "duplicate chat-context ownership markers were accepted",
        )
        if path.read_text(encoding="utf-8") != ambiguous:
            raise SystemExit("ambiguous chat-context ownership refusal changed the note")

    from jarvis_v2.memory import obsidian as obsidian_module

    with TemporaryDirectory(prefix="jarvis-chat-context-atomic-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _save_chat_context(runtime, prompt="preserved snapshot")
        path = _chat_context_path(runtime)
        before = path.read_bytes()
        original_replace = obsidian_module._replace_text

        def fail_target(root, target, content, **kwargs):
            if target == path:
                raise OSError("simulated chat-context replacement failure")
            return original_replace(root, target, content, **kwargs)

        with patch("jarvis_v2.memory.obsidian._replace_text", side_effect=fail_target):
            _expect_projection_refusal(
                lambda: _save_chat_context(runtime, prompt="failed snapshot"),
                "injected chat-context replacement failure was hidden",
            )
        if path.read_bytes() != before:
            raise SystemExit("failed chat-context replacement changed prior bytes")

        foreign_edit = "# Personal edit made outside Jarvis during publication.\n"

        def race_target(root, target, content, **kwargs):
            if target == path:
                path.write_text(foreign_edit, encoding="utf-8")
            return original_replace(root, target, content, **kwargs)

        with patch("jarvis_v2.memory.obsidian._replace_text", side_effect=race_target):
            _expect_projection_refusal(
                lambda: _save_chat_context(runtime, prompt="raced snapshot"),
                "chat-context ownership race was not refused",
            )
        if path.read_text(encoding="utf-8") != foreign_edit:
            raise SystemExit("chat-context CAS refusal did not preserve the outside edit")

        path.write_bytes(before)

        _save_chat_context(runtime, prompt="concurrent one")
        expected_one = path.read_bytes()
        _save_chat_context(runtime, prompt="concurrent two")
        expected_two = path.read_bytes()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [
                future.result(timeout=10)
                for future in (
                    pool.submit(_save_chat_context, runtime, prompt="concurrent one"),
                    pool.submit(_save_chat_context, runtime, prompt="concurrent two"),
                )
            ]
        if any(not result.ok for result in results):
            raise SystemExit(f"concurrent chat-context publication failed: {results}")
        if path.read_bytes() not in (expected_one, expected_two):
            raise SystemExit("concurrent writers left a mixed chat-context projection")

    with TemporaryDirectory(prefix="jarvis-chat-context-large-owned-") as temp:
        runtime = make_temp_runtime(Path(temp))
        owner = runtime.store.get_store_identity()
        first_body = "# Chat Context\n\n" + ("a" * 17_000)
        second_body = "# Chat Context\n\n" + ("b" * 17_000)
        path, _, _ = runtime.vault.write_chat_context_with_evidence(
            first_body,
            store_identity=owner,
            source_payload={"version": 1},
        )
        runtime.vault.write_chat_context_with_evidence(
            second_body,
            store_identity=owner,
            source_payload={"version": 2},
        )
        large_text = path.read_text(encoding="utf-8")
        if second_body not in large_text or first_body in large_text:
            raise SystemExit("large owned chat-context projection could not be rewritten by CAS")


def test_chat_context_source_snapshot_stability() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-context-fence-") as temp:
        runtime = make_temp_runtime(Path(temp))
        entered = Event()
        release = Event()
        original_write = runtime.vault.write_chat_context_with_evidence

        def blocking_write(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("chat-context publication release timed out")
            return original_write(*args, **kwargs)

        runtime.vault.write_chat_context_with_evidence = blocking_write
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                publishing = pool.submit(_save_chat_context, runtime, prompt="fenced source")
                if not entered.wait(timeout=5):
                    raise SystemExit("chat-context save did not reach publication")
                mutating = pool.submit(
                    runtime.store.add_memory,
                    MemoryRecord("context", "Fenced source", "fenced source mutation", "smoke"),
                )
                time.sleep(0.1)
                if mutating.done():
                    raise SystemExit("chat-context publication fence allowed a source commit early")
                release.set()
                published = publishing.result(timeout=5)
                mutating.result(timeout=5)
        finally:
            runtime.vault.write_chat_context_with_evidence = original_write
            release.set()

        if published.metadata.get("memories") != 0:
            raise SystemExit("chat-context snapshot included a mutation committed after capture")
        first = _chat_context_path(runtime).read_bytes()
        refreshed = _save_chat_context(runtime, prompt="fenced source")
        if refreshed.metadata.get("memories") != 1:
            raise SystemExit("refreshed chat-context snapshot missed the later source mutation")
        if _chat_context_path(runtime).read_bytes() == first:
            raise SystemExit("refreshed chat-context projection did not publish the newer snapshot")


def test_chat_context_tolerates_hostile_context_rows() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-context-hostile-") as temp:
        runtime = make_temp_runtime(Path(temp))
        hostile_rows = [_HostileRow()]
        runtime.store.list_preferences = lambda status="active", limit=20: hostile_rows[:]
        runtime.store.search_memories = lambda query, limit=5: hostile_rows[:]
        runtime.store.recent_memories = lambda limit=5: hostile_rows[:]
        runtime.store.search_skills = lambda query, limit=3: hostile_rows[:]
        runtime.store.list_skills = lambda limit=8: hostile_rows[:]
        runtime.store.recent_messages = lambda limit=8, session_id=None: hostile_rows[:]

        cases = [
            "chat context: how should Jarvis talk about memory?",
            "chat prompt preview: how should Jarvis talk about memory?",
            "chat safety",
            "chat loop preview: how should Jarvis talk about memory?",
        ]
        for case in cases:
            result = runtime.handle(case)
            if not result.verified:
                raise SystemExit(f"{case} should survive hostile context rows: {result.response}")
            combined = f"{result.response}\n{result.tool_results[0].metadata}"
            assert_no_hostile_leak(combined, case)
            if case in {"chat safety", "chat loop preview: how should Jarvis talk about memory?"}:
                if "unreadable message row(s) hidden for safety: 1" not in combined:
                    if case == "chat safety":
                        raise SystemExit(f"{case} should report hidden unreadable message rows: {combined}")
                if case.startswith("chat loop preview") and "active preferences: 1" not in combined:
                    raise SystemExit(f"{case} should keep hostile rows bounded to counts: {combined}")
            elif "<unreadable>" not in combined:
                raise SystemExit(f"{case} should collapse hostile row fields to <unreadable>: {combined}")
            if result.tool_results[0].metadata.get("calls_model") or result.tool_results[0].metadata.get("executes_tools") or result.tool_results[0].metadata.get("queues_approval"):
                raise SystemExit(f"{case} should remain read-only under hostile context rows: {result.tool_results[0].metadata}")


def main() -> None:
    test_chat_context_projection_ownership_and_legacy_migration()
    test_chat_context_projection_foreign_store_and_atomicity()
    test_chat_context_source_snapshot_stability()
    test_chat_context_tolerates_hostile_context_rows()
    with TemporaryDirectory(prefix="jarvis-chat-context-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            "add profile note the operator wants Jarvis chat to feel natural and safe",
            "set preference response style to direct and warm category communication",
            "remember that conversational Jarvis should cite useful memory",
            "save skill Memory-Aware Chat when memory chat do Check memory first, then answer naturally, then mention safety boundaries.",
            "Hey Jarvis, talk normally about memory",
            "Can we just talk normally about Jarvis?",
            "chat response health",
            "chat continuity brief",
            "chat safety",
            "chat loop preview: use my computer to organize files",
            "chat loop preview: how should Jarvis talk about memory?",
            "chat context: how should Jarvis talk about memory?",
            "chat prompt preview: how should Jarvis talk about memory?",
            "save chat context: how should Jarvis talk about memory?",
            "session learning preview",
            "help conversation",
            "list tools conversation",
        ]
        for case in cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            if case == "Hey Jarvis, talk normally about memory":
                assert_contains(
                    result.response,
                    [
                        "stay grounded",
                        "Profile context",
                        "natural and safe",
                        "Active preferences",
                        "Relevant memory",
                        "conversational Jarvis should cite useful memory",
                        "Relevant saved skills",
                        "Memory-Aware Chat",
                        "Safety boundary",
                    ],
                    case,
                )
                invented = ["sustainable energy", "advanced materials", "carbon footprint"]
                if any(item in result.response.lower() for item in invented):
                    raise SystemExit("Grounded memory chat invented unsupported personal/project context.")
            if case == "chat safety":
                assert_contains(
                    result.response,
                    [
                        "Jarvis chat safety report",
                        "Grounding rules",
                        "deterministic grounded reply",
                        "If context is thin",
                        "Execution boundary",
                        "approval queue",
                        "Visible context counts",
                        "chat context:",
                        "privacy report",
                    ],
                    case,
                )
            if case == "chat response health":
                assert_contains(
                    result.response,
                    [
                        "Jarvis chat response health",
                        "chat-route responses",
                        "ordinary chat responses",
                        "source counts",
                        "fallback responses",
                        "external model attempts",
                        "latency samples",
                        "WS4 measurement readiness",
                        "Latest chat response",
                        "model provider:",
                        "external model call:",
                        "conversation shared with external model:",
                        "prompt/response content copied into health metadata: no",
                        "does not call models",
                        "does not call models, execute tools",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("chat_responses", 0) < 1:
                    raise SystemExit("chat response health missed previous chat-route response.")
                if metadata.get("ordinary_chat_responses", 0) < 1:
                    raise SystemExit("chat response health missed previous ordinary chat response.")
                if metadata.get("calls_model") or metadata.get("executes_tools") or metadata.get("queues_approval"):
                    raise SystemExit("chat response health should stay read-only.")
                if metadata.get("model_request_content_in_health_metadata") is not False or metadata.get("model_response_content_in_health_metadata") is not False:
                    raise SystemExit("chat response health must not copy model content into its metadata.")
            if case == "chat continuity brief":
                assert_contains(
                    result.response,
                    [
                        "Jarvis chat continuity brief",
                        "read-only",
                        "without calling models",
                        "Recent user asks",
                        "Recent Jarvis responses",
                        "Likely continuation thread",
                        "Latest response path",
                        "Safe next checks",
                        "chat response health",
                        "session learning preview",
                        "approval-gated",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("messages", 0) < 4:
                    raise SystemExit("chat continuity brief missed recent messages.")
                if metadata.get("calls_model") or metadata.get("executes_tools") or metadata.get("queues_approval") or metadata.get("controls_computer"):
                    raise SystemExit("chat continuity brief should stay read-only.")
            if case.startswith("chat loop preview: use my computer"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis chat loop preview",
                        "Perceive text input",
                        "Classify the turn",
                        "turn type: action_or_safety",
                        "reply path: safety_preflight_guidance",
                        "computer-control",
                        "files",
                        "execution governor",
                        "does not call the model",
                        "does not execute tools",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("calls_model") or metadata.get("executes_tools") or metadata.get("queues_approval"):
                    raise SystemExit("Chat loop preview should not call the model, execute tools, or queue approvals.")
                if not str(metadata.get("next_command", "")).startswith("execution governor: "):
                    raise SystemExit("Chat loop preview missed execution-governor next command metadata.")
                if not metadata.get("recommended_next_commands") or not str(metadata["recommended_next_commands"][0]).startswith("execution governor: "):
                    raise SystemExit("Chat loop preview missed execution-governor recommended command metadata.")
            if case.startswith("chat loop preview: how should"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis chat loop preview",
                        "turn type: memory_grounded",
                        "reply path: grounded_memory",
                        "Grounding counts",
                        "does not call the model",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("next_command") or metadata.get("recommended_next_commands"):
                    raise SystemExit("Memory chat loop preview should not expose action next-command metadata.")
            if case.startswith("chat context"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis chat context preview",
                        "read-only",
                        "Prompt",
                        "Profile context",
                        "natural and safe",
                        "Active preferences",
                        "response style",
                        "Relevant memories",
                        "conversational Jarvis should cite useful memory",
                        "Relevant skills",
                        "Memory-Aware Chat",
                        "Recent conversation window",
                        "Safety reminder",
                    ],
                    case,
                )
            if case.startswith("chat prompt preview"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis chat prompt preview",
                        "read-only",
                        "does not call the configured model",
                        "User message",
                        "System prompt preview",
                        "You are J.A.R.V.I.S.",
                        "conversational like ChatGPT or Claude",
                        "Grounding packet preview",
                        "Jarvis chat context preview",
                        "Reply path",
                        "deterministic grounded-memory reply",
                        "Execution boundary",
                        "approval queue",
                    ],
                    case,
                )
            if case == "session learning preview":
                assert_contains(
                    result.response,
                    [
                        "Jarvis session learning preview",
                        "read-only",
                        "does not save memory",
                        "Candidate memories",
                        "conversational Jarvis should cite useful memory",
                        "Candidate preferences",
                        "direct and warm",
                        "Candidate tasks",
                        "Candidate skill/workflow signals",
                        "Safe follow-up commands",
                        "draft skill from this session",
                        "approval-gated",
                    ],
                    case,
                )
            if case.startswith("save chat context"):
                assert_contains(
                    result.response,
                    [
                        "Chat context saved",
                        "Jarvis chat context preview",
                        "Profile context",
                        "Relevant memories",
                        "Relevant skills",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                path_text = str(metadata.get("path") or "")
                path_display = metadata.get("path_display")
                if not path_text:
                    raise SystemExit(f"{case} missed exact saved-note path metadata: {metadata}")
                if path_text in result.response:
                    raise SystemExit(f"{case} should not print the raw local note path.")
                if str(Path(temp)) in result.response or "/private/" in result.response or "/\x55sers/" in result.response:
                    raise SystemExit(f"{case} output should not expose local temp or user paths.")
                if not isinstance(path_display, str) or not path_display.startswith("Reflections/"):
                    raise SystemExit(f"{case} missed vault-relative display metadata: {metadata}")
                if f"Chat context saved: {path_display}" not in result.response:
                    raise SystemExit(f"{case} should print a vault-relative saved-note label.")
                note = Path(path_text)
                if not note.exists():
                    raise SystemExit(f"{case} did not write expected note: {note}")
                assert_contains(
                    note.read_text(encoding="utf-8"),
                    ["# Chat Context", "Jarvis chat context preview", "natural and safe", "Memory-Aware Chat"],
                    f"{case} note",
                )
                if not metadata.get("writes_notes") or not metadata.get("writes_files") or not metadata.get("writes_memory"):
                    raise SystemExit("save chat context should declare note/file/memory writes.")
                handoff = metadata.get("save_chat_context_handoff") or {}
                if handoff.get("path_display") != path_display:
                    raise SystemExit(f"save chat context handoff missed vault-relative path: {handoff}")
                if "/private/" in str(handoff.get("path_display")) or "/\x55sers/" in str(handoff.get("path_display")):
                    raise SystemExit(f"save chat context handoff should not expose raw local paths: {handoff}")
                if handoff.get("active_preference_count") != metadata.get("preferences"):
                    raise SystemExit(f"save chat context handoff missed preference count: {handoff}")
                if handoff.get("relevant_memory_count") != metadata.get("memories"):
                    raise SystemExit(f"save chat context handoff missed memory count: {handoff}")
                if handoff.get("recent_message_count") != metadata.get("recent_messages"):
                    raise SystemExit(f"save chat context handoff missed recent message count: {handoff}")
                expected_handoff = {
                    "command": "save chat context",
                    "writes_notes": True,
                    "writes_files": True,
                    "writes_memory": True,
                    "calls_model": False,
                    "executes_tools": False,
                    "queues_approval": False,
                    "controls_computer": False,
                }
                for key, expected in expected_handoff.items():
                    if handoff.get(key) != expected:
                        raise SystemExit(f"save chat context handoff missed {key}: {handoff}")
            if case == "help conversation":
                assert_contains(result.response, ["chat loop preview:", "chat continuity brief", "chat response health", "chat safety", "chat context:", "chat prompt preview:", "session learning preview", "save chat context:"], case)
            if case == "list tools conversation":
                assert_contains(result.response, ["chat_context", "chat_loop_preview", "chat_prompt_preview", "chat_continuity_brief", "chat_response_health", "chat_safety_report", "save_chat_context"], case)

        long_case = "chat context: " + ("Jarvis should keep the context bounded and safe. " * 80)
        result = runtime.handle(long_case)
        if not result.verified:
            raise SystemExit("Long chat context should run.")
        metadata = result.tool_results[0].metadata
        if metadata.get("calls_model") or metadata.get("executes_tools") or metadata.get("queues_approval") or metadata.get("controls_computer"):
            raise SystemExit("Long chat context should stay read-only.")
        if "Jarvis chat context preview" not in result.response:
            raise SystemExit("Long chat context did not return the expected preview.")


if __name__ == "__main__":
    main()
