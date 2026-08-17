from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from unittest.mock import patch

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
)
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.notes import _metadata_bool, _note_handoff_metadata


def assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def assert_local_productivity_read_recovery(result, label: str) -> None:
    action = LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid the canonical local recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} local recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} local recovery field {key} drifted: {result.metadata}")


def assert_resource_not_found_recovery(result, label: str) -> None:
    action = RESOURCE_NOT_FOUND_RECOVERY_ACTION
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid canonical missing-resource recovery: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": [],
    }:
        raise SystemExit(f"{label} missing-resource declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} missing-resource field {key} drifted: {result.metadata}")


def assert_note_exact_metadata_bool() -> None:
    for value in ("true", "false", "yes", "0", 1, 0, [], ["value"], None, object()):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"note metadata bool should reject malformed value {value!r}")
    if _metadata_bool(True) is not True:
        raise SystemExit("note metadata bool should preserve exact True")
    if _metadata_bool(False) is not False:
        raise SystemExit("note metadata bool should preserve exact False")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("note metadata bool should honor explicit default for malformed values")


def assert_note_malformed_handoff_flags() -> None:
    handoff = {
        "source": "read_jarvis_note",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["list jarvis notes"],
        "boundaries": {"read_only": True},
    }
    metadata = _note_handoff_metadata("note_read_handoff", handoff)
    if metadata.get("state_changed") is not False or metadata.get("note_read_state_changed") is not False:
        raise SystemExit(f"malformed note state_changed should fail closed: {metadata}")
    if metadata.get("content_in_handoff") is not False or metadata.get("note_read_content_in_handoff") is not False:
        raise SystemExit(f"malformed note content_in_handoff should fail closed: {metadata}")


def assert_planner_routes_note_phone_aliases() -> None:
    planner = RuleBasedPlanner()
    for text in (
        "notes please",
        "list notes please",
        "show notes",
        "show latest notes",
        "what notes do i have",
        # Review find 2026-07-09: bare note-SEARCH phrases (no query) previously
        # fell through the query-requiring search regex all the way to the
        # generic web-search pattern -- "search my notes" hit web_lookup with
        # query "my notes" (public web). They must resolve locally, as a listing.
        "search notes",
        "search my notes",
        "find notes",
        "find my notes",
        "list my notes",
        # Real gap found live 2026-07-10, same class as the round-39 "list my
        # open tasks" bug, surfaced via a WS4 mixed-conversation
        # remeasurement: "show latest notes" worked bare, but every
        # my-prefixed combination of a "recent"/"latest" qualifier fell
        # through to chat.
        "recent notes",
        "latest notes",
        "show my recent notes",
        "list my recent notes",
        "my recent notes",
        "show my latest notes",
        "list my latest notes",
        "my latest notes",
        # Real gap found live 2026-07-10 (round 43): trailing "please" broke
        # this whole exact-match set, because the global politeness-retry
        # mechanism's leading-strip side effect corrupts verb-only entries
        # ("show jarvis notes" -> "jarvis notes", not in the set), so
        # list_jarvis_notes was deliberately never added to
        # POLITE_COMMAND_RETRY_TOOLS -- meaning ANY successful retry match got
        # silently discarded. Fixed with a trailing-only strip local to this
        # block instead (safe: a no-op when there's nothing to strip, so it
        # can't break the leading-strip-sensitive entries above).
        "show my recent notes please",
    ):
        plan = planner.plan(text)
        if [(action.tool_name, action.args) for action in plan.actions] != [("list_jarvis_notes", {})]:
            raise SystemExit(f"planner missed note list alias {text!r}: {plan.actions}")
    for text, expected_query in {
        "search notes for Jarvis please": "Jarvis",
        "find notes about approval": "approval",
        "show notes about memory please": "memory",
        # Real bug found live 2026-07-08: "search MY notes for X" (arguably more
        # natural phrasing than "search notes for X") fell through to web_lookup
        # because the regex required "search notes"/"search jarvis notes" with no
        # "my" gap -- Jarvis searched the open web for a note-search request.
        "search my notes for Jarvis": "Jarvis",
        "find my notes about roadmap": "roadmap",
        "show my notes for meeting": "meeting",
        # Real gap found live 2026-07-10, a genuine misroute rather than a
        # refusal: "find my note about meeting" (singular "note") was not
        # recognized by this regex (only plural "notes"/"obsidian"), so it
        # fell all the way through to the generic calendar-events catch-all
        # (which fires on the bare word "meeting" anywhere in the text) and
        # returned a completely unrelated `list_events` result for a private
        # notes-search request.
        "find my note about meeting": "meeting",
        "search my note for meeting": "meeting",
        # Real gap found live 2026-07-10, same privacy-relevant misroute class
        # as the "find my note about meeting" fix above but a different word
        # order: "search for the/a note about X" (verb + preposition/article
        # BEFORE the noun) fell through to a public web_lookup instead of the
        # local notes search.
        "search for the note about the 401k meeting": "the 401k meeting",
        "find the note about vacation": "vacation",
        "search for my note about vacation": "vacation",
        "find a note about vacation": "vacation",
        # Real gap found live 2026-07-10: "search my notes for the budget and
        # then show my tasks" (a compound sentence) swallowed the whole
        # second clause into the search query instead of stopping at "the
        # budget" -- and silently dropped the "show my tasks" intent.
        "search my notes for the budget and then show my tasks": "the budget",
    }.items():
        plan = planner.plan(text)
        if [(action.tool_name, action.args) for action in plan.actions] != [("search_jarvis_notes", {"query": expected_query})]:
            raise SystemExit(f"planner missed note search alias {text!r}: {plan.actions}")


def assert_runtime_routes_note_count_aliases_without_memory_write() -> None:
    with TemporaryDirectory(prefix="jarvis-note-count-alias-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for text in (
            "how many notes do i have",
            "how many jarvis notes do i have",
            "count notes",
            "note count",
            "notes count",
            "노트 몇 개",
            "노트 개수",
            "자비스 노트 몇 개",
        ):
            result = runtime.handle(text)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [action.get("tool_name") for action in planned] != ["list_jarvis_notes"]:
                raise SystemExit(f"runtime missed note-count alias {text!r}: {planned} / {result.response!r}")
            if len(result.tool_results) != 1 or result.tool_results[0].tool_name != "list_jarvis_notes":
                raise SystemExit(f"note-count alias should execute exactly one note-list tool for {text!r}: {result.tool_results}")
            metadata = result.tool_results[0].metadata
            if "Count:" not in result.response or metadata.get("count") is None or metadata.get("total") is None:
                raise SystemExit(f"note-count alias should answer with note counts for {text!r}: {result.response!r} / {metadata}")
            for key in (
                "writes_files",
                "writes_memory",
                "writes_notes",
                "queues_approval",
                "controls_computer",
                "reads_private_data",
                "authorizes_execution",
                "authorizes_completion_claim",
                "approval_granted",
            ):
                if metadata.get(key):
                    raise SystemExit(f"note-count alias unexpectedly set {key} for {text!r}: {metadata}")
        if runtime.store.list_memories(limit=100):
            raise SystemExit("note count aliases must not create memories through the note-capture route.")


def assert_vault_relative_note_receipt(result, *, expected_display: str, label: str) -> None:
    metadata = result.tool_results[0].metadata
    receipt_line = result.response.split("\n", 1)[0]
    path_text = str(metadata.get("path") or "")
    if not path_text or not Path(path_text).exists():
        raise SystemExit(f"{label} should preserve exact saved path metadata: {metadata}")
    if metadata.get("path_display") != expected_display:
        raise SystemExit(f"{label} should expose vault-relative path_display={expected_display!r}: {metadata}")
    if expected_display not in receipt_line:
        raise SystemExit(f"{label} receipt missed vault-relative note label: {receipt_line!r}")
    if path_text in receipt_line:
        raise SystemExit(f"{label} receipt leaked a local path: {receipt_line!r}")
    assert_no_local_path(receipt_line, f"{label} receipt")


def assert_note_boundaries(handoff: dict, *, writes: bool, reads_contents: bool, label: str) -> None:
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("read_only") is not (not writes):
        raise SystemExit(f"{label} read_only boundary mismatch: {handoff}")
    if boundaries.get("reads_note_contents") is not reads_contents:
        raise SystemExit(f"{label} reads_note_contents boundary mismatch: {handoff}")
    if boundaries.get("writes_files") is not writes or boundaries.get("writes_notes") is not writes:
        raise SystemExit(f"{label} write boundary mismatch: {handoff}")
    for key in [
        "writes_memory",
        "queues_approval",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if boundaries.get(key):
            raise SystemExit(f"{label} should keep {key}=False: {handoff}")


def assert_note_contract(metadata: dict, handoff: dict, *, label: str, state_changed: bool, changed: list[str], content_in_handoff: bool) -> None:
    handoff_key = next((key for key, value in metadata.items() if key.endswith("_handoff") and value is handoff), "")
    if not handoff_key:
        raise SystemExit(f"{label} could not discover note handoff key: {metadata}")
    prefix = handoff_key.removesuffix("_handoff")
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        expected_next = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        expected_next = [str(value) for value in raw_next if str(value or "").strip()]
    expected_first = expected_next[0] if expected_next else ""
    for key, value in [
        ("handoff_ready", True),
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ]:
        if handoff.get(key) != value:
            raise SystemExit(f"{label} note handoff {key} failed: {handoff}")
        if key != "handoff_ready" and metadata.get(key) != value:
            raise SystemExit(f"{label} note handoff {key} parity failed: {handoff} / {metadata}")
    for key, value in [
        (f"{handoff_key}_ready", True),
        (f"{prefix}_handoff_ready", True),
        (f"{prefix}_ready_for_operator", True),
        (f"{prefix}_state_changed", state_changed),
        (f"{prefix}_changed", changed),
        (f"{prefix}_content_in_handoff", content_in_handoff),
        (f"{prefix}_authorizes_execution", False),
        (f"{prefix}_authorizes_completion_claim", False),
        (f"{prefix}_approval_granted", False),
    ]:
        if metadata.get(key) != value:
            raise SystemExit(f"{label} note prefixed alias {key} failed: {metadata}")
    for container, container_label in [(handoff, "handoff"), (metadata, "metadata")]:
        if container.get("next_safe_command") != expected_first:
            raise SystemExit(f"{label} {container_label} next_safe_command mismatch: {container}")
        if container.get("next_safe_commands") != expected_next:
            raise SystemExit(f"{label} {container_label} next_safe_commands mismatch: {container}")
        if container.get("next_safe_command_count") != len(expected_next):
            raise SystemExit(f"{label} {container_label} next_safe_command_count mismatch: {container}")
    for key, value in [
        (f"{prefix}_next_safe_command", expected_first),
        (f"{prefix}_next_safe_commands", expected_next),
        (f"{prefix}_next_safe_command_count", len(expected_next)),
    ]:
        if metadata.get(key) != value:
            raise SystemExit(f"{label} note prefixed safe-command alias {key} failed: {metadata}")


def assert_note_list_handoff(metadata: dict, *, expected_count: int, expected_total: int, label: str) -> None:
    handoff = metadata.get("note_list_handoff")
    if not metadata.get("note_list_handoff_ready") or not handoff:
        raise SystemExit(f"{label} missing note_list_handoff: {metadata}")
    if handoff.get("source") != "list_jarvis_notes" or handoff.get("count") != expected_count or handoff.get("total") != expected_total:
        raise SystemExit(f"{label} list handoff count/source mismatch: {handoff}")
    if handoff.get("limit") != metadata.get("limit") or handoff.get("count") != metadata.get("count"):
        raise SystemExit(f"{label} list handoff metadata parity failed: {metadata}")
    assert_note_contract(metadata, handoff, label=label, state_changed=False, changed=[], content_in_handoff=False)
    if expected_count:
        first = handoff.get("first_note_path")
        if not first or f"read jarvis note {first}" not in handoff.get("next_commands", []):
            raise SystemExit(f"{label} list handoff missed first-note command: {handoff}")
        assert_no_local_path(handoff, f"{label} list handoff")
    assert_note_boundaries(handoff, writes=False, reads_contents=False, label=label)


def assert_note_search_handoff(metadata: dict, *, expected_count: int, label: str) -> None:
    handoff = metadata.get("note_search_handoff")
    if not metadata.get("note_search_handoff_ready") or not handoff:
        raise SystemExit(f"{label} missing note_search_handoff: {metadata}")
    if handoff.get("source") != "search_jarvis_notes" or handoff.get("count") != expected_count:
        raise SystemExit(f"{label} search handoff count/source mismatch: {handoff}")
    if handoff.get("limit") != metadata.get("limit") or handoff.get("count") != metadata.get("count"):
        raise SystemExit(f"{label} search handoff metadata parity failed: {metadata}")
    assert_note_contract(metadata, handoff, label=label, state_changed=False, changed=[], content_in_handoff=expected_count > 0)
    if expected_count:
        first = handoff.get("first_note_path")
        if not first or f"outline jarvis note {first}" not in handoff.get("next_commands", []):
            raise SystemExit(f"{label} search handoff missed first-note command: {handoff}")
    assert_no_local_path(handoff, f"{label} search handoff")
    assert_note_boundaries(handoff, writes=False, reads_contents=True, label=label)


def assert_note_read_handoff(metadata: dict, *, expected_display: str, label: str) -> None:
    handoff = metadata.get("note_read_handoff")
    if not metadata.get("note_read_handoff_ready") or not handoff:
        raise SystemExit(f"{label} missing note_read_handoff: {metadata}")
    if handoff.get("source") != "read_jarvis_note" or handoff.get("path_display") != expected_display:
        raise SystemExit(f"{label} read handoff path/source mismatch: {handoff}")
    if handoff.get("chars") != metadata.get("chars") or handoff.get("truncated") != metadata.get("truncated"):
        raise SystemExit(f"{label} read handoff metadata parity failed: {metadata}")
    assert_note_contract(metadata, handoff, label=label, state_changed=False, changed=[], content_in_handoff=False)
    if f"outline jarvis note {expected_display}" not in handoff.get("next_commands", []):
        raise SystemExit(f"{label} read handoff missed outline command: {handoff}")
    assert_no_local_path(handoff, f"{label} read handoff")
    assert_note_boundaries(handoff, writes=False, reads_contents=True, label=label)


def assert_note_outline_handoff(metadata: dict, *, expected_display: str, label: str) -> None:
    handoff = metadata.get("note_outline_handoff")
    if not metadata.get("note_outline_handoff_ready") or not handoff:
        raise SystemExit(f"{label} missing note_outline_handoff: {metadata}")
    if handoff.get("source") != "outline_jarvis_note" or handoff.get("path_display") != expected_display:
        raise SystemExit(f"{label} outline handoff path/source mismatch: {handoff}")
    if handoff.get("heading_count") != metadata.get("headings") or handoff.get("task_count") != metadata.get("tasks") or handoff.get("link_count") != metadata.get("links"):
        raise SystemExit(f"{label} outline handoff metadata parity failed: {metadata}")
    assert_note_contract(
        metadata,
        handoff,
        label=label,
        state_changed=False,
        changed=[],
        content_in_handoff=bool(handoff.get("headings_preview") or handoff.get("tasks_preview") or handoff.get("links_preview")),
    )
    if f"read jarvis note {expected_display}" not in handoff.get("next_commands", []):
        raise SystemExit(f"{label} outline handoff missed read command: {handoff}")
    assert_no_local_path(handoff, f"{label} outline handoff")
    assert_note_boundaries(handoff, writes=False, reads_contents=True, label=label)


def assert_note_write_handoff(metadata: dict, *, expected_display: str, created: bool, label: str) -> None:
    handoff = metadata.get("note_write_handoff")
    if not metadata.get("note_write_handoff_ready") or not handoff:
        raise SystemExit(f"{label} missing note_write_handoff: {metadata}")
    if handoff.get("source") != "write_jarvis_note" or handoff.get("path_display") != expected_display:
        raise SystemExit(f"{label} write handoff path/source mismatch: {handoff}")
    if handoff.get("created") is not created or handoff.get("body_chars") != metadata.get("body_chars"):
        raise SystemExit(f"{label} write handoff metadata parity failed: {metadata}")
    assert_note_contract(metadata, handoff, label=label, state_changed=True, changed=["note_write"], content_in_handoff=handoff.get("body_chars", 0) > 0)
    if f"read jarvis note {expected_display}" not in handoff.get("next_commands", []):
        raise SystemExit(f"{label} write handoff missed read command: {handoff}")
    assert_no_local_path(handoff, f"{label} write handoff")
    assert_note_boundaries(handoff, writes=True, reads_contents=False, label=label)


def assert_note_refusal_handoff(
    metadata: dict,
    *,
    source: str,
    mutation: str,
    reason: str,
    label: str,
    expected_display: str | None = None,
    mode: str | None = None,
    body_chars: int | None = None,
    max_chars: int | None = None,
    exception_type: str | None = None,
) -> None:
    handoff = metadata.get("note_refusal_handoff")
    if not metadata.get("note_refusal_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing note_refusal_handoff: {metadata}")
    for success_key in ("note_read_handoff", "note_outline_handoff", "note_write_handoff"):
        if success_key in metadata:
            raise SystemExit(f"{label} refusal should not emit success handoff {success_key}: {metadata}")
    if handoff.get("source") != source or handoff.get("mutation") != mutation:
        raise SystemExit(f"{label} refusal handoff source/mutation mismatch: {handoff}")
    if metadata.get("reason") != reason or handoff.get("reason") != reason:
        raise SystemExit(f"{label} refusal handoff reason mismatch: {metadata}")
    if handoff.get("ready_for_operator") is not True or handoff.get("refused") is not True:
        raise SystemExit(f"{label} refusal handoff readiness mismatch: {handoff}")
    assert_note_contract(metadata, handoff, label=label, state_changed=False, changed=[], content_in_handoff=False)
    if handoff.get("changed") != []:
        raise SystemExit(f"{label} refusal should report no changed fields: {handoff}")
    if expected_display is not None and handoff.get("path_display") != expected_display:
        raise SystemExit(f"{label} refusal path display mismatch: {handoff}")
    if mode is not None and (metadata.get("mode") != mode or handoff.get("mode") != mode):
        raise SystemExit(f"{label} refusal mode parity failed: {metadata}")
    if body_chars is not None and (metadata.get("chars") != body_chars or handoff.get("body_chars") != body_chars):
        raise SystemExit(f"{label} refusal body_chars parity failed: {metadata}")
    if max_chars is not None and (metadata.get("max_chars") != max_chars or handoff.get("max_chars") != max_chars):
        raise SystemExit(f"{label} refusal max_chars parity failed: {metadata}")
    if exception_type is not None and (
        metadata.get("exception_type") != exception_type or handoff.get("exception_type") != exception_type
    ):
        raise SystemExit(f"{label} refusal exception parity failed: {metadata}")
    commands = handoff.get("next_commands")
    if not isinstance(commands, list) or "list jarvis notes" not in commands or not isinstance(handoff.get("retry_command"), str):
        raise SystemExit(f"{label} refusal missed recovery commands: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} refusal should stay read-only: {handoff}")
    for key in (
        "reads_note_contents",
        "writes_files",
        "writes_notes",
        "writes_memory",
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "calls_model",
        "executes_tools",
        "creates_note",
        "mutates_note",
    ):
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} refusal should keep {key}=False: {handoff}")
    for key in (
        "writes_files",
        "writes_notes",
        "writes_memory",
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} flat refusal metadata should keep {key}=False: {metadata}")
    assert_no_local_path(handoff, f"{label} refusal handoff")


def assert_concurrent_generic_note_writes(runtime) -> None:
    handler = runtime.registry.get("write_jarvis_note").handler
    path = "Projects/Concurrent Generic Note"
    base = "base entry"
    entries = ("unique concurrent alpha", "unique concurrent beta")
    initial = handler({"path": path, "body": base, "mode": "append"})
    if not initial.ok or initial.metadata.get("created") is not True:
        raise SystemExit(f"Concurrent append setup failed: {initial}")

    barrier = Barrier(3)

    def append_entry(entry: str):
        barrier.wait()
        return handler({"path": path, "body": entry, "mode": "append"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(append_entry, entry) for entry in entries]
        barrier.wait()
        results = [future.result() for future in futures]
    if any(not result.ok or result.metadata.get("created") is not False for result in results):
        raise SystemExit(f"Concurrent generic appends failed: {results}")

    note = runtime.vault.root_path / f"{path}.md"
    text = note.read_text(encoding="utf-8")
    expected_values = ("# Concurrent Generic Note", base, *entries)
    if any(text.count(value) != 1 for value in expected_values):
        raise SystemExit(f"Concurrent generic appends lost or duplicated content:\n{text}")

    create_path = "Projects/Concurrent Create"
    create_entries = ("exclusive create alpha", "exclusive create beta")
    create_barrier = Barrier(3)

    def create_entry(entry: str):
        create_barrier.wait()
        return handler({"path": create_path, "body": entry, "mode": "create"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(create_entry, entry) for entry in create_entries]
        create_barrier.wait()
        create_results = [future.result() for future in futures]
    if sum(result.ok for result in create_results) != 1:
        raise SystemExit(f"Concurrent creates should have exactly one winner: {create_results}")
    loser = next(result for result in create_results if not result.ok)
    if loser.metadata.get("reason") != "already_exists":
        raise SystemExit(f"Concurrent create loser used wrong refusal: {loser}")
    create_text = (runtime.vault.root_path / f"{create_path}.md").read_text(encoding="utf-8")
    if sum(create_text.count(entry) for entry in create_entries) != 1:
        raise SystemExit(f"Concurrent create overwrote or combined entries:\n{create_text}")

    outside = runtime.vault.vault_path / "outside-generic-note.md"
    outside_sentinel = "outside generic note sentinel\n"
    outside.write_text(outside_sentinel, encoding="utf-8")
    escape = runtime.vault.root_path / "Projects" / "Generic Escape.md"
    escape.symlink_to(outside)
    for action in (
        lambda: runtime.vault.append_note(escape, "must stay contained", heading="Generic Escape"),
        lambda: runtime.vault.create_note(escape, "must stay contained", heading="Generic Escape"),
    ):
        try:
            action()
        except ValueError:
            pass
        else:
            raise SystemExit("Generic note write followed a symlink outside the Jarvis vault.")
    if outside.read_text(encoding="utf-8") != outside_sentinel:
        raise SystemExit("Rejected generic note symlink write changed the outside target.")


def main() -> None:
    assert_note_exact_metadata_bool()
    assert_note_malformed_handoff_flags()
    assert_planner_routes_note_phone_aliases()
    assert_runtime_routes_note_count_aliases_without_memory_write()

    with TemporaryDirectory(prefix="jarvis-notes-") as temp:
        runtime = make_temp_runtime(Path(temp))
        note_path = runtime.vault.root_path / "Projects" / "Note Search Smoke.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(
            "# Note Search Smoke\n\n"
            "Jarvis can search its Obsidian notes.\n\n"
            "## Tasks\n\n"
            "- [ ] Verify note outline\n"
            "- [x] Keep note reading safe\n\n"
            "[Jarvis](https://example.local/jarvis)\n",
            encoding="utf-8",
        )

        cases = [
            "search jarvis notes Obsidian notes",
            "list jarvis notes",
            "show my notes",
            "list jarvis notes in Projects",
            "outline jarvis note Projects/Note Search Smoke",
            "read jarvis note Projects/Note Search Smoke",
            "create jarvis note Projects/Captured Idea: Jarvis can capture a scoped local note.",
            "append jarvis note Projects/Captured Idea: Second line stays inside the Jarvis vault.",
            "save jarvis note Projects/Saved Shortcut: Save note syntax should route to Obsidian, not memory.",
            "save to jarvis notes: Quick capture should land in the Jarvis inbox note.",
            "take a note: Phone capture should land in the Jarvis inbox note.",
            "take note that Note capture should strip leading that.",
            "jot this down: Jotted capture should land in the Jarvis inbox note.",
            "note to self: Self note should land in the Jarvis inbox note.",
            "make a note that Make-note capture should land in the Jarvis inbox note.",
            "make note of Make-note-of capture should strip leading of.",
            "write down Write-down capture should land in the Jarvis inbox note.",
            "write down that Write-down capture should strip leading that.",
            "save a note that Save-a-note capture should land in the Jarvis inbox note.",
            "save this note: Save-this-note capture should land in the Jarvis inbox note.",
            "read jarvis note ../outside",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1400])
            print()
            if case == "search jarvis notes Obsidian notes":
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 12 or metadata.get("writes_files") is not False:
                    raise SystemExit("Jarvis note search missed sanitized limit/read-only metadata.")
                assert_note_search_handoff(metadata, expected_count=1, label="Jarvis note search")
            if case == "list jarvis notes":
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 40 or metadata.get("reads_private_data") is not False:
                    raise SystemExit("Jarvis note list missed limit/privacy metadata.")
                assert_note_list_handoff(metadata, expected_count=metadata.get("count"), expected_total=metadata.get("total"), label="Jarvis note list")
            if case == "show my notes":
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 40 or metadata.get("reads_private_data") is not False:
                    raise SystemExit("Natural Jarvis note list missed limit/privacy metadata.")
                assert_note_list_handoff(metadata, expected_count=metadata.get("count"), expected_total=metadata.get("total"), label="Natural Jarvis note list")
            if case == "outline jarvis note Projects/Note Search Smoke":
                metadata = result.tool_results[0].metadata
                if metadata.get("tasks") != 2 or metadata.get("reads_private_data") is not False:
                    raise SystemExit("Jarvis note outline missed task/privacy metadata.")
                assert_note_outline_handoff(metadata, expected_display="Projects/Note Search Smoke.md", label="Jarvis note outline")
            if case == "read jarvis note Projects/Note Search Smoke":
                metadata = result.tool_results[0].metadata
                if metadata.get("max_chars") != 6000 or metadata.get("writes_files") is not False:
                    raise SystemExit("Jarvis note read missed bounded read metadata.")
                assert_note_read_handoff(metadata, expected_display="Projects/Note Search Smoke.md", label="Jarvis note read")
            if case == "create jarvis note Projects/Captured Idea: Jarvis can capture a scoped local note.":
                assert_vault_relative_note_receipt(
                    result,
                    expected_display="Projects/Captured Idea.md",
                    label="Created Jarvis note",
                )
                assert_note_write_handoff(result.tool_results[0].metadata, expected_display="Projects/Captured Idea.md", created=True, label="Created Jarvis note")
            if case == "append jarvis note Projects/Captured Idea: Second line stays inside the Jarvis vault.":
                assert_vault_relative_note_receipt(
                    result,
                    expected_display="Projects/Captured Idea.md",
                    label="Updated Jarvis note",
                )
                assert_note_write_handoff(result.tool_results[0].metadata, expected_display="Projects/Captured Idea.md", created=False, label="Updated Jarvis note")
            if case == "save jarvis note Projects/Saved Shortcut: Save note syntax should route to Obsidian, not memory.":
                assert_vault_relative_note_receipt(
                    result,
                    expected_display="Projects/Saved Shortcut.md",
                    label="Saved Jarvis note shortcut",
                )
                assert_note_write_handoff(result.tool_results[0].metadata, expected_display="Projects/Saved Shortcut.md", created=True, label="Saved Jarvis note shortcut")
            if case == "save to jarvis notes: Quick capture should land in the Jarvis inbox note.":
                assert_vault_relative_note_receipt(
                    result,
                    expected_display="Inbox/Captured Notes.md",
                    label="Quick Jarvis note capture",
                )
                assert_note_write_handoff(result.tool_results[0].metadata, expected_display="Inbox/Captured Notes.md", created=True, label="Quick Jarvis note capture")
            if case in {
                "take a note: Phone capture should land in the Jarvis inbox note.",
                "take note that Note capture should strip leading that.",
                "jot this down: Jotted capture should land in the Jarvis inbox note.",
                "note to self: Self note should land in the Jarvis inbox note.",
                "make a note that Make-note capture should land in the Jarvis inbox note.",
                "make note of Make-note-of capture should strip leading of.",
                "write down Write-down capture should land in the Jarvis inbox note.",
                "write down that Write-down capture should strip leading that.",
                "save a note that Save-a-note capture should land in the Jarvis inbox note.",
                "save this note: Save-this-note capture should land in the Jarvis inbox note.",
            }:
                assert_vault_relative_note_receipt(
                    result,
                    expected_display="Inbox/Captured Notes.md",
                    label=f"Natural note capture {case!r}",
                )
                assert_note_write_handoff(
                    result.tool_results[0].metadata,
                    expected_display="Inbox/Captured Notes.md",
                    created=False,
                    label=f"Natural note capture {case!r}",
                )

        captured_note = runtime.vault.root_path / "Projects" / "Captured Idea.md"
        if not captured_note.exists():
            raise SystemExit("Expected captured Jarvis note missing.")
        captured_text = captured_note.read_text(encoding="utf-8")
        for expected in [
            "Jarvis can capture a scoped local note.",
            "Second line stays inside the Jarvis vault.",
        ]:
            if expected not in captured_text:
                raise SystemExit(f"Captured Jarvis note missing expected text: {expected}")

        saved_shortcut = runtime.vault.root_path / "Projects" / "Saved Shortcut.md"
        if not saved_shortcut.exists() or "Save note syntax should route to Obsidian" not in saved_shortcut.read_text(encoding="utf-8"):
            raise SystemExit("Save-note shortcut did not write the expected Jarvis note.")

        inbox_capture = runtime.vault.root_path / "Inbox" / "Captured Notes.md"
        if not inbox_capture.exists() or "Quick capture should land in the Jarvis inbox note." not in inbox_capture.read_text(encoding="utf-8"):
            raise SystemExit("Quick capture did not append to the Jarvis inbox note.")
        inbox_capture_text = inbox_capture.read_text(encoding="utf-8")
        for expected in [
            "Phone capture should land in the Jarvis inbox note.",
            "Note capture should strip leading that.",
            "Jotted capture should land in the Jarvis inbox note.",
            "Self note should land in the Jarvis inbox note.",
            "Make-note capture should land in the Jarvis inbox note.",
            "Make-note-of capture should strip leading of.",
            "Write-down capture should land in the Jarvis inbox note.",
            "Write-down capture should strip leading that.",
            "Save-a-note capture should land in the Jarvis inbox note.",
            "Save-this-note capture should land in the Jarvis inbox note.",
        ]:
            if expected not in inbox_capture_text:
                raise SystemExit(f"Natural note capture missing expected inbox text: {expected}")

        list_result = runtime.handle("list jarvis notes")
        if not list_result.verified or "Projects/Note Search Smoke.md" not in list_result.response:
            raise SystemExit("Jarvis note list did not include expected note metadata.")
        assert_note_list_handoff(
            list_result.tool_results[0].metadata,
            expected_count=list_result.tool_results[0].metadata.get("count"),
            expected_total=list_result.tool_results[0].metadata.get("total"),
            label="Jarvis note list after writes",
        )
        list_projects_result = runtime.handle("list jarvis notes in Projects")
        if not list_projects_result.verified or "Projects/Captured Idea.md" not in list_projects_result.response:
            raise SystemExit("Jarvis folder note list did not include expected project note.")
        assert_note_list_handoff(list_projects_result.tool_results[0].metadata, expected_count=3, expected_total=3, label="Jarvis project note list")
        list_outside_result = runtime.handle("list jarvis notes in ../outside")
        if list_outside_result.verified:
            raise SystemExit("Jarvis note lister allowed path traversal.")
        if "note_list_handoff" in list_outside_result.tool_results[0].metadata:
            raise SystemExit(f"Unsafe list refusal should not emit a handoff: {list_outside_result.tool_results[0].metadata}")
        direct_absolute_folder = runtime.registry.get("list_jarvis_notes").handler({"folder": "/\x55sers/example/private/notes"})
        if direct_absolute_folder.ok or direct_absolute_folder.metadata.get("reason") != "unsafe_folder":
            raise SystemExit("list_jarvis_notes should reject absolute local folders.")
        if direct_absolute_folder.metadata.get("folder") != "<local-path>":
            raise SystemExit(f"list_jarvis_notes leaked absolute folder metadata: {direct_absolute_folder.metadata}")
        assert_no_local_path(direct_absolute_folder.metadata, "list_jarvis_notes absolute folder metadata")
        for folder in ["/var/folders/zc/jarvis/notes", "/tmp/jarvis-notes"]:
            temp_absolute_folder = runtime.registry.get("list_jarvis_notes").handler({"folder": folder})
            if temp_absolute_folder.ok or temp_absolute_folder.metadata.get("reason") != "unsafe_folder":
                raise SystemExit(f"list_jarvis_notes should reject temp-root folders: {temp_absolute_folder.metadata}")
            if temp_absolute_folder.metadata.get("folder") != "<local-path>":
                raise SystemExit(f"list_jarvis_notes leaked temp-root folder metadata: {temp_absolute_folder.metadata}")
            assert_no_local_path(temp_absolute_folder.metadata, "list_jarvis_notes temp-root folder metadata")

        outline_result = runtime.handle("summarize jarvis note Projects/Note Search Smoke")
        if not outline_result.verified:
            raise SystemExit("Jarvis note outline failed.")
        for expected in ["Jarvis note outline", "headings: 2", "tasks: 2", "markdown links: 1", "## Tasks"]:
            if expected not in outline_result.response:
                raise SystemExit(f"Jarvis note outline missed expected detail: {expected}")
        assert_note_outline_handoff(outline_result.tool_results[0].metadata, expected_display="Projects/Note Search Smoke.md", label="Jarvis note summarize")
        outline_outside_result = runtime.handle("outline jarvis note ../outside")
        if outline_outside_result.verified:
            raise SystemExit("Jarvis note outline allowed path traversal.")
        if "note_outline_handoff" in outline_outside_result.tool_results[0].metadata:
            raise SystemExit(f"Unsafe outline refusal should not emit a handoff: {outline_outside_result.tool_results[0].metadata}")
        assert_note_refusal_handoff(
            outline_outside_result.tool_results[0].metadata,
            source="outline_jarvis_note",
            mutation="note_outline",
            reason="unsafe_path",
            label="Unsafe outline refusal",
            expected_display="../outside",
        )
        for tool_name, args in [
            ("read_jarvis_note", {"path": "/\x55sers/example/private/note"}),
            ("outline_jarvis_note", {"path": "/private/tmp/jarvis-note"}),
            ("read_jarvis_note", {"path": "/var/folders/zc/jarvis/note"}),
            ("outline_jarvis_note", {"path": "/tmp/jarvis-note"}),
            ("write_jarvis_note", {"path": "/\x55sers/example/private/write-note", "body": "body"}),
            ("write_jarvis_note", {"path": "/private/tmp/jarvis-empty-body", "body": ""}),
            ("write_jarvis_note", {"path": "/var/folders/zc/jarvis/write-note", "body": "body"}),
            ("write_jarvis_note", {"path": "/tmp/jarvis-empty-body", "body": ""}),
            ("write_jarvis_note", {"path": "/\x55sers/example/private/bad-mode", "body": "body", "mode": "overwrite"}),
            ("write_jarvis_note", {"path": "/var/folders/zc/jarvis/bad-mode", "body": "body", "mode": "overwrite"}),
        ]:
            result = runtime.registry.get(tool_name).handler(args)
            if result.ok:
                raise SystemExit(f"{tool_name} should reject absolute local note paths.")
            if result.metadata.get("path") != "<local-path>":
                raise SystemExit(f"{tool_name} leaked absolute path metadata: {result.metadata}")
            assert_no_local_path(result.metadata, f"{tool_name} absolute path metadata")
            assert_no_local_path(result.output, f"{tool_name} absolute path output")
            assert_note_refusal_handoff(
                result.metadata,
                source=tool_name,
                mutation="note_write" if tool_name == "write_jarvis_note" else ("note_outline" if tool_name == "outline_jarvis_note" else "note_read"),
                reason=result.metadata.get("reason"),
                label=f"{tool_name} absolute path refusal",
                expected_display="<local-path>",
                mode=result.metadata.get("mode"),
            )

        outside_result = runtime.handle("write jarvis note ../outside: should not write")
        if outside_result.verified:
            raise SystemExit("Jarvis note writer allowed path traversal.")
        if outside_result.tool_results[0].metadata.get("reason") != "unsafe_path":
            raise SystemExit("Jarvis note writer path traversal missed refusal metadata.")
        assert_note_refusal_handoff(
            outside_result.tool_results[0].metadata,
            source="write_jarvis_note",
            mutation="note_write",
            reason="unsafe_path",
            label="Unsafe write refusal",
            expected_display="../outside",
            mode="append",
        )
        print("[blocked] write jarvis note ../outside")
        print(outside_result.response[:1400])
        print()

        direct_list = runtime.registry.get("list_jarvis_notes").handler({"limit": "bad"})
        if not direct_list.ok or direct_list.metadata.get("limit") != 40:
            raise SystemExit("list_jarvis_notes did not sanitize a bad limit.")
        if direct_list.metadata.get("raw_limit") != "bad":
            raise SystemExit(f"list_jarvis_notes should preserve bounded raw limit metadata: {direct_list.metadata}")
        assert_note_list_handoff(direct_list.metadata, expected_count=direct_list.metadata.get("count"), expected_total=direct_list.metadata.get("total"), label="Direct bad-limit note list")
        bool_list = runtime.registry.get("list_jarvis_notes").handler({"limit": False})
        if not bool_list.ok or bool_list.metadata.get("limit") != 40 or bool_list.metadata.get("raw_limit") != "False":
            raise SystemExit(f"list_jarvis_notes should treat boolean limits as malformed and preserve raw metadata: {bool_list.metadata}")
        long_bad_list = runtime.registry.get("list_jarvis_notes").handler({"limit": "l" * 120})
        if long_bad_list.metadata.get("raw_limit") != ("l" * 77 + "..."):
            raise SystemExit(f"list_jarvis_notes should bound long raw limit metadata: {long_bad_list.metadata}")
        path_bad_list = runtime.registry.get("list_jarvis_notes").handler({"limit": "/\x55sers/example/private/note-list-limit"})
        if path_bad_list.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"list_jarvis_notes leaked local path in raw limit metadata: {path_bad_list.metadata}")
        for value in ["/var/folders/zc/jarvis/note-list-limit", "/tmp/jarvis-note-list-limit"]:
            temp_path_bad_list = runtime.registry.get("list_jarvis_notes").handler({"limit": value})
            if temp_path_bad_list.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"list_jarvis_notes leaked temp-root raw limit metadata: {temp_path_bad_list.metadata}")
            assert_no_local_path(temp_path_bad_list.metadata, "list_jarvis_notes temp-root raw limit metadata")
        file_as_folder = runtime.registry.get("list_jarvis_notes").handler({"folder": "Projects/Note Search Smoke.md", "limit": "bad"})
        if file_as_folder.ok or file_as_folder.metadata.get("reason") != "not_folder":
            raise SystemExit("list_jarvis_notes should reject file paths used as folders.")
        if file_as_folder.metadata.get("writes_files") or file_as_folder.metadata.get("queues_approval"):
            raise SystemExit("list_jarvis_notes file-as-folder refusal should stay read-only.")
        if file_as_folder.metadata.get("raw_limit") != "bad":
            raise SystemExit(f"list_jarvis_notes refusal should preserve raw limit metadata: {file_as_folder.metadata}")
        direct_search = runtime.registry.get("search_jarvis_notes").handler({"query": "Jarvis", "limit": 999999})
        if not direct_search.ok or direct_search.metadata.get("limit") != 200:
            raise SystemExit("search_jarvis_notes did not clamp a large limit.")
        assert_note_search_handoff(direct_search.metadata, expected_count=direct_search.metadata.get("count"), label="Direct clamped note search")
        bad_search = runtime.registry.get("search_jarvis_notes").handler({"query": "Jarvis", "limit": "bad"})
        if not bad_search.ok or bad_search.metadata.get("limit") != 12 or bad_search.metadata.get("raw_limit") != "bad":
            raise SystemExit(f"search_jarvis_notes should preserve sanitized raw limit metadata: {bad_search.metadata}")
        path_bad_search = runtime.registry.get("search_jarvis_notes").handler({"query": "Jarvis", "limit": "/private/tmp/jarvis-note-search-limit"})
        if path_bad_search.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"search_jarvis_notes leaked local path in raw limit metadata: {path_bad_search.metadata}")
        for value in ["/var/folders/zc/jarvis-note-search-limit", "/tmp/jarvis-note-search-limit"]:
            temp_path_bad_search = runtime.registry.get("search_jarvis_notes").handler({"query": "Jarvis", "limit": value})
            if temp_path_bad_search.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"search_jarvis_notes leaked temp-root raw limit metadata: {temp_path_bad_search.metadata}")
            assert_no_local_path(temp_path_bad_search.metadata, "search_jarvis_notes temp-root raw limit metadata")
        direct_read = runtime.registry.get("read_jarvis_note").handler({"path": "Projects/Note Search Smoke", "max_chars": 999999})
        if not direct_read.ok or direct_read.metadata.get("max_chars") != 20000:
            raise SystemExit("read_jarvis_note did not clamp a large max_chars.")
        assert_note_read_handoff(direct_read.metadata, expected_display="Projects/Note Search Smoke.md", label="Direct clamped note read")
        bool_read = runtime.registry.get("read_jarvis_note").handler({"path": "Projects/Note Search Smoke", "max_chars": True})
        if not bool_read.ok or bool_read.metadata.get("max_chars") != 6000 or bool_read.metadata.get("raw_max_chars") != "True":
            raise SystemExit(f"read_jarvis_note should treat boolean max_chars as malformed and preserve raw metadata: {bool_read.metadata}")
        bad_read = runtime.registry.get("read_jarvis_note").handler({"path": "Projects/Note Search Smoke", "max_chars": "bad"})
        if not bad_read.ok or bad_read.metadata.get("max_chars") != 6000 or bad_read.metadata.get("raw_max_chars") != "bad":
            raise SystemExit(f"read_jarvis_note should preserve sanitized raw max_chars metadata: {bad_read.metadata}")
        long_bad_read = runtime.registry.get("read_jarvis_note").handler({"path": "Projects/Note Search Smoke", "max_chars": "m" * 120})
        if long_bad_read.metadata.get("raw_max_chars") != ("m" * 77 + "..."):
            raise SystemExit(f"read_jarvis_note should bound long raw max_chars metadata: {long_bad_read.metadata}")
        path_bad_read = runtime.registry.get("read_jarvis_note").handler({"path": "Projects/Note Search Smoke", "max_chars": "/\x55sers/example/private/note-read-max-chars"})
        if path_bad_read.metadata.get("raw_max_chars") != "<local-path>":
            raise SystemExit(f"read_jarvis_note leaked local path in raw max_chars metadata: {path_bad_read.metadata}")
        for value in ["/var/folders/zc/jarvis-note-read-max-chars", "/tmp/jarvis-note-read-max-chars"]:
            temp_path_bad_read = runtime.registry.get("read_jarvis_note").handler({"path": "Projects/Note Search Smoke", "max_chars": value})
            if temp_path_bad_read.metadata.get("raw_max_chars") != "<local-path>":
                raise SystemExit(f"read_jarvis_note leaked temp-root raw max_chars metadata: {temp_path_bad_read.metadata}")
            assert_no_local_path(temp_path_bad_read.metadata.get("raw_max_chars"), "read_jarvis_note temp-root raw max_chars metadata")
        direct_small_read = runtime.registry.get("read_jarvis_note").handler({"path": "Projects/Note Search Smoke", "max_chars": 5})
        if not direct_small_read.ok or direct_small_read.metadata.get("truncated") is not True:
            raise SystemExit("read_jarvis_note did not report truncation.")
        assert_note_read_handoff(direct_small_read.metadata, expected_display="Projects/Note Search Smoke.md", label="Direct truncated note read")
        oversized = runtime.registry.get("write_jarvis_note").handler({"path": "Projects/Oversized", "body": "x" * 50001})
        if oversized.ok or "Refusing to write" not in oversized.output:
            raise SystemExit("write_jarvis_note did not refuse oversized note writes.")
        if oversized.metadata.get("max_chars") != 50000 or oversized.metadata.get("writes_files") is not False:
            raise SystemExit("write_jarvis_note oversized refusal missed metadata.")
        if "note_write_handoff" in oversized.metadata:
            raise SystemExit(f"write_jarvis_note oversized refusal should not emit handoff: {oversized.metadata}")
        assert_note_refusal_handoff(
            oversized.metadata,
            source="write_jarvis_note",
            mutation="note_write",
            reason="body_too_large",
            label="Oversized note write refusal",
            expected_display="Projects/Oversized",
            mode="append",
            body_chars=50001,
            max_chars=50000,
        )

        empty_query = runtime.registry.get("search_jarvis_notes").handler({"query": "", "limit": "bad"})
        if empty_query.ok or empty_query.metadata.get("reason") != "missing_query":
            raise SystemExit("search_jarvis_notes missing query missed safe refusal metadata.")
        if empty_query.metadata.get("query_chars") != 0 or empty_query.metadata.get("limit") != 12:
            raise SystemExit(f"search_jarvis_notes missing query should preserve sanitized query metadata: {empty_query.metadata}")
        if empty_query.metadata.get("raw_limit") != "bad":
            raise SystemExit(f"search_jarvis_notes missing query should preserve raw limit metadata: {empty_query.metadata}")
        if "note_search_handoff" in empty_query.metadata:
            raise SystemExit(f"search_jarvis_notes missing query should not emit handoff: {empty_query.metadata}")

        missing_note = runtime.registry.get("read_jarvis_note").handler({"path": "Projects/Missing"})
        if missing_note.ok or missing_note.metadata.get("reason") != "not_found":
            raise SystemExit("read_jarvis_note missing path missed stable refusal metadata.")
        if "note_read_handoff" in missing_note.metadata:
            raise SystemExit(f"read_jarvis_note missing path should not emit handoff: {missing_note.metadata}")
        assert_note_refusal_handoff(
            missing_note.metadata,
            source="read_jarvis_note",
            mutation="note_read",
            reason="not_found",
            label="Missing note read refusal",
            expected_display="Projects/Missing",
        )
        missing_outline = runtime.registry.get("outline_jarvis_note").handler({"path": "Projects/Missing"})
        missing_note_cases = [
            (
                missing_note,
                "missing note read",
                ["list jarvis notes", "read jarvis note Projects/Missing"],
            ),
            (
                missing_outline,
                "missing note outline",
                ["list jarvis notes", "outline jarvis note Projects/Missing"],
            ),
        ]
        for missing_result, label, expected_commands in missing_note_cases:
            if missing_result.ok or missing_result.metadata.get("reason") != "not_found":
                raise SystemExit(f"{label} should fail closed: {missing_result}")
            assert_resource_not_found_recovery(missing_result, label)
            for command in expected_commands:
                if command not in missing_result.output:
                    raise SystemExit(f"{label} missed recovery command {command!r}: {missing_result.output}")
            if missing_result.metadata.get("next_command") != expected_commands[0]:
                raise SystemExit(f"{label} missed first recovery command: {missing_result.metadata}")
            if missing_result.metadata.get("recovery_commands") != expected_commands:
                raise SystemExit(f"{label} missed ordered recovery metadata: {missing_result.metadata}")
            if missing_result.metadata.get("retry_requires_note_refresh") is not True:
                raise SystemExit(f"{label} should require note-list refresh: {missing_result.metadata}")
            if missing_result.metadata.get("authorizes_retry") is not False:
                raise SystemExit(f"{label} should not authorize retry: {missing_result.metadata}")
            assert_note_refusal_handoff(
                missing_result.metadata,
                source="read_jarvis_note" if label.endswith("read") else "outline_jarvis_note",
                mutation="note_read" if label.endswith("read") else "note_outline",
                reason="not_found",
                label=label,
                expected_display="Projects/Missing",
            )

        bad_mode = runtime.registry.get("write_jarvis_note").handler({"path": "Projects/Bad Mode", "body": "body", "mode": "overwrite"})
        if bad_mode.ok or bad_mode.metadata.get("reason") != "bad_mode":
            raise SystemExit("write_jarvis_note bad mode missed safe refusal metadata.")
        if "note_write_handoff" in bad_mode.metadata:
            raise SystemExit(f"write_jarvis_note bad mode should not emit handoff: {bad_mode.metadata}")
        assert_note_refusal_handoff(
            bad_mode.metadata,
            source="write_jarvis_note",
            mutation="note_write",
            reason="bad_mode",
            label="Bad mode note write refusal",
            expected_display="Projects/Bad Mode",
            mode="overwrite",
        )

        bounded_path = runtime.registry.get("write_jarvis_note").handler({"path": "Projects/" + "x" * 600, "body": "bounded path"})
        if bounded_path.ok or bounded_path.metadata.get("reason") != "path_too_large":
            raise SystemExit("write_jarvis_note should refuse an oversized path instead of changing its target.")
        assert_note_refusal_handoff(
            bounded_path.metadata,
            source="write_jarvis_note",
            mutation="note_write",
            reason="path_too_large",
            label="Oversized-path note write refusal",
            expected_display="Projects/" + "x" * 228 + "...",
            mode="append",
        )
        assert_concurrent_generic_note_writes(runtime)

        original_read_note_bounded = runtime.vault.read_note_bounded

        def failing_read_note_bounded(note_path: str | Path, *, max_chars: int):
            if Path(note_path).name == "Note Search Smoke.md":
                raise OSError("permission denied near /\x55sers/example/private/notes")
            return original_read_note_bounded(note_path, max_chars=max_chars)

        with patch.object(
            runtime.vault,
            "read_note_bounded",
            side_effect=failing_read_note_bounded,
        ):
            failed_read = runtime.registry.get("read_jarvis_note").handler({"path": "Projects/Note Search Smoke"})
            failed_outline = runtime.registry.get("outline_jarvis_note").handler({"path": "Projects/Note Search Smoke"})
        for failed in (failed_read, failed_outline):
            if failed.ok or "Could not read Jarvis note" not in failed.output:
                raise SystemExit(f"Jarvis note read failure should return stable guidance: {failed.output}")
            assert_local_productivity_read_recovery(
                failed,
                f"{failed.tool_name} I/O read refusal",
            )
            for expected in ["JARVIS_OBSIDIAN_VAULT", "vault permissions", "setup check", "then retry"]:
                if expected not in failed.output:
                    raise SystemExit(f"Jarvis note read failure missed actionable guidance {expected}: {failed.output}")
            if "/\x55sers/operator" in failed.output or "permission denied" in failed.output:
                raise SystemExit(f"Jarvis note read failure leaked raw filesystem text: {failed.output}")
            if failed.metadata.get("reason") != "read_failed" or failed.metadata.get("exception_type") != "OSError":
                raise SystemExit(f"Jarvis note read failure missed diagnostics: {failed.metadata}")
            if failed.metadata.get("writes_files") or failed.metadata.get("queues_approval"):
                raise SystemExit(f"Jarvis note read failure should stay local/read-only: {failed.metadata}")
            assert_note_refusal_handoff(
                failed.metadata,
                source=failed.tool_name,
                mutation="note_read",
                reason="read_failed",
                label=f"{failed.tool_name} I/O read refusal",
                expected_display="Projects/Note Search Smoke.md",
                exception_type="OSError",
            )

        failure_path = runtime.vault.root_path / "Projects" / "Write Failure.md"
        failure_sentinel = "# Write Failure\n\nexisting content must survive\n"
        failure_path.write_text(failure_sentinel, encoding="utf-8")
        with patch(
            "jarvis_v2.memory.obsidian._rename_exchange",
            side_effect=OSError("disk denied near /\x55sers/example/private/write"),
        ):
            failed_write = runtime.registry.get("write_jarvis_note").handler({"path": "Projects/Write Failure", "body": "body"})
        if failed_write.ok or "Could not write Jarvis note" not in failed_write.output:
            raise SystemExit(f"Jarvis note write failure should return stable guidance: {failed_write.output}")
        if failure_path.read_text(encoding="utf-8") != failure_sentinel:
            raise SystemExit("Failed generic append truncated or changed the existing note.")
        for expected in ["JARVIS_OBSIDIAN_VAULT", "vault permissions", "setup check", "then retry"]:
            if expected not in failed_write.output:
                raise SystemExit(f"Jarvis note write failure missed actionable guidance {expected}: {failed_write.output}")
        if "/\x55sers/operator" in failed_write.output or "disk denied" in failed_write.output:
            raise SystemExit(f"Jarvis note write failure leaked raw filesystem text: {failed_write.output}")
        if failed_write.metadata.get("reason") != "write_failed" or failed_write.metadata.get("exception_type") != "OSError":
            raise SystemExit(f"Jarvis note write failure missed diagnostics: {failed_write.metadata}")
        if failed_write.metadata.get("writes_files") or failed_write.metadata.get("queues_approval"):
            raise SystemExit(f"Jarvis note write failure should not claim completed writes: {failed_write.metadata}")
        assert_note_refusal_handoff(
            failed_write.metadata,
            source="write_jarvis_note",
            mutation="note_write",
            reason="write_failed",
            label="Write I/O refusal",
            expected_display="Projects/Write Failure.md",
            mode="append",
            exception_type="OSError",
        )


if __name__ == "__main__":
    main()
