from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.organize import _metadata_bool, _organize_handoff_metadata, make_organize_tools


def assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def assert_organize_exact_metadata_bool() -> None:
    for value in ("true", "false", "yes", "0", 1, 0, [], ["value"], None, object()):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"organize metadata bool should fail closed for {value!r}")
    if _metadata_bool(True) is not True:
        raise SystemExit("organize metadata bool should preserve exact True")
    if _metadata_bool(False) is not False:
        raise SystemExit("organize metadata bool should preserve exact False")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("organize metadata bool should honor the explicit default")


def assert_organize_malformed_handoff_flags() -> None:
    handoff = {
        "source": "organize_note",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["daily brief"],
        "boundaries": {"read_only": True},
    }
    metadata = _organize_handoff_metadata("organize_note_handoff", handoff)
    if metadata.get("state_changed") is not False:
        raise SystemExit(f"malformed organize state_changed should fail closed: {metadata}")
    if metadata.get("content_in_handoff") is not False:
        raise SystemExit(f"malformed organize content_in_handoff should fail closed: {metadata}")


def assert_organize_contract(
    metadata: dict,
    handoff: dict,
    *,
    label: str,
    state_changed: bool,
    changed: list[str],
    content_in_handoff: bool,
) -> None:
    for key, expected in (
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if metadata.get(key) != expected or handoff.get(key) != expected:
            raise SystemExit(f"{label} missed organize contract {key}={expected}: metadata={metadata} handoff={handoff}")


def assert_organize_handoff(metadata: dict, *, label: str) -> None:
    handoff = metadata.get("organize_note_handoff")
    if not metadata.get("organize_note_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing organize_note_handoff: {metadata}")
    if handoff.get("source") != "organize_note" or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} handoff source/ready mismatch: {handoff}")
    created = handoff.get("created") or {}
    changed = [
        key
        for key, plural in (
            ("tasks", "tasks"),
            ("goals", "goals"),
            ("memories", "memories"),
            ("decisions", "decisions"),
        )
        if metadata.get(plural)
    ]
    assert_organize_contract(
        metadata,
        handoff,
        label=label,
        state_changed=True,
        changed=changed,
        content_in_handoff=True,
    )
    for plural, singular in [("tasks", "task"), ("goals", "goal"), ("memories", "memory"), ("decisions", "decision")]:
        if created.get(f"{singular}_count") != metadata.get(plural):
            raise SystemExit(f"{label} handoff {plural} count mismatch: {handoff} vs {metadata}")
        if created.get(f"{singular}_ids") != metadata.get(f"{singular}_ids"):
            raise SystemExit(f"{label} handoff {singular} ids mismatch: {handoff} vs {metadata}")
    if handoff.get("skipped_count") != metadata.get("skipped") or handoff.get("overflow") != metadata.get("overflow"):
        raise SystemExit(f"{label} handoff skipped/overflow mismatch: {handoff} vs {metadata}")
    if handoff.get("lines_processed") != metadata.get("lines_processed"):
        raise SystemExit(f"{label} handoff lines_processed mismatch: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("read_only") or boundaries.get("writes_files") is not True or boundaries.get("writes_database") is not True:
        raise SystemExit(f"{label} handoff should describe local writes: {handoff}")
    if boundaries.get("writes_memory") != metadata.get("writes_memory") or boundaries.get("writes_notes") != metadata.get("writes_notes"):
        raise SystemExit(f"{label} handoff write boundaries should mirror metadata: {handoff} vs {metadata}")
    for key in ["queues_approval", "controls_computer", "external_side_effect"]:
        if boundaries.get(key):
            raise SystemExit(f"{label} handoff should keep {key}=False: {handoff}")
    if "daily brief" not in handoff.get("next_commands", []) or "export state" not in handoff.get("next_commands", []):
        raise SystemExit(f"{label} handoff missed follow-up commands: {handoff}")
    assert_no_local_path(handoff, f"{label} handoff")


def assert_organize_refusal_handoff(metadata: dict, *, label: str, reason: str, skipped: int = 0, overflow: int = 0, lines_processed: int = 0, limit: int | None = None) -> None:
    handoff = metadata.get("organize_refusal_handoff")
    if not metadata.get("organize_refusal_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing organize_refusal_handoff: {metadata}")
    if "organize_note_handoff" in metadata:
        raise SystemExit(f"{label} refusal should not emit success handoff: {metadata}")
    if handoff.get("source") != "organize_note" or handoff.get("mutation") != "brain_dump_organize":
        raise SystemExit(f"{label} refusal handoff source/mutation mismatch: {handoff}")
    if metadata.get("reason") != reason or handoff.get("reason") != reason:
        raise SystemExit(f"{label} refusal reason parity failed: {metadata}")
    if handoff.get("ready_for_operator") is not True or handoff.get("refused") is not True:
        raise SystemExit(f"{label} refusal readiness failed: {handoff}")
    assert_organize_contract(
        metadata,
        handoff,
        label=label,
        state_changed=False,
        changed=[],
        content_in_handoff=bool(handoff.get("text_chars") or handoff.get("skipped_count")),
    )
    created = handoff.get("created") or {}
    if handoff.get("created_total") != 0 or handoff.get("changed") != []:
        raise SystemExit(f"{label} refusal should report no created/changed state: {handoff}")
    for key in ("task_count", "goal_count", "memory_count", "decision_count"):
        if created.get(key) != 0:
            raise SystemExit(f"{label} refusal created counts should be zero: {handoff}")
    if metadata.get("skipped") != skipped or handoff.get("skipped_count") != skipped:
        raise SystemExit(f"{label} refusal skipped parity failed: {metadata}")
    if metadata.get("overflow") != overflow or handoff.get("overflow") != overflow:
        raise SystemExit(f"{label} refusal overflow parity failed: {metadata}")
    if metadata.get("lines_processed") != lines_processed or handoff.get("lines_processed") != lines_processed:
        raise SystemExit(f"{label} refusal lines_processed parity failed: {metadata}")
    if limit is not None and (metadata.get("limit") != limit or handoff.get("limit") != limit):
        raise SystemExit(f"{label} refusal limit parity failed: {metadata}")
    if "help organize" not in handoff.get("next_commands", []) or not isinstance(handoff.get("retry_command"), str):
        raise SystemExit(f"{label} refusal missed recovery commands: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} refusal should stay read-only: {handoff}")
    for key in (
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "writes_tasks",
        "writes_goals",
        "writes_decisions",
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
        "calls_model",
        "executes_tools",
    ):
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} refusal should keep {key}=False: {handoff}")
    for key in (
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
        "calls_model",
        "executes_tools",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} flat refusal metadata should keep {key}=False: {metadata}")
    assert_no_local_path(handoff, f"{label} refusal handoff")


def main() -> None:
    assert_organize_exact_metadata_bool()
    assert_organize_malformed_handoff_flags()
    with TemporaryDirectory(prefix="jarvis-organize-") as temp:
        runtime = make_temp_runtime(Path(temp))
        organize_note = make_organize_tools(runtime.store, runtime.vault)
        brain_dump = """organize brain dump:
task: review organized Jarvis notes due tomorrow priority high
remember: the operator wants pasted notes converted into durable assistant state
goal: Build brain dump organizer | convert messy notes into useful records | this week
decision: Brain dump organizer stays local-safe | it only writes Jarvis memory and Obsidian notes | no desktop control required
random line that should be skipped
"""
        cases = [
            brain_dump,
            "tasks",
            "goals",
            "decisions",
            "search memory for pasted notes durable assistant",
            "daily brief",
            "export state",
            "help organize",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            label = case.splitlines()[0]
            print(f"[{status}] {label}")
            print(result.response[:2200])
            print()
            if case == brain_dump:
                required = ["tasks: #1", "goals: #1", "memories: #", "decisions: #1", "Skipped lines"]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Organize output missing expected items: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("writes_memory") or not metadata.get("writes_notes"):
                    raise SystemExit(f"Organize brain dump should report memory/note writes: {metadata}")
                assert_organize_handoff(metadata, label="runtime organize brain dump")
            if case == "daily brief":
                required = ["review organized Jarvis notes", "Build brain dump organizer", "Brain dump organizer stays local-safe"]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Daily brief missing organized context: {missing}")

        empty_result = organize_note({"text": ""})
        if empty_result.ok:
            raise SystemExit("organize_note should reject empty text.")
        if "organize_note_handoff" in empty_result.metadata:
            raise SystemExit(f"empty organize_note should not emit handoff: {empty_result.metadata}")
        assert_organize_refusal_handoff(empty_result.metadata, label="empty organize_note", reason="missing_text")

        oversized_result = organize_note({"text": "x" * 50001})
        if oversized_result.ok or "too large" not in oversized_result.output:
            raise SystemExit("organize_note should refuse oversized brain dumps.")
        if oversized_result.metadata.get("writes_files") or oversized_result.metadata.get("writes_database"):
            raise SystemExit("oversized organize_note should not write.")
        if "organize_note_handoff" in oversized_result.metadata:
            raise SystemExit(f"oversized organize_note should not emit handoff: {oversized_result.metadata}")
        assert_organize_refusal_handoff(oversized_result.metadata, label="oversized organize_note", reason="text_too_large", limit=50000)

        no_match_result = organize_note({"text": "just an unprefixed line"})
        if no_match_result.ok or no_match_result.metadata.get("writes_files"):
            raise SystemExit("organize_note should reject unrecognized lines without writing.")
        if "organize_note_handoff" in no_match_result.metadata:
            raise SystemExit(f"unrecognized organize_note should not emit handoff: {no_match_result.metadata}")
        assert_organize_refusal_handoff(no_match_result.metadata, label="unrecognized organize_note", reason="no_recognizable_lines", skipped=1, lines_processed=1)

        for text in [
            "remember: /\x55sers/example/private/brain-dump-memory",
            "remember: /var/folders/zc/jarvis/brain-dump-memory",
            "remember: /tmp/jarvis-brain-dump-memory",
        ]:
            path_only_result = organize_note({"text": text})
            if path_only_result.ok:
                raise SystemExit("organize_note should reject path-shaped recognized content when nothing else is usable.")
            if path_only_result.metadata.get("writes_files") or path_only_result.metadata.get("writes_database"):
                raise SystemExit(f"organize_note path-only content should not write: {path_only_result.metadata}")
            if path_only_result.metadata.get("skipped") != 1:
                raise SystemExit(f"organize_note path-only content should count as skipped: {path_only_result.metadata}")
            assert_no_local_path(path_only_result.output, "organize_note path-only output")
            assert_organize_refusal_handoff(path_only_result.metadata, label="path-only organize_note", reason="no_recognizable_lines", skipped=1, lines_processed=1)

        mixed_path_result = organize_note({"text": "task: safe organizer task\nremember: /private/tmp/organize-memory\nrandom /\x55sers/example/private/skipped-line\nremember: /var/folders/zc/jarvis/organize-memory\nrandom /tmp/jarvis-organize-skipped-line"})
        if not mixed_path_result.ok:
            raise SystemExit("organize_note should still process safe recognized lines when path-shaped lines are skipped.")
        if mixed_path_result.metadata.get("tasks") != 1 or mixed_path_result.metadata.get("memories") != 0 or mixed_path_result.metadata.get("skipped") != 4:
            raise SystemExit(f"organize_note mixed path content should write only safe lines: {mixed_path_result.metadata}")
        if mixed_path_result.metadata.get("writes_memory") or not mixed_path_result.metadata.get("writes_notes"):
            raise SystemExit(f"organize_note mixed path content should report only task-note writes: {mixed_path_result.metadata}")
        assert_organize_handoff(mixed_path_result.metadata, label="mixed path organize_note")
        assert_no_local_path(mixed_path_result.output, "organize_note mixed path output")
        if "<local-path>" not in mixed_path_result.output:
            raise SystemExit(f"organize_note should redact skipped path lines in output: {mixed_path_result.output}")

        overflow_text = "\n".join(["task: bounded brain dump item"] * 205)
        overflow_result = organize_note({"text": overflow_text})
        if overflow_result.ok:
            raise SystemExit("organize_note should reject overflow before silently dropping lines.")
        if overflow_result.metadata.get("writes_files") or overflow_result.metadata.get("writes_database"):
            raise SystemExit(f"overflow organize_note should not write: {overflow_result.metadata}")
        assert_organize_refusal_handoff(
            overflow_result.metadata,
            label="overflow organize_note",
            reason="too_many_lines",
            overflow=5,
            lines_processed=0,
            limit=200,
        )


if __name__ == "__main__":
    main()
