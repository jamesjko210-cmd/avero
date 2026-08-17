from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier

from jarvis_v2.agent.failure_guidance import RESOURCE_NOT_FOUND_RECOVERY_ACTION
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.memory.store import MemoryStore, TaskRecord, normalized_task_identity
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.tasks import _metadata_bool, _task_handoff_metadata, _task_refusal_metadata


class HostileRow:
    def __init__(self, marker: str):
        self.marker = marker

    def __getitem__(self, key: str) -> object:
        raise RuntimeError(f"{self.marker}:{key}")

    def __str__(self) -> str:
        return self.marker


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


def assert_safe(metadata: dict, label: str) -> None:
    for key in (
        "calls_model",
        "executes_tools",
        "queues_approval",
        "controls_computer",
        "reads_private_data",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if metadata.get(key):
            raise SystemExit(f"{label} unexpectedly set {key}: {metadata}")


def assert_recovery_closure_proof_aliases(metadata: dict, label: str) -> None:
    commands = metadata.get("recovery_closure_required_commands") or []
    if metadata.get("recovery_closure_proof_queue") != commands:
        raise SystemExit(f"{label} recovery proof queue should mirror required commands: {metadata}")
    if metadata.get("recovery_closure_proof_queue_count") != len(commands):
        raise SystemExit(f"{label} recovery proof queue count diverged: {metadata}")
    if commands and metadata.get("recovery_closure_next_proof_command") != commands[0]:
        raise SystemExit(f"{label} missed next recovery proof command: {metadata}")
    if metadata.get("recovery_closure_blocks_task_completion"):
        if metadata.get("recovery_closure_checklist_command") != "recovery closure checklist":
            raise SystemExit(f"{label} missed recovery closure checklist command: {metadata}")
        if metadata.get("recovery_closure_should_open_checklist") is not True:
            raise SystemExit(f"{label} missed recovery closure checklist open flag: {metadata}")
    elif metadata.get("recovery_closure_checklist_command"):
        raise SystemExit(f"{label} should not expose checklist command when closure is not blocking: {metadata}")


def assert_task_exact_metadata_bool() -> None:
    for value in ("true", "false", "yes", "0", 1, 0, [], ["value"], None, object()):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"task metadata bool should fail closed for {value!r}")
    if _metadata_bool(True) is not True:
        raise SystemExit("task metadata bool should preserve exact True")
    if _metadata_bool(False) is not False:
        raise SystemExit("task metadata bool should preserve exact False")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("task metadata bool should honor the explicit default")


def assert_task_malformed_handoff_flags() -> None:
    handoff = {
        "source": "show_task",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["show task 1"],
        "boundaries": {"read_only": True},
    }
    metadata = _task_handoff_metadata("task_inspection_handoff", handoff)
    if metadata.get("state_changed") is not False or metadata.get("task_inspection_state_changed") is not False:
        raise SystemExit(f"malformed task state_changed should fail closed: {metadata}")
    if metadata.get("content_in_handoff") is not False or metadata.get("task_inspection_content_in_handoff") is not False:
        raise SystemExit(f"malformed task content_in_handoff should fail closed: {metadata}")


def assert_task_next_commands_fail_closed() -> None:
    handoff = {
        "source": "list_tasks",
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "next_commands": {
            "show": "show task 1",
            "blank": "",
            "path": "open /\x55sers/example/private/task-plan.md",
            "object": {"command": "task board"},
            "long": "task " + ("x" * 260),
        },
        "boundaries": {"read_only": True},
    }
    metadata = _task_handoff_metadata("task_list_handoff", handoff)
    normalized = metadata.get("task_list_handoff")
    if not isinstance(normalized, dict):
        raise SystemExit(f"task handoff should remain available after normalization: {metadata}")
    safe_commands = normalized.get("next_safe_commands")
    if safe_commands != metadata.get("next_safe_commands") or not isinstance(safe_commands, list):
        raise SystemExit(f"task handoff did not preserve only safe next commands: {metadata}")
    if len(safe_commands) != 2 or safe_commands[0] != "show task 1":
        raise SystemExit(f"task handoff did not preserve safe command order: {metadata}")
    if not safe_commands[1].startswith("task ") or not safe_commands[1].endswith("…") or len(safe_commands[1]) > 180:
        raise SystemExit(f"task handoff did not bound long next command safely: {metadata}")
    if normalized.get("next_safe_command") != "show task 1" or metadata.get("task_list_next_safe_command") != "show task 1":
        raise SystemExit(f"task handoff missed first safe next command: {metadata}")
    next_commands = normalized.get("next_commands") or {}
    if next_commands.get("blank") != "":
        raise SystemExit(f"task handoff should preserve blank compatibility slots: {metadata}")
    if "path" in next_commands or "object" in next_commands:
        raise SystemExit(f"task handoff should drop unsafe/non-string command slots: {metadata}")
    if normalized.get("hidden_next_command_count") != 2:
        raise SystemExit(f"task handoff missed hidden unsafe command count: {metadata}")
    if any(fragment in str(metadata) for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"task handoff leaked a local path through metadata: {metadata}")
    assert_safe(metadata, "task handoff next-command sanitization")


def assert_task_malformed_refusal_flags() -> None:
    metadata = _task_refusal_metadata(
        source="complete_task",
        reason="invalid_task_id",
        mutation="task_completion",
        raw_field="task_id",
        raw_value="abc",
    )
    handoff = metadata.get("task_refusal_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"task refusal handoff missing: {metadata}")
    handoff["content_in_handoff"] = "true"
    metadata = _task_refusal_metadata(
        source="complete_task",
        reason="invalid_task_id",
        mutation="task_completion",
        raw_field="task_id",
        raw_value="abc",
    )
    metadata["task_refusal_handoff"]["content_in_handoff"] = "true"
    repaired = _task_handoff_metadata("task_refusal_handoff", metadata["task_refusal_handoff"])
    if repaired.get("content_in_handoff") is not False or repaired.get("task_refusal_content_in_handoff") is not False:
        raise SystemExit(f"malformed task refusal content_in_handoff should fail closed: {repaired}")


def assert_task_readonly_reports_tolerate_malformed_rows() -> None:
    marker = "TASK_HOSTILE_ROW_SHOULD_NOT_LEAK"
    with TemporaryDirectory(prefix="jarvis-tasks-hostile-") as temp:
        runtime = make_temp_runtime(Path(temp))
        created = runtime.registry.get("add_task").handler({
            "body": "review malformed task rows",
            "priority": "high",
            "due": "tomorrow",
        })
        if not created.ok:
            raise SystemExit(f"hostile-row fixture task creation failed: {created}")
        readable_rows = runtime.store.list_tasks(status="open", limit=100)

        def hostile_tasks(status=None, limit=25):  # noqa: ANN001
            return [HostileRow(marker), *readable_rows]

        runtime.store.list_tasks = hostile_tasks  # type: ignore[method-assign]

        listed = runtime.registry.get("list_tasks").handler({"status": "open", "limit": 5})
        list_metadata = listed.metadata
        if not listed.ok:
            raise SystemExit(f"list_tasks should tolerate malformed task rows: {listed}")
        if list_metadata.get("count") != 1 or list_metadata.get("readable_task_rows") != 1:
            raise SystemExit(f"list_tasks should preserve readable task count: {list_metadata}")
        if list_metadata.get("unreadable_task_rows") != 1:
            raise SystemExit(f"list_tasks missed unreadable task counter: {list_metadata}")
        if "review malformed task rows" not in listed.output:
            raise SystemExit("list_tasks should preserve readable task body behind malformed rows.")
        if "unreadable task row(s) hidden for safety" not in listed.output:
            raise SystemExit("list_tasks should report hidden malformed rows.")
        if marker in listed.output or marker in str(list_metadata):
            raise SystemExit("list_tasks leaked raw malformed row text.")
        assert_task_list_handoff(list_metadata, "malformed-row list_tasks")
        assert_safe(list_metadata, "malformed-row list_tasks")

        searched = runtime.registry.get("search_tasks").handler({"query": "malformed", "status": "open", "limit": 5})
        search_metadata = searched.metadata
        if not searched.ok:
            raise SystemExit(f"search_tasks should tolerate malformed task rows: {searched}")
        if search_metadata.get("count") != 1 or search_metadata.get("readable_task_rows") != 1:
            raise SystemExit(f"search_tasks should preserve readable task count: {search_metadata}")
        if search_metadata.get("unreadable_task_rows") != 1:
            raise SystemExit(f"search_tasks missed unreadable task counter: {search_metadata}")
        if "review malformed task rows" not in searched.output:
            raise SystemExit("search_tasks should preserve readable match behind malformed rows.")
        if "unreadable task row(s) hidden for safety" not in searched.output:
            raise SystemExit("search_tasks should report hidden malformed rows.")
        if marker in searched.output or marker in str(search_metadata):
            raise SystemExit("search_tasks leaked raw malformed row text.")
        assert_task_search_handoff(search_metadata, "malformed-row search_tasks")
        assert_safe(search_metadata, "malformed-row search_tasks")


def assert_task_contract(
    metadata: dict,
    handoff: dict,
    label: str,
    *,
    state_changed: bool,
    changed: list[str],
    content_in_handoff: bool,
) -> None:
    prefix = ""
    for key, value in metadata.items():
        if key.endswith("_handoff") and value is handoff:
            prefix = key.removesuffix("_handoff")
            break
    if not prefix:
        for key, value in metadata.items():
            if key.endswith("_handoff") and value == handoff:
                prefix = key.removesuffix("_handoff")
                break
    for key, expected in (
        ("handoff_ready", True),
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if key == "handoff_ready":
            if handoff.get(key) != expected:
                raise SystemExit(f"{label} missed task contract {key}={expected}: metadata={metadata} handoff={handoff}")
            if prefix and metadata.get(f"{prefix}_handoff_ready") != expected:
                raise SystemExit(f"{label} missed prefixed task contract {prefix}_handoff_ready={expected}: metadata={metadata}")
            continue
        if metadata.get(key) != expected or handoff.get(key) != expected:
            raise SystemExit(f"{label} missed task contract {key}={expected}: metadata={metadata} handoff={handoff}")
        if prefix and metadata.get(f"{prefix}_{key}") != expected:
            raise SystemExit(f"{label} missed prefixed task contract {prefix}_{key}={expected}: metadata={metadata}")
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        expected_commands = []
        for value in raw_next.values():
            if isinstance(value, list):
                expected_commands.extend(str(item) for item in value if str(item or "").strip())
            elif str(value or "").strip():
                expected_commands.append(str(value))
    else:
        expected_commands = [str(value) for value in raw_next if str(value or "").strip()]
    expected_command = expected_commands[0] if expected_commands else ""
    if handoff.get("next_safe_command") != expected_command or metadata.get("next_safe_command") != expected_command:
        raise SystemExit(f"{label} missed next_safe_command parity: metadata={metadata} handoff={handoff}")
    if handoff.get("next_safe_commands") != expected_commands or metadata.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} missed next_safe_commands parity: metadata={metadata} handoff={handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands) or metadata.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} missed next_safe_command_count parity: metadata={metadata} handoff={handoff}")
    if prefix:
        if metadata.get(f"{prefix}_next_safe_command") != expected_command:
            raise SystemExit(f"{label} missed prefixed next_safe_command parity: metadata={metadata} handoff={handoff}")
        if metadata.get(f"{prefix}_next_safe_commands") != expected_commands:
            raise SystemExit(f"{label} missed prefixed next_safe_commands parity: metadata={metadata} handoff={handoff}")
        if metadata.get(f"{prefix}_next_safe_command_count") != len(expected_commands):
            raise SystemExit(f"{label} missed prefixed next_safe_command_count parity: metadata={metadata} handoff={handoff}")


def assert_no_boundary_authority(boundaries: dict, label: str) -> None:
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} handoff boundary {key} should be false: {boundaries}")


def assert_task_completion_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("task_completion_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed task completion handoff: {metadata}")
    if metadata.get("task_completion_handoff_ready") is not True:
        raise SystemExit(f"{label} missed task completion handoff ready flag: {metadata}")
    task = handoff.get("task") or {}
    evidence = handoff.get("evidence") or {}
    recent_audit_context = handoff.get("recent_audit_context") or {}
    recovery = handoff.get("recovery_closure") or {}
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    commands = metadata.get("recovery_closure_required_commands") or []
    if handoff.get("source") != "task_completion_packet":
        raise SystemExit(f"{label} handoff source diverged: {handoff}")
    assert_task_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=True)
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("task_id") != metadata.get("task_id") or task.get("id") != metadata.get("task_id"):
        raise SystemExit(f"{label} task id parity failed: {metadata}")
    if task.get("status") != metadata.get("status"):
        raise SystemExit(f"{label} task status parity failed: {metadata}")
    if handoff.get("verdict") != metadata.get("verdict"):
        raise SystemExit(f"{label} verdict parity failed: {metadata}")
    if handoff.get("has_evidence") is not metadata.get("has_evidence"):
        raise SystemExit(f"{label} evidence parity failed: {metadata}")
    if evidence.get("supplied") is not bool(metadata.get("has_evidence")) and not evidence.get("verification_run_supplied"):
        raise SystemExit(f"{label} evidence supplied flag diverged: {metadata}")
    if evidence.get("verification_run_id_chars") != metadata.get("verification_run_id_chars"):
        raise SystemExit(f"{label} verification id length parity failed: {metadata}")
    if recent_audit_context.get("recent_tool_runs") != metadata.get("recent_tool_runs"):
        raise SystemExit(f"{label} recent audit count parity failed: {metadata}")
    for handoff_key, metadata_key in (
        ("recent_ok_tool_runs", "recent_ok_tool_runs"),
        ("recent_failed_tool_runs", "recent_failed_tool_runs"),
        ("recent_approval_held_tool_runs", "recent_approval_held_tool_runs"),
        ("recent_unreadable_tool_run_rows", "recent_unreadable_tool_run_rows"),
    ):
        if recent_audit_context.get(handoff_key) != metadata.get(metadata_key):
            raise SystemExit(f"{label} recent audit {handoff_key} parity failed: {metadata}")
    if recovery.get("state") != metadata.get("recovery_closure_state"):
        raise SystemExit(f"{label} recovery state parity failed: {metadata}")
    if recovery.get("ready_to_retry") is not metadata.get("recovery_closure_ready_to_retry"):
        raise SystemExit(f"{label} recovery retry parity failed: {metadata}")
    if recovery.get("missing") != metadata.get("recovery_closure_missing"):
        raise SystemExit(f"{label} recovery missing parity failed: {metadata}")
    if recovery.get("missing_count") != metadata.get("recovery_closure_missing_count"):
        raise SystemExit(f"{label} recovery missing count parity failed: {metadata}")
    if recovery.get("required_commands") != commands or recovery.get("proof_queue") != commands:
        raise SystemExit(f"{label} recovery command queue parity failed: {metadata}")
    if recovery.get("proof_queue_count") != metadata.get("recovery_closure_proof_queue_count"):
        raise SystemExit(f"{label} recovery proof count parity failed: {metadata}")
    if recovery.get("next_required_command") != metadata.get("recovery_closure_next_required_command"):
        raise SystemExit(f"{label} next required recovery command parity failed: {metadata}")
    if recovery.get("next_proof_command") != metadata.get("recovery_closure_next_proof_command"):
        raise SystemExit(f"{label} next proof recovery command parity failed: {metadata}")
    if recovery.get("checklist_command") != metadata.get("recovery_closure_checklist_command"):
        raise SystemExit(f"{label} checklist command parity failed: {metadata}")
    if recovery.get("should_open_checklist") is not metadata.get("recovery_closure_should_open_checklist"):
        raise SystemExit(f"{label} checklist flag parity failed: {metadata}")
    if recovery.get("blocks_task_completion") is not metadata.get("recovery_closure_blocks_task_completion"):
        raise SystemExit(f"{label} recovery block parity failed: {metadata}")
    if recovery.get("target_run_id") != metadata.get("recovery_closure_target_run_id"):
        raise SystemExit(f"{label} recovery target run parity failed: {metadata}")
    if recovery.get("target_tool_name") != metadata.get("recovery_closure_target_tool_name"):
        raise SystemExit(f"{label} recovery target tool parity failed: {metadata}")
    if next_commands.get("recovery_proof_queue") != commands:
        raise SystemExit(f"{label} next command recovery queue parity failed: {metadata}")
    if commands and next_commands.get("next_recovery_proof") != commands[0]:
        raise SystemExit(f"{label} next recovery proof should be first queued command: {metadata}")
    if "complete task" not in str(next_commands.get("complete_with_evidence")):
        raise SystemExit(f"{label} missed completion evidence command: {metadata}")
    if handoff.get("completion_allowed") is not (metadata.get("verdict") == "READY_TO_COMPLETE"):
        raise SystemExit(f"{label} completion allowed derivation failed: {metadata}")
    if handoff.get("already_done") is not (metadata.get("verdict") == "ALREADY_DONE"):
        raise SystemExit(f"{label} already done derivation failed: {metadata}")
    if handoff.get("evidence_required") is not (metadata.get("verdict") == "EVIDENCE_REQUIRED"):
        raise SystemExit(f"{label} evidence required derivation failed: {metadata}")
    if handoff.get("recovery_closure_required") is not (metadata.get("verdict") == "RECOVERY_CLOSURE_REQUIRED"):
        raise SystemExit(f"{label} recovery required derivation failed: {metadata}")
    for key in (
        "completes_task",
        "writes_files",
        "writes_memory",
        "writes_notes",
        "calls_model",
        "executes_tools",
        "queues_approval",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "executes_side_effect",
        "controls_computer",
    ):
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} handoff boundary {key} should be false: {metadata}")


def assert_task_inspection_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("task_inspection_handoff")
    if not isinstance(handoff, dict) or metadata.get("task_inspection_handoff_ready") is not True:
        raise SystemExit(f"{label} missed task inspection handoff: {metadata}")
    task = handoff.get("task") or {}
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("source") != "inspect_task":
        raise SystemExit(f"{label} inspection handoff source diverged: {handoff}")
    assert_task_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=True)
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("task_id") != metadata.get("task_id") or task.get("id") != metadata.get("task_id"):
        raise SystemExit(f"{label} inspection task id parity failed: {metadata}")
    for key in ["status", "priority", "due", "source"]:
        if task.get(key) != metadata.get(key):
            raise SystemExit(f"{label} inspection {key} parity failed: {metadata}")
    if not isinstance(task.get("body_chars"), int) or task.get("body_chars") <= 0:
        raise SystemExit(f"{label} inspection handoff missed body length: {handoff}")
    if next_commands.get("show") != f"show task {metadata.get('task_id')}":
        raise SystemExit(f"{label} inspection handoff missed show command: {handoff}")
    if next_commands.get("completion_packet") != f"task completion packet {metadata.get('task_id')}":
        raise SystemExit(f"{label} inspection handoff missed completion packet command: {handoff}")
    for key in ["changes_task_status", "completes_task", "writes_files", "writes_memory", "writes_notes", "calls_model", "executes_tools", "queues_approval", "controls_computer"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} inspection handoff boundary {key} should be false: {handoff}")


def assert_task_list_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("task_list_handoff")
    if not isinstance(handoff, dict) or metadata.get("task_list_handoff_ready") is not True:
        raise SystemExit(f"{label} missed task list handoff: {metadata}")
    tasks = handoff.get("tasks") or []
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("source") != "list_tasks":
        raise SystemExit(f"{label} task list handoff source diverged: {handoff}")
    assert_task_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=bool(tasks))
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("status") != metadata.get("status") or handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} task list status/limit parity failed: {metadata}")
    if handoff.get("task_count") != metadata.get("count") or len(tasks) != metadata.get("count"):
        raise SystemExit(f"{label} task list count parity failed: {metadata}")
    if handoff.get("task_ids") != [row.get("id") for row in tasks]:
        raise SystemExit(f"{label} task list ids diverged: {handoff}")
    if tasks:
        first_id = tasks[0].get("id")
        if handoff.get("first_task_id") != first_id:
            raise SystemExit(f"{label} task list first task diverged: {handoff}")
        if next_commands.get("show_first_task") != f"show task {first_id}":
            raise SystemExit(f"{label} task list missed show command: {handoff}")
        if next_commands.get("completion_packet_first_task") != f"task completion packet {first_id}":
            raise SystemExit(f"{label} task list missed completion packet command: {handoff}")
    elif handoff.get("first_task_id") is not None or next_commands.get("show_first_task"):
        raise SystemExit(f"{label} empty task list should not name a first task: {handoff}")
    for row in tasks:
        if not isinstance(row.get("body_chars"), int) or row.get("body_chars") <= 0:
            raise SystemExit(f"{label} task list missed body length: {handoff}")
    for key in ["changes_task_status", "completes_task", "writes_files", "writes_memory", "writes_notes", "calls_model", "executes_tools", "queues_approval", "controls_computer"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} task list handoff boundary {key} should be false: {handoff}")


def assert_task_overview_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("task_overview_handoff")
    if not isinstance(handoff, dict) or metadata.get("task_overview_handoff_ready") is not True:
        raise SystemExit(f"{label} missed task overview handoff: {metadata}")
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("source") != "task_overview":
        raise SystemExit(f"{label} task overview handoff source diverged: {handoff}")
    assert_task_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=bool(handoff.get("total")))
    assert_no_boundary_authority(boundaries, label)
    for key in ["limit", "total", "status_counts", "priority_counts"]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} task overview {key} parity failed: {metadata}")
    for key, metadata_key in [("open_count", "open"), ("paused_count", "paused"), ("done_count", "done"), ("dropped_count", "dropped")]:
        if handoff.get(key) != metadata.get(metadata_key):
            raise SystemExit(f"{label} task overview {key} parity failed: {handoff} vs {metadata}")
    first_open = handoff.get("first_open_task_id")
    if metadata.get("open") and next_commands.get("show_first_open") != f"show task {first_open}":
        raise SystemExit(f"{label} task overview missed first-open show command: {handoff}")
    if metadata.get("open") and next_commands.get("completion_packet_first_open") != f"task completion packet {first_open}":
        raise SystemExit(f"{label} task overview missed first-open completion command: {handoff}")
    for key in ["changes_task_status", "completes_task", "writes_files", "writes_memory", "writes_notes", "calls_model", "executes_tools", "queues_approval", "controls_computer"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} task overview handoff boundary {key} should be false: {handoff}")


def assert_next_task_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("next_task_handoff")
    if not isinstance(handoff, dict) or metadata.get("next_task_handoff_ready") is not True:
        raise SystemExit(f"{label} missed next task handoff: {metadata}")
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("source") != "next_task":
        raise SystemExit(f"{label} next task handoff source diverged: {handoff}")
    assert_task_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=bool(metadata.get("task_id")))
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("task_id") != metadata.get("task_id"):
        raise SystemExit(f"{label} next task id parity failed: {metadata}")
    if metadata.get("task_id") is None:
        if handoff.get("task_selected") is not False or next_commands.get("add_task") != "add task <task body>":
            raise SystemExit(f"{label} empty next task handoff malformed: {handoff}")
    else:
        task = handoff.get("task") or {}
        if handoff.get("task_selected") is not True:
            raise SystemExit(f"{label} next task should be selected: {handoff}")
        for key in ["status", "priority", "due"]:
            if task.get(key) != metadata.get(key):
                raise SystemExit(f"{label} next task {key} parity failed: {metadata}")
        if handoff.get("reason") != metadata.get("reason"):
            raise SystemExit(f"{label} next task reason parity failed: {metadata}")
        if next_commands.get("show") != f"show task {metadata.get('task_id')}":
            raise SystemExit(f"{label} next task missed show command: {handoff}")
        if next_commands.get("completion_packet") != f"task completion packet {metadata.get('task_id')}":
            raise SystemExit(f"{label} next task missed completion command: {handoff}")
    for key in ["changes_task_status", "completes_task", "writes_files", "writes_memory", "writes_notes", "calls_model", "executes_tools", "queues_approval", "controls_computer"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} next task handoff boundary {key} should be false: {handoff}")


def assert_task_board_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("task_board_handoff")
    if not isinstance(handoff, dict) or metadata.get("task_board_handoff_ready") is not True:
        raise SystemExit(f"{label} missed task board handoff: {metadata}")
    board = handoff.get("board") or {}
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("source") != "task_board":
        raise SystemExit(f"{label} task board handoff source diverged: {handoff}")
    assert_task_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=any(bool(rows) for rows in board.values()))
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} task board limit parity failed: {metadata}")
    for status in ["open", "paused", "done", "dropped"]:
        if status not in board:
            raise SystemExit(f"{label} task board missed {status}: {handoff}")
        if handoff.get("status_counts", {}).get(status) != len(board.get(status) or []):
            raise SystemExit(f"{label} task board count diverged for {status}: {handoff}")
    first_open = handoff.get("first_open_task_id")
    if board.get("open"):
        if next_commands.get("show_first_open") != f"show task {first_open}":
            raise SystemExit(f"{label} task board missed first-open show command: {handoff}")
        if next_commands.get("completion_packet_first_open") != f"task completion packet {first_open}":
            raise SystemExit(f"{label} task board missed first-open completion command: {handoff}")
    for rows in board.values():
        for row in rows:
            if not isinstance(row.get("body_chars"), int) or row.get("body_chars") <= 0:
                raise SystemExit(f"{label} task board missed body length: {handoff}")
    for key in ["changes_task_status", "completes_task", "writes_files", "writes_memory", "writes_notes", "calls_model", "executes_tools", "queues_approval", "controls_computer"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} task board handoff boundary {key} should be false: {handoff}")


def assert_task_mutation_handoff(metadata: dict, label: str, *, mutation: str, changed: list[str]) -> None:
    handoff = metadata.get("task_mutation_handoff")
    if not isinstance(handoff, dict) or metadata.get("task_mutation_handoff_ready") is not True:
        raise SystemExit(f"{label} missed task mutation handoff: {metadata}")
    task = handoff.get("task") or {}
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("mutation") != mutation:
        raise SystemExit(f"{label} mutation handoff type diverged: {handoff}")
    assert_task_contract(metadata, handoff, label, state_changed=True, changed=changed, content_in_handoff=True)
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("task_id") != metadata.get("task_id") or task.get("id") != metadata.get("task_id"):
        raise SystemExit(f"{label} mutation task id parity failed: {metadata}")
    if handoff.get("changed") != changed:
        raise SystemExit(f"{label} mutation changed fields diverged: {handoff}")
    if not str(handoff.get("path_display") or ""):
        raise SystemExit(f"{label} mutation handoff missed refreshed task export path: {handoff}")
    if any(fragment in str(handoff.get("path_display")) for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} mutation handoff leaked raw local path: {handoff}")
    if next_commands.get("show") != f"show task {metadata.get('task_id')}":
        raise SystemExit(f"{label} mutation handoff missed show command: {handoff}")
    if next_commands.get("completion_packet") != f"task completion packet {metadata.get('task_id')}":
        raise SystemExit(f"{label} mutation handoff missed completion packet command: {handoff}")
    for key in ["writes_files", "writes_memory", "writes_notes"]:
        if boundaries.get(key) is not True:
            raise SystemExit(f"{label} mutation handoff boundary {key} should be true: {handoff}")
    for key in ["calls_model", "executes_tools", "queues_approval", "approves_request", "dismisses_request", "reads_private_data", "reads_personal_data", "executes_side_effect", "controls_computer"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} mutation handoff boundary {key} should be false: {handoff}")


def assert_task_refusal_handoff(metadata: dict, label: str, *, source: str, mutation: str, reason: str) -> None:
    handoff = metadata.get("task_refusal_handoff")
    if not isinstance(handoff, dict) or metadata.get("task_refusal_handoff_ready") is not True:
        raise SystemExit(f"{label} missed task refusal handoff: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    next_commands = handoff.get("next_commands") or {}
    if metadata.get("task_mutation_handoff_ready") is not False:
        raise SystemExit(f"{label} refusal should explicitly mark mutation handoff not ready: {metadata}")
    if handoff.get("source") != source or handoff.get("mutation") != mutation or handoff.get("reason") != reason:
        raise SystemExit(f"{label} refusal handoff source/mutation/reason diverged: {handoff}")
    assert_task_contract(
        metadata,
        handoff,
        label,
        state_changed=False,
        changed=[],
        content_in_handoff="raw_field" in handoff,
    )
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("changed") != [] or handoff.get("refused") is not True:
        raise SystemExit(f"{label} refusal handoff should report no changed fields: {handoff}")
    if handoff.get("task_id") != metadata.get("task_id"):
        raise SystemExit(f"{label} refusal handoff task id parity failed: {metadata}")
    raw_field = handoff.get("raw_field")
    if raw_field and handoff.get("raw_value") != metadata.get(f"raw_{raw_field}"):
        raise SystemExit(f"{label} refusal handoff raw value parity failed: {metadata}")
    for key in ["retry", "list_tasks", "task_board", "next_task"]:
        if key not in next_commands:
            raise SystemExit(f"{label} refusal handoff missed {key} command: {handoff}")
    for key in [
        "changes_task_status",
        "completes_task",
        "writes_files",
        "writes_memory",
        "writes_notes",
        "calls_model",
        "executes_tools",
        "queues_approval",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "executes_side_effect",
        "controls_computer",
    ]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} refusal handoff boundary {key} should be false: {handoff}")


def assert_task_search_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("task_search_handoff")
    if not isinstance(handoff, dict) or metadata.get("task_search_handoff_ready") is not True:
        raise SystemExit(f"{label} missed task search handoff: {metadata}")
    matches = handoff.get("matches") or []
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("source") != "search_tasks":
        raise SystemExit(f"{label} search handoff source diverged: {handoff}")
    assert_task_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=bool(matches))
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("query") != metadata.get("query") or handoff.get("status") != metadata.get("status"):
        raise SystemExit(f"{label} search handoff query/status parity failed: {metadata}")
    if handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} search handoff limit parity failed: {metadata}")
    if handoff.get("match_count") != metadata.get("count") or len(matches) != metadata.get("count"):
        raise SystemExit(f"{label} search handoff count parity failed: {metadata}")
    if handoff.get("matched_task_ids") != [row.get("id") for row in matches]:
        raise SystemExit(f"{label} search handoff matched ids diverged: {handoff}")
    if matches:
        first_id = matches[0].get("id")
        if handoff.get("first_match_task_id") != first_id:
            raise SystemExit(f"{label} search handoff first match diverged: {handoff}")
        if next_commands.get("show_first_match") != f"show task {first_id}":
            raise SystemExit(f"{label} search handoff missed first show command: {handoff}")
        if next_commands.get("completion_packet_first_match") != f"task completion packet {first_id}":
            raise SystemExit(f"{label} search handoff missed first completion packet command: {handoff}")
    elif handoff.get("first_match_task_id") is not None or next_commands.get("show_first_match"):
        raise SystemExit(f"{label} empty search handoff should not name a first task: {handoff}")
    for row in matches:
        if not isinstance(row.get("body_chars"), int) or row.get("body_chars") <= 0:
            raise SystemExit(f"{label} search handoff missed match body length: {handoff}")
    for key in ["changes_task_status", "completes_task", "writes_files", "writes_memory", "writes_notes", "calls_model", "executes_tools", "queues_approval", "controls_computer"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} search handoff boundary {key} should be false: {handoff}")


def assert_task_overdue_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("task_overdue_handoff")
    if not isinstance(handoff, dict) or metadata.get("task_overdue_handoff_ready") is not True:
        raise SystemExit(f"{label} missed overdue handoff: {metadata}")
    overdue = handoff.get("overdue_tasks") or []
    unparsed = handoff.get("unparsed_due_tasks") or []
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("source") != "overdue_tasks":
        raise SystemExit(f"{label} overdue handoff source diverged: {handoff}")
    assert_task_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=bool(overdue or unparsed))
    assert_no_boundary_authority(boundaries, label)
    for key in ["limit", "parsed_due_count", "overdue_count", "unparsed_due_count", "as_of"]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} overdue {key} parity failed: {metadata}")
    if handoff.get("overdue_task_ids") != [row.get("id") for row in overdue]:
        raise SystemExit(f"{label} overdue task ids diverged: {handoff}")
    if handoff.get("unparsed_due_task_ids") != [row.get("id") for row in unparsed]:
        raise SystemExit(f"{label} unparsed due task ids diverged: {handoff}")
    if overdue:
        first_id = overdue[0].get("id")
        if handoff.get("first_overdue_task_id") != first_id:
            raise SystemExit(f"{label} first overdue task diverged: {handoff}")
        if next_commands.get("show_first_overdue") != f"show task {first_id}":
            raise SystemExit(f"{label} missed show-first-overdue command: {handoff}")
        if next_commands.get("completion_packet_first_overdue") != f"task completion packet {first_id}":
            raise SystemExit(f"{label} missed completion packet command: {handoff}")
    elif handoff.get("first_overdue_task_id") is not None or next_commands.get("show_first_overdue"):
        raise SystemExit(f"{label} empty overdue list should not name a first task: {handoff}")
    for row in overdue + unparsed:
        if not isinstance(row.get("body_chars"), int) or row.get("body_chars") <= 0:
            raise SystemExit(f"{label} overdue handoff missed body length: {handoff}")
    for key in ["changes_task_status", "completes_task", "writes_files", "writes_memory", "writes_notes", "calls_model", "executes_tools", "queues_approval", "controls_computer", "reads_private_data", "reads_personal_data", "executes_side_effect"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} overdue handoff boundary {key} should be false: {handoff}")


def assert_vault_relative_receipt(result, root: Path, prefix: str, label: str) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    response = result.response if hasattr(result, "response") else result.output
    path_text = str(metadata.get("path") or "")
    path_display = metadata.get("path_display")
    receipt_line = response.split("\n", 1)[0]
    if not path_text or not Path(path_text).exists():
        raise SystemExit(f"{label} should preserve exact saved path metadata: {metadata}")
    if path_text in receipt_line:
        raise SystemExit(f"{label} should not print the raw local note path.")
    if any(fragment in receipt_line for fragment in (str(root), "/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} receipt should not expose local temp or user paths.")
    if not isinstance(path_display, str) or not path_display.startswith(prefix):
        raise SystemExit(f"{label} missed vault-relative display metadata: {metadata}")
    if path_display not in receipt_line:
        raise SystemExit(f"{label} should print the vault-relative saved-note label.")


def assert_task_export_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("task_export_handoff")
    if not isinstance(handoff, dict) or metadata.get("task_export_handoff_ready") is not True:
        raise SystemExit(f"{label} missed task export handoff: {metadata}")
    tasks = handoff.get("tasks") or []
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    path_display = str(handoff.get("path_display") or "")
    if handoff.get("source") != "export_tasks":
        raise SystemExit(f"{label} export handoff source diverged: {handoff}")
    assert_task_contract(metadata, handoff, label, state_changed=True, changed=["task_export"], content_in_handoff=bool(tasks))
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} export handoff limit parity failed: {metadata}")
    if handoff.get("exported_count") != metadata.get("exported_count") or len(tasks) != metadata.get("exported_count"):
        raise SystemExit(f"{label} export handoff count parity failed: {metadata}")
    if handoff.get("task_ids") != [row.get("id") for row in tasks]:
        raise SystemExit(f"{label} export handoff task ids diverged: {handoff}")
    if path_display != metadata.get("path_display") or any(fragment in path_display for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} export handoff leaked/missed safe path: {handoff}")
    if next_commands.get("export_again") != "export tasks" or next_commands.get("task_board") != "task board":
        raise SystemExit(f"{label} export handoff missed next commands: {handoff}")
    for row in tasks:
        if not isinstance(row.get("body_chars"), int) or row.get("body_chars") <= 0:
            raise SystemExit(f"{label} export handoff missed task body length: {handoff}")
    for key in ["exports_tasks", "writes_files", "writes_notes"]:
        if boundaries.get(key) is not True:
            raise SystemExit(f"{label} export handoff boundary {key} should be true: {handoff}")
    for key in ["writes_memory", "imports_tasks", "changes_task_status", "completes_task", "calls_model", "executes_tools", "queues_approval", "controls_computer", "reads_private_data", "reads_personal_data", "executes_side_effect"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} export handoff boundary {key} should be false: {handoff}")


def assert_planner_routes_task_phone_aliases() -> None:
    planner = RuleBasedPlanner()
    for text in (
        "show my tasks",
        "list my tasks",
        "show latest tasks",
        "show current tasks",
        "tasks please",
        "task list please",
        "what tasks do I have",
        "what are my tasks",
        "what do i need to do",
        "what do I have to do",
        "anything I need to do",
        "what's on my list",
        "what is on my list",
        "what's on my plate",
        "what is on my plate",
        "todo list",
        "todo list please",
        "todos please",
        "show my todo",
        "to do list",
        "show to do list",
        "show todos",
        "what are my todos",
        "show todo list",
        "what is on my todo list",
        "my to do list",
        # Real gap found live 2026-07-10 during a fresh WS4/DoD-A mixed-
        # conversation latency remeasurement: "list my open tasks" (and the
        # same phrasing with show/what-are/bare-my prefixes) fell through
        # every task pattern to chat -- the exact-match set had "open tasks"
        # bare but none of the show/list/my-prefixed combinations that the
        # equivalent bare "tasks" phrasing already covered. Worse than a
        # routing miss: the chat fallback then claimed "I'm not aware of any
        # open tasks" mid-conversation, directly contradicting a real
        # `list_tasks` result shown earlier in the SAME session (Count: 10).
        "list my open tasks",
        "show my open tasks",
        "Show my tasks.",
        "Show my open tasks.",
        "What are my open tasks?",
        "what are my open tasks",
        "my open tasks",
        "list open tasks",
        "show open tasks",
        "what open tasks do i have",
        "내 할 일 뭐야?",
        "내 할 일 알려줘",
        "내 할 일 알려주세요",
        "할 일 뭐야",
        "할 일 알려줘",
        "내 작업 뭐야",
        "내 작업 알려줘",
        "작업 뭐야",
    ):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("list_tasks", {"status": "open"})]:
            raise SystemExit(f"planner missed task list phone alias {text!r}: {plan.actions}")
    status_task_cases = {
        "done tasks": "done",
        "tasks done": "done",
        "show done tasks": "done",
        "completed tasks": "done",
        "list completed tasks": "done",
        "show completed todos": "done",
        "finished todo list": "done",
        "paused tasks": "paused",
        "tasks paused": "paused",
        "show paused tasks": "paused",
        "dropped tasks": "dropped",
        "tasks dropped": "dropped",
        "show dropped todos": "dropped",
    }
    for text, expected_status in status_task_cases.items():
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("list_tasks", {"status": expected_status})]:
            raise SystemExit(f"planner missed status task list alias {text!r}: {plan.actions}")
    for text in ("overdue tasks", "tasks overdue", "show overdue tasks", "what tasks are overdue", "anything overdue", "past due tasks", "tasks past due"):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("overdue_tasks", {})]:
            raise SystemExit(f"planner missed overdue task alias {text!r}: {plan.actions}")
    for text in ("complete task 1", "complete task 1 please", "finish task 1", "mark task 1 done"):
        complete_mutation = planner.plan(text)
        if [(a.tool_name, a.args) for a in complete_mutation.actions] != [("complete_task", {"task_id": 1})]:
            raise SystemExit(f"planner should preserve complete-task mutation route {text!r}: {complete_mutation.actions}")
    for text in ("add task buy milk please", "add todo buy milk", "add to do buy milk", "todo buy milk", "new todo buy milk"):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("add_task", {"body": "buy milk", "due": "", "priority": "normal"})]:
            raise SystemExit(f"planner missed todo capture alias {text!r}: {plan.actions}")
    # Real gap found live 2026-07-09: "new task buy milk" and "add a task to buy
    # milk" (the two most natural phrasings besides the bare "add task buy milk"
    # already covered above) fell through to chat because add_task_match only
    # recognized the literal verbs "add"/"create"/"capture" directly followed by
    # "task " with no article and no "to" lead-in before the body.
    for text in ("new task buy milk", "add a task to buy milk", "create a task to buy milk"):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("add_task", {"body": "buy milk", "due": "", "priority": "normal"})]:
            raise SystemExit(f"planner missed natural add-task phrasing {text!r}: {plan.actions}")
    # Real gap found live 2026-07-10: "add task buy milk and then show my
    # goals" (a compound sentence) swallowed the whole second clause into the
    # task body, writing a confusing task titled "buy milk and then show my
    # goals" instead of just "buy milk" -- a data-quality bug, not just a
    # routing one, since it silently drops the second intent and pollutes
    # the saved task content.
    compound_plan = planner.plan("add task buy milk and then show my goals")
    if [(a.tool_name, a.args) for a in compound_plan.actions] != [("add_task", {"body": "buy milk", "due": "", "priority": "normal"})]:
        raise SystemExit(f"planner should stop task body at a compound-sentence boundary: {compound_plan.actions}")
    # Real gap found live 2026-07-10, same privacy-relevant misroute class as
    # the round-21/26 notes-search word-order fixes: "search for the/a task
    # about X" / "find the/a task about X" (article BEFORE the noun) fell
    # through to the generic public web-search fallback instead of searching
    # Jarvis's own tracked tasks.
    for text, expected_query in {
        "find the task about groceries": "groceries",
        "search for the task about groceries": "groceries",
        "search for a task about groceries": "groceries",
        "search tasks for groceries": "groceries",
    }.items():
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("search_tasks", {"query": expected_query, "status": "all"})]:
            raise SystemExit(f"planner missed task-search word-order alias {text!r}: {plan.actions}")
    # Real gap found live 2026-07-10: compound sentences swallowed the whole
    # second clause into the task-search query instead of stopping at the
    # intended keyword.
    for text in ("search tasks for groceries and then show my calendar", "find the task about groceries and then call mom"):
        compound_search_plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in compound_search_plan.actions] != [("search_tasks", {"query": "groceries", "status": "all"})]:
            raise SystemExit(f"planner should stop task-search query at a compound-sentence boundary: {text!r} -> {compound_search_plan.actions}")
    for text in ("todo board", "todos board", "todo dashboard", "todo kanban", "show task board", "show tasks dashboard"):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("task_board", {})]:
            raise SystemExit(f"planner should route todo/task board alias read-only {text!r}: {plan.actions}")
    for text in (
        "todo overview",
        "todos summary",
        "todo report",
        "show task overview",
        "show tasks report",
        "prioritize my tasks",
        "prioritize my todo list",
        "task priorities",
        "todo priorities",
        "what's due",
        "what is due",
        "anything due",
    ):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("task_overview", {})]:
            raise SystemExit(f"planner should route todo/task overview alias read-only {text!r}: {plan.actions}")
    todo_due_plan = planner.plan("add todo buy milk due tomorrow priority high")
    if [(a.tool_name, a.args) for a in todo_due_plan.actions] != [
        ("add_task", {"body": "buy milk", "due": "tomorrow", "priority": "high"})
    ]:
        raise SystemExit(f"planner missed todo due/priority alias: {todo_due_plan.actions}")
    due_plan = planner.plan("tasks due today")
    if [(a.tool_name, a.args) for a in due_plan.actions] != [("search_tasks", {"query": "today", "status": "open"})]:
        raise SystemExit(f"planner missed due task phone alias: {due_plan.actions}")
    due_question_plan = planner.plan("what tasks are due tomorrow")
    if [(a.tool_name, a.args) for a in due_question_plan.actions] != [("search_tasks", {"query": "tomorrow", "status": "open"})]:
        raise SystemExit(f"planner missed due task question alias: {due_question_plan.actions}")
    shorthand_due_cases = {
        "what's due tomorrow": "tomorrow",
        "what is due tomorrow": "tomorrow",
        "anything due tomorrow": "tomorrow",
        "show due tomorrow": "tomorrow",
        "list due this week": "this week",
        "do I have anything due today?": "today",
        "do I have any tasks due tomorrow?": "tomorrow",
        "do I have anything due Friday?": "friday",
        "do I have anything due next Monday?": "next monday",
        "do I have anything due July 15?": "july 15",
        "do I have anything due 2026-07-15?": "2026-07-15",
        "do I have anything due today ?": "today",
        "do I have anything due today??": "today",
        "do I have anything due today？": "today",
        "do I have anything due today, please?": "today",
        "what’s due tomorrow?": "tomorrow",
        "do I have anything due 07/15/2026?": "07/15/2026",
        "do I have anything due 7/15?": "7/15",
        "do I have anything due 7-15-2026?": "7-15-2026",
        "do I have anything due Jul 15?": "jul 15",
        "do I have anything due Sept 15?": "sept 15",
        "do I have anything due 15 Sept?": "15 sept",
        "do I have anything due July 15th?": "july 15th",
        "do I have anything due 15 Jul?": "15 jul",
    }
    for text, expected_query in shorthand_due_cases.items():
        shorthand_due_plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in shorthand_due_plan.actions] != [("search_tasks", {"query": expected_query, "status": "open"})]:
            raise SystemExit(f"planner missed shorthand due task alias {text!r}: {shorthand_due_plan.actions}")
    for text in ("do I have anything tomorrow?", "am I free today?"):
        availability_plan = planner.plan(text)
        if [action.tool_name for action in availability_plan.actions] != ["check_availability"]:
            raise SystemExit(f"due-task routing stole calendar availability question {text!r}: {availability_plan.actions}")
    due_diligence_plan = planner.plan("what is due diligence?")
    if [action.tool_name for action in due_diligence_plan.actions] != ["wiki_summary"]:
        raise SystemExit(f"due-task routing stole encyclopedia intent: {due_diligence_plan.actions}")
    compound_due_calendar_plan = planner.plan("tasks due today, show calendar?")
    if [(action.tool_name, action.args) for action in compound_due_calendar_plan.actions] != [
        ("search_tasks", {"query": "today", "status": "open"}),
        ("list_events", {"range": "today"}),
    ]:
        raise SystemExit(f"compound due-task/calendar intent was not preserved: {compound_due_calendar_plan.actions}")
    and_then_due_calendar_plan = planner.plan("tasks due today and then show calendar")
    if [(action.tool_name, action.args) for action in and_then_due_calendar_plan.actions] != [
        ("search_tasks", {"query": "today", "status": "open"}),
        ("list_events", {"range": "today"}),
    ]:
        raise SystemExit(f"and-then due-task/calendar intent was not preserved: {and_then_due_calendar_plan.actions}")
    absolute_due_calendar_plan = planner.plan("tasks due July 15th, show calendar?")
    if [(action.tool_name, action.args) for action in absolute_due_calendar_plan.actions] != [
        ("search_tasks", {"query": "july 15th", "status": "open"}),
        ("list_events", {"range": "july 15th"}),
    ]:
        raise SystemExit(f"absolute-date due-task/calendar intent was not preserved: {absolute_due_calendar_plan.actions}")
    calendar_scoped_task_plan = planner.plan("what tasks are due tomorrow on my calendar?")
    if [(action.tool_name, action.args) for action in calendar_scoped_task_plan.actions] != [
        ("list_events", {"range": "tomorrow"})
    ]:
        raise SystemExit(f"calendar-scoped task wording should stay with calendar events: {calendar_scoped_task_plan.actions}")
    dated_task_cases = {
        "today tasks": "today",
        "today's tasks": "today",
        "tasks today": "today",
        "show today tasks": "today",
        "show tasks today": "today",
        "tomorrow tasks": "tomorrow",
        "tomorrow's tasks": "tomorrow",
        "tasks tomorrow": "tomorrow",
        "show tomorrow tasks": "tomorrow",
        "show tasks tomorrow": "tomorrow",
        "this week tasks": "this week",
        "tasks this week": "this week",
        "next week tasks": "next week",
        "tasks next week": "next week",
        "what do i need to do today": "today",
        "what do i need to do tomorrow": "tomorrow",
        "what do i have to do today": "today",
        "what do i have to do tomorrow": "tomorrow",
        "what's on my plate today": "today",
        "what is on my plate tomorrow": "tomorrow",
        "오늘 할 일": "today",
        "오늘 할 일 뭐야?": "today",
        "오늘 내 할 일 보여줘": "today",
        "오늘 작업 알려주세요": "today",
        "내일 할 일": "tomorrow",
        "내일 할 일 뭐야?": "tomorrow",
        "내일 내 할 일 알려줘": "tomorrow",
        "내일 작업 보여주세요": "tomorrow",
    }
    for text, expected_query in dated_task_cases.items():
        dated_task_plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in dated_task_plan.actions] != [("search_tasks", {"query": expected_query, "status": "open"})]:
            raise SystemExit(f"planner missed dated task alias {text!r}: {dated_task_plan.actions}")
    priority_card = planner.plan("what should i do today")
    if [(a.tool_name, a.args) for a in priority_card.actions] != [("next_action_packet", {})]:
        raise SystemExit(f"planner should preserve priority-card route for what-should-i-do: {priority_card.actions}")
    priority_task_cases = {
        "show high priority tasks": "high",
        "high priority todos": "high",
        "urgent tasks": "high",
        "show urgent todos": "high",
        "important task list": "high",
        "priority tasks": "high",
        "show priority tasks": "high",
        "tasks priority high": "high",
        "todo priority high": "high",
        "low priority tasks": "low",
        "show low priority todos": "low",
        "tasks priority low": "low",
        "normal priority tasks": "normal",
        "medium priority todos": "normal",
        "tasks priority normal": "normal",
    }
    for text, expected_query in priority_task_cases.items():
        priority_plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in priority_plan.actions] != [("search_tasks", {"query": expected_query, "status": "open"})]:
            raise SystemExit(f"planner missed priority task alias {text!r}: {priority_plan.actions}")
    for text in (
        "what is my next task",
        "next todo",
        "next thing to do",
        "show next task",
        "show my next todo",
        "what is next task",
        "what's next todo",
        "what task should i do",
        "which todo should i do",
        "what should I do first",
        "what should I tackle first",
        "what should I work on first",
        "which task should I do first",
        "which todo should I do first",
        "pick next todo",
        "pick my next task",
    ):
        next_plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in next_plan.actions] != [("next_task", {})]:
            raise SystemExit(f"planner missed next-task phone alias {text!r}: {next_plan.actions}")

    for text in (
        "오늘 할 일 추가해줘",
        "내일 작업 만들어줘",
        "내 할 일 완료해줘",
        "내 작업 삭제해줘",
        "오늘 할 일 보여줘 그리고 새 작업 추가해줘",
    ):
        plan = planner.plan(text)
        if any(action.tool_name in {"list_tasks", "search_tasks"} for action in plan.actions):
            raise SystemExit(f"Korean task read route captured a mutation or compound request: {text!r} -> {plan.actions}")


def assert_note_task_preview_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("note_task_preview_handoff")
    if not isinstance(handoff, dict) or metadata.get("note_task_preview_handoff_ready") is not True:
        raise SystemExit(f"{label} missed note task preview handoff: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    next_commands = handoff.get("next_commands") or {}
    if handoff.get("source") != "preview_tasks_from_note":
        raise SystemExit(f"{label} preview handoff source diverged: {handoff}")
    assert_task_contract(
        metadata,
        handoff,
        label,
        state_changed=False,
        changed=[],
        content_in_handoff=bool(
            handoff.get("open_checkbox_count")
            or handoff.get("completed_checkbox_count")
            or handoff.get("new_importable_count")
            or handoff.get("duplicate_count")
        ),
    )
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("open_checkbox_count") != metadata.get("open_checkboxes"):
        raise SystemExit(f"{label} preview handoff open checkbox count diverged: {handoff}")
    if handoff.get("new_importable_count") != metadata.get("new_tasks"):
        raise SystemExit(f"{label} preview handoff new task count diverged: {handoff}")
    if handoff.get("duplicate_count") != metadata.get("duplicates"):
        raise SystemExit(f"{label} preview handoff duplicate count diverged: {handoff}")
    if handoff.get("completed_checkbox_count") != metadata.get("completed_checkboxes"):
        raise SystemExit(f"{label} preview handoff completed count diverged: {handoff}")
    path_display = str(handoff.get("path_display") or "")
    if not path_display or any(fragment in path_display for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} preview handoff missed safe vault-relative path: {handoff}")
    if next_commands.get("import") != f"import tasks from note {path_display}":
        raise SystemExit(f"{label} preview handoff missed import command: {handoff}")
    for key in ["imports_tasks", "writes_files", "writes_memory", "writes_notes", "changes_task_status", "completes_task", "calls_model", "executes_tools", "queues_approval", "controls_computer"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} preview handoff boundary {key} should be false: {handoff}")


def assert_note_task_import_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("note_task_import_handoff")
    if not isinstance(handoff, dict) or metadata.get("note_task_import_handoff_ready") is not True:
        raise SystemExit(f"{label} missed note task import handoff: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    next_commands = handoff.get("next_commands") or {}
    if handoff.get("source") != "import_tasks_from_note":
        raise SystemExit(f"{label} import handoff source diverged: {handoff}")
    assert_task_contract(
        metadata,
        handoff,
        label,
        state_changed=bool(handoff.get("imported_count")),
        changed=["tasks"] if handoff.get("imported_count") else [],
        content_in_handoff=bool(handoff.get("imported_count") or handoff.get("skipped_count")),
    )
    assert_no_boundary_authority(boundaries, label)
    if handoff.get("priority") != metadata.get("priority"):
        raise SystemExit(f"{label} import handoff priority diverged: {handoff}")
    if handoff.get("imported_count") != metadata.get("imported") or handoff.get("skipped_count") != metadata.get("skipped"):
        raise SystemExit(f"{label} import handoff counts diverged: {handoff}")
    for key in ["note_path_display", "tasks_path_display"]:
        path_display = str(handoff.get(key) or "")
        if not path_display or any(fragment in path_display for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
            raise SystemExit(f"{label} import handoff leaked/missed safe path for {key}: {handoff}")
    if next_commands.get("preview_again") != f"preview tasks from note {handoff.get('note_path_display')}":
        raise SystemExit(f"{label} import handoff missed preview-again command: {handoff}")
    for key in ["imports_tasks", "writes_files", "writes_memory", "writes_notes"]:
        if boundaries.get(key) is not True:
            raise SystemExit(f"{label} import handoff boundary {key} should be true: {handoff}")
    for key in ["changes_task_status", "completes_task", "calls_model", "executes_tools", "queues_approval", "controls_computer", "reads_private_data", "reads_personal_data", "executes_side_effect"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} import handoff boundary {key} should be false: {handoff}")


def assert_runtime_routes_task_count_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-task-count-alias-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        runtime.registry.get("add_task").handler({"body": "verify task count aliases", "priority": "normal"})
        paused = runtime.registry.get("add_task").handler({"body": "paused count sample", "priority": "normal"})
        paused_id = paused.metadata.get("task_id")
        runtime.registry.get("update_task_status").handler({"task_id": paused_id, "status": "paused"})
        for text in (
            "how many tasks do i have",
            "how many tasks",
            "count tasks",
            "task count",
            "tasks count",
            "how many todos do i have",
            "todo count",
            "todos count",
            "작업 몇 개",
            "작업 개수",
            "태스크 몇 개",
            "할일 몇 개",
        ):
            result = runtime.handle(text)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [action.get("tool_name") for action in planned] != ["task_overview"]:
                raise SystemExit(f"runtime missed task-count alias {text!r}: {planned} / {result.response!r}")
            if len(result.tool_results) != 1:
                raise SystemExit(f"task-count alias should execute exactly one read-only tool for {text!r}: {result.tool_results}")
            metadata = result.tool_results[0].metadata
            if metadata.get("open") != 1 or metadata.get("paused") != 1 or metadata.get("total") != 2:
                raise SystemExit(f"task-count alias missed task overview counts for {text!r}: {metadata}")
            if "- open: 1" not in result.response or "- paused: 1" not in result.response:
                raise SystemExit(f"task-count alias should answer with task counts for {text!r}: {result.response!r}")
            assert_task_overview_handoff(metadata, text)
            assert_safe(metadata, text)
        if len(runtime.store.list_tasks(status=None, limit=100)) != 2:
            raise SystemExit("task count aliases must not create or mutate tasks.")


def assert_missing_task_id_recovery() -> None:
    with TemporaryDirectory(prefix="jarvis-missing-task-recovery-") as temp:
        runtime = make_temp_runtime(Path(temp))
        private_proof = "PRIVATE_TASK_PROOF_SHOULD_NOT_APPEAR"
        cases = (
            (
                "inspect_task",
                {"task_id": 9999},
                "task_read",
                ["list tasks", "show task <correct task id>"],
            ),
            (
                "task_completion_packet",
                {"task_id": 9999},
                "completion_read",
                ["list tasks", "show task <correct task id>", "task completion packet <correct task id>"],
            ),
            (
                "complete_task",
                {"task_id": 9999},
                "status_update",
                ["list tasks", "show task <correct task id>", "complete task <correct task id>"],
            ),
            (
                "update_task_status",
                {"task_id": 9999, "status": "paused"},
                "status_update",
                ["list tasks", "show task <correct task id>", "pause task <correct task id>"],
            ),
            (
                "update_task_details",
                {"task_id": 9999, "body": "PRIVATE_TASK_BODY_SHOULD_NOT_APPEAR"},
                "field_update",
                [
                    "list tasks",
                    "show task <correct task id>",
                    "rename task <correct task id>: <body>",
                ],
            ),
            (
                "complete_task_with_evidence",
                {"task_id": 9999, "evidence": private_proof},
                "evidence_completion",
                [
                    "list tasks",
                    "show task <correct task id>",
                    "complete task <correct task id> with evidence: <proof>",
                ],
            ),
        )
        for tool_name, args, mutation, expected_commands in cases:
            result = runtime.registry.get(tool_name).handler(args)
            if result.ok or result.metadata.get("reason") != "missing_task":
                raise SystemExit(f"{tool_name} should fail closed for a missing task: {result}")
            assert_resource_not_found_recovery(result, f"{tool_name} missing task")
            for token in ("Run `list tasks`", "refresh task IDs", "show task <correct task id>", "normal policy"):
                if token not in result.output:
                    raise SystemExit(f"{tool_name} missing-task output missed {token!r}: {result.output}")
            metadata = result.metadata
            if metadata.get("recovery_commands") != expected_commands:
                raise SystemExit(f"{tool_name} missing-task recovery order drifted: {metadata}")
            for key in ("retry_requires_task_refresh", "retry_requires_corrected_id", "recovery_commands_require_normal_policy"):
                if metadata.get(key) is not True:
                    raise SystemExit(f"{tool_name} missing-task recovery missed {key}: {metadata}")
            for key in (
                "authorizes_retry",
                "authorizes_task_mutation",
                "writes_files",
                "writes_memory",
                "writes_notes",
                "authorizes_execution",
                "authorizes_completion_claim",
            ):
                if metadata.get(key):
                    raise SystemExit(f"{tool_name} missing-task recovery unexpectedly set {key}: {metadata}")
            assert_task_refusal_handoff(
                metadata,
                f"{tool_name} missing task",
                source=tool_name,
                mutation=mutation,
                reason="missing_task",
            )
            assert_safe(metadata, f"{tool_name} missing task")
            rendered = f"{result.output}\n{metadata}"
            for forbidden in (private_proof, "PRIVATE_TASK_BODY_SHOULD_NOT_APPEAR"):
                if forbidden in rendered:
                    raise SystemExit(f"{tool_name} missing-task recovery leaked supplied content: {rendered}")


def assert_note_task_import_identity_is_complete_and_atomic() -> None:
    with TemporaryDirectory(prefix="jarvis-task-import-old-identity-") as temp:
        runtime = make_temp_runtime(Path(temp))
        old_body = "Track the oldest imported task identity"
        now = "2026-07-11T00:00:00"
        with runtime.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks(body, source, due, priority, status, created_at, updated_at)
                VALUES (?, 'seed', '', 'normal', 'done', ?, ?)
                """,
                (old_body, now, now),
            )
            conn.executemany(
                """
                INSERT INTO tasks(body, source, due, priority, status, created_at, updated_at, completed_at)
                VALUES (?, 'seed', '', 'normal', 'done', ?, ?, ?)
                """,
                [(f"newer completed filler task {index}", now, now, now) for index in range(1000)],
            )

        note_path = runtime.vault.root_path / "Projects" / "Old Task Identity.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(f"# Import\n\n- [ ] {old_body.upper()}\n", encoding="utf-8")

        preview = runtime.registry.get("preview_tasks_from_note").handler(
            {"path": "Projects/Old Task Identity"}
        )
        if (
            not preview.ok
            or preview.metadata.get("new_tasks") != 0
            or preview.metadata.get("duplicates") != 1
        ):
            raise SystemExit(f"Preview treated an older-than-1000 task as importable: {preview}")

        imported = runtime.registry.get("import_tasks_from_note").handler(
            {"path": "Projects/Old Task Identity", "priority": "high"}
        )
        if (
            not imported.ok
            or imported.metadata.get("imported") != 0
            or imported.metadata.get("skipped") != 1
        ):
            raise SystemExit(f"Import duplicated an older-than-1000 task identity: {imported}")
        with runtime.store.connect() as conn:
            matching = sum(
                normalized_task_identity(row["body"]) == normalized_task_identity(old_body)
                for row in conn.execute("SELECT body FROM tasks")
            )
        if matching != 1:
            raise SystemExit(f"Older-than-1000 task identity count diverged: {matching}")

    with TemporaryDirectory(prefix="jarvis-task-import-contention-") as temp:
        db_path = Path(temp) / "memory.db"
        first_store = MemoryStore(db_path)
        first_store.init()
        second_store = MemoryStore(db_path)
        barrier = Barrier(2)

        def contend(store: MemoryStore, body: str, status: str) -> int | None:
            barrier.wait(timeout=5)
            return store.add_task_if_identity_absent(
                TaskRecord(body=body, source="contention-smoke", status=status)
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(contend, first_store, "Atomic imported task", "open"),
                pool.submit(contend, second_store, "  ATOMIC IMPORTED TASK  ", "done"),
            ]
            results = [future.result(timeout=10) for future in futures]
        if sum(result is not None for result in results) != 1:
            raise SystemExit(f"Two-connection task identity contention did not choose one insert: {results}")
        with first_store.connect() as conn:
            rows = list(conn.execute("SELECT body FROM tasks"))
        if len(rows) != 1:
            raise SystemExit(f"Two-connection task identity contention inserted {len(rows)} rows")


def main() -> None:
    assert_note_task_import_identity_is_complete_and_atomic()
    assert_missing_task_id_recovery()
    assert_task_exact_metadata_bool()
    assert_task_malformed_handoff_flags()
    assert_task_next_commands_fail_closed()
    assert_task_malformed_refusal_flags()
    assert_planner_routes_task_phone_aliases()
    assert_runtime_routes_task_count_aliases()
    assert_task_readonly_reports_tolerate_malformed_rows()
    with TemporaryDirectory(prefix="jarvis-tasks-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        cases = [
            "add task review Jarvis approval queue due tomorrow priority high",
            "add task write one new assistant smoke test",
            "create jarvis note Projects/Task Import Smoke: # Task Import Smoke\n\n- [ ] Wire note tasks into Jarvis task tracking\n- [x] Ignore completed markdown tasks\n- [ ] Verify duplicate markdown tasks are skipped",
            "preview tasks from jarvis note Projects/Task Import Smoke",
            "import tasks from jarvis note Projects/Task Import Smoke priority high",
            "preview tasks from note Projects/Task Import Smoke",
            "import tasks from note Projects/Task Import Smoke",
            "tasks",
            "show my tasks",
            "what is on my todo list",
            "task overview",
            "task board",
            "next task",
            "show task 2",
            "task completion packet 1",
            "search tasks assistant smoke",
            "set task 2 priority high",
            "set task 2 due Friday",
            "rename task 2: write one polished assistant smoke test",
            "pause task 2",
            "reopen task 2",
            "drop task 2",
            "set task 2 to open",
            "complete task 1 with evidence: task review was verified through task detail and task overview output",
            "all tasks",
            "export tasks",
            "export state",
        ]
        for case in cases:
            result = runtime.handle(case)
            if case.startswith("preview tasks"):
                if not result.verified or "Jarvis note task preview" not in result.response:
                    raise SystemExit(f"Task preview failed: {result.response}")
                if "completed checkboxes ignored: 1" not in result.response:
                    raise SystemExit("Task preview should report completed checkboxes as ignored.")
                assert_note_task_preview_handoff(result.tool_results[0].metadata, case)
            if case.startswith("import tasks"):
                if not result.verified or "Imported" not in result.response:
                    raise SystemExit(f"Task import failed: {result.response}")
                if "Ignore completed markdown tasks" in result.response:
                    raise SystemExit("Completed markdown checkbox should not be imported.")
                assert_note_task_import_handoff(result.tool_results[0].metadata, case)
            if case == "show task 2":
                if not result.verified or "Task #2" not in result.response or "status: open" not in result.response:
                    raise SystemExit(f"Task detail command failed: {result.response}")
                metadata = result.tool_results[0].metadata
                assert_task_inspection_handoff(metadata, case)
                assert_safe(metadata, case)
            if case == "task completion packet 1":
                if not result.verified or "Jarvis task completion packet" not in result.response or "Verdict: EVIDENCE_REQUIRED" not in result.response:
                    raise SystemExit(f"Task completion packet command failed: {result.response}")
                metadata = result.tool_results[0].metadata
                if metadata.get("has_evidence") is not False or metadata.get("writes_files") is not False:
                    raise SystemExit("Task completion packet should be read-only and require evidence.")
                if metadata.get("recovery_closure_state") in {None, ""} or "recovery_closure_blocks_task_completion" not in metadata:
                    raise SystemExit(f"Task completion packet missed recovery closure metadata: {metadata}")
                if "recovery_closure_next_required_command" not in metadata:
                    raise SystemExit(f"Task completion packet missed next required recovery proof metadata: {metadata}")
                assert_recovery_closure_proof_aliases(metadata, case)
                assert_task_completion_handoff(metadata, case)
                assert_safe(metadata, case)
            if case == "tasks":
                if not result.verified or "Tasks:" not in result.response or "#1" not in result.response:
                    raise SystemExit(f"List tasks command failed: {result.response}")
                metadata = result.tool_results[0].metadata
                assert_task_list_handoff(metadata, case)
                assert_safe(metadata, case)
            if case == "task overview":
                if not result.verified or "Task overview:" not in result.response or "open: 4" not in result.response:
                    raise SystemExit(f"Task overview command failed: {result.response}")
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 8 or metadata.get("writes_files") is not False:
                    raise SystemExit("Task overview missed sanitized limit or read-only metadata.")
                assert_task_overview_handoff(metadata, case)
                assert_safe(metadata, case)
            if case == "task board":
                if not result.verified or "Task board:" not in result.response or "Open (4 shown):" not in result.response:
                    raise SystemExit(f"Task board command failed: {result.response}")
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 6 or metadata.get("reads_private_data") is not False:
                    raise SystemExit("Task board missed sanitized limit or privacy metadata.")
                assert_task_board_handoff(metadata, case)
                assert_safe(metadata, case)
            if case == "next task":
                if not result.verified or "Next task: #1 review Jarvis approval queue" not in result.response:
                    raise SystemExit(f"Next task command failed: {result.response}")
                metadata = result.tool_results[0].metadata
                assert_next_task_handoff(metadata, case)
                assert_safe(metadata, case)
            if case == "search tasks assistant smoke":
                if not result.verified or "Task search: assistant smoke" not in result.response or "write one new assistant smoke test" not in result.response:
                    raise SystemExit(f"Task search command failed: {result.response}")
                metadata = result.tool_results[0].metadata
                assert_task_search_handoff(metadata, case)
                assert_safe(metadata, case)
            if case == "set task 2 priority high" and "Updated task #2 priority" not in result.response:
                raise SystemExit("Set task priority command did not update task #2.")
            if case == "set task 2 priority high":
                assert_task_mutation_handoff(result.tool_results[0].metadata, case, mutation="field_update", changed=["priority"])
            if case == "set task 2 due Friday" and "Updated task #2 due" not in result.response:
                raise SystemExit("Set task due command did not update task #2.")
            if case == "set task 2 due Friday":
                assert_task_mutation_handoff(result.tool_results[0].metadata, case, mutation="field_update", changed=["due"])
            if case == "rename task 2: write one polished assistant smoke test" and "write one polished assistant smoke test" not in result.response:
                raise SystemExit("Rename task command did not update task #2.")
            if case == "rename task 2: write one polished assistant smoke test":
                assert_task_mutation_handoff(result.tool_results[0].metadata, case, mutation="field_update", changed=["body"])
            if case == "pause task 2" and "Paused task #2" not in result.response:
                raise SystemExit("Pause task command did not pause task #2.")
            if case == "pause task 2":
                assert_task_mutation_handoff(result.tool_results[0].metadata, case, mutation="status_update", changed=["status"])
            if case == "reopen task 2" and "Reopened task #2" not in result.response:
                raise SystemExit("Reopen task command did not reopen task #2.")
            if case == "reopen task 2":
                assert_task_mutation_handoff(result.tool_results[0].metadata, case, mutation="status_update", changed=["status"])
            if case == "drop task 2" and "Dropped task #2" not in result.response:
                raise SystemExit("Drop task command did not drop task #2.")
            if case == "drop task 2":
                assert_task_mutation_handoff(result.tool_results[0].metadata, case, mutation="status_update", changed=["status"])
            if case == "set task 2 to open" and "Reopened task #2" not in result.response:
                raise SystemExit("Set task open command did not reopen task #2.")
            if case == "set task 2 to open":
                assert_task_mutation_handoff(result.tool_results[0].metadata, case, mutation="status_update", changed=["status"])
            if case.startswith("complete task 1 with evidence"):
                if "Completed task #1 with evidence" not in result.response or "Evidence receipt" not in result.response:
                    raise SystemExit("Evidence-backed complete task command did not complete task #1.")
                metadata = result.tool_results[0].metadata
                if metadata.get("has_evidence") is not True or metadata.get("writes_files") is not True:
                    raise SystemExit("Evidence-backed complete task missed write/evidence metadata.")
                if metadata.get("recovery_closure_blocks_task_completion") is not False:
                    raise SystemExit(f"Evidence-backed complete task should not be recovery-blocked in clean flow: {metadata}")
                assert_task_mutation_handoff(metadata, case, mutation="evidence_completion", changed=["status"])
            if case == "export tasks":
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 100 or not metadata.get("writes_notes") or metadata.get("writes_memory") is not False:
                    raise SystemExit(f"Runtime export_tasks missed write metadata: {metadata}")
                assert_task_export_handoff(metadata, case)
                assert_vault_relative_receipt(result, root, "Tasks/", case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1400])
            print()

        runtime.registry.get("add_task").handler({"body": "overdue parser smoke", "due": "2000-01-01"})
        runtime.registry.get("add_task").handler({"body": "ambiguous due label smoke", "due": "Friday"})
        overdue_result = runtime.handle("overdue tasks")
        if not overdue_result.verified or "Overdue task report" not in overdue_result.response:
            raise SystemExit(f"Overdue tasks command failed: {overdue_result.response}")
        if "overdue parser smoke" not in overdue_result.response or "ambiguous due label smoke" not in overdue_result.response:
            raise SystemExit(f"Overdue tasks should report overdue and unparsed due-label examples: {overdue_result.response}")
        overdue_metadata = overdue_result.tool_results[0].metadata
        if overdue_metadata.get("overdue_count", 0) < 1 or overdue_metadata.get("unparsed_due_count", 0) < 1:
            raise SystemExit(f"Overdue tasks missed expected counts: {overdue_metadata}")
        assert_task_overdue_handoff(overdue_metadata, "overdue tasks")
        assert_safe(overdue_metadata, "overdue tasks")
        print("[ok] overdue tasks")
        print(overdue_result.response[:1400])
        print()

        direct_list = runtime.registry.get("list_tasks").handler({"limit": "bad"})
        if not direct_list.ok or direct_list.metadata.get("limit") != 25 or direct_list.metadata.get("raw_limit") != "bad":
            raise SystemExit("list_tasks did not sanitize a bad limit.")
        if direct_list.metadata.get("writes_files") is not False or direct_list.metadata.get("reads_private_data") is not False:
            raise SystemExit("list_tasks missed read-only safety metadata.")
        assert_task_list_handoff(direct_list.metadata, "direct list tasks")
        assert_safe(direct_list.metadata, "direct list tasks")
        empty_paused_list = runtime.registry.get("list_tasks").handler({"status": "paused", "limit": 5})
        if not empty_paused_list.ok or empty_paused_list.metadata.get("count") != 0:
            raise SystemExit(f"empty paused task list should succeed with zero tasks: {empty_paused_list.metadata}")
        assert_task_list_handoff(empty_paused_list.metadata, "empty paused task list")
        assert_safe(empty_paused_list.metadata, "empty paused task list")
        bool_list = runtime.registry.get("list_tasks").handler({"limit": False})
        if not bool_list.ok or bool_list.metadata.get("limit") != 25 or bool_list.metadata.get("raw_limit") != "False":
            raise SystemExit(f"list_tasks should treat boolean limits as malformed and preserve raw metadata: {bool_list.metadata}")
        direct_list_long_limit = runtime.registry.get("list_tasks").handler({"limit": "l" * 200})
        if direct_list_long_limit.metadata.get("raw_limit") != ("l" * 79 + "…"):
            raise SystemExit(f"list_tasks did not bound raw bad limit metadata: {direct_list_long_limit.metadata}")
        for path_limit in (
            "/\x55sers/example/private/task-list-limit",
            "/var/folders/zc/jarvis/task-list-limit",
            "/tmp/jarvis/task-list-limit",
        ):
            path_bad_list = runtime.registry.get("list_tasks").handler({"limit": path_limit})
            if path_bad_list.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"list_tasks leaked local path in raw limit metadata: {path_bad_list.metadata}")
        bad_list_status = runtime.registry.get("list_tasks").handler({"status": "archived"})
        if bad_list_status.ok or bad_list_status.metadata.get("reason") != "bad_status" or bad_list_status.metadata.get("raw_status") != "archived":
            raise SystemExit(f"list_tasks bad status should preserve bounded raw status: {bad_list_status.metadata}")
        assert_safe(bad_list_status.metadata, "bad list task status")
        for path_status in (
            "/private/tmp/jarvis-task-status",
            "/var/folders/zc/jarvis/task-status",
            "/tmp/jarvis/task-status",
        ):
            path_bad_list_status = runtime.registry.get("list_tasks").handler({"status": path_status})
            if path_bad_list_status.metadata.get("raw_status") != "<local-path>":
                raise SystemExit(f"list_tasks leaked local path in raw status metadata: {path_bad_list_status.metadata}")

        direct_search = runtime.registry.get("search_tasks").handler({"query": "Jarvis", "limit": "bad"})
        if not direct_search.ok or direct_search.metadata.get("limit") != 25 or direct_search.metadata.get("raw_limit") != "bad":
            raise SystemExit("search_tasks did not sanitize a bad limit.")
        assert_safe(direct_search.metadata, "direct search tasks")
        bool_search = runtime.registry.get("search_tasks").handler({"query": "Jarvis", "limit": True})
        if not bool_search.ok or bool_search.metadata.get("limit") != 25 or bool_search.metadata.get("raw_limit") != "True":
            raise SystemExit(f"search_tasks should treat boolean limits as malformed and preserve raw metadata: {bool_search.metadata}")
        direct_search_large = runtime.registry.get("search_tasks").handler({"query": "Jarvis", "limit": 999999})
        if not direct_search_large.ok or direct_search_large.metadata.get("limit") != 200:
            raise SystemExit("search_tasks did not clamp a large limit.")
        assert_safe(direct_search_large.metadata, "direct search tasks large")
        for path_limit in (
            "/private/tmp/jarvis-task-search-limit",
            "/var/folders/zc/jarvis/task-search-limit",
            "/tmp/jarvis/task-search-limit",
        ):
            path_bad_search = runtime.registry.get("search_tasks").handler({"query": "Jarvis", "limit": path_limit})
            if path_bad_search.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"search_tasks leaked local path in raw limit metadata: {path_bad_search.metadata}")
        bad_search_status = runtime.registry.get("search_tasks").handler({"query": "Jarvis", "status": "archived"})
        if bad_search_status.ok or bad_search_status.metadata.get("raw_status") != "archived":
            raise SystemExit(f"search_tasks bad status should preserve bounded raw status: {bad_search_status.metadata}")
        assert_safe(bad_search_status.metadata, "bad search task status")
        for path_status in (
            "/\x55sers/example/private/search-status",
            "/var/folders/zc/jarvis/search-status",
            "/tmp/jarvis/search-status",
        ):
            path_bad_search_status = runtime.registry.get("search_tasks").handler({"query": "Jarvis", "status": path_status})
            if path_bad_search_status.metadata.get("raw_status") != "<local-path>":
                raise SystemExit(f"search_tasks leaked local path in raw status metadata: {path_bad_search_status.metadata}")
        missing_query = runtime.registry.get("search_tasks").handler({"query": ""})
        if missing_query.ok or missing_query.metadata.get("reason") != "missing_query" or missing_query.metadata.get("raw_query") != "":
            raise SystemExit(f"search_tasks missing query should preserve bounded raw query: {missing_query.metadata}")
        assert_safe(missing_query.metadata, "missing task search query")
        long_search = runtime.registry.get("search_tasks").handler({"query": "Jarvis " * 200, "limit": 2})
        if len(long_search.metadata.get("query", "")) > 260:
            raise SystemExit("search_tasks did not bound long queries.")

        direct_overview = runtime.registry.get("task_overview").handler({"limit": "bad"})
        if not direct_overview.ok or direct_overview.metadata.get("limit") != 8 or direct_overview.metadata.get("raw_limit") != "bad":
            raise SystemExit("task_overview did not sanitize a bad limit.")
        assert_task_overview_handoff(direct_overview.metadata, "direct task overview")
        assert_safe(direct_overview.metadata, "direct task overview")
        bool_overview = runtime.registry.get("task_overview").handler({"limit": False})
        if not bool_overview.ok or bool_overview.metadata.get("limit") != 8 or bool_overview.metadata.get("raw_limit") != "False":
            raise SystemExit(f"task_overview should treat boolean limits as malformed and preserve raw metadata: {bool_overview.metadata}")
        for path_limit in (
            "/\x55sers/example/private/task-overview-limit",
            "/var/folders/zc/jarvis/task-overview-limit",
            "/tmp/jarvis/task-overview-limit",
        ):
            path_bad_overview = runtime.registry.get("task_overview").handler({"limit": path_limit})
            if path_bad_overview.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"task_overview leaked local path in raw limit metadata: {path_bad_overview.metadata}")
        direct_overview_low = runtime.registry.get("task_overview").handler({"limit": -1})
        if not direct_overview_low.ok or direct_overview_low.metadata.get("limit") != 1:
            raise SystemExit("task_overview did not clamp a low limit.")
        assert_task_overview_handoff(direct_overview_low.metadata, "direct task overview low")
        assert_safe(direct_overview_low.metadata, "direct task overview low")

        direct_board = runtime.registry.get("task_board").handler({"limit": "bad"})
        if not direct_board.ok or direct_board.metadata.get("limit") != 6 or direct_board.metadata.get("raw_limit") != "bad":
            raise SystemExit("task_board did not sanitize a bad limit.")
        assert_task_board_handoff(direct_board.metadata, "direct task board")
        assert_safe(direct_board.metadata, "direct task board")
        bool_board = runtime.registry.get("task_board").handler({"limit": True})
        if not bool_board.ok or bool_board.metadata.get("limit") != 6 or bool_board.metadata.get("raw_limit") != "True":
            raise SystemExit(f"task_board should treat boolean limits as malformed and preserve raw metadata: {bool_board.metadata}")
        for path_limit in (
            "/private/tmp/jarvis-task-board-limit",
            "/var/folders/zc/jarvis/task-board-limit",
            "/tmp/jarvis/task-board-limit",
        ):
            path_bad_board = runtime.registry.get("task_board").handler({"limit": path_limit})
            if path_bad_board.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"task_board leaked local path in raw limit metadata: {path_bad_board.metadata}")

        direct_export = runtime.registry.get("export_tasks").handler({"limit": "bad"})
        if not direct_export.ok or direct_export.metadata.get("limit") != 100 or direct_export.metadata.get("raw_limit") != "bad":
            raise SystemExit("export_tasks did not sanitize a bad limit.")
        assert_task_export_handoff(direct_export.metadata, "direct export_tasks")
        canonical_export = runtime.registry.get("export_tasks").handler({"limit": 1})
        if canonical_export.metadata.get("exported_count") != 1:
            raise SystemExit(f"export_tasks request metadata should preserve its requested subset: {canonical_export.metadata}")
        canonical_note = (runtime.vault.root_path / "Tasks" / "Open Tasks.md").read_text(encoding="utf-8")
        if canonical_note.count("- [ ] #") <= 1:
            raise SystemExit(f"A narrow export_tasks request truncated the canonical shared mirror:\n{canonical_note}")
        assert_task_export_handoff(canonical_export.metadata, "canonical export_tasks")
        bool_export = runtime.registry.get("export_tasks").handler({"limit": False})
        if not bool_export.ok or bool_export.metadata.get("limit") != 100 or bool_export.metadata.get("raw_limit") != "False":
            raise SystemExit(f"export_tasks should treat boolean limits as malformed and preserve raw metadata: {bool_export.metadata}")
        assert_task_export_handoff(bool_export.metadata, "bool export_tasks")
        for path_limit in (
            "/\x55sers/example/private/export-limit",
            "/var/folders/zc/jarvis/export-limit",
            "/tmp/jarvis/export-limit",
        ):
            path_bad_export = runtime.registry.get("export_tasks").handler({"limit": path_limit})
            if path_bad_export.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"export_tasks leaked local path in raw limit metadata: {path_bad_export.metadata}")
            assert_task_export_handoff(path_bad_export.metadata, "path export_tasks")
        direct_export_large = runtime.registry.get("export_tasks").handler({"limit": 999999})
        if not direct_export_large.ok or direct_export_large.metadata.get("limit") != 200:
            raise SystemExit("export_tasks did not clamp a large limit.")
        assert_task_export_handoff(direct_export_large.metadata, "direct export_tasks large")
        if direct_export.metadata.get("writes_files") is not True or direct_export.metadata.get("controls_computer") is not False:
            raise SystemExit("export_tasks missed safety metadata.")
        if not direct_export.metadata.get("writes_notes") or direct_export.metadata.get("writes_memory") is not False:
            raise SystemExit("export_tasks exposed incorrect note/memory write metadata.")
        assert_vault_relative_receipt(direct_export, root, "Tasks/", "direct export_tasks")

        missing_body = runtime.registry.get("add_task").handler({"body": ""})
        if missing_body.ok or missing_body.metadata.get("reason") != "missing_body" or missing_body.metadata.get("raw_body") != "":
            raise SystemExit(f"add_task missing body should preserve bounded raw body metadata: {missing_body.metadata}")
        assert_task_refusal_handoff(missing_body.metadata, "missing task body", source="add_task", mutation="task_create", reason="missing_body")
        assert_safe(missing_body.metadata, "missing task body")

        bad_priority = runtime.registry.get("add_task").handler({"body": "bad priority task", "priority": "urgent"})
        if bad_priority.ok or bad_priority.metadata.get("raw_priority") != "urgent":
            raise SystemExit(f"add_task bad priority should preserve bounded raw priority metadata: {bad_priority.metadata}")
        assert_task_refusal_handoff(bad_priority.metadata, "bad task priority", source="add_task", mutation="task_create", reason="bad_priority")
        assert_safe(bad_priority.metadata, "bad task priority")
        for path_priority in (
            "/private/tmp/jarvis-task-priority",
            "/var/folders/zc/jarvis/task-priority",
            "/tmp/jarvis/task-priority",
        ):
            path_bad_priority = runtime.registry.get("add_task").handler({"body": "bad priority task", "priority": path_priority})
            if path_bad_priority.metadata.get("raw_priority") != "<local-path>":
                raise SystemExit(f"add_task leaked local path in raw priority metadata: {path_bad_priority.metadata}")
            assert_task_refusal_handoff(path_bad_priority.metadata, "path bad task priority", source="add_task", mutation="task_create", reason="bad_priority")

        long_add = runtime.registry.get("add_task").handler({"body": "Jarvis task bounds " * 100, "priority": "high"})
        if not long_add.ok or long_add.metadata.get("body_chars", 9999) > 620:
            raise SystemExit("add_task did not bound long task bodies.")
        if not long_add.metadata.get("writes_notes") or not long_add.metadata.get("writes_memory"):
            raise SystemExit("add_task missed task write metadata.")
        assert_task_mutation_handoff(long_add.metadata, "long add task", mutation="task_create", changed=["task"])

        direct_update = runtime.registry.get("update_task_details").handler({"task_id": 3, "due": "next sprint", "priority": "low"})
        if not direct_update.ok or direct_update.metadata.get("changed") != ["due", "priority"]:
            raise SystemExit(f"direct update_task_details should update due and priority: {direct_update.metadata}")
        assert_task_mutation_handoff(direct_update.metadata, "direct update task details", mutation="field_update", changed=["due", "priority"])

        direct_status = runtime.registry.get("update_task_status").handler({"task_id": 3, "status": "paused"})
        if not direct_status.ok or direct_status.metadata.get("status") != "paused":
            raise SystemExit(f"direct update_task_status should pause task: {direct_status.metadata}")
        assert_task_mutation_handoff(direct_status.metadata, "direct update task status", mutation="status_update", changed=["status"])
        direct_reopen = runtime.registry.get("update_task_status").handler({"task_id": 3, "status": "open"})
        if not direct_reopen.ok or direct_reopen.metadata.get("status") != "open":
            raise SystemExit(f"direct update_task_status should reopen task: {direct_reopen.metadata}")
        assert_task_mutation_handoff(direct_reopen.metadata, "direct reopen task status", mutation="status_update", changed=["status"])

        bad_update = runtime.registry.get("update_task_details").handler({"task_id": "bad", "body": "x"})
        if bad_update.ok:
            raise SystemExit("update_task_details should reject bad task ids.")
        if bad_update.metadata.get("raw_task_id") != "bad":
            raise SystemExit(f"update_task_details should preserve bounded bad task id metadata: {bad_update.metadata}")
        assert_task_refusal_handoff(bad_update.metadata, "bad update task", source="update_task_details", mutation="field_update", reason="bad_task_id")
        assert_safe(bad_update.metadata, "bad update task")
        bool_update = runtime.registry.get("update_task_details").handler({"task_id": True, "body": "x"})
        if bool_update.ok or bool_update.metadata.get("raw_task_id") != "True":
            raise SystemExit(f"update_task_details should reject boolean task ids before mutation: {bool_update.metadata}")
        assert_task_refusal_handoff(bool_update.metadata, "bool update task", source="update_task_details", mutation="field_update", reason="bad_task_id")
        assert_safe(bool_update.metadata, "bool update task")
        for path_id in (
            "/\x55sers/example/private/task-id",
            "/var/folders/zc/jarvis/task-id",
            "/tmp/jarvis/task-id",
        ):
            path_bad_update = runtime.registry.get("update_task_details").handler({"task_id": path_id, "body": "x"})
            if path_bad_update.metadata.get("raw_task_id") != "<local-path>":
                raise SystemExit(f"update_task_details leaked local path in raw task id metadata: {path_bad_update.metadata}")
            assert_task_refusal_handoff(path_bad_update.metadata, "path bad update task", source="update_task_details", mutation="field_update", reason="bad_task_id")

        bad_inspect = runtime.registry.get("inspect_task").handler({"task_id": "task-abc"})
        if bad_inspect.ok or bad_inspect.metadata.get("raw_task_id") != "task-abc":
            raise SystemExit(f"inspect_task should preserve bounded bad task id metadata: {bad_inspect.metadata}")
        assert_safe(bad_inspect.metadata, "bad inspect task")
        bool_inspect = runtime.registry.get("inspect_task").handler({"task_id": True})
        if bool_inspect.ok or bool_inspect.metadata.get("raw_task_id") != "True":
            raise SystemExit(f"inspect_task should reject boolean task ids before lookup: {bool_inspect.metadata}")
        assert_safe(bool_inspect.metadata, "bool inspect task")

        zero_inspect = runtime.registry.get("inspect_task").handler({"task_id": 0})
        if zero_inspect.ok or "positive number" not in zero_inspect.output:
            raise SystemExit("inspect_task should reject zero task ids before lookup.")
        if zero_inspect.metadata.get("raw_task_id") != "0" or zero_inspect.metadata.get("task_id") is not None:
            raise SystemExit(f"inspect_task should preserve explicit zero id metadata: {zero_inspect.metadata}")
        assert_safe(zero_inspect.metadata, "zero inspect task")

        long_bad_update = runtime.registry.get("update_task_status").handler({"task_id": "t" * 200, "status": "done"})
        if long_bad_update.ok or long_bad_update.metadata.get("raw_task_id") != ("t" * 79 + "…"):
            raise SystemExit(f"update_task_status should bound bad task id metadata: {long_bad_update.metadata}")
        assert_task_refusal_handoff(long_bad_update.metadata, "bad update task status", source="update_task_status", mutation="status_update", reason="bad_task_id")
        assert_safe(long_bad_update.metadata, "bad update task status")
        bool_status_update = runtime.registry.get("update_task_status").handler({"task_id": True, "status": "done"})
        if bool_status_update.ok or bool_status_update.metadata.get("raw_task_id") != "True":
            raise SystemExit(f"update_task_status should reject boolean task ids before mutation: {bool_status_update.metadata}")
        assert_task_refusal_handoff(bool_status_update.metadata, "bool update task status", source="update_task_status", mutation="status_update", reason="bad_task_id")
        assert_safe(bool_status_update.metadata, "bool update task status")

        negative_update = runtime.registry.get("update_task_status").handler({"task_id": -1, "status": "done"})
        if negative_update.ok or "positive number" not in negative_update.output:
            raise SystemExit("update_task_status should reject negative task ids before mutation.")
        if negative_update.metadata.get("raw_task_id") != "-1" or negative_update.metadata.get("task_id") is not None:
            raise SystemExit(f"update_task_status should preserve negative id metadata: {negative_update.metadata}")
        if negative_update.metadata.get("writes_files") or negative_update.metadata.get("writes_memory") or negative_update.metadata.get("writes_notes"):
            raise SystemExit(f"update_task_status negative id path should not write: {negative_update.metadata}")
        assert_task_refusal_handoff(negative_update.metadata, "negative update task status", source="update_task_status", mutation="status_update", reason="bad_task_id")
        assert_safe(negative_update.metadata, "negative update task status")

        bad_task_status = runtime.registry.get("update_task_status").handler({"task_id": 2, "status": "archived"})
        if bad_task_status.ok or bad_task_status.metadata.get("reason") != "bad_status" or bad_task_status.metadata.get("raw_status") != "archived":
            raise SystemExit(f"update_task_status bad status should preserve bounded raw status: {bad_task_status.metadata}")
        assert_task_refusal_handoff(bad_task_status.metadata, "bad task status", source="update_task_status", mutation="status_update", reason="bad_status")
        assert_safe(bad_task_status.metadata, "bad task status")
        for path_status in (
            "/\x55sers/example/private/update-status",
            "/var/folders/zc/jarvis/update-status",
            "/tmp/jarvis/update-status",
        ):
            path_bad_task_status = runtime.registry.get("update_task_status").handler({"task_id": 2, "status": path_status})
            if path_bad_task_status.metadata.get("raw_status") != "<local-path>":
                raise SystemExit(f"update_task_status leaked local path in raw status metadata: {path_bad_task_status.metadata}")
            assert_task_refusal_handoff(path_bad_task_status.metadata, "path bad task status", source="update_task_status", mutation="status_update", reason="bad_status")

        bad_complete = runtime.registry.get("complete_task").handler({"task_id": "task-abc"})
        if bad_complete.ok or bad_complete.metadata.get("raw_task_id") != "task-abc":
            raise SystemExit(f"complete_task should preserve bounded bad task id metadata: {bad_complete.metadata}")
        assert_task_refusal_handoff(bad_complete.metadata, "bad complete task", source="complete_task", mutation="status_update", reason="bad_task_id")
        assert_safe(bad_complete.metadata, "bad complete task")
        bool_complete = runtime.registry.get("complete_task").handler({"task_id": True})
        if bool_complete.ok or bool_complete.metadata.get("raw_task_id") != "True":
            raise SystemExit(f"complete_task should reject boolean task ids before mutation: {bool_complete.metadata}")
        assert_task_refusal_handoff(bool_complete.metadata, "bool complete task", source="complete_task", mutation="status_update", reason="bad_task_id")
        assert_safe(bool_complete.metadata, "bool complete task")

        missing_detail_body = runtime.registry.get("update_task_details").handler({"task_id": 2, "body": ""})
        if missing_detail_body.ok or missing_detail_body.metadata.get("reason") != "missing_body":
            raise SystemExit(f"update_task_details missing body should refuse locally: {missing_detail_body.metadata}")
        assert_task_refusal_handoff(missing_detail_body.metadata, "missing update task body", source="update_task_details", mutation="field_update", reason="missing_body")
        assert_safe(missing_detail_body.metadata, "missing update task body")

        bad_detail_priority = runtime.registry.get("update_task_details").handler({"task_id": 2, "priority": "/private/tmp/task-priority"})
        if bad_detail_priority.ok or bad_detail_priority.metadata.get("raw_priority") != "<local-path>":
            raise SystemExit(f"update_task_details bad priority should redact raw priority: {bad_detail_priority.metadata}")
        assert_task_refusal_handoff(bad_detail_priority.metadata, "bad update task priority", source="update_task_details", mutation="field_update", reason="bad_priority")
        assert_safe(bad_detail_priority.metadata, "bad update task priority")

        missing_update_fields = runtime.registry.get("update_task_details").handler({"task_id": 2})
        if missing_update_fields.ok or missing_update_fields.metadata.get("reason") != "missing_update":
            raise SystemExit(f"update_task_details missing fields should refuse locally: {missing_update_fields.metadata}")
        assert_task_refusal_handoff(missing_update_fields.metadata, "missing update task fields", source="update_task_details", mutation="field_update", reason="missing_update")
        assert_safe(missing_update_fields.metadata, "missing update task fields")

        missing_evidence = runtime.registry.get("complete_task_with_evidence").handler({"task_id": 2})
        if missing_evidence.ok or missing_evidence.metadata.get("reason") != "missing_evidence":
            raise SystemExit(f"complete_task_with_evidence missing evidence should refuse locally: {missing_evidence.metadata}")
        assert_task_refusal_handoff(missing_evidence.metadata, "missing completion evidence", source="complete_task_with_evidence", mutation="evidence_completion", reason="missing_evidence")
        assert_safe(missing_evidence.metadata, "missing completion evidence")

        bad_packet = runtime.registry.get("task_completion_packet").handler({"task_id": "task-abc"})
        if bad_packet.ok or bad_packet.metadata.get("raw_task_id") != "task-abc":
            raise SystemExit(f"task_completion_packet should preserve bounded bad task id metadata: {bad_packet.metadata}")
        assert_safe(bad_packet.metadata, "bad task completion packet")
        bool_packet = runtime.registry.get("task_completion_packet").handler({"task_id": True})
        if bool_packet.ok or bool_packet.metadata.get("raw_task_id") != "True":
            raise SystemExit(f"task_completion_packet should reject boolean task ids before reading audit context: {bool_packet.metadata}")
        assert_safe(bool_packet.metadata, "bool task completion packet")

        zero_packet = runtime.registry.get("task_completion_packet").handler({"task_id": 0, "evidence": "proof"})
        if zero_packet.ok or "positive number" not in zero_packet.output:
            raise SystemExit("task_completion_packet should reject zero task ids before reading audit context.")
        if zero_packet.metadata.get("raw_task_id") != "0" or zero_packet.metadata.get("task_id") is not None:
            raise SystemExit(f"task_completion_packet should preserve zero id metadata: {zero_packet.metadata}")
        assert_safe(zero_packet.metadata, "zero task completion packet")

        bad_evidence_complete = runtime.registry.get("complete_task_with_evidence").handler({"task_id": "task-abc", "evidence": "proof"})
        if bad_evidence_complete.ok or bad_evidence_complete.metadata.get("raw_task_id") != "task-abc":
            raise SystemExit(f"complete_task_with_evidence should preserve bounded bad task id metadata: {bad_evidence_complete.metadata}")
        assert_safe(bad_evidence_complete.metadata, "bad evidence complete task")
        bool_evidence_complete = runtime.registry.get("complete_task_with_evidence").handler({"task_id": True, "evidence": "proof"})
        if bool_evidence_complete.ok or bool_evidence_complete.metadata.get("raw_task_id") != "True":
            raise SystemExit(f"complete_task_with_evidence should reject boolean task ids before mutation: {bool_evidence_complete.metadata}")
        assert_safe(bool_evidence_complete.metadata, "bool evidence complete task")

        zero_evidence_complete = runtime.registry.get("complete_task_with_evidence").handler({"task_id": 0, "evidence": "proof"})
        if zero_evidence_complete.ok or "positive number" not in zero_evidence_complete.output:
            raise SystemExit("complete_task_with_evidence should reject zero task ids before mutation.")
        if zero_evidence_complete.metadata.get("raw_task_id") != "0" or zero_evidence_complete.metadata.get("task_id") is not None:
            raise SystemExit(f"complete_task_with_evidence should preserve zero id metadata: {zero_evidence_complete.metadata}")
        if zero_evidence_complete.metadata.get("writes_files") or zero_evidence_complete.metadata.get("writes_memory") or zero_evidence_complete.metadata.get("writes_notes"):
            raise SystemExit(f"complete_task_with_evidence zero id path should not write: {zero_evidence_complete.metadata}")
        assert_safe(zero_evidence_complete.metadata, "zero evidence complete task")

        no_evidence_complete = runtime.registry.get("complete_task_with_evidence").handler({"task_id": 3})
        if no_evidence_complete.ok or no_evidence_complete.metadata.get("has_evidence") is not False:
            raise SystemExit("complete_task_with_evidence should reject missing evidence.")
        assert_safe(no_evidence_complete.metadata, "complete task without evidence")

        empty_search = runtime.registry.get("search_tasks").handler({"query": "definitely absent task", "status": "all", "limit": 7})
        if not empty_search.ok or empty_search.metadata.get("count") != 0:
            raise SystemExit(f"empty task search should succeed with zero matches: {empty_search.metadata}")
        assert_task_search_handoff(empty_search.metadata, "empty task search")
        assert_safe(empty_search.metadata, "empty task search")

        packet_with_evidence = runtime.registry.get("task_completion_packet").handler({"task_id": 3, "evidence": "verified by task board output"})
        if not packet_with_evidence.ok or packet_with_evidence.metadata.get("verdict") != "READY_TO_COMPLETE":
            raise SystemExit("task_completion_packet should report READY_TO_COMPLETE when evidence is supplied.")
        if packet_with_evidence.metadata.get("recovery_closure_blocks_task_completion") is not False:
            raise SystemExit(f"task_completion_packet should not block clean evidence: {packet_with_evidence.metadata}")
        if packet_with_evidence.metadata.get("recovery_closure_next_required_command") not in {"", None}:
            raise SystemExit(f"task_completion_packet should not require recovery proof in clean evidence flow: {packet_with_evidence.metadata}")
        assert_task_completion_handoff(packet_with_evidence.metadata, "task packet with evidence")
        assert_safe(packet_with_evidence.metadata, "task packet with evidence")

        blocked_run = runtime.handle("run command python3 --version")
        if blocked_run.verified or "approval #" not in blocked_run.response:
            raise SystemExit(f"Risky command should queue approval before blocked task completion test: {blocked_run.response}")
        blocked_task = runtime.registry.get("add_task").handler({"body": "close only after recovery closure", "priority": "high"})
        blocked_task_id = blocked_task.metadata.get("task_id")
        blocked_packet = runtime.registry.get("task_completion_packet").handler({"task_id": blocked_task_id, "evidence": "manual evidence is not enough"})
        if not blocked_packet.ok or blocked_packet.metadata.get("verdict") != "RECOVERY_CLOSURE_REQUIRED":
            raise SystemExit(f"task_completion_packet should require recovery closure after a blocked run: {blocked_packet.metadata}")
        if blocked_packet.metadata.get("recovery_closure_blocks_task_completion") is not True:
            raise SystemExit(f"task_completion_packet missed recovery closure blocker: {blocked_packet.metadata}")
        if not blocked_packet.metadata.get("recovery_closure_required_commands"):
            raise SystemExit(f"task_completion_packet missed recovery closure commands: {blocked_packet.metadata}")
        if blocked_packet.metadata.get("recovery_closure_next_required_command") != blocked_packet.metadata["recovery_closure_required_commands"][0]:
            raise SystemExit(f"task_completion_packet missed first recovery proof command: {blocked_packet.metadata}")
        assert_recovery_closure_proof_aliases(blocked_packet.metadata, "blocked task packet")
        assert_task_completion_handoff(blocked_packet.metadata, "blocked task packet")
        if blocked_packet.metadata.get("recent_approval_held_tool_runs") < 1:
            raise SystemExit(f"task_completion_packet missed approval-held audit count: {blocked_packet.metadata}")
        if blocked_packet.metadata.get("recent_failed_tool_runs") != 0:
            raise SystemExit(f"task_completion_packet should not count approval-held rows as failed: {blocked_packet.metadata}")
        if "approval held" not in blocked_packet.output:
            raise SystemExit(f"task_completion_packet should render approval-held audit rows honestly: {blocked_packet.output}")
        if "failed/blocked" in blocked_packet.output:
            raise SystemExit(f"task_completion_packet mislabeled approval-held audit row as failed/blocked: {blocked_packet.output}")
        for command in ("approval readiness 1", "approval packet 1", "approval chain proof 1"):
            if command not in blocked_packet.output:
                raise SystemExit(f"task_completion_packet missed approval review command {command!r}: {blocked_packet.output}")
        if "next required:" not in blocked_packet.output or "checklist overview" not in blocked_packet.output:
            raise SystemExit(f"task_completion_packet should name the next required recovery command in output: {blocked_packet.output}")
        if "next required proof" in blocked_packet.output:
            raise SystemExit(f"task_completion_packet should not use ambiguous next required proof prose: {blocked_packet.output}")
        blocked_complete = runtime.registry.get("complete_task_with_evidence").handler({"task_id": blocked_task_id, "evidence": "manual evidence is not enough"})
        if blocked_complete.ok or blocked_complete.metadata.get("recovery_closure_blocks_task_completion") is not True:
            raise SystemExit(f"complete_task_with_evidence should reject unresolved recovery closure: {blocked_complete.metadata}")
        if blocked_complete.metadata.get("recovery_closure_next_required_command") != blocked_complete.metadata.get("recovery_closure_required_commands", [""])[0]:
            raise SystemExit(f"complete_task_with_evidence missed first recovery proof command: {blocked_complete.metadata}")
        assert_recovery_closure_proof_aliases(blocked_complete.metadata, "blocked complete task")
        if blocked_complete.metadata.get("writes_files") is not False or blocked_complete.metadata.get("writes_memory") is not False:
            raise SystemExit(f"blocked complete_task_with_evidence should remain read-only: {blocked_complete.metadata}")
        assert_safe(blocked_packet.metadata, "blocked task packet")
        assert_safe(blocked_complete.metadata, "blocked complete task")

        bad_preview = runtime.registry.get("preview_tasks_from_note").handler({"path": "../outside.md"})
        if bad_preview.ok or bad_preview.metadata.get("queues_approval"):
            raise SystemExit("preview_tasks_from_note should reject unsafe relative paths without approval.")
        missing_note_cases = [
            (
                runtime.registry.get("preview_tasks_from_note").handler({"path": "Projects/Missing Tasks"}),
                "missing task preview note",
                ["list jarvis notes", "preview tasks from note Projects/Missing Tasks"],
            ),
            (
                runtime.registry.get("import_tasks_from_note").handler(
                    {"path": "Projects/Missing Tasks", "priority": "high"}
                ),
                "missing task import note",
                ["list jarvis notes", "import tasks from note Projects/Missing Tasks"],
            ),
        ]
        for missing_result, label, expected_commands in missing_note_cases:
            if missing_result.ok or missing_result.metadata.get("reason") != "note_not_found":
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
                raise SystemExit(f"{label} should require a note-list refresh: {missing_result.metadata}")
            if missing_result.metadata.get("authorizes_retry") is not False:
                raise SystemExit(f"{label} should not authorize retry: {missing_result.metadata}")
            assert_safe(missing_result.metadata, label)
        for tool_name, args in [
            ("preview_tasks_from_note", {"path": "/\x55sers/example/private/task-note"}),
            ("preview_tasks_from_note", {"path": "/var/folders/zc/jarvis/task-note"}),
            ("preview_tasks_from_note", {"path": "/tmp/jarvis/task-note"}),
            ("import_tasks_from_note", {"path": "/private/tmp/jarvis-task-note", "priority": "high"}),
            ("import_tasks_from_note", {"path": "/var/folders/zc/jarvis/task-note", "priority": "high"}),
            ("import_tasks_from_note", {"path": "/tmp/jarvis/task-note", "priority": "high"}),
            ("import_tasks_from_note", {"path": "/\x55sers/example/private/task-note-priority", "priority": "urgent"}),
            ("import_tasks_from_note", {"path": "/var/folders/zc/jarvis/task-note-priority", "priority": "urgent"}),
            ("import_tasks_from_note", {"path": "/tmp/jarvis/task-note-priority", "priority": "urgent"}),
        ]:
            result = runtime.registry.get(tool_name).handler(args)
            if result.ok:
                raise SystemExit(f"{tool_name} should reject absolute local note paths/refusals.")
            if result.metadata.get("raw_path") != "<local-path>" or any(fragment in str(result.metadata) for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"{tool_name} leaked absolute raw_path metadata: {result.metadata}")
            if any(fragment in result.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"{tool_name} leaked absolute path output: {result.output}")

        original_bounded_read = runtime.vault.read_note_bounded

        def failing_bounded_read(*_args, **_kwargs):
            raise OSError("permission denied near /\x55sers/example/private/tasks")

        runtime.vault.read_note_bounded = failing_bounded_read  # type: ignore[method-assign]
        try:
            failed_preview = runtime.registry.get("preview_tasks_from_note").handler(
                {"path": "Projects/Task Import Smoke"}
            )
            failed_import = runtime.registry.get("import_tasks_from_note").handler(
                {"path": "Projects/Task Import Smoke", "priority": "high"}
            )
        finally:
            runtime.vault.read_note_bounded = original_bounded_read  # type: ignore[method-assign]

        for label, failed in (("preview", failed_preview), ("import", failed_import)):
            if failed.ok or "Could not read Jarvis note for task" not in failed.output:
                raise SystemExit(f"Task note {label} failure should return stable guidance: {failed.output}")
            for expected in ["JARVIS_OBSIDIAN_VAULT", "vault permissions", "setup check", "then retry"]:
                if expected not in failed.output:
                    raise SystemExit(f"Task note {label} failure missed actionable guidance {expected}: {failed.output}")
            if "/\x55sers/operator" in failed.output or "permission denied" in failed.output:
                raise SystemExit(f"Task note {label} failure leaked raw filesystem text: {failed.output}")
            if failed.metadata.get("reason") != "note_read_failed" or failed.metadata.get("exception_type") != "OSError":
                raise SystemExit(f"Task note {label} failure missed diagnostics: {failed.metadata}")
            if failed.metadata.get("writes_files") or failed.metadata.get("writes_memory") or failed.metadata.get("writes_notes"):
                raise SystemExit(f"Task note {label} failure should not claim writes: {failed.metadata}")
            assert_safe(failed.metadata, f"failed task note {label}")


if __name__ == "__main__":
    main()
